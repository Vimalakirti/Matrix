// Morpheus S&P'27: GPU kernels of the PCS authentication data path
// (mirrors zk-torch-4/src/morpheus/pcs_maps.rs and auth.rs).
//
//  * wtable / ev_partial / ev_reduce: Ev_raw via the eq tensor factorization,
//      P_m = Σ_{l,j} (e_l e_j) W_l[j + 64 m],  m = 0..63  (K coefficients);
//    the host finishes Ev = g_k(X)·P(X) (one 64x64 A product).
//  * decode: Σ_h 2^h col[h + 64 m] mod q for every column and m (H and E_O).
//  * combine: Z_r += β_{r,e} · V_e for both rows at once (S action X = Y^64
//    is a negacyclic 64-tap convolution along k of each j-slice).
//  * norms: per-block squared norm (exact 128-bit) and max |coefficient|.
#pragma once
#include <cstdint>
#include "almost_goldilocks.cuh"
#include "morpheus_k8.cuh"
#include <mma.h>

namespace mauth {

typedef unsigned long long u64;
constexpr u64 Q = 18446744069414584289ULL;

struct K8d { u64 c[8]; };

__device__ __forceinline__ u64 fmul(u64 a, u64 b) { return agl_mul(AlmostGoldilocksField(a), AlmostGoldilocksField(b)).value; }
__device__ __forceinline__ u64 fadd(u64 a, u64 b) { return agl_add(AlmostGoldilocksField(a), AlmostGoldilocksField(b)).value; }
__device__ __forceinline__ u64 fcanon(u64 a) { return agl_canonicalize(a); }
__device__ __forceinline__ u64 to_field(int64_t v) { return v >= 0 ? (u64)v : Q - (u64)(-v); }

/// K = F_q[Θ]/(Θ^8 − 3) product (lazy reduction, canonical output).
__device__ __forceinline__ K8d k8_mul(const K8d& a, const K8d& b) {
    K8d r;
    mk8::mul(a.c, b.c, r.c);
    return r;
}

/// w[l][j] = e_l · e_j for l < cols, j < 64.
__global__ void wtable_kernel(const K8d* __restrict__ el, const K8d* __restrict__ ej, int cols, K8d* __restrict__ w) {
    for (size_t id = blockIdx.x * (size_t)blockDim.x + threadIdx.x; id < (size_t)cols * 64; id += (size_t)gridDim.x * blockDim.x)
        w[id] = k8_mul(el[id / 64], ej[id % 64]);
}

/// Per-block partial sums P_m over a grid-stride set of columns.
/// 256 threads: m = t % 64, quarter q = t / 64 handles j in [16q, 16q+16).
/// ev_partial_kernel for many sources at once: source s = blockIdx.y at
/// W + s · stride elements, partial sums at partial + s · gridDim.x · 64.
template <typename TW, bool PACKED = false>
__global__ void ev_partial_kernel(const TW* __restrict__ W, int cols, const K8d* __restrict__ w, K8d* __restrict__ partial, size_t stride = 0) {
    W += (size_t)blockIdx.y * stride;
    partial += (size_t)blockIdx.y * gridDim.x * 64;
    // The column tile and the final 4x64x8 reduction share one buffer
    // (int64 tile 33,280 B; reduction 16,384 B), keeping the int64 variant
    // under the 48 KB static limit.
    constexpr size_t TILE = sizeof(TW) * 64 * 65, RED = 4 * 64 * 8 * 8;
    __shared__ __align__(16) unsigned char raw[TILE > RED ? TILE : RED];
    TW* s = reinterpret_cast<TW*>(raw);
    __shared__ K8d sw[64];
    // −w per column: digits ±1 (every raw source digit) add w or −w.
    __shared__ K8d snw[64];
    const int t = threadIdx.x, m = t & 63, qtr = t >> 6;
    u64 acc[8];
#pragma unroll
    for (int a = 0; a < 8; a++) acc[a] = 0;
    for (int l = blockIdx.x; l < cols; l += gridDim.x) {
        if (PACKED) {
            // Packed physical column (1024 bytes, 2 bits per digit).
            const uint8_t* col = reinterpret_cast<const uint8_t*>(W) + (size_t)l * 1024;
            for (int i = t; i < 4096; i += 256) s[(i >> 6) * 65 + (i & 63)] = (TW)(int8_t)((int8_t)(col[i >> 2] << (6 - 2 * (i & 3))) >> 6);
        } else {
            const TW* col = W + (size_t)l * 4096;
            for (int i = t; i < 4096; i += 256) s[(i >> 6) * 65 + (i & 63)] = col[i];  // phys i = j + 64 m -> s[m][j]
        }
        if (t < 64) {
            K8d x = w[(size_t)l * 64 + t];
            sw[t] = x;
#pragma unroll
            for (int a = 0; a < 8; a++) { u64 c = fcanon(x.c[a]); x.c[a] = c ? Q - c : 0; }
            snw[t] = x;
        }
        __syncthreads();
        for (int j = qtr * 16; j < qtr * 16 + 16; j++) {
            int64_t v = (int64_t)s[m * 65 + j];
            if (v == 0) continue;
            if (v == 1 || v == -1) {
                const u64* c = v > 0 ? sw[j].c : snw[j].c;
#pragma unroll
                for (int a = 0; a < 8; a++) acc[a] = fadd(acc[a], c[a]);
                continue;
            }
            u64 x = to_field(v);
#pragma unroll
            for (int a = 0; a < 8; a++) acc[a] = fadd(acc[a], fmul(sw[j].c[a], x));
        }
        __syncthreads();
    }
    u64 (*red)[64][8] = reinterpret_cast<u64 (*)[64][8]>(raw);
#pragma unroll
    for (int a = 0; a < 8; a++) red[qtr][m][a] = acc[a];
    __syncthreads();
    if (qtr == 0) {
        K8d o;
#pragma unroll
        for (int a = 0; a < 8; a++) o.c[a] = fcanon(fadd(fadd(red[0][m][a], red[1][m][a]), fadd(red[2][m][a], red[3][m][a])));
        partial[(size_t)blockIdx.x * 64 + m] = o;
    }
}

/// out[m] = Σ_b partial[b][m]; 512 threads = 64 m x 8 coordinates (one
/// source per blockIdx.x).
__global__ void ev_reduce_kernel(const K8d* __restrict__ partial, int blocks, K8d* __restrict__ out) {
    partial += (size_t)blockIdx.x * blocks * 64; out += (size_t)blockIdx.x * 64;
    int t = threadIdx.x, m = t >> 3, a = t & 7;
    u64 s = 0;
    for (int b = 0; b < blocks; b++) s = fadd(s, partial[(size_t)b * 64 + m].c[a]);
    out[m].c[a] = fcanon(s);
}

/// out[c][m] = Σ_h 2^h col_c[h + 64 m] mod q (int64 columns).
__global__ void decode_kernel(const int64_t* __restrict__ cols, int ncols, u64* __restrict__ out) {
    for (size_t id = blockIdx.x * (size_t)blockDim.x + threadIdx.x; id < (size_t)ncols * 64; id += (size_t)gridDim.x * blockDim.x) {
        size_t c = id / 64; int m = (int)(id % 64);
        const int64_t* col = cols + c * 4096 + 64 * m;
        u64 s = 0, p2 = 1;
        for (int h = 0; h < 64; h++) {
            int64_t v = col[h];
            if (v) s = fadd(s, fmul(to_field(v), p2));
            p2 = fadd(p2, p2);
        }
        out[id] = fcanon(s);
    }
}

/// Z_r[col][i] += Σ_m β_r[m] · sgn · V[col][i − 64 m (mod 4096)], r = 0, 1.
/// One block per column; V is int8 (source/link/aux witnesses are digits).
__global__ void combine_kernel(const int8_t* __restrict__ V, int cols, const int8_t* __restrict__ beta2 /* [2][64] */,
                               int64_t* __restrict__ Z0, int64_t* __restrict__ Z1) {
    __shared__ int8_t s[4096];
    __shared__ int8_t b[2][64];
    if (threadIdx.x < 128) b[threadIdx.x >> 6][threadIdx.x & 63] = beta2[threadIdx.x];
    for (int l = blockIdx.x; l < cols; l += gridDim.x) {
        const int8_t* col = V + (size_t)l * 4096;
        for (int i = threadIdx.x; i < 4096; i += blockDim.x) s[i] = col[i];
        __syncthreads();
        for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
            int j = i & 63, k = i >> 6;
            int a0 = 0, a1 = 0;
#pragma unroll 8
            for (int m = 0; m < 64; m++) {
                int kk = k - m;
                int v = kk >= 0 ? s[j + 64 * kk] : -s[j + 64 * (kk + 64)];
                a0 += b[0][m] * v;
                a1 += b[1][m] * v;
            }
            Z0[(size_t)l * 4096 + i] += a0;
            Z1[(size_t)l * 4096 + i] += a1;
        }
        __syncthreads();
    }
}

/// combine_kernel over several sources at once: source e's column l is at
/// Vp[e] + l·4096 with challenge rows beta[e][2][64]. Each block
/// keeps a column's 2×4096 sums in registers across the sources (thread t
/// owns outputs t + 512q, q < 8) and updates Z once, instead of one
/// read-modify-write of Z per source.
__global__ void __launch_bounds__(512) combine_multi_kernel(const int8_t* const* __restrict__ Vp, int nsrc, int cols,
                                                            const int8_t* __restrict__ beta, int64_t* __restrict__ Z0, int64_t* __restrict__ Z1) {
    __shared__ __align__(16) int8_t s[4096];
    __shared__ int8_t b[2][64];
    const int tid = threadIdx.x, j = tid & 63, k0 = tid >> 6;
    for (int l = blockIdx.x; l < cols; l += gridDim.x) {
        int a0[8], a1[8];
#pragma unroll
        for (int q = 0; q < 8; q++) { a0[q] = 0; a1[q] = 0; }
        for (int e = 0; e < nsrc; e++) {
            const int8_t* col = Vp[e] + (size_t)l * 4096;
            reinterpret_cast<long long*>(s)[tid] = __ldg(reinterpret_cast<const long long*>(col) + tid);
            if (tid < 128) b[tid >> 6][tid & 63] = beta[(size_t)e * 128 + tid];
            __syncthreads();
#pragma unroll 4
            for (int m = 0; m < 64; m++) {
                const int bb0 = b[0][m], bb1 = b[1][m];
#pragma unroll
                for (int q = 0; q < 8; q++) {
                    const int kk = k0 + 8 * q - m;
                    const int v = kk >= 0 ? s[j + 64 * kk] : -s[j + 64 * (kk + 64)];
                    a0[q] += bb0 * v; a1[q] += bb1 * v;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int q = 0; q < 8; q++) {
            const size_t i = (size_t)l * 4096 + tid + 512 * q;
            Z0[i] += a0[q]; Z1[i] += a1[q];
        }
    }
}

/// combine_multi_kernel with packed int8 dot products. For lane position j
/// the column's negacyclic sequence is ext[p] = −x_p (p < 64), x_{p−64}
/// (p ≥ 64), x_t = V[j + 64t], and output k is
///   y_k = Σ_m β_m ext[k − m + 64] = Σ_u βr[u] ext[k + 1 + u],  βr[u] = β[63 − u],
/// a 64-byte window: 16 __dp4a per row. Four byte-shifted, word-packed copies
/// E[sh][w][j] = ext[4w + sh .. 4w + sh + 3] make every window word-aligned.
template <bool PACKED = false>
__global__ void __launch_bounds__(512) combine_dp4a_kernel(const int8_t* const* __restrict__ Vp, int nsrc, int cols,
                                                           const int8_t* __restrict__ beta, int64_t* __restrict__ Z0, int64_t* __restrict__ Z1) {
    __shared__ __align__(16) int8_t s[4096];
    __shared__ int E[4][33][64];
    __shared__ int Br[2][16];
    const int tid = threadIdx.x, j = tid & 63, k0 = tid >> 6;
    for (int l = blockIdx.x; l < cols; l += gridDim.x) {
        int a0[8], a1[8];
#pragma unroll
        for (int q = 0; q < 8; q++) { a0[q] = 0; a1[q] = 0; }
        for (int e = 0; e < nsrc; e++) {
            if (PACKED) {
                // Packed physical column: thread tid unpacks digits 8 tid .. 8 tid + 7.
                const uint16_t x = __ldg(reinterpret_cast<const uint16_t*>(Vp[e] + (size_t)l * 1024) + tid);
#pragma unroll
                for (int q = 0; q < 8; q++) s[8 * tid + q] = (int8_t)((int8_t)((x >> (2 * q)) << 6) >> 6);
            } else {
                const int8_t* col = Vp[e] + (size_t)l * 4096;
                reinterpret_cast<long long*>(s)[tid] = __ldg(reinterpret_cast<const long long*>(col) + tid);
            }
            if (tid < 32) {
                // Reversed rows packed: word w = βr[4w .. 4w+3], βr[u] = β[63 − u].
                const int r = tid >> 4, w = tid & 15;
                const int8_t* bb = beta + (size_t)e * 128 + r * 64;
                Br[r][w] = (int)((uint32_t)(uint8_t)bb[63 - 4 * w] | ((uint32_t)(uint8_t)bb[62 - 4 * w] << 8)
                               | ((uint32_t)(uint8_t)bb[61 - 4 * w] << 16) | ((uint32_t)(uint8_t)bb[60 - 4 * w] << 24));
            }
            __syncthreads();
            // E[sh][w][j] for w < 33 (window words up to index 32 when sh > 0).
            for (int t = tid; t < 4 * 33 * 64; t += 512) {
                const int jj = t & 63, w = (t >> 6) % 33, sh = t / (33 * 64);
                uint32_t word = 0;
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const int p = 4 * w + sh + b;
                    int v = 0;
                    if (p < 64) v = -s[jj + 64 * p];
                    else if (p < 128) v = s[jj + 64 * (p - 64)];
                    word |= (uint32_t)(uint8_t)(int8_t)v << (8 * b);
                }
                E[sh][w][jj] = (int)word;
            }
            __syncthreads();
#pragma unroll
            for (int q = 0; q < 8; q++) {
                const int st = k0 + 8 * q + 1, sh = st & 3, w0 = st >> 2;
                int x0 = a0[q], x1 = a1[q];
#pragma unroll
                for (int w = 0; w < 16; w++) {
                    const int ev = E[sh][w0 + w][j];
                    x0 = __dp4a(ev, Br[0][w], x0);
                    x1 = __dp4a(ev, Br[1][w], x1);
                }
                a0[q] = x0; a1[q] = x1;
            }
            __syncthreads();
        }
#pragma unroll
        for (int q = 0; q < 8; q++) {
            const size_t i = (size_t)l * 4096 + tid + 512 * q;
            Z0[i] += a0[q]; Z1[i] += a1[q];
        }
    }
}

/// Wire codec of the responses: v ∈ (−B_Z, B_Z) is sent as the 37-bit
/// unsigned u = v + (B_Z − 1), LSB first, eight coefficients per 37 bytes.
__global__ void pack37_kernel(const int64_t* __restrict__ z, u64 groups, int64_t off, uint8_t* __restrict__ out) {
    for (u64 g = blockIdx.x * (u64)blockDim.x + threadIdx.x; g < groups; g += (u64)gridDim.x * blockDim.x) {
        u64 w[5] = {0, 0, 0, 0, 0};
        for (int t = 0; t < 8; t++) {
            u64 u = (u64)(z[g * 8 + t] + off) & ((1ULL << 37) - 1);
            int bp = 37 * t, i = bp >> 6, sh = bp & 63;
            w[i] |= u << sh;
            if (sh + 37 > 64) w[i + 1] |= u >> (64 - sh);
        }
        uint8_t* o = out + g * 37;
        for (int b = 0; b < 37; b++) o[b] = (uint8_t)(w[b >> 3] >> (8 * (b & 7)));
    }
}

/// Inverse of pack37; a code above `umax` (= 2 B_Z − 2) sets *bad.
__global__ void unpack37_kernel(const uint8_t* __restrict__ in, u64 groups, int64_t off, u64 umax, int64_t* __restrict__ z, int* __restrict__ bad) {
    for (u64 g = blockIdx.x * (u64)blockDim.x + threadIdx.x; g < groups; g += (u64)gridDim.x * blockDim.x) {
        u64 w[5] = {0, 0, 0, 0, 0};
        const uint8_t* p = in + g * 37;
        for (int b = 0; b < 37; b++) w[b >> 3] |= (u64)p[b] << (8 * (b & 7));
        for (int t = 0; t < 8; t++) {
            int bp = 37 * t, i = bp >> 6, sh = bp & 63;
            u64 u = w[i] >> sh;
            if (sh + 37 > 64) u |= w[i + 1] << (64 - sh);
            u &= (1ULL << 37) - 1;
            if (u > umax) *bad = 1;
            z[g * 8 + t] = (int64_t)u - off;
        }
    }
}

/// Per-block partial squared norms (lo, hi words of a 128-bit sum) and max |v|.
__global__ void norm_kernel(const int64_t* __restrict__ z, size_t n, u64* __restrict__ part /* [grid][3] */) {
    u64 lo = 0, hi = 0, mx = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int64_t v = z[i];
        u64 a = (u64)(v < 0 ? -v : v);
        u64 sl = a * a, sh = __umul64hi(a, a);
        u64 nl = lo + sl; hi += sh + (nl < lo); lo = nl;
        mx = a > mx ? a : mx;
    }
    __shared__ u64 L[256], H[256], M[256];
    L[threadIdx.x] = lo; H[threadIdx.x] = hi; M[threadIdx.x] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        u64 rl = 0, rh = 0, rm = 0;
        for (int t = 0; t < blockDim.x; t++) { u64 nl = rl + L[t]; rh += H[t] + (nl < rl); rl = nl; rm = M[t] > rm ? M[t] : rm; }
        part[blockIdx.x * 3] = rl; part[blockIdx.x * 3 + 1] = rh; part[blockIdx.x * 3 + 2] = rm;
    }
}

/// Logical raw digits -> physical columns: dst[c·4096 + perm(l)] = src[c·4096 + l],
/// perm(l) = j + 64k for l = 64j + k.
__global__ void permute_kernel(const int8_t* __restrict__ src, int8_t* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        size_t c = i >> 12; int l = (int)(i & 4095);
        int j = l >> 6, k = l & 63;
        dst[(c << 12) + j + 64 * k] = src[i];
    }
}

/// Tensor-core combine. For column l and lane j (physical digit j + 64 k),
///   Z_r[l][j + 64 k] += Σ_e Σ_m' C_e[r·64 + k][m'] · V_e[l][j + 64 m'],
///   C_e[r·64 + k][m'] = β_{r,e}[(k − m') mod 64] · (k ≥ m' ? 1 : −1),
/// the negacyclic S action of combine_kernel as one int8 GEMM per column
/// (M = 128 rows (r, k), N = 64 lanes, K = 64 per source), accumulated
/// exactly in int32 (|Σ| ≤ 128 · nsrc). build_c_kernel lays out C as
/// [e][128][64]. A block takes TC_COLS columns and 8 warps; warp w owns output
/// rows 16w..16w+15 for every column and lane.
constexpr int TC_COLS = 2;
__global__ void build_c_kernel(const int8_t* __restrict__ beta /* [n][2][64] */, int nsrc, int8_t* __restrict__ C) {
    for (size_t id = blockIdx.x * (size_t)blockDim.x + threadIdx.x; id < (size_t)nsrc * 128 * 64; id += (size_t)gridDim.x * blockDim.x) {
        const int mp = (int)(id & 63), row = (int)((id >> 6) & 127), e = (int)(id >> 13);
        const int r = row >> 6, k = row & 63, m = (k - mp) & 63;
        const int b = beta[(size_t)e * 128 + r * 64 + m];
        C[id] = (int8_t)(k >= mp ? b : -b);
    }
}
/// V_e[l] at Vp[e] + l · colbytes: packed physical columns (PACKED, 1024
/// bytes) or int8 (4096 bytes).
template <bool PACKED>
__global__ void __launch_bounds__(256) combine_tc_kernel(const uint8_t* const* __restrict__ Vp, int nsrc, int cols,
                                                         const int8_t* __restrict__ C, int64_t* __restrict__ Z0, int64_t* __restrict__ Z1) {
    using namespace nvcuda;
    __shared__ __align__(32) int8_t sA[128 * 64];
    __shared__ __align__(32) int8_t sB[TC_COLS][64 * 64];
    __shared__ __align__(32) int sOut[8][16 * 16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t colbytes = PACKED ? 1024 : 4096;
    for (int l0 = blockIdx.x * TC_COLS; l0 < cols; l0 += gridDim.x * TC_COLS) {
        const int nc = cols - l0 < TC_COLS ? cols - l0 : TC_COLS;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[TC_COLS][4];
#pragma unroll
        for (int c = 0; c < TC_COLS; c++)
#pragma unroll
            for (int t = 0; t < 4; t++) wmma::fill_fragment(acc[c][t], 0);
        for (int e = 0; e < nsrc; e++) {
            // A: C_e (8 KB), B: the source's columns (64 m' × 64 j each).
            const int4* ca = reinterpret_cast<const int4*>(C + (size_t)e * 8192);
            for (int q = threadIdx.x; q < 512; q += 256) reinterpret_cast<int4*>(sA)[q] = __ldg(ca + q);
            for (int c = 0; c < TC_COLS; c++) {
                if (c >= nc) break;
                const uint8_t* v = Vp[e] + (size_t)(l0 + c) * colbytes;
                if (PACKED) {
                    // 1024 bytes: thread t unpacks 4 bytes (16 digits).
                    const uint32_t x = __ldg(reinterpret_cast<const uint32_t*>(v) + threadIdx.x);
                    uint32_t w[4];
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        uint32_t o = 0;
#pragma unroll
                        for (int b = 0; b < 4; b++) {
                            const int f = (int)((x >> (2 * (4 * q + b))) & 3u);
                            o |= (uint32_t)(uint8_t)(int8_t)(f == 1 ? 1 : f == 3 ? -1 : 0) << (8 * b);
                        }
                        w[q] = o;
                    }
                    reinterpret_cast<int4*>(sB[c])[threadIdx.x] = make_int4((int)w[0], (int)w[1], (int)w[2], (int)w[3]);
                } else {
                    reinterpret_cast<int4*>(sB[c])[threadIdx.x] = __ldg(reinterpret_cast<const int4*>(v) + threadIdx.x);
                }
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
                wmma::load_matrix_sync(a, sA + warp * 16 * 64 + kk * 16, 64);
#pragma unroll
                for (int c = 0; c < TC_COLS; c++) {
#pragma unroll
                    for (int t = 0; t < 4; t++) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b;
                        wmma::load_matrix_sync(b, sB[c] + kk * 16 * 64 + t * 16, 64);
                        wmma::mma_sync(acc[c][t], a, b, acc[c][t]);
                    }
                }
            }
            __syncthreads();
        }
        // Rows 16w..16w+15 are (r, k) = (w / 4, 16 (w % 4) + i).
        const int r = warp >> 2, k0 = (warp & 3) * 16;
        int64_t* Z = r ? Z1 : Z0;
        for (int c = 0; c < nc; c++) {
            for (int t = 0; t < 4; t++) {
                wmma::store_matrix_sync(sOut[warp], acc[c][t], 16, wmma::mem_row_major);
                __syncwarp();
                for (int q = lane; q < 256; q += 32) {
                    const int i = q >> 4, jj = q & 15;
                    Z[(size_t)(l0 + c) * 4096 + (t * 16 + jj) + 64 * (k0 + i)] += sOut[warp][q];
                }
                __syncwarp();
            }
        }
    }
}

/// Packed logical columns -> packed physical columns (as permute_kernel):
/// output digit j + 64 k of a column is input digit 64 j + k. One thread per
/// output byte; ncols columns of 1024 bytes.
__global__ void permute2_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t ncols) {
    for (size_t ob = blockIdx.x * (size_t)blockDim.x + threadIdx.x; ob < ncols * 1024; ob += (size_t)gridDim.x * blockDim.x) {
        const size_t c = ob >> 10;
        const uint8_t* col = src + (c << 10);
        uint32_t out = 0;
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int p = (int)((ob & 1023) * 4 + q), j = p & 63, k = p >> 6, l = 64 * j + k;
            out |= (uint32_t)((col[l >> 2] >> (2 * (l & 3))) & 3) << (2 * q);
        }
        dst[ob] = (uint8_t)out;
    }
}

/// Packed raw digits (2 bits each, 00 = 0, 01 = 1, 11 = -1) -> int8,
/// optionally permuted to physical columns as permute_kernel.
__device__ __forceinline__ int8_t unpack_digit(const uint8_t* __restrict__ p, size_t i) {
    return (int8_t)((int8_t)(p[i >> 2] << (6 - 2 * (int)(i & 3))) >> 6);
}
__global__ void unpack2_kernel(const uint8_t* __restrict__ src, int8_t* __restrict__ dst, size_t n, int phys) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int8_t v = unpack_digit(src, i);
        if (phys) { size_t c = i >> 12; int l = (int)(i & 4095); dst[(c << 12) + (l >> 6) + 64 * (l & 63)] = v; }
        else dst[i] = v;
    }
}

}  // namespace mauth

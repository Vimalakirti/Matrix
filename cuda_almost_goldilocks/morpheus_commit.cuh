// Morpheus S&P'27: GPU randomized Ajtai commitment over R = F_q[Y]/(Y^4096+1).
//
//   C = P·x + B·s + e   (rank rows, payload+hiding columns, identity error)
//
// computed exactly: the integer negacyclic products are formed modulo k of
// the five 31-bit NTT primes and recombined by Garner CRT, then reduced mod q
// with centering (mirrors zk-torch-4/src/morpheus/ring.rs::matvec and
// commit.rs::commit, including the Kyber-style twiddle order).
//
// Pipeline:
//   keygen   : public key elements (SHAKE256 v2, as keys.rs) -> centered -> mod p_i
//              -> forward NTT, cached as key[prime][col][row][4096] (u32);
//   inputs   : int8 or int64 columns -> mod p_i -> forward NTT;
//   mac      : NTT-domain multiply-accumulate over column chunks into
//              partial[chunk][v][prime][row][4096];
//   finish   : sum partials (also across GPUs: the NTT is linear), inverse
//              NTT, Garner CRT, + identity error, mod q.
#pragma once
#include <cstdint>
#include "morpheus_shake.cuh"

namespace mcommit {

typedef unsigned long long u64;
constexpr int N = 4096;
constexpr int MAXP = 5;
constexpr int MAXROWS = 16;
constexpr u64 Q = 18446744069414584289ULL;  // 2^64 - 2^32 - 31

struct Tables {
    uint32_t p[MAXP];
    uint32_t ninv[MAXP], ninv_sh[MAXP];
    u64 barrett[MAXP];               // floor(2^64 / p)
    uint32_t zeta[MAXP][N], zeta_sh[MAXP][N];
    uint32_t izeta[MAXP][N], izeta_sh[MAXP][N];
    // Garner: inv[i][j] = p_j^{-1} mod p_i; half digits of floor(P/2); P mod q.
    uint32_t ginv[MAXP][MAXP];
    uint32_t half[MAXP][MAXP];       // half digits for each k (k-1 indexed)
    u64 pmodq[MAXP];                 // (prod_{i<k} p_i) mod q, k-1 indexed
};

// Shoup: a*w mod p for a < 2^32, w < p < 2^31, wsh = floor(w 2^32 / p).
__device__ __forceinline__ uint32_t mul_shoup(uint32_t a, uint32_t w, uint32_t wsh, uint32_t p) {
    uint32_t q = __umulhi(a, wsh);
    uint32_t r = a * w - q * p;
    return r >= p ? r - p : r;
}
__device__ __forceinline__ uint32_t barrett64(u64 x, uint32_t p, u64 m) {
    u64 q = __umul64hi(x, m);
    u64 r = x - q * p;
    while (r >= p) r -= p;
    return (uint32_t)r;
}
__device__ __forceinline__ uint32_t mulmod(uint32_t a, uint32_t b, uint32_t p, u64 m) {
    return barrett64((u64)a * b, p, m);
}

// ---------------------------------------------------------------- NTT
// One 512-thread block per 4096-point transform, in shared memory.
// Transform t: prime = (t / count) % k (layouts [.. ][prime][count][4096]).
__global__ void ntt_fwd_kernel(uint32_t* __restrict__ a, int count, int k, const Tables* __restrict__ T) {
    __shared__ uint32_t s[N];
    const size_t t = blockIdx.x;
    const int pi = (int)((t / count) % k);
    const uint32_t p = T->p[pi];
    uint32_t* g = a + (size_t)t * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) s[i] = g[i];
    __syncthreads();
    for (int len = N / 2; len >= 1; len >>= 1) {
        for (int b = threadIdx.x; b < N / 2; b += blockDim.x) {
            int grp = b / len, off = b % len;
            int j = grp * 2 * len + off;
            int zi = N / (2 * len) + grp;
            uint32_t tt = mul_shoup(s[j + len], T->zeta[pi][zi], T->zeta_sh[pi][zi], p);
            uint32_t u = s[j];
            s[j + len] = u >= tt ? u - tt : u + p - tt;
            uint32_t w = u + tt;
            s[j] = w >= p ? w - p : w;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < N; i += blockDim.x) g[i] = s[i];
}

__global__ void ntt_inv_kernel(uint32_t* __restrict__ a, int count, int k, const Tables* __restrict__ T) {
    __shared__ uint32_t s[N];
    const size_t t = blockIdx.x;
    const int pi = (int)((t / count) % k);
    const uint32_t p = T->p[pi];
    uint32_t* g = a + (size_t)t * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) s[i] = g[i];
    __syncthreads();
    for (int len = 1; len < N; len <<= 1) {
        for (int b = threadIdx.x; b < N / 2; b += blockDim.x) {
            int grp = b / len, off = b % len;
            int j = grp * 2 * len + off;
            int zi = N / (2 * len) + grp;
            uint32_t u = s[j], v = s[j + len];
            uint32_t w = u + v;
            s[j] = w >= p ? w - p : w;
            uint32_t d = u >= v ? u - v : u + p - v;
            s[j + len] = mul_shoup(d, T->izeta[pi][zi], T->izeta_sh[pi][zi], p);
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < N; i += blockDim.x) g[i] = mul_shoup(s[i], T->ninv[pi], T->ninv_sh[pi], p);
}

// ---------------------------------------------------------------- keys
__device__ __forceinline__ int64_t center_q(u64 a) {
    return a > (Q - 1) / 2 ? -(int64_t)(Q - a) : (int64_t)a;
}
__device__ __forceinline__ uint32_t res_i64(int64_t x, uint32_t p) {
    int64_t r = x % (int64_t)p;
    return (uint32_t)(r < 0 ? r + p : r);
}

/// key[prime][c][row][4096] for columns col0.. col0+ncols and rows [0,rows),
/// residues only (the NTT runs afterwards). Thread per (c, row, block b).
__global__ void keygen_kernel(const uint32_t* __restrict__ seed, uint32_t kind, int rows, u64 col0, int ncols, int k,
                              const Tables* __restrict__ T, uint32_t* __restrict__ key) {
    uint32_t sd[8];
#pragma unroll
    for (int i = 0; i < 8; i++) sd[i] = seed[i];
    const size_t total = (size_t)ncols * rows * (N / 8);
    for (size_t id = blockIdx.x * (size_t)blockDim.x + threadIdx.x; id < total; id += (size_t)gridDim.x * blockDim.x) {
        int b = (int)(id % (N / 8));
        int row = (int)((id / (N / 8)) % rows);
        int c = (int)(id / ((size_t)(N / 8) * rows));
        u64 col = col0 + c;
        uint32_t nonce[3] = {(kind << 16) | (uint32_t)row, (uint32_t)col, (uint32_t)(col >> 32)};
        u64 vals[8];
        int written = 0;
        for (uint32_t retry = 0; retry < 16 && written < 8; retry++) {
            uint32_t w[16];
            mshake::key_block(sd, (uint32_t)b + 512u * retry, nonce, w);
            for (int t = 0; t < 8 && written < 8; t++) {
                u64 v = (u64)w[2 * t] | ((u64)w[2 * t + 1] << 32);
                if (v < Q) vals[written++] = v;
            }
        }
        for (; written < 8; written++) vals[written] = 0;
        for (int pi = 0; pi < k; pi++) {
            uint32_t* dst = key + ((((size_t)pi * ncols + c) * rows + row) * N) + (size_t)b * 8;
#pragma unroll
            for (int t = 0; t < 8; t++) dst[t] = res_i64(center_q(vals[t]), T->p[pi]);
        }
    }
}

// ---------------------------------------------------------------- inputs
/// out[v][prime][c][4096] residues of input columns x[v·vstride + c·4096 + i] (int8 or int64).
template <typename TX>
__global__ void residues_kernel(const TX* __restrict__ x, size_t vstride, int V, int C, int k, const Tables* __restrict__ T,
                                uint32_t* __restrict__ out) {
    const size_t total = (size_t)V * C * N;
    for (size_t id = blockIdx.x * (size_t)blockDim.x + threadIdx.x; id < total; id += (size_t)gridDim.x * blockDim.x) {
        size_t v = id / ((size_t)C * N), rem = id % ((size_t)C * N);
        int64_t xv = (int64_t)x[v * vstride + rem];
        for (int pi = 0; pi < k; pi++) out[((v * k + pi) * C) * N + rem] = res_i64(xv, T->p[pi]);
    }
}

// ---------------------------------------------------------------- MAC
/// partial[chunk][v][prime][row][j] = sum over the chunk's columns c of
/// key[prime][kc(c)][row][j] * xh[v][prime][c][j]  (mod p),
/// where kc(c) = colidx ? colidx[c] : c indexes the key cache. Thread per
/// (prime, j); grid.y = column chunks. V <= 2.
template <int V>
__global__ void mac_kernel(const uint32_t* __restrict__ key, int kcols, int rows, const uint32_t* __restrict__ xh,
                           const uint32_t* __restrict__ colidx, int C, int k, int chunk_cols,
                           const Tables* __restrict__ T, uint32_t* __restrict__ partial) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // over k*N
    if (j >= k * N) return;
    const int pi = j / N, jj = j % N;
    const uint32_t p = T->p[pi];
    const u64 m = T->barrett[pi];
    const int c0 = blockIdx.y * chunk_cols;
    const int c1 = min(C, c0 + chunk_cols);
    u64 acc[V][MAXROWS];
#pragma unroll
    for (int v = 0; v < V; v++)
#pragma unroll
        for (int r = 0; r < MAXROWS; r++) acc[v][r] = 0;
    int pending = 0;
    for (int c = c0; c < c1; c++) {
        const int kc = colidx ? (int)colidx[c] : c;
        const uint32_t* kp = key + (((size_t)pi * kcols + kc) * rows) * N + jj;
        uint32_t xv[V];
#pragma unroll
        for (int v = 0; v < V; v++) xv[v] = xh[(((size_t)v * k + pi) * C + c) * N + jj];
#pragma unroll
        for (int r = 0; r < MAXROWS; r++) {
            if (r < rows) {
                uint32_t a = kp[(size_t)r * N];
#pragma unroll
                for (int v = 0; v < V; v++) acc[v][r] += (u64)a * xv[v];
            }
        }
        if (++pending == 4) {
            pending = 0;
#pragma unroll
            for (int v = 0; v < V; v++)
#pragma unroll
                for (int r = 0; r < MAXROWS; r++) acc[v][r] = barrett64(acc[v][r], p, m);
        }
    }
    for (int v = 0; v < V; v++)
        for (int r = 0; r < rows; r++)
            partial[((((size_t)blockIdx.y * V + v) * k + pi) * rows + r) * N + jj] = barrett64(acc[v][r], p, m);
}

/// mac_kernel for up to MV vectors per pass over the key cache: block =
/// 64 positions j of one prime × (rows / RPT) row groups; a column's V input
/// residues are staged in shared memory once for all row groups, so each key
/// element is read once per pass of V sources (mac_kernel<1> re-reads the
/// whole key per source). Output layout as mac_kernel.
constexpr int MV = 8, RPT = 2;
template <int V>
__global__ void mac_v_kernel(const uint32_t* __restrict__ key, int kcols, int rows, const uint32_t* __restrict__ xh,
                             const uint32_t* __restrict__ colidx, int C, int k, int chunk_cols,
                             const Tables* __restrict__ T, uint32_t* __restrict__ partial) {
    __shared__ uint32_t sx[4][V][64];
    const int lane = threadIdx.x & 63, rg = threadIdx.x >> 6, ngroups = blockDim.x >> 6;
    const int jt = blockIdx.x * 64 + lane;   // over k*N
    const int pi = (blockIdx.x * 64) / N, jj = jt % N;
    const uint32_t p = T->p[pi];
    const u64 m = T->barrett[pi];
    const int r0 = rg * RPT;
    const int c0 = blockIdx.y * chunk_cols;
    const int c1 = min(C, c0 + chunk_cols);
    u64 acc[V][RPT];
#pragma unroll
    for (int v = 0; v < V; v++)
#pragma unroll
        for (int r = 0; r < RPT; r++) acc[v][r] = 0;
    int pending = 0;
    for (int cb = c0; cb < c1; cb += 4) {
        const int nb = min(4, c1 - cb);
        // Stage the residues of up to 4 columns × V vectors.
        for (int t = threadIdx.x; t < 4 * V * 64; t += blockDim.x) {
            const int q = t / (V * 64), v = (t / 64) % V, l = t & 63;
            if (q < nb) sx[q][v][l] = xh[(((size_t)v * k + pi) * C + cb + q) * N + (size_t)(blockIdx.x * 64 % N) + l];
        }
        __syncthreads();
        for (int q = 0; q < nb; q++) {
            const int c = cb + q;
            const int kc = colidx ? (int)colidx[c] : c;
            const uint32_t* kp = key + (((size_t)pi * kcols + kc) * rows) * N + jj;
#pragma unroll
            for (int r = 0; r < RPT; r++) {
                if (r0 + r < rows) {
                    const uint32_t a = kp[(size_t)(r0 + r) * N];
#pragma unroll
                    for (int v = 0; v < V; v++) acc[v][r] += (u64)a * sx[q][v][lane];
                }
            }
            if (++pending == 4) {
                pending = 0;
#pragma unroll
                for (int v = 0; v < V; v++)
#pragma unroll
                    for (int r = 0; r < RPT; r++) acc[v][r] = barrett64(acc[v][r], p, m);
            }
        }
        __syncthreads();
    }
    for (int v = 0; v < V; v++)
        for (int r = 0; r < RPT; r++)
            if (r0 + r < rows) partial[((((size_t)blockIdx.y * V + v) * k + pi) * rows + r0 + r) * N + jj] = barrett64(acc[v][r], p, m);
    (void)ngroups;
}

/// acc[v][prime][row][j] = sum over `parts` partial blocks (mod p)./// acc[v][prime][row][j] = sum over `parts` partial blocks (mod p).
__global__ void sum_partials_kernel(const uint32_t* __restrict__ partial, int parts, size_t stride, int k, int rows,
                                    const Tables* __restrict__ T, uint32_t* __restrict__ acc, size_t n_elems) {
    for (size_t id = blockIdx.x * (size_t)blockDim.x + threadIdx.x; id < n_elems; id += (size_t)gridDim.x * blockDim.x) {
        int pi = (int)((id / ((size_t)rows * N)) % k);
        uint32_t p = T->p[pi];
        u64 s = 0;
        for (int b = 0; b < parts; b++) s += partial[b * stride + id];
        acc[id] = (uint32_t)(s % p);
    }
}

// ---------------------------------------------------------------- CRT
// x*p + d mod q for x < q, p < 2^31, d < 2^31.
__device__ __forceinline__ u64 horner_q(u64 x, uint32_t p, uint32_t d) {
    u64 lo = x * (u64)p, hi = __umul64hi(x, (u64)p);
    // 2^64 = 2^32 + 31 (mod q); hi < 2^31.
    u64 t = hi * ((1ULL << 32) + 31);
    u64 s = lo + t;
    if (s < lo) s += (1ULL << 32) + 31;  // wrapped: add 2^64 mod q
    u64 s2 = s + d;
    if (s2 < s) s2 += (1ULL << 32) + 31;
    while (s2 >= Q) s2 -= Q;
    return s2;
}

/// out[v][row][j] = CRT(acc[v][*][row][j]) mod q (+ err[v][row][j]).
__global__ void crt_kernel(const uint32_t* __restrict__ acc, int V, int k, int rows, const Tables* __restrict__ T,
                           const int64_t* __restrict__ err, u64* __restrict__ out) {
    const size_t total = (size_t)V * rows * N;
    for (size_t id = blockIdx.x * (size_t)blockDim.x + threadIdx.x; id < total; id += (size_t)gridDim.x * blockDim.x) {
        size_t v = id / ((size_t)rows * N), rj = id % ((size_t)rows * N);
        uint32_t d[MAXP];
        for (int i = 0; i < k; i++) {
            uint32_t pi = T->p[i];
            u64 m = T->barrett[i];
            uint32_t x = acc[((v * k + i) * rows) * N + rj];
            for (int j = 0; j < i; j++) {
                uint32_t dj = d[j] % pi;
                x = x >= dj ? x - dj : x + pi - dj;
                x = mulmod(x, T->ginv[i][j], pi, m);
            }
            d[i] = x;
        }
        u64 x = 0;
        for (int i = k - 1; i >= 0; i--) x = horner_q(x, T->p[i], d[i]);
        bool greater = false;
        for (int i = k - 1; i >= 0; i--) {
            if (d[i] != T->half[k - 1][i]) { greater = d[i] > T->half[k - 1][i]; break; }
        }
        if (greater) { u64 pm = T->pmodq[k - 1]; x = x >= pm ? x - pm : x + (Q - pm); }
        if (err) {
            int64_t e = err[id];
            u64 eu = e >= 0 ? (u64)e : Q - (u64)(-e);
            u64 s = x + eu;
            if (s < x || s >= Q) s -= Q;
            x = s;
        }
        out[id] = x;
    }
}

}  // namespace mcommit

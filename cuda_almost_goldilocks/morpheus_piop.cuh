// Morpheus S&P'27: GPU kernels of the masked graph PIOP (tables over K).
//  * pe_kernel: partial evaluation of a base-field tensor at K coordinates:
//      out[r] = Σ_f e[f]·vals[dep(r, rest) | dep(f, fix)]
//  * eq_level_kernel: one doubling level of an eq table.
//  * gather_kernel: out[x] = in[Σ_j bit(x, pos_j) << j] (variable maps).
//  * axpy_kernel: out += c·in.
//  * sc_round_kernel / sc_fold_kernel: product sumcheck rounds over K tables.
#pragma once
#include <cstdint>
#include "morpheus_link.cuh"

namespace mpiop {

using mauth::K8d;
using mlink::kzero; using mlink::kadd; using mlink::ksub; using mlink::kcanon; using mlink::shfl_down_k;
typedef unsigned long long u64;
constexpr int MAXV = 32;
constexpr int MAXTAB = 6;

struct VarMap { int n; int pos[MAXV]; };

__device__ __forceinline__ uint64_t deposit(uint64_t x, const VarMap& m) {
    uint64_t r = 0;
    for (int j = 0; j < m.n; j++) r |= ((x >> j) & 1ULL) << m.pos[j];
    return r;
}
__device__ __forceinline__ K8d kmulb(const K8d& a, u64 v) { K8d r; for (int j = 0; j < 8; j++) r.c[j] = mauth::fmul(a.c[j], v); return r; }

/// MLE of a base table at a K point without a full eq table: the point is
/// split low/mid/high (≤ 8 variables each, eq tables built per block in
/// shared memory) and
///   Σ_{h,m} eh[h]·em[m]·Σ_l el[l]·vals[(((h << nmid) | m) << nlo) | l],
/// a warp per row (h, m); partial[block] = the block's sum.
__device__ __forceinline__ void eq_smem(K8d* e, const K8d* r, int n) {
    if (threadIdx.x == 0) { e[0] = kzero(); e[0].c[0] = 1; }
    __syncthreads();
    for (int j = 0; j < n; j++) {
        const int half = 1 << j;
        K8d omr = kzero(); omr.c[0] = 1; omr = ksub(omr, r[j]);
        for (int t = threadIdx.x; t < half; t += blockDim.x) { const K8d x = e[t]; e[t | half] = mauth::k8_mul(x, r[j]); e[t] = mauth::k8_mul(x, omr); }
        __syncthreads();
    }
}
/// The three eq tables of `mle2_kernel`: block b builds tab[256b..256b+256)
/// (el, em, eh), entry t = Π_j (bit_j(t) ? r_j : 1 − r_j) (no dependent levels).
__global__ void __launch_bounds__(256) eq3_kernel(const K8d* __restrict__ point, int nlo, int nmid, int nhi, K8d* __restrict__ tab) {
    const int b = blockIdx.x, t = threadIdx.x;
    const int nb = b == 0 ? nlo : b == 1 ? nmid : nhi;
    const K8d* r = point + (b == 0 ? 0 : b == 1 ? nlo : nlo + nmid);
    if (t >= (1 << nb)) return;
    K8d acc = kzero(); acc.c[0] = 1;
    for (int j = 0; j < nb; j++) {
        K8d f;
        if ((t >> j) & 1) f = r[j]; else { f = kzero(); f.c[0] = 1; f = ksub(f, r[j]); }
        acc = j == 0 ? f : mauth::k8_mul(acc, f);
    }
    tab[256 * b + t] = kcanon(acc);
}
__global__ void __launch_bounds__(256) mle2_kernel(const u64* __restrict__ vals, const K8d* __restrict__ tab, int nlo, int nmid, int nhi,
                                                   K8d* __restrict__ partial) {
    __shared__ K8d wsum[8];
    const K8d* el = tab; const K8d* em = tab + 256; const K8d* eh = tab + 512;
    const uint64_t nrows = 1ULL << (nmid + nhi);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint64_t gw = (uint64_t)blockIdx.x * 8 + warp, nw = (uint64_t)gridDim.x * 8;
    K8d accw = kzero();
    for (uint64_t row = gw; row < nrows; row += nw) {
        K8d acc = kzero();
        const u64* vr = vals + (row << nlo);
        for (uint32_t l = lane; l < (1u << nlo); l += 32) {
            const u64 v = vr[l];
            if (v) acc = kadd(acc, v == 1 ? el[l] : kmulb(el[l], v));
        }
        acc = kcanon(acc);
        for (int off = 16; off > 0; off >>= 1) acc = kadd(acc, shfl_down_k(acc, off));
        if (lane == 0 && !mlink::kzero_p(acc)) {
            const K8d w = mauth::k8_mul(eh[row >> nmid], em[row & ((1ULL << nmid) - 1)]);
            accw = kadd(accw, mauth::k8_mul(acc, w));
        }
    }
    if (lane == 0) wsum[warp] = kcanon(accw);
    __syncthreads();
    if (threadIdx.x == 0) {
        K8d s = kzero();
        for (int w = 0; w < 8; w++) s = kadd(s, wsum[w]);
        partial[blockIdx.x] = kcanon(s);
    }
}

/// 256 threads = 32 outputs × 8 f-lanes./// 256 threads = 32 outputs × 8 f-lanes.
__global__ void __launch_bounds__(256) pe_kernel(const u64* __restrict__ vals, VarMap rest, VarMap fix,
                                                 const K8d* __restrict__ e, uint64_t nrest, uint64_t nfix, K8d* __restrict__ out) {
    __shared__ K8d part[8][32];
    const int rr = threadIdx.x & 31, fl = threadIdx.x >> 5;
    for (uint64_t r0 = blockIdx.x * 32ULL; r0 < nrest; r0 += gridDim.x * 32ULL) {
        const uint64_t r = r0 + rr;
        K8d acc = kzero();
        if (r < nrest) {
            const uint64_t ro = deposit(r, rest);
            for (uint64_t f = fl; f < nfix; f += 8) {
                u64 v = vals[ro | deposit(f, fix)];
                if (v) acc = kadd(acc, v == 1 ? e[f] : kmulb(e[f], v));
            }
        }
        part[fl][rr] = kcanon(acc);
        __syncthreads();
        if (fl == 0 && r < nrest) {
            K8d s = part[0][rr];
            for (int k = 1; k < 8; k++) s = kadd(s, part[k][rr]);
            out[r] = kcanon(s);
        }
        __syncthreads();
    }
}

/// Partial evaluation with few outputs: block per output, all threads over f.
__global__ void __launch_bounds__(256) pe_wide_kernel(const u64* __restrict__ vals, VarMap rest, VarMap fix,
                                                      const K8d* __restrict__ e, uint64_t nrest, uint64_t nfix, K8d* __restrict__ out) {
    __shared__ K8d wsum[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (uint64_t r = blockIdx.y; r < nrest; r += gridDim.y) {
        const uint64_t ro = deposit(r, rest);
        K8d acc = kzero();
        for (uint64_t f = blockIdx.x * 256ULL + threadIdx.x; f < nfix; f += gridDim.x * 256ULL) {
            u64 v = vals[ro | deposit(f, fix)];
            if (v) acc = kadd(acc, v == 1 ? e[f] : kmulb(e[f], v));
        }
        acc = kcanon(acc);
        for (int off = 16; off > 0; off >>= 1) acc = kadd(acc, shfl_down_k(acc, off));
        if (lane == 0) wsum[warp] = kcanon(acc);
        __syncthreads();
        if (threadIdx.x == 0) {
            K8d s = kzero();
            for (int w = 0; w < 8; w++) s = kadd(s, wsum[w]);
            out[r * gridDim.x + blockIdx.x] = kcanon(s);   // partial per x-block
        }
        __syncthreads();
    }
}

/// Out-of-place doubling: out[2i + b] = in[i]·(b ? r : 1 − r). Variable j is bit 0 of
/// the new index when processed last-variable-first, so call with variables in
/// reverse order to get little-endian tables.
__global__ void eq_double_kernel(const K8d* __restrict__ in, K8d* __restrict__ out, uint64_t half, K8d r) {
    K8d one_m = kzero(); one_m.c[0] = 1; one_m = ksub(one_m, r);
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < half; i += (uint64_t)gridDim.x * blockDim.x) {
        const K8d v = in[i];
        out[2 * i] = mauth::k8_mul(v, one_m);
        out[2 * i + 1] = mauth::k8_mul(v, r);
    }
}

/// out[x] = lo[x mod 2^nlo]·hi[x >> nlo] (an eq table from its two halves).
__global__ void eq_outer_kernel(const K8d* __restrict__ lo, int nlo, const K8d* __restrict__ hi, uint64_t n, K8d* __restrict__ out) {
    const uint64_t m = (1ULL << nlo) - 1;
    for (uint64_t x = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; x < n; x += (uint64_t)gridDim.x * blockDim.x)
        out[x] = mauth::k8_mul(lo[x & m], hi[x >> nlo]);
}

/// out[i] = table[(sel[i] >> shift) & mask] (lookup tables over selector indices).
__global__ void sel_lookup_kernel(const u64* __restrict__ sel, uint64_t n, int shift, u64 mask, const K8d* __restrict__ table, K8d* __restrict__ out) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        out[i] = table[(sel[i] >> shift) & mask];
}

__global__ void gather_kernel(const K8d* __restrict__ in, VarMap m, uint64_t n, K8d* __restrict__ out) {
    for (uint64_t x = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; x < n; x += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t idx = 0;
        for (int j = 0; j < m.n; j++) idx |= ((x >> m.pos[j]) & 1ULL) << j;
        out[x] = in[idx];
    }
}

__global__ void axpy_kernel(K8d* __restrict__ out, const K8d* __restrict__ in, uint64_t n, K8d c) {
    for (uint64_t x = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; x < n; x += (uint64_t)gridDim.x * blockDim.x)
        out[x] = kcanon(kadd(out[x], mauth::k8_mul(in[x], c)));
}

/// From base values to K (canonical words → K with zero high coordinates).
__global__ void lift_kernel(const u64* __restrict__ in, uint64_t n, K8d* __restrict__ out) {
    for (uint64_t x = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; x < n; x += (uint64_t)gridDim.x * blockDim.x) {
        K8d r = kzero(); r.c[0] = in[x]; out[x] = r;
    }
}

struct Tabs { const K8d* t[MAXTAB]; int ntab; };

/// Σ_k Π_j (T_j[2k] + t (T_j[2k+1] − T_j[2k])) for t = 0..d; partial per block.
__global__ void __launch_bounds__(256) sc_round_kernel(Tabs T, uint64_t len, int d, K8d* __restrict__ partial) {
    __shared__ K8d red[8][MAXTAB + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    K8d acc[MAXTAB + 1];
    for (int t = 0; t <= d; t++) acc[t] = kzero();
    const uint64_t half = len >> 1;
    for (uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; k < half; k += (uint64_t)gridDim.x * blockDim.x) {
        K8d v[MAXTAB], dv[MAXTAB];
        for (int j = 0; j < T.ntab; j++) { v[j] = T.t[j][2 * k]; dv[j] = ksub(T.t[j][2 * k + 1], v[j]); }
        for (int t = 0; t <= d; t++) {
            if (t > 0) for (int j = 0; j < T.ntab; j++) v[j] = kadd(v[j], dv[j]);
            K8d p = v[0];
            for (int j = 1; j < T.ntab; j++) p = mauth::k8_mul(p, v[j]);
            acc[t] = kadd(acc[t], p);
        }
    }
    for (int t = 0; t <= d; t++) {
        K8d x = kcanon(acc[t]);
        for (int off = 16; off > 0; off >>= 1) x = kadd(x, shfl_down_k(x, off));
        if (lane == 0) red[warp][t] = kcanon(x);
    }
    __syncthreads();
    if (threadIdx.x <= d) {
        K8d s = kzero();
        for (int w = 0; w < 8; w++) s = kadd(s, red[w][threadIdx.x]);
        partial[(size_t)blockIdx.x * (MAXTAB + 1) + threadIdx.x] = kcanon(s);
    }
}

__global__ void sc_fold_kernel(const K8d* __restrict__ in, K8d* __restrict__ out, uint64_t len, K8d r) {
    const uint64_t half = len >> 1;
    for (uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; k < half; k += (uint64_t)gridDim.x * blockDim.x) {
        K8d a = in[2 * k], b = in[2 * k + 1];
        out[k] = kcanon(kadd(a, mauth::k8_mul(r, ksub(b, a))));
    }
}

/// One fused round: fold every table with r (out_j[k] = T_j[2k] + r (T_j[2k+1] − T_j[2k]),
/// len/2 entries) and, over the folded pairs, the next round's
/// Σ_k Π_j (F_j[2k] + t (F_j[2k+1] − F_j[2k])) for t = 0..d; partial per block.
struct OutTabs { K8d* t[MAXTAB]; };
__global__ void __launch_bounds__(256) sc_fold_round_kernel(Tabs T, OutTabs O, uint64_t len, K8d r, int d, K8d* __restrict__ partial) {
    __shared__ K8d red[8][MAXTAB + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    K8d acc[MAXTAB + 1];
    for (int t = 0; t <= d; t++) acc[t] = kzero();
    const uint64_t quarter = len >> 2;   // next-round pairs
    for (uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; k < quarter; k += (uint64_t)gridDim.x * blockDim.x) {
        K8d v[MAXTAB], dv[MAXTAB];
        for (int j = 0; j < T.ntab; j++) {
            const K8d a0 = T.t[j][4 * k], a1 = T.t[j][4 * k + 1], b0 = T.t[j][4 * k + 2], b1 = T.t[j][4 * k + 3];
            const K8d f0 = kcanon(kadd(a0, mauth::k8_mul(r, ksub(a1, a0))));
            const K8d f1 = kcanon(kadd(b0, mauth::k8_mul(r, ksub(b1, b0))));
            O.t[j][2 * k] = f0; O.t[j][2 * k + 1] = f1;
            v[j] = f0; dv[j] = ksub(f1, f0);
        }
        for (int t = 0; t <= d; t++) {
            if (t > 0) for (int j = 0; j < T.ntab; j++) v[j] = kadd(v[j], dv[j]);
            K8d p = v[0];
            for (int j = 1; j < T.ntab; j++) p = mauth::k8_mul(p, v[j]);
            acc[t] = kadd(acc[t], p);
        }
    }
    for (int t = 0; t <= d; t++) {
        K8d x = kcanon(acc[t]);
        for (int off = 16; off > 0; off >>= 1) x = kadd(x, shfl_down_k(x, off));
        if (lane == 0) red[warp][t] = kcanon(x);
    }
    __syncthreads();
    if (threadIdx.x <= d) {
        K8d s = kzero();
        for (int w = 0; w < 8; w++) s = kadd(s, red[w][threadIdx.x]);
        partial[(size_t)blockIdx.x * (MAXTAB + 1) + threadIdx.x] = kcanon(s);
    }
}

/// out[t] = Σ_b partial[b][t] for t = 0..d (one block, grid-stride over b).
__global__ void __launch_bounds__(256) sc_partial_sum_kernel(const K8d* __restrict__ partial, int nblk, int d, K8d* __restrict__ out) {
    __shared__ K8d red[8][MAXTAB + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int t = 0; t <= d; t++) {
        K8d x = kzero();
        for (int b = threadIdx.x; b < nblk; b += blockDim.x) x = kadd(x, partial[(size_t)b * (MAXTAB + 1) + t]);
        x = kcanon(x);
        for (int off = 16; off > 0; off >>= 1) x = kadd(x, shfl_down_k(x, off));
        if (lane == 0) red[warp][t] = kcanon(x);
    }
    __syncthreads();
    if (threadIdx.x <= d) {
        K8d s = kzero();
        for (int w = 0; w < 8; w++) s = kadd(s, red[w][threadIdx.x]);
        out[threadIdx.x] = kcanon(s);
    }
}

}  // namespace mpiop

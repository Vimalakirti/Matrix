// Morpheus S&P'27: GPU phase 1 of the structured Π_link prover
// (mirrors zk-torch-4/src/morpheus/link_fast.rs::phase1_block / block_fold).
//
// A data block is an aligned 2^B0 run of one source's ±1 digits lying inside
// one region. In round i < B0 its folded values F[y] = Σ_{p<2^i} E_i[p]·
// digit[p + y·2^i] are rebuilt in shared memory from the digits, then each
// pair (F[2·rest], F[2·rest+1]) contributes, for t = 0..d,
//   range[t] += e_alpha[rest]·(cr·Q(w_t) + cb·(w_t² − w_t)),
//   query[t] += Σ_terms c·A[t][pr]·B[ir] · w_t,
// with Q(w) = w³ − w, w_t = w0 + t (w1 − w0), rest = pr + (ir << pa).
#pragma once
#include <cstdint>
#include "morpheus_auth.cuh"

namespace mlink {

using mauth::K8d;
typedef unsigned long long u64;
constexpr int MAXD = 4;            // old-variable degree cap (phase 1)
constexpr int NACC = 2 * (MAXD + 1);

__device__ __forceinline__ K8d kzero() { K8d r; for (int a = 0; a < 8; a++) r.c[a] = 0; return r; }
__device__ __forceinline__ K8d kadd(const K8d& x, const K8d& y) { K8d r; for (int a = 0; a < 8; a++) r.c[a] = mauth::fadd(x.c[a], y.c[a]); return r; }
__device__ __forceinline__ K8d ksub(const K8d& x, const K8d& y) {
    K8d r; for (int a = 0; a < 8; a++) r.c[a] = agl_sub(AlmostGoldilocksField(x.c[a]), AlmostGoldilocksField(y.c[a])).value; return r;
}
__device__ __forceinline__ bool kzero_p(const K8d& x) { u64 o = 0; for (int a = 0; a < 8; a++) o |= mauth::fcanon(x.c[a]); return o == 0; }
__device__ __forceinline__ K8d kcanon(const K8d& x) { K8d r; for (int a = 0; a < 8; a++) r.c[a] = mauth::fcanon(x.c[a]); return r; }

struct BlockDesc {
    uint32_t src, id, toff, tcnt;
    uint32_t qoff, pad0, pad1, pad2;  // dense query table offset (K units), or 0xffffffff
    K8d cr, cb;   // c_alpha·γ_s and c_alpha·γ'_s·χ
};
struct BlockTerm { uint32_t term; uint32_t pad; K8d c; };
struct TermTab { uint32_t aoff, boff, pa, pad; };

__device__ __forceinline__ K8d shfl_down_k(const K8d& x, int off) {
    K8d r; for (int a = 0; a < 8; a++) r.c[a] = __shfl_down_sync(0xffffffffu, x.c[a], off); return r;
}

/// One phase-1 round. Grid-stride over data blocks; warp t (blockDim =
/// 32·(d+1)) accumulates the contributions at evaluation point t, so each
/// thread holds two K accumulators. Each CUDA block writes NACC partial K
/// values (range[0..d], query[0..d]) to partial[blockIdx.x].
__global__ void __launch_bounds__(160) phase1_kernel(
    const int8_t* __restrict__ raw, uint64_t src_len, int b0, int i, int d,
    const BlockDesc* __restrict__ blocks, uint32_t nblocks,
    const BlockTerm* __restrict__ bterms, const TermTab* __restrict__ ttab,
    const K8d* __restrict__ A, const K8d* __restrict__ B,
    const K8d* __restrict__ e_pre, const K8d* __restrict__ e_alpha,
    const K8d* __restrict__ qd, K8d* __restrict__ partial)
{
    extern __shared__ K8d F[];   // 2^{b0−i} entries (i ≥ 1)
    const int lane = threadIdx.x & 31, t = threadIdx.x >> 5;
    const u64 tq = (u64)t;
    K8d acc_r = kzero(), acc_q = kzero();
    const int ny = 1 << (b0 - i);
    const int nrest = ny >> 1;
    for (uint32_t bi = blockIdx.x; bi < nblocks; bi += gridDim.x) {
        const BlockDesc bd = blocks[bi];
        const int8_t* dg = raw + (uint64_t)bd.src * src_len + ((uint64_t)bd.id << b0);
        if (i > 0) {
            for (int y = threadIdx.x; y < ny; y += blockDim.x) {
                K8d s = kzero();
                const int8_t* p0 = dg + ((size_t)y << i);
                for (int p = 0; p < (1 << i); p++) {
                    int8_t v = p0[p];
                    if (v == 1) s = kadd(s, e_pre[p]); else if (v == -1) s = ksub(s, e_pre[p]);
                }
                F[y] = s;
            }
            __syncthreads();
        }
        if (t <= d) {
            for (int rest = lane; rest < nrest; rest += 32) {
                K8d w0, w1;
                if (i == 0) {
                    w0 = kzero(); w1 = kzero();
                    int8_t a = dg[2 * rest], b = dg[2 * rest + 1];
                    if (a == 0 && b == 0) continue;
                    w0.c[0] = a == 1 ? 1ULL : (a == -1 ? mauth::Q - 1 : 0ULL);
                    w1.c[0] = b == 1 ? 1ULL : (b == -1 ? mauth::Q - 1 : 0ULL);
                } else {
                    w0 = F[2 * rest]; w1 = F[2 * rest + 1];
                    if (kzero_p(w0) && kzero_p(w1)) continue;
                }
                // w_t = w0 + t·(w1 − w0)
                K8d dw = ksub(w1, w0);
                K8d wt;
                for (int a = 0; a < 8; a++) wt.c[a] = mauth::fadd(w0.c[a], mauth::fmul(dw.c[a], tq));
                K8d w2 = mauth::k8_mul(wt, wt);
                K8d w3 = mauth::k8_mul(w2, wt);
                K8d r = mauth::k8_mul(bd.cr, ksub(w3, wt));
                r = kadd(r, mauth::k8_mul(bd.cb, ksub(w2, wt)));
                acc_r = kadd(acc_r, mauth::k8_mul(e_alpha[rest], r));
                // Query weight at t.
                K8d qw = kzero();
                if (bd.qoff != 0xffffffffu) {
                    const K8d* qb = qd + bd.qoff;
                    K8d q0 = kzero(), q1 = kzero();
                    for (int p = 0; p < (1 << i); p++) {
                        q0 = kadd(q0, mauth::k8_mul(e_pre[p], qb[p + (2 * rest << i)]));
                        q1 = kadd(q1, mauth::k8_mul(e_pre[p], qb[p + ((2 * rest + 1) << i)]));
                    }
                    K8d dq = ksub(q1, q0);
                    for (int a = 0; a < 8; a++) qw.c[a] = mauth::fadd(q0.c[a], mauth::fmul(dq.c[a], tq));
                }
                for (uint32_t k = 0; k < bd.tcnt; k++) {
                    const BlockTerm bt = bterms[bd.toff + k];
                    const TermTab tt = ttab[bt.term];
                    int pr = rest & ((1 << tt.pa) - 1), ir = rest >> tt.pa;
                    K8d bv = mauth::k8_mul(B[tt.boff + ir], bt.c);
                    qw = kadd(qw, mauth::k8_mul(A[tt.aoff + (t << tt.pa) + pr], bv));
                }
                acc_q = kadd(acc_q, mauth::k8_mul(qw, wt));
            }
        }
        __syncthreads();
    }
    // Warp reductions; lane 0 of warp t holds acc at t.
    K8d vr = kcanon(acc_r), vq = kcanon(acc_q);
    for (int off = 16; off > 0; off >>= 1) { vr = kadd(vr, shfl_down_k(vr, off)); vq = kadd(vq, shfl_down_k(vq, off)); }
    if (lane == 0 && t <= d) {
        partial[(size_t)blockIdx.x * NACC + t] = kcanon(vr);
        partial[(size_t)blockIdx.x * NACC + d + 1 + t] = kcanon(vq);
    }
}

/// out[b] = Σ_p e_full[p]·digit[p] over each data block (fully folded value).
__global__ void __launch_bounds__(256) fold_kernel(
    const int8_t* __restrict__ raw, uint64_t src_len, int b0,
    const BlockDesc* __restrict__ blocks, uint32_t nblocks,
    const K8d* __restrict__ e_full, K8d* __restrict__ out)
{
    __shared__ K8d wsum[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (uint32_t bi = blockIdx.x; bi < nblocks; bi += gridDim.x) {
        const BlockDesc bd = blocks[bi];
        const int8_t* dg = raw + (uint64_t)bd.src * src_len + ((uint64_t)bd.id << b0);
        K8d s = kzero();
        for (int p = threadIdx.x; p < (1 << b0); p += blockDim.x) {
            int8_t v = dg[p];
            if (v == 1) s = kadd(s, e_full[p]); else if (v == -1) s = ksub(s, e_full[p]);
        }
        s = kcanon(s);
        for (int off = 16; off > 0; off >>= 1) s = kadd(s, shfl_down_k(s, off));
        if (lane == 0) wsum[warp] = kcanon(s);
        __syncthreads();
        if (threadIdx.x == 0) {
            K8d t = kzero();
            for (int w = 0; w < (int)(blockDim.x >> 5); w++) t = kadd(t, wsum[w]);
            out[bi] = kcanon(t);
        }
        __syncthreads();
    }
}

}  // namespace mlink

// Morpheus S&P'27: structured Π_link prover, GPU v2.
//
// Same rounds as zk-torch-4/src/morpheus/link_fast.rs (the CPU reference);
// the work is organised by region instead of by (block, term):
//   * per-block constants come from split tables: c_alpha(id) =
//     CA_lo[id_lo]·CA_hi[id_hi]; a term's high-index factor hi_T(k) =
//     HL_T[k_lo]·HH_T[k_hi] (k = block index inside the region);
//   * one-hot regions (lookup selectors) in their plane rounds (i < p): an
//     index with selected plane t* has folded value e = E_i[t* mod 2^i] on the
//     side bit_i(t*) of its pair, so with λ_0 = 1 − t, λ_1 = t,
//       Q(λe) = λ³e³ − λe,  λe(λe − 1) = λ²e² − λe,
//     and per index only ea·e, ea·e², ea·e³ (and the query terms) are
//     accumulated, split by bit_i(t*);
//   * other rounds: folded digits (and dense query tables) are rebuilt in
//     shared memory, nonzero pairs are compacted, warp t evaluates point t;
//   * coarse rounds (i ≥ B0) run on dense per-source K tables.
#pragma once
#include <cstdint>
#include "morpheus_link.cuh"

namespace mlink2 {

using mauth::K8d;
using mlink::kzero; using mlink::kadd; using mlink::ksub; using mlink::kzero_p; using mlink::kcanon; using mlink::shfl_down_k;
typedef unsigned long long u64;
constexpr int MAXD = 4;
constexpr int NACC = 2 * (MAXD + 1);
constexpr int MAXT = 32;          // terms per region

struct Region {
    uint32_t src, base_blk, h, p;      // h: block-index bits inside the region
    uint32_t flags, toff, tcnt, qd_off; // flags: 1 chi, 2 one-hot, 4 dense-q; qd_off: start of this region's entries in qidx
};
struct Term {
    uint32_t pa, aoff, boff, hl_off, hh_off, h1, pad0, pad1;
    K8d cs;                             // coef·scale
};
struct Blk { uint32_t src, id, region, k; };   // region 0xffffffff: no terms, no flags

// Raw source digits are packed two bits each (zk-torch-4 morpheus/packed.rs):
// digit k is bits 2(k mod 4)..2(k mod 4)+2 of byte k/4, 00 = 0, 01 = 1,
// 11 = -1. A block starts on a byte boundary (2^b0 digits, b0 >= 2).
__device__ __forceinline__ const uint8_t* blk_bytes(const uint8_t* raw, uint64_t src_len, const Blk& b, int b0) {
    return raw + (((uint64_t)b.src * src_len + ((uint64_t)b.id << b0)) >> 2);
}
__device__ __forceinline__ int dig(const uint8_t* __restrict__ bp, uint32_t k) {
    return (int)((int8_t)(__ldg(bp + (k >> 2)) << (6 - 2 * (int)(k & 3))) >> 6);
}
/// 16 digits starting at digit k (k a multiple of 16) as one word: field j
/// (bits 2j, 2j+1) is digit k + j.
__device__ __forceinline__ uint32_t dig16(const uint8_t* __restrict__ bp, uint32_t k) {
    return __ldg(reinterpret_cast<const uint32_t*>(bp + (k >> 2)));
}

__device__ __forceinline__ K8d kmul(const K8d& a, const K8d& b) { return mauth::k8_mul(a, b); }
__device__ __forceinline__ K8d kscale(const K8d& a, u64 s) { K8d r; for (int j = 0; j < 8; j++) r.c[j] = mauth::fmul(a.c[j], s); return r; }

struct Consts {
    const K8d* ca_lo; const K8d* ca_hi; int ca_lo_bits;
    const K8d* g_range; const K8d* g_bit;
    const Region* regions; const Term* terms; const K8d* hl; const K8d* hh;
    const K8d* A; const K8d* B;            // per-round term tables
    const K8d* qd;                         // dense query tables (level 0: 2^b0 K each; rounds: folded to level i)
    const uint32_t* qidx;                  // per region block: table number or 0xffffffff
    const u64* bco;                        // per block (gen_all order): offset of its constants in bc (K units)
    u64 oh_base;                           // index of the first one-hot block in gen_all (one-hot pre pass)
};

/// Per-block constants: c_alpha·γ (cr), c_alpha·γ'·χ (cb), term factors c_T.
__device__ __forceinline__ void block_consts(const Consts& C, const Blk& b, K8d& cr, K8d& cb, K8d* ct, int& tcnt, uint32_t& flags) {
    const uint32_t mlo = (1u << C.ca_lo_bits) - 1;
    K8d ca = kmul(C.ca_lo[b.id & mlo], C.ca_hi[b.id >> C.ca_lo_bits]);
    cr = kmul(ca, C.g_range[b.src]);
    flags = 0; tcnt = 0;
    if (b.region != 0xffffffffu) {
        const Region R = C.regions[b.region];
        flags = R.flags;
        tcnt = (int)R.tcnt;
        for (int t = 0; t < tcnt; t++) {
            const Term T = C.terms[R.toff + t];
            uint32_t klo = b.k & ((1u << T.h1) - 1), khi = b.k >> T.h1;
            ct[t] = kmul(T.cs, kmul(C.hl[T.hl_off + klo], C.hh[T.hh_off + khi]));
        }
    }
    cb = (flags & 1) ? kmul(ca, C.g_bit[b.src]) : kzero();
}

// ------------------------------------------------------------- constants
/// Per-block constants, once per link: bc[b·(2+MAXT)+0] = cr, +1 = cb,
/// +2+k = c_T for the region's k-th term (k < tcnt ≤ MAXT).
__global__ void block_consts_kernel(const Blk* __restrict__ blks, uint32_t nblk, Consts C, K8d* __restrict__ bc) {
    for (uint32_t bi = blockIdx.x * blockDim.x + threadIdx.x; bi < nblk; bi += gridDim.x * blockDim.x) {
        const Blk b = blks[bi];
        K8d* o = bc + C.bco[bi];
        int tc; uint32_t fl;
        block_consts(C, b, o[0], o[1], o + 2, tc, fl);
    }
}

// ------------------------------------------------------------- one-hot
/// Plane round i < p of one-hot regions. Output per CUDA block: NACC K.
__global__ void __launch_bounds__(256) onehot_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0, int i, int d,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C,
    const K8d* __restrict__ E1, const K8d* __restrict__ E2, const K8d* __restrict__ E3,
    const K8d* __restrict__ e_alpha, K8d* __restrict__ partial)
{
    __shared__ K8d s_ct[MAXT];
    __shared__ K8d s_cr, s_cb;
    __shared__ int s_tcnt, s_p;
    __shared__ uint32_t s_toff;
    __shared__ K8d red[8][10];
    __shared__ K8d tot[NACC];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x < NACC) tot[threadIdx.x] = kzero();
    for (uint32_t bi = blockIdx.x; bi < nblk; bi += gridDim.x) {
        const Blk b = blks[bi];
        if (threadIdx.x == 0) {
            int tc; uint32_t fl;
            K8d cr, cb;
            block_consts(C, b, cr, cb, s_ct, tc, fl);
            s_cr = cr; s_cb = cb; s_tcnt = tc;
            const Region R = C.regions[b.region];
            s_p = (int)R.p; s_toff = R.toff;
        }
        __syncthreads();
        const int p = s_p;
        const int nidx = 1 << (b0 - p);
        const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
        K8d acc[10];  // S1[2], S2[2], S3[2], P0[2], P1[2]
        for (int k = 0; k < 10; k++) acc[k] = kzero();
        for (int idx = threadIdx.x; idx < nidx; idx += blockDim.x) {
            const uint32_t pl = (uint32_t)idx << p;
            int ts = -1;
            for (int q = 0; q < (1 << p); q++) if (dig(dg, pl + q) == 1) { ts = q; break; }
            if (ts < 0) continue;
            const int bit = (ts >> i) & 1, low = ts & ((1 << i) - 1), pr = ts >> (i + 1);
            const int pa = p - i - 1;
            const int rest = pr + (idx << pa);
            const K8d ea = e_alpha[rest];
            const K8d e = E1[low];
            acc[0 + bit] = kadd(acc[0 + bit], kmul(ea, e));
            acc[2 + bit] = kadd(acc[2 + bit], kmul(ea, E2[low]));
            acc[4 + bit] = kadd(acc[4 + bit], kmul(ea, E3[low]));
            K8d qa = kzero(), qd = kzero();
            for (int t = 0; t < s_tcnt; t++) {
                const Term T = C.terms[s_toff + t];
                K8d bv = kmul(C.B[T.boff + idx], s_ct[t]);
                K8d a0 = C.A[T.aoff + pr], a1 = C.A[T.aoff + (1 << T.pa) + pr];
                qa = kadd(qa, kmul(a0, bv));
                qd = kadd(qd, kmul(ksub(a1, a0), bv));
            }
            acc[6 + bit] = kadd(acc[6 + bit], kmul(qa, e));
            acc[8 + bit] = kadd(acc[8 + bit], kmul(qd, e));
        }
        for (int k = 0; k < 10; k++) {
            K8d v = kcanon(acc[k]);
            for (int off = 16; off > 0; off >>= 1) v = kadd(v, shfl_down_k(v, off));
            if (lane == 0) red[warp][k] = kcanon(v);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            K8d S[10];
            for (int k = 0; k < 10; k++) { S[k] = kzero(); for (int w = 0; w < (int)(blockDim.x >> 5); w++) S[k] = kadd(S[k], red[w][k]); }
            for (int t = 0; t <= d; t++) {
                // λ_0 = 1 − t, λ_1 = t as field elements.
                u64 l[2] = { t == 0 ? 1ULL : (t == 1 ? 0ULL : mauth::Q - (u64)(t - 1)), (u64)t };
                K8d r1 = kzero(), r2 = kzero(), qv = kzero();
                for (int bt = 0; bt < 2; bt++) {
                    u64 l1 = l[bt], l2 = mauth::fmul(l1, l1), l3 = mauth::fmul(l2, l1);
                    // cr·(λ³S3 − λS1) + cb·(λ²S2 − λS1)
                    r1 = kadd(r1, ksub(kscale(S[4 + bt], l3), kscale(S[0 + bt], l1)));
                    r2 = kadd(r2, ksub(kscale(S[2 + bt], l2), kscale(S[0 + bt], l1)));
                    qv = kadd(qv, kscale(kadd(S[6 + bt], kscale(S[8 + bt], (u64)t)), l1));
                }
                tot[t] = kadd(tot[t], kadd(kmul(s_cr, r1), kmul(s_cb, r2)));
                tot[d + 1 + t] = kadd(tot[d + 1 + t], qv);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < NACC) partial[(size_t)blockIdx.x * NACC + threadIdx.x] = kcanon(tot[threadIdx.x]);
}

// ------------------------------------------------------------- generic
/// Generic block-local round, blockDim = 32·(d+1), warp t evaluates point t.
/// A data block's pairs are processed in chunks of GCH:
///   fold the chunk's digits into F (i ≥ 1; round 0 reads digits directly),
///   compact the nonzero pairs, build their query weights QF at t = 0, 1 once
///   (level-i dense table + Σ_T c_T B_T[ir] A_T[side][pr]; all weights are
///   multilinear, so linear in t), then evaluate.
/// The range constants cr, cb are applied once per block per lane:
///   Σ ea·(cr(w³ − w) + cb(w² − w)) = cr·Σ ea(w³ − w) + cb·Σ ea(w² − w).
/// Reduced mode (blockDim 96): warps evaluate t = 0, 2, 3 (range) and t = 0, 2
/// (query); the host gets t = 1 from the previous round's claims.
constexpr int GCH = 128;
/// MODE (specialised by block kind, each launched over its own index list):
/// 0 range only (no query), 1 range + fast query, 2 range + general query.
template <int MINB, int MODE>
__global__ void __launch_bounds__(160, MINB) generic_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0, int i, int d, int reduced_dbg,
    const Blk* __restrict__ blks, const uint32_t* __restrict__ list, uint32_t nblk, Consts C, const K8d* __restrict__ bc,
    const K8d* __restrict__ e_pre, const K8d* __restrict__ e_alpha, K8d* __restrict__ partial)
{
    // Debug (timing only; results wrong): bit 1 skips range evaluation, bit 2 the query.
    const int reduced = reduced_dbg & 1, dbg = reduced_dbg >> 1;
    __shared__ K8d F[2 * GCH], QF[2 * GCH];
    __shared__ uint16_t nz[GCH];
    __shared__ K8d s_ct[MAXT];
    __shared__ K8d s_cr, s_cb;
    __shared__ int s_tcnt, s_cnt, s_pa;
    __shared__ uint32_t s_toff, s_qd;
    const int ny = 1 << (b0 - i), nrest = ny >> 1;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int t = reduced ? (wp == 0 ? 0 : wp + 1) : wp;
    const bool doq = !reduced || t < 3;
    const u64 tq = (u64)t;
    K8d acc_r = kzero(), acc_q = kzero();
    if (threadIdx.x == 0) s_cnt = 0;
    for (uint32_t li = blockIdx.x; li < nblk; li += gridDim.x) {
        const uint32_t bi = list[li];
        const Blk b = blks[bi];
        if (threadIdx.x == 0) {
            uint32_t fl = 0; int tc = 0; uint32_t toff = 0, qdo = 0;
            if (b.region != 0xffffffffu) { const Region R = C.regions[b.region]; fl = R.flags; tc = (int)R.tcnt; toff = R.toff; qdo = R.qd_off; }
            s_tcnt = tc; s_toff = toff;
            s_pa = tc > 0 ? (int)C.terms[toff].pa : 0;
            uint32_t qi = (fl & 4) ? C.qidx[qdo + b.k] : 0xffffffffu;
            s_qd = qi == 0xffffffffu ? 0xffffffffu : (qi << (b0 - i));   // level-i folded tables
        }
        __syncthreads();
        {
            const K8d* o = bc + C.bco[bi];
            if (threadIdx.x == 0) s_cr = o[0];
            if (threadIdx.x == 1) s_cb = o[1];
            for (int k = (int)threadIdx.x - 2; k < s_tcnt; k += blockDim.x) if (k >= 0) s_ct[k] = o[2 + k];
        }
        const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
        const int tcnt = s_tcnt;
        const uint32_t toff = s_toff, qd = s_qd;
        K8d b3 = kzero(), b2 = kzero();          // this lane's Σ ea(w³ − w), Σ ea(w² − w) in the block
        // Fast query (no dense table, ≤ 2 terms, 2^pa ≤ 8): with q_t(r) =
        // Σ_T c_T B_T[ir](A_T[0][pr] + t dA_T[pr]) and w_t = (1−t)w0 + t w1,
        //   Σ_r q_t w_t = Σ_T c_T Σ_pr (A0 + t dA)[pr]·((1−t)S0 + t S1),
        //   S_side[T][pr] = Σ_ir B_T[ir]·w_side(ir, pr)  (t-independent).
        // A thread's rests are ≡ threadIdx (mod blockDim), so its pr is fixed.
        const int pa = s_pa;
        const bool fastq = MODE == 1 && !(dbg & 2);
        K8d S00 = kzero(), S01 = kzero(), S10 = kzero(), S11 = kzero();   // S{term}{side}
        for (int c0 = 0; c0 < nrest; c0 += GCH) {
            const int cp = nrest - c0 < GCH ? nrest - c0 : GCH, cy = 2 * cp, y0 = 2 * c0;
            if (i > 0) {
                // A group of G threads (G ≤ 32, inside a warp) shares one y.
                int G = 1;
                while (G < 32 && G * cy < (int)blockDim.x && G < (1 << i)) G <<= 1;
                const int groups = blockDim.x / G;
                const int gid = threadIdx.x / G, gl = threadIdx.x % G;
                for (int yr = 0; yr < cy; yr += groups) {
                    const int yl = yr + gid;
                    const bool live = yl < cy;
                    K8d sacc = kzero();
                    if (live) {
                        const uint32_t p0 = (uint32_t)(y0 + yl) << i;
                        for (int p = gl; p < (1 << i); p += G) { int v = dig(dg, p0 + p); if (v == 1) sacc = kadd(sacc, e_pre[p]); else if (v == -1) sacc = ksub(sacc, e_pre[p]); }
                    }
                    if (G > 1) {
                        sacc = kcanon(sacc);
                        for (int off = G >> 1; off > 0; off >>= 1) {
                            K8d x;
                            for (int w = 0; w < 8; w++) x.c[w] = __shfl_down_sync(0xffffffffu, sacc.c[w], off, G);
                            sacc = kadd(sacc, x);
                        }
                    }
                    if (live && gl == 0) F[yl] = kcanon(sacc);
                }
                __syncthreads();
            }
            // Compact the nonzero pairs.
            for (int r = threadIdx.x; r < cp; r += blockDim.x) {
                bool z;
                if (i == 0) z = dig(dg, y0 + 2 * r) == 0 && dig(dg, y0 + 2 * r + 1) == 0;
                else z = kzero_p(F[2 * r]) && kzero_p(F[2 * r + 1]);
                if (!z) { int pos = atomicAdd(&s_cnt, 1); nz[pos] = (uint16_t)r; }
            }
            __syncthreads();
            const int cnt = s_cnt;
            if (fastq) {
                const Term T0 = C.terms[toff];
                const Term T1 = tcnt > 1 ? C.terms[toff + 1] : T0;
                for (int r = threadIdx.x; r < cp; r += blockDim.x) {
                    const int ir = (c0 + r) >> pa;
                    if (i == 0) {
                        const int d0 = dig(dg, y0 + 2 * r), d1 = dig(dg, y0 + 2 * r + 1);
                        if ((d0 | d1) == 0) continue;
                        const K8d bb0 = C.B[T0.boff + ir];
                        if (d0 == 1) S00 = kadd(S00, bb0); else if (d0 == -1) S00 = ksub(S00, bb0);
                        if (d1 == 1) S01 = kadd(S01, bb0); else if (d1 == -1) S01 = ksub(S01, bb0);
                        if (tcnt > 1) {
                            const K8d bb1 = C.B[T1.boff + ir];
                            if (d0 == 1) S10 = kadd(S10, bb1); else if (d0 == -1) S10 = ksub(S10, bb1);
                            if (d1 == 1) S11 = kadd(S11, bb1); else if (d1 == -1) S11 = ksub(S11, bb1);
                        }
                    } else {
                        const K8d w0 = F[2 * r], w1 = F[2 * r + 1];
                        if (kzero_p(w0) && kzero_p(w1)) continue;
                        const K8d bb0 = C.B[T0.boff + ir];
                        S00 = kadd(S00, kmul(bb0, w0)); S01 = kadd(S01, kmul(bb0, w1));
                        if (tcnt > 1) { const K8d bb1 = C.B[T1.boff + ir]; S10 = kadd(S10, kmul(bb1, w0)); S11 = kadd(S11, kmul(bb1, w1)); }
                    }
                }
            }
            // Query weights of the compacted pairs at t = 0, 1.
            if (MODE == 2 && (qd != 0xffffffffu || tcnt > 0) && !(dbg & 2)) {
                for (int j = threadIdx.x; j < cnt; j += blockDim.x) {
                    const int rest = c0 + nz[j];
                    K8d q0 = qd != 0xffffffffu ? C.qd[qd + 2 * rest] : kzero();
                    K8d q1 = qd != 0xffffffffu ? C.qd[qd + 2 * rest + 1] : kzero();
                    for (int k = 0; k < tcnt; k++) {
                        const Term T = C.terms[toff + k];
                        const int pr = rest & ((1 << T.pa) - 1), ir = rest >> T.pa;
                        const K8d bv = kmul(C.B[T.boff + ir], s_ct[k]);
                        q0 = kadd(q0, kmul(C.A[T.aoff + pr], bv));
                        q1 = kadd(q1, kmul(C.A[T.aoff + (1 << T.pa) + pr], bv));
                    }
                    QF[2 * j] = kcanon(q0); QF[2 * j + 1] = kcanon(q1);
                }
                __syncthreads();
            }
            const bool hasq = MODE == 2 && (qd != 0xffffffffu || tcnt > 0) && doq && !(dbg & 2);
            {
                for (int j = lane; j < cnt; j += 32) {
                    const int rl = nz[j], rest = c0 + rl;
                    const K8d ea = e_alpha[rest];
                    K8d qt = kzero();
                    if (hasq) {
                        const K8d q0 = QF[2 * j], dq = ksub(QF[2 * j + 1], q0);
                        for (int a = 0; a < 8; a++) qt.c[a] = mauth::fadd(q0.c[a], mauth::fmul(dq.c[a], tq));
                    }
                    if (i == 0) {
                        // Digits: w_t = a + t (c − a) is a small integer.
                        const long long a = dig(dg, y0 + 2 * rl), c = dig(dg, y0 + 2 * rl + 1);
                        const long long wv = a + (long long)t * (c - a);
                        auto fe = [](long long x) -> u64 { return x >= 0 ? (u64)x : mauth::Q - (u64)(-x); };
                        if (wv != 0) {
                            b3 = kadd(b3, kscale(ea, fe(wv * wv * wv - wv)));
                            b2 = kadd(b2, kscale(ea, fe(wv * wv - wv)));
                            if (hasq) acc_q = kadd(acc_q, kscale(qt, fe(wv)));
                        }
                    } else {
                        const K8d w0 = F[2 * rl], dw = ksub(F[2 * rl + 1], w0);
                        K8d wt;
                        for (int a = 0; a < 8; a++) wt.c[a] = mauth::fadd(w0.c[a], mauth::fmul(dw.c[a], tq));
                        // ea·w, ea·w², ea·w³ by three products.
                        if (!(dbg & 1)) {
                            const K8d ew = kmul(ea, wt), ew2 = kmul(ew, wt), ew3 = kmul(ew2, wt);
                            b3 = kadd(b3, ksub(ew3, ew));
                            b2 = kadd(b2, ksub(ew2, ew));
                        }
                        if (hasq) acc_q = kadd(acc_q, kmul(qt, wt));
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) s_cnt = 0;
            __syncthreads();
        }
        if (!(kzero_p(b3) && kzero_p(b2))) acc_r = kadd(acc_r, kadd(kmul(s_cr, b3), kmul(s_cb, b2)));
        if (MODE == 3 && !(dbg & 2)) {
            // Many-term query (no dense table, 2^{i+1+pa} ≤ 32): as the fast
            // query, with rest = (ir << pa) | pr,
            //   Σ_r q_t w_t = Σ_T Σ_pr c_T (A0 + t dA)[pr]·((1−t)S0 + t S1)[pr],
            //   S_side[T][pr] = Σ_ir B_T[ir] w_side(ir, pr).
            // Over the block's digits x = ((2 rest + side) << i) | p, with
            // w_side = Σ_p e_pre[p] digit(x): S = Σ_p e_pre[p] U_T(pr, side, p),
            // U_T = Σ_ir ±B_T[ir] by additions only. One warp per term; lane ℓ
            // takes x ≡ ℓ (mod 32), so its (pr, side, p) = ℓ mod 2^{i+1+pa}.
            // S lands in QF (unused here); then (term, pr) pairs across lanes.
            const int nw = (int)(blockDim.x >> 5), np2 = 1 << i, nx = 1 << b0;
            for (int k = wp; k < tcnt; k += nw) {
                const Term T = C.terms[toff + k];
                const int sh = i + 1 + (int)T.pa, cb = 1 << sh;
                const K8d* Bk = C.B + T.boff;
                K8d u = kzero(), um = kzero();
                for (int x = lane; x < nx; x += 32) {
                    const int v = dig(dg, x);
                    if (v == 1) u = kadd(u, Bk[x >> sh]); else if (v == -1) um = kadd(um, Bk[x >> sh]);
                }
                u = kcanon(ksub(u, um));
                for (int off = 16; off >= cb; off >>= 1) u = kadd(u, shfl_down_k(u, off));
                // Lanes < cb: U(pr = ℓ >> (i+1), side = (ℓ >> i) & 1, p = ℓ & (2^i − 1)).
                if (i > 0) u = kmul(kcanon(u), e_pre[lane & (np2 - 1)]);
                u = kcanon(u);
                for (int off = np2 >> 1; off > 0; off >>= 1) u = kadd(u, shfl_down_k(u, off));
                // Lane (pr << (i+1)) | (side << i): S_side[pr], kept in QF.
                if (lane < cb && (lane & (np2 - 1)) == 0) QF[((k << T.pa) + (lane >> (i + 1))) * 2 + ((lane >> i) & 1)] = kcanon(u);
            }
            __syncthreads();
            if (doq) {
                const int pa = s_pa, np = 1 << pa;
                for (int idx = lane; idx < (tcnt << pa); idx += 32) {
                    const int k = idx >> pa, pr = idx & (np - 1);
                    const Term T = C.terms[toff + k];
                    const K8d a0 = C.A[T.aoff + pr], a1 = C.A[T.aoff + np + pr];
                    const K8d x = kmul(s_ct[k], a0), y = kmul(s_ct[k], ksub(a1, a0));
                    const K8d sa = QF[2 * idx], dsx = ksub(QF[2 * idx + 1], sa);
                    K8d qv, sv;
                    for (int a = 0; a < 8; a++) { qv.c[a] = mauth::fadd(x.c[a], mauth::fmul(y.c[a], tq)); sv.c[a] = mauth::fadd(sa.c[a], mauth::fmul(dsx.c[a], tq)); }
                    acc_q = kadd(acc_q, kmul(qv, sv));
                }
            }
        }
        if (fastq) {
            // Sums per pr: within warps over lanes ≡ pr (mod 2^pa), then across
            // warps through QF (unused by fast blocks): RED[warp][pr][4].
            const int np = 1 << pa;
            S00 = kcanon(S00); S01 = kcanon(S01); S10 = kcanon(S10); S11 = kcanon(S11);
            for (int off = 16; off >= np; off >>= 1) {
                S00 = kadd(S00, shfl_down_k(S00, off)); S01 = kadd(S01, shfl_down_k(S01, off));
                S10 = kadd(S10, shfl_down_k(S10, off)); S11 = kadd(S11, shfl_down_k(S11, off));
            }
            K8d* RED = QF;
            if (lane < np) { K8d* o = RED + ((size_t)wp * np + lane) * 4; o[0] = kcanon(S00); o[1] = kcanon(S01); o[2] = kcanon(S10); o[3] = kcanon(S11); }
            __syncthreads();
            if (doq && lane < np) {
                K8d s4[4] = {kzero(), kzero(), kzero(), kzero()};
                for (int w = 0; w < (int)(blockDim.x >> 5); w++) for (int q = 0; q < 4; q++) s4[q] = kadd(s4[q], RED[((size_t)w * np + lane) * 4 + q]);
                for (int k = 0; k < tcnt; k++) {
                    const Term T = C.terms[toff + k];
                    const K8d a0 = C.A[T.aoff + lane], a1 = C.A[T.aoff + (1 << T.pa) + lane];
                    const K8d x = kmul(s_ct[k], a0), y = kmul(s_ct[k], ksub(a1, a0));
                    const K8d sa = s4[2 * k], dsx = ksub(s4[2 * k + 1], sa);
                    K8d qv, sv;
                    for (int a = 0; a < 8; a++) { qv.c[a] = mauth::fadd(x.c[a], mauth::fmul(y.c[a], tq)); sv.c[a] = mauth::fadd(sa.c[a], mauth::fmul(dsx.c[a], tq)); }
                    acc_q = kadd(acc_q, kmul(qv, sv));
                }
            }
            S00 = S01 = S10 = S11 = kzero();
        }
        __syncthreads();
    }
    K8d vr = kcanon(acc_r), vq = kcanon(acc_q);
    for (int off = 16; off > 0; off >>= 1) { vr = kadd(vr, shfl_down_k(vr, off)); vq = kadd(vq, shfl_down_k(vq, off)); }
    if (lane == 0) {
        partial[(size_t)blockIdx.x * NACC + t] = kcanon(vr);
        if (doq) partial[(size_t)blockIdx.x * NACC + d + 1 + t] = kcanon(vq);
    }
}

// ------------------------------------------------------------- coarse init
/// w_coarse[src][id] = Σ_p e_full[p]·digit[p] for each nonzero block.
__global__ void __launch_bounds__(256) fold_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0, const Blk* __restrict__ blks, uint32_t nblk,
    const K8d* __restrict__ e_full, uint64_t coarse, K8d* __restrict__ w)
{
    __shared__ K8d wsum[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (uint32_t bi = blockIdx.x; bi < nblk; bi += gridDim.x) {
        const Blk b = blks[bi];
        const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
        K8d s = kzero();
        for (int p = threadIdx.x; p < (1 << b0); p += blockDim.x) { int v = dig(dg, p); if (v == 1) s = kadd(s, e_full[p]); else if (v == -1) s = ksub(s, e_full[p]); }
        s = kcanon(s);
        for (int off = 16; off > 0; off >>= 1) s = kadd(s, shfl_down_k(s, off));
        if (lane == 0) wsum[warp] = kcanon(s);
        __syncthreads();
        if (threadIdx.x == 0) { K8d tt = kzero(); for (int x = 0; x < (int)(blockDim.x >> 5); x++) tt = kadd(tt, wsum[x]); w[(uint64_t)b.src * coarse + b.id] = kcanon(tt); }
        __syncthreads();
    }
}

/// q_coarse and χ_coarse for every block of every region (thread per block).
/// Block g of the flattened list belongs to region rid[g] with index k[g].
/// Dense tables have 2^qbits entries (b0, or 0 for factored tables already
/// reduced against e_full).
__global__ void coarse_q_kernel(Consts C, const uint32_t* __restrict__ rid, const uint32_t* __restrict__ kk, uint64_t n,
                                const K8d* __restrict__ tfold, const K8d* __restrict__ e_full, int qbits, uint64_t coarse,
                                K8d* __restrict__ q, K8d* __restrict__ chi)
{
    for (uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; g < n; g += (uint64_t)gridDim.x * blockDim.x) {
        const Region R = C.regions[rid[g]];
        const uint32_t k = kk[g];
        K8d acc = kzero();
        for (uint32_t t = 0; t < R.tcnt; t++) {
            const Term T = C.terms[R.toff + t];
            uint32_t klo = k & ((1u << T.h1) - 1), khi = k >> T.h1;
            K8d c = kmul(T.cs, kmul(C.hl[T.hl_off + klo], C.hh[T.hh_off + khi]));
            acc = kadd(acc, kmul(c, tfold[R.toff + t]));
        }
        uint32_t qi = (R.flags & 4) ? C.qidx[R.qd_off + k] : 0xffffffffu;
        if (qi != 0xffffffffu) {
            const K8d* qb = C.qd + ((uint64_t)qi << qbits);
            if (qbits == 0) acc = kadd(acc, qb[0]);
            else for (int p = 0; p < (1 << qbits); p++) if (!kzero_p(qb[p])) acc = kadd(acc, kmul(qb[p], e_full[p]));
        }
        const uint64_t at = (uint64_t)R.src * coarse + R.base_blk + k;
        q[at] = kcanon(acc);
        K8d c1 = kzero(); if (R.flags & 1) c1.c[0] = 1;
        chi[at] = c1;
    }
}

/// Factored dense query tables (table t = a_t ⊗ pw of its region) at level i:
/// out[t][y] = af[t][y >> pbits]·pf[pf_off[t] + (y & (2^pbits − 1))].
__global__ void dense_expand_kernel(const K8d* __restrict__ af, int afn, const K8d* __restrict__ pf, const uint32_t* __restrict__ pf_off,
                                    int pbits, int ybits, uint64_t ntab, K8d* __restrict__ out)
{
    const uint64_t tot = ntab << ybits;
    for (uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; g < tot; g += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t t = g >> ybits; const uint32_t y = (uint32_t)(g & ((1u << ybits) - 1));
        out[g] = kmul(af[t * afn + (y >> pbits)], pf[pf_off[t] + (y & ((1u << pbits) - 1))]);
    }
}

/// Round-0 per-term B tables: entry x of term s (at off[s], 2^nv[s] entries)
/// is Π_j (bit_j(x) ? r_j : 1 − r_j) over the term's index coordinates
/// coords[co[s] .. co[s] + nv[s]). pre[s] = Σ_{s' < s} 2^nv[s'].
__global__ void b_build_kernel(const K8d* __restrict__ coords, const uint32_t* __restrict__ co, const uint8_t* __restrict__ nv,
                               const uint32_t* __restrict__ off, const uint64_t* __restrict__ pre, uint32_t nseg, uint64_t total,
                               K8d* __restrict__ out)
{
    for (uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; g < total; g += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t lo = 0, hi = nseg;
        while (hi - lo > 1) { const uint32_t m = (lo + hi) >> 1; if (pre[m] <= g) lo = m; else hi = m; }
        const uint32_t x = (uint32_t)(g - pre[lo]);
        const K8d* r = coords + co[lo];
        K8d acc = kzero(); acc.c[0] = 1;
        for (int j = 0; j < nv[lo]; j++) {
            K8d f;
            if ((x >> j) & 1) f = r[j]; else { f = kzero(); f.c[0] = 1; f = ksub(f, r[j]); }
            acc = j == 0 ? f : kmul(acc, f);
        }
        out[off[lo] + x] = kcanon(acc);
    }
}

/// Index-round step of the per-term B tables: dst[off_s + y] = src[off_s + 2y]
/// + src[off_s + 2y + 1] for y < len_s (eq tables over the remaining index
/// bits: summing out the bound bit). pre[s] = Σ_{s' < s} len_{s'}.
__global__ void b_halve_kernel(const K8d* __restrict__ src, K8d* __restrict__ dst, const uint32_t* __restrict__ off,
                               const uint64_t* __restrict__ pre, uint32_t nseg, uint64_t total)
{
    for (uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; g < total; g += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t lo = 0, hi = nseg;   // last s with pre[s] <= g
        while (hi - lo > 1) { const uint32_t m = (lo + hi) >> 1; if (pre[m] <= g) lo = m; else hi = m; }
        const uint64_t y = g - pre[lo], o = off[lo];
        dst[o + y] = kcanon(kadd(src[o + 2 * y], src[o + 2 * y + 1]));
    }
}

// ------------------------------------------------------------- coarse rounds
/// One coarse round over len-entry tables per source (pairs k: (2k, 2k+1)).
__global__ void __launch_bounds__(160) coarse_round_kernel(
    const K8d* __restrict__ w, const K8d* __restrict__ q, const K8d* __restrict__ chi, uint32_t nsrc, uint64_t len,
    const K8d* __restrict__ g_range, const K8d* __restrict__ g_bit, const K8d* __restrict__ e_alpha, int d,
    K8d* __restrict__ partial)
{
    const int lane = threadIdx.x & 31, t = threadIdx.x >> 5;
    const u64 tq = (u64)t;
    K8d acc_r = kzero(), acc_q = kzero();
    const uint64_t half = len >> 1, total = half * nsrc;
    if (t <= d) {
        for (uint64_t g = blockIdx.x * 32ULL + lane; g < total; g += (uint64_t)gridDim.x * 32) {
            const uint64_t s = g / half, k = g % half;
            const K8d* ws = w + s * len;
            K8d w0 = ws[2 * k], w1 = ws[2 * k + 1];
            if (kzero_p(w0) && kzero_p(w1)) continue;
            const K8d* qs = q + s * len; const K8d* cs = chi + s * len;
            K8d dw = ksub(w1, w0), dq = ksub(qs[2 * k + 1], qs[2 * k]), dc = ksub(cs[2 * k + 1], cs[2 * k]);
            K8d wt, qt, ct;
            for (int a = 0; a < 8; a++) {
                wt.c[a] = mauth::fadd(w0.c[a], mauth::fmul(dw.c[a], tq));
                qt.c[a] = mauth::fadd(qs[2 * k].c[a], mauth::fmul(dq.c[a], tq));
                ct.c[a] = mauth::fadd(cs[2 * k].c[a], mauth::fmul(dc.c[a], tq));
            }
            K8d w2 = kmul(wt, wt), w3 = kmul(w2, wt);
            K8d r = kadd(kmul(g_range[s], ksub(w3, wt)), kmul(kmul(g_bit[s], ct), ksub(w2, wt)));
            acc_r = kadd(acc_r, kmul(e_alpha[k], r));
            acc_q = kadd(acc_q, kmul(qt, wt));
        }
    }
    K8d vr = kcanon(acc_r), vq = kcanon(acc_q);
    for (int off = 16; off > 0; off >>= 1) { vr = kadd(vr, shfl_down_k(vr, off)); vq = kadd(vq, shfl_down_k(vq, off)); }
    if (lane == 0 && t <= d) {
        partial[(size_t)blockIdx.x * NACC + t] = kcanon(vr);
        partial[(size_t)blockIdx.x * NACC + d + 1 + t] = kcanon(vq);
    }
}

/// Fold every per-source table with challenge r: out[s][k] = a + r (b − a).
__global__ void coarse_fold_kernel(const K8d* __restrict__ in, K8d* __restrict__ out, uint32_t nsrc, uint64_t len, K8d r) {
    const uint64_t half = len >> 1, total = half * nsrc;
    for (uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; g < total; g += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t s = g / half, k = g % half;
        K8d a = in[s * len + 2 * k], b = in[s * len + 2 * k + 1];
        out[s * half + k] = (kzero_p(a) && kzero_p(b)) ? kzero() : kcanon(kadd(a, kmul(r, ksub(b, a))));
    }
}

// ------------------------------------------------------------- one-hot v3
/// Plane round i < p of one-hot regions, warp per data block, lane t owns
/// plane bin t (t < 2^p ≤ 32). In round i the range/query sums only depend on
/// the bins C[t] = Σ_{idx: sel = t} eq(α_[p..B0), idx) and, per term,
/// CB_T[t] = Σ_{idx: sel = t} B_T[idx] (both independent of i). Each lane
/// accumulates polynomial coefficients in t:
///   range:  cr·H·(λ³e³ − λe) + cb·H·(λ²e² − λe),  H = eq(α_(i..p), t>>(i+1))·C[t]
///   query:  Σ_T c_T CB_T (A_T[0][pr] + t·dA_T[pr])·λe
/// with λ = 1 − t (bit 0) or t (bit 1). Coefficients (degree ≤ 3) are reduced
/// once per CUDA block: partial[block][0..3] range, [4..6] query.
__global__ void __launch_bounds__(128) onehot3_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0, int i,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C,
    const K8d* __restrict__ E1, const K8d* __restrict__ E2, const K8d* __restrict__ E3,
    const K8d* __restrict__ eai,      // per p: eq(α_[p..B0), idx), at offsets eai_off[p]
    const uint32_t* __restrict__ eai_off,
    const K8d* __restrict__ eapr,     // eq(α_(i..p), pr) per p, at offsets eapr_off[p]
    const uint32_t* __restrict__ eapr_off,
    K8d* __restrict__ partial)
{
    __shared__ K8d s_ct[4][MAXT];
    __shared__ K8d s_cr[4], s_cb[4];
    __shared__ uint32_t s_tcnt[4], s_toff[4], s_p[4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = blockIdx.x * 4 + warp, nw = gridDim.x * 4;
    K8d cr3 = kzero(), cr2 = kzero(), cr1 = kzero(), cr0 = kzero();   // range coefficients
    K8d cq2 = kzero(), cq1 = kzero(), cq0 = kzero();                  // query coefficients
    for (uint32_t bi = gw; bi < nblk; bi += nw) {
        const Blk b = blks[bi];
        const Region R = C.regions[b.region];
        const int p = (int)R.p, np = 1 << p;
        if (lane == 0) {
            const uint32_t mlo = (1u << C.ca_lo_bits) - 1;
            K8d ca = kmul(C.ca_lo[b.id & mlo], C.ca_hi[b.id >> C.ca_lo_bits]);
            s_cr[warp] = kmul(ca, C.g_range[b.src]);
            s_cb[warp] = (R.flags & 1) ? kmul(ca, C.g_bit[b.src]) : kzero();
            s_tcnt[warp] = R.tcnt; s_toff[warp] = R.toff; s_p[warp] = R.p;
        }
        if (lane < (int)R.tcnt) {
            const Term T = C.terms[R.toff + lane];
            uint32_t klo = b.k & ((1u << T.h1) - 1), khi = b.k >> T.h1;
            s_ct[warp][lane] = kmul(T.cs, kmul(C.hl[T.hl_off + klo], C.hh[T.hh_off + khi]));
        }
        __syncwarp();
        if (lane >= np) continue;
        // Bin t = lane: scan the block's indices.
        const int nidx = 1 << (b0 - p);
        const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
        const K8d* ea_i = eai + eai_off[p];
        K8d Cb = kzero();
        K8d CB[4] = {kzero(), kzero(), kzero(), kzero()};   // up to 4 structured terms here
        const int tc = (int)s_tcnt[warp] < 4 ? (int)s_tcnt[warp] : 4;
        bool any = false;
        for (int idx = 0; idx < nidx; idx++) {
            if (dig(dg, ((uint32_t)idx << p) + lane) != 1) continue;
            any = true;
            Cb = kadd(Cb, ea_i[idx]);
            for (int k = 0; k < tc; k++) {
                const Term T = C.terms[s_toff[warp] + k];
                CB[k] = kadd(CB[k], C.B[T.boff + idx]);
            }
        }
        if (!any) continue;
        const int t = lane;
        const int bit = (t >> i) & 1, low = t & ((1 << i) - 1), pr = t >> (i + 1);
        const K8d e = E1[low];
        const K8d H = kmul(eapr[eapr_off[p] + pr], Cb);
        // Range: X3·λ³ − X1r·λ + X2·λ² − X1b·λ
        const K8d X3 = kmul(kmul(s_cr[warp], H), E3[low]);
        const K8d X2 = kmul(kmul(s_cb[warp], H), E2[low]);
        const K8d X1 = kmul(kmul(kadd(s_cr[warp], s_cb[warp]), H), e);
        // Query: Σ_T c_T CB_T (a0 + t·da)·λ·e = λ·(Y0 + t·Y1)
        K8d Y0 = kzero(), Y1 = kzero();
        for (int k = 0; k < tc; k++) {
            const Term T = C.terms[s_toff[warp] + k];
            K8d cbk = kmul(s_ct[warp][k], CB[k]);
            K8d a0 = C.A[T.aoff + pr], a1 = C.A[T.aoff + (1 << T.pa) + pr];
            Y0 = kadd(Y0, kmul(cbk, a0));
            Y1 = kadd(Y1, kmul(cbk, ksub(a1, a0)));
        }
        Y0 = kmul(Y0, e); Y1 = kmul(Y1, e);
        if (bit == 1) {
            // λ = t: range = X3 t³ + X2 t² − X1 t; query = Y0 t + Y1 t²
            cr3 = kadd(cr3, X3); cr2 = kadd(cr2, X2); cr1 = ksub(cr1, X1);
            cq1 = kadd(cq1, Y0); cq2 = kadd(cq2, Y1);
        } else {
            // λ = 1 − t: λ³ = 1 − 3t + 3t² − t³, λ² = 1 − 2t + t².
            cr0 = kadd(cr0, ksub(kadd(X3, X2), X1));
            cr1 = kadd(cr1, kadd(ksub(kscale(X1, 1), kscale(X3, 3)), ksub(kzero(), kscale(X2, 2))));
            cr2 = kadd(cr2, kadd(kscale(X3, 3), X2));
            cr3 = ksub(cr3, X3);
            // (1 − t)(Y0 + t Y1) = Y0 + t (Y1 − Y0) − t² Y1
            cq0 = kadd(cq0, Y0); cq1 = kadd(cq1, ksub(Y1, Y0)); cq2 = ksub(cq2, Y1);
        }
    }
    // Block reduction of the 7 coefficients.
    __shared__ K8d red[4][7];
    K8d v[7] = {cr0, cr1, cr2, cr3, cq0, cq1, cq2};
    for (int k = 0; k < 7; k++) {
        K8d x = kcanon(v[k]);
        for (int off = 16; off > 0; off >>= 1) x = kadd(x, shfl_down_k(x, off));
        if (lane == 0) red[warp][k] = kcanon(x);
    }
    __syncthreads();
    if (threadIdx.x < 7) {
        K8d sacc = kzero();
        for (int w = 0; w < 4; w++) sacc = kadd(sacc, red[w][threadIdx.x]);
        partial[(size_t)blockIdx.x * 7 + threadIdx.x] = kcanon(sacc);
    }
}

// ------------------------------------------------------------- one-hot sums
/// One-hot regions, all plane rounds at once. With C[t] = Σ_{idx: sel = t}
/// eq(α_[p..B0), idx) and CB_T[t] = Σ_{idx: sel = t} B_T[idx] (B_T is fixed
/// while i < p), every round's one-hot contribution is a function of
///   G3[p][t] = Σ_b cr_b·C_b[t],  G2[p][t] = Σ_b cb_b·C_b[t],
///   W[T][t]  = Σ_b c_T(b)·CB_T,b[t].
/// Each warp takes a contiguous run of blocks, lane t owns bin t, and flushes
/// its sums on a region change as 32-bit halves into u64 atomics (16 per K).
/// bc_oh: block constants of this list in reverse order.
__device__ __forceinline__ void kacc_atomic(u64* dst, const K8d& v) {
    for (int w = 0; w < 8; w++) {
        atomicAdd(dst + 2 * w, v.c[w] & 0xffffffffULL);
        atomicAdd(dst + 2 * w + 1, v.c[w] >> 32);
    }
}
__global__ void __launch_bounds__(128) onehot_pre_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C, const K8d* __restrict__ bc_oh,
    const K8d* __restrict__ eai, const uint32_t* __restrict__ eai_off, uint32_t per_warp,
    u64* __restrict__ G, u64* __restrict__ W)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint64_t s0 = (uint64_t)(blockIdx.x * 4 + warp) * per_warp;
    const uint64_t e0 = s0 + per_warp < nblk ? s0 + per_warp : nblk;
    int cur = -1, cp = 0, ctc = 0;
    uint32_t ctoff = 0, cw = 0;   // cw: compact per-term bin base (one-hot regions keep it in qd_off)
    K8d g3 = kzero(), g2 = kzero(), w0 = kzero(), w1 = kzero(), w2 = kzero(), w3 = kzero();
    auto flush = [&]() {
        if (cur < 0 || lane >= (1 << cp)) return;
        kacc_atomic(G + ((size_t)(cp * 32 + lane) * 2 + 0) * 16, kcanon(g3));
        kacc_atomic(G + ((size_t)(cp * 32 + lane) * 2 + 1) * 16, kcanon(g2));
        if (ctc > 0) kacc_atomic(W + ((size_t)(cw + 0) * 32 + lane) * 16, kcanon(w0));
        if (ctc > 1) kacc_atomic(W + ((size_t)(cw + 1) * 32 + lane) * 16, kcanon(w1));
        if (ctc > 2) kacc_atomic(W + ((size_t)(cw + 2) * 32 + lane) * 16, kcanon(w2));
        if (ctc > 3) kacc_atomic(W + ((size_t)(cw + 3) * 32 + lane) * 16, kcanon(w3));
    };
    for (uint64_t bi = s0; bi < e0; bi++) {
        const Blk b = blks[bi];
        const Region R = C.regions[b.region];
        if (R.flags & 8) continue;
        if ((int)b.region != cur) {
            flush();
            g3 = g2 = w0 = w1 = w2 = w3 = kzero();
            cur = (int)b.region; cp = (int)R.p; ctoff = R.toff; cw = R.qd_off; ctc = R.tcnt < 4 ? (int)R.tcnt : 4;
        }
        const int p = cp;
        if (lane >= (1 << p)) continue;
        const int nidx = 1 << (b0 - p);
        const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
        const K8d* ea_i = eai + eai_off[p];
        K8d Cb = kzero(), CB0 = kzero(), CB1 = kzero(), CB2 = kzero(), CB3 = kzero();
        const Term* Ts = C.terms + ctoff;
        bool any = false;
        for (int idx = 0; idx < nidx; idx++) {
            if (dig(dg, ((uint32_t)idx << p) + lane) != 1) continue;
            any = true;
            Cb = kadd(Cb, ea_i[idx]);
            if (ctc > 0) CB0 = kadd(CB0, C.B[Ts[0].boff + idx]);
            if (ctc > 1) CB1 = kadd(CB1, C.B[Ts[1].boff + idx]);
            if (ctc > 2) CB2 = kadd(CB2, C.B[Ts[2].boff + idx]);
            if (ctc > 3) CB3 = kadd(CB3, C.B[Ts[3].boff + idx]);
        }
        if (!any) continue;
        const K8d* o = bc_oh + C.bco[C.oh_base + nblk - 1 - bi];
        g3 = kadd(g3, kmul(o[0], Cb));
        if (!kzero_p(o[1])) g2 = kadd(g2, kmul(o[1], Cb));
        if (ctc > 0) w0 = kadd(w0, kmul(o[2], CB0));
        if (ctc > 1) w1 = kadd(w1, kmul(o[3], CB1));
        if (ctc > 2) w2 = kadd(w2, kmul(o[4], CB2));
        if (ctc > 3) w3 = kadd(w3, kmul(o[5], CB3));
    }
    flush();
}

__device__ __forceinline__ K8d shfl_down_seg(const K8d& v, int off, int width) {
    K8d r; for (int w = 0; w < 8; w++) r.c[w] = __shfl_down_sync(0xffffffffu, v.c[w], off, width); return r;
}

// ------------------------------------------------------------- one-hot bins
/// Binned one-hot blocks (p = 4, ≤ 2 terms; region flag 8). One pass over the
/// digits gives what the plane rounds and round i = p need, so these blocks
/// never enter the round kernels at i ≤ p. With V_b[r] = ca(b)·eq(α_(p..B0), r):
///   GV[par][s]   = Σ_b Σ_{idx: sel = s, idx&1 = par} V_b[idx >> 1]
///                  (G3[s] = γ_range·Σ_par eqb(α_p, par)·GV[par][s], G2 likewise),
///   U[T][b][par][s] = Σ_blk c_T Σ_{idx: sel = s, idx&1 = par, (idx>>1)&1 = b} B5_T[idx >> 2]
///                  (B5_T the round-5 table; the host derives the round-4 and
///                  plane-round bins: U4[par][s] = Σ_b eqb(r_T1, b)·U[b][par][s],
///                  W_T[s] = Σ_par eqb(r_T0, par)·U4[par][s]),
///   J[s0][s1]    = Σ_b Σ_{r: (sel(2r), sel(2r+1)) = (s0, s1)} V_b[r],
///   J5[c]        = (Σ cr·ea5[r], Σ cb·ea5[r]) over round-5 pairs r (indices
///                  4r..4r+3) of quad class c = (s0, s1, s2, s3) (global atomics;
///                  each block's most frequent class by complement, Σ_r ea5[r] = 1).
/// GV and J are kept per (source, χ) run and flushed times γ_range, γ_bit:
/// out GV[0..1] = (γ_range·GV, γ_bit·GV), J[0..1] = (γ_range·J, γ_bit·J).
/// Warp per data block; lane (s, par) = (lane & 15, lane >> 4) owns class s
/// and index parity par; J rows s0 are owned by lanes 0..15.
constexpr int BIN_P = 4, BIN_NP = 1 << BIN_P;
constexpr int J5_REP = 8;   // replicas of the round-5 bins (spread hot-bin atomics; summed on the host)
__device__ unsigned long long g_bin_prof[8];
#define BPROF(k) do { if (prof && lane == 0) { long long _t = clock64(); atomicAdd(&g_bin_prof[k], (unsigned long long)(_t - _t0)); _t0 = _t; } } while (0)
__global__ void __launch_bounds__(128) onehot_bin_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C, const K8d* __restrict__ bc_oh,
    const K8d* __restrict__ ea1, const K8d* __restrict__ ea5, const K8d* __restrict__ B5, uint32_t chunk, int prof_dbg,
    u64* __restrict__ GV, u64* __restrict__ U, u64* __restrict__ J, u64* __restrict__ J5)
{
    extern __shared__ K8d bin_sh[];   // per warp: J (256 K), V (64 K), sel (128 B)
    // Debug (timing only; results wrong): bit 1 skips round 5, 2 the class loop, 4 the U products, 8 the V products.
    const int prof = prof_dbg & 1, dbg = prof_dbg >> 1;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cls = lane & 15, par = lane >> 4;
    J5 += (size_t)(blockIdx.x % J5_REP) * 2 * 65536 * 16;
    // Warps take interleaved chunks of consecutive blocks.
    const uint64_t gw = blockIdx.x * 4 + warp, nw = (uint64_t)gridDim.x * 4;
    K8d* Jw = bin_sh + (size_t)warp * (BIN_NP * BIN_NP + 64 + 2);
    K8d* V = Jw + BIN_NP * BIN_NP;
    uint8_t* sel = reinterpret_cast<uint8_t*>(V + 64);
    for (int j = lane; j < BIN_NP * BIN_NP; j += 32) Jw[j] = kzero();
    int jsrc = -1, jchi = 0;
    K8d gv = kzero();
    auto flush_j = [&]() {
        __syncwarp();
        if (jsrc < 0) return;
        const K8d gr = C.g_range[jsrc], gb = C.g_bit[jsrc];
        if (!kzero_p(gv)) {
            kacc_atomic(GV + ((size_t)(0 * 2 + par) * 32 + cls) * 16, kcanon(kmul(gr, gv)));
            if (jchi) kacc_atomic(GV + ((size_t)(1 * 2 + par) * 32 + cls) * 16, kcanon(kmul(gb, gv)));
        }
        gv = kzero();
        for (int j = lane; j < BIN_NP * BIN_NP; j += 32) {
            const K8d x = Jw[j];
            if (kzero_p(x)) continue;
            kacc_atomic(J + ((size_t)0 * BIN_NP * BIN_NP + j) * 16, kcanon(kmul(gr, x)));
            if (jchi) kacc_atomic(J + ((size_t)1 * BIN_NP * BIN_NP + j) * 16, kcanon(kmul(gb, x)));
            Jw[j] = kzero();
        }
        __syncwarp();
    };
    const uint32_t mlo = (1u << C.ca_lo_bits) - 1;
    const uint32_t pmask = par ? 0xAAAAAAAAu : 0x55555555u;
    long long _t0 = clock64();
    for (uint64_t ch = gw; ch * chunk < nblk; ch += nw)
    for (uint64_t bi = ch * chunk; bi < nblk && bi < (ch + 1) * chunk; bi++) {
        const Blk b = blks[bi];
        const Region R = C.regions[b.region];
        if (!(R.flags & 8)) continue;
        BPROF(7);
        const int chi = (int)(R.flags & 1);
        if ((int)b.src != jsrc || chi != jchi) { flush_j(); jsrc = (int)b.src; jchi = chi; }
        BPROF(0);
        // Selected plane of each of the 128 indices (16 bytes each).
        const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            const int idx = lane * 4 + q;
            // The index's 16 plane digits; the selected plane is the lowest nonzero.
            const uint32_t w = dig16(dg, (uint32_t)idx * 16);
            sel[idx] = (uint8_t)(w ? (__ffs(w) - 1) >> 1 : 0);
        }
        const K8d ca = (dbg & 8) ? C.ca_lo[b.id & mlo] : kmul(C.ca_lo[b.id & mlo], C.ca_hi[b.id >> C.ca_lo_bits]);
        V[lane] = (dbg & 8) ? ea1[lane] : kmul(ca, ea1[lane]);
        V[lane + 32] = (dbg & 8) ? ea1[lane + 32] : kmul(ca, ea1[lane + 32]);
        __syncwarp();
        BPROF(1);
        uint32_t mk[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const int sv = sel[c * 32 + lane];
            uint32_t mine = 0;
            #pragma unroll
            for (int s = 0; s < BIN_NP; s++) { const uint32_t m = __ballot_sync(0xffffffffu, sv == s); if (s == cls) mine = m; }
            mk[c] = mine & pmask;
        }
        // Dominant class (most indices): its bins follow by complement, so
        // no lane walks its (typically most) indices.
        int n = 0;
        #pragma unroll
        for (int c = 0; c < 4; c++) n += __popc(mk[c]);
        const int tot = n + __shfl_xor_sync(0xffffffffu, n, 16);
        const int dom = (int)(__reduce_max_sync(0xffffffffu, (unsigned)((tot << 4) | cls)) & 15);
        if (prof) {
            const int mx = __reduce_max_sync(0xffffffffu, (unsigned)n);
            const int m2 = __reduce_max_sync(0xffffffffu, (unsigned)(cls == dom ? 0 : n));
            if (lane == 0) { atomicAdd(&g_bin_prof[5], (unsigned long long)mx); atomicAdd(&g_bin_prof[6], (unsigned long long)m2); }
        }
        BPROF(2);
        // Block-local sum of this lane's class and parity; js: this lane's
        // J additions (pairs not both dominant).
        K8d gvb = kzero(), js = kzero();
        if (cls != dom && !(dbg & 2)) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                uint32_t m = mk[c];
                while (m) {
                    const int bt = __ffs(m) - 1; m &= m - 1;
                    const int idx = c * 32 + bt, r = idx >> 1;
                    const K8d vr = V[r];
                    gvb = kadd(gvb, vr);
                    // Pair (s0, s1): row s0 ≠ dom by lane (s0, 0); (dom, s1) by lane (s1, 1).
                    if (par == 0) { K8d& jj = Jw[cls * BIN_NP + sel[idx + 1]]; jj = kadd(jj, vr); js = kadd(js, vr); }
                    else if (sel[idx - 1] == dom) { K8d& jj = Jw[dom * BIN_NP + cls]; jj = kadd(jj, vr); js = kadd(js, vr); }
                }
            }
        }
        // Complements: Σ_par-lanes gvb = Σ_r V[r] = ca (Σ ea1 = 1), Σ J = ca.
        const K8d tvb = ca;
        js = kcanon(js);
        for (int off = 16; off > 0; off >>= 1) js = kadd(js, shfl_down_k(js, off));
        js = kcanon(js);
        K8d jsb;
        for (int w = 0; w < 8; w++) jsb.c[w] = __shfl_sync(0xffffffffu, js.c[w], 0);
        // Half (parity) totals of the non-dominant lanes; Σ_{y5} B5[y5] = 1 per (b, par).
        auto half_sum = [&](K8d x) -> K8d {
            x = kcanon(x);
            for (int off = 8; off > 0; off >>= 1) x = kadd(x, shfl_down_seg(x, off, 16));
            K8d h; for (int w = 0; w < 8; w++) h.c[w] = __shfl_sync(0xffffffffu, x.c[w], lane & 16);
            return h;
        };
        const K8d hg = half_sum(gvb);
        if (cls == dom) {
            gvb = ksub(tvb, hg);
            if (par == 0) { K8d& jj = Jw[dom * BIN_NP + dom]; jj = kadd(jj, ksub(tvb, jsb)); }
        }
        gv = kadd(gv, gvb);
        __syncwarp();
        BPROF(3);
    }
    flush_j();
}

/// Binned one-hot blocks, part B: the query moments U (see above) and the
/// round-5 range bins J5. Per block the round-5 tables are pre-scaled,
/// B5c_T = c_T(b)·B5_T (lane y computes entry y), so the class loop only adds
/// (dominant class by complement: Σ_y B5c_T[y] = c_T); the run sums of U live
/// in shared memory (RU[term·2 + b][lane]).
__global__ void __launch_bounds__(128, 3) onehot_q5_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C, const K8d* __restrict__ bc_oh,
    const K8d* __restrict__ ea5, const K8d* __restrict__ B5, uint32_t chunk,
    u64* __restrict__ U, u64* __restrict__ J5)
{
    extern __shared__ K8d q5_sh[];   // per warp: B5c[2][32], RU[4][32], sel (128 B)
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cls = lane & 15, par = lane >> 4;
    J5 += (size_t)(blockIdx.x % J5_REP) * 2 * 65536 * 16;
    const uint64_t gw = blockIdx.x * 4 + warp, nw = (uint64_t)gridDim.x * 4;
    K8d* B5c = q5_sh + (size_t)warp * (64 + 128 + 2);
    K8d* RU = B5c + 64;
    uint8_t* sel = reinterpret_cast<uint8_t*>(RU + 128);
    for (int q = 0; q < 4; q++) RU[q * 32 + lane] = kzero();
    int cur = -1, ctc = 0;
    uint32_t cw = 0;
    auto flush = [&]() {
        if (cur < 0) return;
        for (int q = 0; q < 2 * ctc; q++) {
            const K8d x = RU[q * 32 + lane];
            if (!kzero_p(x)) kacc_atomic(U + (((size_t)(cw + (q >> 1)) * 4 + (q & 1) * 2 + par) * 32 + cls) * 16, kcanon(x));
            RU[q * 32 + lane] = kzero();
        }
    };
    int d5 = -1; K8d d5r = kzero(), d5b = kzero();
    auto flush5 = [&]() {
        if (d5 >= 0 && lane == 0) {
            kacc_atomic(J5 + ((size_t)0 * 65536 + d5) * 16, kcanon(d5r));
            kacc_atomic(J5 + ((size_t)1 * 65536 + d5) * 16, kcanon(d5b));
        }
        d5 = -1; d5r = d5b = kzero();
    };
    const uint32_t pmask = par ? 0xAAAAAAAAu : 0x55555555u;
    for (uint64_t ch = gw; ch * chunk < nblk; ch += nw)
    for (uint64_t bi = ch * chunk; bi < nblk && bi < (ch + 1) * chunk; bi++) {
        const Blk b = blks[bi];
        const Region R = C.regions[b.region];
        if (!(R.flags & 8)) continue;
        if ((int)b.region != cur) { flush(); cur = (int)b.region; cw = R.qd_off; ctc = (int)R.tcnt; }
        const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            const int idx = lane * 4 + q;
            // The index's 16 plane digits; the selected plane is the lowest nonzero.
            const uint32_t w = dig16(dg, (uint32_t)idx * 16);
            sel[idx] = (uint8_t)(w ? (__ffs(w) - 1) >> 1 : 0);
        }
        const K8d* o = bc_oh + C.bco[C.oh_base + nblk - 1 - bi];
        for (int t = 0; t < ctc; t++) B5c[t * 32 + lane] = kmul(o[2 + t], B5[(size_t)(cw + t) * 32 + lane]);
        __syncwarp();
        {
            // Round-5 pair r = lane: quad class of indices 4r..4r+3.
            const int c5 = (sel[4 * lane] << 12) | (sel[4 * lane + 1] << 8) | (sel[4 * lane + 2] << 4) | sel[4 * lane + 3];
            const unsigned peers = __match_any_sync(0xffffffffu, c5);
            const unsigned best = __reduce_max_sync(0xffffffffu, ((unsigned)__popc(peers) << 16) | (unsigned)c5);
            const int dom5 = (int)(best & 0xffff);
            const K8d cr = o[0], cb = o[1];
            const bool hb = !kzero_p(cb);
            K8d vr = kzero(), vb = kzero();
            if (c5 != dom5) {
                vr = kmul(cr, ea5[lane]);
                kacc_atomic(J5 + ((size_t)0 * 65536 + c5) * 16, kcanon(vr));
                if (hb) { vb = kmul(cb, ea5[lane]); kacc_atomic(J5 + ((size_t)1 * 65536 + c5) * 16, kcanon(vb)); }
            }
            vr = kcanon(vr); vb = kcanon(vb);
            for (int off = 16; off > 0; off >>= 1) { vr = kadd(vr, shfl_down_k(vr, off)); vb = kadd(vb, shfl_down_k(vb, off)); }
            if (lane == 0) {
                if (dom5 != d5) { flush5(); d5 = dom5; }
                d5r = kadd(d5r, ksub(cr, vr));
                if (hb) d5b = kadd(d5b, ksub(cb, vb));
            }
            d5 = __shfl_sync(0xffffffffu, d5, 0);
        }
        uint32_t mk[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const int sv = sel[c * 32 + lane];
            uint32_t mine = 0;
            #pragma unroll
            for (int s = 0; s < BIN_NP; s++) { const uint32_t m = __ballot_sync(0xffffffffu, sv == s); if (s == cls) mine = m; }
            mk[c] = mine & pmask;
        }
        int n = 0;
        #pragma unroll
        for (int c = 0; c < 4; c++) n += __popc(mk[c]);
        const int tot = n + __shfl_xor_sync(0xffffffffu, n, 16);
        const int dom = (int)(__reduce_max_sync(0xffffffffu, (unsigned)((tot << 4) | cls)) & 15);
        K8d ub[4] = {kzero(), kzero(), kzero(), kzero()};   // ub[term·2 + b]
        if (cls != dom) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                uint32_t m = mk[c];
                while (m) {
                    const int bt = __ffs(m) - 1; m &= m - 1;
                    const int idx = c * 32 + bt, y5 = idx >> 2, bb = (idx >> 1) & 1;
                    for (int t = 0; t < ctc; t++) ub[t * 2 + bb] = kadd(ub[t * 2 + bb], B5c[t * 32 + y5]);
                }
            }
        }
        // Dominant: Σ over the half (parity) of B5c sums is c_T per (term, b).
        for (int q = 0; q < 2 * ctc; q++) {
            K8d x = kcanon(ub[q]);
            for (int off = 8; off > 0; off >>= 1) x = kadd(x, shfl_down_seg(x, off, 16));
            K8d h; for (int w = 0; w < 8; w++) h.c[w] = __shfl_sync(0xffffffffu, x.c[w], lane & 16);
            const K8d v = cls == dom ? ksub(o[2 + (q >> 1)], h) : ub[q];
            RU[q * 32 + lane] = kadd(RU[q * 32 + lane], v);
        }
        __syncwarp();
    }
    flush();
    flush5();
}

// ------------------------------------------------------------- index rounds
/// Late block-local rounds (2^{b0−i} ≤ 128, 2^i ≥ 16): one warp per data
/// block, folded digits F and query weights QF in a per-warp shared slice.
///   A: digits → F[y] = Σ_p e_pre[p]·d[(y << i) + p]; lane owns 2^b0/32
///      contiguous bytes (16-byte loads), lanes sharing a y reduce.
///   B: QF[y] = folded dense table (level i) + Σ_T c_T B_T[ir] A_T[side][pr];
///      lanes split pairs, then terms.
///   C: lanes 0..29 (pair slot pl = lane/3, point t ∈ {0, 2, 3}) add the range
///      sum at t and (t < 3) the query sum at t. Two accumulators per lane.
/// The host gets t = 1 from the previous round's claims and extrapolates
/// range (degree 3) and query (degree 2) to 5 points.
/// partial[block][t] range at t (slot 1 zero), [4 + t] query at t (slot 5 zero).
constexpr int IDX_WARPS = 4;
constexpr int IDX_MAXY = 128;
template <int MINB>
__global__ void __launch_bounds__(128, MINB) index_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0, int i,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C, const K8d* __restrict__ bc,
    const K8d* __restrict__ e_pre, const K8d* __restrict__ e_alpha, K8d* __restrict__ partial)
{
    extern __shared__ K8d dyn_sh[];   // [IDX_WARPS][2][ny]
    __shared__ K8d s_c[IDX_WARPS][2 + MAXT];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = blockIdx.x * IDX_WARPS + warp, nw = gridDim.x * IDX_WARPS;
    const int ny = 1 << (b0 - i), nrest = ny >> 1;
    K8d* Fw = dyn_sh + (size_t)warp * 2 * ny;
    K8d* QFw = Fw + ny;
    // Phase A geometry.
    const int P = 1 << (b0 - 5);                       // bytes per lane
    const int span = P < (1 << i) ? P : (1 << i);      // bytes of one y per lane
    const int G = (1 << i) > P ? (1 << i) / P : 1;     // lanes per y
    const int nys = P >> (31 - __clz(span));           // ys per lane
    // Phase B geometry.
    const int GB = nrest >= 32 ? 1 : 32 / nrest;
    // Phase C slot.
    const int tl = lane < 30 ? (lane % 3 == 0 ? 0 : lane % 3 + 1) : 0;
    const int pl = lane < 30 ? lane / 3 : 1 << 20;
    const u64 tq = (u64)tl;
    K8d ar = kzero(), aq = kzero();
    for (uint32_t bi = gw; bi < nblk; bi += nw) {
        const Blk b = blks[bi];
        uint32_t tcnt = 0, toff = 0, qd = 0xffffffffu;
        if (b.region != 0xffffffffu) {
            const Region R = C.regions[b.region];
            tcnt = R.tcnt; toff = R.toff;
            if (R.flags & 4) { uint32_t qi = C.qidx[R.qd_off + b.k]; qd = qi == 0xffffffffu ? 0xffffffffu : (qi << (b0 - i)); }
        }
        const K8d* o = bc + C.bco[bi];
        for (int k = lane; k < 2 + (int)tcnt; k += 32) s_c[warp][k] = o[k];
        // ---- A: fold the digits.
        const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
        const uint32_t dg0 = (uint32_t)lane * P;
        for (int s = 0; s < nys; s++) {
            const int start = lane * P + s * span;
            const int poff = start & ((1 << i) - 1);
            K8d acc = kzero();
            for (int c = 0; c < span; c += 16) {
                // 16 digits: each nonzero 2-bit field is +1 (01) or -1 (11).
                uint32_t x = dig16(dg, dg0 + s * span + c);
                while (x) {
                    const int fd = (__ffs(x) - 1) >> 1;
                    const uint32_t code = (x >> (2 * fd)) & 3u;
                    x &= ~(3u << (2 * fd));
                    const K8d e = e_pre[poff + c + fd];
                    acc = code == 1u ? kadd(acc, e) : ksub(acc, e);
                }
            }
            if (G > 1) {
                acc = kcanon(acc);
                for (int off = G >> 1; off > 0; off >>= 1) acc = kadd(acc, shfl_down_seg(acc, off, G));
            }
            if (lane % G == 0) Fw[start >> i] = kcanon(acc);
        }
        // ---- B: query weights.
        for (int y = lane; y < ny; y += 32) QFw[y] = qd != 0xffffffffu ? C.qd[qd + y] : kzero();
        __syncwarp();
        if (tcnt > 0) {
            for (int r0 = 0; r0 < nrest; r0 += 32 / GB) {
                const int rest = r0 + lane / GB, gl = lane % GB;
                K8d q0 = kzero(), q1 = kzero();
                if (rest < nrest) {
                    for (int k = gl; k < (int)tcnt; k += GB) {
                        const Term T = C.terms[toff + k];
                        const int pr = rest & ((1 << T.pa) - 1), ir = rest >> T.pa;
                        const K8d bv = kmul(C.B[T.boff + ir], s_c[warp][2 + k]);
                        q0 = kadd(q0, kmul(C.A[T.aoff + pr], bv));
                        q1 = kadd(q1, kmul(C.A[T.aoff + (1 << T.pa) + pr], bv));
                    }
                }
                if (GB > 1) {
                    q0 = kcanon(q0); q1 = kcanon(q1);
                    for (int off = GB >> 1; off > 0; off >>= 1) { q0 = kadd(q0, shfl_down_seg(q0, off, GB)); q1 = kadd(q1, shfl_down_seg(q1, off, GB)); }
                }
                if (rest < nrest && gl == 0) {
                    QFw[2 * rest] = kcanon(kadd(QFw[2 * rest], q0));
                    QFw[2 * rest + 1] = kcanon(kadd(QFw[2 * rest + 1], q1));
                }
            }
            __syncwarp();
        }
        // ---- C: evaluations; the block's range constants are applied once:
        // Σ ea(cr(w³ − w) + cb(w² − w)) = cr·Σ(ea w³ − ea w) + cb·Σ(ea w² − ea w).
        const K8d cr = s_c[warp][0], cb = s_c[warp][1];
        K8d b3 = kzero(), b2 = kzero();
        for (int rest = pl; rest < nrest; rest += 10) {
            const K8d w0 = Fw[2 * rest], w1 = Fw[2 * rest + 1];
            if (kzero_p(w0) && kzero_p(w1)) continue;
            const K8d dw = ksub(w1, w0);
            K8d wt;
            for (int a = 0; a < 8; a++) wt.c[a] = mauth::fadd(w0.c[a], mauth::fmul(dw.c[a], tq));
            const K8d ew = kmul(e_alpha[rest], wt), ew2 = kmul(ew, wt), ew3 = kmul(ew2, wt);
            b3 = kadd(b3, ksub(ew3, ew));
            b2 = kadd(b2, ksub(ew2, ew));
            if (tl < 3) {
                const K8d q0 = QFw[2 * rest], q1 = QFw[2 * rest + 1];
                const K8d dq = ksub(q1, q0);
                K8d qt;
                for (int a = 0; a < 8; a++) qt.c[a] = mauth::fadd(q0.c[a], mauth::fmul(dq.c[a], tq));
                aq = kadd(aq, kmul(qt, wt));
            }
        }
        if (nrest > pl) ar = kadd(ar, kadd(kmul(cr, b3), kmul(cb, b2)));
        __syncwarp();
    }
    // Lanes with equal t reduce through shared memory (once per kernel; the
    // dynamic slice holds at least IDX_WARPS·32·2 K).
    __syncthreads();
    K8d (*red)[2] = reinterpret_cast<K8d (*)[2]>(dyn_sh);
    red[threadIdx.x][0] = kcanon(ar); red[threadIdx.x][1] = kcanon(aq);
    __syncthreads();
    if (threadIdx.x < 7) {
        const int slot = threadIdx.x;
        const int tt = slot < 4 ? slot : slot - 4, which = slot < 4 ? 0 : 1;
        K8d sacc = kzero();
        if (tt != 1 && !(which == 1 && tt == 3))
            for (int l = 0; l < IDX_WARPS * 32; l++) {
                const int ln = l & 31;
                if (ln >= 30) continue;
                const int lt = ln % 3 == 0 ? 0 : ln % 3 + 1;
                if (lt == tt) sacc = kcanon(kadd(sacc, red[l][which]));
            }
        partial[(size_t)blockIdx.x * 7 + slot] = kcanon(sacc);
    }
}

// ------------------------------------------------------------- range bins, round 1
/// Round-1 range sums of generic blocks from digit classes. Round-1 pair r
/// covers digits 4r..4r+3 (all in {−1, 0, 1}); its w_t depends only on the
/// class c = Σ_j (d_j + 1)·3^j (81 classes), so
///   R[0][c] = Σ_b cr_b Σ_{r: class c} eq(α_(1..B0), r),  R[1][c] likewise with cb_b,
/// and the host evaluates Σ_c R·(w³ − w), R·(w² − w). Warp per block: a
/// per-warp histogram of ea (u64 halves, shared atomics), then each lane
/// scales its classes by cr, cb into per-warp run bins (flushed once).
constexpr int RB1_C = 81;
__global__ void __launch_bounds__(128) range_bin1_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C, const K8d* __restrict__ bc,
    const K8d* __restrict__ ea1, u64* __restrict__ R)
{
    extern __shared__ u64 rb_sh[];   // per warp: H[81][16] u64, run[2][81] K
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    u64* H = rb_sh + (size_t)warp * (RB1_C * 16 + 2 * RB1_C * 8);
    K8d* run = reinterpret_cast<K8d*>(H + RB1_C * 16);
    for (int j = lane; j < RB1_C * 16; j += 32) H[j] = 0;
    for (int j = lane; j < 2 * RB1_C; j += 32) run[j] = kzero();
    __syncwarp();
    const uint64_t gw = blockIdx.x * 4 + warp, nw = (uint64_t)gridDim.x * 4;
    const int npair = 1 << (b0 - 2);
    for (uint64_t bi = gw; bi < nblk; bi += nw) {
        const Blk b = blks[bi];
        const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
        for (int r = lane; r < npair; r += 32) {
            // Four digits in one byte.
            const uint32_t x = __ldg(dg + r);
            // Every digit must be ternary: code 10 is not (R[2·81] counts violations).
            if (((x >> 1) & ~x & 0x55u) != 0) { atomicAdd(R + 2 * RB1_C * 16, 1ULL); continue; }
            int c = 0, m = 1;
            #pragma unroll
            for (int j = 0; j < 4; j++) { const uint32_t f = (x >> (2 * j)) & 3u; c += (f == 1u ? 2 : f == 3u ? 0 : 1) * m; m *= 3; }
            if (c == 40) continue;   // all-zero digits: w ≡ 0
            const K8d e = ea1[r];
            u64* h = H + (size_t)c * 16;
            #pragma unroll
            for (int w = 0; w < 8; w++) { atomicAdd(h + 2 * w, e.c[w] & 0xffffffffULL); atomicAdd(h + 2 * w + 1, e.c[w] >> 32); }
        }
        __syncwarp();
        const K8d* o = bc + C.bco[bi];
        const K8d cr = o[0], cb = o[1];
        const bool hb = !kzero_p(cb);
        for (int c = lane; c < RB1_C; c += 32) {
            u64* h = H + (size_t)c * 16;
            bool nz = false;
            #pragma unroll
            for (int w = 0; w < 16; w++) nz |= h[w] != 0;
            if (!nz) continue;
            K8d v;
            #pragma unroll
            for (int w = 0; w < 8; w++) { v.c[w] = mauth::fadd(h[2 * w], mauth::fmul(h[2 * w + 1], 1ULL << 32)); h[2 * w] = 0; h[2 * w + 1] = 0; }
            run[c] = kadd(run[c], kmul(cr, v));
            if (hb) run[RB1_C + c] = kadd(run[RB1_C + c], kmul(cb, v));
        }
        __syncwarp();
    }
    for (int j = lane; j < 2 * RB1_C; j += 32) if (!kzero_p(run[j])) kacc_atomic(R + (size_t)j * 16, kcanon(run[j]));
}

// ------------------------------------------------------------- late index rounds
/// Index rounds with few pairs per block (2^{b0−i} ≤ 8): a warp takes K =
/// 10/nrest blocks at once so its 30 evaluation lanes stay busy. Slot sl =
/// lane/3 (block kk = sl/nrest, pair rest = sl%nrest); in phase B the slot's
/// three lanes split the terms, in phase C they evaluate t ∈ {0, 2, 3}.
/// Phase A (digit fold) is index_kernel's, per block of the batch. Output as
/// index_kernel.
constexpr int IB_SLOTS = 10;
__global__ void __launch_bounds__(128, 4) index_batch_kernel(
    const uint8_t* __restrict__ raw, uint64_t src_len, int b0, int i,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C, const K8d* __restrict__ bc,
    const K8d* __restrict__ e_pre, const K8d* __restrict__ e_alpha, K8d* __restrict__ partial)
{
    extern __shared__ K8d dyn_sh[];   // per warp: F[2·IB_SLOTS], QF[2·IB_SLOTS]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ny = 1 << (b0 - i), nrest = ny >> 1;
    const int K = IB_SLOTS / nrest;
    const uint64_t gw = blockIdx.x * IDX_WARPS + warp, nw = (uint64_t)gridDim.x * IDX_WARPS;
    K8d* Fw = dyn_sh + (size_t)warp * 4 * IB_SLOTS;
    K8d* QFw = Fw + 2 * IB_SLOTS;
    const int P = 1 << (b0 - 5);
    const int span = P < (1 << i) ? P : (1 << i);
    const int G = (1 << i) > P ? (1 << i) / P : 1;
    const int nys = P >> (31 - __clz(span));
    const int sl = lane < 30 ? lane / 3 : IB_SLOTS, sub = lane % 3;
    const int tl = lane < 30 ? (sub == 0 ? 0 : sub + 1) : 0;
    const u64 tq = (u64)tl;
    const int kk = sl / nrest, rest = sl % nrest;
    K8d ar = kzero(), aq = kzero();
    for (uint64_t bb = gw * K; bb < nblk; bb += nw * K) {
        const int nb = nblk - bb < (uint64_t)K ? (int)(nblk - bb) : K;
        for (int q = 0; q < nb; q++) {
            const Blk b = blks[bb + q];
            uint32_t qd = 0xffffffffu;
            if (b.region != 0xffffffffu) {
                const Region R = C.regions[b.region];
                if (R.flags & 4) { uint32_t qi = C.qidx[R.qd_off + b.k]; qd = qi == 0xffffffffu ? 0xffffffffu : (qi << (b0 - i)); }
            }
            K8d* F = Fw + q * ny;
            const uint8_t* dg = blk_bytes(raw, src_len, b, b0);
            const uint32_t dg0 = (uint32_t)lane * P;
            for (int s = 0; s < nys; s++) {
                const int start = lane * P + s * span;
                const int poff = start & ((1 << i) - 1);
                K8d acc = kzero();
                for (int c = 0; c < span; c += 16) {
                    // 16 digits: each nonzero 2-bit field is +1 (01) or -1 (11).
                    uint32_t x = dig16(dg, dg0 + s * span + c);
                    while (x) {
                        const int fd = (__ffs(x) - 1) >> 1;
                        const uint32_t code = (x >> (2 * fd)) & 3u;
                        x &= ~(3u << (2 * fd));
                        const K8d e = e_pre[poff + c + fd];
                        acc = code == 1u ? kadd(acc, e) : ksub(acc, e);
                    }
                }
                if (G > 1) {
                    acc = kcanon(acc);
                    for (int off = G >> 1; off > 0; off >>= 1) acc = kadd(acc, shfl_down_seg(acc, off, G));
                }
                if (lane % G == 0) F[start >> i] = kcanon(acc);
            }
            if (lane < ny) QFw[q * ny + lane] = qd != 0xffffffffu ? C.qd[qd + lane] : kzero();
        }
        __syncwarp();
        // ---- B: the slot's three lanes split its block's terms.
        const bool live = sl < IB_SLOTS && kk < nb;
        K8d q0 = kzero(), q1 = kzero();
        const K8d* o = nullptr;
        if (live) {
            const Blk b = blks[bb + kk];
            o = bc + C.bco[bb + kk];
            if (b.region != 0xffffffffu) {
                const Region R = C.regions[b.region];
                for (int k = sub; k < (int)R.tcnt; k += 3) {
                    const Term T = C.terms[R.toff + k];
                    const int pr = rest & ((1 << T.pa) - 1), ir = rest >> T.pa;
                    const K8d bv = kmul(C.B[T.boff + ir], o[2 + k]);
                    q0 = kadd(q0, kmul(C.A[T.aoff + pr], bv));
                    q1 = kadd(q1, kmul(C.A[T.aoff + (1 << T.pa) + pr], bv));
                }
            }
        }
        q0 = kcanon(q0); q1 = kcanon(q1);
        K8d x0, x1, y0, y1;
        for (int w = 0; w < 8; w++) {
            x0.c[w] = __shfl_down_sync(0xffffffffu, q0.c[w], 1); x1.c[w] = __shfl_down_sync(0xffffffffu, q1.c[w], 1);
            y0.c[w] = __shfl_down_sync(0xffffffffu, q0.c[w], 2); y1.c[w] = __shfl_down_sync(0xffffffffu, q1.c[w], 2);
        }
        if (live && sub == 0) {
            K8d* QF = QFw + kk * ny;
            QF[2 * rest] = kcanon(kadd(QF[2 * rest], kadd(q0, kadd(x0, y0))));
            QF[2 * rest + 1] = kcanon(kadd(QF[2 * rest + 1], kadd(q1, kadd(x1, y1))));
        }
        __syncwarp();
        // ---- C: evaluations.
        if (live) {
            const K8d* F = Fw + kk * ny;
            const K8d w0 = F[2 * rest], w1 = F[2 * rest + 1];
            if (!(kzero_p(w0) && kzero_p(w1))) {
                const K8d cr = o[0], cb = o[1];
                const K8d dw = ksub(w1, w0);
                K8d wt;
                for (int a = 0; a < 8; a++) wt.c[a] = mauth::fadd(w0.c[a], mauth::fmul(dw.c[a], tq));
                const K8d ew = kmul(e_alpha[rest], wt);
                const K8d ew2 = kmul(ew, wt), ew3 = kmul(ew2, wt);
                // e·(cr(w³ − w) + cb(w² − w))
                ar = kadd(ar, kadd(kmul(cr, ksub(ew3, ew)), kmul(cb, ksub(ew2, ew))));
                if (tl < 3) {
                    const K8d* QF = QFw + kk * ny;
                    const K8d qa = QF[2 * rest], dq = ksub(QF[2 * rest + 1], qa);
                    K8d qt;
                    for (int a = 0; a < 8; a++) qt.c[a] = mauth::fadd(qa.c[a], mauth::fmul(dq.c[a], tq));
                    aq = kadd(aq, kmul(qt, wt));
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();
    K8d (*red)[2] = reinterpret_cast<K8d (*)[2]>(dyn_sh);
    red[threadIdx.x][0] = kcanon(ar); red[threadIdx.x][1] = kcanon(aq);
    __syncthreads();
    if (threadIdx.x < 7) {
        const int slot = threadIdx.x;
        const int tt = slot < 4 ? slot : slot - 4, which = slot < 4 ? 0 : 1;
        K8d sacc = kzero();
        if (tt != 1 && !(which == 1 && tt == 3))
            for (int l = 0; l < IDX_WARPS * 32; l++) {
                const int ln = l & 31;
                if (ln >= 30) continue;
                const int lt = ln % 3 == 0 ? 0 : ln % 3 + 1;
                if (lt == tt) sacc = kcanon(kadd(sacc, red[l][which]));
            }
        partial[(size_t)blockIdx.x * 7 + slot] = kcanon(sacc);
    }
}

}  // namespace mlink2

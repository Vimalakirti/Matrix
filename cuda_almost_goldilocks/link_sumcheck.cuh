/**
 * GPU kernels for the packed PCS link sumcheck (see zk-torch-4 src/pcs/link.rs).
 *
 * Per round the prover computes
 *
 *   S_i(k) = sum_e gamma_e * sum_idx eq_suffix[idx] * R(w_e,k[idx])   k = 0..3
 *   E_i(k) =         sum_e sum_idx omega_e,k[idx] * w_e,k[idx]        k = 0..2
 *
 * with R(u) = u^3 - u, and the interpolants w_k = w_lo + k*w_d obtained by
 * repeated addition rather than a multiply each.
 *
 * The two halves ship separately because their degrees differ (Gruen's
 * factorization drops the norm half from 5 evaluations to 4 by pulling the
 * eq(t, alpha_i) factor out for the verifier to reapply).
 *
 * Parallel decomposition mirrors the multi-GPU one exactly: the round message
 * is a sum over commitments, so a block owns (commitment, index-chunk) and the
 * per-commitment tag is applied once at block level. Sharding across devices is
 * then just a partition of gridDim.y with a 7-element all-reduce.
 *
 * Support skipping: where a witness entry and its partner are both zero the
 * contribution to both halves is identically zero (R(0) = 0, omega*0 = 0). On
 * CPU that skip is per element; here it is effectively per warp, since a warp
 * only retires early when all its lanes are zero. One-hot Shout advice has
 * nonzeros spaced 2^table_commit_log apart, so whole warps do skip in practice,
 * but the win is bounded by warp width rather than by raw density.
 */

#ifndef ALMOST_LINK_SUMCHECK_CUH
#define ALMOST_LINK_SUMCHECK_CUH

#include "almost_goldilocks.cuh"
#include "almost_extension.cuh"

namespace link_sumcheck {

typedef AlmostGoldilocksExt2 E2;

constexpr int NORM_EVALS = 4;
constexpr int EVAL_EVALS = 3;
constexpr int MSG_SLOTS  = NORM_EVALS + EVAL_EVALS;   // 7
constexpr int BLOCK      = 256;

__device__ __forceinline__ E2 e2_zero() {
    return E2((uint64_t)0, (uint64_t)0);
}

__device__ __forceinline__ bool e2_is_zero(const E2& v) {
    return v.c[0].value == 0 && v.c[1].value == 0;
}

/** R(u) = u^3 - u. Vanishes exactly on {-1, 0, 1}. */
__device__ __forceinline__ E2 range_poly(const E2& u) {
    E2 u2 = aext2_mul(u, u);
    E2 u3 = aext2_mul(u2, u);
    return aext2_sub(u3, u);
}

/**
 * Per-block partial round message.
 *
 *   gridDim  = (chunks, n_commit)
 *   blockDim = BLOCK
 *   partial  = [n_commit][chunks][MSG_SLOTS]
 *
 * `first_round` skips the k = 0 and k = 1 norm evaluations: on the first round
 * the witness still holds committed coefficients, which are in {-1, 0, 1}, so R
 * vanishes there.
 */
__global__ void link_round_kernel(
    const E2* __restrict__ w,          // [n_commit][stride]
    const E2* __restrict__ omega,      // [n_commit][stride]
    const E2* __restrict__ eq_suffix,  // [half]
    const E2* __restrict__ tags,       // [n_commit]
    uint64_t stride,
    uint64_t omega_stride,
    uint64_t half,
    int      first_round,
    E2*      __restrict__ partial
) {
    const uint64_t e     = blockIdx.y;
    const uint64_t chunk = blockIdx.x;
    const uint64_t nchunk = gridDim.x;
    const int tid = threadIdx.x;

    const E2* __restrict__ we = w + e * stride;
    const E2* __restrict__ oe = omega + e * omega_stride;

    E2 acc[MSG_SLOTS];
#pragma unroll
    for (int k = 0; k < MSG_SLOTS; k++) acc[k] = e2_zero();

    // Grid-stride within this commitment's index range.
    for (uint64_t idx = chunk * BLOCK + tid; idx < half; idx += nchunk * BLOCK) {
        E2 w_lo = we[idx];
        E2 w_hi = we[half + idx];
        if (e2_is_zero(w_lo) && e2_is_zero(w_hi)) continue;

        E2 w_d = aext2_sub(w_hi, w_lo);
        E2 o_lo = oe[idx];
        E2 o_d  = aext2_sub(oe[half + idx], o_lo);
        E2 eqs  = eq_suffix[idx];

        // Interpolants by repeated addition.
        E2 w0 = w_lo;
        E2 w1 = aext2_add(w0, w_d);
        E2 w2 = aext2_add(w1, w_d);
        E2 w3 = aext2_add(w2, w_d);

        if (!first_round) {
            acc[0] = aext2_add(acc[0], aext2_mul(eqs, range_poly(w0)));
            acc[1] = aext2_add(acc[1], aext2_mul(eqs, range_poly(w1)));
        }
        acc[2] = aext2_add(acc[2], aext2_mul(eqs, range_poly(w2)));
        acc[3] = aext2_add(acc[3], aext2_mul(eqs, range_poly(w3)));

        E2 o0 = o_lo;
        E2 o1 = aext2_add(o0, o_d);
        E2 o2 = aext2_add(o1, o_d);
        acc[4] = aext2_add(acc[4], aext2_mul(o0, w0));
        acc[5] = aext2_add(acc[5], aext2_mul(o1, w1));
        acc[6] = aext2_add(acc[6], aext2_mul(o2, w2));
    }

    // Block reduction, one slot at a time: 4 KB of shared instead of 28.
    __shared__ E2 sdata[BLOCK];
    const E2 tag = tags[e];
    for (int k = 0; k < MSG_SLOTS; k++) {
        sdata[tid] = acc[k];
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = aext2_add(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }
        if (tid == 0) {
            // gamma_e multiplies the norm half only, once per block rather than
            // once per element.
            E2 v = sdata[0];
            if (k < NORM_EVALS) v = aext2_mul(tag, v);
            partial[(e * nchunk + chunk) * MSG_SLOTS + k] = v;
        }
        __syncthreads();
    }
}


/**
 * Round 0 straight off the bit-packed witness.
 *
 * A leaf witness is one bit per coefficient. Expanding it to Ext2 to run the
 * first round costs 16 bytes per bit for a buffer that is discarded after one
 * fold — and round 0 is the largest round, so that buffer sets the peak. These
 * kernels read the bits directly and never materialize the full-size table:
 * only the folded, half-size output exists as Ext2.
 *
 * Values are in {0,1}, so `R` vanishes at k = 0 and k = 1 exactly as the
 * first_round shortcut assumes.
 */
__device__ __forceinline__ E2 bit_at(const uint64_t* __restrict__ bits, uint64_t i) {
    return E2((uint64_t)((bits[i >> 6] >> (i & 63)) & 1ULL), (uint64_t)0);
}

__global__ void link_round0_bits_kernel(
    const uint64_t* __restrict__ bits,     // [n_commit][stride/64]
    const E2* __restrict__ omega,          // [n_commit][stride]
    const E2* __restrict__ eq_suffix,      // [half]
    const E2* __restrict__ tags,
    uint64_t stride,
    uint64_t half,
    E2* __restrict__ partial
) {
    const uint64_t e      = blockIdx.y;
    const uint64_t chunk  = blockIdx.x;
    const uint64_t nchunk = gridDim.x;
    const int tid = threadIdx.x;

    const uint64_t* __restrict__ be = bits + e * (stride >> 6);
    const E2* __restrict__ oe = omega + e * stride;

    E2 acc[MSG_SLOTS];
#pragma unroll
    for (int k = 0; k < MSG_SLOTS; k++) acc[k] = e2_zero();

    for (uint64_t idx = chunk * BLOCK + tid; idx < half; idx += nchunk * BLOCK) {
        E2 w_lo = bit_at(be, idx);
        E2 w_hi = bit_at(be, half + idx);
        if (e2_is_zero(w_lo) && e2_is_zero(w_hi)) continue;

        E2 w_d = aext2_sub(w_hi, w_lo);
        E2 o_lo = oe[idx];
        E2 o_d  = aext2_sub(oe[half + idx], o_lo);
        E2 eqs  = eq_suffix[idx];

        E2 w0 = w_lo;
        E2 w1 = aext2_add(w0, w_d);
        E2 w2 = aext2_add(w1, w_d);
        E2 w3 = aext2_add(w2, w_d);

        // k = 0, 1 are free: the witness is still binary here.
        acc[2] = aext2_add(acc[2], aext2_mul(eqs, range_poly(w2)));
        acc[3] = aext2_add(acc[3], aext2_mul(eqs, range_poly(w3)));

        E2 o0 = o_lo;
        E2 o1 = aext2_add(o0, o_d);
        E2 o2 = aext2_add(o1, o_d);
        acc[4] = aext2_add(acc[4], aext2_mul(o0, w0));
        acc[5] = aext2_add(acc[5], aext2_mul(o1, w1));
        acc[6] = aext2_add(acc[6], aext2_mul(o2, w2));
    }

    __shared__ E2 sdata[BLOCK];
    const E2 tag = tags[e];
    for (int k = 0; k < MSG_SLOTS; k++) {
        sdata[tid] = acc[k];
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = aext2_add(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }
        if (tid == 0) {
            E2 v = sdata[0];
            if (k < NORM_EVALS) v = aext2_mul(tag, v);
            partial[(e * nchunk + chunk) * MSG_SLOTS + k] = v;
        }
        __syncthreads();
    }
}


/** Round 0 off the bit-packed witness, driven by the support list. */
__global__ void link_round0_bits_sparse_kernel(
    const uint64_t* __restrict__ bits,
    const E2* __restrict__ omega,
    const E2* __restrict__ eq_gathered,   // indexed by list position
    const E2* __restrict__ tags,
    const uint32_t* __restrict__ list,
    const uint64_t* __restrict__ list_off,
    const uint64_t* __restrict__ list_len,
    uint64_t stride,
    uint64_t half,
    E2* __restrict__ partial
) {
    const uint64_t e      = blockIdx.y;
    const uint64_t chunk  = blockIdx.x;
    const uint64_t nchunk = gridDim.x;
    const int tid = threadIdx.x;

    const uint64_t* __restrict__ be = bits + e * (stride >> 6);
    const E2* __restrict__ oe = omega + e * stride;
    const uint32_t* __restrict__ le = list + list_off[e];
    const uint64_t n = list_len[e];

    E2 acc[MSG_SLOTS];
#pragma unroll
    for (int k = 0; k < MSG_SLOTS; k++) acc[k] = e2_zero();

    for (uint64_t t = chunk * BLOCK + tid; t < n; t += nchunk * BLOCK) {
        const uint64_t idx = le[t];
        E2 w_lo = bit_at(be, idx);
        E2 w_hi = bit_at(be, half + idx);

        E2 w_d = aext2_sub(w_hi, w_lo);
        E2 o_lo = oe[idx];
        E2 o_d  = aext2_sub(oe[half + idx], o_lo);
        E2 eqs  = eq_gathered[list_off[e] + t];

        E2 w0 = w_lo;
        E2 w1 = aext2_add(w0, w_d);
        E2 w2 = aext2_add(w1, w_d);
        E2 w3 = aext2_add(w2, w_d);

        // k = 0, 1 free: witness still binary.
        acc[2] = aext2_add(acc[2], aext2_mul(eqs, range_poly(w2)));
        acc[3] = aext2_add(acc[3], aext2_mul(eqs, range_poly(w3)));

        E2 o0 = o_lo;
        E2 o1 = aext2_add(o0, o_d);
        E2 o2 = aext2_add(o1, o_d);
        acc[4] = aext2_add(acc[4], aext2_mul(o0, w0));
        acc[5] = aext2_add(acc[5], aext2_mul(o1, w1));
        acc[6] = aext2_add(acc[6], aext2_mul(o2, w2));
    }

    __shared__ E2 sdata[BLOCK];
    const E2 tag = tags[e];
    for (int k = 0; k < MSG_SLOTS; k++) {
        sdata[tid] = acc[k];
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = aext2_add(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }
        if (tid == 0) {
            E2 v = sdata[0];
            if (k < NORM_EVALS) v = aext2_mul(tag, v);
            partial[(e * nchunk + chunk) * MSG_SLOTS + k] = v;
        }
        __syncthreads();
    }
}

/** Round-0 fold: bits in, half-size Ext2 out. Never allocates the full table. */
__global__ void link_fold0_bits_kernel(
    const uint64_t* __restrict__ bits,
    E2* __restrict__ w_out,        // [n_commit][half]
    E2* __restrict__ omega,        // folded in place at [n_commit][stride]
    uint64_t stride,
    uint64_t half,
    uint64_t n_commit,
    uint64_t r_c0,
    uint64_t r_c1
) {
    const E2 r(r_c0, r_c1);
    const uint64_t total = half * n_commit;
    for (uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         t < total; t += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t e = t / half;
        const uint64_t i = t - e * half;
        const uint64_t* __restrict__ be = bits + e * (stride >> 6);

        E2 lo = bit_at(be, i);
        E2 hi = bit_at(be, half + i);
        w_out[e * half + i] = aext2_add(lo, aext2_mul(r, aext2_sub(hi, lo)));

        const uint64_t ob = e * stride + i;
        E2 olo = omega[ob];
        omega[ob] = aext2_add(olo, aext2_mul(r, aext2_sub(omega[ob + half], olo)));
    }
}

/** Sum all per-block partials into the round message. */
__global__ void link_reduce_kernel(
    const E2* __restrict__ partial,
    uint64_t n_blocks,
    E2*      __restrict__ out        // [MSG_SLOTS]
) {
    const int k = blockIdx.x;        // one block per slot
    const int tid = threadIdx.x;
    __shared__ E2 sdata[BLOCK];

    E2 acc = e2_zero();
    for (uint64_t b = tid; b < n_blocks; b += BLOCK) {
        acc = aext2_add(acc, partial[b * MSG_SLOTS + k]);
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = aext2_add(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[k] = sdata[0];
}

/** Fold both tables with the round challenge: t[i] = lo + r*(hi - lo). */
__global__ void link_fold_kernel(
    E2*      __restrict__ w,
    E2*      __restrict__ omega,
    uint64_t stride,
    uint64_t omega_stride,
    uint64_t half,
    uint64_t n_commit,
    uint64_t r_c0,
    uint64_t r_c1
) {
    const uint64_t total = half * n_commit;
    const E2 r(r_c0, r_c1);
    for (uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         t < total; t += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t e = t / half;
        const uint64_t i = t - e * half;
        const uint64_t base = e * stride + i;
        const uint64_t obase = e * omega_stride + i;

        E2 lo = w[base];
        w[base] = aext2_add(lo, aext2_mul(r, aext2_sub(w[base + half], lo)));

        E2 olo = omega[obase];
        omega[obase] = aext2_add(olo, aext2_mul(r, aext2_sub(omega[obase + half], olo)));
    }
}

/**
 * eq(., point) table over 2^len Boolean indices, MOST significant variable
 * first. Built one variable at a time; each pass doubles the live span, so the
 * whole build is O(2^len) work like the host version.
 */
__global__ void link_eq_expand_kernel(
    E2*      __restrict__ table,
    uint64_t span,          // current live length before this pass
    uint64_t r_c0,
    uint64_t r_c1
) {
    const E2 r(r_c0, r_c1);
    const E2 one((uint64_t)1, (uint64_t)0);
    const E2 om = aext2_sub(one, r);
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         i < span; i += (uint64_t)gridDim.x * blockDim.x) {
        E2 v = table[i];
        table[i]        = aext2_mul(v, om);
        table[span + i] = aext2_mul(v, r);
    }
}



/**
 * Expand a bit-packed witness into Ext2 coefficients on device.
 *
 * A leaf witness is one bit per coefficient, so uploading it as Ext2 inflates it
 * 128x — gigabytes of PCIe traffic and host materialization to carry data that
 * fits in tens of megabytes. Upload the bits; expand here.
 */
__global__ void link_expand_bits_kernel(
    const uint64_t* __restrict__ bits,   // [n_words]
    E2* __restrict__ out,                // [n_words * 64]
    uint64_t n_words
) {
    for (uint64_t w = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         w < n_words; w += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t word = bits[w];
        for (int b = 0; b < 64; b++) {
            out[w * 64 + b] = E2((uint64_t)((word >> b) & 1ULL), (uint64_t)0);
        }
    }
}

/**
 * Batched query-weight construction.
 *
 * omega_e(X) = sum_{j: e_j = e} gamma^tag(j) * eq(X, r_j). Because a packed
 * query's point has a Boolean block prefix, eq factorizes across it and the
 * weight is supported on that leaf's sub-cube alone. So each query's eq table is
 * built directly into its slice of omega.
 *
 * Batching is by *expansion level*, not by query: at level k every query whose
 * block is still larger doubles its live span, so the whole construction is
 * `max_block_arity` launches with total work proportional to the packed witness
 * size — rather than one launch chain per query, which at realistic query counts
 * would be six figures of launches.
 *
 *   gridDim = (ceil(span / BLOCK), n_active)
 */
__global__ void link_omega_expand_kernel(
    E2* __restrict__ omega,
    const uint64_t* __restrict__ bases,   // [n_active] element offset per query
    const uint64_t* __restrict__ rs,      // [n_active][2] challenge limbs
    uint64_t span                          // live length before this pass
) {
    const uint64_t q = blockIdx.y;
    const uint64_t base = bases[q];
    const E2 r(rs[2 * q], rs[2 * q + 1]);
    const E2 one((uint64_t)1, (uint64_t)0);
    const E2 om = aext2_sub(one, r);

    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         i < span; i += (uint64_t)gridDim.x * blockDim.x) {
        E2 v = omega[base + i];
        omega[base + i]        = aext2_mul(v, om);
        omega[base + span + i] = aext2_mul(v, r);
    }
}


/**
 * Gather eq(idx, alpha_suffix) at exactly the listed positions.
 *
 * The suffix eq factor is dense by nature, but on the sparse path it is only
 * ever read at list entries. Materializing all 2^(A-r-1) of it costs more than
 * the round message it feeds; evaluating it pointwise costs (A-r) multiplies per
 * listed entry, which summed over rounds is a small fraction of one dense table.
 */
__global__ void link_eq_gather_kernel(
    const uint64_t* __restrict__ alpha,   // [n_vars][2], MSB-first
    uint64_t n_vars,
    const uint32_t* __restrict__ list,
    E2* __restrict__ out,
    uint64_t total
) {
    const E2 one((uint64_t)1, (uint64_t)0);
    for (uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         t < total; t += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t idx = list[t];
        E2 acc = one;
        for (uint64_t k = 0; k < n_vars; k++) {
            const E2 a(alpha[2 * k], alpha[2 * k + 1]);
            const uint64_t bit = (idx >> (n_vars - 1 - k)) & 1ULL;
            acc = aext2_mul(acc, bit ? a : aext2_sub(one, a));
        }
        out[t] = acc;
    }
}

/**
 * Round message over an explicit support list.
 *
 * The dense kernel's zero test retires a warp only when all 32 lanes are zero.
 * One-hot Shout advice puts one nonzero every 2^table_commit_log slots, so that
 * caps the win at warp width no matter how sparse the witness really is. Driving
 * the loop from a compacted list of active pair indices removes the cap: the
 * kernel touches exactly the pairs that can contribute.
 *
 * The lists are position-only and precomputed per round, which is sound because
 * folding never creates a nonzero outside its parents' positions — the support
 * of round r is determined at setup, independent of the challenges.
 *
 *   gridDim  = (chunks, n_commit)
 *   list     = [n_commit] slices, described by list_off / list_len
 */
__global__ void link_round_sparse_kernel(
    const E2* __restrict__ w,
    const E2* __restrict__ omega,
    const E2* __restrict__ eq_suffix,
    const E2* __restrict__ tags,
    const uint32_t* __restrict__ list,
    const uint64_t* __restrict__ list_off,
    const uint64_t* __restrict__ list_len,
    uint64_t stride,
    uint64_t omega_stride,
    uint64_t half,
    int      first_round,
    E2*      __restrict__ partial
) {
    const uint64_t e      = blockIdx.y;
    const uint64_t chunk  = blockIdx.x;
    const uint64_t nchunk = gridDim.x;
    const int tid = threadIdx.x;

    const E2* __restrict__ we = w + e * stride;
    const E2* __restrict__ oe = omega + e * omega_stride;
    const uint32_t* __restrict__ le = list + list_off[e];
    const uint64_t n = list_len[e];

    E2 acc[MSG_SLOTS];
#pragma unroll
    for (int k = 0; k < MSG_SLOTS; k++) acc[k] = e2_zero();

    for (uint64_t t = chunk * BLOCK + tid; t < n; t += nchunk * BLOCK) {
        const uint64_t idx = le[t];
        E2 w_lo = we[idx];
        E2 w_hi = we[half + idx];

        E2 w_d = aext2_sub(w_hi, w_lo);
        E2 o_lo = oe[idx];
        E2 o_d  = aext2_sub(oe[half + idx], o_lo);
        // eq_suffix is gathered, so it indexes by list position not by idx.
        E2 eqs  = eq_suffix[list_off[e] + t];

        E2 w0 = w_lo;
        E2 w1 = aext2_add(w0, w_d);
        E2 w2 = aext2_add(w1, w_d);
        E2 w3 = aext2_add(w2, w_d);

        if (!first_round) {
            acc[0] = aext2_add(acc[0], aext2_mul(eqs, range_poly(w0)));
            acc[1] = aext2_add(acc[1], aext2_mul(eqs, range_poly(w1)));
        }
        acc[2] = aext2_add(acc[2], aext2_mul(eqs, range_poly(w2)));
        acc[3] = aext2_add(acc[3], aext2_mul(eqs, range_poly(w3)));

        E2 o0 = o_lo;
        E2 o1 = aext2_add(o0, o_d);
        E2 o2 = aext2_add(o1, o_d);
        acc[4] = aext2_add(acc[4], aext2_mul(o0, w0));
        acc[5] = aext2_add(acc[5], aext2_mul(o1, w1));
        acc[6] = aext2_add(acc[6], aext2_mul(o2, w2));
    }

    __shared__ E2 sdata[BLOCK];
    const E2 tag = tags[e];
    for (int k = 0; k < MSG_SLOTS; k++) {
        sdata[tid] = acc[k];
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = aext2_add(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }
        if (tid == 0) {
            E2 v = sdata[0];
            if (k < NORM_EVALS) v = aext2_mul(tag, v);
            partial[(e * nchunk + chunk) * MSG_SLOTS + k] = v;
        }
        __syncthreads();
    }
}

/** Fold driven by the same support list: untouched entries are already zero. */
__global__ void link_fold_sparse_kernel(
    E2*      __restrict__ w,
    E2*      __restrict__ omega,
    const uint32_t* __restrict__ list,
    const uint64_t* __restrict__ list_off,
    const uint64_t* __restrict__ list_len,
    uint64_t stride,
    uint64_t half,
    uint64_t n_commit,
    uint64_t r_c0,
    uint64_t r_c1
) {
    const E2 r(r_c0, r_c1);
    const uint64_t e = blockIdx.y;
    if (e >= n_commit) return;
    const uint32_t* __restrict__ le = list + list_off[e];
    const uint64_t n = list_len[e];
    for (uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         t < n; t += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t base = e * stride + le[t];
        E2 lo = w[base];
        w[base] = aext2_add(lo, aext2_mul(r, aext2_sub(w[base + half], lo)));
        E2 olo = omega[base];
        omega[base] = aext2_add(olo, aext2_mul(r, aext2_sub(omega[base + half], olo)));
    }
}


// ============================================================================
// Interleaved layout: query weights evaluated on demand
// ============================================================================
//
// With the block index in the LOW bits, folding the high (leaf) variables never
// mixes two blocks, so every live position belongs to exactly one query and its
// weight is
//
//     omega(idx) = scale[block] * eq(l', point[block][r..a])
//
// where l' is the remaining leaf index and scale carries gamma_j together with
// the eq factors of the already-bound rounds. No folded weight table exists at
// all, which takes the link's live state from 16 bytes per witness bit to 8 and
// is what lets a batch hold enough commitments to matter.

__device__ __forceinline__ E2 eq_pointwise(
    const E2* __restrict__ pt,   // remaining leaf point, MSB-first, m entries
    uint64_t l,                  // remaining leaf index, m bits
    int m
) {
    const E2 one((uint64_t)1, (uint64_t)0);
    E2 acc = one;
    for (int k = 0; k < m; k++) {
        const E2 r = pt[k];
        const uint64_t bit = (l >> (m - 1 - k)) & 1ULL;
        acc = aext2_mul(acc, bit ? r : aext2_sub(one, r));
    }
    return acc;
}

/**
 * Round message for the interleaved layout, driven by a support list, with the
 * witness read from bits (round 0) or from the folded table (later rounds).
 *
 *   pts    [n_commit][blocks][leaf_arity]  leaf points, MSB-first
 *   scale  [n_commit][blocks]              gamma_j * prod of bound eq factors
 */
__global__ void link_round_interleaved_kernel(
    const uint64_t* __restrict__ bits,     // null when reading w
    const E2* __restrict__ w,              // null on round 0
    const E2* __restrict__ pts,
    const E2* __restrict__ scale,
    const E2* __restrict__ eq_gathered,
    const E2* __restrict__ tags,
    const uint32_t* __restrict__ list,
    const uint64_t* __restrict__ list_off,
    const uint64_t* __restrict__ list_len,
    uint64_t w_stride,
    uint64_t bits_stride_words,
    uint64_t half,
    uint32_t block_mask,      // blocks - 1
    int      block_bits,
    int      leaf_arity,
    int      round,           // rounds already bound
    int      first_round,
    E2*      __restrict__ partial
) {
    const uint64_t e      = blockIdx.y;
    const uint64_t chunk  = blockIdx.x;
    const uint64_t nchunk = gridDim.x;
    const int tid = threadIdx.x;
    const int m = leaf_arity - round;          // remaining leaf variables

    const uint32_t* __restrict__ le = list + list_off[e];
    const uint64_t n = list_len[e];
    const int blocks = block_mask + 1;
    const E2* __restrict__ pe = pts + e * (uint64_t)blocks * leaf_arity;
    const E2* __restrict__ se = scale + e * (uint64_t)blocks;

    E2 acc[MSG_SLOTS];
#pragma unroll
    for (int k = 0; k < MSG_SLOTS; k++) acc[k] = e2_zero();

    for (uint64_t t = chunk * BLOCK + tid; t < n; t += nchunk * BLOCK) {
        const uint64_t idx = le[t];
        E2 w_lo, w_hi;
        if (first_round) {
            const uint64_t* __restrict__ be = bits + e * bits_stride_words;
            w_lo = bit_at(be, idx);
            w_hi = bit_at(be, half + idx);
        } else {
            const E2* __restrict__ we = w + e * w_stride;
            w_lo = we[idx];
            w_hi = we[half + idx];
        }

        const uint32_t b = (uint32_t)idx & block_mask;
        const uint64_t l_lo = idx >> block_bits;
        const uint64_t l_hi = (half + idx) >> block_bits;
        const E2* __restrict__ ptb = pe + (uint64_t)b * leaf_arity + round;
        const E2 sc = se[b];

        E2 o_lo = aext2_mul(sc, eq_pointwise(ptb, l_lo, m));
        E2 o_hi = aext2_mul(sc, eq_pointwise(ptb, l_hi, m));

        E2 w_d = aext2_sub(w_hi, w_lo);
        E2 o_d = aext2_sub(o_hi, o_lo);
        E2 eqs = eq_gathered[list_off[e] + t];

        E2 w0 = w_lo;
        E2 w1 = aext2_add(w0, w_d);
        E2 w2 = aext2_add(w1, w_d);
        E2 w3 = aext2_add(w2, w_d);

        if (!first_round) {
            acc[0] = aext2_add(acc[0], aext2_mul(eqs, range_poly(w0)));
            acc[1] = aext2_add(acc[1], aext2_mul(eqs, range_poly(w1)));
        }
        acc[2] = aext2_add(acc[2], aext2_mul(eqs, range_poly(w2)));
        acc[3] = aext2_add(acc[3], aext2_mul(eqs, range_poly(w3)));

        E2 oo1 = aext2_add(o_lo, o_d);
        E2 oo2 = aext2_add(oo1, o_d);
        acc[4] = aext2_add(acc[4], aext2_mul(o_lo, w0));
        acc[5] = aext2_add(acc[5], aext2_mul(oo1, w1));
        acc[6] = aext2_add(acc[6], aext2_mul(oo2, w2));
    }

    __shared__ E2 sdata[BLOCK];
    const E2 tag = tags[e];
    for (int k = 0; k < MSG_SLOTS; k++) {
        sdata[tid] = acc[k];
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = aext2_add(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }
        if (tid == 0) {
            E2 v = sdata[0];
            if (k < NORM_EVALS) v = aext2_mul(tag, v);
            partial[(e * nchunk + chunk) * MSG_SLOTS + k] = v;
        }
        __syncthreads();
    }
}

/** Witness-only fold for the interleaved layout: there is no weight table. */
__global__ void link_fold_w_kernel(
    const uint64_t* __restrict__ bits,   // round 0 source, else null
    const E2* __restrict__ w_in,
    E2* __restrict__ w_out,
    uint64_t in_stride,
    uint64_t bits_stride_words,
    uint64_t out_stride,
    uint64_t half,
    uint64_t n_commit,
    int      first_round,
    uint64_t r_c0,
    uint64_t r_c1
) {
    const E2 r(r_c0, r_c1);
    const uint64_t total = half * n_commit;
    for (uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         t < total; t += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t e = t / half;
        const uint64_t i = t - e * half;
        E2 lo, hi;
        if (first_round) {
            const uint64_t* __restrict__ be = bits + e * bits_stride_words;
            lo = bit_at(be, i);
            hi = bit_at(be, half + i);
        } else {
            const E2* __restrict__ we = w_in + e * in_stride;
            lo = we[i];
            hi = we[half + i];
        }
        w_out[e * out_stride + i] = aext2_add(lo, aext2_mul(r, aext2_sub(hi, lo)));
    }
}

}  // namespace link_sumcheck

#endif  // ALMOST_LINK_SUMCHECK_CUH

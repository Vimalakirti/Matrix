/**
 * Basefold Polynomial Commitment Scheme - CUDA Implementation
 *
 * GPU-accelerated kernels for the Basefold PCS over the Goldilocks field,
 * with extension field (GoldilocksExt2) opening support.
 *
 * Phases 1-7: Device functions and CUDA kernels.
 * See basefold_kernels.cu for host wrappers, orchestration, and tests.
 */

#ifndef BASEFOLD_CUH
#define BASEFOLD_CUH

#include "goldilocks.cuh"
#include "extension.cuh"
#include "poseidon2.cuh"
#include "eq_lagrange.cuh"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// ============================================================================
// Data Structures
// ============================================================================

/**
 * A folding table entry: stores a folding point and precomputed weight.
 * weight = 1 / (x1 - x0) where x0 = point and x1 is the paired point.
 */
struct FoldingEntry {
    GoldilocksField point;   // x0
    GoldilocksField weight;  // 1 / (x1 - x0)
};

// ============================================================================
// Phase 1: Bit-Reversal Permutation
// ============================================================================

/**
 * Reverse the bits of an integer of width log_n.
 */
__device__ __forceinline__
size_t bit_reverse(size_t x, int log_n) {
    size_t result = 0;
    for (int i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/**
 * Bit-reversal permutation kernel for GoldilocksField.
 * Converts between Type1 and Type2 orderings (the transform is its own inverse).
 */
__global__ void bit_reverse_permute_gl_kernel(
    GoldilocksField* __restrict__ data,
    int log_n
) {
    size_t n = 1ULL << log_n;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t rev = bit_reverse(idx, log_n);
    if (idx < rev) {
        GoldilocksField tmp = data[idx];
        data[idx] = data[rev];
        data[rev] = tmp;
    }
}

/**
 * Bit-reversal permutation kernel for GoldilocksExt2.
 */
__global__ void bit_reverse_permute_ext2_kernel(
    GoldilocksExt2* __restrict__ data,
    int log_n
) {
    size_t n = 1ULL << log_n;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t rev = bit_reverse(idx, log_n);
    if (idx < rev) {
        GoldilocksExt2 tmp = data[idx];
        data[idx] = data[rev];
        data[rev] = tmp;
    }
}

// ============================================================================
// Phase 2: Boolean Hypercube Interpolation (Evals -> Coefficients)
// ============================================================================

/**
 * First pass of BHC interpolation:
 *   coeffs[2i]   = evals[2i]
 *   coeffs[2i+1] = evals[2i+1] - evals[2i]
 *
 * Also copies evals into bh_evals_copy for later bit-reversal.
 */
__global__ void bhc_interp_first_pass_kernel(
    const GoldilocksField* __restrict__ evals,
    GoldilocksField* __restrict__ coeffs,
    GoldilocksField* __restrict__ bh_evals_copy,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pair_count = n / 2;
    if (idx >= pair_count) return;

    GoldilocksField a = evals[2 * idx];
    GoldilocksField b = evals[2 * idx + 1];

    coeffs[2 * idx]     = a;
    coeffs[2 * idx + 1] = gl_sub(b, a);

    bh_evals_copy[2 * idx]     = a;
    bh_evals_copy[2 * idx + 1] = b;
}

/**
 * Subsequent layers of BHC interpolation.
 * For level k (k >= 1), chunk_size = 2^(k+1), half_chunk = 2^k.
 * For the second half of each chunk:
 *   coeffs[base + half_chunk + j] -= coeffs[base + j]
 */
__global__ void bhc_interp_layer_kernel(
    GoldilocksField* __restrict__ coeffs,
    size_t half_chunk,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;

    // Determine which chunk and position within second half
    size_t chunk_size = half_chunk * 2;
    size_t chunk_idx = idx / half_chunk;
    size_t pos_in_half = idx % half_chunk;
    size_t base = chunk_idx * chunk_size;

    size_t upper_idx = base + half_chunk + pos_in_half;
    size_t lower_idx = base + pos_in_half;

    coeffs[upper_idx] = gl_sub(coeffs[upper_idx], coeffs[lower_idx]);
}

// ============================================================================
// Phase 3: Foldable Domain Encoding
// ============================================================================

/**
 * Repetition code: each coefficient repeated rate times.
 * output[i] = coeffs[i / rate]
 */
__global__ void repetition_encode_kernel(
    const GoldilocksField* __restrict__ coeffs,
    GoldilocksField* __restrict__ output,
    int rate,
    size_t n_output
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_output) return;

    output[idx] = coeffs[idx / rate];
}

/**
 * One layer of the foldable domain encoding butterfly.
 * For each pair (j, j - half_chunk) within a chunk:
 *   rhs = data[j]  (original value)
 *   lhs = -rhs     (negation for AES-based table)
 *   data[j]          = data[j - half_chunk] + lhs
 *   data[j - half_chunk] = data[j - half_chunk] + rhs
 *
 * Note: This uses the negation-based butterfly (non-binary_rs mode).
 * The table is not used here because the "random folding points" in AES mode
 * use negation (lhs = -rhs).
 */
__global__ void foldable_domain_layer_kernel(
    GoldilocksField* __restrict__ data,
    size_t half_chunk,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t chunk_size = half_chunk * 2;
    size_t num_pairs = n / 2;
    if (idx >= num_pairs) return;

    // Map thread to the correct pair within its chunk
    size_t chunk_idx = idx / half_chunk;
    size_t pos_in_half = idx % half_chunk;
    size_t base = chunk_idx * chunk_size;

    size_t lo_idx = base + pos_in_half;
    size_t hi_idx = lo_idx + half_chunk;

    GoldilocksField lo_val = data[lo_idx];
    GoldilocksField hi_val = data[hi_idx];

    // Butterfly: lo_new = lo + hi, hi_new = lo - hi
    data[lo_idx] = gl_add(lo_val, hi_val);
    data[hi_idx] = gl_sub(lo_val, hi_val);
}

/**
 * One layer of the foldable domain encoding butterfly with table twiddle factors.
 * Used when code_type == "binary_rs" or when actual table values are needed.
 *
 * For each pair:
 *   tw = table[level_offset + pair_index_within_level]
 *   data[hi] = data[lo] + tw * data[hi]   (conceptually)
 *   Actually follows the Rust reference more precisely.
 */
__global__ void foldable_domain_layer_table_kernel(
    GoldilocksField* __restrict__ data,
    const GoldilocksField* __restrict__ table_level,
    size_t half_chunk,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_pairs = n / 2;
    if (idx >= num_pairs) return;

    size_t chunk_size = half_chunk * 2;
    size_t chunk_idx = idx / half_chunk;
    size_t pos_in_half = idx % half_chunk;
    size_t base = chunk_idx * chunk_size;

    size_t lo_idx = base + pos_in_half;
    size_t hi_idx = lo_idx + half_chunk;

    GoldilocksField lo_val = data[lo_idx];
    GoldilocksField hi_val = data[hi_idx];

    // Get twiddle factor for this pair
    GoldilocksField tw = table_level[idx];

    // Butterfly with twiddle
    GoldilocksField tw_hi = gl_mul(tw, hi_val);
    data[lo_idx] = gl_add(lo_val, tw_hi);
    data[hi_idx] = gl_sub(lo_val, tw_hi);
}

/**
 * RS basecode encoding kernel.
 * Each thread evaluates one chunk polynomial at one domain point using Horner's method.
 * Domain point = (eval_idx + 1) as a field element.
 */
__global__ void rs_basecode_encode_kernel(
    const GoldilocksField* __restrict__ coeffs,
    GoldilocksField* __restrict__ output,
    int chunk_size,
    int total_eval_points,
    size_t n_chunks
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_outputs = n_chunks * total_eval_points;
    if (idx >= total_outputs) return;

    size_t chunk_idx = idx / total_eval_points;
    size_t eval_idx = idx % total_eval_points;

    // Domain point: 1-indexed
    GoldilocksField x(eval_idx + 1);

    // Horner's method: evaluate polynomial at x
    const GoldilocksField* chunk_coeffs = coeffs + chunk_idx * chunk_size;
    GoldilocksField result(0);
    for (int i = chunk_size - 1; i >= 0; i--) {
        result = gl_add(gl_mul(result, x), chunk_coeffs[i]);
    }

    output[idx] = result;
}

// ============================================================================
// Phase 5: Sum-Check Kernels (Base Field)
// ============================================================================

/**
 * Single-level BHC interpolation for sum-check (one_level_interp_hc).
 * For each pair [a, b]: write [a, b - a].
 * Operates in-place.
 */
__global__ void sumcheck_interp_kernel(
    GoldilocksField* __restrict__ data,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksField a = data[2 * idx];
    GoldilocksField b = data[2 * idx + 1];

    data[2 * idx + 1] = gl_sub(b, a);
    // data[2*idx] = a (unchanged)
}

/**
 * Sum-check evaluate at challenge and compact (one_level_eval_hc).
 * For each pair [c, d]: output[i] = c + challenge * d.
 * Input has 2*pair_count elements, output has pair_count elements.
 */
__global__ void sumcheck_eval_kernel(
    const GoldilocksField* __restrict__ data,
    GoldilocksField challenge,
    GoldilocksField* __restrict__ output,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksField c = data[2 * idx];
    GoldilocksField d = data[2 * idx + 1];

    output[idx] = gl_add(c, gl_mul(challenge, d));
}

/**
 * Sum-check product kernel (parallel_pi).
 * Computes per-pair contributions to the 3 degree-2 polynomial coefficients,
 * then performs block-level reduction.
 *
 * eq and bh are in interleaved coefficient form:
 *   eq[2i] = eq_even, eq[2i+1] = eq_odd
 *   bh[2i] = bh_even, bh[2i+1] = bh_odd
 *
 * c0 = sum(eq_even * bh_even)
 * c1 = sum(eq_even * bh_odd + eq_odd * bh_even)
 * c2 = sum(eq_odd * bh_odd)
 */
__global__ void sumcheck_product_kernel(
    const GoldilocksField* __restrict__ eq,
    const GoldilocksField* __restrict__ bh,
    GoldilocksField* __restrict__ partial_c0,
    GoldilocksField* __restrict__ partial_c1,
    GoldilocksField* __restrict__ partial_c2,
    size_t pair_count
) {
    __shared__ uint64_t s_c0[BLOCK_SIZE];
    __shared__ uint64_t s_c1[BLOCK_SIZE];
    __shared__ uint64_t s_c2[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    GoldilocksField acc_c0(0), acc_c1(0), acc_c2(0);

    for (size_t i = idx; i < pair_count; i += grid_size) {
        GoldilocksField eq_even = eq[2 * i];
        GoldilocksField eq_odd  = eq[2 * i + 1];
        GoldilocksField bh_even = bh[2 * i];
        GoldilocksField bh_odd  = bh[2 * i + 1];

        acc_c0 = gl_add(acc_c0, gl_mul(eq_even, bh_even));
        acc_c1 = gl_add(acc_c1, gl_add(gl_mul(eq_even, bh_odd), gl_mul(eq_odd, bh_even)));
        acc_c2 = gl_add(acc_c2, gl_mul(eq_odd, bh_odd));
    }

    s_c0[tid] = acc_c0.value;
    s_c1[tid] = acc_c1.value;
    s_c2[tid] = acc_c2.value;
    __syncthreads();

    // Block-level tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_c0[tid] = gl_add(GoldilocksField(s_c0[tid]), GoldilocksField(s_c0[tid + s])).value;
            s_c1[tid] = gl_add(GoldilocksField(s_c1[tid]), GoldilocksField(s_c1[tid + s])).value;
            s_c2[tid] = gl_add(GoldilocksField(s_c2[tid]), GoldilocksField(s_c2[tid + s])).value;
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_c0[blockIdx.x] = GoldilocksField(s_c0[0]);
        partial_c1[blockIdx.x] = GoldilocksField(s_c1[0]);
        partial_c2[blockIdx.x] = GoldilocksField(s_c2[0]);
    }
}

// ============================================================================
// Phase 6: Basefold Codeword Folding (Base Field)
// ============================================================================

/**
 * Basefold one-round folding via Lagrange interpolation with precomputed weights.
 *
 * For each pair index i:
 *   val0 = codeword[2*i], val1 = codeword[2*i + 1]
 *   x0 = table[i].point, w = table[i].weight  (w = 1/(x1 - x0))
 *   output[i] = val0 + (challenge - x0) * (val1 - val0) * w
 */
__global__ void basefold_fold_kernel(
    const GoldilocksField* __restrict__ codeword,
    const FoldingEntry* __restrict__ table,
    GoldilocksField challenge,
    GoldilocksField* __restrict__ output,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksField val0 = codeword[2 * idx];
    GoldilocksField val1 = codeword[2 * idx + 1];
    GoldilocksField x0 = table[idx].point;
    GoldilocksField w  = table[idx].weight;

    // Lagrange interpolation at challenge:
    // result = val0 + (challenge - x0) * (val1 - val0) * w
    GoldilocksField diff = gl_sub(val1, val0);
    GoldilocksField cx = gl_sub(challenge, x0);
    GoldilocksField result = gl_add(val0, gl_mul(gl_mul(cx, diff), w));

    output[idx] = result;
}

// ============================================================================
// Phase 7: Extension Field Kernels
// ============================================================================

// --- 7a: Mixed sum-check product (bh in F_p, eq in F_{p^2}) ---

__global__ void sumcheck_product_mixed_kernel(
    const GoldilocksExt2* __restrict__ eq,
    const GoldilocksField* __restrict__ bh,
    GoldilocksExt2* __restrict__ partial_c0,
    GoldilocksExt2* __restrict__ partial_c1,
    GoldilocksExt2* __restrict__ partial_c2,
    size_t pair_count
) {
    // Shared memory: 2 uint64_t per ext2 element, 3 accumulators per thread
    __shared__ uint64_t s_c0_0[BLOCK_SIZE];
    __shared__ uint64_t s_c0_1[BLOCK_SIZE];
    __shared__ uint64_t s_c1_0[BLOCK_SIZE];
    __shared__ uint64_t s_c1_1[BLOCK_SIZE];
    __shared__ uint64_t s_c2_0[BLOCK_SIZE];
    __shared__ uint64_t s_c2_1[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    GoldilocksExt2 acc_c0, acc_c1, acc_c2;

    for (size_t i = idx; i < pair_count; i += grid_size) {
        GoldilocksExt2 eq_even = eq[2 * i];
        GoldilocksExt2 eq_odd  = eq[2 * i + 1];
        // bh values promoted to ext2 via scalar mul
        GoldilocksField bh_even = bh[2 * i];
        GoldilocksField bh_odd  = bh[2 * i + 1];

        // eq_even * bh_even (scalar mul: F_p * F_{p^2})
        acc_c0 = ext2_add(acc_c0, ext2_scalar_mul(bh_even, eq_even));
        // eq_even * bh_odd + eq_odd * bh_even
        acc_c1 = ext2_add(acc_c1, ext2_add(
            ext2_scalar_mul(bh_odd, eq_even),
            ext2_scalar_mul(bh_even, eq_odd)
        ));
        // eq_odd * bh_odd
        acc_c2 = ext2_add(acc_c2, ext2_scalar_mul(bh_odd, eq_odd));
    }

    s_c0_0[tid] = acc_c0.c[0].value; s_c0_1[tid] = acc_c0.c[1].value;
    s_c1_0[tid] = acc_c1.c[0].value; s_c1_1[tid] = acc_c1.c[1].value;
    s_c2_0[tid] = acc_c2.c[0].value; s_c2_1[tid] = acc_c2.c[1].value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            GoldilocksExt2 a, b;
            a = GoldilocksExt2(GoldilocksField(s_c0_0[tid]), GoldilocksField(s_c0_1[tid]));
            b = GoldilocksExt2(GoldilocksField(s_c0_0[tid + s]), GoldilocksField(s_c0_1[tid + s]));
            a = ext2_add(a, b);
            s_c0_0[tid] = a.c[0].value; s_c0_1[tid] = a.c[1].value;

            a = GoldilocksExt2(GoldilocksField(s_c1_0[tid]), GoldilocksField(s_c1_1[tid]));
            b = GoldilocksExt2(GoldilocksField(s_c1_0[tid + s]), GoldilocksField(s_c1_1[tid + s]));
            a = ext2_add(a, b);
            s_c1_0[tid] = a.c[0].value; s_c1_1[tid] = a.c[1].value;

            a = GoldilocksExt2(GoldilocksField(s_c2_0[tid]), GoldilocksField(s_c2_1[tid]));
            b = GoldilocksExt2(GoldilocksField(s_c2_0[tid + s]), GoldilocksField(s_c2_1[tid + s]));
            a = ext2_add(a, b);
            s_c2_0[tid] = a.c[0].value; s_c2_1[tid] = a.c[1].value;
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_c0[blockIdx.x] = GoldilocksExt2(GoldilocksField(s_c0_0[0]), GoldilocksField(s_c0_1[0]));
        partial_c1[blockIdx.x] = GoldilocksExt2(GoldilocksField(s_c1_0[0]), GoldilocksField(s_c1_1[0]));
        partial_c2[blockIdx.x] = GoldilocksExt2(GoldilocksField(s_c2_0[0]), GoldilocksField(s_c2_1[0]));
    }
}

// --- 7b: Mixed eval (F_p data, F_{p^2} challenge -> F_{p^2} output) ---

__global__ void sumcheck_eval_mixed_kernel(
    const GoldilocksField* __restrict__ data,
    GoldilocksExt2 challenge,
    GoldilocksExt2* __restrict__ output,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksField c = data[2 * idx];
    GoldilocksField d = data[2 * idx + 1];

    // c + challenge * d  (promote c, d to ext2)
    GoldilocksExt2 c_ext = gl_to_ext2(c);
    GoldilocksExt2 d_ext = gl_to_ext2(d);
    output[idx] = ext2_add(c_ext, ext2_mul(challenge, d_ext));
}

// --- 7c: Mixed codeword fold (F_p codeword, F_{p^2} challenge -> F_{p^2}) ---

__global__ void basefold_fold_mixed_kernel(
    const GoldilocksField* __restrict__ codeword,
    const FoldingEntry* __restrict__ table,
    GoldilocksExt2 challenge,
    GoldilocksExt2* __restrict__ output,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksField val0 = codeword[2 * idx];
    GoldilocksField val1 = codeword[2 * idx + 1];
    GoldilocksField x0 = table[idx].point;
    GoldilocksField w  = table[idx].weight;

    // result = val0 + (challenge - x0) * (val1 - val0) * w
    // val0, val1, x0, w are F_p; challenge is F_{p^2}
    GoldilocksField diff = gl_sub(val1, val0);
    GoldilocksField diff_w = gl_mul(diff, w);  // (val1 - val0) * w in F_p
    GoldilocksExt2 cx = ext2_sub(challenge, gl_to_ext2(x0));  // challenge - x0 in F_{p^2}
    GoldilocksExt2 result = ext2_add(
        gl_to_ext2(val0),
        ext2_scalar_mul(diff_w, cx)  // diff_w * cx, scalar_mul: F_p * F_{p^2}
    );

    output[idx] = result;
}

// --- 7d: Ext2 sum-check interp ---

__global__ void sumcheck_interp_ext2_kernel(
    GoldilocksExt2* __restrict__ data,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksExt2 a = data[2 * idx];
    GoldilocksExt2 b = data[2 * idx + 1];

    data[2 * idx + 1] = ext2_sub(b, a);
}

// --- 7e: Ext2 sum-check eval ---

__global__ void sumcheck_eval_ext2_kernel(
    const GoldilocksExt2* __restrict__ data,
    GoldilocksExt2 challenge,
    GoldilocksExt2* __restrict__ output,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksExt2 c = data[2 * idx];
    GoldilocksExt2 d = data[2 * idx + 1];

    output[idx] = ext2_add(c, ext2_mul(challenge, d));
}

// --- 7f: Ext2 sum-check product ---

__global__ void sumcheck_product_ext2_kernel(
    const GoldilocksExt2* __restrict__ eq,
    const GoldilocksExt2* __restrict__ bh,
    GoldilocksExt2* __restrict__ partial_c0,
    GoldilocksExt2* __restrict__ partial_c1,
    GoldilocksExt2* __restrict__ partial_c2,
    size_t pair_count
) {
    __shared__ uint64_t s_c0_0[BLOCK_SIZE];
    __shared__ uint64_t s_c0_1[BLOCK_SIZE];
    __shared__ uint64_t s_c1_0[BLOCK_SIZE];
    __shared__ uint64_t s_c1_1[BLOCK_SIZE];
    __shared__ uint64_t s_c2_0[BLOCK_SIZE];
    __shared__ uint64_t s_c2_1[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    GoldilocksExt2 acc_c0, acc_c1, acc_c2;

    for (size_t i = idx; i < pair_count; i += grid_size) {
        GoldilocksExt2 eq_even = eq[2 * i];
        GoldilocksExt2 eq_odd  = eq[2 * i + 1];
        GoldilocksExt2 bh_even = bh[2 * i];
        GoldilocksExt2 bh_odd  = bh[2 * i + 1];

        acc_c0 = ext2_add(acc_c0, ext2_mul(eq_even, bh_even));
        acc_c1 = ext2_add(acc_c1, ext2_add(ext2_mul(eq_even, bh_odd), ext2_mul(eq_odd, bh_even)));
        acc_c2 = ext2_add(acc_c2, ext2_mul(eq_odd, bh_odd));
    }

    s_c0_0[tid] = acc_c0.c[0].value; s_c0_1[tid] = acc_c0.c[1].value;
    s_c1_0[tid] = acc_c1.c[0].value; s_c1_1[tid] = acc_c1.c[1].value;
    s_c2_0[tid] = acc_c2.c[0].value; s_c2_1[tid] = acc_c2.c[1].value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            GoldilocksExt2 a, b;
            a = GoldilocksExt2(GoldilocksField(s_c0_0[tid]), GoldilocksField(s_c0_1[tid]));
            b = GoldilocksExt2(GoldilocksField(s_c0_0[tid + s]), GoldilocksField(s_c0_1[tid + s]));
            a = ext2_add(a, b);
            s_c0_0[tid] = a.c[0].value; s_c0_1[tid] = a.c[1].value;

            a = GoldilocksExt2(GoldilocksField(s_c1_0[tid]), GoldilocksField(s_c1_1[tid]));
            b = GoldilocksExt2(GoldilocksField(s_c1_0[tid + s]), GoldilocksField(s_c1_1[tid + s]));
            a = ext2_add(a, b);
            s_c1_0[tid] = a.c[0].value; s_c1_1[tid] = a.c[1].value;

            a = GoldilocksExt2(GoldilocksField(s_c2_0[tid]), GoldilocksField(s_c2_1[tid]));
            b = GoldilocksExt2(GoldilocksField(s_c2_0[tid + s]), GoldilocksField(s_c2_1[tid + s]));
            a = ext2_add(a, b);
            s_c2_0[tid] = a.c[0].value; s_c2_1[tid] = a.c[1].value;
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_c0[blockIdx.x] = GoldilocksExt2(GoldilocksField(s_c0_0[0]), GoldilocksField(s_c0_1[0]));
        partial_c1[blockIdx.x] = GoldilocksExt2(GoldilocksField(s_c1_0[0]), GoldilocksField(s_c1_1[0]));
        partial_c2[blockIdx.x] = GoldilocksExt2(GoldilocksField(s_c2_0[0]), GoldilocksField(s_c2_1[0]));
    }
}

// --- 7f2: Fused eval+interp+product for Ext2 sum-check round ---
// Fuses sumcheck_eval_ext2 × 2 + sumcheck_interp_ext2 × 2 + sumcheck_product_ext2
// into a single kernel, reducing global memory traffic by ~65%.
// pair_count = number of product pairs = (input element count) / 4.

__global__ void fused_sumcheck_round_ext2_kernel(
    const GoldilocksExt2* __restrict__ eq_in,   // 4*pair_count Ext2 elements
    const GoldilocksExt2* __restrict__ bh_in,   // 4*pair_count Ext2 elements
    GoldilocksExt2 challenge,
    GoldilocksExt2* __restrict__ eq_out,        // 2*pair_count Ext2 elements (eval'd + interp'd)
    GoldilocksExt2* __restrict__ bh_out,        // 2*pair_count Ext2 elements
    GoldilocksExt2* __restrict__ partial_c0,
    GoldilocksExt2* __restrict__ partial_c1,
    GoldilocksExt2* __restrict__ partial_c2,
    size_t pair_count
) {
    __shared__ uint64_t s_c0_0[BLOCK_SIZE];
    __shared__ uint64_t s_c0_1[BLOCK_SIZE];
    __shared__ uint64_t s_c1_0[BLOCK_SIZE];
    __shared__ uint64_t s_c1_1[BLOCK_SIZE];
    __shared__ uint64_t s_c2_0[BLOCK_SIZE];
    __shared__ uint64_t s_c2_1[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    GoldilocksExt2 acc_c0, acc_c1, acc_c2;

    for (size_t i = idx; i < pair_count; i += grid_size) {
        // --- Eval: fold 4 elements into 2 using challenge ---
        GoldilocksExt2 eq_a = ext2_add(eq_in[4 * i],     ext2_mul(challenge, eq_in[4 * i + 1]));
        GoldilocksExt2 eq_b = ext2_add(eq_in[4 * i + 2], ext2_mul(challenge, eq_in[4 * i + 3]));
        GoldilocksExt2 bh_a = ext2_add(bh_in[4 * i],     ext2_mul(challenge, bh_in[4 * i + 1]));
        GoldilocksExt2 bh_b = ext2_add(bh_in[4 * i + 2], ext2_mul(challenge, bh_in[4 * i + 3]));

        // --- Interp: compute (value, difference) pairs ---
        GoldilocksExt2 eq_even = eq_a;
        GoldilocksExt2 eq_odd  = ext2_sub(eq_b, eq_a);
        GoldilocksExt2 bh_even = bh_a;
        GoldilocksExt2 bh_odd  = ext2_sub(bh_b, bh_a);

        // --- Product: accumulate oracle terms ---
        acc_c0 = ext2_add(acc_c0, ext2_mul(eq_even, bh_even));
        acc_c1 = ext2_add(acc_c1, ext2_add(ext2_mul(eq_even, bh_odd), ext2_mul(eq_odd, bh_even)));
        acc_c2 = ext2_add(acc_c2, ext2_mul(eq_odd, bh_odd));

        // --- Write back eval'd+interp'd data for next round ---
        eq_out[2 * i]     = eq_even;
        eq_out[2 * i + 1] = eq_odd;
        bh_out[2 * i]     = bh_even;
        bh_out[2 * i + 1] = bh_odd;
    }

    // --- Block-level tree reduction (identical to sumcheck_product_ext2_kernel) ---
    s_c0_0[tid] = acc_c0.c[0].value; s_c0_1[tid] = acc_c0.c[1].value;
    s_c1_0[tid] = acc_c1.c[0].value; s_c1_1[tid] = acc_c1.c[1].value;
    s_c2_0[tid] = acc_c2.c[0].value; s_c2_1[tid] = acc_c2.c[1].value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            GoldilocksExt2 a, b;
            a = GoldilocksExt2(GoldilocksField(s_c0_0[tid]), GoldilocksField(s_c0_1[tid]));
            b = GoldilocksExt2(GoldilocksField(s_c0_0[tid + s]), GoldilocksField(s_c0_1[tid + s]));
            a = ext2_add(a, b);
            s_c0_0[tid] = a.c[0].value; s_c0_1[tid] = a.c[1].value;

            a = GoldilocksExt2(GoldilocksField(s_c1_0[tid]), GoldilocksField(s_c1_1[tid]));
            b = GoldilocksExt2(GoldilocksField(s_c1_0[tid + s]), GoldilocksField(s_c1_1[tid + s]));
            a = ext2_add(a, b);
            s_c1_0[tid] = a.c[0].value; s_c1_1[tid] = a.c[1].value;

            a = GoldilocksExt2(GoldilocksField(s_c2_0[tid]), GoldilocksField(s_c2_1[tid]));
            b = GoldilocksExt2(GoldilocksField(s_c2_0[tid + s]), GoldilocksField(s_c2_1[tid + s]));
            a = ext2_add(a, b);
            s_c2_0[tid] = a.c[0].value; s_c2_1[tid] = a.c[1].value;
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_c0[blockIdx.x] = GoldilocksExt2(GoldilocksField(s_c0_0[0]), GoldilocksField(s_c0_1[0]));
        partial_c1[blockIdx.x] = GoldilocksExt2(GoldilocksField(s_c1_0[0]), GoldilocksField(s_c1_1[0]));
        partial_c2[blockIdx.x] = GoldilocksExt2(GoldilocksField(s_c2_0[0]), GoldilocksField(s_c2_1[0]));
    }
}

// --- 7g: Ext2 codeword fold (F_{p^2} codeword, F_p table, F_{p^2} challenge) ---

__global__ void basefold_fold_ext2_kernel(
    const GoldilocksExt2* __restrict__ codeword,
    const FoldingEntry* __restrict__ table,
    GoldilocksExt2 challenge,
    GoldilocksExt2* __restrict__ output,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksExt2 val0 = codeword[2 * idx];
    GoldilocksExt2 val1 = codeword[2 * idx + 1];
    GoldilocksField x0 = table[idx].point;   // F_p
    GoldilocksField w  = table[idx].weight;   // F_p

    // result = val0 + (challenge - x0) * (val1 - val0) * w
    GoldilocksExt2 diff = ext2_sub(val1, val0);
    GoldilocksExt2 diff_w = ext2_scalar_mul(w, diff);  // w * diff
    GoldilocksExt2 cx = ext2_sub(challenge, gl_to_ext2(x0));
    GoldilocksExt2 result = ext2_add(val0, ext2_mul(cx, diff_w));

    output[idx] = result;
}

// ============================================================================
// Phase 10: Query Phase Kernels
// ============================================================================

/**
 * Extract query pairs from a codeword.
 * For each query index q, extract (codeword[2*(q/2)], codeword[2*(q/2)+1]).
 */
__global__ void basefold_extract_queries_kernel(
    const GoldilocksField* __restrict__ codeword,
    const int* __restrict__ query_indices,
    GoldilocksField* __restrict__ output,  // output[2*i], output[2*i+1]
    int num_queries,
    size_t codeword_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;

    int q = query_indices[idx];
    int pair_base = (q / 2) * 2;  // round down to even
    if (pair_base + 1 < codeword_len) {
        output[2 * idx]     = codeword[pair_base];
        output[2 * idx + 1] = codeword[pair_base + 1];
    }
}

/**
 * Extract query pairs from an ext2 codeword.
 */
__global__ void basefold_extract_queries_ext2_kernel(
    const GoldilocksExt2* __restrict__ codeword,
    const int* __restrict__ query_indices,
    GoldilocksExt2* __restrict__ output,
    int num_queries,
    size_t codeword_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;

    int q = query_indices[idx];
    int pair_base = (q / 2) * 2;
    if (pair_base + 1 < codeword_len) {
        output[2 * idx]     = codeword[pair_base];
        output[2 * idx + 1] = codeword[pair_base + 1];
    }
}

/**
 * Merkle path extraction kernel.
 * For each query, extract sibling hashes at each tree level.
 * tree layout: flat array, leaves at offset 0 (num_leaves * chunk_size),
 *   parents at offset num_leaves * chunk_size, etc.
 */
__global__ void merkle_extract_path_kernel(
    const GoldilocksField* __restrict__ tree,
    const int* __restrict__ query_indices,
    GoldilocksField* __restrict__ paths,   // output: paths[query * depth * CHUNK + level * CHUNK + j]
    int num_queries,
    int num_leaves,
    int depth,
    int chunk_size
) {
    int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= num_queries) return;

    int node_idx = query_indices[qidx];
    int layer_start = 0;
    int layer_size = num_leaves;

    for (int level = 0; level < depth; level++) {
        // Sibling index
        int sibling = (node_idx ^ 1);
        // Copy sibling hash
        for (int j = 0; j < chunk_size; j++) {
            paths[qidx * depth * chunk_size + level * chunk_size + j] =
                tree[layer_start * chunk_size + sibling * chunk_size + j];
        }
        node_idx /= 2;
        layer_start += layer_size;
        layer_size /= 2;
    }
}

// ============================================================================
// Phase 11: Verifier Kernels
// ============================================================================

/**
 * Verify interpolation consistency for one query across all rounds.
 * For each query × round:
 *   interpolate2(x0, val0, x1, val1, fold_challenge) should equal next_oracle_val
 *
 * query_vals[query][round] = (val0, val1) pair
 * fold_challenges[round] = challenge used for folding
 * table[round][pair_index] = FoldingEntry
 */
__global__ void basefold_verify_query_kernel(
    const GoldilocksField* __restrict__ query_vals,  // [num_queries * (num_rounds+1) * 2]
    const GoldilocksField* __restrict__ fold_challenges,
    const FoldingEntry* __restrict__ table_flat,
    const int* __restrict__ table_offsets,  // offset into table_flat for each round
    const int* __restrict__ query_indices,
    int num_queries,
    int num_rounds,
    int log_rate,
    int num_vars,
    int* __restrict__ results  // 0 = pass, 1 = fail
) {
    int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= num_queries) return;

    int pass = 0;  // 0 = OK

    int current_index = query_indices[qidx];

    for (int round = 0; round < num_rounds; round++) {
        int pair_idx = current_index / 2;
        GoldilocksField val0 = query_vals[(qidx * (num_rounds + 1) + round) * 2];
        GoldilocksField val1 = query_vals[(qidx * (num_rounds + 1) + round) * 2 + 1];

        FoldingEntry entry = table_flat[table_offsets[round] + pair_idx];
        GoldilocksField challenge = fold_challenges[round];

        // Lagrange interpolation
        GoldilocksField diff = gl_sub(val1, val0);
        GoldilocksField cx = gl_sub(challenge, entry.point);
        GoldilocksField expected = gl_add(val0, gl_mul(gl_mul(cx, diff), entry.weight));

        // Next oracle value
        GoldilocksField next_val;
        if (current_index % 2 == 0) {
            next_val = query_vals[(qidx * (num_rounds + 1) + round + 1) * 2];
        } else {
            next_val = query_vals[(qidx * (num_rounds + 1) + round + 1) * 2 + 1];
        }

        // Check consistency
        GoldilocksField diff_check = gl_sub(expected, next_val);
        if (canonicalize(diff_check.value) != 0) {
            pass = 1;
        }

        current_index = pair_idx;
    }

    results[qidx] = pass;
}

/**
 * Merkle path verification kernel.
 * For each query, recompute hashes up the tree and check against roots.
 */
__global__ void merkle_verify_path_kernel(
    const GoldilocksField* __restrict__ leaf_vals,   // [num_queries * 2] paired values
    const GoldilocksField* __restrict__ paths,       // sibling hashes
    const GoldilocksField* __restrict__ roots,       // expected root per round
    const int* __restrict__ query_indices,
    int num_queries,
    int depth,
    int chunk_size,
    int* __restrict__ results  // 0 = pass, 1 = fail
) {
    int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= num_queries) return;

    // Hash the leaf pair to get initial node hash
    GoldilocksField current_hash[4];
    const GoldilocksField* left_leaf  = leaf_vals + qidx * 2 * chunk_size;
    const GoldilocksField* right_leaf = leaf_vals + qidx * 2 * chunk_size + chunk_size;
    poseidon2_compress_8(left_leaf, right_leaf, current_hash);

    int node_idx = query_indices[qidx] / 2;

    for (int level = 1; level < depth; level++) {
        // Get sibling hash from path
        const GoldilocksField* sibling = paths + qidx * depth * chunk_size + level * chunk_size;

        GoldilocksField new_hash[4];
        if (node_idx % 2 == 0) {
            poseidon2_compress_8(current_hash, sibling, new_hash);
        } else {
            poseidon2_compress_8(sibling, current_hash, new_hash);
        }
        for (int j = 0; j < chunk_size; j++) current_hash[j] = new_hash[j];
        node_idx /= 2;
    }

    // Check against root
    int pass = 0;
    for (int j = 0; j < chunk_size; j++) {
        GoldilocksField diff = gl_sub(current_hash[j], roots[j]);
        if (canonicalize(diff.value) != 0) {
            pass = 1;
        }
    }

    results[qidx] = pass;
}

// ============================================================================
// Ext2 dot product kernel (for mixed inner product: F_p * F_{p^2} -> F_{p^2})
// ============================================================================

__global__ void ext2_dot_product_mixed_kernel(
    const GoldilocksField* __restrict__ a,   // F_p values
    const GoldilocksExt2* __restrict__ b,    // F_{p^2} values
    GoldilocksExt2* __restrict__ output,
    size_t n
) {
    __shared__ uint64_t s0[BLOCK_SIZE];
    __shared__ uint64_t s1[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    GoldilocksExt2 acc;

    for (size_t i = idx; i < n; i += grid_size) {
        acc = ext2_add(acc, ext2_scalar_mul(a[i], b[i]));
    }

    s0[tid] = acc.c[0].value;
    s1[tid] = acc.c[1].value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            uint64_t a0 = s0[tid], a1 = s1[tid];
            uint64_t b0 = s0[tid + s], b1 = s1[tid + s];
            GoldilocksExt2 va; va.c[0] = GoldilocksField(a0); va.c[1] = GoldilocksField(a1);
            GoldilocksExt2 vb; vb.c[0] = GoldilocksField(b0); vb.c[1] = GoldilocksField(b1);
            va = ext2_add(va, vb);
            s0[tid] = va.c[0].value;
            s1[tid] = va.c[1].value;
        }
        __syncthreads();
    }

    if (tid == 0) {
        GoldilocksExt2 res; res.c[0] = GoldilocksField(s0[0]); res.c[1] = GoldilocksField(s1[0]);
        output[blockIdx.x] = res;
    }
}

/**
 * Ext2 sum reduction kernel.
 */
__global__ void ext2_sum_reduce_kernel(
    const GoldilocksExt2* __restrict__ input,
    GoldilocksExt2* __restrict__ output,
    size_t n
) {
    __shared__ uint64_t s0[BLOCK_SIZE];
    __shared__ uint64_t s1[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    GoldilocksExt2 acc;

    for (size_t i = idx; i < n; i += grid_size) {
        acc = ext2_add(acc, input[i]);
    }

    s0[tid] = acc.c[0].value;
    s1[tid] = acc.c[1].value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            uint64_t a0 = s0[tid], a1 = s1[tid];
            uint64_t b0 = s0[tid + s], b1 = s1[tid + s];
            GoldilocksExt2 va; va.c[0] = GoldilocksField(a0); va.c[1] = GoldilocksField(a1);
            GoldilocksExt2 vb; vb.c[0] = GoldilocksField(b0); vb.c[1] = GoldilocksField(b1);
            va = ext2_add(va, vb);
            s0[tid] = va.c[0].value;
            s1[tid] = va.c[1].value;
        }
        __syncthreads();
    }

    if (tid == 0) {
        GoldilocksExt2 res; res.c[0] = GoldilocksField(s0[0]); res.c[1] = GoldilocksField(s1[0]);
        output[blockIdx.x] = res;
    }
}

// ============================================================================
// Interp kernel for mixed F_p (used for bh_evals in first round before ext2)
// ============================================================================

__global__ void sumcheck_interp_mixed_kernel(
    GoldilocksField* __restrict__ data,
    size_t pair_count
) {
    // Same as sumcheck_interp_kernel (interp on F_p data)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksField a = data[2 * idx];
    GoldilocksField b = data[2 * idx + 1];

    data[2 * idx + 1] = gl_sub(b, a);
}

#endif // BASEFOLD_CUH

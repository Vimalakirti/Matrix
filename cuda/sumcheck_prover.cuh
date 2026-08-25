/**
 * GPU Sumcheck Prover Kernels
 *
 * Two kernels for the linear sumcheck protocol:
 * 1. sumcheck_round_message_kernel: computes g(c) = Σ_y Π_i p_i(c,y) for c ∈ {0,...,d}
 * 2. sumcheck_fold_kernel: folds all d polynomials at a challenge point
 *
 * Memory layout: d polynomials packed contiguously with stride = original_size
 *   [poly_0: original_size | poly_1: original_size | ... | poly_{d-1}: original_size]
 * After round m, only first (original_size >> (m+1)) elements per poly are valid.
 */

#ifndef SUMCHECK_PROVER_CUH
#define SUMCHECK_PROVER_CUH

#include "goldilocks.cuh"

#define SUMCHECK_BLOCK_SIZE 256
#define MAX_DEGREE 8

/**
 * Compute round message for the sumcheck protocol.
 *
 * For each evaluation point c ∈ {0, 1, ..., d}, computes:
 *   g(c) = Σ_{y=0}^{half-1} Π_{i=0}^{d-1} p_i(c, y)
 * where p_i(c, y) = p_i[2y] + c * (p_i[2y+1] - p_i[2y])
 *
 * Uses block-level reduction: each block outputs (d+1) partial sums.
 * Host sums across blocks.
 *
 * @param d_polys       Packed polynomials on device (d * original_size elements)
 * @param d_partial     Output: num_blocks * (d+1) partial sums
 * @param d             Number of polynomials
 * @param original_size Stride between polynomials (constant across rounds)
 * @param half          Number of pairs in current round = current_size / 2
 */
__global__ void sumcheck_round_message_kernel(
    const uint64_t* __restrict__ d_polys,
    uint64_t* __restrict__ d_partial,
    int d,
    size_t original_size,
    size_t half
) {
    // Shared memory: (d+1) lanes, one per eval point, each with SUMCHECK_BLOCK_SIZE entries
    __shared__ uint64_t shared[(MAX_DEGREE + 1) * SUMCHECK_BLOCK_SIZE];

    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    int dp1 = d + 1;  // number of evaluation points

    // Initialize accumulators
    GoldilocksField acc[MAX_DEGREE + 1];
    for (int c = 0; c < dp1; c++) {
        acc[c] = GoldilocksField(0);
    }

    // Grid-stride loop over y values
    for (size_t y = idx; y < half; y += grid_size) {
        // Load even/odd values for each polynomial
        GoldilocksField even[MAX_DEGREE];
        GoldilocksField diff[MAX_DEGREE];  // odd - even

        for (int i = 0; i < d; i++) {
            size_t base = i * original_size;
            even[i] = GoldilocksField(d_polys[base + 2 * y]);
            GoldilocksField odd(d_polys[base + 2 * y + 1]);
            diff[i] = gl_sub(odd, even[i]);
        }

        // For each evaluation point c, compute Π_i (even[i] + c * diff[i])
        for (int c = 0; c < dp1; c++) {
            GoldilocksField c_val((uint64_t)c);
            GoldilocksField product(1);

            for (int i = 0; i < d; i++) {
                GoldilocksField val = gl_add(even[i], gl_mul(c_val, diff[i]));
                product = gl_mul(product, val);
            }

            acc[c] = gl_add(acc[c], product);
        }
    }

    // Store accumulators in shared memory
    for (int c = 0; c < dp1; c++) {
        shared[c * SUMCHECK_BLOCK_SIZE + tid] = acc[c].value;
    }
    __syncthreads();

    // Block-level tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < dp1; c++) {
                int offset = c * SUMCHECK_BLOCK_SIZE;
                shared[offset + tid] = gl_add(
                    GoldilocksField(shared[offset + tid]),
                    GoldilocksField(shared[offset + tid + s])
                ).value;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes partial sums for this block
    if (tid == 0) {
        for (int c = 0; c < dp1; c++) {
            d_partial[blockIdx.x * dp1 + c] = shared[c * SUMCHECK_BLOCK_SIZE];
        }
    }
}

/**
 * Fold all d polynomials at a challenge value.
 *
 * Reads from d_input, writes to d_output (must be separate buffers to avoid
 * cross-warp race conditions on write-to-position-y / read-from-position-2y).
 *
 * For each polynomial i and each y in [0, half):
 *   d_output[base + y] = d_input[base + 2y] + challenge * (d_input[base + 2y+1] - d_input[base + 2y])
 *
 * @param d_input       Source packed polynomials (read-only)
 * @param d_output      Destination packed polynomials (write-only)
 * @param challenge     The folding challenge
 * @param d             Number of polynomials
 * @param original_size Stride between polynomials
 * @param half          Number of output elements per poly = current_size / 2
 */
__global__ void sumcheck_fold_kernel(
    const uint64_t* __restrict__ d_input,
    uint64_t* __restrict__ d_output,
    uint64_t challenge,
    int d,
    size_t original_size,
    size_t half
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    GoldilocksField ch(challenge);

    for (size_t y = idx; y < half; y += grid_size) {
        for (int i = 0; i < d; i++) {
            size_t base = i * original_size;
            GoldilocksField a(d_input[base + 2 * y]);
            GoldilocksField b(d_input[base + 2 * y + 1]);
            d_output[base + y] = gl_add(a, gl_mul(ch, gl_sub(b, a))).value;
        }
    }
}

// ============================================================================
// Ext2 Sumcheck Kernels
// ============================================================================

/**
 * Compute round message for the sumcheck protocol over Ext2.
 *
 * Polynomials are stored as interleaved Ext2: [c0, c1, c0, c1, ...]
 * Each polynomial has original_size Ext2 elements = original_size * 2 u64s.
 * Stride between polynomials = original_size * 2 u64s.
 *
 * For each evaluation point c ∈ {0, 1, ..., d}, computes:
 *   g(c) = Σ_{y=0}^{half-1} Π_{i=0}^{d-1} p_i(c, y)
 * where p_i(c, y) = p_i[2y] + c * (p_i[2y+1] - p_i[2y]) in Ext2
 * and c is embedded as Ext2(c, 0).
 *
 * Output: num_blocks * (d+1) * 2 u64s (Ext2 partial sums).
 */
__global__ void sumcheck_round_message_ext2_kernel(
    const uint64_t* __restrict__ d_polys,
    uint64_t* __restrict__ d_partial,
    int d,
    size_t original_size,
    size_t half
) {
    // Shared memory: (d+1) lanes, each with SUMCHECK_BLOCK_SIZE Ext2 entries = 2 u64s each
    __shared__ uint64_t shared[(MAX_DEGREE + 1) * SUMCHECK_BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    int dp1 = d + 1;
    size_t stride = original_size * 2; // u64s per polynomial

    // Initialize Ext2 accumulators
    GoldilocksExt2 acc[MAX_DEGREE + 1];
    for (int c = 0; c < dp1; c++) {
        acc[c] = GoldilocksExt2();
    }

    // Grid-stride loop over y values
    for (size_t y = idx; y < half; y += grid_size) {
        // Load even/odd Ext2 values for each polynomial
        GoldilocksExt2 even[MAX_DEGREE];
        GoldilocksExt2 diff[MAX_DEGREE];

        for (int i = 0; i < d; i++) {
            size_t base = i * stride;
            size_t even_off = base + 4 * y;      // 2 u64s per Ext2 element, pairs of 2
            size_t odd_off  = base + 4 * y + 2;
            even[i] = GoldilocksExt2(d_polys[even_off], d_polys[even_off + 1]);
            GoldilocksExt2 odd(d_polys[odd_off], d_polys[odd_off + 1]);
            diff[i] = ext2_sub(odd, even[i]);
        }

        // For each evaluation point c (base field integer), compute Π_i (even[i] + c * diff[i])
        for (int c = 0; c < dp1; c++) {
            GoldilocksExt2 c_ext(GoldilocksField((uint64_t)c));  // embed c as Ext2
            GoldilocksExt2 product(GoldilocksField(1));           // Ext2 one

            for (int i = 0; i < d; i++) {
                GoldilocksExt2 val = ext2_add(even[i], ext2_mul(c_ext, diff[i]));
                product = ext2_mul(product, val);
            }

            acc[c] = ext2_add(acc[c], product);
        }
    }

    // Store Ext2 accumulators in shared memory
    for (int c = 0; c < dp1; c++) {
        int offset = c * SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
        shared[offset]     = acc[c].c[0].value;
        shared[offset + 1] = acc[c].c[1].value;
    }
    __syncthreads();

    // Block-level tree reduction over Ext2
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < dp1; c++) {
                int offset = c * SUMCHECK_BLOCK_SIZE * 2;
                GoldilocksExt2 a(shared[offset + tid * 2], shared[offset + tid * 2 + 1]);
                GoldilocksExt2 b(shared[offset + (tid + s) * 2], shared[offset + (tid + s) * 2 + 1]);
                GoldilocksExt2 sum = ext2_add(a, b);
                shared[offset + tid * 2]     = sum.c[0].value;
                shared[offset + tid * 2 + 1] = sum.c[1].value;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes Ext2 partial sums for this block
    if (tid == 0) {
        for (int c = 0; c < dp1; c++) {
            int offset = c * SUMCHECK_BLOCK_SIZE * 2;
            d_partial[(blockIdx.x * dp1 + c) * 2]     = shared[offset];
            d_partial[(blockIdx.x * dp1 + c) * 2 + 1] = shared[offset + 1];
        }
    }
}

/**
 * Fold all d polynomials at an Ext2 challenge value.
 *
 * Reads from d_input, writes to d_output (separate buffers).
 * Polynomials are interleaved Ext2: stride = original_size * 2 u64s.
 *
 * For each polynomial i and each y in [0, half):
 *   d_output[i*stride + y*2..y*2+2] = d_input[i*stride + 2y*2..] + challenge * (d_input[i*stride + (2y+1)*2..] - d_input[i*stride + 2y*2..])
 */
__global__ void sumcheck_fold_ext2_kernel(
    const uint64_t* __restrict__ d_input,
    uint64_t* __restrict__ d_output,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    int d,
    size_t original_size,
    size_t half
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    GoldilocksExt2 ch(challenge_c0, challenge_c1);
    size_t stride = original_size * 2; // u64s per polynomial

    for (size_t y = idx; y < half; y += grid_size) {
        for (int i = 0; i < d; i++) {
            size_t base = i * stride;
            size_t even_off = base + 4 * y;
            size_t odd_off  = base + 4 * y + 2;
            GoldilocksExt2 a(d_input[even_off], d_input[even_off + 1]);
            GoldilocksExt2 b(d_input[odd_off], d_input[odd_off + 1]);
            GoldilocksExt2 result = ext2_add(a, ext2_mul(ch, ext2_sub(b, a)));
            d_output[base + y * 2]     = result.c[0].value;
            d_output[base + y * 2 + 1] = result.c[1].value;
        }
    }
}

#endif // SUMCHECK_PROVER_CUH

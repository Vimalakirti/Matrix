// fused_permute_peval.cuh
// ============================================================
// Fused GPU permute + partial evaluation kernel.
//
// Combines the bit-permutation (variable reordering) and partial
// evaluation into a single GPU pass, eliminating the need for an
// intermediate permuted array.
//
// For each output position j (0..2^{n-m}):
//   output[j] = sum_{b=0}^{2^m-1} evals[perm(b + j*2^m)] * eq(r, b)
//
// where perm() uses split-LUTs (lo_lut + hi_lut) loaded into shared
// memory for O(1) per-element permutation lookup, and eq(r, b) is the
// precomputed eq polynomial at the challenge points r.
// ============================================================

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "goldilocks.cuh"
#include "extension.cuh"

#ifndef FUSED_BLOCK_SIZE
#define FUSED_BLOCK_SIZE 256
#endif

// ============================================================
// Main fused kernel: one block per output position
// ============================================================

__global__ void fused_permute_partial_eval_kernel(
    const uint64_t* __restrict__ d_evals,     // base field, 2^n elements
    uint64_t* __restrict__ d_output,           // Ext2 output, output_size * 2 u64
    const uint64_t* __restrict__ d_eq_table,   // Ext2, 2^m elements (2 u64 each)
    const uint32_t* __restrict__ d_lo_lut,     // 2^half entries
    const uint32_t* __restrict__ d_hi_lut,     // 2^(n-half) entries
    int n, int m, int half, int output_size
) {
    // Dynamic shared memory layout:
    //   [lo_lut: lo_size uint32] [hi_lut: hi_size uint32] [warp_results: num_warps*2 uint64]
    extern __shared__ char s_bytes[];

    int lo_size = 1 << half;
    int hi_size = 1 << (n - half);

    uint32_t* s_lo = (uint32_t*)s_bytes;
    uint32_t* s_hi = s_lo + lo_size;

    // Warp result storage (aligned to 8 bytes) after LUTs
    size_t lut_bytes = (size_t)(lo_size + hi_size) * sizeof(uint32_t);
    size_t aligned_lut = (lut_bytes + 7) & ~(size_t)7;
    uint64_t* s_warp = (uint64_t*)(s_bytes + aligned_lut);

    int tid = threadIdx.x;

    // Cooperatively load LUTs into shared memory
    for (int i = tid; i < lo_size; i += blockDim.x)
        s_lo[i] = d_lo_lut[i];
    for (int i = tid; i < hi_size; i += blockDim.x)
        s_hi[i] = d_hi_lut[i];
    __syncthreads();

    int inner_size = 1 << m;
    uint32_t lo_mask = (uint32_t)(lo_size - 1);
    int num_warps = blockDim.x / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;

    // Grid-stride loop over output positions
    for (int j = blockIdx.x; j < output_size; j += gridDim.x) {
        // Each thread accumulates its partial dot product
        GoldilocksExt2 acc;  // default constructor: (0, 0)
        uint32_t base_idx = (uint32_t)j << m;

        for (int b = tid; b < inner_size; b += blockDim.x) {
            uint32_t idx_new = base_idx + (uint32_t)b;
            uint32_t idx_old = s_lo[idx_new & lo_mask] | s_hi[idx_new >> half];

            GoldilocksField val(d_evals[idx_old]);
            GoldilocksExt2 eq(d_eq_table[2*b], d_eq_table[2*b+1]);

            // acc += val * eq (base-field scalar * Ext2)
            acc = ext2_add(acc, ext2_scalar_mul(val, eq));
        }

        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            uint64_t c0_other = __shfl_down_sync(0xffffffff, acc.c[0].value, offset);
            uint64_t c1_other = __shfl_down_sync(0xffffffff, acc.c[1].value, offset);
            acc.c[0] = gl_add(acc.c[0], GoldilocksField(c0_other));
            acc.c[1] = gl_add(acc.c[1], GoldilocksField(c1_other));
        }

        // Cross-warp reduction: lane 0 of each warp writes to shared memory
        if (lane == 0) {
            s_warp[warp_id * 2]     = acc.c[0].value;
            s_warp[warp_id * 2 + 1] = acc.c[1].value;
        }
        __syncthreads();

        // First warp reduces across all warps
        if (warp_id == 0) {
            GoldilocksField c0(lane < num_warps ? s_warp[lane * 2]     : 0);
            GoldilocksField c1(lane < num_warps ? s_warp[lane * 2 + 1] : 0);

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                c0 = gl_add(c0, GoldilocksField(__shfl_down_sync(0xffffffff, c0.value, offset)));
                c1 = gl_add(c1, GoldilocksField(__shfl_down_sync(0xffffffff, c1.value, offset)));
            }

            if (lane == 0) {
                d_output[2 * j]     = c0.value;
                d_output[2 * j + 1] = c1.value;
            }
        }
        __syncthreads();  // Ensure reduction complete before next iteration
    }
}

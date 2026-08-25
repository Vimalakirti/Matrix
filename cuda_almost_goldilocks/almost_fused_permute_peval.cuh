// almost_fused_permute_peval.cuh
// ============================================================
// Fused GPU permute + partial evaluation kernel for the
// almost-Goldilocks field. Mirrors cuda/fused_permute_peval.cuh.
// ============================================================

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "almost_goldilocks.cuh"
#include "almost_extension.cuh"

#ifndef AGL_FUSED_BLOCK_SIZE
#define AGL_FUSED_BLOCK_SIZE 256
#endif

__global__ void agl_fused_permute_partial_eval_kernel(
    const uint64_t* __restrict__ d_evals,
    uint64_t* __restrict__ d_output,
    const uint64_t* __restrict__ d_eq_table,
    const uint32_t* __restrict__ d_lo_lut,
    const uint32_t* __restrict__ d_hi_lut,
    int n, int m, int half, int output_size
) {
    extern __shared__ char s_bytes[];

    int lo_size = 1 << half;
    int hi_size = 1 << (n - half);

    uint32_t* s_lo = (uint32_t*)s_bytes;
    uint32_t* s_hi = s_lo + lo_size;

    size_t lut_bytes = (size_t)(lo_size + hi_size) * sizeof(uint32_t);
    size_t aligned_lut = (lut_bytes + 7) & ~(size_t)7;
    uint64_t* s_warp = (uint64_t*)(s_bytes + aligned_lut);

    int tid = threadIdx.x;

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

    for (int j = blockIdx.x; j < output_size; j += gridDim.x) {
        AlmostGoldilocksExt2 acc;
        uint32_t base_idx = (uint32_t)j << m;

        for (int b = tid; b < inner_size; b += blockDim.x) {
            uint32_t idx_new = base_idx + (uint32_t)b;
            uint32_t idx_old = s_lo[idx_new & lo_mask] | s_hi[idx_new >> half];

            AlmostGoldilocksField val(d_evals[idx_old]);
            AlmostGoldilocksExt2 eq(d_eq_table[2*b], d_eq_table[2*b+1]);

            acc = aext2_add(acc, aext2_scalar_mul(val, eq));
        }

        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            uint64_t c0_other = __shfl_down_sync(0xffffffff, acc.c[0].value, offset);
            uint64_t c1_other = __shfl_down_sync(0xffffffff, acc.c[1].value, offset);
            acc.c[0] = agl_add(acc.c[0], AlmostGoldilocksField(c0_other));
            acc.c[1] = agl_add(acc.c[1], AlmostGoldilocksField(c1_other));
        }

        if (lane == 0) {
            s_warp[warp_id * 2]     = acc.c[0].value;
            s_warp[warp_id * 2 + 1] = acc.c[1].value;
        }
        __syncthreads();

        if (warp_id == 0) {
            AlmostGoldilocksField c0(lane < num_warps ? s_warp[lane * 2]     : 0);
            AlmostGoldilocksField c1(lane < num_warps ? s_warp[lane * 2 + 1] : 0);

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                c0 = agl_add(c0, AlmostGoldilocksField(__shfl_down_sync(0xffffffff, c0.value, offset)));
                c1 = agl_add(c1, AlmostGoldilocksField(__shfl_down_sync(0xffffffff, c1.value, offset)));
            }

            if (lane == 0) {
                d_output[2 * j]     = c0.value;
                d_output[2 * j + 1] = c1.value;
            }
        }
        __syncthreads();
    }
}

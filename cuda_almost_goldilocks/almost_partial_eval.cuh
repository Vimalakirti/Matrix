// almost_partial_eval.cuh
// ============================================================
// GPU partial evaluation of multilinear polynomials over the
// almost-Goldilocks field. Mirrors cuda/partial_eval.cuh; only
// types/prefixes differ.
// ============================================================

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "almost_goldilocks.cuh"
#include "almost_extension.cuh"

#ifndef AGL_PEVAL_BLOCK_SIZE
#define AGL_PEVAL_BLOCK_SIZE 256
#endif

// ============================================================
// Kernel 1: Base field folding layer (AGL -> AGL)
// ============================================================

__global__ void agl_partial_eval_layer_kernel(
    const AlmostGoldilocksField* __restrict__ input,
    AlmostGoldilocksField* __restrict__ output,
    const AlmostGoldilocksField* __restrict__ d_r,
    int layer_idx,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    AlmostGoldilocksField r_i = d_r[layer_idx];
    AlmostGoldilocksField a = input[2 * idx];
    AlmostGoldilocksField b = input[2 * idx + 1];
    AlmostGoldilocksField diff = agl_sub(b, a);
    output[idx] = agl_add(a, agl_mul(r_i, diff));
}

// ============================================================
// Kernel 2: Mixed first round (AGL input, ext2 r -> ext2 output)
// ============================================================

__global__ void agl_partial_eval_mixed_first_kernel(
    const AlmostGoldilocksField* __restrict__ input,
    AlmostGoldilocksExt2* __restrict__ output,
    const AlmostGoldilocksExt2* __restrict__ d_r,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    AlmostGoldilocksExt2 r_0 = d_r[0];
    AlmostGoldilocksField a = input[2 * idx];
    AlmostGoldilocksField b = input[2 * idx + 1];
    AlmostGoldilocksField diff = agl_sub(b, a);

    output[idx] = AlmostGoldilocksExt2(
        agl_add(a, agl_mul(r_0.c[0], diff)),
        agl_mul(r_0.c[1], diff)
    );
}

// ============================================================
// Kernel 3: Ext2 folding layer (ext2 -> ext2)
// ============================================================

__global__ void aext2_partial_eval_layer_kernel(
    const AlmostGoldilocksExt2* __restrict__ input,
    AlmostGoldilocksExt2* __restrict__ output,
    const AlmostGoldilocksExt2* __restrict__ d_r,
    int layer_idx,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    AlmostGoldilocksExt2 r_i = d_r[layer_idx];
    AlmostGoldilocksExt2 a = input[2 * idx];
    AlmostGoldilocksExt2 b = input[2 * idx + 1];
    AlmostGoldilocksExt2 diff = aext2_sub(b, a);
    output[idx] = aext2_add(a, aext2_mul(r_i, diff));
}

// ============================================================
// Host: base field partial eval with ping-pong
// ============================================================

inline cudaError_t agl_partial_eval(
    AlmostGoldilocksField* d_data,
    AlmostGoldilocksField* d_scratch,
    const AlmostGoldilocksField* d_r,
    int log_n,
    int m,
    cudaStream_t stream = 0
) {
    AlmostGoldilocksField* src = d_data;
    AlmostGoldilocksField* dst = d_scratch;
    size_t current_size = 1ULL << log_n;

    for (int i = 0; i < m; i++) {
        size_t pair_count = current_size / 2;
        int grid = (int)((pair_count + AGL_PEVAL_BLOCK_SIZE - 1) / AGL_PEVAL_BLOCK_SIZE);
        agl_partial_eval_layer_kernel<<<grid, AGL_PEVAL_BLOCK_SIZE, 0, stream>>>(
            src, dst, d_r, i, pair_count
        );
        AlmostGoldilocksField* tmp = src;
        src = dst;
        dst = tmp;
        current_size = pair_count;
    }

    if (m > 0 && src != d_data) {
        cudaError_t err = cudaMemcpyAsync(
            d_data, src,
            current_size * sizeof(AlmostGoldilocksField),
            cudaMemcpyDeviceToDevice, stream
        );
        if (err != cudaSuccess) return err;
    }
    return cudaGetLastError();
}

// ============================================================
// Host: AGL -> Ext2 partial eval with ping-pong
// ============================================================

inline cudaError_t agl_partial_eval_ext2_from_base(
    const AlmostGoldilocksField* d_input,
    AlmostGoldilocksExt2* d_output,
    AlmostGoldilocksExt2* d_scratch,
    const AlmostGoldilocksExt2* d_r,
    int log_n,
    int m,
    cudaStream_t stream = 0
) {
    size_t current_size = 1ULL << log_n;

    {
        size_t pair_count = current_size / 2;
        int grid = (int)((pair_count + AGL_PEVAL_BLOCK_SIZE - 1) / AGL_PEVAL_BLOCK_SIZE);
        agl_partial_eval_mixed_first_kernel<<<grid, AGL_PEVAL_BLOCK_SIZE, 0, stream>>>(
            d_input, d_output, d_r, pair_count
        );
        current_size = pair_count;
    }

    AlmostGoldilocksExt2* src = d_output;
    AlmostGoldilocksExt2* dst = d_scratch;

    for (int i = 1; i < m; i++) {
        size_t pair_count = current_size / 2;
        int grid = (int)((pair_count + AGL_PEVAL_BLOCK_SIZE - 1) / AGL_PEVAL_BLOCK_SIZE);
        aext2_partial_eval_layer_kernel<<<grid, AGL_PEVAL_BLOCK_SIZE, 0, stream>>>(
            src, dst, d_r, i, pair_count
        );
        AlmostGoldilocksExt2* tmp = src;
        src = dst;
        dst = tmp;
        current_size = pair_count;
    }

    if (m > 1 && src != d_output) {
        cudaError_t err = cudaMemcpyAsync(
            d_output, src,
            current_size * sizeof(AlmostGoldilocksExt2),
            cudaMemcpyDeviceToDevice, stream
        );
        if (err != cudaSuccess) return err;
    }
    return cudaGetLastError();
}

// partial_eval.cuh
// ============================================================
// GPU partial evaluation of multilinear polynomials.
//
// Given f(x_1,...,x_N) with 2^N evaluations and a point r=(r_1,...,r_m),
// compute g(x_{m+1},...,x_N) = f(r, x_{m+1},...,x_N) yielding
// 2^{N-m} evaluations via layer-by-layer folding.
//
// Each round halves the data:
//   out[j] = in[2j] + r_i * (in[2j+1] - in[2j])
//
// Note: in-place folding has a WAR race across warps (thread j writes
// position j, while thread j/2 reads from it), so we use a ping-pong
// approach with a scratch buffer.
// ============================================================

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "goldilocks.cuh"
#include "extension.cuh"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// ============================================================
// Kernel 1: Base field folding layer (GL -> GL)
// ============================================================

__global__ void partial_eval_gl_layer_kernel(
    const GoldilocksField* __restrict__ input,
    GoldilocksField* __restrict__ output,
    const GoldilocksField* __restrict__ d_r,
    int layer_idx,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksField r_i = d_r[layer_idx];
    GoldilocksField a = input[2 * idx];
    GoldilocksField b = input[2 * idx + 1];
    GoldilocksField diff = gl_sub(b, a);

    output[idx] = gl_add(a, gl_mul(r_i, diff));
}

// ============================================================
// Kernel 2: Mixed first round (GL input, ext2 r -> ext2 output)
// ============================================================

__global__ void partial_eval_mixed_first_kernel(
    const GoldilocksField* __restrict__ input,
    GoldilocksExt2* __restrict__ output,
    const GoldilocksExt2* __restrict__ d_r,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksExt2 r_0 = d_r[0];
    GoldilocksField a = input[2 * idx];
    GoldilocksField b = input[2 * idx + 1];
    GoldilocksField diff = gl_sub(b, a);

    output[idx] = GoldilocksExt2(
        gl_add(a, gl_mul(r_0.c[0], diff)),
        gl_mul(r_0.c[1], diff)
    );
}

// ============================================================
// Kernel 3: Ext2 folding layer (ext2 -> ext2)
// ============================================================

__global__ void partial_eval_ext2_layer_kernel(
    const GoldilocksExt2* __restrict__ input,
    GoldilocksExt2* __restrict__ output,
    const GoldilocksExt2* __restrict__ d_r,
    int layer_idx,
    size_t pair_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    GoldilocksExt2 r_i = d_r[layer_idx];
    GoldilocksExt2 a = input[2 * idx];
    GoldilocksExt2 b = input[2 * idx + 1];
    GoldilocksExt2 diff = ext2_sub(b, a);

    output[idx] = ext2_add(a, ext2_mul(r_i, diff));
}

// ============================================================
// Host function: GL partial eval with ping-pong
// ============================================================

/**
 * Partial evaluate a multilinear polynomial at r (base field).
 *
 * Uses d_data and d_scratch as ping-pong buffers.  The result ends
 * up in d_data (first 2^{log_n - m} positions).
 *
 * @param d_data    Device buffer with 2^log_n GL elements.
 * @param d_scratch Device scratch buffer (>= 2^{log_n - 1} elements).
 * @param d_r       Device pointer to m GL elements.
 * @param log_n     Log of the input size.
 * @param m         Number of variables to evaluate (m <= log_n).
 * @param stream    CUDA stream.
 * @return cudaError_t
 */
inline cudaError_t partial_eval_gl(
    GoldilocksField* d_data,
    GoldilocksField* d_scratch,
    const GoldilocksField* d_r,
    int log_n,
    int m,
    cudaStream_t stream = 0
) {
    GoldilocksField* src = d_data;
    GoldilocksField* dst = d_scratch;
    size_t current_size = 1ULL << log_n;

    for (int i = 0; i < m; i++) {
        size_t pair_count = current_size / 2;
        int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        partial_eval_gl_layer_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, d_r, i, pair_count
        );
        // Swap buffers
        GoldilocksField* tmp = src;
        src = dst;
        dst = tmp;
        current_size = pair_count;
    }

    // If result ended up in d_scratch (m is odd), copy to d_data
    if (m > 0 && src != d_data) {
        cudaError_t err = cudaMemcpyAsync(
            d_data, src,
            current_size * sizeof(GoldilocksField),
            cudaMemcpyDeviceToDevice, stream
        );
        if (err != cudaSuccess) return err;
    }

    return cudaGetLastError();
}

// ============================================================
// Host function: GL -> ext2 partial eval with ping-pong
// ============================================================

/**
 * Partial evaluate a GL multilinear polynomial at ext2 r.
 *
 * Round 0: mixed kernel (GL input -> ext2 d_output).
 * Rounds 1+: ext2 folding with ping-pong between d_output and d_scratch.
 * Result guaranteed in d_output.
 *
 * @param d_input    Device pointer to 2^log_n GL elements (read-only).
 * @param d_output   Device buffer for ext2 output; needs 2^{log_n-1} ext2 elems.
 * @param d_scratch  Device scratch buffer; needs 2^{log_n-2} ext2 elems (or NULL if m <= 1).
 * @param d_r        Device pointer to m ext2 elements.
 * @param log_n      Log of the input size.
 * @param m          Number of variables to evaluate (1 <= m <= log_n).
 * @param stream     CUDA stream.
 * @return cudaError_t
 */
inline cudaError_t partial_eval_ext2_from_gl(
    const GoldilocksField* d_input,
    GoldilocksExt2* d_output,
    GoldilocksExt2* d_scratch,
    const GoldilocksExt2* d_r,
    int log_n,
    int m,
    cudaStream_t stream = 0
) {
    size_t current_size = 1ULL << log_n;

    // Round 0: mixed (GL -> ext2), always writes to d_output
    {
        size_t pair_count = current_size / 2;
        int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        partial_eval_mixed_first_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_input, d_output, d_r, pair_count
        );
        current_size = pair_count;
    }

    // Rounds 1+: ext2 ping-pong
    GoldilocksExt2* src = d_output;
    GoldilocksExt2* dst = d_scratch;

    for (int i = 1; i < m; i++) {
        size_t pair_count = current_size / 2;
        int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        partial_eval_ext2_layer_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, d_r, i, pair_count
        );
        GoldilocksExt2* tmp = src;
        src = dst;
        dst = tmp;
        current_size = pair_count;
    }

    // If result ended up in d_scratch (m-1 is odd => m is even), copy to d_output
    if (m > 1 && src != d_output) {
        cudaError_t err = cudaMemcpyAsync(
            d_output, src,
            current_size * sizeof(GoldilocksExt2),
            cudaMemcpyDeviceToDevice, stream
        );
        if (err != cudaSuccess) return err;
    }

    return cudaGetLastError();
}

// almost_eq_lagrange.cuh
// ============================================================
// GPU implementations of eq(r, x) over the Boolean hypercube,
// for the almost-Goldilocks base field and its Ext2.
//
// Mirrors cuda/eq_lagrange.cuh. Only field-specific constants and
// type names differ; the algorithms (DP and Walsh–Hadamard Transform)
// are identical.
// ============================================================

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "almost_goldilocks.cuh"
#include "almost_extension.cuh"

#ifndef AGL_EQ_BLOCK_SIZE
#define AGL_EQ_BLOCK_SIZE 256
#endif

// ============================================================
// 1) Θ(N) DP implementation (base field)
// ============================================================

__global__ void agl_eq_dp_layer_kernel(
    const AlmostGoldilocksField* __restrict__ in,
    AlmostGoldilocksField* __restrict__ out,
    const AlmostGoldilocksField* __restrict__ d_r,
    int layer_idx,
    size_t half_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    AlmostGoldilocksField r_i = d_r[layer_idx];
    AlmostGoldilocksField a   = in[idx];
    AlmostGoldilocksField ar  = agl_mul(a, r_i);

    out[idx]          = agl_sub(a, ar);  // a * (1 - r_i)
    out[idx + half_n] = ar;
}

__global__ void agl_eq_init_one_kernel(AlmostGoldilocksField* __restrict__ buf, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    buf[idx] = (idx == 0) ? AlmostGoldilocksField(1) : AlmostGoldilocksField(0);
}

inline cudaError_t agl_eq_dp_all(
    const AlmostGoldilocksField* d_r,
    AlmostGoldilocksField* d_buf_a,
    AlmostGoldilocksField* d_buf_b,
    int log_n,
    AlmostGoldilocksField** d_result,
    cudaStream_t stream = 0
) {
    size_t n = 1ULL << log_n;

    int init_grid = (int)((n + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE);
    agl_eq_init_one_kernel<<<init_grid, AGL_EQ_BLOCK_SIZE, 0, stream>>>(d_buf_a, n);

    size_t half_n = 1;
    for (int i = 0; i < log_n; i++) {
        int grid = (int)((half_n + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE);
        agl_eq_dp_layer_kernel<<<grid, AGL_EQ_BLOCK_SIZE, 0, stream>>>(
            d_buf_a, d_buf_b, d_r, i, half_n
        );
        AlmostGoldilocksField* tmp = d_buf_a;
        d_buf_a = d_buf_b;
        d_buf_b = tmp;
        half_n <<= 1;
    }

    if (d_result) *d_result = d_buf_a;
    return cudaGetLastError();
}

// ============================================================
// 1b) Θ(N) DP implementation (Ext2)
// ============================================================

__global__ void aext2_eq_dp_layer_kernel(
    const AlmostGoldilocksExt2* __restrict__ in,
    AlmostGoldilocksExt2* __restrict__ out,
    const AlmostGoldilocksExt2* __restrict__ d_r,
    int layer_idx,
    size_t half_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    AlmostGoldilocksExt2 r_i = d_r[layer_idx];
    AlmostGoldilocksExt2 a   = in[idx];
    AlmostGoldilocksExt2 ar  = aext2_mul(a, r_i);

    out[idx]          = aext2_sub(a, ar);
    out[idx + half_n] = ar;
}

__global__ void aext2_eq_init_one_kernel(AlmostGoldilocksExt2* __restrict__ buf, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    buf[idx] = (idx == 0)
        ? AlmostGoldilocksExt2(AlmostGoldilocksField(1), AlmostGoldilocksField(0))
        : AlmostGoldilocksExt2(AlmostGoldilocksField(0), AlmostGoldilocksField(0));
}

inline cudaError_t aext2_eq_dp_all(
    const AlmostGoldilocksExt2* d_r,
    AlmostGoldilocksExt2* d_buf_a,
    AlmostGoldilocksExt2* d_buf_b,
    int log_n,
    AlmostGoldilocksExt2** d_result,
    cudaStream_t stream = 0
) {
    size_t n = 1ULL << log_n;

    int init_grid = (int)((n + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE);
    aext2_eq_init_one_kernel<<<init_grid, AGL_EQ_BLOCK_SIZE, 0, stream>>>(d_buf_a, n);

    size_t half_n = 1;
    for (int i = 0; i < log_n; i++) {
        int grid = (int)((half_n + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE);
        aext2_eq_dp_layer_kernel<<<grid, AGL_EQ_BLOCK_SIZE, 0, stream>>>(
            d_buf_a, d_buf_b, d_r, i, half_n
        );
        AlmostGoldilocksExt2* tmp = d_buf_a;
        d_buf_a = d_buf_b;
        d_buf_b = tmp;
        half_n <<= 1;
    }

    if (d_result) *d_result = d_buf_a;
    return cudaGetLastError();
}

// ============================================================
// 1b) BATCHED eq DP — N independent eq tables in one launch per stage.
//
// Layout:
//   d_r_all       : [leaf_0_r (log_n Ext2) | leaf_1_r | …]
//   d_buf_a_all   : [leaf_0_buf (n Ext2)   | leaf_1_buf | …]   (in/out)
//   d_buf_b_all   : same shape (scratch)
//
// Per stage, grid Y = num_leaves; each block processes its leaf's
// slice. After the loop, the result is in `d_buf_a_all` if log_n is
// even, else in `d_buf_b_all` (caller checks via the returned pointer).
// ============================================================

__global__ void aext2_eq_init_one_batched_kernel(
    AlmostGoldilocksExt2* __restrict__ buf_all,
    size_t n,
    size_t leaf_stride
) {
    int leaf = blockIdx.y;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    AlmostGoldilocksExt2* buf = buf_all + (size_t)leaf * leaf_stride;
    buf[idx] = (idx == 0)
        ? AlmostGoldilocksExt2(AlmostGoldilocksField(1), AlmostGoldilocksField(0))
        : AlmostGoldilocksExt2(AlmostGoldilocksField(0), AlmostGoldilocksField(0));
}

__global__ void aext2_eq_dp_layer_batched_kernel(
    const AlmostGoldilocksExt2* __restrict__ in_all,
    AlmostGoldilocksExt2* __restrict__ out_all,
    const AlmostGoldilocksExt2* __restrict__ d_r_all,
    int layer_idx,
    int log_n,
    size_t half_n,
    size_t leaf_stride
) {
    int leaf = blockIdx.y;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    const AlmostGoldilocksExt2* in  = in_all  + (size_t)leaf * leaf_stride;
    AlmostGoldilocksExt2*       out = out_all + (size_t)leaf * leaf_stride;
    const AlmostGoldilocksExt2* d_r = d_r_all + (size_t)leaf * log_n;

    AlmostGoldilocksExt2 r_i = d_r[layer_idx];
    AlmostGoldilocksExt2 a   = in[idx];
    AlmostGoldilocksExt2 ar  = aext2_mul(a, r_i);

    out[idx]          = aext2_sub(a, ar);
    out[idx + half_n] = ar;
}

inline cudaError_t aext2_eq_dp_all_batched(
    const AlmostGoldilocksExt2* d_r_all,
    AlmostGoldilocksExt2* d_buf_a_all,
    AlmostGoldilocksExt2* d_buf_b_all,
    int log_n,
    int num_leaves,
    size_t leaf_stride,
    AlmostGoldilocksExt2** d_result,
    cudaStream_t stream = 0
) {
    if (num_leaves <= 0) return cudaSuccess;
    size_t n = 1ULL << log_n;

    int init_grid_x = (int)((n + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE);
    dim3 init_grid(init_grid_x, num_leaves);
    aext2_eq_init_one_batched_kernel<<<init_grid, AGL_EQ_BLOCK_SIZE, 0, stream>>>(
        d_buf_a_all, n, leaf_stride
    );

    size_t half_n = 1;
    for (int i = 0; i < log_n; i++) {
        int grid_x = (int)((half_n + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE);
        dim3 grid(grid_x, num_leaves);
        aext2_eq_dp_layer_batched_kernel<<<grid, AGL_EQ_BLOCK_SIZE, 0, stream>>>(
            d_buf_a_all, d_buf_b_all, d_r_all, i, log_n, half_n, leaf_stride
        );
        AlmostGoldilocksExt2* tmp = d_buf_a_all;
        d_buf_a_all = d_buf_b_all;
        d_buf_b_all = tmp;
        half_n <<= 1;
    }
    if (d_result) *d_result = d_buf_a_all;
    return cudaGetLastError();
}

// ============================================================
// 2) Θ(N log N) Walsh–Hadamard Transform (base field)
// ============================================================

__global__ void agl_eq_compute_c_kernel(
    const AlmostGoldilocksField* __restrict__ r,
    AlmostGoldilocksField* __restrict__ c,
    int log_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)log_n) return;

    AlmostGoldilocksField r_i = r[idx];
    // c_i = 1 - 2*r_i (WHT sign convention)
    c[idx] = agl_sub(AlmostGoldilocksField(1), agl_add(r_i, r_i));
}

__global__ void agl_eq_init_fourier_kernel(
    const AlmostGoldilocksField* __restrict__ c,
    AlmostGoldilocksField* __restrict__ f_hat,
    int log_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n   = 1ULL << log_n;
    if (idx >= n) return;

    AlmostGoldilocksField acc(1);
    size_t mask = idx;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (i >= log_n) break;
        if (mask & 1) acc = agl_mul(acc, c[i]);
        mask >>= 1;
    }
    f_hat[idx] = acc;
}

__global__ void agl_wht_stage_kernel(
    AlmostGoldilocksField* __restrict__ data,
    size_t n,
    size_t stride
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pairs = n / 2;
    if (idx >= pairs) return;

    size_t base = (idx / stride) * stride * 2 + (idx % stride);
    if (base + stride >= n) return;

    AlmostGoldilocksField a = data[base];
    AlmostGoldilocksField b = data[base + stride];
    data[base]          = agl_add(a, b);
    data[base + stride] = agl_sub(a, b);
}

/**
 * Compute inverse normalization for WHT: scale = (2^log_n)^(-1).
 * Uses the almost-Goldilocks inverse-of-2 constant.
 */
__global__ void agl_wht_compute_scale_kernel(
    AlmostGoldilocksField* __restrict__ scale_out,
    int log_n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // inv2 for almost-Goldilocks: (P+1)/2 = HALF_P_PLUS_ONE
    AlmostGoldilocksField inv2(ALMOST_HALF_P_PLUS_ONE);
    AlmostGoldilocksField scale(1);
    for (int i = 0; i < log_n; i++) {
        scale = agl_mul(scale, inv2);
    }
    *scale_out = scale;
}

__global__ void agl_wht_scale_kernel(
    AlmostGoldilocksField* __restrict__ data,
    const AlmostGoldilocksField* __restrict__ scale_ptr,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = agl_mul(data[idx], *scale_ptr);
}

inline cudaError_t agl_eq_wht_all(
    const AlmostGoldilocksField* d_r,
    AlmostGoldilocksField* d_data,
    int log_n,
    cudaStream_t stream = 0
) {
    size_t n = 1ULL << log_n;

    AlmostGoldilocksField* d_c = nullptr;
    cudaMalloc(&d_c, sizeof(AlmostGoldilocksField) * log_n);

    int c_grid = (log_n + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE;
    agl_eq_compute_c_kernel<<<c_grid, AGL_EQ_BLOCK_SIZE, 0, stream>>>(d_r, d_c, log_n);

    int grid = (int)((n + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE);
    agl_eq_init_fourier_kernel<<<grid, AGL_EQ_BLOCK_SIZE, 0, stream>>>(d_c, d_data, log_n);

    for (int k = 0; k < log_n; k++) {
        size_t stride = 1ULL << k;
        size_t pairs  = n / 2;
        int grid2 = (int)((pairs + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE);
        agl_wht_stage_kernel<<<grid2, AGL_EQ_BLOCK_SIZE, 0, stream>>>(d_data, n, stride);
    }

    AlmostGoldilocksField* d_scale = nullptr;
    cudaMalloc(&d_scale, sizeof(AlmostGoldilocksField));
    agl_wht_compute_scale_kernel<<<1, 1, 0, stream>>>(d_scale, log_n);

    int grid3 = (int)((n + AGL_EQ_BLOCK_SIZE - 1) / AGL_EQ_BLOCK_SIZE);
    agl_wht_scale_kernel<<<grid3, AGL_EQ_BLOCK_SIZE, 0, stream>>>(d_data, d_scale, n);

    cudaFree(d_scale);
    cudaFree(d_c);
    return cudaGetLastError();
}

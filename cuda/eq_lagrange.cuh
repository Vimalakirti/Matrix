// eq_lagrange_gpu.cuh
// ============================================================
// GPU implementations of eq(r, x) over the Boolean hypercube
//   1) Θ(N) DP (layer-by-layer expansion)
//   2) Θ(N log N) Walsh–Hadamard Transform (WHT)
// ============================================================

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "goldilocks.cuh"   // must define GoldilocksField + gl_add/gl_sub/gl_mul
#include "extension.cuh"    // GoldilocksExt2 + ext2_add/ext2_sub/ext2_mul

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// ============================================================
// 1) Θ(N) DP implementation
// ============================================================

__global__ void eq_dp_layer_kernel(
    const GoldilocksField* __restrict__ in,
    GoldilocksField* __restrict__ out,
    const GoldilocksField* __restrict__ d_r,
    int layer_idx,
    size_t half_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    GoldilocksField r_i = d_r[layer_idx];
    GoldilocksField a  = in[idx];
    GoldilocksField ar = gl_mul(a, r_i);

    out[idx]          = gl_sub(a, ar);
    out[idx + half_n] = ar;
}

__global__ void eq_init_one_kernel(GoldilocksField* __restrict__ buf, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    buf[idx] = (idx == 0) ? GoldilocksField(1) : GoldilocksField(0);
}

/**
 * Compute eq(r, x) for all x in {0,1}^log_n using DP.
 *
 * @param d_r        Device pointer to r values (length log_n)
 * @param d_buf_a    Device buffer of size 2^log_n (used as workspace)
 * @param d_buf_b    Device buffer of size 2^log_n (used as workspace)
 * @param log_n      Log of the hypercube dimension
 * @param d_result   Output: pointer to the buffer containing the result
 * @param stream     CUDA stream
 * @return cudaError_t
 */
inline cudaError_t eq_dp_all(
    const GoldilocksField* d_r,
    GoldilocksField* d_buf_a,
    GoldilocksField* d_buf_b,
    int log_n,
    GoldilocksField** d_result,
    cudaStream_t stream = 0
) {
    size_t n = 1ULL << log_n;

    int init_grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    eq_init_one_kernel<<<init_grid, BLOCK_SIZE, 0, stream>>>(d_buf_a, n);

    size_t half_n = 1;
    for (int i = 0; i < log_n; i++) {
        int grid = (half_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        eq_dp_layer_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_buf_a, d_buf_b, d_r, i, half_n
        );
        GoldilocksField* tmp = d_buf_a;
        d_buf_a = d_buf_b;
        d_buf_b = tmp;
        half_n <<= 1;
    }

    // Return pointer to buffer containing result
    if (d_result) *d_result = d_buf_a;

    return cudaGetLastError();
}

// ============================================================
// 1b) Θ(N) DP implementation for GoldilocksExt2
// ============================================================

__global__ void ext2_eq_dp_layer_kernel(
    const GoldilocksExt2* __restrict__ in,
    GoldilocksExt2* __restrict__ out,
    const GoldilocksExt2* __restrict__ d_r,
    int layer_idx,
    size_t half_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    GoldilocksExt2 r_i = d_r[layer_idx];
    GoldilocksExt2 a   = in[idx];
    GoldilocksExt2 ar  = ext2_mul(a, r_i);

    out[idx]          = ext2_sub(a, ar);   // a * (1 - r_i)
    out[idx + half_n] = ar;                // a * r_i
}

__global__ void ext2_eq_init_one_kernel(GoldilocksExt2* __restrict__ buf, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    buf[idx] = (idx == 0)
        ? GoldilocksExt2(GoldilocksField(1), GoldilocksField(0))
        : GoldilocksExt2(GoldilocksField(0), GoldilocksField(0));
}

/**
 * Compute eq(r, x) for all x in {0,1}^log_n using DP (Ext2 version).
 *
 * @param d_r        Device pointer to r values (length log_n)
 * @param d_buf_a    Device buffer of size 2^log_n (used as workspace)
 * @param d_buf_b    Device buffer of size 2^log_n (used as workspace)
 * @param log_n      Log of the hypercube dimension
 * @param d_result   Output: pointer to the buffer containing the result
 * @param stream     CUDA stream
 * @return cudaError_t
 */
inline cudaError_t ext2_eq_dp_all(
    const GoldilocksExt2* d_r,
    GoldilocksExt2* d_buf_a,
    GoldilocksExt2* d_buf_b,
    int log_n,
    GoldilocksExt2** d_result,
    cudaStream_t stream = 0
) {
    size_t n = 1ULL << log_n;

    int init_grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_eq_init_one_kernel<<<init_grid, BLOCK_SIZE, 0, stream>>>(d_buf_a, n);

    size_t half_n = 1;
    for (int i = 0; i < log_n; i++) {
        int grid = (half_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ext2_eq_dp_layer_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_buf_a, d_buf_b, d_r, i, half_n
        );
        GoldilocksExt2* tmp = d_buf_a;
        d_buf_a = d_buf_b;
        d_buf_b = tmp;
        half_n <<= 1;
    }

    // Return pointer to buffer containing result
    if (d_result) *d_result = d_buf_a;

    return cudaGetLastError();
}

// ============================================================
// 2) Θ(N log N) Walsh–Hadamard Transform implementation
// ============================================================

__global__ void eq_compute_c_kernel(
    const GoldilocksField* __restrict__ r,
    GoldilocksField* __restrict__ c,
    int log_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= log_n) return;

    GoldilocksField r_i = r[idx];
    // Use 1 - 2*r_i (negated) to account for WHT sign convention
    // This ensures WHT output matches eq(r, x) directly
    c[idx] = gl_sub(GoldilocksField(1), gl_add(r_i, r_i)); // 1 - 2*r_i
}

__global__ void eq_init_fourier_kernel(
    const GoldilocksField* __restrict__ c,
    GoldilocksField* __restrict__ f_hat,
    int log_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n   = 1ULL << log_n;
    if (idx >= n) return;

    GoldilocksField acc(1);
    size_t mask = idx;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (i >= log_n) break;
        if (mask & 1) acc = gl_mul(acc, c[i]);
        mask >>= 1;
    }

    f_hat[idx] = acc;
}

__global__ void wht_stage_kernel(
    GoldilocksField* __restrict__ data,
    size_t n,
    size_t stride
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pairs = n / 2;
    if (idx >= pairs) return;

    // Each thread processes one butterfly pair
    // pair index idx maps to elements (base, base + stride)
    size_t base = (idx / stride) * stride * 2 + (idx % stride);
    if (base + stride >= n) return;

    GoldilocksField a = data[base];
    GoldilocksField b = data[base + stride];

    data[base]          = gl_add(a, b);
    data[base + stride] = gl_sub(a, b);
}

/**
 * Compute scale = inv2^log_n on GPU (single thread)
 */
__global__ void wht_compute_scale_kernel(
    GoldilocksField* __restrict__ scale_out,
    int log_n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    GoldilocksField inv2(0x7FFFFFFF80000001ULL);  // 2^(-1) in Goldilocks
    GoldilocksField scale(1);

    for (int i = 0; i < log_n; i++) {
        scale = gl_mul(scale, inv2);
    }

    *scale_out = scale;
}

/**
 * Scale by (2^log_n)^(-1) — REQUIRED normalization for inverse WHT
 * Reads the scale factor from device memory
 */
__global__ void wht_scale_kernel(
    GoldilocksField* __restrict__ data,
    const GoldilocksField* __restrict__ scale_ptr,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = gl_mul(data[idx], *scale_ptr);
}

inline cudaError_t eq_wht_all(
    const GoldilocksField* d_r,
    GoldilocksField* d_data,
    int log_n,
    cudaStream_t stream = 0
) {
    size_t n = 1ULL << log_n;

    GoldilocksField* d_c = nullptr;
    cudaMalloc(&d_c, sizeof(GoldilocksField) * log_n);

    int c_grid = (log_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    eq_compute_c_kernel<<<c_grid, BLOCK_SIZE, 0, stream>>>(d_r, d_c, log_n);

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    eq_init_fourier_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_c, d_data, log_n
    );

    for (int k = 0; k < log_n; k++) {
        size_t stride = 1ULL << k;
        size_t pairs  = n / 2;
        int grid2 = (pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

        wht_stage_kernel<<<grid2, BLOCK_SIZE, 0, stream>>>(
            d_data, n, stride
        );
    }

    // Compute scale = inv2^log_n on GPU
    GoldilocksField* d_scale = nullptr;
    cudaMalloc(&d_scale, sizeof(GoldilocksField));

    wht_compute_scale_kernel<<<1, 1, 0, stream>>>(d_scale, log_n);

    int grid3 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    wht_scale_kernel<<<grid3, BLOCK_SIZE, 0, stream>>>(
        d_data, d_scale, n
    );

    cudaFree(d_scale);
    cudaFree(d_c);
    return cudaGetLastError();
}

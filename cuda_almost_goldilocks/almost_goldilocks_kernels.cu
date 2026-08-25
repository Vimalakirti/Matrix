/**
 * Almost-Goldilocks Field CUDA Kernels
 *
 * Batch kernels for the almost-Goldilocks field P = 2^64 - 2^32 - 31.
 * Mirrors cuda/goldilocks_kernels.cu (kernel-side, no test main here —
 * see almost_field_test.cu for the test executable).
 */

#include "almost_goldilocks.cuh"
#include <stdio.h>

// ============================================================================
// Configuration
// ============================================================================

#ifndef AGL_BLOCK_SIZE
#define AGL_BLOCK_SIZE 256
#endif
#define AGL_WARP_SIZE 32

// ============================================================================
// Batch Arithmetic Kernels
// ============================================================================

__global__ void agl_batch_add_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    const AlmostGoldilocksField* __restrict__ b,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = agl_add(a[idx], b[idx]);
}

__global__ void agl_batch_sub_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    const AlmostGoldilocksField* __restrict__ b,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = agl_sub(a[idx], b[idx]);
}

__global__ void agl_batch_mul_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    const AlmostGoldilocksField* __restrict__ b,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = agl_mul(a[idx], b[idx]);
}

__global__ void agl_batch_square_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = agl_square(a[idx]);
}

__global__ void agl_batch_neg_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = agl_neg(a[idx]);
}

__global__ void agl_batch_inverse_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = agl_inverse(a[idx]);
}

__global__ void agl_scalar_mul_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    AlmostGoldilocksField scalar,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = agl_mul(scalar, a[idx]);
}

__global__ void agl_batch_exp_kernel(
    const AlmostGoldilocksField* __restrict__ base,
    uint64_t exp,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = agl_exp(base[idx], exp);
}

__global__ void agl_batch_fma_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    const AlmostGoldilocksField* __restrict__ b,
    const AlmostGoldilocksField* __restrict__ c,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = agl_add(agl_mul(a[idx], b[idx]), c[idx]);
}

__global__ void agl_batch_canonicalize_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    AlmostGoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = AlmostGoldilocksField(agl_canonicalize(a[idx].value));
}

// ============================================================================
// Reductions
// ============================================================================

__global__ void agl_sum_reduce_kernel(
    const AlmostGoldilocksField* __restrict__ input,
    AlmostGoldilocksField* __restrict__ output,
    size_t n
) {
    __shared__ uint64_t shared[AGL_BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    AlmostGoldilocksField sum(0);
    for (size_t i = idx; i < n; i += grid_size) {
        sum = agl_add(sum, input[i]);
    }
    shared[tid] = sum.value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = agl_add(
                AlmostGoldilocksField(shared[tid]),
                AlmostGoldilocksField(shared[tid + s])
            ).value;
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = AlmostGoldilocksField(shared[0]);
}

__global__ void agl_dot_product_kernel(
    const AlmostGoldilocksField* __restrict__ a,
    const AlmostGoldilocksField* __restrict__ b,
    AlmostGoldilocksField* __restrict__ output,
    size_t n
) {
    __shared__ uint64_t shared[AGL_BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    AlmostGoldilocksField sum(0);
    for (size_t i = idx; i < n; i += grid_size) {
        sum = agl_add(sum, agl_mul(a[i], b[i]));
    }
    shared[tid] = sum.value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = agl_add(
                AlmostGoldilocksField(shared[tid]),
                AlmostGoldilocksField(shared[tid + s])
            ).value;
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = AlmostGoldilocksField(shared[0]);
}

// ============================================================================
// Polynomial / Matrix
// ============================================================================

__global__ void agl_poly_eval_kernel(
    const AlmostGoldilocksField* __restrict__ coeffs,
    const AlmostGoldilocksField* __restrict__ points,
    AlmostGoldilocksField* __restrict__ results,
    int n_coeffs,
    int n_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    AlmostGoldilocksField x = points[idx];
    AlmostGoldilocksField result(0);
    for (int i = n_coeffs - 1; i >= 0; i--) {
        result = agl_add(agl_mul(result, x), coeffs[i]);
    }
    results[idx] = result;
}

__global__ void agl_matrix_vec_mul_kernel(
    const AlmostGoldilocksField* __restrict__ A,
    const AlmostGoldilocksField* __restrict__ v,
    AlmostGoldilocksField* __restrict__ result,
    int m,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    AlmostGoldilocksField sum(0);
    for (int col = 0; col < n; col++) {
        sum = agl_add(sum, agl_mul(A[row * n + col], v[col]));
    }
    result[row] = sum;
}

__global__ void agl_matrix_mul_kernel(
    const AlmostGoldilocksField* __restrict__ A,
    const AlmostGoldilocksField* __restrict__ B,
    AlmostGoldilocksField* __restrict__ C,
    int m,
    int k,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    AlmostGoldilocksField sum(0);
    for (int i = 0; i < k; i++) {
        sum = agl_add(sum, agl_mul(A[row * k + i], B[i * n + col]));
    }
    C[row * n + col] = sum;
}

// ============================================================================
// Host Wrappers
// ============================================================================

inline void agl_get_grid_dims(size_t n, int& grid_size, int& block_size) {
    block_size = AGL_BLOCK_SIZE;
    grid_size = (int)((n + block_size - 1) / block_size);
}

inline cudaError_t agl_batch_add(
    const AlmostGoldilocksField* d_a,
    const AlmostGoldilocksField* d_b,
    AlmostGoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid, block;
    agl_get_grid_dims(n, grid, block);
    agl_batch_add_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t agl_batch_sub(
    const AlmostGoldilocksField* d_a,
    const AlmostGoldilocksField* d_b,
    AlmostGoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid, block;
    agl_get_grid_dims(n, grid, block);
    agl_batch_sub_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t agl_batch_mul(
    const AlmostGoldilocksField* d_a,
    const AlmostGoldilocksField* d_b,
    AlmostGoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid, block;
    agl_get_grid_dims(n, grid, block);
    agl_batch_mul_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t agl_batch_square(
    const AlmostGoldilocksField* d_a,
    AlmostGoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid, block;
    agl_get_grid_dims(n, grid, block);
    agl_batch_square_kernel<<<grid, block, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t agl_batch_inverse(
    const AlmostGoldilocksField* d_a,
    AlmostGoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid, block;
    agl_get_grid_dims(n, grid, block);
    agl_batch_inverse_kernel<<<grid, block, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t agl_batch_neg(
    const AlmostGoldilocksField* d_a,
    AlmostGoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid, block;
    agl_get_grid_dims(n, grid, block);
    agl_batch_neg_kernel<<<grid, block, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

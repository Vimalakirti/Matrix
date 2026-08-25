/**
 * Almost-Goldilocks Quadratic Extension CUDA Kernels
 *
 * Ext2-only batch kernels and host wrappers. Quintic extension is omitted.
 */

#include "almost_extension.cuh"

#ifndef AGL_EXT2_BLOCK_SIZE
#define AGL_EXT2_BLOCK_SIZE 256
#endif

// ============================================================================
// Batch kernels
// ============================================================================

__global__ void aext2_batch_add_kernel(
    const AlmostGoldilocksExt2* __restrict__ a,
    const AlmostGoldilocksExt2* __restrict__ b,
    AlmostGoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = aext2_add(a[idx], b[idx]);
}

__global__ void aext2_batch_sub_kernel(
    const AlmostGoldilocksExt2* __restrict__ a,
    const AlmostGoldilocksExt2* __restrict__ b,
    AlmostGoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = aext2_sub(a[idx], b[idx]);
}

__global__ void aext2_batch_mul_kernel(
    const AlmostGoldilocksExt2* __restrict__ a,
    const AlmostGoldilocksExt2* __restrict__ b,
    AlmostGoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = aext2_mul(a[idx], b[idx]);
}

__global__ void aext2_batch_square_kernel(
    const AlmostGoldilocksExt2* __restrict__ a,
    AlmostGoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = aext2_square(a[idx]);
}

__global__ void aext2_batch_inverse_kernel(
    const AlmostGoldilocksExt2* __restrict__ a,
    AlmostGoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = aext2_inverse(a[idx]);
}

__global__ void aext2_batch_frobenius_kernel(
    const AlmostGoldilocksExt2* __restrict__ a,
    AlmostGoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = aext2_frobenius(a[idx]);
}

__global__ void agl_to_aext2_batch_kernel(
    const AlmostGoldilocksField* __restrict__ input,
    AlmostGoldilocksExt2* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = agl_to_ext2(input[idx]);
}

__global__ void aext2_to_agl_batch_kernel(
    const AlmostGoldilocksExt2* __restrict__ input,
    AlmostGoldilocksField* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = aext2_to_agl(input[idx]);
}

__global__ void aext2_batch_scalar_mul_kernel(
    AlmostGoldilocksField scalar,
    const AlmostGoldilocksExt2* __restrict__ a,
    AlmostGoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = aext2_scalar_mul(scalar, a[idx]);
}

__global__ void aext2_batch_exp_kernel(
    const AlmostGoldilocksExt2* __restrict__ base,
    uint64_t exp,
    AlmostGoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = aext2_exp(base[idx], exp);
}

// ============================================================================
// Host wrappers
// ============================================================================

inline cudaError_t aext2_batch_add(
    const AlmostGoldilocksExt2* d_a,
    const AlmostGoldilocksExt2* d_b,
    AlmostGoldilocksExt2* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid = (n + AGL_EXT2_BLOCK_SIZE - 1) / AGL_EXT2_BLOCK_SIZE;
    aext2_batch_add_kernel<<<grid, AGL_EXT2_BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t aext2_batch_sub(
    const AlmostGoldilocksExt2* d_a,
    const AlmostGoldilocksExt2* d_b,
    AlmostGoldilocksExt2* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid = (n + AGL_EXT2_BLOCK_SIZE - 1) / AGL_EXT2_BLOCK_SIZE;
    aext2_batch_sub_kernel<<<grid, AGL_EXT2_BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t aext2_batch_mul(
    const AlmostGoldilocksExt2* d_a,
    const AlmostGoldilocksExt2* d_b,
    AlmostGoldilocksExt2* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid = (n + AGL_EXT2_BLOCK_SIZE - 1) / AGL_EXT2_BLOCK_SIZE;
    aext2_batch_mul_kernel<<<grid, AGL_EXT2_BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t aext2_batch_square(
    const AlmostGoldilocksExt2* d_a,
    AlmostGoldilocksExt2* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid = (n + AGL_EXT2_BLOCK_SIZE - 1) / AGL_EXT2_BLOCK_SIZE;
    aext2_batch_square_kernel<<<grid, AGL_EXT2_BLOCK_SIZE, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t aext2_batch_inverse(
    const AlmostGoldilocksExt2* d_a,
    AlmostGoldilocksExt2* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid = (n + AGL_EXT2_BLOCK_SIZE - 1) / AGL_EXT2_BLOCK_SIZE;
    aext2_batch_inverse_kernel<<<grid, AGL_EXT2_BLOCK_SIZE, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t agl_to_aext2_batch(
    const AlmostGoldilocksField* d_input,
    AlmostGoldilocksExt2* d_output,
    int n,
    cudaStream_t stream = 0
) {
    int grid = (n + AGL_EXT2_BLOCK_SIZE - 1) / AGL_EXT2_BLOCK_SIZE;
    agl_to_aext2_batch_kernel<<<grid, AGL_EXT2_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    return cudaGetLastError();
}

inline cudaError_t aext2_to_agl_batch(
    const AlmostGoldilocksExt2* d_input,
    AlmostGoldilocksField* d_output,
    int n,
    cudaStream_t stream = 0
) {
    int grid = (n + AGL_EXT2_BLOCK_SIZE - 1) / AGL_EXT2_BLOCK_SIZE;
    aext2_to_agl_batch_kernel<<<grid, AGL_EXT2_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    return cudaGetLastError();
}

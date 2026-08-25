/**
 * C FFI Wrapper for Goldilocks CUDA Kernels
 *
 * This file provides C-compatible functions that can be called from Rust via FFI.
 */

#include "goldilocks.cuh"
#include "extension.cuh"
#include "poseidon2.cuh"
#include "challenger.cuh"
#include "sumcheck_prover.cuh"
#include "fused_permute_peval.cuh"
#include "monolith.cuh"
#include "monolith_kernels.cu"
#include <cstdio>

// ============================================================================
// C-compatible type definitions
// ============================================================================

extern "C" {

// ============================================================================
// Initialization
// ============================================================================

int goldilocks_cuda_init() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        return -1;
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return -1;
    }

    err = goldilocks_init();
    return (err == cudaSuccess) ? 0 : -1;
}

int poseidon2_cuda_init() {
    cudaError_t err = poseidon2_init();
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Memory Management
// ============================================================================

int cuda_malloc(void** ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_free(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_memcpy_htod(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_memcpy_dtoh(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_memcpy_dtod(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_device_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_get_last_error() {
    cudaError_t err = cudaGetLastError();
    return (int)err;
}

int cuda_peek_at_last_error() {
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

int cuda_mem_get_info(size_t* free, size_t* total) {
    cudaError_t err = cudaMemGetInfo(free, total);
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Goldilocks Field Batch Operations
// ============================================================================

#define BLOCK_SIZE 256

__global__ void gl_batch_add_kernel_ffi(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fa(a[idx]);
        GoldilocksField fb(b[idx]);
        result[idx] = gl_add(fa, fb).value;
    }
}

__global__ void gl_batch_sub_kernel_ffi(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fa(a[idx]);
        GoldilocksField fb(b[idx]);
        result[idx] = gl_sub(fa, fb).value;
    }
}

__global__ void gl_batch_mul_kernel_ffi(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fa(a[idx]);
        GoldilocksField fb(b[idx]);
        result[idx] = gl_mul(fa, fb).value;
    }
}

__global__ void gl_batch_inverse_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fa(a[idx]);
        result[idx] = canonicalize(gl_inverse(fa).value);
    }
}

int gl_batch_add(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_add_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int gl_batch_sub(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_sub_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int gl_batch_mul(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_mul_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int gl_batch_inverse(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_inverse_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Scalar multiplication: result[i] = scalar * a[i]
__global__ void gl_batch_mul_scalar_kernel_ffi(
    uint64_t scalar,
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fs(scalar);
        GoldilocksField fa(a[idx]);
        result[idx] = gl_mul(fs, fa).value;
    }
}

int gl_batch_mul_scalar(uint64_t scalar, const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_mul_scalar_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(scalar, d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Negation: result[i] = -a[i]
__global__ void gl_batch_neg_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fa(a[idx]);
        result[idx] = gl_neg(fa).value;
    }
}

int gl_batch_neg(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_neg_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Square: result[i] = a[i]^2
__global__ void gl_batch_square_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fa(a[idx]);
        result[idx] = gl_square(fa).value;
    }
}

int gl_batch_square(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_square_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Double: result[i] = 2 * a[i]
__global__ void gl_batch_double_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fa(a[idx]);
        result[idx] = gl_double(fa).value;
    }
}

int gl_batch_double(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_double_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Exponentiation: result[i] = a[i]^exp
__global__ void gl_batch_exp_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t exp,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fa(a[idx]);
        result[idx] = gl_exp(fa, exp).value;
    }
}

int gl_batch_exp(const uint64_t* d_a, uint64_t exp, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_exp_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, exp, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Division: result[i] = a[i] / b[i]
__global__ void gl_batch_div_kernel_ffi(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fa(a[idx]);
        GoldilocksField fb(b[idx]);
        result[idx] = canonicalize(gl_div(fa, fb).value);
    }
}

int gl_batch_div(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_batch_div_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Extension Field (Ext2) Batch Operations
// ============================================================================

// Ext2 is represented as [c0, c1] = 2 uint64_t values per element

__global__ void ext2_batch_add_kernel_ffi(
    const uint64_t* __restrict__ a,  // [c0, c1, c0, c1, ...]
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 eb(b[2*idx], b[2*idx+1]);
        GoldilocksExt2 er = ext2_add(ea, eb);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

__global__ void ext2_batch_sub_kernel_ffi(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 eb(b[2*idx], b[2*idx+1]);
        GoldilocksExt2 er = ext2_sub(ea, eb);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

__global__ void ext2_batch_mul_kernel_ffi(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 eb(b[2*idx], b[2*idx+1]);
        GoldilocksExt2 er = ext2_mul(ea, eb);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

__global__ void ext2_batch_inverse_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 er = ext2_inverse(ea);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

int ext2_batch_add_ffi(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_add_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int ext2_batch_sub_ffi(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_sub_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int ext2_batch_mul_ffi(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_mul_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int ext2_batch_inverse_ffi(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_inverse_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext2 scalar multiplication: result[i] = scalar * a[i] (scalar is base field)
__global__ void ext2_batch_mul_scalar_kernel_ffi(
    uint64_t scalar,
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fs(scalar);
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 er = ext2_scalar_mul(fs, ea);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

int ext2_batch_mul_scalar_ffi(uint64_t scalar, const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_mul_scalar_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(scalar, d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext2 negation: result[i] = -a[i]
__global__ void ext2_batch_neg_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 er = ext2_neg(ea);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

int ext2_batch_neg_ffi(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_neg_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext2 square: result[i] = a[i]^2
__global__ void ext2_batch_square_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 er = ext2_square(ea);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

int ext2_batch_square_ffi(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_square_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext2 Frobenius: result[i] = a[i]^p
__global__ void ext2_batch_frobenius_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 er = ext2_frobenius(ea);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

int ext2_batch_frobenius_ffi(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_frobenius_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext2 conjugate: result[i] = conj(a[i]) = (c0, -c1)
__global__ void ext2_batch_conjugate_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 er = ext2_conjugate(ea);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

int ext2_batch_conjugate_ffi(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_conjugate_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext2 exponentiation: result[i] = a[i]^exp
__global__ void ext2_batch_exp_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t exp,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt2 ea(a[2*idx], a[2*idx+1]);
        GoldilocksExt2 er = ext2_exp(ea, exp);
        result[2*idx] = er.c[0].value;
        result[2*idx+1] = er.c[1].value;
    }
}

int ext2_batch_exp_ffi(const uint64_t* d_a, uint64_t exp, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_exp_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, exp, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Conversion: Goldilocks <-> Ext2
// ============================================================================

__global__ void gl_to_ext2_kernel_ffi(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,  // [c0, c1, c0, c1, ...]
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[2*idx] = input[idx];
        output[2*idx+1] = 0;
    }
}

__global__ void ext2_to_gl_kernel_ffi(
    const uint64_t* __restrict__ input,  // [c0, c1, c0, c1, ...]
    uint64_t* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[2*idx];
    }
}

int gl_to_ext2_batch_ffi(const uint64_t* d_input, uint64_t* d_output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_to_ext2_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int ext2_to_gl_batch_ffi(const uint64_t* d_input, uint64_t* d_output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_to_gl_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Extension Field (Ext5) Batch Operations
// ============================================================================

// Ext5 is represented as [c0, c1, c2, c3, c4] = 5 uint64_t values per element

// Ext5 addition: result[i] = a[i] + b[i]
__global__ void ext5_batch_add_kernel_ffi(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt5 ea, eb;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ea.c[i] = GoldilocksField(a[5*idx + i]);
            eb.c[i] = GoldilocksField(b[5*idx + i]);
        }
        GoldilocksExt5 er = ext5_add(ea, eb);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            result[5*idx + i] = er.c[i].value;
        }
    }
}

int ext5_batch_add_ffi(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_add_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext5 subtraction: result[i] = a[i] - b[i]
__global__ void ext5_batch_sub_kernel_ffi(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt5 ea, eb;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ea.c[i] = GoldilocksField(a[5*idx + i]);
            eb.c[i] = GoldilocksField(b[5*idx + i]);
        }
        GoldilocksExt5 er = ext5_sub(ea, eb);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            result[5*idx + i] = er.c[i].value;
        }
    }
}

int ext5_batch_sub_ffi(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_sub_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext5 multiplication: result[i] = a[i] * b[i]
__global__ void ext5_batch_mul_kernel_ffi(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt5 ea, eb;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ea.c[i] = GoldilocksField(a[5*idx + i]);
            eb.c[i] = GoldilocksField(b[5*idx + i]);
        }
        GoldilocksExt5 er = ext5_mul(ea, eb);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            result[5*idx + i] = canonicalize(er.c[i].value);
        }
    }
}

int ext5_batch_mul_ffi(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_mul_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext5 negation: result[i] = -a[i]
__global__ void ext5_batch_neg_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt5 ea;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ea.c[i] = GoldilocksField(a[5*idx + i]);
        }
        GoldilocksExt5 er = ext5_neg(ea);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            result[5*idx + i] = er.c[i].value;
        }
    }
}

int ext5_batch_neg_ffi(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_neg_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext5 square: result[i] = a[i]^2
__global__ void ext5_batch_square_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt5 ea;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ea.c[i] = GoldilocksField(a[5*idx + i]);
        }
        GoldilocksExt5 er = ext5_square(ea);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            result[5*idx + i] = er.c[i].value;
        }
    }
}

int ext5_batch_square_ffi(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_square_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext5 inverse: result[i] = a[i]^(-1)
__global__ void ext5_batch_inverse_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt5 ea;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ea.c[i] = GoldilocksField(a[5*idx + i]);
        }
        GoldilocksExt5 er = ext5_inverse(ea);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            result[5*idx + i] = canonicalize(er.c[i].value);
        }
    }
}

int ext5_batch_inverse_ffi(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_inverse_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext5 scalar multiplication: result[i] = scalar * a[i]
__global__ void ext5_batch_mul_scalar_kernel_ffi(
    uint64_t scalar,
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField fs(scalar);
        GoldilocksExt5 ea;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ea.c[i] = GoldilocksField(a[5*idx + i]);
        }
        GoldilocksExt5 er = ext5_scalar_mul(fs, ea);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            result[5*idx + i] = er.c[i].value;
        }
    }
}

int ext5_batch_mul_scalar_ffi(uint64_t scalar, const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_mul_scalar_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(scalar, d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext5 Frobenius: result[i] = a[i]^p
__global__ void ext5_batch_frobenius_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt5 ea;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ea.c[i] = GoldilocksField(a[5*idx + i]);
        }
        GoldilocksExt5 er = ext5_frobenius(ea);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            result[5*idx + i] = er.c[i].value;
        }
    }
}

int ext5_batch_frobenius_ffi(const uint64_t* d_a, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_frobenius_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext5 exponentiation: result[i] = a[i]^exp
__global__ void ext5_batch_exp_kernel_ffi(
    const uint64_t* __restrict__ a,
    uint64_t exp,
    uint64_t* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksExt5 ea;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ea.c[i] = GoldilocksField(a[5*idx + i]);
        }
        GoldilocksExt5 er = ext5_exp(ea, exp);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            result[5*idx + i] = er.c[i].value;
        }
    }
}

int ext5_batch_exp_ffi(const uint64_t* d_a, uint64_t exp, uint64_t* d_result, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_exp_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_a, exp, d_result, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Conversion: Goldilocks <-> Ext5
// ============================================================================

__global__ void gl_to_ext5_kernel_ffi(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[5*idx] = input[idx];
        output[5*idx + 1] = 0;
        output[5*idx + 2] = 0;
        output[5*idx + 3] = 0;
        output[5*idx + 4] = 0;
    }
}

__global__ void ext5_to_gl_kernel_ffi(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[5*idx];
    }
}

int gl_to_ext5_batch_ffi(const uint64_t* d_input, uint64_t* d_output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_to_ext5_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int ext5_to_gl_batch_ffi(const uint64_t* d_input, uint64_t* d_output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_to_gl_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Poseidon2 Hash Operations
// ============================================================================

__global__ void poseidon2_hash_batch_kernel_ffi(
    const uint64_t* __restrict__ input,  // 8 elements per hash
    uint64_t* __restrict__ output,       // 8 elements per hash
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField state[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            state[i] = GoldilocksField(input[idx * 8 + i]);
        }

        poseidon2_permute_8(state);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            output[idx * 8 + i] = state[i].value;
        }
    }
}

__global__ void poseidon2_compress_kernel_ffi(
    const uint64_t* __restrict__ left,   // 4 elements
    const uint64_t* __restrict__ right,  // 4 elements
    uint64_t* __restrict__ output,       // 4 elements
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField state[8];

        // Load left (4 elements)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            state[i] = GoldilocksField(left[idx * 4 + i]);
        }
        // Load right (4 elements)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            state[4 + i] = GoldilocksField(right[idx * 4 + i]);
        }

        poseidon2_permute_8(state);

        // Output first 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            output[idx * 4 + i] = state[i].value;
        }
    }
}

int poseidon2_hash_batch_ffi(const uint64_t* d_input, uint64_t* d_output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_hash_batch_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int poseidon2_compress_batch_ffi(const uint64_t* d_left, const uint64_t* d_right,
                                  uint64_t* d_output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_compress_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_left, d_right, d_output, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Merkle Tree
// ============================================================================

__global__ void poseidon2_merkle_layer_kernel_ffi(
    const uint64_t* __restrict__ input,  // 4 elements per node
    uint64_t* __restrict__ output,       // 4 elements per node
    int n  // number of output nodes (input has 2*n nodes)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        GoldilocksField state[8];

        // Load left child (4 elements)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            state[i] = GoldilocksField(input[(2*idx) * 4 + i]);
        }
        // Load right child (4 elements)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            state[4 + i] = GoldilocksField(input[(2*idx + 1) * 4 + i]);
        }

        poseidon2_permute_8(state);

        // Output first 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            output[idx * 4 + i] = state[i].value;
        }
    }
}

int poseidon2_merkle_layer_ffi(const uint64_t* d_input, uint64_t* d_output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_merkle_layer_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Device Info
// ============================================================================

int cuda_set_device(int device) {
    cudaError_t err = cudaSetDevice(device);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_get_device(int* device) {
    cudaError_t err = cudaGetDevice(device);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_get_device_count(int* count) {
    cudaError_t err = cudaGetDeviceCount(count);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_get_device_name(int device, char* name, int max_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return -1;

    strncpy(name, prop.name, max_len - 1);
    name[max_len - 1] = '\0';
    return 0;
}

// ============================================================================
// Fiat-Shamir Challenger Operations
// ============================================================================

// Challenger state size in bytes
int challenger_state_size() {
    return sizeof(DuplexChallengerState);
}

// Allocate challenger states on device
int challenger_alloc_states(void** d_states, int n) {
    cudaError_t err = cudaMalloc(d_states, n * sizeof(DuplexChallengerState));
    return (err == cudaSuccess) ? 0 : -1;
}

// Initialize challenger states
int challenger_init_states(void* d_states, int n) {
    cudaError_t err = challenger_batch_init(
        static_cast<DuplexChallengerState*>(d_states), n
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Batch observe: each challenger observes one value
int challenger_observe_ffi(void* d_states, const uint64_t* d_values, int n) {
    cudaError_t err = challenger_batch_observe(
        static_cast<DuplexChallengerState*>(d_states),
        reinterpret_cast<const GoldilocksField*>(d_values),
        n
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Batch observe slice: each challenger observes `count` values
int challenger_observe_slice_ffi(void* d_states, const uint64_t* d_values, int count, int n) {
    cudaError_t err = challenger_batch_observe_slice(
        static_cast<DuplexChallengerState*>(d_states),
        reinterpret_cast<const GoldilocksField*>(d_values),
        count,
        n
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Batch sample: each challenger samples one value
int challenger_sample_ffi(void* d_states, uint64_t* d_outputs, int n) {
    cudaError_t err = challenger_batch_sample(
        static_cast<DuplexChallengerState*>(d_states),
        reinterpret_cast<GoldilocksField*>(d_outputs),
        n
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Batch sample array: each challenger samples `count` values
int challenger_sample_array_ffi(void* d_states, uint64_t* d_outputs, int count, int n) {
    cudaError_t err = challenger_batch_sample_array(
        static_cast<DuplexChallengerState*>(d_states),
        reinterpret_cast<GoldilocksField*>(d_outputs),
        count,
        n
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Batch sample extension field (GF(p^2))
int challenger_sample_ext2_ffi(void* d_states, uint64_t* d_outputs, int n) {
    cudaError_t err = challenger_batch_sample_ext2(
        static_cast<DuplexChallengerState*>(d_states),
        reinterpret_cast<GoldilocksExt2*>(d_outputs),
        n
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Batch sample extension field (GF(p^5))
int challenger_sample_ext5_ffi(void* d_states, uint64_t* d_outputs, int n) {
    cudaError_t err = challenger_batch_sample_ext5(
        static_cast<DuplexChallengerState*>(d_states),
        reinterpret_cast<GoldilocksExt5*>(d_outputs),
        n
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Batch observe extension field (GF(p^2))
int challenger_observe_ext2_ffi(void* d_states, const uint64_t* d_values, int n) {
    cudaError_t err = challenger_batch_observe_ext2(
        static_cast<DuplexChallengerState*>(d_states),
        reinterpret_cast<const GoldilocksExt2*>(d_values),
        n
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Copy challenger state to host
int challenger_copy_state_to_host(void* h_state, const void* d_state) {
    cudaError_t err = cudaMemcpy(
        h_state, d_state, sizeof(DuplexChallengerState), cudaMemcpyDeviceToHost
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Copy challenger state to device
int challenger_copy_state_to_device(void* d_state, const void* h_state) {
    cudaError_t err = cudaMemcpy(
        d_state, h_state, sizeof(DuplexChallengerState), cudaMemcpyHostToDevice
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Eq Lagrange Operations
// ============================================================================

#include "eq_lagrange.cuh"

// eq_dp_all for base field
int eq_dp_all_ffi(
    const uint64_t* d_r,
    uint64_t* d_buf_a,
    uint64_t* d_buf_b,
    int log_n,
    uint64_t** d_result
) {
    GoldilocksField* result_ptr = nullptr;
    cudaError_t err = eq_dp_all(
        reinterpret_cast<const GoldilocksField*>(d_r),
        reinterpret_cast<GoldilocksField*>(d_buf_a),
        reinterpret_cast<GoldilocksField*>(d_buf_b),
        log_n,
        &result_ptr
    );
    if (d_result) {
        *d_result = reinterpret_cast<uint64_t*>(result_ptr);
    }
    return (err == cudaSuccess) ? 0 : -1;
}

// ext2_eq_dp_all for quadratic extension field
int ext2_eq_dp_all_ffi(
    const uint64_t* d_r,
    uint64_t* d_buf_a,
    uint64_t* d_buf_b,
    int log_n,
    uint64_t** d_result
) {
    GoldilocksExt2* result_ptr = nullptr;
    cudaError_t err = ext2_eq_dp_all(
        reinterpret_cast<const GoldilocksExt2*>(d_r),
        reinterpret_cast<GoldilocksExt2*>(d_buf_a),
        reinterpret_cast<GoldilocksExt2*>(d_buf_b),
        log_n,
        &result_ptr
    );
    if (d_result) {
        *d_result = reinterpret_cast<uint64_t*>(result_ptr);
    }
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Partial Evaluation Operations
// ============================================================================

#include "partial_eval.cuh"

int partial_eval_gl_ffi(uint64_t* d_data, const uint64_t* d_r, int log_n, int m) {
    if (m <= 0) return 0;

    // Allocate scratch buffer (needs 2^{log_n - 1} elements)
    size_t scratch_size = (1ULL << log_n) / 2;
    GoldilocksField* d_scratch = nullptr;
    cudaError_t alloc_err = cudaMalloc(&d_scratch, scratch_size * sizeof(GoldilocksField));
    if (alloc_err != cudaSuccess) return -1;

    cudaError_t err = partial_eval_gl(
        reinterpret_cast<GoldilocksField*>(d_data),
        d_scratch,
        reinterpret_cast<const GoldilocksField*>(d_r),
        log_n, m
    );

    cudaFree(d_scratch);
    return (err == cudaSuccess) ? 0 : -1;
}

int partial_eval_ext2_from_gl_ffi(const uint64_t* d_input, uint64_t* d_output, const uint64_t* d_r, int log_n, int m) {
    if (m <= 0) return 0;

    // Allocate scratch buffer for ext2 ping-pong (needs 2^{log_n - 2} ext2 elements)
    GoldilocksExt2* d_scratch = nullptr;
    if (m > 1) {
        size_t scratch_size = (1ULL << log_n) / 4;
        if (scratch_size == 0) scratch_size = 1;
        cudaError_t alloc_err = cudaMalloc(&d_scratch, scratch_size * sizeof(GoldilocksExt2));
        if (alloc_err != cudaSuccess) return -1;
    }

    cudaError_t err = partial_eval_ext2_from_gl(
        reinterpret_cast<const GoldilocksField*>(d_input),
        reinterpret_cast<GoldilocksExt2*>(d_output),
        d_scratch,
        reinterpret_cast<const GoldilocksExt2*>(d_r),
        log_n, m
    );

    if (d_scratch) cudaFree(d_scratch);
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Basefold Operations
// ============================================================================

#include "basefold.cuh"
// Include basefold host wrappers (they depend on goldilocks_kernels.cu and poseidon2_kernels.cu)
// We only need the kernel wrappers and table types, not the test code.
// Since basefold_kernels.cu includes goldilocks_kernels.cu and poseidon2_kernels.cu,
// and those have global kernels that would conflict with wrapper.cu's BLOCK_SIZE,
// we only include the .cuh headers (already included above) and reimplement
// the minimal host wrappers needed for FFI.

// Bit-reversal permutation (in-place)
int basefold_bit_reverse_gl_ffi(uint64_t* d_data, int log_n) {
    size_t n = 1ULL << log_n;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_reverse_permute_gl_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<GoldilocksField*>(d_data), log_n
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int basefold_bit_reverse_ext2_ffi(uint64_t* d_data, int log_n) {
    size_t n = 1ULL << log_n;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_reverse_permute_ext2_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<GoldilocksExt2*>(d_data), log_n
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// BHC interpolation: evals (Type2) -> coeffs (Type2) + bh_evals (Type1)
int basefold_bhc_interpolate_ffi(
    const uint64_t* d_evals,
    uint64_t* d_coeffs,
    uint64_t* d_bh_evals,
    int num_vars
) {
    size_t n = 1ULL << num_vars;
    int grid = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    bhc_interp_first_pass_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksField*>(d_evals),
        reinterpret_cast<GoldilocksField*>(d_coeffs),
        reinterpret_cast<GoldilocksField*>(d_bh_evals),
        n
    );

    for (int k = 1; k < num_vars; k++) {
        size_t half_chunk = 1ULL << k;
        int layer_grid = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bhc_interp_layer_kernel<<<layer_grid, BLOCK_SIZE>>>(
            reinterpret_cast<GoldilocksField*>(d_coeffs), half_chunk, n
        );
    }

    // Bit-reverse bh_evals: Type2 -> Type1
    int br_grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_reverse_permute_gl_kernel<<<br_grid, BLOCK_SIZE>>>(
        reinterpret_cast<GoldilocksField*>(d_bh_evals), num_vars
    );

    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

// Foldable domain encoding: coeffs (Type2) -> codeword (Type1)
int basefold_encode_ffi(
    const uint64_t* d_coeffs,
    uint64_t* d_codeword,
    int num_vars,
    int log_rate
) {
    size_t k = 1ULL << num_vars;
    int rate = 1 << log_rate;
    size_t n_output = k * rate;

    // Step 1: Repetition code
    int grid = (n_output + BLOCK_SIZE - 1) / BLOCK_SIZE;
    repetition_encode_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksField*>(d_coeffs),
        reinterpret_cast<GoldilocksField*>(d_codeword),
        rate, n_output
    );

    // Step 2: Butterfly layers
    for (int i = 0; i < num_vars; i++) {
        size_t half_chunk = 1ULL << (i + log_rate);
        int layer_grid = (n_output / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        foldable_domain_layer_kernel<<<layer_grid, BLOCK_SIZE>>>(
            reinterpret_cast<GoldilocksField*>(d_codeword), half_chunk, n_output
        );
    }

    // Step 3: Bit-reverse to Type1
    int br_grid = (n_output + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_reverse_permute_gl_kernel<<<br_grid, BLOCK_SIZE>>>(
        reinterpret_cast<GoldilocksField*>(d_codeword), num_vars + log_rate
    );

    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

// Base field codeword fold
// table is flat uint64_t array: [point0, weight0, point1, weight1, ...]
int basefold_fold_gl_ffi(
    const uint64_t* d_codeword,
    const uint64_t* d_table,
    uint64_t challenge,
    uint64_t* d_output,
    int pair_count
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    basefold_fold_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksField*>(d_codeword),
        reinterpret_cast<const FoldingEntry*>(d_table),
        GoldilocksField(challenge),
        reinterpret_cast<GoldilocksField*>(d_output),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Mixed codeword fold: F_p codeword, F_{p^2} challenge -> F_{p^2} output
int basefold_fold_mixed_ffi(
    const uint64_t* d_codeword,
    const uint64_t* d_table,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    uint64_t* d_output,
    int pair_count
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GoldilocksExt2 challenge(challenge_c0, challenge_c1);
    basefold_fold_mixed_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksField*>(d_codeword),
        reinterpret_cast<const FoldingEntry*>(d_table),
        challenge,
        reinterpret_cast<GoldilocksExt2*>(d_output),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Ext2 codeword fold: F_{p^2} codeword, F_{p^2} challenge -> F_{p^2} output
int basefold_fold_ext2_ffi(
    const uint64_t* d_codeword,
    const uint64_t* d_table,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    uint64_t* d_output,
    int pair_count
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GoldilocksExt2 challenge(challenge_c0, challenge_c1);
    basefold_fold_ext2_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksExt2*>(d_codeword),
        reinterpret_cast<const FoldingEntry*>(d_table),
        challenge,
        reinterpret_cast<GoldilocksExt2*>(d_output),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Sum-check: interp (in-place), base field
int basefold_sumcheck_interp_gl_ffi(uint64_t* d_data, int pair_count) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sumcheck_interp_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<GoldilocksField*>(d_data), pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Sum-check: interp (in-place), ext2
int basefold_sumcheck_interp_ext2_ffi(uint64_t* d_data, int pair_count) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sumcheck_interp_ext2_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<GoldilocksExt2*>(d_data), pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Sum-check: product, base field. Output: partial block sums for c0, c1, c2.
int basefold_sumcheck_product_gl_ffi(
    const uint64_t* d_eq,
    const uint64_t* d_bh,
    uint64_t* d_partial_c0,
    uint64_t* d_partial_c1,
    uint64_t* d_partial_c2,
    int pair_count,
    int num_blocks
) {
    sumcheck_product_kernel<<<num_blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksField*>(d_eq),
        reinterpret_cast<const GoldilocksField*>(d_bh),
        reinterpret_cast<GoldilocksField*>(d_partial_c0),
        reinterpret_cast<GoldilocksField*>(d_partial_c1),
        reinterpret_cast<GoldilocksField*>(d_partial_c2),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Sum-check: product, mixed (F_p bh, F_{p^2} eq)
int basefold_sumcheck_product_mixed_ffi(
    const uint64_t* d_eq,
    const uint64_t* d_bh,
    uint64_t* d_partial_c0,
    uint64_t* d_partial_c1,
    uint64_t* d_partial_c2,
    int pair_count,
    int num_blocks
) {
    sumcheck_product_mixed_kernel<<<num_blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksExt2*>(d_eq),
        reinterpret_cast<const GoldilocksField*>(d_bh),
        reinterpret_cast<GoldilocksExt2*>(d_partial_c0),
        reinterpret_cast<GoldilocksExt2*>(d_partial_c1),
        reinterpret_cast<GoldilocksExt2*>(d_partial_c2),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Sum-check: product, ext2
int basefold_sumcheck_product_ext2_ffi(
    const uint64_t* d_eq,
    const uint64_t* d_bh,
    uint64_t* d_partial_c0,
    uint64_t* d_partial_c1,
    uint64_t* d_partial_c2,
    int pair_count,
    int num_blocks
) {
    sumcheck_product_ext2_kernel<<<num_blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksExt2*>(d_eq),
        reinterpret_cast<const GoldilocksExt2*>(d_bh),
        reinterpret_cast<GoldilocksExt2*>(d_partial_c0),
        reinterpret_cast<GoldilocksExt2*>(d_partial_c1),
        reinterpret_cast<GoldilocksExt2*>(d_partial_c2),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Fused eval+interp+product for ext2 sum-check round
int fused_sumcheck_round_ext2_ffi(
    const uint64_t* d_eq_in,
    const uint64_t* d_bh_in,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    uint64_t* d_eq_out,
    uint64_t* d_bh_out,
    uint64_t* d_partial_c0,
    uint64_t* d_partial_c1,
    uint64_t* d_partial_c2,
    int pair_count,
    int num_blocks
) {
    GoldilocksExt2 challenge(challenge_c0, challenge_c1);
    fused_sumcheck_round_ext2_kernel<<<num_blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksExt2*>(d_eq_in),
        reinterpret_cast<const GoldilocksExt2*>(d_bh_in),
        challenge,
        reinterpret_cast<GoldilocksExt2*>(d_eq_out),
        reinterpret_cast<GoldilocksExt2*>(d_bh_out),
        reinterpret_cast<GoldilocksExt2*>(d_partial_c0),
        reinterpret_cast<GoldilocksExt2*>(d_partial_c1),
        reinterpret_cast<GoldilocksExt2*>(d_partial_c2),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Sum-check: eval at challenge, base field
int basefold_sumcheck_eval_gl_ffi(
    const uint64_t* d_data,
    uint64_t challenge,
    uint64_t* d_output,
    int pair_count
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sumcheck_eval_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksField*>(d_data),
        GoldilocksField(challenge),
        reinterpret_cast<GoldilocksField*>(d_output),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Sum-check: mixed eval (F_p data, F_{p^2} challenge -> F_{p^2} output)
int basefold_sumcheck_eval_mixed_ffi(
    const uint64_t* d_data,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    uint64_t* d_output,
    int pair_count
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GoldilocksExt2 challenge(challenge_c0, challenge_c1);
    sumcheck_eval_mixed_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksField*>(d_data),
        challenge,
        reinterpret_cast<GoldilocksExt2*>(d_output),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Sum-check: ext2 eval
int basefold_sumcheck_eval_ext2_ffi(
    const uint64_t* d_data,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    uint64_t* d_output,
    int pair_count
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GoldilocksExt2 challenge(challenge_c0, challenge_c1);
    sumcheck_eval_ext2_kernel<<<grid, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksExt2*>(d_data),
        challenge,
        reinterpret_cast<GoldilocksExt2*>(d_output),
        pair_count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Dot product kernel (inline version for FFI, avoids including goldilocks_kernels.cu)
__global__ void basefold_dot_product_gl_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ output,
    int n
) {
    __shared__ uint64_t shared[256];
    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    GoldilocksField sum(0);
    for (size_t i = idx; i < (size_t)n; i += grid_size) {
        GoldilocksField fa(a[i]), fb(b[i]);
        sum = gl_add(sum, gl_mul(fa, fb));
    }
    shared[tid] = sum.value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = gl_add(GoldilocksField(shared[tid]), GoldilocksField(shared[tid + s])).value;
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = shared[0];
}

int basefold_dot_product_gl_ffi(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_partial,
    int n,
    int num_blocks
) {
    basefold_dot_product_gl_kernel<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_partial, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Mixed dot product: F_p * F_{p^2} -> F_{p^2}
int basefold_dot_product_mixed_ffi(
    const uint64_t* d_a,
    const uint64_t* d_b,
    uint64_t* d_partial,
    int n,
    int num_blocks
) {
    ext2_dot_product_mixed_kernel<<<num_blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const GoldilocksField*>(d_a),
        reinterpret_cast<const GoldilocksExt2*>(d_b),
        reinterpret_cast<GoldilocksExt2*>(d_partial),
        n
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Merkle Tree Operations (GPU-resident)
// ============================================================================

// Hash pairs of base-field codeword elements into 4-element Poseidon2 leaf digests.
// Input: 2 field elements per leaf, output: 4 u64 per leaf digest.
__global__ void hash_gl_leaves_kernel(
    const uint64_t* __restrict__ d_codeword,
    uint64_t* __restrict__ d_digests,
    int num_leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_leaves) {
        GoldilocksField state[8];
        // Load 2 codeword elements, zero-pad to width 8
        state[0] = GoldilocksField(d_codeword[2 * idx]);
        state[1] = GoldilocksField(d_codeword[2 * idx + 1]);
        for (int i = 2; i < 8; i++) state[i] = GoldilocksField(0);

        poseidon2_permute_8(state);

        // Output first 4 elements as digest
        for (int i = 0; i < 4; i++) {
            d_digests[idx * 4 + i] = state[i].value;
        }
    }
}

// Hash pairs of ext2 codeword elements into 4-element Poseidon2 leaf digests.
// Input: 2 ext2 elements (4 u64) per leaf, output: 4 u64 per leaf digest.
__global__ void hash_ext2_leaves_kernel(
    const uint64_t* __restrict__ d_codeword,
    uint64_t* __restrict__ d_digests,
    int num_leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_leaves) {
        GoldilocksField state[8];
        // Load 2 ext2 elements = 4 base field elements, zero-pad to 8
        for (int i = 0; i < 4; i++) {
            state[i] = GoldilocksField(d_codeword[4 * idx + i]);
        }
        for (int i = 4; i < 8; i++) state[i] = GoldilocksField(0);

        poseidon2_permute_8(state);

        for (int i = 0; i < 4; i++) {
            d_digests[idx * 4 + i] = state[i].value;
        }
    }
}

// Build merkle tree using in-place layer compression on a contiguous buffer.
// d_tree layout: [leaves: N*4 u64] [layer1: N/2*4 u64] ... [root: 4 u64]
// Assumes leaf digests are already in the first N*4 positions.
static int build_merkle_tree_inplace(uint64_t* d_tree, int num_leaves) {
    int layer_offset = 0;
    int layer_size = num_leaves;
    while (layer_size > 1) {
        int num_pairs = layer_size / 2;
        int grid_size = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t* input = d_tree + layer_offset * 4;
        uint64_t* output = d_tree + (layer_offset + layer_size) * 4;

        poseidon2_merkle_layer_kernel_ffi<<<grid_size, BLOCK_SIZE>>>(input, output, num_pairs);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -1;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return -1;

        layer_offset += layer_size;
        layer_size = num_pairs;
    }
    return 0;
}

// Hash pairs of GL codeword elements into leaf digests, then build full merkle tree.
// d_codeword: codeword of length num_leaves * 2 (base field)
// d_tree: pre-allocated with (2 * num_leaves - 1) * 4 uint64_t
int poseidon2_merkle_tree_gl_ffi(
    const uint64_t* d_codeword,
    uint64_t* d_tree,
    int num_leaves
) {
    // Step 1: Hash leaf pairs
    int grid = (num_leaves + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hash_gl_leaves_kernel<<<grid, BLOCK_SIZE>>>(d_codeword, d_tree, num_leaves);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;

    // Step 2: Build tree layers
    return build_merkle_tree_inplace(d_tree, num_leaves);
}

// Hash pairs of ext2 codeword elements into leaf digests, then build full merkle tree.
// d_codeword: codeword of length num_leaves * 2 ext2 elements = num_leaves * 4 u64
// d_tree: pre-allocated with (2 * num_leaves - 1) * 4 uint64_t
int poseidon2_merkle_tree_ext2_ffi(
    const uint64_t* d_codeword,
    uint64_t* d_tree,
    int num_leaves
) {
    int grid = (num_leaves + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hash_ext2_leaves_kernel<<<grid, BLOCK_SIZE>>>(d_codeword, d_tree, num_leaves);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;

    return build_merkle_tree_inplace(d_tree, num_leaves);
}

// ============================================================================
// Sumcheck Prover Operations
// ============================================================================

int sumcheck_round_message_ffi(
    const uint64_t* d_polys,
    uint64_t* d_partial,
    int d,
    size_t original_size,
    size_t half,
    int num_blocks
) {
    sumcheck_round_message_kernel<<<num_blocks, SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_partial, d, original_size, half
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

int sumcheck_fold_ffi(
    const uint64_t* d_input,
    uint64_t* d_output,
    uint64_t challenge,
    int d,
    size_t original_size,
    size_t half
) {
    int grid = ((int)half + SUMCHECK_BLOCK_SIZE - 1) / SUMCHECK_BLOCK_SIZE;
    sumcheck_fold_kernel<<<grid, SUMCHECK_BLOCK_SIZE>>>(
        d_input, d_output, challenge, d, original_size, half
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

int sumcheck_round_message_ext2_ffi(
    const uint64_t* d_polys,
    uint64_t* d_partial,
    int d,
    size_t original_size,
    size_t half,
    int num_blocks
) {
    sumcheck_round_message_ext2_kernel<<<num_blocks, SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_partial, d, original_size, half
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

int sumcheck_fold_ext2_ffi(
    const uint64_t* d_input,
    uint64_t* d_output,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    int d,
    size_t original_size,
    size_t half
) {
    int grid = ((int)half + SUMCHECK_BLOCK_SIZE - 1) / SUMCHECK_BLOCK_SIZE;
    sumcheck_fold_ext2_kernel<<<grid, SUMCHECK_BLOCK_SIZE>>>(
        d_input, d_output, challenge_c0, challenge_c1, d, original_size, half
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Ext2 Scale-Accumulate: acc[i] += scalar * src[i]
// ============================================================================

__global__ void ext2_scale_accumulate_kernel(
    uint64_t scalar_c0, uint64_t scalar_c1,
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ acc,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    GoldilocksExt2 s(scalar_c0, scalar_c1);
    GoldilocksExt2 v(src[2*idx], src[2*idx+1]);
    GoldilocksExt2 a(acc[2*idx], acc[2*idx+1]);
    GoldilocksExt2 result = ext2_add(a, ext2_mul(s, v));
    acc[2*idx] = result.c[0].value;
    acc[2*idx+1] = result.c[1].value;
}

int ext2_scale_accumulate_ffi(
    uint64_t scalar_c0, uint64_t scalar_c1,
    const uint64_t* d_src, uint64_t* d_acc,
    int n
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_scale_accumulate_kernel<<<grid_size, BLOCK_SIZE>>>(scalar_c0, scalar_c1, d_src, d_acc, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Bit permutation: out[permuted_idx] = in[idx]
// perm_map[old_var] = new_var, applied to index bits
// ============================================================================

// Gather version: thread at new_idx reads from old_idx (good write coalescing)
// inv_perm[new_var] = old_var
__global__ void bit_permute_kernel(
    const uint64_t* __restrict__ d_input,
    uint64_t* __restrict__ d_output,
    const int* __restrict__ d_inv_perm,
    int n_bits,
    int total
) {
    int new_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_idx >= total) return;

    int old_idx = 0;
    for (int bit = 0; bit < n_bits; bit++) {
        if (new_idx & (1 << bit)) {
            old_idx |= (1 << d_inv_perm[bit]);
        }
    }
    d_output[new_idx] = d_input[old_idx];
}

int bit_permute_gl_ffi(
    const uint64_t* d_input, uint64_t* d_output,
    const int* d_perm_map,
    int n_bits, int total
) {
    int grid_size = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_permute_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, d_perm_map, n_bits, total);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Fused Permute + Partial Eval
// ============================================================================

int fused_permute_partial_eval_ffi(
    const uint64_t* d_evals,
    uint64_t* d_output,
    const uint64_t* d_eq_table,
    const uint32_t* d_lo_lut,
    const uint32_t* d_hi_lut,
    int n, int m, int half, int output_size,
    int smem_bytes
) {
    // Set max dynamic shared memory for the kernel (needed for large LUTs)
    cudaFuncSetAttribute(
        fused_permute_partial_eval_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );

    int num_blocks = output_size;
    // Cap blocks to avoid launching too many for very large output
    if (num_blocks > 65536) num_blocks = 65536;

    fused_permute_partial_eval_kernel<<<num_blocks, FUSED_BLOCK_SIZE, smem_bytes>>>(
        d_evals, d_output, d_eq_table,
        d_lo_lut, d_hi_lut,
        n, m, half, output_size
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Monolith Hash Operations
// ============================================================================

int monolith_cuda_init() {
    cudaError_t err = monolith_init();
    return (err == cudaSuccess) ? 0 : -1;
}

int monolith_merkle_tree_gl_ffi(
    const uint64_t* d_codeword,
    uint64_t* d_tree,
    int num_leaves
) {
    cudaError_t err = monolith_build_merkle_tree_8(
        (const GoldilocksField*)d_codeword,
        (GoldilocksField*)d_tree,
        num_leaves
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Monolith ext2 leaf hash kernel: hash pairs of ext2 elements into 4-u64 digests.
// Layout matches poseidon2: d_codeword[4*idx..4*idx+3] -> d_digests[4*idx..4*idx+3]
__global__ void monolith_hash_ext2_leaf_4u64_kernel(
    const uint64_t* __restrict__ d_codeword,
    uint64_t* __restrict__ d_digests,
    int num_leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves) return;

    GoldilocksField state[MONOLITH_WIDTH];
    // Load 2 ext2 elements = 4 base field elements
    for (int i = 0; i < 4; i++) {
        state[i] = GoldilocksField(d_codeword[4 * idx + i]);
    }
    // Zero padding
    for (int i = 4; i < MONOLITH_WIDTH; i++) state[i] = GoldilocksField(0);

    monolith_permute_12(state);

    // Output first 4 elements as 4-u64 digest
    for (int i = 0; i < 4; i++) {
        d_digests[idx * 4 + i] = state[i].value;
    }
}

// Monolith tree layer kernel for 4-u64 digests (matches poseidon2 tree layout)
__global__ void monolith_tree_compress_4u64_kernel(
    GoldilocksField* tree,
    int num_leaves,
    int current_layer_start,
    int current_layer_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_layer_size / 2) return;

    int parent_layer_start = current_layer_start + current_layer_size * 4;

    const GoldilocksField* left = tree + current_layer_start + idx * 8;
    const GoldilocksField* right = tree + current_layer_start + idx * 8 + 4;
    GoldilocksField* output = tree + parent_layer_start + idx * 4;

    monolith_compress(left, right, output);
}

int monolith_merkle_tree_ext2_ffi(
    const uint64_t* d_codeword,
    uint64_t* d_tree,
    int num_leaves
) {
    // Step 1: Hash ext2 leaf pairs into 4-u64 digests
    int grid = (num_leaves + 256 - 1) / 256;
    monolith_hash_ext2_leaf_4u64_kernel<<<grid, 256>>>(d_codeword, d_tree, num_leaves);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;

    // Step 2: Build tree layers (4 u64 per node)
    const int chunk_size = 4;
    int current_layer_start = 0;
    int current_layer_size = num_leaves;

    while (current_layer_size > 1) {
        int num_pairs = current_layer_size / 2;
        grid = (num_pairs + 256 - 1) / 256;

        monolith_tree_compress_4u64_kernel<<<grid, 256>>>(
            (GoldilocksField*)d_tree, num_leaves, current_layer_start, current_layer_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) return -1;
        err = cudaStreamSynchronize(0);
        if (err != cudaSuccess) return -1;

        current_layer_start += current_layer_size * chunk_size;
        current_layer_size = num_pairs;
    }

    return 0;
}

// Test kernel: run monolith_permute_12 on a single state
__global__ void monolith_permute_test_kernel(GoldilocksField* state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        monolith_permute_12(state);
    }
}

// FFI: run single Monolith permutation on GPU, compare with host
int monolith_permute_test_ffi(
    const uint64_t* h_input,
    uint64_t* h_output
) {
    GoldilocksField* d_state;
    cudaMalloc(&d_state, 12 * sizeof(GoldilocksField));
    cudaMemcpy(d_state, h_input, 12 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    monolith_permute_test_kernel<<<1, 1>>>(d_state);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(d_state); return -1; }
    cudaMemcpy(h_output, d_state, 12 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    return 0;
}

// ============================================================================
// Generic einsum kernel (Goldilocks base field, 2 inputs)
// ============================================================================
//
// Each thread computes one output element by iterating over the summation
// indices and accumulating the product of the two inputs. Indices use the
// little-endian flat-index convention (first dim has stride 1) that matches
// `einsum_compute` on the CPU side of zk-torch-3.
//
// All stride/dim arrays are passed by value as fixed-size int[8] structs to
// keep them in registers without an extra device allocation.

#define EINSUM_MAX_NDIM 8

struct EinsumDimSpec {
    int ndim;
    int dims[EINSUM_MAX_NDIM];
    int strides_a[EINSUM_MAX_NDIM];
    int strides_b[EINSUM_MAX_NDIM];
};

__global__ void gl_einsum2_kernel_ffi(
    const uint64_t* __restrict__ A,
    const uint64_t* __restrict__ B,
    uint64_t* __restrict__ C,
    int out_size,
    int sum_size,
    EinsumDimSpec out_spec,
    EinsumDimSpec sum_spec
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int base_a = 0;
    int base_b = 0;
    int rem = idx;
    for (int d = 0; d < out_spec.ndim; d++) {
        int c = rem % out_spec.dims[d];
        rem /= out_spec.dims[d];
        base_a += c * out_spec.strides_a[d];
        base_b += c * out_spec.strides_b[d];
    }

    GoldilocksField acc(0);
    for (int s = 0; s < sum_size; s++) {
        int sa = base_a;
        int sb = base_b;
        int sr = s;
        for (int d = 0; d < sum_spec.ndim; d++) {
            int c = sr % sum_spec.dims[d];
            sr /= sum_spec.dims[d];
            sa += c * sum_spec.strides_a[d];
            sb += c * sum_spec.strides_b[d];
        }
        GoldilocksField a_val(A[sa]);
        GoldilocksField b_val(B[sb]);
        acc = gl_add(acc, gl_mul(a_val, b_val));
    }
    C[idx] = acc.value;
}

int gl_einsum2(
    const uint64_t* d_A,
    const uint64_t* d_B,
    uint64_t* d_C,
    int out_size,
    int sum_size,
    int out_ndim,
    const int* out_dims,
    const int* out_strides_a,
    const int* out_strides_b,
    int sum_ndim,
    const int* sum_dims,
    const int* sum_strides_a,
    const int* sum_strides_b
) {
    if (out_ndim > EINSUM_MAX_NDIM || sum_ndim > EINSUM_MAX_NDIM) return -1;

    EinsumDimSpec out_spec = {0};
    out_spec.ndim = out_ndim;
    for (int i = 0; i < out_ndim; i++) {
        out_spec.dims[i] = out_dims[i];
        out_spec.strides_a[i] = out_strides_a[i];
        out_spec.strides_b[i] = out_strides_b[i];
    }

    EinsumDimSpec sum_spec = {0};
    sum_spec.ndim = sum_ndim;
    for (int i = 0; i < sum_ndim; i++) {
        sum_spec.dims[i] = sum_dims[i];
        sum_spec.strides_a[i] = sum_strides_a[i];
        sum_spec.strides_b[i] = sum_strides_b[i];
    }

    int block = 256;
    int grid = (out_size + block - 1) / block;
    gl_einsum2_kernel_ffi<<<grid, block>>>(d_A, d_B, d_C, out_size, sum_size, out_spec, sum_spec);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Single-input variant for unary einsums (transpose / sum-reduction).
__global__ void gl_einsum1_kernel_ffi(
    const uint64_t* __restrict__ A,
    uint64_t* __restrict__ C,
    int out_size,
    int sum_size,
    EinsumDimSpec out_spec,
    EinsumDimSpec sum_spec
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int base_a = 0;
    int rem = idx;
    for (int d = 0; d < out_spec.ndim; d++) {
        int c = rem % out_spec.dims[d];
        rem /= out_spec.dims[d];
        base_a += c * out_spec.strides_a[d];
    }

    GoldilocksField acc(0);
    for (int s = 0; s < sum_size; s++) {
        int sa = base_a;
        int sr = s;
        for (int d = 0; d < sum_spec.ndim; d++) {
            int c = sr % sum_spec.dims[d];
            sr /= sum_spec.dims[d];
            sa += c * sum_spec.strides_a[d];
        }
        GoldilocksField a_val(A[sa]);
        acc = gl_add(acc, a_val);
    }
    C[idx] = acc.value;
}

// ============================================================================
// Bit decomposition: for ScaleDown / ScaleUp / NonNegative auxiliary witnesses.
// ============================================================================
//
// Mirrors zk_torch_3::basicblock::scale::ScaleDown::run on the CPU side.
// Each input element x (Goldilocks field, interpreted as signed via
// int_val = (v < p/2) ? v : v - p) is split into:
//   q = floor-div toward zero by sf  (rounded toward -inf for negatives via
//       -ceil_div(-x, sf))
//   r = x - q * sf, always in [0, sf-1]
// Quotient is written back as a field element. Remainder is decomposed into
// 32 bits and scattered into `bits[i + bit * n]` (little-endian: input dim has
// stride 1, bit dim has stride n).

__device__ __forceinline__ int64_t gl_to_signed(uint64_t v) {
    // p/2 = 0x7FFFFFFF80000000
    if (v < (GOLDILOCKS_PRIME >> 1)) {
        return (int64_t)v;
    }
    return (int64_t)v - (int64_t)GOLDILOCKS_PRIME;
}

__device__ __forceinline__ uint64_t gl_from_signed(int64_t s) {
    if (s >= 0) {
        uint64_t u = (uint64_t)s;
        return (u >= GOLDILOCKS_PRIME) ? (u - GOLDILOCKS_PRIME) : u;
    }
    // s in (-p, 0)
    uint64_t neg = (uint64_t)(-s);
    return (neg >= GOLDILOCKS_PRIME) ? 0 : (GOLDILOCKS_PRIME - neg);
}

__global__ void gl_scale_down_kernel_ffi(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ quotients,
    uint64_t* __restrict__ bits,
    int n,
    uint64_t sf
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int64_t x = gl_to_signed(input[i]);
    int64_t sf_i = (int64_t)sf;
    int64_t q;
    if (x >= 0) {
        q = x / sf_i;
    } else {
        int64_t neg = -x;
        q = -((neg + sf_i - 1) / sf_i);
    }
    int64_t r = x - q * sf_i;  // in [0, sf-1]

    quotients[i] = gl_from_signed(q);

    uint32_t value = (uint32_t)r;
    for (int bit = 0; bit < 32; bit++) {
        bits[(size_t)i + (size_t)bit * (size_t)n] = (uint64_t)((value >> bit) & 1u);
    }
}

int gl_scale_down(
    const uint64_t* d_input,
    uint64_t* d_quotients,
    uint64_t* d_bits,
    int n,
    uint64_t sf
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gl_scale_down_kernel_ffi<<<grid, block>>>(d_input, d_quotients, d_bits, n, sf);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ScaleUp: y = x * sf in the field (no integer overflow concern — Goldilocks
// modular mul handles it). Bit polynomial is all zeros. We expose only the
// quotient kernel; the bit buffer can just be cudaMemset to 0 on the host.
__global__ void gl_scale_up_kernel_ffi(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int n,
    uint64_t sf
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    GoldilocksField fa(input[i]);
    GoldilocksField fs(sf);
    output[i] = gl_mul(fa, fs).value;
}

int gl_scale_up(
    const uint64_t* d_input,
    uint64_t* d_output,
    int n,
    uint64_t sf
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gl_scale_up_kernel_ffi<<<grid, block>>>(d_input, d_output, n, sf);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// NonNegative bit-decomp: decompose each input into 32 bits using the raw
// unsigned u64 value (not signed). Matches NonNegative::run: only values
// < 2^32 produce real bits; everything else produces all-zero bits (the
// verifier's eval_to_check catches the mismatch and rejects).
__global__ void gl_decompose_bits32_kernel_ffi(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ bits,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t x = input[i];
    if (x < (1ULL << 32)) {
        uint32_t value = (uint32_t)x;
        for (int bit = 0; bit < 32; bit++) {
            bits[(size_t)i + (size_t)bit * (size_t)n] = (uint64_t)((value >> bit) & 1u);
        }
    } else {
        for (int bit = 0; bit < 32; bit++) {
            bits[(size_t)i + (size_t)bit * (size_t)n] = 0ULL;
        }
    }
}

int gl_decompose_bits32(const uint64_t* d_input, uint64_t* d_bits, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gl_decompose_bits32_kernel_ffi<<<grid, block>>>(d_input, d_bits, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Memset a device buffer to zero (helper for ScaleUp's bit witness).
int gl_memset_zero(uint64_t* d_buf, size_t n_u64) {
    cudaError_t err = cudaMemset(d_buf, 0, n_u64 * sizeof(uint64_t));
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Conv2D — direct convolution (no im2col/matmul)
// ============================================================================
//
// Mirrors zk_torch_3::basicblock::conv::Conv2D::run on the CPU side.
//   X     : [c_in,  h_in_pad,  w_in_pad]    little-endian (w_in stride 1)
//   W_flat: [c_out, c_in_pad,  s_kernel_pad] little-endian (j stride 1)
//   Y     : [c_out_pad, h_out_pad, w_out_pad] (must be pre-zeroed; only valid
//                                              region [c_out, h_out, w_out] is
//                                              written).
// One thread per (d, ho, wo) accumulates the c_in × kernel_h × kernel_w sum.

__global__ void gl_conv2d_kernel_ffi(
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ W_flat,
    uint64_t* __restrict__ Y,
    int c_out, int h_out, int w_out,
    int c_in,  int kernel_h, int kernel_w,
    int conv_stride_h, int conv_stride_w,
    int dilation_h, int dilation_w,
    int w_in_pad, int h_in_pad,
    int c_in_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int stride_w_val
) {
    int total = c_out * h_out * w_out;
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total) return;

    int wo = flat_idx % w_out;
    int tmp = flat_idx / w_out;
    int ho = tmp % h_out;
    int d  = tmp / h_out;

    GoldilocksField acc(0);
    for (int c = 0; c < c_in; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int ih = ho * conv_stride_h + kh * dilation_h;
                int iw = wo * conv_stride_w + kw * dilation_w;
                int x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                int j = kh * dilation_h * stride_w_val + kw * dilation_w;
                int wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                GoldilocksField xv(X[x_idx]);
                GoldilocksField wv(W_flat[wf_idx]);
                acc = gl_add(acc, gl_mul(xv, wv));
            }
        }
    }
    int out_idx = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
    Y[out_idx] = acc.value;
}

int gl_conv2d(
    const uint64_t* d_X,
    const uint64_t* d_W_flat,
    uint64_t* d_Y,
    int c_out, int h_out, int w_out,
    int c_in,  int kernel_h, int kernel_w,
    int conv_stride_h, int conv_stride_w,
    int dilation_h, int dilation_w,
    int w_in_pad, int h_in_pad,
    int c_in_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int stride_w_val
) {
    int total = c_out * h_out * w_out;
    if (total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    gl_conv2d_kernel_ffi<<<grid, block>>>(
        d_X, d_W_flat, d_Y,
        c_out, h_out, w_out,
        c_in, kernel_h, kernel_w,
        conv_stride_h, conv_stride_w,
        dilation_h, dilation_w,
        w_in_pad, h_in_pad,
        c_in_pad, s_kernel_pad,
        w_out_pad, h_out_pad,
        stride_w_val
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// FlattenKernel: scatter W[C_out, C_in, kH, kW] → W_flat[C_out, C_in, S_kernel_pad]
// where the inner index maps as j = kh*dilation_h*S_w + kw*dilation_w. Output
// must be pre-zeroed (most positions stay 0 due to dilation gaps).

__global__ void gl_flatten_kernel_ffi(
    const uint64_t* __restrict__ W,
    uint64_t* __restrict__ W_flat,
    int c_out, int c_in, int kh_size, int kw_size,
    int kw_pad, int kh_pad,
    int c_in_pad, int s_kernel_pad,
    int dilation_h, int dilation_w, int s_w
) {
    int total = c_out * c_in * kh_size * kw_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int kw = idx % kw_size;
    int t1 = idx / kw_size;
    int kh = t1 % kh_size;
    int t2 = t1 / kh_size;
    int c  = t2 % c_in;
    int d  = t2 / c_in;

    int w_idx  = kw + kh * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad;
    int j      = kh * dilation_h * s_w + kw * dilation_w;
    int wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
    W_flat[wf_idx] = W[w_idx];
}

int gl_flatten_kernel2d(
    const uint64_t* d_W,
    uint64_t* d_W_flat,
    int c_out, int c_in, int kh, int kw,
    int kw_pad, int kh_pad,
    int c_in_pad, int s_kernel_pad,
    int dilation_h, int dilation_w, int s_w
) {
    int total = c_out * c_in * kh * kw;
    if (total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    gl_flatten_kernel_ffi<<<grid, block>>>(
        d_W, d_W_flat, c_out, c_in, kh, kw,
        kw_pad, kh_pad, c_in_pad, s_kernel_pad,
        dilation_h, dilation_w, s_w
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// ReLUHelper: neg[i] = max(0, -x[i]) — for each element, if x is "negative"
// (field value > p/2) write (p - x), else write 0.
// ============================================================================

__global__ void gl_relu_helper_kernel_ffi(
    const uint64_t* __restrict__ x,
    uint64_t* __restrict__ neg,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t v = x[i];
    // p/2 = 0x7FFFFFFF80000000; "negative" if strictly greater.
    neg[i] = (v > (GOLDILOCKS_PRIME >> 1)) ? (GOLDILOCKS_PRIME - v) : 0ULL;
}

int gl_relu_helper(const uint64_t* d_x, uint64_t* d_neg, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gl_relu_helper_kernel_ffi<<<grid, block>>>(d_x, d_neg, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ProductZeroCheck: certificate is just a zero buffer of the input's size.
// We expose this as a helper (caller can also just memset).
int gl_zero_buffer(uint64_t* d_buf, int n) {
    cudaError_t err = cudaMemset(d_buf, 0, (size_t)n * sizeof(uint64_t));
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Conv3D — direct 3D convolution
// ============================================================================
//
//   X     : [c_in,  d_in_pad,  h_in_pad,  w_in_pad]
//   W_flat: [c_out, c_in_pad,  s_kernel_pad]   (already flattened in 3D form)
//   Y     : [c_out_pad, d_out_pad, h_out_pad, w_out_pad] (pre-zeroed)

__global__ void gl_conv3d_kernel_ffi(
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ W_flat,
    uint64_t* __restrict__ Y,
    int c_out, int d_out, int h_out, int w_out,
    int c_in,  int kernel_d, int kernel_h, int kernel_w,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int w_in_pad, int h_in_pad, int d_in_pad,
    int c_in_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad, int d_out_pad,
    int stride_h_val, int stride_w_val
) {
    int total = c_out * d_out * h_out * w_out;
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total) return;

    int wo = flat_idx % w_out;
    int t1 = flat_idx / w_out;
    int ho = t1 % h_out;
    int t2 = t1 / h_out;
    int do_ = t2 % d_out;
    int d   = t2 / d_out;

    GoldilocksField acc(0);
    for (int c = 0; c < c_in; c++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int id = do_ * conv_stride_d + kd;
                    int ih = ho  * conv_stride_h + kh;
                    int iw = wo  * conv_stride_w + kw;
                    int x_idx = iw + ih * w_in_pad
                              + id * w_in_pad * h_in_pad
                              + c  * w_in_pad * h_in_pad * d_in_pad;
                    int j = kd * stride_h_val + kh * stride_w_val + kw;
                    int wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                    GoldilocksField xv(X[x_idx]);
                    GoldilocksField wv(W_flat[wf_idx]);
                    acc = gl_add(acc, gl_mul(xv, wv));
                }
            }
        }
    }
    int out_idx = wo + ho * w_out_pad
                + do_ * w_out_pad * h_out_pad
                + d   * w_out_pad * h_out_pad * d_out_pad;
    Y[out_idx] = acc.value;
}

int gl_conv3d(
    const uint64_t* d_X, const uint64_t* d_W_flat, uint64_t* d_Y,
    int c_out, int d_out, int h_out, int w_out,
    int c_in,  int kernel_d, int kernel_h, int kernel_w,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int w_in_pad, int h_in_pad, int d_in_pad,
    int c_in_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad, int d_out_pad,
    int stride_h_val, int stride_w_val
) {
    int total = c_out * d_out * h_out * w_out;
    if (total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    gl_conv3d_kernel_ffi<<<grid, block>>>(
        d_X, d_W_flat, d_Y,
        c_out, d_out, h_out, w_out,
        c_in, kernel_d, kernel_h, kernel_w,
        conv_stride_d, conv_stride_h, conv_stride_w,
        w_in_pad, h_in_pad, d_in_pad,
        c_in_pad, s_kernel_pad,
        w_out_pad, h_out_pad, d_out_pad,
        stride_h_val, stride_w_val
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// FlattenKernel3D: scatter W[C_out, C_in, kD, kH, kW] → W_flat[C_out, C_in, S_kernel_pad]
// j = kd*stride_h + kh*stride_w + kw

__global__ void gl_flatten_kernel3d_kernel_ffi(
    const uint64_t* __restrict__ W,
    uint64_t* __restrict__ W_flat,
    int c_out, int c_in, int kd_size, int kh_size, int kw_size,
    int kw_pad, int kh_pad, int kd_pad,
    int c_in_pad, int s_kernel_pad,
    int stride_h, int stride_w
) {
    int total = c_out * c_in * kd_size * kh_size * kw_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int kw = idx % kw_size;
    int t1 = idx / kw_size;
    int kh = t1 % kh_size;
    int t2 = t1 / kh_size;
    int kd = t2 % kd_size;
    int t3 = t2 / kd_size;
    int c  = t3 % c_in;
    int d  = t3 / c_in;

    int w_idx = kw + kh * kw_pad + kd * kw_pad * kh_pad
              + c  * kw_pad * kh_pad * kd_pad
              + d  * kw_pad * kh_pad * kd_pad * c_in_pad;
    int j = kd * stride_h + kh * stride_w + kw;
    int wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
    W_flat[wf_idx] = W[w_idx];
}

int gl_flatten_kernel3d(
    const uint64_t* d_W, uint64_t* d_W_flat,
    int c_out, int c_in, int kd, int kh, int kw,
    int kw_pad, int kh_pad, int kd_pad,
    int c_in_pad, int s_kernel_pad,
    int stride_h, int stride_w
) {
    int total = c_out * c_in * kd * kh * kw;
    if (total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    gl_flatten_kernel3d_kernel_ffi<<<grid, block>>>(
        d_W, d_W_flat, c_out, c_in, kd, kh, kw,
        kw_pad, kh_pad, kd_pad, c_in_pad, s_kernel_pad,
        stride_h, stride_w
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// DepthwiseConv2D: each channel convolved independently.
//   X[C, H_in_pad, W_in_pad], W_flat[C, S_kernel_pad] → Y[C_pad, H_out_pad, W_out_pad]

__global__ void gl_depthwise_conv2d_kernel_ffi(
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ W_flat,
    uint64_t* __restrict__ Y,
    int channels, int h_out, int w_out,
    int kernel_h, int kernel_w,
    int conv_stride_h, int conv_stride_w,
    int w_in_pad, int h_in_pad,
    int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int stride_w_val
) {
    int total = channels * h_out * w_out;
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total) return;

    int wo = flat_idx % w_out;
    int t1 = flat_idx / w_out;
    int ho = t1 % h_out;
    int c  = t1 / h_out;

    GoldilocksField acc(0);
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int ih = ho * conv_stride_h + kh;
            int iw = wo * conv_stride_w + kw;
            int x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
            int j = kh * stride_w_val + kw;
            int wf_idx = j + c * s_kernel_pad;
            GoldilocksField xv(X[x_idx]);
            GoldilocksField wv(W_flat[wf_idx]);
            acc = gl_add(acc, gl_mul(xv, wv));
        }
    }
    int out_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
    Y[out_idx] = acc.value;
}

int gl_depthwise_conv2d(
    const uint64_t* d_X, const uint64_t* d_W_flat, uint64_t* d_Y,
    int channels, int h_out, int w_out,
    int kernel_h, int kernel_w,
    int conv_stride_h, int conv_stride_w,
    int w_in_pad, int h_in_pad,
    int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int stride_w_val
) {
    int total = channels * h_out * w_out;
    if (total <= 0) return 0;
    int block = 256;
    int grid = (total + block - 1) / block;
    gl_depthwise_conv2d_kernel_ffi<<<grid, block>>>(
        d_X, d_W_flat, d_Y,
        channels, h_out, w_out,
        kernel_h, kernel_w,
        conv_stride_h, conv_stride_w,
        w_in_pad, h_in_pad,
        s_kernel_pad,
        w_out_pad, h_out_pad,
        stride_w_val
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int gl_einsum1(
    const uint64_t* d_A,
    uint64_t* d_C,
    int out_size,
    int sum_size,
    int out_ndim,
    const int* out_dims,
    const int* out_strides_a,
    int sum_ndim,
    const int* sum_dims,
    const int* sum_strides_a
) {
    if (out_ndim > EINSUM_MAX_NDIM || sum_ndim > EINSUM_MAX_NDIM) return -1;

    EinsumDimSpec out_spec = {0};
    out_spec.ndim = out_ndim;
    for (int i = 0; i < out_ndim; i++) {
        out_spec.dims[i] = out_dims[i];
        out_spec.strides_a[i] = out_strides_a[i];
    }

    EinsumDimSpec sum_spec = {0};
    sum_spec.ndim = sum_ndim;
    for (int i = 0; i < sum_ndim; i++) {
        sum_spec.dims[i] = sum_dims[i];
        sum_spec.strides_a[i] = sum_strides_a[i];
    }

    int block = 256;
    int grid = (out_size + block - 1) / block;
    gl_einsum1_kernel_ffi<<<grid, block>>>(d_A, d_C, out_size, sum_size, out_spec, sum_spec);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

} // extern "C"

/**
 * Goldilocks Extension Fields - CUDA Kernels
 *
 * Batch processing kernels for quadratic (degree 2) and quintic (degree 5) extensions.
 */

#include "extension.cuh"
#include <stdio.h>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256

// ============================================================================
// Quadratic Extension (Degree 2) Batch Kernels
// ============================================================================

/**
 * Batch addition for GF(p^2)
 */
__global__ void ext2_batch_add_kernel(
    const GoldilocksExt2* __restrict__ a,
    const GoldilocksExt2* __restrict__ b,
    GoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext2_add(a[idx], b[idx]);
    }
}

/**
 * Batch subtraction for GF(p^2)
 */
__global__ void ext2_batch_sub_kernel(
    const GoldilocksExt2* __restrict__ a,
    const GoldilocksExt2* __restrict__ b,
    GoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext2_sub(a[idx], b[idx]);
    }
}

/**
 * Batch multiplication for GF(p^2)
 */
__global__ void ext2_batch_mul_kernel(
    const GoldilocksExt2* __restrict__ a,
    const GoldilocksExt2* __restrict__ b,
    GoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext2_mul(a[idx], b[idx]);
    }
}

/**
 * Batch squaring for GF(p^2)
 */
__global__ void ext2_batch_square_kernel(
    const GoldilocksExt2* __restrict__ a,
    GoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext2_square(a[idx]);
    }
}

/**
 * Batch inversion for GF(p^2)
 */
__global__ void ext2_batch_inverse_kernel(
    const GoldilocksExt2* __restrict__ a,
    GoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext2_inverse(a[idx]);
    }
}

/**
 * Batch Frobenius for GF(p^2)
 */
__global__ void ext2_batch_frobenius_kernel(
    const GoldilocksExt2* __restrict__ a,
    GoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext2_frobenius(a[idx]);
    }
}

/**
 * Batch conversion: Goldilocks -> GF(p^2)
 * Maps a -> (a, 0)
 */
__global__ void gl_to_ext2_batch_kernel(
    const GoldilocksField* __restrict__ input,
    GoldilocksExt2* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gl_to_ext2(input[idx]);
    }
}

/**
 * Batch conversion: GF(p^2) -> Goldilocks (extracts c[0])
 */
__global__ void ext2_to_gl_batch_kernel(
    const GoldilocksExt2* __restrict__ input,
    GoldilocksField* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = ext2_to_gl(input[idx]);
    }
}

/**
 * Batch scalar multiplication for GF(p^2)
 */
__global__ void ext2_batch_scalar_mul_kernel(
    GoldilocksField scalar,
    const GoldilocksExt2* __restrict__ a,
    GoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext2_scalar_mul(scalar, a[idx]);
    }
}

/**
 * Batch exponentiation for GF(p^2)
 */
__global__ void ext2_batch_exp_kernel(
    const GoldilocksExt2* __restrict__ base,
    uint64_t exp,
    GoldilocksExt2* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext2_exp(base[idx], exp);
    }
}

// ============================================================================
// Quintic Extension (Degree 5) Batch Kernels
// ============================================================================

/**
 * Batch addition for GF(p^5)
 */
__global__ void ext5_batch_add_kernel(
    const GoldilocksExt5* __restrict__ a,
    const GoldilocksExt5* __restrict__ b,
    GoldilocksExt5* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext5_add(a[idx], b[idx]);
    }
}

/**
 * Batch subtraction for GF(p^5)
 */
__global__ void ext5_batch_sub_kernel(
    const GoldilocksExt5* __restrict__ a,
    const GoldilocksExt5* __restrict__ b,
    GoldilocksExt5* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext5_sub(a[idx], b[idx]);
    }
}

/**
 * Batch multiplication for GF(p^5)
 */
__global__ void ext5_batch_mul_kernel(
    const GoldilocksExt5* __restrict__ a,
    const GoldilocksExt5* __restrict__ b,
    GoldilocksExt5* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext5_mul(a[idx], b[idx]);
    }
}

/**
 * Batch squaring for GF(p^5)
 */
__global__ void ext5_batch_square_kernel(
    const GoldilocksExt5* __restrict__ a,
    GoldilocksExt5* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext5_square(a[idx]);
    }
}

/**
 * Batch inversion for GF(p^5)
 */
__global__ void ext5_batch_inverse_kernel(
    const GoldilocksExt5* __restrict__ a,
    GoldilocksExt5* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext5_inverse(a[idx]);
    }
}

/**
 * Batch Frobenius for GF(p^5)
 */
__global__ void ext5_batch_frobenius_kernel(
    const GoldilocksExt5* __restrict__ a,
    GoldilocksExt5* __restrict__ result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = ext5_frobenius(a[idx]);
    }
}

/**
 * Batch conversion: Goldilocks -> GF(p^5)
 * Maps a -> (a, 0, 0, 0, 0)
 */
__global__ void gl_to_ext5_batch_kernel(
    const GoldilocksField* __restrict__ input,
    GoldilocksExt5* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gl_to_ext5(input[idx]);
    }
}

/**
 * Batch conversion: GF(p^5) -> Goldilocks (extracts c[0])
 */
__global__ void ext5_to_gl_batch_kernel(
    const GoldilocksExt5* __restrict__ input,
    GoldilocksField* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = ext5_to_gl(input[idx]);
    }
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

// Quadratic extension wrappers
inline cudaError_t ext2_batch_add(
    const GoldilocksExt2* d_a,
    const GoldilocksExt2* d_b,
    GoldilocksExt2* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_add_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t ext2_batch_mul(
    const GoldilocksExt2* d_a,
    const GoldilocksExt2* d_b,
    GoldilocksExt2* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_mul_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t ext2_batch_square(
    const GoldilocksExt2* d_a,
    GoldilocksExt2* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_square_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t ext2_batch_inverse(
    const GoldilocksExt2* d_a,
    GoldilocksExt2* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_batch_inverse_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

// Batch conversion: Goldilocks -> Ext2
inline cudaError_t gl_to_ext2_batch(
    const GoldilocksField* d_input,
    GoldilocksExt2* d_output,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_to_ext2_batch_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    return cudaGetLastError();
}

// Batch conversion: Ext2 -> Goldilocks
inline cudaError_t ext2_to_gl_batch(
    const GoldilocksExt2* d_input,
    GoldilocksField* d_output,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext2_to_gl_batch_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    return cudaGetLastError();
}

// Quintic extension wrappers
inline cudaError_t ext5_batch_add(
    const GoldilocksExt5* d_a,
    const GoldilocksExt5* d_b,
    GoldilocksExt5* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_add_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t ext5_batch_mul(
    const GoldilocksExt5* d_a,
    const GoldilocksExt5* d_b,
    GoldilocksExt5* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_mul_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

inline cudaError_t ext5_batch_inverse(
    const GoldilocksExt5* d_a,
    GoldilocksExt5* d_result,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_batch_inverse_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

// Batch conversion: Goldilocks -> Ext5
inline cudaError_t gl_to_ext5_batch(
    const GoldilocksField* d_input,
    GoldilocksExt5* d_output,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gl_to_ext5_batch_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    return cudaGetLastError();
}

// Batch conversion: Ext5 -> Goldilocks
inline cudaError_t ext5_to_gl_batch(
    const GoldilocksExt5* d_input,
    GoldilocksField* d_output,
    int n,
    cudaStream_t stream = 0
) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ext5_to_gl_batch_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    return cudaGetLastError();
}

// ============================================================================
// Test Code
// ============================================================================

#ifdef EXTENSION_TEST

#include <iostream>
#include <vector>
#include <random>

void test_ext2_basic() {
    std::cout << "Testing GF(p^2) basic operations..." << std::endl;

    // Initialize
    cudaError_t err = goldilocks_init();
    if (err != cudaSuccess) {
        std::cerr << "Failed to init Goldilocks: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Test: (1 + 2X) * (3 + 4X) = 3 + 4X + 6X + 8X^2
    //                          = 3 + 8*7 + (4+6)X = 59 + 10X
    GoldilocksExt2 h_a(1, 2);
    GoldilocksExt2 h_b(3, 4);

    GoldilocksExt2 *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, sizeof(GoldilocksExt2));
    cudaMalloc(&d_b, sizeof(GoldilocksExt2));
    cudaMalloc(&d_result, sizeof(GoldilocksExt2));

    cudaMemcpy(d_a, &h_a, sizeof(GoldilocksExt2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(GoldilocksExt2), cudaMemcpyHostToDevice);

    ext2_batch_mul(d_a, d_b, d_result, 1);
    cudaDeviceSynchronize();

    GoldilocksExt2 h_result;
    cudaMemcpy(&h_result, d_result, sizeof(GoldilocksExt2), cudaMemcpyDeviceToHost);

    uint64_t r0 = canonicalize(h_result.c[0].value);
    uint64_t r1 = canonicalize(h_result.c[1].value);

    std::cout << "(1 + 2X) * (3 + 4X) = " << r0 << " + " << r1 << "X" << std::endl;
    std::cout << "Expected: 59 + 10X" << std::endl;

    if (r0 == 59 && r1 == 10) {
        std::cout << "GF(p^2) multiplication test PASSED!" << std::endl;
    } else {
        std::cout << "GF(p^2) multiplication test FAILED!" << std::endl;
    }

    // Test inversion: a * a^(-1) = 1
    ext2_batch_inverse(d_a, d_result, 1);
    cudaDeviceSynchronize();

    GoldilocksExt2 h_inv;
    cudaMemcpy(&h_inv, d_result, sizeof(GoldilocksExt2), cudaMemcpyDeviceToHost);

    // Multiply a * a^(-1)
    cudaMemcpy(d_b, &h_inv, sizeof(GoldilocksExt2), cudaMemcpyHostToDevice);
    ext2_batch_mul(d_a, d_b, d_result, 1);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(GoldilocksExt2), cudaMemcpyDeviceToHost);

    r0 = canonicalize(h_result.c[0].value);
    r1 = canonicalize(h_result.c[1].value);

    std::cout << "(1 + 2X) * (1 + 2X)^(-1) = " << r0 << " + " << r1 << "X" << std::endl;

    if (r0 == 1 && r1 == 0) {
        std::cout << "GF(p^2) inversion test PASSED!" << std::endl;
    } else {
        std::cout << "GF(p^2) inversion test FAILED!" << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

void test_ext5_basic() {
    std::cout << "\nTesting GF(p^5) basic operations..." << std::endl;

    // Test: simple multiplication
    GoldilocksExt5 h_a(1, 1, 0, 0, 0);  // 1 + X
    GoldilocksExt5 h_b(1, 1, 0, 0, 0);  // 1 + X

    // (1 + X)^2 = 1 + 2X + X^2
    GoldilocksExt5 *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, sizeof(GoldilocksExt5));
    cudaMalloc(&d_b, sizeof(GoldilocksExt5));
    cudaMalloc(&d_result, sizeof(GoldilocksExt5));

    cudaMemcpy(d_a, &h_a, sizeof(GoldilocksExt5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(GoldilocksExt5), cudaMemcpyHostToDevice);

    ext5_batch_mul(d_a, d_b, d_result, 1);
    cudaDeviceSynchronize();

    GoldilocksExt5 h_result;
    cudaMemcpy(&h_result, d_result, sizeof(GoldilocksExt5), cudaMemcpyDeviceToHost);

    std::cout << "(1 + X)^2 = ";
    for (int i = 0; i < 5; i++) {
        uint64_t ci = canonicalize(h_result.c[i].value);
        if (i > 0) std::cout << " + ";
        std::cout << ci;
        if (i > 0) std::cout << "X^" << i;
    }
    std::cout << std::endl;
    std::cout << "Expected: 1 + 2X + X^2" << std::endl;

    uint64_t c0 = canonicalize(h_result.c[0].value);
    uint64_t c1 = canonicalize(h_result.c[1].value);
    uint64_t c2 = canonicalize(h_result.c[2].value);

    if (c0 == 1 && c1 == 2 && c2 == 1) {
        std::cout << "GF(p^5) multiplication test PASSED!" << std::endl;
    } else {
        std::cout << "GF(p^5) multiplication test FAILED!" << std::endl;
    }

    // Test inversion
    h_a = GoldilocksExt5(2, 1, 0, 0, 0);  // 2 + X
    cudaMemcpy(d_a, &h_a, sizeof(GoldilocksExt5), cudaMemcpyHostToDevice);

    ext5_batch_inverse(d_a, d_result, 1);
    cudaDeviceSynchronize();

    GoldilocksExt5 h_inv;
    cudaMemcpy(&h_inv, d_result, sizeof(GoldilocksExt5), cudaMemcpyDeviceToHost);

    // Multiply a * a^(-1)
    cudaMemcpy(d_b, &h_inv, sizeof(GoldilocksExt5), cudaMemcpyHostToDevice);
    ext5_batch_mul(d_a, d_b, d_result, 1);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(GoldilocksExt5), cudaMemcpyDeviceToHost);

    std::cout << "(2 + X) * (2 + X)^(-1) = ";
    bool is_one = true;
    for (int i = 0; i < 5; i++) {
        uint64_t ci = canonicalize(h_result.c[i].value);
        if (i > 0) std::cout << " + ";
        std::cout << ci;
        if (i > 0) std::cout << "X^" << i;
        if (i == 0 && ci != 1) is_one = false;
        if (i > 0 && ci != 0) is_one = false;
    }
    std::cout << std::endl;

    if (is_one) {
        std::cout << "GF(p^5) inversion test PASSED!" << std::endl;
    } else {
        std::cout << "GF(p^5) inversion test FAILED!" << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

void test_ext2_performance() {
    std::cout << "\nTesting GF(p^2) batch performance..." << std::endl;

    const int BATCH_SIZE = 1024 * 1024;

    GoldilocksExt2 *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, BATCH_SIZE * sizeof(GoldilocksExt2));
    cudaMalloc(&d_b, BATCH_SIZE * sizeof(GoldilocksExt2));
    cudaMalloc(&d_result, BATCH_SIZE * sizeof(GoldilocksExt2));

    // Initialize with simple values
    cudaMemset(d_a, 1, BATCH_SIZE * sizeof(GoldilocksExt2));
    cudaMemset(d_b, 2, BATCH_SIZE * sizeof(GoldilocksExt2));

    // Warm up
    ext2_batch_mul(d_a, d_b, d_result, BATCH_SIZE);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    ext2_batch_mul(d_a, d_b, d_result, BATCH_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "GF(p^2) batch multiplication:" << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Time: " << ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (BATCH_SIZE / ms / 1000.0) << " M ops/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

void test_ext5_performance() {
    std::cout << "\nTesting GF(p^5) batch performance..." << std::endl;

    const int BATCH_SIZE = 1024 * 1024;

    GoldilocksExt5 *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, BATCH_SIZE * sizeof(GoldilocksExt5));
    cudaMalloc(&d_b, BATCH_SIZE * sizeof(GoldilocksExt5));
    cudaMalloc(&d_result, BATCH_SIZE * sizeof(GoldilocksExt5));

    cudaMemset(d_a, 1, BATCH_SIZE * sizeof(GoldilocksExt5));
    cudaMemset(d_b, 2, BATCH_SIZE * sizeof(GoldilocksExt5));

    // Warm up
    ext5_batch_mul(d_a, d_b, d_result, BATCH_SIZE);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    ext5_batch_mul(d_a, d_b, d_result, BATCH_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "GF(p^5) batch multiplication:" << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Time: " << ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (BATCH_SIZE / ms / 1000.0) << " M ops/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

int main() {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    test_ext2_basic();
    test_ext5_basic();
    test_ext2_performance();
    test_ext5_performance();

    return 0;
}

#endif // EXTENSION_TEST

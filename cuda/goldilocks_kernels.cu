/**
 * Goldilocks Field CUDA Kernels
 *
 * This file contains CUDA kernels for batch operations on Goldilocks field elements.
 * Include this file in your CUDA project along with goldilocks.cuh.
 */

#include "goldilocks.cuh"
#include <stdio.h>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// Batch Arithmetic Kernels
// ============================================================================

/**
 * Batch addition: result[i] = a[i] + b[i]
 */
__global__ void gl_batch_add_kernel(
    const GoldilocksField* __restrict__ a,
    const GoldilocksField* __restrict__ b,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_add(a[idx], b[idx]);
    }
}

/**
 * Batch subtraction: result[i] = a[i] - b[i]
 */
__global__ void gl_batch_sub_kernel(
    const GoldilocksField* __restrict__ a,
    const GoldilocksField* __restrict__ b,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_sub(a[idx], b[idx]);
    }
}

/**
 * Batch multiplication: result[i] = a[i] * b[i]
 */
__global__ void gl_batch_mul_kernel(
    const GoldilocksField* __restrict__ a,
    const GoldilocksField* __restrict__ b,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_mul(a[idx], b[idx]);
    }
}

/**
 * Batch squaring: result[i] = a[i]^2
 */
__global__ void gl_batch_square_kernel(
    const GoldilocksField* __restrict__ a,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_square(a[idx]);
    }
}

/**
 * Batch negation: result[i] = -a[i]
 */
__global__ void gl_batch_neg_kernel(
    const GoldilocksField* __restrict__ a,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_neg(a[idx]);
    }
}

/**
 * Batch inverse: result[i] = 1/a[i]
 */
__global__ void gl_batch_inverse_kernel(
    const GoldilocksField* __restrict__ a,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_inverse(a[idx]);
    }
}

/**
 * Scalar multiplication: result[i] = scalar * a[i]
 */
__global__ void gl_scalar_mul_kernel(
    const GoldilocksField* __restrict__ a,
    GoldilocksField scalar,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_mul(scalar, a[idx]);
    }
}

/**
 * Batch exponentiation: result[i] = base[i]^exp
 */
__global__ void gl_batch_exp_kernel(
    const GoldilocksField* __restrict__ base,
    uint64_t exp,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_exp(base[idx], exp);
    }
}

/**
 * Fused multiply-add: result[i] = a[i] * b[i] + c[i]
 */
__global__ void gl_batch_fma_kernel(
    const GoldilocksField* __restrict__ a,
    const GoldilocksField* __restrict__ b,
    const GoldilocksField* __restrict__ c,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_add(gl_mul(a[idx], b[idx]), c[idx]);
    }
}

/**
 * Batch canonicalization: result[i] = canonical form of a[i]
 */
__global__ void gl_batch_canonicalize_kernel(
    const GoldilocksField* __restrict__ a,
    GoldilocksField* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = GoldilocksField(canonicalize(a[idx].value));
    }
}

// ============================================================================
// Reduction Kernels (for sum, dot product)
// ============================================================================

/**
 * Block-level sum reduction using shared memory
 */
__device__ __forceinline__
GoldilocksField block_reduce_sum(GoldilocksField val) {
    __shared__ uint64_t shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    shared[tid] = val.value;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = gl_add(GoldilocksField(shared[tid]), GoldilocksField(shared[tid + s])).value;
        }
        __syncthreads();
    }

    return GoldilocksField(shared[0]);
}

/**
 * Sum reduction kernel - first pass
 * Each block reduces a portion of the array
 */
__global__ void gl_sum_reduce_kernel(
    const GoldilocksField* __restrict__ input,
    GoldilocksField* __restrict__ output,
    size_t n
) {
    __shared__ uint64_t shared[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and accumulate multiple elements per thread for coalescing
    GoldilocksField sum(0);
    size_t grid_size = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += grid_size) {
        sum = gl_add(sum, input[i]);
    }

    shared[tid] = sum.value;
    __syncthreads();

    // Tree reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = gl_add(GoldilocksField(shared[tid]), GoldilocksField(shared[tid + s])).value;
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = GoldilocksField(shared[0]);
    }
}

/**
 * Dot product kernel - first pass
 * Computes partial dot products per block
 */
__global__ void gl_dot_product_kernel(
    const GoldilocksField* __restrict__ a,
    const GoldilocksField* __restrict__ b,
    GoldilocksField* __restrict__ output,
    size_t n
) {
    __shared__ uint64_t shared[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    GoldilocksField sum(0);
    size_t grid_size = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += grid_size) {
        sum = gl_add(sum, gl_mul(a[i], b[i]));
    }

    shared[tid] = sum.value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = gl_add(GoldilocksField(shared[tid]), GoldilocksField(shared[tid + s])).value;
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = GoldilocksField(shared[0]);
    }
}

// ============================================================================
// Polynomial Operations
// ============================================================================

/**
 * Polynomial evaluation using Horner's method
 * Evaluates poly[0] + poly[1]*x + poly[2]*x^2 + ... + poly[n-1]*x^(n-1)
 *
 * Each thread evaluates the polynomial at a different point.
 */
__global__ void gl_poly_eval_kernel(
    const GoldilocksField* __restrict__ coeffs,
    const GoldilocksField* __restrict__ points,
    GoldilocksField* __restrict__ results,
    int n_coeffs,
    int n_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    GoldilocksField x = points[idx];
    GoldilocksField result(0);

    // Horner's method: start from highest degree
    for (int i = n_coeffs - 1; i >= 0; i--) {
        result = gl_add(gl_mul(result, x), coeffs[i]);
    }

    results[idx] = result;
}

/**
 * Polynomial multiplication (coefficient-wise contribution)
 * This is a naive O(n^2) algorithm - for large polynomials, use FFT
 */
__global__ void gl_poly_mul_naive_kernel(
    const GoldilocksField* __restrict__ a,
    const GoldilocksField* __restrict__ b,
    GoldilocksField* __restrict__ result,
    int n_a,
    int n_b
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int result_len = n_a + n_b - 1;

    if (idx >= result_len) return;

    GoldilocksField sum(0);

    // Compute coefficient at index idx
    int start_i = max(0, idx - n_b + 1);
    int end_i = min(idx, n_a - 1);

    for (int i = start_i; i <= end_i; i++) {
        sum = gl_add(sum, gl_mul(a[i], b[idx - i]));
    }

    result[idx] = sum;
}

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * Matrix-vector multiplication: result = A * v
 * A is m x n, v is n x 1, result is m x 1
 */
__global__ void gl_matrix_vec_mul_kernel(
    const GoldilocksField* __restrict__ A,
    const GoldilocksField* __restrict__ v,
    GoldilocksField* __restrict__ result,
    int m,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    GoldilocksField sum(0);
    for (int col = 0; col < n; col++) {
        sum = gl_add(sum, gl_mul(A[row * n + col], v[col]));
    }

    result[row] = sum;
}

/**
 * Matrix-matrix multiplication: C = A * B
 * A is m x k, B is k x n, C is m x n
 *
 * Simple implementation - for large matrices, use tiled/shared memory version
 */
__global__ void gl_matrix_mul_kernel(
    const GoldilocksField* __restrict__ A,
    const GoldilocksField* __restrict__ B,
    GoldilocksField* __restrict__ C,
    int m,
    int k,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    GoldilocksField sum(0);
    for (int i = 0; i < k; i++) {
        sum = gl_add(sum, gl_mul(A[row * k + i], B[i * n + col]));
    }

    C[row * n + col] = sum;
}

// ============================================================================
// Montgomery Batch Inverse (more efficient for multiple inverses)
// ============================================================================

/**
 * Montgomery batch inversion
 * Computes inverses of n elements using only 1 field inversion and 3n multiplications
 *
 * Algorithm:
 * 1. Compute cumulative products: prod[i] = a[0] * a[1] * ... * a[i]
 * 2. Compute inverse of final product: inv_all = 1 / prod[n-1]
 * 3. Backtrack to get individual inverses:
 *    inv[i] = inv_all * prod[i-1]
 *    inv_all = inv_all * a[i]
 */
__global__ void gl_montgomery_batch_inverse_prefix_kernel(
    const GoldilocksField* __restrict__ input,
    GoldilocksField* __restrict__ products,
    size_t n
) {
    // This kernel computes prefix products within each block
    __shared__ uint64_t shared[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element
    GoldilocksField val = (idx < n) ? input[idx] : GoldilocksField(1);
    shared[tid] = val.value;
    __syncthreads();

    // Inclusive scan (prefix product)
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        uint64_t temp = (tid >= offset) ? gl_mul(GoldilocksField(shared[tid - offset]), GoldilocksField(shared[tid])).value : shared[tid];
        __syncthreads();
        shared[tid] = temp;
        __syncthreads();
    }

    if (idx < n) {
        products[idx] = GoldilocksField(shared[tid]);
    }
}

// ============================================================================
// Extension Field Batch Kernels
// ============================================================================

/**
 * Batch quadratic extension multiplication
 */
__global__ void gl_ext_batch_mul_kernel(
    const GoldilocksExtQuad* __restrict__ a,
    const GoldilocksExtQuad* __restrict__ b,
    GoldilocksExtQuad* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_ext_mul(a[idx], b[idx]);
    }
}

/**
 * Batch quadratic extension addition
 */
__global__ void gl_ext_batch_add_kernel(
    const GoldilocksExtQuad* __restrict__ a,
    const GoldilocksExtQuad* __restrict__ b,
    GoldilocksExtQuad* __restrict__ result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = gl_ext_add(a[idx], b[idx]);
    }
}

// ============================================================================
// Host-side Wrapper Functions
// ============================================================================

/**
 * Calculate optimal grid dimensions
 */
inline void get_grid_dims(size_t n, int& grid_size, int& block_size) {
    block_size = BLOCK_SIZE;
    grid_size = (n + block_size - 1) / block_size;
}

/**
 * Host wrapper for batch addition
 */
inline cudaError_t gl_batch_add(
    const GoldilocksField* d_a,
    const GoldilocksField* d_b,
    GoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid_size, block_size;
    get_grid_dims(n, grid_size, block_size);
    gl_batch_add_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

/**
 * Host wrapper for batch subtraction
 */
inline cudaError_t gl_batch_sub(
    const GoldilocksField* d_a,
    const GoldilocksField* d_b,
    GoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid_size, block_size;
    get_grid_dims(n, grid_size, block_size);
    gl_batch_sub_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

/**
 * Host wrapper for batch multiplication
 */
inline cudaError_t gl_batch_mul(
    const GoldilocksField* d_a,
    const GoldilocksField* d_b,
    GoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid_size, block_size;
    get_grid_dims(n, grid_size, block_size);
    gl_batch_mul_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_result, n);
    return cudaGetLastError();
}

/**
 * Host wrapper for batch squaring
 */
inline cudaError_t gl_batch_square(
    const GoldilocksField* d_a,
    GoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid_size, block_size;
    get_grid_dims(n, grid_size, block_size);
    gl_batch_square_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

/**
 * Host wrapper for batch inverse
 */
inline cudaError_t gl_batch_inverse(
    const GoldilocksField* d_a,
    GoldilocksField* d_result,
    size_t n,
    cudaStream_t stream = 0
) {
    int grid_size, block_size;
    get_grid_dims(n, grid_size, block_size);
    gl_batch_inverse_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_result, n);
    return cudaGetLastError();
}

// ============================================================================
// Test/Example Code
// ============================================================================

#ifdef GOLDILOCKS_TEST

#include <iostream>
#include <vector>
#include <random>

void test_basic_operations() {
    std::cout << "Testing Goldilocks CUDA kernels..." << std::endl;

    // Initialize constant memory
    cudaError_t err = goldilocks_init();
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    const size_t N = 1024 * 1024;  // 1M elements

    // Allocate host memory
    std::vector<GoldilocksField> h_a(N), h_b(N), h_result(N);

    // Generate random field elements
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < N; i++) {
        h_a[i] = GoldilocksField(rng() % GOLDILOCKS_PRIME);
        h_b[i] = GoldilocksField(rng() % GOLDILOCKS_PRIME);
    }

    // Allocate device memory
    GoldilocksField *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, N * sizeof(GoldilocksField));
    cudaMalloc(&d_b, N * sizeof(GoldilocksField));
    cudaMalloc(&d_result, N * sizeof(GoldilocksField));

    // Copy to device
    cudaMemcpy(d_a, h_a.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test multiplication
    cudaEventRecord(start);
    gl_batch_mul(d_a, d_b, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Batch multiplication of " << N << " elements: " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << (N / ms / 1000.0) << " M elements/ms" << std::endl;

    // Verify a few results
    cudaMemcpy(h_result.data(), d_result, N * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < 10; i++) {
        uint128_t prod = mul_u64_u64(h_a[i].value, h_b[i].value);
        uint64_t expected = reduce128(prod);
        uint64_t got = canonicalize(h_result[i].value);
        expected = canonicalize(expected);

        if (got != expected) {
            std::cout << "Mismatch at " << i << ": expected " << expected << ", got " << got << std::endl;
            correct = false;
        }
    }

    if (correct) {
        std::cout << "Verification passed!" << std::endl;
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_basic_operations();
    return 0;
}

#endif // GOLDILOCKS_TEST

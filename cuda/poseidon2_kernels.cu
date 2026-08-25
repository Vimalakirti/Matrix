/**
 * Poseidon2 CUDA Kernels for Goldilocks Field
 *
 * Batch processing kernels for:
 * - Permutation
 * - Hashing
 * - Merkle tree compression
 */

#include "poseidon2.cuh"
#include <stdio.h>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256

// ============================================================================
// Batch Permutation Kernels
// ============================================================================

/**
 * Batch Poseidon2 permutation (width 8)
 * Each thread processes one state
 */
__global__ void poseidon2_batch_permute_8_kernel(
    const GoldilocksField* __restrict__ states_in,
    GoldilocksField* __restrict__ states_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Load state into registers
    GoldilocksField state[8];
    const GoldilocksField* in_ptr = states_in + idx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = in_ptr[i];
    }

    // Apply permutation
    poseidon2_permute_8(state);

    // Store result
    GoldilocksField* out_ptr = states_out + idx * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out_ptr[i] = state[i];
    }
}

/**
 * Batch Poseidon2 permutation (width 16)
 */
__global__ void poseidon2_batch_permute_16_kernel(
    const GoldilocksField* __restrict__ states_in,
    GoldilocksField* __restrict__ states_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    GoldilocksField state[16];
    const GoldilocksField* in_ptr = states_in + idx * 16;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        state[i] = in_ptr[i];
    }

    poseidon2_permute_16(state);

    GoldilocksField* out_ptr = states_out + idx * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out_ptr[i] = state[i];
    }
}

// ============================================================================
// Batch Hash Kernels
// ============================================================================

/**
 * Batch hash fixed-size inputs (8 elements each) -> 4-element outputs
 */
__global__ void poseidon2_batch_hash_8_to_4_kernel(
    const GoldilocksField* __restrict__ inputs,
    GoldilocksField* __restrict__ outputs,
    int input_size,  // Elements per input (must be multiple of 4)
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const GoldilocksField* in_ptr = inputs + idx * input_size;
    GoldilocksField* out_ptr = outputs + idx * 4;

    poseidon2_hash_8_4(in_ptr, input_size, out_ptr);
}

/**
 * Batch hash using width 16 (16 elements input) -> 8-element outputs
 */
__global__ void poseidon2_batch_hash_16_to_8_kernel(
    const GoldilocksField* __restrict__ inputs,
    GoldilocksField* __restrict__ outputs,
    int input_size,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const GoldilocksField* in_ptr = inputs + idx * input_size;
    GoldilocksField* out_ptr = outputs + idx * 8;

    poseidon2_hash_16_8(in_ptr, input_size, out_ptr);
}

// ============================================================================
// Merkle Tree Kernels
// ============================================================================

/**
 * Batch 2-to-1 compression for Merkle tree layer
 *
 * Compresses pairs of 4-element chunks into single 4-element outputs
 * Used for building one layer of a Merkle tree
 */
__global__ void poseidon2_merkle_layer_8_kernel(
    const GoldilocksField* __restrict__ leaves,  // n * 4 elements
    GoldilocksField* __restrict__ parents,       // (n/2) * 4 elements
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    const GoldilocksField* left = leaves + idx * 8;      // First child
    const GoldilocksField* right = leaves + idx * 8 + 4; // Second child
    GoldilocksField* output = parents + idx * 4;

    poseidon2_compress_8(left, right, output);
}

/**
 * Batch 2-to-1 compression using width-16 permutation
 * Compresses pairs of 8-element chunks
 */
__global__ void poseidon2_merkle_layer_16_kernel(
    const GoldilocksField* __restrict__ leaves,
    GoldilocksField* __restrict__ parents,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    const GoldilocksField* left = leaves + idx * 16;
    const GoldilocksField* right = leaves + idx * 16 + 8;
    GoldilocksField* output = parents + idx * 8;

    poseidon2_compress_16(left, right, output);
}

/**
 * Build entire Merkle tree in one kernel launch
 * Leaves are in the first n*chunk_size positions
 * Parents are computed iteratively
 *
 * This kernel handles one layer at a time with synchronization
 */
__global__ void poseidon2_merkle_tree_8_kernel(
    GoldilocksField* tree,       // Full tree storage: leaves + all parent levels
    int num_leaves,              // Number of leaf nodes
    int current_layer_start,     // Starting index of current layer in tree
    int current_layer_size       // Number of nodes in current layer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_layer_size / 2) return;

    int parent_layer_start = current_layer_start + current_layer_size * 4;

    const GoldilocksField* left = tree + current_layer_start + idx * 8;
    const GoldilocksField* right = tree + current_layer_start + idx * 8 + 4;
    GoldilocksField* output = tree + parent_layer_start + idx * 4;

    poseidon2_compress_8(left, right, output);
}

// ============================================================================
// Extension Field Hashing Kernels
// ============================================================================

/**
 * Batch hash GF(p²) elements
 * Each thread hashes one extension element -> 4 base field output
 */
__global__ void poseidon2_batch_hash_ext2_kernel(
    const GoldilocksExt2* __restrict__ inputs,
    GoldilocksField* __restrict__ outputs,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    poseidon2_hash_ext2(inputs[idx], outputs + idx * 4);
}

/**
 * Batch hash arrays of GF(p²) elements
 * Each thread hashes input_len extension elements -> 4 base field output
 */
__global__ void poseidon2_batch_hash_ext2_array_kernel(
    const GoldilocksExt2* __restrict__ inputs,
    GoldilocksField* __restrict__ outputs,
    int input_len,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    poseidon2_hash_ext2_array(inputs + idx * input_len, input_len, outputs + idx * 4);
}

/**
 * Batch hash GF(p⁵) elements
 */
__global__ void poseidon2_batch_hash_ext5_kernel(
    const GoldilocksExt5* __restrict__ inputs,
    GoldilocksField* __restrict__ outputs,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    poseidon2_hash_ext5(inputs[idx], outputs + idx * 4);
}

/**
 * Batch 2-to-1 compression for Merkle tree over GF(p²) elements
 */
__global__ void poseidon2_merkle_layer_ext2_kernel(
    const GoldilocksExt2* __restrict__ leaves,
    GoldilocksExt2* __restrict__ parents,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    poseidon2_compress_ext2(leaves[idx * 2], leaves[idx * 2 + 1], &parents[idx]);
}

/**
 * Batch 2-to-1 compression for Merkle tree over GF(p⁵) elements
 */
__global__ void poseidon2_merkle_layer_ext5_kernel(
    const GoldilocksExt5* __restrict__ leaves,
    GoldilocksExt5* __restrict__ parents,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    poseidon2_compress_ext5(leaves[idx * 2], leaves[idx * 2 + 1], &parents[idx]);
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

inline cudaError_t poseidon2_batch_permute_8(
    const GoldilocksField* d_states_in,
    GoldilocksField* d_states_out,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_batch_permute_8_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_states_in, d_states_out, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t poseidon2_batch_permute_16(
    const GoldilocksField* d_states_in,
    GoldilocksField* d_states_out,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_batch_permute_16_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_states_in, d_states_out, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t poseidon2_batch_hash_8(
    const GoldilocksField* d_inputs,
    GoldilocksField* d_outputs,
    int input_size,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_batch_hash_8_to_4_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_inputs, d_outputs, input_size, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t poseidon2_merkle_layer_8(
    const GoldilocksField* d_leaves,
    GoldilocksField* d_parents,
    int num_pairs,
    cudaStream_t stream = 0
) {
    int grid_size = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_merkle_layer_8_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_leaves, d_parents, num_pairs
    );
    return cudaGetLastError();
}

/**
 * Build a complete Merkle tree on GPU
 *
 * @param d_tree Pre-allocated device memory for entire tree
 *               Size: (2 * num_leaves - 1) * chunk_size elements
 *               Leaves should already be copied to first num_leaves * chunk_size positions
 * @param num_leaves Number of leaf nodes (must be power of 2)
 * @param chunk_size Size of each node (4 for width-8)
 */
inline cudaError_t poseidon2_build_merkle_tree_8(
    GoldilocksField* d_tree,
    int num_leaves,
    cudaStream_t stream = 0
) {
    const int chunk_size = 4;
    int current_layer_start = 0;
    int current_layer_size = num_leaves;

    while (current_layer_size > 1) {
        int num_pairs = current_layer_size / 2;
        int grid_size = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

        poseidon2_merkle_tree_8_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            d_tree, num_leaves, current_layer_start, current_layer_size
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;

        // Sync between layers
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return err;

        current_layer_start += current_layer_size * chunk_size;
        current_layer_size = num_pairs;
    }

    return cudaSuccess;
}

// ============================================================================
// Extension Field Hashing Host Wrappers
// ============================================================================

/**
 * Batch hash GF(p²) elements
 * Each element produces 4 base field outputs
 */
inline cudaError_t poseidon2_batch_hash_ext2(
    const GoldilocksExt2* d_inputs,
    GoldilocksField* d_outputs,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_batch_hash_ext2_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_inputs, d_outputs, batch_size
    );
    return cudaGetLastError();
}

/**
 * Batch hash arrays of GF(p²) elements
 */
inline cudaError_t poseidon2_batch_hash_ext2_array(
    const GoldilocksExt2* d_inputs,
    GoldilocksField* d_outputs,
    int input_len,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_batch_hash_ext2_array_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_inputs, d_outputs, input_len, batch_size
    );
    return cudaGetLastError();
}

/**
 * Batch hash GF(p⁵) elements
 */
inline cudaError_t poseidon2_batch_hash_ext5(
    const GoldilocksExt5* d_inputs,
    GoldilocksField* d_outputs,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_batch_hash_ext5_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_inputs, d_outputs, batch_size
    );
    return cudaGetLastError();
}

/**
 * Build Merkle tree layer over GF(p²) elements
 */
inline cudaError_t poseidon2_merkle_layer_ext2(
    const GoldilocksExt2* d_leaves,
    GoldilocksExt2* d_parents,
    int num_pairs,
    cudaStream_t stream = 0
) {
    int grid_size = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_merkle_layer_ext2_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_leaves, d_parents, num_pairs
    );
    return cudaGetLastError();
}

/**
 * Build Merkle tree layer over GF(p⁵) elements
 */
inline cudaError_t poseidon2_merkle_layer_ext5(
    const GoldilocksExt5* d_leaves,
    GoldilocksExt5* d_parents,
    int num_pairs,
    cudaStream_t stream = 0
) {
    int grid_size = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poseidon2_merkle_layer_ext5_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_leaves, d_parents, num_pairs
    );
    return cudaGetLastError();
}

/**
 * Build complete Merkle tree over GF(p²) elements
 */
inline cudaError_t poseidon2_build_merkle_tree_ext2(
    GoldilocksExt2* d_tree,
    int num_leaves,
    cudaStream_t stream = 0
) {
    int current_layer_start = 0;
    int current_layer_size = num_leaves;

    while (current_layer_size > 1) {
        int num_pairs = current_layer_size / 2;

        poseidon2_merkle_layer_ext2(
            d_tree + current_layer_start,
            d_tree + current_layer_start + current_layer_size,
            num_pairs,
            stream
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return err;

        current_layer_start += current_layer_size;
        current_layer_size = num_pairs;
    }

    return cudaSuccess;
}

// ============================================================================
// Test Code
// ============================================================================

#ifdef POSEIDON2_TEST

#include <iostream>
#include <vector>
#include <cstring>

// Test vectors from Plonky3 (width 8)
void test_poseidon2_permute() {
    std::cout << "Testing Poseidon2 permutation (width 8)..." << std::endl;

    // Initialize
    cudaError_t err = goldilocks_init();
    if (err != cudaSuccess) {
        std::cerr << "Failed to init Goldilocks: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = poseidon2_init();
    if (err != cudaSuccess) {
        std::cerr << "Failed to init Poseidon2: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Test input: [0, 1, 2, 3, 4, 5, 6, 7]
    std::vector<GoldilocksField> h_input(8);
    for (int i = 0; i < 8; i++) {
        h_input[i] = GoldilocksField(i);
    }

    // Allocate device memory
    GoldilocksField *d_input, *d_output;
    cudaMalloc(&d_input, 8 * sizeof(GoldilocksField));
    cudaMalloc(&d_output, 8 * sizeof(GoldilocksField));

    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), 8 * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    // Run permutation
    poseidon2_batch_permute_8(d_input, d_output, 1);
    cudaDeviceSynchronize();

    // Copy result back
    std::vector<GoldilocksField> h_output(8);
    cudaMemcpy(h_output.data(), d_output, 8 * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    std::cout << "Input:  [";
    for (int i = 0; i < 8; i++) {
        std::cout << h_input[i].value << (i < 7 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    std::cout << "Output: [";
    for (int i = 0; i < 8; i++) {
        std::cout << canonicalize(h_output[i].value) << (i < 7 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    // Expected output from Plonky3 test (range 0..8):
    // [14266028122062624699, 5353147180106052723, 15203350112844181434,
    //  17630919042639565165, 16601551015858213987, 10184091939013874068,
    //  16774100645754596496, 12047415603622314780]
    uint64_t expected[8] = {
        14266028122062624699ULL, 5353147180106052723ULL, 15203350112844181434ULL,
        17630919042639565165ULL, 16601551015858213987ULL, 10184091939013874068ULL,
        16774100645754596496ULL, 12047415603622314780ULL
    };

    std::cout << "Expected: [";
    for (int i = 0; i < 8; i++) {
        std::cout << expected[i] << (i < 7 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    bool correct = true;
    for (int i = 0; i < 8; i++) {
        if (canonicalize(h_output[i].value) != expected[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << std::endl;
        }
    }

    if (correct) {
        std::cout << "Permutation test PASSED!" << std::endl;
    } else {
        std::cout << "Permutation test FAILED!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

void test_batch_performance() {
    std::cout << "\nTesting batch permutation performance..." << std::endl;

    const int BATCH_SIZE = 1024 * 1024;  // 1M permutations

    // Allocate
    GoldilocksField *d_input, *d_output;
    cudaMalloc(&d_input, BATCH_SIZE * 8 * sizeof(GoldilocksField));
    cudaMalloc(&d_output, BATCH_SIZE * 8 * sizeof(GoldilocksField));

    // Initialize with zeros
    cudaMemset(d_input, 0, BATCH_SIZE * 8 * sizeof(GoldilocksField));

    // Warm up
    poseidon2_batch_permute_8(d_input, d_output, BATCH_SIZE);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    poseidon2_batch_permute_8(d_input, d_output, BATCH_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Batch size: " << BATCH_SIZE << " permutations" << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << (BATCH_SIZE / ms / 1000.0) << " M permutations/s" << std::endl;
    std::cout << "Per permutation: " << (ms * 1000.0 / BATCH_SIZE) << " us" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

void test_merkle_tree() {
    std::cout << "\nTesting Merkle tree construction..." << std::endl;

    const int NUM_LEAVES = 1024;  // Must be power of 2
    const int CHUNK_SIZE = 4;
    const int TREE_SIZE = (2 * NUM_LEAVES - 1) * CHUNK_SIZE;

    // Allocate tree
    GoldilocksField* d_tree;
    cudaMalloc(&d_tree, TREE_SIZE * sizeof(GoldilocksField));

    // Initialize leaves (simple test data)
    std::vector<GoldilocksField> h_leaves(NUM_LEAVES * CHUNK_SIZE);
    for (int i = 0; i < NUM_LEAVES * CHUNK_SIZE; i++) {
        h_leaves[i] = GoldilocksField(i);
    }
    cudaMemcpy(d_tree, h_leaves.data(), NUM_LEAVES * CHUNK_SIZE * sizeof(GoldilocksField),
               cudaMemcpyHostToDevice);

    // Build tree
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    poseidon2_build_merkle_tree_8(d_tree, NUM_LEAVES);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Leaves: " << NUM_LEAVES << std::endl;
    std::cout << "Tree build time: " << ms << " ms" << std::endl;

    // Get root
    std::vector<GoldilocksField> h_root(CHUNK_SIZE);
    int root_offset = (2 * NUM_LEAVES - 2) * CHUNK_SIZE;  // Last chunk is root
    cudaMemcpy(h_root.data(), d_tree + root_offset, CHUNK_SIZE * sizeof(GoldilocksField),
               cudaMemcpyDeviceToHost);

    std::cout << "Root: [";
    for (int i = 0; i < CHUNK_SIZE; i++) {
        std::cout << canonicalize(h_root[i].value) << (i < CHUNK_SIZE - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_tree);
}

void test_ext2_hashing() {
    std::cout << "\nTesting GF(p²) element hashing..." << std::endl;

    // Test single element hashing
    GoldilocksExt2 h_input(123, 456);  // 123 + 456*X

    GoldilocksExt2* d_input;
    GoldilocksField* d_output;
    cudaMalloc(&d_input, sizeof(GoldilocksExt2));
    cudaMalloc(&d_output, 4 * sizeof(GoldilocksField));

    cudaMemcpy(d_input, &h_input, sizeof(GoldilocksExt2), cudaMemcpyHostToDevice);

    poseidon2_batch_hash_ext2(d_input, d_output, 1);
    cudaDeviceSynchronize();

    GoldilocksField h_output[4];
    cudaMemcpy(h_output, d_output, 4 * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    std::cout << "Hash of (123 + 456X): [";
    for (int i = 0; i < 4; i++) {
        std::cout << canonicalize(h_output[i].value) << (i < 3 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    // Batch performance test
    const int BATCH_SIZE = 1024 * 1024;

    GoldilocksExt2* d_batch_input;
    GoldilocksField* d_batch_output;
    cudaMalloc(&d_batch_input, BATCH_SIZE * sizeof(GoldilocksExt2));
    cudaMalloc(&d_batch_output, BATCH_SIZE * 4 * sizeof(GoldilocksField));

    cudaMemset(d_batch_input, 1, BATCH_SIZE * sizeof(GoldilocksExt2));

    // Warm up
    poseidon2_batch_hash_ext2(d_batch_input, d_batch_output, BATCH_SIZE);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    poseidon2_batch_hash_ext2(d_batch_input, d_batch_output, BATCH_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Batch GF(p²) hashing:" << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Time: " << ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (BATCH_SIZE / ms / 1000.0) << " M hashes/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_batch_input);
    cudaFree(d_batch_output);
}

void test_ext2_merkle_tree() {
    std::cout << "\nTesting Merkle tree over GF(p²) elements..." << std::endl;

    const int NUM_LEAVES = 1024;
    const int TREE_SIZE = 2 * NUM_LEAVES - 1;

    GoldilocksExt2* d_tree;
    cudaMalloc(&d_tree, TREE_SIZE * sizeof(GoldilocksExt2));

    // Initialize leaves
    std::vector<GoldilocksExt2> h_leaves(NUM_LEAVES);
    for (int i = 0; i < NUM_LEAVES; i++) {
        h_leaves[i] = GoldilocksExt2(i, i + 1);
    }
    cudaMemcpy(d_tree, h_leaves.data(), NUM_LEAVES * sizeof(GoldilocksExt2),
               cudaMemcpyHostToDevice);

    // Build tree
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    poseidon2_build_merkle_tree_ext2(d_tree, NUM_LEAVES);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Leaves: " << NUM_LEAVES << std::endl;
    std::cout << "Tree build time: " << ms << " ms" << std::endl;

    // Get root (last element in tree)
    GoldilocksExt2 h_root;
    cudaMemcpy(&h_root, d_tree + TREE_SIZE - 1, sizeof(GoldilocksExt2),
               cudaMemcpyDeviceToHost);

    std::cout << "Root: " << canonicalize(h_root.c[0].value)
              << " + " << canonicalize(h_root.c[1].value) << "X" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_tree);
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

    test_poseidon2_permute();
    test_batch_performance();
    test_merkle_tree();
    test_ext2_hashing();
    test_ext2_merkle_tree();

    return 0;
}

#endif // POSEIDON2_TEST

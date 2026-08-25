/**
 * Monolith GPU kernel launches — mirrors poseidon2_kernels.cu structure.
 */

#pragma once

#include "monolith.cuh"

static const int MONO_BLOCK_SIZE = 256;

// ============================================================================
// Merkle Tree Kernels
// ============================================================================

/**
 * Batch 2-to-1 compression for Merkle tree over base-field digests.
 * Each thread compresses one pair: left[4] || right[4] → parent[4].
 */
__global__ void monolith_merkle_tree_8_kernel(
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

/**
 * Batch 2-to-1 compression for Merkle tree over GF(p^2) elements.
 */
__global__ void monolith_merkle_layer_ext2_kernel(
    const GoldilocksExt2* __restrict__ leaves,
    GoldilocksExt2* __restrict__ parents,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    monolith_compress_ext2(leaves[idx * 2], leaves[idx * 2 + 1], &parents[idx]);
}

// ============================================================================
// Leaf Hashing Kernels
// ============================================================================

/**
 * Hash pairs of base-field elements into leaf digests.
 */
__global__ void monolith_hash_gl_leaves_kernel(
    const GoldilocksField* __restrict__ codeword,
    GoldilocksField* __restrict__ tree,
    int num_leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves) return;

    monolith_hash_gl_leaf(
        codeword[idx * 2],
        codeword[idx * 2 + 1],
        tree + idx * 4
    );
}

/**
 * Hash pairs of Ext2 elements into leaf digests.
 * Each pair = 4 base field elements → compress into 1 Ext2 parent.
 */
__global__ void monolith_hash_ext2_leaves_kernel(
    const GoldilocksExt2* __restrict__ codeword,
    GoldilocksExt2* __restrict__ tree,
    int num_leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves) return;

    monolith_compress_ext2(codeword[idx * 2], codeword[idx * 2 + 1], &tree[idx]);
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

inline cudaError_t monolith_merkle_layer_8(
    const GoldilocksField* d_leaves,
    GoldilocksField* d_parents,
    int num_pairs,
    cudaStream_t stream = 0
) {
    int grid_size = (num_pairs + MONO_BLOCK_SIZE - 1) / MONO_BLOCK_SIZE;
    monolith_merkle_tree_8_kernel<<<grid_size, MONO_BLOCK_SIZE, 0, stream>>>(
        (GoldilocksField*)d_leaves, 0, 0, num_pairs * 2
    );
    return cudaGetLastError();
}

inline cudaError_t monolith_merkle_layer_ext2(
    const GoldilocksExt2* d_leaves,
    GoldilocksExt2* d_parents,
    int num_pairs,
    cudaStream_t stream = 0
) {
    int grid_size = (num_pairs + MONO_BLOCK_SIZE - 1) / MONO_BLOCK_SIZE;
    monolith_merkle_layer_ext2_kernel<<<grid_size, MONO_BLOCK_SIZE, 0, stream>>>(
        d_leaves, d_parents, num_pairs
    );
    return cudaGetLastError();
}

/**
 * Build complete Merkle tree from base-field codeword.
 * 1. Hash pairs of codeword elements into leaf digests.
 * 2. Build parent layers bottom-up.
 */
inline cudaError_t monolith_build_merkle_tree_8(
    const GoldilocksField* d_codeword,
    GoldilocksField* d_tree,
    int num_leaves,
    cudaStream_t stream = 0
) {
    // Step 1: Hash leaves
    int grid_size = (num_leaves + MONO_BLOCK_SIZE - 1) / MONO_BLOCK_SIZE;
    monolith_hash_gl_leaves_kernel<<<grid_size, MONO_BLOCK_SIZE, 0, stream>>>(
        d_codeword, d_tree, num_leaves
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    // Step 2: Build parent layers
    const int chunk_size = 4;
    int current_layer_start = 0;
    int current_layer_size = num_leaves;

    while (current_layer_size > 1) {
        int num_pairs = current_layer_size / 2;
        grid_size = (num_pairs + MONO_BLOCK_SIZE - 1) / MONO_BLOCK_SIZE;

        monolith_merkle_tree_8_kernel<<<grid_size, MONO_BLOCK_SIZE, 0, stream>>>(
            d_tree, num_leaves, current_layer_start, current_layer_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return err;

        current_layer_start += current_layer_size * chunk_size;
        current_layer_size = num_pairs;
    }

    return cudaSuccess;
}

/**
 * Build complete Merkle tree over GF(p^2) elements.
 */
inline cudaError_t monolith_build_merkle_tree_ext2(
    GoldilocksExt2* d_tree,
    int num_leaves,
    cudaStream_t stream = 0
) {
    int current_layer_start = 0;
    int current_layer_size = num_leaves;

    while (current_layer_size > 1) {
        int num_pairs = current_layer_size / 2;

        monolith_merkle_layer_ext2(
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

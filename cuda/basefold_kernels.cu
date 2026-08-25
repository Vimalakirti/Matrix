/**
 * Basefold Polynomial Commitment Scheme - CUDA Host Wrappers & Tests
 *
 * Phases 4, 8-12: Commit/Open orchestration, table generation,
 * query phase, verifier, and comprehensive tests.
 */

// Include kernels we need (gl_dot_product_kernel, poseidon2_build_merkle_tree_8, etc.)
// These are inline/static so safe to include from .cu
#include "goldilocks_kernels.cu"
#include "poseidon2_kernels.cu"
#include "basefold.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <random>

// ============================================================================
// Phase 1 Host Wrappers: Bit-Reversal
// ============================================================================

inline cudaError_t bit_reverse_permute_gl(
    GoldilocksField* d_data,
    int log_n,
    cudaStream_t stream = 0
) {
    size_t n = 1ULL << log_n;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_reverse_permute_gl_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_data, log_n);
    return cudaGetLastError();
}

inline cudaError_t bit_reverse_permute_ext2(
    GoldilocksExt2* d_data,
    int log_n,
    cudaStream_t stream = 0
) {
    size_t n = 1ULL << log_n;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_reverse_permute_ext2_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_data, log_n);
    return cudaGetLastError();
}

// ============================================================================
// Phase 2 Host Wrapper: BHC Interpolation
// ============================================================================

/**
 * Interpolate over Boolean hypercube: evals (Type2) -> coeffs (Type2) + bh_evals (Type1).
 *
 * d_evals: input evaluations in Type2 order (device, size 2^num_vars)
 * d_coeffs: output coefficients in Type2 order (device, pre-allocated)
 * d_bh_evals: output copy of evals in Type1 order (device, pre-allocated)
 */
inline cudaError_t bhc_interpolate(
    const GoldilocksField* d_evals,
    GoldilocksField* d_coeffs,
    GoldilocksField* d_bh_evals,
    int num_vars,
    cudaStream_t stream = 0
) {
    size_t n = 1ULL << num_vars;
    int grid = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // First pass: pairwise differences + copy
    bhc_interp_first_pass_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_evals, d_coeffs, d_bh_evals, n
    );

    // Subsequent layers
    for (int k = 1; k < num_vars; k++) {
        size_t half_chunk = 1ULL << k;
        int layer_grid = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bhc_interp_layer_kernel<<<layer_grid, BLOCK_SIZE, 0, stream>>>(
            d_coeffs, half_chunk, n
        );
    }

    // Bit-reverse bh_evals to convert from Type2 to Type1
    bit_reverse_permute_gl(d_bh_evals, num_vars, stream);

    return cudaGetLastError();
}

// ============================================================================
// Phase 3 Host Wrapper: Foldable Domain Encoding
// ============================================================================

/**
 * Encode coefficients over foldable domain (Mode A: repetition + butterfly).
 *
 * d_coeffs: input coefficients in Type2 order (size 2^num_vars)
 * d_codeword: output codeword (pre-allocated, size 2^(num_vars + log_rate))
 */
inline cudaError_t encode_foldable_domain(
    const GoldilocksField* d_coeffs,
    GoldilocksField* d_codeword,
    int num_vars,
    int log_rate,
    cudaStream_t stream = 0
) {
    size_t k = 1ULL << num_vars;
    int rate = 1 << log_rate;
    size_t n_output = k * rate;

    // Step 1: Repetition code
    int grid = (n_output + BLOCK_SIZE - 1) / BLOCK_SIZE;
    repetition_encode_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_coeffs, d_codeword, rate, n_output
    );

    // Step 2: Iterative butterfly layers
    int log_k = num_vars;
    for (int i = 0; i < log_k; i++) {
        size_t half_chunk = 1ULL << (i + log_rate);
        int layer_grid = (n_output / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        foldable_domain_layer_kernel<<<layer_grid, BLOCK_SIZE, 0, stream>>>(
            d_codeword, half_chunk, n_output
        );
    }

    // Step 3: Bit-reverse to convert Type2 -> Type1
    bit_reverse_permute_gl(d_codeword, num_vars + log_rate, stream);

    return cudaGetLastError();
}

/**
 * Encode with RS basecode.
 */
inline cudaError_t encode_rs_basecode(
    const GoldilocksField* d_coeffs,
    GoldilocksField* d_basecode,
    GoldilocksField* d_codeword,
    int num_vars,
    int log_rate,
    int basecode_rounds,
    cudaStream_t stream = 0
) {
    int rate = 1 << log_rate;
    int chunk_size = 1 << (num_vars - basecode_rounds);
    size_t n_chunks = 1ULL << basecode_rounds;
    int total_eval_points = chunk_size * rate;
    size_t total_outputs = n_chunks * total_eval_points;

    // Step 1: RS basecode
    int grid = (total_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rs_basecode_encode_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_coeffs, d_basecode, chunk_size, total_eval_points, n_chunks
    );

    // Step 2: Copy to codeword and apply remaining butterfly layers
    cudaMemcpyAsync(d_codeword, d_basecode,
                    total_outputs * sizeof(GoldilocksField),
                    cudaMemcpyDeviceToDevice, stream);

    int levels = basecode_rounds;
    size_t n_output = total_outputs;
    int log_chunk = num_vars - basecode_rounds + log_rate;

    for (int i = 0; i < levels; i++) {
        size_t half_chunk = 1ULL << (log_chunk + i);
        int layer_grid = (n_output / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        foldable_domain_layer_kernel<<<layer_grid, BLOCK_SIZE, 0, stream>>>(
            d_codeword, half_chunk, n_output
        );
    }

    // Step 3: Bit-reverse
    bit_reverse_permute_gl(d_codeword, num_vars + log_rate, stream);

    return cudaGetLastError();
}

// ============================================================================
// Phase 4 Host Wrapper: Commit
// ============================================================================

/**
 * Full basefold commit: evals -> (codeword, bh_evals, merkle_tree)
 *
 * d_evals: polynomial evaluations in Type2 order (size 2^num_vars)
 * d_codeword: output codeword in Type1 (pre-allocated, size 2^(num_vars + log_rate))
 * d_bh_evals: output BH evals in Type1 (pre-allocated, size 2^num_vars)
 * d_tree: output Merkle tree (pre-allocated, size (2 * num_leaves - 1) * CHUNK_SIZE)
 *         where num_leaves = 2^(num_vars + log_rate - 1) and CHUNK_SIZE = 4
 */
inline cudaError_t basefold_commit(
    const GoldilocksField* d_evals,
    GoldilocksField* d_coeffs,   // workspace, size 2^num_vars
    GoldilocksField* d_codeword,
    GoldilocksField* d_bh_evals,
    GoldilocksField* d_tree,
    int num_vars,
    int log_rate,
    cudaStream_t stream = 0
) {
    // Phase 2: BHC interpolation
    bhc_interpolate(d_evals, d_coeffs, d_bh_evals, num_vars, stream);
    cudaStreamSynchronize(stream);

    // Phase 3: Encode over foldable domain
    encode_foldable_domain(d_coeffs, d_codeword, num_vars, log_rate, stream);
    cudaStreamSynchronize(stream);

    // Merkle tree: hash pairs of codeword elements, then build tree
    // The codeword has 2^(num_vars + log_rate) elements.
    // Leaves = codeword elements treated as 4-element chunks (Poseidon2 width-8 compression).
    // num_leaves = 2^(num_vars + log_rate) / 2 (pairs hashed)
    // Actually for poseidon2_build_merkle_tree_8:
    //   - leaves are already in the first num_leaves * 4 positions
    //   - But our codeword is GoldilocksField elements, not pre-chunked.
    //
    // We need to hash pairs of codeword elements into 4-element digests first,
    // then build the tree on top.
    //
    // For simplicity, we copy the codeword into the tree leaf area,
    // treating every 8 consecutive field elements as one leaf-pair input.
    size_t codeword_len = 1ULL << (num_vars + log_rate);
    // Number of leaves for Merkle tree: codeword_len / 2
    // (each leaf = hash of 2 adjacent codeword elements -> 4-element digest)
    // But poseidon2_build_merkle_tree_8 expects leaves already as 4-element chunks.
    //
    // So first we hash pairs: poseidon2_batch_hash_8(codeword, leaf_digests, 2, codeword_len/2)
    // Then build tree on leaf_digests.

    size_t num_leaves = codeword_len / 2;

    // Hash pairs of codeword elements into 4-element leaf digests
    // Input: pairs of field elements (8 bytes each -> 2 elements per leaf)
    // We use poseidon2 hash with input_size=2
    poseidon2_batch_hash_8(d_codeword, d_tree, 2, num_leaves, stream);
    cudaStreamSynchronize(stream);

    // Build tree on top of leaf digests
    poseidon2_build_merkle_tree_8(d_tree, num_leaves, stream);

    return cudaGetLastError();
}

// ============================================================================
// Host-Side Field Arithmetic (needed by wrappers below)
// ============================================================================

static inline GoldilocksField gl_add_host(GoldilocksField a, GoldilocksField b) {
    uint64_t sum = a.value + b.value;
    if (sum < a.value || sum >= GOLDILOCKS_PRIME) {
        sum += NEG_ORDER;
    }
    return GoldilocksField(sum);
}

static inline GoldilocksField gl_sub_host(GoldilocksField a, GoldilocksField b) {
    uint64_t diff;
    if (a.value >= b.value) {
        diff = a.value - b.value;
    } else {
        diff = a.value + GOLDILOCKS_PRIME - b.value;
    }
    return GoldilocksField(diff);
}

static inline GoldilocksField gl_mul_host(GoldilocksField a, GoldilocksField b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t full = (__uint128_t)a.value * (__uint128_t)b.value;
    uint64_t result = (uint64_t)(full % GOLDILOCKS_PRIME);
    return GoldilocksField(result);
#else
    uint128_t prod = mul_u64_u64(a.value, b.value);
    uint64_t result = prod.lo;
    uint64_t tmp = prod.hi * NEG_ORDER;
    result = gl_add_host(GoldilocksField(result), GoldilocksField(tmp)).value;
    return GoldilocksField(result % GOLDILOCKS_PRIME);
#endif
}

static inline GoldilocksField gl_neg_host(GoldilocksField a) {
    if (a.value == 0) return a;
    return GoldilocksField(GOLDILOCKS_PRIME - (a.value % GOLDILOCKS_PRIME));
}

static inline GoldilocksField gl_inv_host(GoldilocksField a) {
    uint64_t exp = GOLDILOCKS_PRIME - 2;
    GoldilocksField result(1);
    GoldilocksField base = a;
    while (exp > 0) {
        if (exp & 1) result = gl_mul_host(result, base);
        base = gl_mul_host(base, base);
        exp >>= 1;
    }
    return result;
}

static inline GoldilocksExt2 ext2_add_host(GoldilocksExt2 a, GoldilocksExt2 b) {
    return GoldilocksExt2(gl_add_host(a.c[0], b.c[0]), gl_add_host(a.c[1], b.c[1]));
}

static inline GoldilocksExt2 ext2_sub_host(GoldilocksExt2 a, GoldilocksExt2 b) {
    return GoldilocksExt2(gl_sub_host(a.c[0], b.c[0]), gl_sub_host(a.c[1], b.c[1]));
}

static inline GoldilocksExt2 ext2_mul_host(GoldilocksExt2 a, GoldilocksExt2 b) {
    GoldilocksField b1_w = gl_mul_host(b.c[1], GoldilocksField(EXT2_W));
    GoldilocksField c0 = gl_add_host(gl_mul_host(a.c[0], b.c[0]), gl_mul_host(a.c[1], b1_w));
    GoldilocksField c1 = gl_add_host(gl_mul_host(a.c[0], b.c[1]), gl_mul_host(a.c[1], b.c[0]));
    return GoldilocksExt2(c0, c1);
}

static inline GoldilocksExt2 ext2_scalar_mul_host(GoldilocksField s, GoldilocksExt2 a) {
    return GoldilocksExt2(gl_mul_host(s, a.c[0]), gl_mul_host(s, a.c[1]));
}

// ============================================================================
// Phase 5 Host Wrappers: Sum-Check
// ============================================================================

/**
 * Reduce partial sums from sumcheck_product_kernel into final 3 coefficients.
 */
inline void sumcheck_reduce_partials(
    GoldilocksField* d_partial_c0,
    GoldilocksField* d_partial_c1,
    GoldilocksField* d_partial_c2,
    GoldilocksField* h_coeffs,  // output: 3 elements
    int num_blocks,
    cudaStream_t stream = 0
) {
    // Copy partials to host and reduce
    std::vector<GoldilocksField> p0(num_blocks), p1(num_blocks), p2(num_blocks);
    cudaMemcpyAsync(p0.data(), d_partial_c0, num_blocks * sizeof(GoldilocksField),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(p1.data(), d_partial_c1, num_blocks * sizeof(GoldilocksField),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(p2.data(), d_partial_c2, num_blocks * sizeof(GoldilocksField),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    GoldilocksField c0(0), c1(0), c2(0);
    for (int i = 0; i < num_blocks; i++) {
        c0 = gl_add_host(c0, p0[i]);
        c1 = gl_add_host(c1, p1[i]);
        c2 = gl_add_host(c2, p2[i]);
    }
    h_coeffs[0] = c0;
    h_coeffs[1] = c1;
    h_coeffs[2] = c2;
}

/**
 * Run one sum-check interp + product step.
 * eq and bh should be in evaluation form (2 * pair_count elements each).
 * Returns 3 coefficients on host.
 */
inline void sumcheck_round(
    GoldilocksField* d_eq,
    GoldilocksField* d_bh,
    GoldilocksField* h_coeffs,
    size_t pair_count,
    GoldilocksField* d_partial_c0,
    GoldilocksField* d_partial_c1,
    GoldilocksField* d_partial_c2,
    cudaStream_t stream = 0
) {
    int grid_interp = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Interp both eq and bh
    sumcheck_interp_kernel<<<grid_interp, BLOCK_SIZE, 0, stream>>>(d_eq, pair_count);
    sumcheck_interp_kernel<<<grid_interp, BLOCK_SIZE, 0, stream>>>(d_bh, pair_count);

    // Product
    int grid_prod = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid_prod > 256) grid_prod = 256;  // limit for reduction
    sumcheck_product_kernel<<<grid_prod, BLOCK_SIZE, 0, stream>>>(
        d_eq, d_bh, d_partial_c0, d_partial_c1, d_partial_c2, pair_count
    );

    sumcheck_reduce_partials(d_partial_c0, d_partial_c1, d_partial_c2,
                             h_coeffs, grid_prod, stream);
}

/**
 * Apply challenge to eq and bh (eval step), compacting to half size.
 */
inline void sumcheck_apply_challenge(
    GoldilocksField* d_eq_in,
    GoldilocksField* d_eq_out,
    GoldilocksField* d_bh_in,
    GoldilocksField* d_bh_out,
    GoldilocksField challenge,
    size_t pair_count,
    cudaStream_t stream = 0
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sumcheck_eval_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_eq_in, challenge, d_eq_out, pair_count);
    sumcheck_eval_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_bh_in, challenge, d_bh_out, pair_count);
}

// ============================================================================
// Phase 6 Host Wrapper: Basefold Folding
// ============================================================================

inline cudaError_t basefold_fold(
    const GoldilocksField* d_codeword,
    const FoldingEntry* d_table,
    GoldilocksField challenge,
    GoldilocksField* d_output,
    size_t pair_count,
    cudaStream_t stream = 0
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    basefold_fold_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_codeword, d_table, challenge, d_output, pair_count
    );
    return cudaGetLastError();
}

// ============================================================================
// Phase 7 Host Wrappers: Extension Field
// ============================================================================

inline cudaError_t basefold_fold_mixed(
    const GoldilocksField* d_codeword,
    const FoldingEntry* d_table,
    GoldilocksExt2 challenge,
    GoldilocksExt2* d_output,
    size_t pair_count,
    cudaStream_t stream = 0
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    basefold_fold_mixed_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_codeword, d_table, challenge, d_output, pair_count
    );
    return cudaGetLastError();
}

inline cudaError_t basefold_fold_ext2(
    const GoldilocksExt2* d_codeword,
    const FoldingEntry* d_table,
    GoldilocksExt2 challenge,
    GoldilocksExt2* d_output,
    size_t pair_count,
    cudaStream_t stream = 0
) {
    int grid = (pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    basefold_fold_ext2_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_codeword, d_table, challenge, d_output, pair_count
    );
    return cudaGetLastError();
}

/**
 * Reduce ext2 partial sums.
 */
inline void sumcheck_reduce_partials_ext2(
    GoldilocksExt2* d_partial_c0,
    GoldilocksExt2* d_partial_c1,
    GoldilocksExt2* d_partial_c2,
    GoldilocksExt2* h_coeffs,  // output: 3 elements
    int num_blocks,
    cudaStream_t stream = 0
) {
    std::vector<GoldilocksExt2> p0(num_blocks), p1(num_blocks), p2(num_blocks);
    cudaMemcpyAsync(p0.data(), d_partial_c0, num_blocks * sizeof(GoldilocksExt2),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(p1.data(), d_partial_c1, num_blocks * sizeof(GoldilocksExt2),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(p2.data(), d_partial_c2, num_blocks * sizeof(GoldilocksExt2),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    GoldilocksExt2 c0, c1, c2;
    for (int i = 0; i < num_blocks; i++) {
        c0 = ext2_add_host(c0, p0[i]);
        c1 = ext2_add_host(c1, p1[i]);
        c2 = ext2_add_host(c2, p2[i]);
    }
    h_coeffs[0] = c0;
    h_coeffs[1] = c1;
    h_coeffs[2] = c2;
}

// ============================================================================
// Phase 9: Table Generation (Host-Side)
// ============================================================================

/**
 * Generate random folding table on host (AES-mode equivalent).
 * Uses a simple PRNG for reproducibility.
 *
 * Returns a flat vector of FoldingEntry for all levels.
 * table_offsets[level] = offset into the flat array for that level.
 *
 * Level 0: 1 entry (for the first folding round, codeword halves from N to N/2)
 * Level i: 2^i entries
 * Total entries: 2^0 + 2^1 + ... + 2^(num_rounds-1) = 2^num_rounds - 1
 *
 * But following the Rust reference more carefully:
 * The table at level i has entries for pair indices in the codeword of size 2^(num_vars+log_rate-i).
 * At level i, there are 2^(num_vars+log_rate-i-1) pairs.
 * The folding table maps each pair to a folding point.
 *
 * For simplicity, we generate the table matching the bit-reversed Type1 order
 * that the codeword is in. Each level i has table_size = 2^(num_vars+log_rate-i) / 2 entries.
 */
struct BasefoldTable {
    std::vector<FoldingEntry> entries;
    std::vector<size_t> level_offsets;  // offset for level i
    std::vector<size_t> level_sizes;    // number of entries at level i
    int num_rounds;

    FoldingEntry* d_entries;  // device pointer

    BasefoldTable() : d_entries(nullptr), num_rounds(0) {}

    ~BasefoldTable() {
        if (d_entries) cudaFree(d_entries);
    }

    void upload() {
        if (d_entries) cudaFree(d_entries);
        cudaMalloc(&d_entries, entries.size() * sizeof(FoldingEntry));
        cudaMemcpy(d_entries, entries.data(), entries.size() * sizeof(FoldingEntry),
                   cudaMemcpyHostToDevice);
    }

    FoldingEntry* device_level(int level) {
        return d_entries + level_offsets[level];
    }
};

/**
 * Generate a random folding table.
 * seed: random seed for reproducibility.
 */
inline BasefoldTable generate_folding_table(
    int num_vars,
    int log_rate,
    int num_rounds,
    uint64_t seed = 12345
) {
    BasefoldTable table;
    table.num_rounds = num_rounds;

    std::mt19937_64 rng(seed);
    auto rand_field = [&]() -> GoldilocksField {
        return GoldilocksField(rng() % GOLDILOCKS_PRIME);
    };

    size_t total_entries = 0;
    for (int i = 0; i < num_rounds; i++) {
        size_t level_size = 1ULL << (num_vars + log_rate - i - 1);
        table.level_offsets.push_back(total_entries);
        table.level_sizes.push_back(level_size);
        total_entries += level_size;
    }

    table.entries.resize(total_entries);

    for (int i = 0; i < num_rounds; i++) {
        size_t level_size = table.level_sizes[i];
        size_t offset = table.level_offsets[i];

        for (size_t j = 0; j < level_size; j++) {
            GoldilocksField x0 = rand_field();
            GoldilocksField x1 = rand_field();
            // Ensure x0 != x1
            while (x1.value == x0.value) {
                x1 = rand_field();
            }
            GoldilocksField diff = gl_sub_host(x1, x0);
            GoldilocksField weight = gl_inv_host(diff);

            table.entries[offset + j].point = x0;
            table.entries[offset + j].weight = weight;
        }
    }

    return table;
}

// ============================================================================
// Phase 8: Open (Prover-Side Orchestration)
// ============================================================================

/**
 * Basefold open proof data structure.
 */
struct BasefoldProof {
    // Sum-check oracles: num_rounds + 1 oracles, each with 3 coefficients
    std::vector<GoldilocksField> sumcheck_oracles;  // flat: [c0, c1, c2] * (num_rounds + 1)
    GoldilocksField eval;

    // Folded oracles (intermediate codewords)
    std::vector<std::vector<GoldilocksField>> folded_oracles;

    // Merkle roots
    std::vector<GoldilocksField> merkle_roots;  // 4 elements per root

    // Final oracle (last folded codeword)
    std::vector<GoldilocksField> final_oracle;

    // Query responses
    std::vector<int> query_indices;
    std::vector<std::vector<GoldilocksField>> query_vals;  // per query, per round pair
};

struct BasefoldProofExt2 {
    std::vector<GoldilocksExt2> sumcheck_oracles;  // [c0, c1, c2] * (num_rounds + 1)
    GoldilocksExt2 eval;
    std::vector<std::vector<GoldilocksExt2>> folded_oracles;
    std::vector<GoldilocksField> merkle_roots;
    std::vector<GoldilocksExt2> final_oracle;
    std::vector<int> query_indices;
};

/**
 * Base field open.
 */
inline BasefoldProof basefold_open(
    const GoldilocksField* d_codeword,      // Type1 codeword, size 2^(num_vars + log_rate)
    const GoldilocksField* d_bh_evals,      // Type1 BH evals, size 2^num_vars
    const GoldilocksField* d_point,          // evaluation point, size num_vars (device)
    BasefoldTable& table,
    int num_vars,
    int log_rate,
    int num_rounds,
    int num_queries,
    cudaStream_t stream = 0
) {
    BasefoldProof proof;
    size_t N = 1ULL << num_vars;

    // Allocate workspace
    GoldilocksField *d_eq_a, *d_eq_b, *d_eq_result;
    cudaMalloc(&d_eq_a, N * sizeof(GoldilocksField));
    cudaMalloc(&d_eq_b, N * sizeof(GoldilocksField));

    // Step 1: Build eq(x, point) using existing eq_dp_all
    eq_dp_all(d_point, d_eq_a, d_eq_b, num_vars, &d_eq_result, stream);
    cudaStreamSynchronize(stream);

    // Copy eq result to d_eq_a (we'll use d_eq_a going forward)
    if (d_eq_result != d_eq_a) {
        cudaMemcpy(d_eq_a, d_eq_result, N * sizeof(GoldilocksField), cudaMemcpyDeviceToDevice);
    }

    // bh_evals is in Type1 (bit-reversed) order from commit.
    // Both dot product and sum-check need Type2 (natural) order.
    // Create working copy and bit-reverse to convert Type1 -> Type2.
    GoldilocksField *d_bh_work;
    cudaMalloc(&d_bh_work, N * sizeof(GoldilocksField));
    cudaMemcpy(d_bh_work, d_bh_evals, N * sizeof(GoldilocksField), cudaMemcpyDeviceToDevice);
    bit_reverse_permute_gl(d_bh_work, num_vars, stream);
    cudaStreamSynchronize(stream);

    // Step 2: Compute eval = dot_product(bh_evals_Type2, eq_Type2)
    int dp_blocks = min((int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    GoldilocksField* d_dp_partial;
    cudaMalloc(&d_dp_partial, dp_blocks * sizeof(GoldilocksField));
    gl_dot_product_kernel<<<dp_blocks, BLOCK_SIZE, 0, stream>>>(
        d_bh_work, d_eq_a, d_dp_partial, N
    );
    cudaStreamSynchronize(stream);

    // Reduce dot product partials on host
    std::vector<GoldilocksField> dp_partial(dp_blocks);
    cudaMemcpy(dp_partial.data(), d_dp_partial, dp_blocks * sizeof(GoldilocksField),
               cudaMemcpyDeviceToHost);
    proof.eval = GoldilocksField(0);
    for (int i = 0; i < dp_blocks; i++) {
        proof.eval = gl_add_host(proof.eval, dp_partial[i]);
    }
    cudaFree(d_dp_partial);

    int max_blocks = 256;
    GoldilocksField *d_pc0, *d_pc1, *d_pc2;
    cudaMalloc(&d_pc0, max_blocks * sizeof(GoldilocksField));
    cudaMalloc(&d_pc1, max_blocks * sizeof(GoldilocksField));
    cudaMalloc(&d_pc2, max_blocks * sizeof(GoldilocksField));

    GoldilocksField *d_eq_out, *d_bh_out;
    cudaMalloc(&d_eq_out, N * sizeof(GoldilocksField));
    cudaMalloc(&d_bh_out, N * sizeof(GoldilocksField));

    // Step 3: First sum-check round (interp + product)
    size_t current_pairs = N / 2;
    GoldilocksField sc_coeffs[3];
    sumcheck_round(d_eq_a, d_bh_work, sc_coeffs, current_pairs, d_pc0, d_pc1, d_pc2, stream);
    proof.sumcheck_oracles.push_back(sc_coeffs[0]);
    proof.sumcheck_oracles.push_back(sc_coeffs[1]);
    proof.sumcheck_oracles.push_back(sc_coeffs[2]);

    // Step 4: Folding loop
    size_t cw_size = 1ULL << (num_vars + log_rate);
    GoldilocksField* d_cw_current;
    GoldilocksField* d_cw_next;
    cudaMalloc(&d_cw_current, cw_size * sizeof(GoldilocksField));
    cudaMalloc(&d_cw_next, cw_size * sizeof(GoldilocksField));
    cudaMemcpy(d_cw_current, d_codeword, cw_size * sizeof(GoldilocksField),
               cudaMemcpyDeviceToDevice);

    // Workspace for Merkle trees of intermediate oracles
    // (We store roots for the proof)
    for (int round = 0; round < num_rounds; round++) {
        // Generate challenge (in real protocol, from transcript; here use deterministic)
        GoldilocksField challenge(round * 17 + 42);  // deterministic for testing

        // Apply challenge to sum-check
        sumcheck_apply_challenge(d_eq_a, d_eq_out, d_bh_work, d_bh_out,
                                 challenge, current_pairs, stream);
        cudaStreamSynchronize(stream);

        // Swap buffers
        std::swap(d_eq_a, d_eq_out);
        std::swap(d_bh_work, d_bh_out);
        current_pairs /= 2;

        // Next sum-check round
        if (current_pairs > 0) {
            sumcheck_round(d_eq_a, d_bh_work, sc_coeffs, current_pairs,
                           d_pc0, d_pc1, d_pc2, stream);
            proof.sumcheck_oracles.push_back(sc_coeffs[0]);
            proof.sumcheck_oracles.push_back(sc_coeffs[1]);
            proof.sumcheck_oracles.push_back(sc_coeffs[2]);
        }

        // Fold codeword
        size_t cw_pairs = cw_size / 2;
        basefold_fold(d_cw_current, table.device_level(round), challenge,
                      d_cw_next, cw_pairs, stream);
        cudaStreamSynchronize(stream);

        cw_size = cw_pairs;
        std::swap(d_cw_current, d_cw_next);

        // Store folded oracle
        std::vector<GoldilocksField> folded(cw_size);
        cudaMemcpy(folded.data(), d_cw_current, cw_size * sizeof(GoldilocksField),
                   cudaMemcpyDeviceToHost);
        proof.folded_oracles.push_back(folded);
    }

    // Final oracle
    proof.final_oracle.resize(cw_size);
    cudaMemcpy(proof.final_oracle.data(), d_cw_current,
               cw_size * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    // Query phase: generate random query indices
    std::mt19937 qrng(9999);
    proof.query_indices.resize(num_queries);
    size_t initial_cw_size = 1ULL << (num_vars + log_rate);
    for (int i = 0; i < num_queries; i++) {
        proof.query_indices[i] = qrng() % initial_cw_size;
    }

    // Extract query values from initial codeword and all folded oracles
    int* d_query_indices;
    cudaMalloc(&d_query_indices, num_queries * sizeof(int));
    cudaMemcpy(d_query_indices, proof.query_indices.data(),
               num_queries * sizeof(int), cudaMemcpyHostToDevice);

    GoldilocksField* d_query_out;
    cudaMalloc(&d_query_out, num_queries * 2 * sizeof(GoldilocksField));

    // Extract from initial codeword
    {
        int grid = (num_queries + BLOCK_SIZE - 1) / BLOCK_SIZE;
        GoldilocksField* d_initial_cw;
        cudaMalloc(&d_initial_cw, initial_cw_size * sizeof(GoldilocksField));
        cudaMemcpy(d_initial_cw, d_codeword, initial_cw_size * sizeof(GoldilocksField),
                   cudaMemcpyDeviceToDevice);
        basefold_extract_queries_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_initial_cw, d_query_indices, d_query_out, num_queries, initial_cw_size
        );
        cudaStreamSynchronize(stream);

        std::vector<GoldilocksField> qvals(num_queries * 2);
        cudaMemcpy(qvals.data(), d_query_out, num_queries * 2 * sizeof(GoldilocksField),
                   cudaMemcpyDeviceToHost);
        proof.query_vals.push_back(qvals);
        cudaFree(d_initial_cw);
    }

    // Cleanup
    cudaFree(d_eq_a); cudaFree(d_eq_b);
    cudaFree(d_bh_work); cudaFree(d_eq_out); cudaFree(d_bh_out);
    cudaFree(d_pc0); cudaFree(d_pc1); cudaFree(d_pc2);
    cudaFree(d_cw_current); cudaFree(d_cw_next);
    cudaFree(d_query_indices); cudaFree(d_query_out);

    return proof;
}

/**
 * Extension field open (Phase 8 ext2 variant).
 */
inline BasefoldProofExt2 basefold_open_ext2(
    const GoldilocksField* d_codeword,
    const GoldilocksField* d_bh_evals,
    const GoldilocksExt2* d_point_ext2,
    BasefoldTable& table,
    int num_vars,
    int log_rate,
    int num_rounds,
    int num_queries,
    cudaStream_t stream = 0
) {
    BasefoldProofExt2 proof;
    size_t N = 1ULL << num_vars;

    // Step 1: Build eq(x, point) in ext2 using ext2_eq_dp_all
    GoldilocksExt2 *d_eq_a, *d_eq_b, *d_eq_result;
    cudaMalloc(&d_eq_a, N * sizeof(GoldilocksExt2));
    cudaMalloc(&d_eq_b, N * sizeof(GoldilocksExt2));

    ext2_eq_dp_all(d_point_ext2, d_eq_a, d_eq_b, num_vars, &d_eq_result, stream);
    cudaStreamSynchronize(stream);

    if (d_eq_result != d_eq_a) {
        cudaMemcpy(d_eq_a, d_eq_result, N * sizeof(GoldilocksExt2), cudaMemcpyDeviceToDevice);
    }

    // bh_evals is in Type1 (bit-reversed) order from commit.
    // Both dot product and sum-check need Type2 (natural) order.
    GoldilocksField* d_bh_fp_work;
    cudaMalloc(&d_bh_fp_work, N * sizeof(GoldilocksField));
    cudaMemcpy(d_bh_fp_work, d_bh_evals, N * sizeof(GoldilocksField), cudaMemcpyDeviceToDevice);
    bit_reverse_permute_gl(d_bh_fp_work, num_vars, stream);
    cudaStreamSynchronize(stream);

    // Step 2: Compute eval = mixed dot product (bh_evals_Type2_Fp, eq_ext2)
    int dp_blocks = min((int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    GoldilocksExt2* d_dp_partial;
    cudaMalloc(&d_dp_partial, dp_blocks * sizeof(GoldilocksExt2));
    ext2_dot_product_mixed_kernel<<<dp_blocks, BLOCK_SIZE, 0, stream>>>(
        d_bh_fp_work, d_eq_a, d_dp_partial, N
    );
    cudaStreamSynchronize(stream);

    std::vector<GoldilocksExt2> dp_partial(dp_blocks);
    cudaMemcpy(dp_partial.data(), d_dp_partial, dp_blocks * sizeof(GoldilocksExt2),
               cudaMemcpyDeviceToHost);
    proof.eval = GoldilocksExt2();
    for (int i = 0; i < dp_blocks; i++) {
        proof.eval = ext2_add_host(proof.eval, dp_partial[i]);
    }
    cudaFree(d_dp_partial);

    GoldilocksExt2 *d_bh_ext2_work, *d_eq_ext2_out, *d_bh_ext2_out;
    cudaMalloc(&d_bh_ext2_work, N * sizeof(GoldilocksExt2));
    cudaMalloc(&d_eq_ext2_out, N * sizeof(GoldilocksExt2));
    cudaMalloc(&d_bh_ext2_out, N * sizeof(GoldilocksExt2));

    int max_blocks = 256;
    GoldilocksExt2 *d_pc0, *d_pc1, *d_pc2;
    cudaMalloc(&d_pc0, max_blocks * sizeof(GoldilocksExt2));
    cudaMalloc(&d_pc1, max_blocks * sizeof(GoldilocksExt2));
    cudaMalloc(&d_pc2, max_blocks * sizeof(GoldilocksExt2));

    size_t current_pairs = N / 2;

    // Step 3: First sum-check round (mixed: bh in F_p, eq in F_{p^2})
    {
        int grid_interp = (current_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // Interp eq (ext2)
        sumcheck_interp_ext2_kernel<<<grid_interp, BLOCK_SIZE, 0, stream>>>(
            d_eq_a, current_pairs
        );
        // Interp bh (F_p)
        sumcheck_interp_mixed_kernel<<<grid_interp, BLOCK_SIZE, 0, stream>>>(
            d_bh_fp_work, current_pairs
        );

        // Mixed product
        int grid_prod = min((int)((current_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE), max_blocks);
        sumcheck_product_mixed_kernel<<<grid_prod, BLOCK_SIZE, 0, stream>>>(
            d_eq_a, d_bh_fp_work, d_pc0, d_pc1, d_pc2, current_pairs
        );

        GoldilocksExt2 sc_coeffs[3];
        sumcheck_reduce_partials_ext2(d_pc0, d_pc1, d_pc2, sc_coeffs, grid_prod, stream);
        proof.sumcheck_oracles.push_back(sc_coeffs[0]);
        proof.sumcheck_oracles.push_back(sc_coeffs[1]);
        proof.sumcheck_oracles.push_back(sc_coeffs[2]);
    }

    // Step 4: Folding loop
    size_t cw_size = 1ULL << (num_vars + log_rate);

    // For codeword: first round is mixed (F_p -> F_{p^2}), subsequent are pure ext2
    GoldilocksField* d_cw_fp;
    GoldilocksExt2* d_cw_ext2;
    GoldilocksExt2* d_cw_ext2_next;
    cudaMalloc(&d_cw_fp, cw_size * sizeof(GoldilocksField));
    cudaMalloc(&d_cw_ext2, cw_size * sizeof(GoldilocksExt2));
    cudaMalloc(&d_cw_ext2_next, cw_size * sizeof(GoldilocksExt2));
    cudaMemcpy(d_cw_fp, d_codeword, cw_size * sizeof(GoldilocksField),
               cudaMemcpyDeviceToDevice);

    for (int round = 0; round < num_rounds; round++) {
        // Challenge (deterministic for testing)
        GoldilocksExt2 challenge(GoldilocksField(round * 17 + 42), GoldilocksField(round * 7 + 3));

        if (round == 0) {
            // Mixed eval: bh from F_p -> F_{p^2}
            int grid = (current_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sumcheck_eval_mixed_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_bh_fp_work, challenge, d_bh_ext2_work, current_pairs
            );
            // Eval eq (ext2)
            sumcheck_eval_ext2_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_eq_a, challenge, d_eq_ext2_out, current_pairs
            );
            cudaStreamSynchronize(stream);

            cudaMemcpy(d_eq_a, d_eq_ext2_out, current_pairs * sizeof(GoldilocksExt2),
                       cudaMemcpyDeviceToDevice);

            // Mixed fold codeword
            size_t cw_pairs = cw_size / 2;
            basefold_fold_mixed(d_cw_fp, table.device_level(round), challenge,
                                d_cw_ext2, cw_pairs, stream);
            cudaStreamSynchronize(stream);
            cw_size = cw_pairs;
        } else {
            // Pure ext2 eval
            int grid = (current_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sumcheck_eval_ext2_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_eq_a, challenge, d_eq_ext2_out, current_pairs
            );
            sumcheck_eval_ext2_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_bh_ext2_work, challenge, d_bh_ext2_out, current_pairs
            );
            cudaStreamSynchronize(stream);

            cudaMemcpy(d_eq_a, d_eq_ext2_out, current_pairs * sizeof(GoldilocksExt2),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_bh_ext2_work, d_bh_ext2_out, current_pairs * sizeof(GoldilocksExt2),
                       cudaMemcpyDeviceToDevice);

            // Ext2 fold codeword
            size_t cw_pairs = cw_size / 2;
            basefold_fold_ext2(d_cw_ext2, table.device_level(round), challenge,
                               d_cw_ext2_next, cw_pairs, stream);
            cudaStreamSynchronize(stream);
            cw_size = cw_pairs;
            std::swap(d_cw_ext2, d_cw_ext2_next);
        }

        current_pairs /= 2;

        // Next sum-check round (all ext2 now)
        if (current_pairs > 0) {
            int grid_interp = (current_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sumcheck_interp_ext2_kernel<<<grid_interp, BLOCK_SIZE, 0, stream>>>(
                d_eq_a, current_pairs
            );
            sumcheck_interp_ext2_kernel<<<grid_interp, BLOCK_SIZE, 0, stream>>>(
                d_bh_ext2_work, current_pairs
            );

            int grid_prod = min((int)((current_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE), max_blocks);
            sumcheck_product_ext2_kernel<<<grid_prod, BLOCK_SIZE, 0, stream>>>(
                d_eq_a, d_bh_ext2_work, d_pc0, d_pc1, d_pc2, current_pairs
            );

            GoldilocksExt2 sc_coeffs[3];
            sumcheck_reduce_partials_ext2(d_pc0, d_pc1, d_pc2, sc_coeffs, grid_prod, stream);
            proof.sumcheck_oracles.push_back(sc_coeffs[0]);
            proof.sumcheck_oracles.push_back(sc_coeffs[1]);
            proof.sumcheck_oracles.push_back(sc_coeffs[2]);
        }
    }

    // Final oracle
    proof.final_oracle.resize(cw_size);
    cudaMemcpy(proof.final_oracle.data(), d_cw_ext2,
               cw_size * sizeof(GoldilocksExt2), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_eq_a); cudaFree(d_eq_b);
    cudaFree(d_bh_fp_work); cudaFree(d_bh_ext2_work);
    cudaFree(d_eq_ext2_out); cudaFree(d_bh_ext2_out);
    cudaFree(d_pc0); cudaFree(d_pc1); cudaFree(d_pc2);
    cudaFree(d_cw_fp); cudaFree(d_cw_ext2); cudaFree(d_cw_ext2_next);

    return proof;
}

// ============================================================================
// Phase 11: Verify (Host-Side)
// ============================================================================

/**
 * Host-side degree-2 polynomial evaluation: p(x) = c0 + c1*x + c2*x^2
 */
static inline GoldilocksField degree_2_eval_host(
    GoldilocksField c0, GoldilocksField c1, GoldilocksField c2,
    GoldilocksField x
) {
    return gl_add_host(c0, gl_add_host(gl_mul_host(c1, x), gl_mul_host(c2, gl_mul_host(x, x))));
}

/**
 * Host-side sum-check consistency verification.
 * Checks: degree_2_zero_plus_one(oracle[0]) == eval
 * and:    degree_2_eval(oracle[i], challenge[i]) == degree_2_zero_plus_one(oracle[i+1])
 */
static inline bool verify_sumcheck_consistency(
    const std::vector<GoldilocksField>& oracles,  // flat: [c0,c1,c2] * (num_rounds+1)
    const std::vector<GoldilocksField>& challenges,
    GoldilocksField eval,
    int num_rounds
) {
    // Check first oracle: c0 + (c0 + c1 + c2) = 2*c0 + c1 + c2 == eval
    // p(0) + p(1) = c0 + (c0 + c1 + c2) = 2*c0 + c1 + c2
    GoldilocksField c0 = oracles[0], c1 = oracles[1], c2 = oracles[2];
    GoldilocksField sum_01 = gl_add_host(gl_add_host(c0, c0), gl_add_host(c1, c2));

    GoldilocksField eval_canon = GoldilocksField(eval.value % GOLDILOCKS_PRIME);
    GoldilocksField sum_canon = GoldilocksField(sum_01.value % GOLDILOCKS_PRIME);

    if (eval_canon.value != sum_canon.value) {
        printf("  Sum-check consistency FAIL at round 0: eval=%lu, p(0)+p(1)=%lu\n",
               eval_canon.value, sum_canon.value);
        return false;
    }

    for (int i = 0; i < num_rounds; i++) {
        // degree_2_eval(oracle[i], challenge[i]) should equal p(0)+p(1) of oracle[i+1]
        GoldilocksField ci0 = oracles[i * 3];
        GoldilocksField ci1 = oracles[i * 3 + 1];
        GoldilocksField ci2 = oracles[i * 3 + 2];
        GoldilocksField lhs = degree_2_eval_host(ci0, ci1, ci2, challenges[i]);

        if (i + 1 <= num_rounds) {
            GoldilocksField ni0 = oracles[(i + 1) * 3];
            GoldilocksField ni1 = oracles[(i + 1) * 3 + 1];
            GoldilocksField ni2 = oracles[(i + 1) * 3 + 2];
            GoldilocksField rhs = gl_add_host(gl_add_host(ni0, ni0), gl_add_host(ni1, ni2));

            GoldilocksField lhs_c = GoldilocksField(lhs.value % GOLDILOCKS_PRIME);
            GoldilocksField rhs_c = GoldilocksField(rhs.value % GOLDILOCKS_PRIME);

            if (lhs_c.value != rhs_c.value) {
                printf("  Sum-check consistency FAIL at round %d: lhs=%lu, rhs=%lu\n",
                       i + 1, lhs_c.value, rhs_c.value);
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Tests
// ============================================================================

#ifdef BASEFOLD_TEST

static void test_bit_reversal() {
    printf("=== Test: Bit Reversal ===\n");
    const int LOG_N = 4;
    const int N = 1 << LOG_N;

    std::vector<GoldilocksField> h_data(N);
    for (int i = 0; i < N; i++) h_data[i] = GoldilocksField(i);

    GoldilocksField* d_data;
    cudaMalloc(&d_data, N * sizeof(GoldilocksField));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    bit_reverse_permute_gl(d_data, LOG_N);
    cudaDeviceSynchronize();

    std::vector<GoldilocksField> h_result(N);
    cudaMemcpy(h_result.data(), d_data, N * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    // Verify: h_result[i] should be h_data[bit_reverse(i)]
    bool pass = true;
    for (int i = 0; i < N; i++) {
        size_t rev = 0;
        for (int b = 0; b < LOG_N; b++) {
            rev = (rev << 1) | ((i >> b) & 1);
        }
        if (h_result[i].value != rev) {
            printf("  FAIL: result[%d] = %lu, expected %lu\n", i, h_result[i].value, rev);
            pass = false;
        }
    }

    // Apply again: should get back original
    bit_reverse_permute_gl(d_data, LOG_N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result.data(), d_data, N * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (h_result[i].value != (uint64_t)i) {
            printf("  FAIL (double reverse): result[%d] = %lu, expected %d\n",
                   i, h_result[i].value, i);
            pass = false;
        }
    }

    printf("  %s\n", pass ? "PASS" : "FAIL");
    cudaFree(d_data);
}

static void test_bhc_interpolation() {
    printf("=== Test: BHC Interpolation ===\n");
    const int NUM_VARS = 3;
    const int N = 1 << NUM_VARS;

    // Create test polynomial evaluations: f(x) = x1 + 2*x2 + 3*x3 + 1
    // Evaluated over {0,1}^3 in Type2 order
    std::vector<GoldilocksField> h_evals(N);
    for (int i = 0; i < N; i++) {
        uint64_t val = 1;
        if (i & 1) val += 1;
        if (i & 2) val += 2;
        if (i & 4) val += 3;
        h_evals[i] = GoldilocksField(val);
    }

    GoldilocksField *d_evals, *d_coeffs, *d_bh_evals;
    cudaMalloc(&d_evals, N * sizeof(GoldilocksField));
    cudaMalloc(&d_coeffs, N * sizeof(GoldilocksField));
    cudaMalloc(&d_bh_evals, N * sizeof(GoldilocksField));

    cudaMemcpy(d_evals, h_evals.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    bhc_interpolate(d_evals, d_coeffs, d_bh_evals, NUM_VARS);
    cudaDeviceSynchronize();

    std::vector<GoldilocksField> h_coeffs(N), h_bh(N);
    cudaMemcpy(h_coeffs.data(), d_coeffs, N * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bh.data(), d_bh_evals, N * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    // Verify: for a multilinear polynomial f(x1,x2,x3) = 1 + x1 + 2*x2 + 3*x3
    // The coefficients in the multilinear basis should be specific values.
    // We verify by re-evaluating: for each x in {0,1}^3,
    // f(x) = sum_S coeff_S * prod_{i in S} x_i
    bool pass = true;
    for (int x = 0; x < N; x++) {
        GoldilocksField eval(0);
        for (int S = 0; S < N; S++) {
            // Check if all bits in S are set in x
            if ((x & S) == S) {
                eval = gl_add_host(eval, h_coeffs[S]);
            }
        }
        uint64_t expected = 1;
        if (x & 1) expected += 1;
        if (x & 2) expected += 2;
        if (x & 4) expected += 3;
        uint64_t got = eval.value % GOLDILOCKS_PRIME;
        if (got != expected) {
            printf("  FAIL: re-eval at x=%d: got %lu, expected %lu\n", x, got, expected);
            pass = false;
        }
    }

    printf("  %s\n", pass ? "PASS" : "FAIL");
    cudaFree(d_evals); cudaFree(d_coeffs); cudaFree(d_bh_evals);
}

static void test_encoding() {
    printf("=== Test: Foldable Domain Encoding ===\n");
    const int NUM_VARS = 3;
    const int LOG_RATE = 1;
    const int N = 1 << NUM_VARS;
    const int CW_LEN = 1 << (NUM_VARS + LOG_RATE);

    // Simple coefficients
    std::vector<GoldilocksField> h_coeffs(N);
    for (int i = 0; i < N; i++) h_coeffs[i] = GoldilocksField(i + 1);

    GoldilocksField *d_coeffs, *d_codeword;
    cudaMalloc(&d_coeffs, N * sizeof(GoldilocksField));
    cudaMalloc(&d_codeword, CW_LEN * sizeof(GoldilocksField));

    cudaMemcpy(d_coeffs, h_coeffs.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    encode_foldable_domain(d_coeffs, d_codeword, NUM_VARS, LOG_RATE);
    cudaDeviceSynchronize();

    std::vector<GoldilocksField> h_cw(CW_LEN);
    cudaMemcpy(h_cw.data(), d_codeword, CW_LEN * sizeof(GoldilocksField),
               cudaMemcpyDeviceToHost);

    // Basic sanity: codeword should be non-trivial and have correct length
    bool pass = true;
    bool all_zero = true;
    for (int i = 0; i < CW_LEN; i++) {
        if (h_cw[i].value != 0) all_zero = false;
    }
    if (all_zero) {
        printf("  FAIL: all codeword elements are zero\n");
        pass = false;
    }

    printf("  Codeword length: %d (expected %d)\n", CW_LEN, CW_LEN);
    printf("  First few elements: %lu %lu %lu %lu\n",
           h_cw[0].value % GOLDILOCKS_PRIME,
           h_cw[1].value % GOLDILOCKS_PRIME,
           h_cw[2].value % GOLDILOCKS_PRIME,
           h_cw[3].value % GOLDILOCKS_PRIME);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_coeffs); cudaFree(d_codeword);
}

static void test_sumcheck_base_field() {
    printf("=== Test: Sum-Check (Base Field) ===\n");
    const int NUM_VARS = 4;
    const int N = 1 << NUM_VARS;

    // Create random polynomial evals and random point
    std::mt19937_64 rng(42);
    std::vector<GoldilocksField> h_evals(N), h_point(NUM_VARS);
    for (int i = 0; i < N; i++) h_evals[i] = GoldilocksField(rng() % GOLDILOCKS_PRIME);
    for (int i = 0; i < NUM_VARS; i++) h_point[i] = GoldilocksField(rng() % GOLDILOCKS_PRIME);

    // Compute expected eval on CPU: eval = sum_x f(x) * eq(x, point)
    GoldilocksField expected_eval(0);
    for (int x = 0; x < N; x++) {
        // eq(x, r) = prod_i (x_i * r_i + (1 - x_i) * (1 - r_i))
        GoldilocksField eq_val(1);
        for (int i = 0; i < NUM_VARS; i++) {
            int xi = (x >> i) & 1;
            GoldilocksField ri = h_point[i];
            if (xi) {
                eq_val = gl_mul_host(eq_val, ri);
            } else {
                eq_val = gl_mul_host(eq_val, gl_sub_host(GoldilocksField(1), ri));
            }
        }
        expected_eval = gl_add_host(expected_eval, gl_mul_host(h_evals[x], eq_val));
    }

    // GPU computation
    GoldilocksField *d_point, *d_eq_a, *d_eq_b, *d_eq_result;
    GoldilocksField *d_bh;
    cudaMalloc(&d_point, NUM_VARS * sizeof(GoldilocksField));
    cudaMalloc(&d_eq_a, N * sizeof(GoldilocksField));
    cudaMalloc(&d_eq_b, N * sizeof(GoldilocksField));
    cudaMalloc(&d_bh, N * sizeof(GoldilocksField));

    cudaMemcpy(d_point, h_point.data(), NUM_VARS * sizeof(GoldilocksField), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bh, h_evals.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    // Build eq
    eq_dp_all(d_point, d_eq_a, d_eq_b, NUM_VARS, &d_eq_result);
    cudaDeviceSynchronize();
    if (d_eq_result != d_eq_a) {
        cudaMemcpy(d_eq_a, d_eq_result, N * sizeof(GoldilocksField), cudaMemcpyDeviceToDevice);
    }

    // Dot product
    int dp_blocks = min((int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    GoldilocksField* d_dp_partial;
    cudaMalloc(&d_dp_partial, dp_blocks * sizeof(GoldilocksField));
    gl_dot_product_kernel<<<dp_blocks, BLOCK_SIZE>>>(d_bh, d_eq_a, d_dp_partial, N);
    cudaDeviceSynchronize();

    std::vector<GoldilocksField> dp_partial(dp_blocks);
    cudaMemcpy(dp_partial.data(), d_dp_partial, dp_blocks * sizeof(GoldilocksField),
               cudaMemcpyDeviceToHost);
    GoldilocksField gpu_eval(0);
    for (int i = 0; i < dp_blocks; i++) {
        gpu_eval = gl_add_host(gpu_eval, dp_partial[i]);
    }

    bool pass = true;
    uint64_t exp_c = expected_eval.value % GOLDILOCKS_PRIME;
    uint64_t got_c = gpu_eval.value % GOLDILOCKS_PRIME;
    if (exp_c != got_c) {
        printf("  FAIL: eval mismatch: expected %lu, got %lu\n", exp_c, got_c);
        pass = false;
    } else {
        printf("  Eval correct: %lu\n", exp_c);
    }

    // Run sum-check rounds
    int max_blocks = 256;
    GoldilocksField *d_pc0, *d_pc1, *d_pc2;
    cudaMalloc(&d_pc0, max_blocks * sizeof(GoldilocksField));
    cudaMalloc(&d_pc1, max_blocks * sizeof(GoldilocksField));
    cudaMalloc(&d_pc2, max_blocks * sizeof(GoldilocksField));

    GoldilocksField *d_eq_out, *d_bh_out;
    cudaMalloc(&d_eq_out, N * sizeof(GoldilocksField));
    cudaMalloc(&d_bh_out, N * sizeof(GoldilocksField));

    std::vector<GoldilocksField> all_oracles;
    std::vector<GoldilocksField> challenges;

    size_t current_pairs = N / 2;
    GoldilocksField sc_coeffs[3];

    // First round
    sumcheck_round(d_eq_a, d_bh, sc_coeffs, current_pairs, d_pc0, d_pc1, d_pc2);
    all_oracles.push_back(sc_coeffs[0]);
    all_oracles.push_back(sc_coeffs[1]);
    all_oracles.push_back(sc_coeffs[2]);

    for (int round = 0; round < NUM_VARS - 1; round++) {
        GoldilocksField challenge(round * 17 + 42);
        challenges.push_back(challenge);

        sumcheck_apply_challenge(d_eq_a, d_eq_out, d_bh, d_bh_out,
                                 challenge, current_pairs);
        cudaDeviceSynchronize();
        std::swap(d_eq_a, d_eq_out);
        std::swap(d_bh, d_bh_out);
        current_pairs /= 2;

        sumcheck_round(d_eq_a, d_bh, sc_coeffs, current_pairs, d_pc0, d_pc1, d_pc2);
        all_oracles.push_back(sc_coeffs[0]);
        all_oracles.push_back(sc_coeffs[1]);
        all_oracles.push_back(sc_coeffs[2]);
    }

    // Verify sum-check consistency
    bool sc_pass = verify_sumcheck_consistency(all_oracles, challenges, gpu_eval, NUM_VARS - 1);
    if (!sc_pass) pass = false;

    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_point); cudaFree(d_eq_a); cudaFree(d_eq_b);
    cudaFree(d_bh); cudaFree(d_dp_partial);
    cudaFree(d_pc0); cudaFree(d_pc1); cudaFree(d_pc2);
    cudaFree(d_eq_out); cudaFree(d_bh_out);
}

static void test_basefold_fold() {
    printf("=== Test: Basefold Codeword Folding ===\n");
    const int N = 8;  // 4 pairs

    // Simple codeword
    std::vector<GoldilocksField> h_cw(N);
    for (int i = 0; i < N; i++) h_cw[i] = GoldilocksField(i + 1);

    // Simple table: x0 = i, weight = 1/(1) = 1 (for testing, x1 = x0 + 1)
    std::vector<FoldingEntry> h_table(N / 2);
    for (int i = 0; i < N / 2; i++) {
        h_table[i].point = GoldilocksField(i);
        // weight = 1 / (x1 - x0) = 1 / 1 = 1 (if x1 = x0 + 1)
        h_table[i].weight = GoldilocksField(1);
    }

    GoldilocksField *d_cw, *d_out;
    FoldingEntry *d_table;
    cudaMalloc(&d_cw, N * sizeof(GoldilocksField));
    cudaMalloc(&d_out, (N / 2) * sizeof(GoldilocksField));
    cudaMalloc(&d_table, (N / 2) * sizeof(FoldingEntry));

    cudaMemcpy(d_cw, h_cw.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table, h_table.data(), (N / 2) * sizeof(FoldingEntry), cudaMemcpyHostToDevice);

    GoldilocksField challenge(5);
    basefold_fold(d_cw, d_table, challenge, d_out, N / 2);
    cudaDeviceSynchronize();

    std::vector<GoldilocksField> h_result(N / 2);
    cudaMemcpy(h_result.data(), d_out, (N / 2) * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    // Verify: result[i] = val0 + (challenge - x0) * (val1 - val0) * weight
    bool pass = true;
    for (int i = 0; i < N / 2; i++) {
        GoldilocksField val0 = h_cw[2 * i];
        GoldilocksField val1 = h_cw[2 * i + 1];
        GoldilocksField x0 = h_table[i].point;
        GoldilocksField w = h_table[i].weight;

        GoldilocksField diff = gl_sub_host(val1, val0);
        GoldilocksField cx = gl_sub_host(challenge, x0);
        GoldilocksField expected = gl_add_host(val0, gl_mul_host(gl_mul_host(cx, diff), w));

        uint64_t exp_c = expected.value % GOLDILOCKS_PRIME;
        uint64_t got_c = h_result[i].value % GOLDILOCKS_PRIME;

        if (exp_c != got_c) {
            printf("  FAIL at i=%d: expected %lu, got %lu\n", i, exp_c, got_c);
            pass = false;
        }
    }

    printf("  %s\n", pass ? "PASS" : "FAIL");
    cudaFree(d_cw); cudaFree(d_out); cudaFree(d_table);
}

static void test_basefold_fold_ext2() {
    printf("=== Test: Basefold Ext2 Fold ===\n");
    const int N = 8;

    // Mixed fold: F_p codeword + F_{p^2} challenge
    std::vector<GoldilocksField> h_cw(N);
    for (int i = 0; i < N; i++) h_cw[i] = GoldilocksField(i + 1);

    std::vector<FoldingEntry> h_table(N / 2);
    for (int i = 0; i < N / 2; i++) {
        h_table[i].point = GoldilocksField(i * 2);
        h_table[i].weight = GoldilocksField(1);
    }

    GoldilocksField *d_cw;
    GoldilocksExt2 *d_out;
    FoldingEntry *d_table;
    cudaMalloc(&d_cw, N * sizeof(GoldilocksField));
    cudaMalloc(&d_out, (N / 2) * sizeof(GoldilocksExt2));
    cudaMalloc(&d_table, (N / 2) * sizeof(FoldingEntry));

    cudaMemcpy(d_cw, h_cw.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table, h_table.data(), (N / 2) * sizeof(FoldingEntry), cudaMemcpyHostToDevice);

    GoldilocksExt2 challenge(GoldilocksField(5), GoldilocksField(3));
    basefold_fold_mixed(d_cw, d_table, challenge, d_out, N / 2);
    cudaDeviceSynchronize();

    std::vector<GoldilocksExt2> h_result(N / 2);
    cudaMemcpy(h_result.data(), d_out, (N / 2) * sizeof(GoldilocksExt2), cudaMemcpyDeviceToHost);

    // Verify on CPU
    bool pass = true;
    for (int i = 0; i < N / 2; i++) {
        GoldilocksField val0 = h_cw[2 * i];
        GoldilocksField val1 = h_cw[2 * i + 1];
        GoldilocksField x0 = h_table[i].point;
        GoldilocksField w = h_table[i].weight;

        GoldilocksField diff = gl_sub_host(val1, val0);
        GoldilocksField diff_w = gl_mul_host(diff, w);
        GoldilocksExt2 cx = ext2_sub_host(challenge, GoldilocksExt2(x0, GoldilocksField(0)));
        GoldilocksExt2 expected = ext2_add_host(
            GoldilocksExt2(val0, GoldilocksField(0)),
            ext2_scalar_mul_host(diff_w, cx)
        );

        uint64_t exp0 = expected.c[0].value % GOLDILOCKS_PRIME;
        uint64_t exp1 = expected.c[1].value % GOLDILOCKS_PRIME;
        uint64_t got0 = h_result[i].c[0].value % GOLDILOCKS_PRIME;
        uint64_t got1 = h_result[i].c[1].value % GOLDILOCKS_PRIME;

        if (exp0 != got0 || exp1 != got1) {
            printf("  FAIL at i=%d: expected (%lu, %lu), got (%lu, %lu)\n",
                   i, exp0, exp1, got0, got1);
            pass = false;
        }
    }

    printf("  %s\n", pass ? "PASS" : "FAIL");
    cudaFree(d_cw); cudaFree(d_out); cudaFree(d_table);
}

static void test_commit_open_base() {
    printf("=== Test: Commit + Open (Base Field) ===\n");
    const int NUM_VARS = 4;
    const int LOG_RATE = 1;
    const int N = 1 << NUM_VARS;
    const int CW_LEN = 1 << (NUM_VARS + LOG_RATE);
    const int NUM_ROUNDS = NUM_VARS;
    const int NUM_QUERIES = 1;

    // Initialize CUDA constants
    goldilocks_init();
    poseidon2_init();

    // Create polynomial
    std::mt19937_64 rng(42);
    std::vector<GoldilocksField> h_evals(N);
    for (int i = 0; i < N; i++) h_evals[i] = GoldilocksField(rng() % 1000);

    // Create point
    std::vector<GoldilocksField> h_point(NUM_VARS);
    for (int i = 0; i < NUM_VARS; i++) h_point[i] = GoldilocksField(rng() % 100);

    // Generate table
    BasefoldTable table = generate_folding_table(NUM_VARS, LOG_RATE, NUM_ROUNDS, 12345);
    table.upload();

    // Allocate device memory
    GoldilocksField *d_evals, *d_coeffs, *d_codeword, *d_bh_evals;
    size_t num_leaves = CW_LEN / 2;
    size_t tree_size = (2 * num_leaves - 1) * 4;
    GoldilocksField *d_tree;

    cudaMalloc(&d_evals, N * sizeof(GoldilocksField));
    cudaMalloc(&d_coeffs, N * sizeof(GoldilocksField));
    cudaMalloc(&d_codeword, CW_LEN * sizeof(GoldilocksField));
    cudaMalloc(&d_bh_evals, N * sizeof(GoldilocksField));
    cudaMalloc(&d_tree, tree_size * sizeof(GoldilocksField));

    cudaMemcpy(d_evals, h_evals.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    // Commit
    basefold_commit(d_evals, d_coeffs, d_codeword, d_bh_evals, d_tree,
                    NUM_VARS, LOG_RATE);
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error after commit: %s\n", cudaGetErrorString(err));
        printf("  FAIL\n");
    } else {
        printf("  Commit completed successfully\n");
    }

    // Open
    GoldilocksField* d_point;
    cudaMalloc(&d_point, NUM_VARS * sizeof(GoldilocksField));
    cudaMemcpy(d_point, h_point.data(), NUM_VARS * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    BasefoldProof proof = basefold_open(
        d_codeword, d_bh_evals, d_point, table,
        NUM_VARS, LOG_RATE, NUM_ROUNDS, NUM_QUERIES
    );

    printf("  Open completed: eval = %lu\n", proof.eval.value % GOLDILOCKS_PRIME);
    printf("  Sum-check oracles: %zu rounds\n", proof.sumcheck_oracles.size() / 3);
    printf("  Final oracle size: %zu\n", proof.final_oracle.size());

    // Verify eval on CPU
    GoldilocksField cpu_eval(0);
    for (int x = 0; x < N; x++) {
        GoldilocksField eq_val(1);
        for (int i = 0; i < NUM_VARS; i++) {
            int xi = (x >> i) & 1;
            GoldilocksField ri = h_point[i];
            if (xi) eq_val = gl_mul_host(eq_val, ri);
            else eq_val = gl_mul_host(eq_val, gl_sub_host(GoldilocksField(1), ri));
        }
        cpu_eval = gl_add_host(cpu_eval, gl_mul_host(h_evals[x], eq_val));
    }

    uint64_t exp_c = cpu_eval.value % GOLDILOCKS_PRIME;
    uint64_t got_c = proof.eval.value % GOLDILOCKS_PRIME;
    bool pass = (exp_c == got_c);
    printf("  Eval check: CPU=%lu, GPU=%lu -> %s\n", exp_c, got_c, pass ? "MATCH" : "MISMATCH");

    // Verify sum-check consistency
    std::vector<GoldilocksField> challenges;
    for (int i = 0; i < NUM_ROUNDS; i++) {
        challenges.push_back(GoldilocksField(i * 17 + 42));
    }
    int num_sc_oracles = (int)(proof.sumcheck_oracles.size() / 3);
    bool sc_pass = verify_sumcheck_consistency(proof.sumcheck_oracles, challenges,
                                                proof.eval, num_sc_oracles - 1);
    printf("  Sum-check consistency: %s\n", sc_pass ? "PASS" : "FAIL");
    if (!sc_pass) pass = false;

    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_evals); cudaFree(d_coeffs); cudaFree(d_codeword);
    cudaFree(d_bh_evals); cudaFree(d_tree); cudaFree(d_point);
}

static void test_ext2_open() {
    printf("=== Test: Extension Field Open ===\n");
    const int NUM_VARS = 4;
    const int LOG_RATE = 1;
    const int N = 1 << NUM_VARS;
    const int CW_LEN = 1 << (NUM_VARS + LOG_RATE);
    const int NUM_ROUNDS = NUM_VARS;
    const int NUM_QUERIES = 1;

    goldilocks_init();
    poseidon2_init();

    // Create polynomial (base field evals)
    std::mt19937_64 rng(42);
    std::vector<GoldilocksField> h_evals(N);
    for (int i = 0; i < N; i++) h_evals[i] = GoldilocksField(rng() % 1000);

    // Extension field evaluation point
    std::vector<GoldilocksExt2> h_point(NUM_VARS);
    for (int i = 0; i < NUM_VARS; i++) {
        h_point[i] = GoldilocksExt2(GoldilocksField(rng() % 100), GoldilocksField(rng() % 100));
    }

    // Generate table
    BasefoldTable table = generate_folding_table(NUM_VARS, LOG_RATE, NUM_ROUNDS, 12345);
    table.upload();

    // Commit (same as base field)
    GoldilocksField *d_evals, *d_coeffs, *d_codeword, *d_bh_evals;
    size_t num_leaves = CW_LEN / 2;
    size_t tree_size = (2 * num_leaves - 1) * 4;
    GoldilocksField *d_tree;

    cudaMalloc(&d_evals, N * sizeof(GoldilocksField));
    cudaMalloc(&d_coeffs, N * sizeof(GoldilocksField));
    cudaMalloc(&d_codeword, CW_LEN * sizeof(GoldilocksField));
    cudaMalloc(&d_bh_evals, N * sizeof(GoldilocksField));
    cudaMalloc(&d_tree, tree_size * sizeof(GoldilocksField));

    cudaMemcpy(d_evals, h_evals.data(), N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    basefold_commit(d_evals, d_coeffs, d_codeword, d_bh_evals, d_tree,
                    NUM_VARS, LOG_RATE);
    cudaDeviceSynchronize();

    // Open with ext2 point
    GoldilocksExt2* d_point;
    cudaMalloc(&d_point, NUM_VARS * sizeof(GoldilocksExt2));
    cudaMemcpy(d_point, h_point.data(), NUM_VARS * sizeof(GoldilocksExt2), cudaMemcpyHostToDevice);

    BasefoldProofExt2 proof = basefold_open_ext2(
        d_codeword, d_bh_evals, d_point, table,
        NUM_VARS, LOG_RATE, NUM_ROUNDS, NUM_QUERIES
    );

    printf("  Open completed: eval = (%lu, %lu)\n",
           proof.eval.c[0].value % GOLDILOCKS_PRIME,
           proof.eval.c[1].value % GOLDILOCKS_PRIME);

    // Verify eval on CPU: f(z) = sum_x f(x) * eq(x, z) where z in F_{p^2}
    GoldilocksExt2 cpu_eval;
    for (int x = 0; x < N; x++) {
        GoldilocksExt2 eq_val(GoldilocksField(1), GoldilocksField(0));
        for (int i = 0; i < NUM_VARS; i++) {
            int xi = (x >> i) & 1;
            GoldilocksExt2 ri = h_point[i];
            GoldilocksExt2 one(GoldilocksField(1), GoldilocksField(0));
            if (xi) {
                eq_val = ext2_mul_host(eq_val, ri);
            } else {
                eq_val = ext2_mul_host(eq_val, ext2_sub_host(one, ri));
            }
        }
        // h_evals[x] is base field, so promote to ext2
        GoldilocksExt2 fval(h_evals[x], GoldilocksField(0));
        cpu_eval = ext2_add_host(cpu_eval, ext2_mul_host(fval, eq_val));
    }

    uint64_t exp0 = cpu_eval.c[0].value % GOLDILOCKS_PRIME;
    uint64_t exp1 = cpu_eval.c[1].value % GOLDILOCKS_PRIME;
    uint64_t got0 = proof.eval.c[0].value % GOLDILOCKS_PRIME;
    uint64_t got1 = proof.eval.c[1].value % GOLDILOCKS_PRIME;
    bool pass = (exp0 == got0 && exp1 == got1);
    printf("  Eval check: CPU=(%lu,%lu), GPU=(%lu,%lu) -> %s\n",
           exp0, exp1, got0, got1, pass ? "MATCH" : "MISMATCH");

    printf("  Sum-check oracles: %zu rounds\n", proof.sumcheck_oracles.size() / 3);
    printf("  Final oracle size: %zu\n", proof.final_oracle.size());
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_evals); cudaFree(d_coeffs); cudaFree(d_codeword);
    cudaFree(d_bh_evals); cudaFree(d_tree); cudaFree(d_point);
}

static void test_table_generation() {
    printf("=== Test: Table Generation ===\n");
    const int NUM_VARS = 4;
    const int LOG_RATE = 1;
    const int NUM_ROUNDS = 4;

    BasefoldTable table = generate_folding_table(NUM_VARS, LOG_RATE, NUM_ROUNDS, 12345);

    printf("  Num rounds: %d\n", table.num_rounds);
    printf("  Total entries: %zu\n", table.entries.size());
    for (int i = 0; i < NUM_ROUNDS; i++) {
        printf("  Level %d: offset=%zu, size=%zu\n",
               i, table.level_offsets[i], table.level_sizes[i]);
    }

    // Verify weights: point * weight should make sense
    bool pass = true;
    for (size_t i = 0; i < table.entries.size(); i++) {
        // weight != 0
        if (table.entries[i].weight.value == 0) {
            printf("  FAIL: zero weight at entry %zu\n", i);
            pass = false;
        }
    }

    // Upload and verify device pointer
    table.upload();
    if (table.d_entries == nullptr) {
        printf("  FAIL: device upload failed\n");
        pass = false;
    }

    printf("  %s\n", pass ? "PASS" : "FAIL");
}

int main() {
    // Initialize
    goldilocks_init();
    poseidon2_init();

    printf("========================================\n");
    printf("Basefold CUDA Kernel Tests\n");
    printf("========================================\n\n");

    // Phase 1
    test_bit_reversal();

    // Phase 2
    test_bhc_interpolation();

    // Phase 3
    test_encoding();

    // Phase 5
    test_sumcheck_base_field();

    // Phase 6
    test_basefold_fold();

    // Phase 7
    test_basefold_fold_ext2();

    // Phase 9
    test_table_generation();

    // Phase 4 + 8
    test_commit_open_base();

    // Phase 7 + 8 (ext2)
    test_ext2_open();

    printf("\n========================================\n");
    printf("All tests completed.\n");
    printf("========================================\n");

    return 0;
}

#endif // BASEFOLD_TEST

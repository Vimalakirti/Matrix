/**
 * GPU Sumcheck Prover Kernels — Almost-Goldilocks field
 *
 * Mirrors cuda/sumcheck_prover.cuh; only the field types/prefixes change.
 *
 * Memory layout: d polynomials packed contiguously with stride = original_size
 *   [poly_0: original_size | poly_1: original_size | ... | poly_{d-1}: original_size]
 * After round m, only first (original_size >> (m+1)) elements per poly are valid.
 */

#ifndef ALMOST_SUMCHECK_PROVER_CUH
#define ALMOST_SUMCHECK_PROVER_CUH

#include "almost_goldilocks.cuh"
#include "almost_extension.cuh"

#define AGL_SUMCHECK_BLOCK_SIZE 256
#define AGL_MAX_DEGREE 8

// ============================================================================
// Base field sumcheck round + fold
// ============================================================================

__global__ void agl_sumcheck_round_message_kernel(
    const uint64_t* __restrict__ d_polys,
    uint64_t* __restrict__ d_partial,
    int d,
    size_t original_size,
    size_t half
) {
    __shared__ uint64_t shared[(AGL_MAX_DEGREE + 1) * AGL_SUMCHECK_BLOCK_SIZE];

    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    int dp1 = d + 1;

    AlmostGoldilocksField acc[AGL_MAX_DEGREE + 1];
    for (int c = 0; c < dp1; c++) {
        acc[c] = AlmostGoldilocksField(0);
    }

    for (size_t y = idx; y < half; y += grid_size) {
        AlmostGoldilocksField even[AGL_MAX_DEGREE];
        AlmostGoldilocksField diff[AGL_MAX_DEGREE];

        for (int i = 0; i < d; i++) {
            size_t base = i * original_size;
            even[i] = AlmostGoldilocksField(d_polys[base + 2 * y]);
            AlmostGoldilocksField odd(d_polys[base + 2 * y + 1]);
            diff[i] = agl_sub(odd, even[i]);
        }

        for (int c = 0; c < dp1; c++) {
            AlmostGoldilocksField c_val((uint64_t)c);
            AlmostGoldilocksField product(1);
            for (int i = 0; i < d; i++) {
                AlmostGoldilocksField val = agl_add(even[i], agl_mul(c_val, diff[i]));
                product = agl_mul(product, val);
            }
            acc[c] = agl_add(acc[c], product);
        }
    }

    for (int c = 0; c < dp1; c++) {
        shared[c * AGL_SUMCHECK_BLOCK_SIZE + tid] = acc[c].value;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < dp1; c++) {
                int offset = c * AGL_SUMCHECK_BLOCK_SIZE;
                shared[offset + tid] = agl_add(
                    AlmostGoldilocksField(shared[offset + tid]),
                    AlmostGoldilocksField(shared[offset + tid + s])
                ).value;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int c = 0; c < dp1; c++) {
            d_partial[blockIdx.x * dp1 + c] = shared[c * AGL_SUMCHECK_BLOCK_SIZE];
        }
    }
}

__global__ void agl_sumcheck_fold_kernel(
    const uint64_t* __restrict__ d_input,
    uint64_t* __restrict__ d_output,
    uint64_t challenge,
    int d,
    size_t original_size,
    size_t half
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    AlmostGoldilocksField ch(challenge);

    for (size_t y = idx; y < half; y += grid_size) {
        for (int i = 0; i < d; i++) {
            size_t base = i * original_size;
            AlmostGoldilocksField a(d_input[base + 2 * y]);
            AlmostGoldilocksField b(d_input[base + 2 * y + 1]);
            d_output[base + y] = agl_add(a, agl_mul(ch, agl_sub(b, a))).value;
        }
    }
}

// ============================================================================
// Ext2 sumcheck round + fold
// ============================================================================

__global__ void aext2_sumcheck_round_message_kernel(
    const uint64_t* __restrict__ d_polys,
    uint64_t* __restrict__ d_partial,
    int d,
    size_t original_size,
    size_t half
) {
    __shared__ uint64_t shared[(AGL_MAX_DEGREE + 1) * AGL_SUMCHECK_BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    int dp1 = d + 1;
    size_t stride = original_size * 2;

    AlmostGoldilocksExt2 acc[AGL_MAX_DEGREE + 1];
    for (int c = 0; c < dp1; c++) acc[c] = AlmostGoldilocksExt2();

    for (size_t y = idx; y < half; y += grid_size) {
        AlmostGoldilocksExt2 even[AGL_MAX_DEGREE];
        AlmostGoldilocksExt2 diff[AGL_MAX_DEGREE];

        for (int i = 0; i < d; i++) {
            size_t base = i * stride;
            size_t even_off = base + 4 * y;
            size_t odd_off  = base + 4 * y + 2;
            even[i] = AlmostGoldilocksExt2(d_polys[even_off], d_polys[even_off + 1]);
            AlmostGoldilocksExt2 odd(d_polys[odd_off], d_polys[odd_off + 1]);
            diff[i] = aext2_sub(odd, even[i]);
        }

        for (int c = 0; c < dp1; c++) {
            AlmostGoldilocksExt2 c_ext(AlmostGoldilocksField((uint64_t)c));
            AlmostGoldilocksExt2 product(AlmostGoldilocksField(1));
            for (int i = 0; i < d; i++) {
                AlmostGoldilocksExt2 val = aext2_add(even[i], aext2_mul(c_ext, diff[i]));
                product = aext2_mul(product, val);
            }
            acc[c] = aext2_add(acc[c], product);
        }
    }

    for (int c = 0; c < dp1; c++) {
        int offset = c * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
        shared[offset]     = acc[c].c[0].value;
        shared[offset + 1] = acc[c].c[1].value;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < dp1; c++) {
                int offset = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
                AlmostGoldilocksExt2 a(shared[offset + tid * 2], shared[offset + tid * 2 + 1]);
                AlmostGoldilocksExt2 b(shared[offset + (tid + s) * 2], shared[offset + (tid + s) * 2 + 1]);
                AlmostGoldilocksExt2 sum = aext2_add(a, b);
                shared[offset + tid * 2]     = sum.c[0].value;
                shared[offset + tid * 2 + 1] = sum.c[1].value;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int c = 0; c < dp1; c++) {
            int offset = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
            d_partial[(blockIdx.x * dp1 + c) * 2]     = shared[offset];
            d_partial[(blockIdx.x * dp1 + c) * 2 + 1] = shared[offset + 1];
        }
    }
}

__global__ void aext2_sumcheck_fold_kernel(
    const uint64_t* __restrict__ d_input,
    uint64_t* __restrict__ d_output,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    int d,
    size_t original_size,
    size_t half
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    AlmostGoldilocksExt2 ch(challenge_c0, challenge_c1);
    size_t stride = original_size * 2;

    for (size_t y = idx; y < half; y += grid_size) {
        for (int i = 0; i < d; i++) {
            size_t base = i * stride;
            size_t even_off = base + 4 * y;
            size_t odd_off  = base + 4 * y + 2;
            AlmostGoldilocksExt2 a(d_input[even_off], d_input[even_off + 1]);
            AlmostGoldilocksExt2 b(d_input[odd_off], d_input[odd_off + 1]);
            AlmostGoldilocksExt2 result = aext2_add(a, aext2_mul(ch, aext2_sub(b, a)));
            d_output[base + y * 2]     = result.c[0].value;
            d_output[base + y * 2 + 1] = result.c[1].value;
        }
    }
}

// ============================================================================
// Batched per-leaf same-point sumcheck round + fold
//
// Layout (degree-2 only — eq + f per leaf):
//   d_polys = [leaf_0_eq | leaf_0_f | leaf_1_eq | leaf_1_f | ...]
//   each section is `original_size` Ext2 values = `2 * original_size` u64.
//   stride_leaf_u64 = 2 * 2 * original_size  (= 2 polys × 2 u64/Ext2 × N)
//
// Grid: (num_blocks_x, num_leaves). Each block handles one leaf's chunk
// of y positions; partial sums per (block, leaf, c) are written to
//   d_partial[block_x * (num_leaves * 3 * 2) + leaf * (3 * 2) + c * 2 + {0,1}].
// ============================================================================

__global__ void aext2_sumcheck_batched_round_message_kernel(
    const uint64_t* __restrict__ d_polys,
    uint64_t* __restrict__ d_partial,
    size_t original_size,
    size_t half,
    int num_leaves
) {
    int leaf = blockIdx.y;
    if (leaf >= num_leaves) return;

    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    const size_t poly_stride_u64 = original_size * 2;
    const size_t leaf_stride_u64 = 2 * poly_stride_u64;
    const uint64_t* leaf_polys = d_polys + (size_t)leaf * leaf_stride_u64;

    __shared__ uint64_t shared[3 * AGL_SUMCHECK_BLOCK_SIZE * 2];

    AlmostGoldilocksExt2 acc0, acc1, acc2;

    for (size_t y = idx; y < half; y += grid_size) {
        size_t e_even = 0 * poly_stride_u64 + 4 * y;
        size_t f_even = 1 * poly_stride_u64 + 4 * y;
        AlmostGoldilocksExt2 e0(leaf_polys[e_even],     leaf_polys[e_even + 1]);
        AlmostGoldilocksExt2 e1(leaf_polys[e_even + 2], leaf_polys[e_even + 3]);
        AlmostGoldilocksExt2 f0(leaf_polys[f_even],     leaf_polys[f_even + 1]);
        AlmostGoldilocksExt2 f1(leaf_polys[f_even + 2], leaf_polys[f_even + 3]);

        // c = 0
        acc0 = aext2_add(acc0, aext2_mul(e0, f0));
        // c = 1
        acc1 = aext2_add(acc1, aext2_mul(e1, f1));
        // c = 2: (2*e1 - e0) * (2*f1 - f0)
        AlmostGoldilocksExt2 e2 = aext2_sub(aext2_add(e1, e1), e0);
        AlmostGoldilocksExt2 f2 = aext2_sub(aext2_add(f1, f1), f0);
        acc2 = aext2_add(acc2, aext2_mul(e2, f2));
    }

    int base0 = 0 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    int base1 = 1 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    int base2 = 2 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    shared[base0]     = acc0.c[0].value;  shared[base0 + 1] = acc0.c[1].value;
    shared[base1]     = acc1.c[0].value;  shared[base1 + 1] = acc1.c[1].value;
    shared[base2]     = acc2.c[0].value;  shared[base2 + 1] = acc2.c[1].value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < 3; c++) {
                int off = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
                AlmostGoldilocksExt2 a(shared[off + tid * 2], shared[off + tid * 2 + 1]);
                AlmostGoldilocksExt2 b(shared[off + (tid + s) * 2], shared[off + (tid + s) * 2 + 1]);
                AlmostGoldilocksExt2 sum = aext2_add(a, b);
                shared[off + tid * 2]     = sum.c[0].value;
                shared[off + tid * 2 + 1] = sum.c[1].value;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Per-block layout: [block_x][leaf][c][c0, c1]
        size_t out_base = ((size_t)blockIdx.x * num_leaves + leaf) * 3 * 2;
        for (int c = 0; c < 3; c++) {
            int off = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
            d_partial[out_base + c * 2]     = shared[off];
            d_partial[out_base + c * 2 + 1] = shared[off + 1];
        }
    }
}

__global__ void aext2_sumcheck_batched_fold_kernel(
    const uint64_t* __restrict__ d_input,
    uint64_t* __restrict__ d_output,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    size_t original_size,
    size_t half,
    int num_leaves
) {
    int leaf = blockIdx.y;
    if (leaf >= num_leaves) return;

    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    AlmostGoldilocksExt2 ch(challenge_c0, challenge_c1);
    const size_t poly_stride_u64 = original_size * 2;
    const size_t leaf_stride_u64 = 2 * poly_stride_u64;
    const uint64_t* in_leaf  = d_input  + (size_t)leaf * leaf_stride_u64;
    uint64_t*       out_leaf = d_output + (size_t)leaf * leaf_stride_u64;

    for (size_t y = idx; y < half; y += grid_size) {
        for (int i = 0; i < 2; i++) {
            size_t base = i * poly_stride_u64;
            size_t e_off = base + 4 * y;
            AlmostGoldilocksExt2 a(in_leaf[e_off],     in_leaf[e_off + 1]);
            AlmostGoldilocksExt2 b(in_leaf[e_off + 2], in_leaf[e_off + 3]);
            AlmostGoldilocksExt2 r = aext2_add(a, aext2_mul(ch, aext2_sub(b, a)));
            out_leaf[base + y * 2]     = r.c[0].value;
            out_leaf[base + y * 2 + 1] = r.c[1].value;
        }
    }
}

// ============================================================================
// Shared-eq batched same-point sumcheck.
//
// In a fold-tree group the leaves share their claim_pt: the 21 bit-planes of
// one dense edge have ONE extended_point (so a 63-leaf L0 group of ~3 edges
// has ~3 unique eq tables), and every leaf at level 1+ shares the previous
// level's `shared_r` (1 unique). The original interleaved layout stored and
// FOLDED eq once per leaf (63x), pure redundancy. Here eq lives in `d_eq`
// (num_unique tables) and f in `d_f` (num_leaves); the sumcheck folds eq only
// `num_unique` times and reads it via a leaf->unique map. Cuts both the
// dense-Ext2 eq fold work and ~num_unique/num_leaves of the eq storage.
//
// Layouts: eq table u at `d_eq + u*poly_stride_u64`; f table leaf at
// `d_f + leaf*poly_stride_u64`; poly_stride_u64 = original_size*2 (stays full
// even as the active region halves each round). Folds the LOW variable: pairs
// adjacent elements (2y, 2y+1).
// ============================================================================

// Build the per-unique combined f-poly F_u[x] = Σ_{leaf : unique[leaf]==u}
// α_leaf · f_leaf[x], where f_leaf is binary (packed bits). This pre-applies
// the same-point α-weights so the sumcheck runs only num_unique-wide (the
// combined round message is then just Σ_u of the per-unique eq_u·F_u msgs,
// no further α).
//
// grid.y = num_unique. The host pre-sorts leaf indices by unique:
// `d_leaf_idx_sorted[d_unique_offsets[u] .. d_unique_offsets[u+1]]` are the
// leaves mapped to u, so each (x,u) thread loops ONLY its own unique's
// leaves — no scan over all leaves, no divergent `continue` branch. Adjacent
// threads share each leaf's packed word (64 consecutive x per word), so the
// inner loads hit L1.
__global__ void aext2_build_fu_kernel(
    const uint64_t* __restrict__ d_packed,         // num_leaves × packed_size_u64
    const uint64_t* __restrict__ d_alphas,         // num_leaves Ext2 (c0,c1)
    const int* __restrict__ d_leaf_idx_sorted,     // num_leaves, grouped by unique
    const int* __restrict__ d_unique_offsets,      // num_unique + 1
    uint64_t* __restrict__ d_Fu,                   // num_unique × original_size*2
    size_t original_size,
    size_t packed_size_u64
) {
    int u = blockIdx.y;
    int lo = d_unique_offsets[u];
    int hi = d_unique_offsets[u + 1];
    size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;
    for (; x < original_size; x += grid_size) {
        AlmostGoldilocksExt2 acc(0ULL, 0ULL);
        size_t word_idx = x >> 6;
        uint64_t bitmask = 1ULL << (x & 63);
        for (int s = lo; s < hi; s++) {
            int leaf = d_leaf_idx_sorted[s];
            uint64_t word = d_packed[(size_t)leaf * packed_size_u64 + word_idx];
            if (word & bitmask) {
                AlmostGoldilocksExt2 a(d_alphas[(size_t)leaf * 2], d_alphas[(size_t)leaf * 2 + 1]);
                acc = aext2_add(acc, a);
            }
        }
        d_Fu[(size_t)u * original_size * 2 + x * 2]     = acc.c[0].value;
        d_Fu[(size_t)u * original_size * 2 + x * 2 + 1] = acc.c[1].value;
    }
}

// Ternary variant of build_fu: F_u[x] = Σ_{leaf : unique[leaf]==u} α_leaf ·
// (pos_leaf[x] - neg_leaf[x]), pos/neg packed bits (single-chunk ternary).
// Same sorted-leaf-range layout as the binary variant above.
__global__ void aext2_build_fu_ternary_kernel(
    const uint64_t* __restrict__ d_pos,            // num_leaves × packed_size_u64
    const uint64_t* __restrict__ d_neg,
    const uint64_t* __restrict__ d_alphas,         // num_leaves Ext2
    const int* __restrict__ d_leaf_idx_sorted,     // num_leaves, grouped by unique
    const int* __restrict__ d_unique_offsets,      // num_unique + 1
    uint64_t* __restrict__ d_Fu,                   // num_unique × original_size*2
    size_t original_size,
    size_t packed_size_u64
) {
    int u = blockIdx.y;
    int lo = d_unique_offsets[u];
    int hi = d_unique_offsets[u + 1];
    size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;
    for (; x < original_size; x += grid_size) {
        AlmostGoldilocksExt2 acc(0ULL, 0ULL);
        size_t word_idx = x >> 6;
        uint64_t bitmask = 1ULL << (x & 63);
        for (int s = lo; s < hi; s++) {
            int leaf = d_leaf_idx_sorted[s];
            size_t off = (size_t)leaf * packed_size_u64 + word_idx;
            uint64_t pb = d_pos[off] & bitmask;
            uint64_t nb = d_neg[off] & bitmask;
            if (pb | nb) {
                AlmostGoldilocksExt2 a(d_alphas[(size_t)leaf * 2], d_alphas[(size_t)leaf * 2 + 1]);
                if (pb) acc = aext2_add(acc, a);
                if (nb) acc = aext2_sub(acc, a);
            }
        }
        d_Fu[(size_t)u * original_size * 2 + x * 2]     = acc.c[0].value;
        d_Fu[(size_t)u * original_size * 2 + x * 2 + 1] = acc.c[1].value;
    }
}

// ============================================================================
// Device-side split decomposition: wide i16 multifold output → k_chunks
// signed-binary (pos/neg bit-plane) ternary chunks. Replicates the host
// encode exactly: v < 0 → bits of |v| land in the neg planes, else pos.
// One thread per ring word j: reads 64 i16 coefficients, writes word j of
// every chunk plane. |v| ≥ 2^k_chunks (norm-bound violation) sets the
// error flag instead of asserting.
// ============================================================================
__global__ void aext2_wide_to_ternary_kernel(
    const int16_t* __restrict__ d_wide,   // n_ring × 64
    uint64_t* __restrict__ d_pos,         // k_chunks × n_ring
    uint64_t* __restrict__ d_neg,
    int* __restrict__ d_err,
    size_t n_ring,
    int k_chunks
) {
    size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;
    for (; j < n_ring; j += grid_size) {
        uint64_t p[16];
        uint64_t n[16];
        for (int i = 0; i < k_chunks; i++) { p[i] = 0; n[i] = 0; }
        for (int k = 0; k < 64; k++) {
            int16_t v = d_wide[j * 64 + k];
            bool negv = v < 0;
            uint32_t a = negv ? (uint32_t)(-(int32_t)v) : (uint32_t)v;
            if (a >> k_chunks) { atomicExch(d_err, 1); }
            uint64_t bit = 1ULL << k;
            for (int i = 0; i < k_chunks; i++) {
                if ((a >> i) & 1u) {
                    if (negv) n[i] |= bit; else p[i] |= bit;
                }
            }
        }
        for (int i = 0; i < k_chunks; i++) {
            d_pos[(size_t)i * n_ring + j] = p[i];
            d_neg[(size_t)i * n_ring + j] = n[i];
        }
    }
}

// ============================================================================
// Factored-eq (Gruen) shared-eq backend.
//
// eq(R, x) factorizes per variable, so the folded eq table at round t is
// always `p_t · eqsuf_t[y]` where p_t = Π_{i<t} eq1(R_i, r_i) is a HOST
// scalar and eqsuf_t[y] = Π_{i≥t} eq1(R_i, y_{i-t}) is challenge-
// independent. So eq is never materialized at full size and never folded:
// we precompute ALL suffix stages once (total 2^n − 1 elements per unique,
// half of the old eq+scratch footprint) and each round reads the right
// stage. Round messages are mathematically identical to the fold-based
// path: T(0) = p(1−R_t)·A, T(1) = p·R_t·B, T(2) = p(3R_t−1)(2B−A) with
// A = Σ_y eqsuf[y]·F[2y], B = Σ_y eqsuf[y]·F[2y+1].
//
// Suffix-stage layout per unique (stride eqsuf_stride_u64 = 2^n Ext2):
// eqsuf_t lives at element offset 2^{n−t} − 1 with 2^{n−t} elements
// (t = n..1, built smallest-first); round t reads eqsuf_{t+1} at offset
// 2^{n−t−1} − 1, length half = 2^{n−t−1}.
// ============================================================================

// One dp layer: eqsuf_t[x_t + 2y] = eq1(R_t, x_t) · eqsuf_{t+1}[y]. The new
// variable becomes the LOW index bit (interleave write), matching the
// sumcheck's low-variable fold order. Each thread writes one adjacent
// (2i, 2i+1) Ext2 pair — coalesced.
__global__ void aext2_eq_suffix_layer_kernel(
    const uint64_t* __restrict__ d_r_all,    // num_unique × log_n Ext2
    uint64_t* __restrict__ d_eqsuf,          // num_unique × eqsuf_stride_u64
    int t,
    int log_n,
    size_t in_off_elems,
    size_t out_off_elems,
    size_t in_size,
    size_t eqsuf_stride_u64
) {
    int u = blockIdx.y;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;
    AlmostGoldilocksExt2 r(d_r_all[((size_t)u * log_n + t) * 2],
                           d_r_all[((size_t)u * log_n + t) * 2 + 1]);
    uint64_t* base = d_eqsuf + (size_t)u * eqsuf_stride_u64;
    const uint64_t* in  = base + in_off_elems * 2;
    uint64_t*       out = base + out_off_elems * 2;
    for (; i < in_size; i += grid_size) {
        AlmostGoldilocksExt2 a(in[2 * i], in[2 * i + 1]);
        AlmostGoldilocksExt2 ar = aext2_mul(a, r);
        AlmostGoldilocksExt2 lo = aext2_sub(a, ar);
        out[4 * i]     = lo.c[0].value;
        out[4 * i + 1] = lo.c[1].value;
        out[4 * i + 2] = ar.c[0].value;
        out[4 * i + 3] = ar.c[1].value;
    }
}

// Init: eqsuf_n = [1] (empty suffix) at offset 0 of each unique's region.
__global__ void aext2_eq_suffix_init_kernel(
    uint64_t* __restrict__ d_eqsuf,
    size_t eqsuf_stride_u64
) {
    int u = blockIdx.x;
    d_eqsuf[(size_t)u * eqsuf_stride_u64]     = 1ULL;
    d_eqsuf[(size_t)u * eqsuf_stride_u64 + 1] = 0ULL;
}

// Per-round partial sums A_u = Σ_y eqsuf[y]·F_u[2y], B_u = Σ_y eqsuf[y]·
// F_u[2y+1]. The host turns (A,B) into the degree-2 message [T(0),T(1),T(2)]
// with the prefix scalar p_u and R_{u,t}. Reads 1.5·S elements per round vs
// the fold-based kernel's 2·S, and there is no eq fold at all.
__global__ void aext2_sharedeq_factored_msg_kernel(
    const uint64_t* __restrict__ d_eqsuf,
    const uint64_t* __restrict__ d_fu,
    uint64_t* __restrict__ d_partial,        // blocks_x × num_unique × 2 Ext2
    size_t eqsuf_off_elems,
    size_t eqsuf_stride_u64,
    size_t poly_stride_u64,
    size_t half,
    int num_unique
) {
    int u = blockIdx.y;
    if (u >= num_unique) return;
    int tid = threadIdx.x;
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    const uint64_t* eqs = d_eqsuf + (size_t)u * eqsuf_stride_u64 + eqsuf_off_elems * 2;
    const uint64_t* f   = d_fu    + (size_t)u * poly_stride_u64;

    __shared__ uint64_t shared[2 * AGL_SUMCHECK_BLOCK_SIZE * 2];

    AlmostGoldilocksExt2 accA, accB;
    for (size_t y = idx; y < half; y += grid_size) {
        AlmostGoldilocksExt2 e(eqs[2 * y], eqs[2 * y + 1]);
        size_t o = 4 * y;
        AlmostGoldilocksExt2 f0(f[o],     f[o + 1]);
        AlmostGoldilocksExt2 f1(f[o + 2], f[o + 3]);
        accA = aext2_add(accA, aext2_mul(e, f0));
        accB = aext2_add(accB, aext2_mul(e, f1));
    }

    int baseA = 0 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    int baseB = 1 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    shared[baseA]     = accA.c[0].value;  shared[baseA + 1] = accA.c[1].value;
    shared[baseB]     = accB.c[0].value;  shared[baseB + 1] = accB.c[1].value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < 2; c++) {
                int off = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
                AlmostGoldilocksExt2 a(shared[off + tid * 2], shared[off + tid * 2 + 1]);
                AlmostGoldilocksExt2 b(shared[off + (tid + s) * 2], shared[off + (tid + s) * 2 + 1]);
                AlmostGoldilocksExt2 sum = aext2_add(a, b);
                shared[off + tid * 2]     = sum.c[0].value;
                shared[off + tid * 2 + 1] = sum.c[1].value;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        size_t out_base = ((size_t)blockIdx.x * num_unique + u) * 2 * 2;
        for (int c = 0; c < 2; c++) {
            int off = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
            d_partial[out_base + c * 2]     = shared[off];
            d_partial[out_base + c * 2 + 1] = shared[off + 1];
        }
    }
}

__global__ void aext2_sharedeq_msg_kernel(
    const uint64_t* __restrict__ d_eq,
    const uint64_t* __restrict__ d_f,
    const int* __restrict__ d_leaf_to_unique,
    uint64_t* __restrict__ d_partial,
    size_t original_size,
    size_t half,
    int num_leaves
) {
    int leaf = blockIdx.y;
    if (leaf >= num_leaves) return;

    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    const size_t poly_stride_u64 = original_size * 2;
    const uint64_t* eq = d_eq + (size_t)d_leaf_to_unique[leaf] * poly_stride_u64;
    const uint64_t* f  = d_f  + (size_t)leaf * poly_stride_u64;

    __shared__ uint64_t shared[3 * AGL_SUMCHECK_BLOCK_SIZE * 2];

    AlmostGoldilocksExt2 acc0, acc1, acc2;

    for (size_t y = idx; y < half; y += grid_size) {
        size_t o = 4 * y;  // elements 2y (lo), 2y+1 (hi)
        AlmostGoldilocksExt2 e0(eq[o],     eq[o + 1]);
        AlmostGoldilocksExt2 e1(eq[o + 2], eq[o + 3]);
        AlmostGoldilocksExt2 f0(f[o],      f[o + 1]);
        AlmostGoldilocksExt2 f1(f[o + 2],  f[o + 3]);
        acc0 = aext2_add(acc0, aext2_mul(e0, f0));
        acc1 = aext2_add(acc1, aext2_mul(e1, f1));
        AlmostGoldilocksExt2 e2 = aext2_sub(aext2_add(e1, e1), e0);
        AlmostGoldilocksExt2 f2 = aext2_sub(aext2_add(f1, f1), f0);
        acc2 = aext2_add(acc2, aext2_mul(e2, f2));
    }

    int base0 = 0 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    int base1 = 1 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    int base2 = 2 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    shared[base0]     = acc0.c[0].value;  shared[base0 + 1] = acc0.c[1].value;
    shared[base1]     = acc1.c[0].value;  shared[base1 + 1] = acc1.c[1].value;
    shared[base2]     = acc2.c[0].value;  shared[base2 + 1] = acc2.c[1].value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < 3; c++) {
                int off = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
                AlmostGoldilocksExt2 a(shared[off + tid * 2], shared[off + tid * 2 + 1]);
                AlmostGoldilocksExt2 b(shared[off + (tid + s) * 2], shared[off + (tid + s) * 2 + 1]);
                AlmostGoldilocksExt2 sum = aext2_add(a, b);
                shared[off + tid * 2]     = sum.c[0].value;
                shared[off + tid * 2 + 1] = sum.c[1].value;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        size_t out_base = ((size_t)blockIdx.x * num_leaves + leaf) * 3 * 2;
        for (int c = 0; c < 3; c++) {
            int off = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
            d_partial[out_base + c * 2]     = shared[off];
            d_partial[out_base + c * 2 + 1] = shared[off + 1];
        }
    }
}

// Fold `count` independent Ext2 polynomials (each `original_size` long,
// `poly_stride_u64 = original_size*2`) by one challenge. Used for eq
// (count = num_unique) and f (count = num_leaves) separately.
__global__ void aext2_fold_single_kernel(
    const uint64_t* __restrict__ d_in,
    uint64_t* __restrict__ d_out,
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    size_t original_size,
    size_t half,
    int count
) {
    int p = blockIdx.y;
    if (p >= count) return;

    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    AlmostGoldilocksExt2 ch(challenge_c0, challenge_c1);
    const size_t poly_stride_u64 = original_size * 2;
    const uint64_t* in_p  = d_in  + (size_t)p * poly_stride_u64;
    uint64_t*       out_p = d_out + (size_t)p * poly_stride_u64;

    for (size_t y = idx; y < half; y += grid_size) {
        size_t o = 4 * y;
        AlmostGoldilocksExt2 a(in_p[o],     in_p[o + 1]);
        AlmostGoldilocksExt2 b(in_p[o + 2], in_p[o + 3]);
        AlmostGoldilocksExt2 r = aext2_add(a, aext2_mul(ch, aext2_sub(b, a)));
        out_p[y * 2]     = r.c[0].value;
        out_p[y * 2 + 1] = r.c[1].value;
    }
}

// ============================================================================
// Binary round-0 fused kernels for the batched same-point sumcheck.
//
// At round 0 the f-poly is still binary (packed bits), so we can compute the
// degree-2 message and the fold WITHOUT lifting f to Ext2 (skipping the
// `2^arity` lift write and replacing the f-side Ext2 muls with selective
// add/sub/double). The eq-poly is dense Ext2 (lives in d_polys' eq-slot) and
// is handled normally. After the fold the output holds Ext2 eq'/f' at half
// size, so rounds 1+ use the standard kernels.
//
// The sumcheck folds the LOW variable: it pairs adjacent elements (2y, 2y+1).
// Since 2y is even, bits 2y and 2y+1 always live in the same packed word.
// ============================================================================

__global__ void aext2_sumcheck_batched_round0_binary_msg_kernel(
    const uint64_t* __restrict__ d_polys,   // eq-slot used; f-slot ignored
    const uint64_t* __restrict__ d_packed,  // binary f bits
    uint64_t* __restrict__ d_partial,
    size_t original_size,
    size_t half,
    int num_leaves,
    size_t packed_size_u64
) {
    int leaf = blockIdx.y;
    if (leaf >= num_leaves) return;

    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    const size_t poly_stride_u64 = original_size * 2;
    const size_t leaf_stride_u64 = 2 * poly_stride_u64;
    const uint64_t* leaf_eq = d_polys + (size_t)leaf * leaf_stride_u64; // eq slot (offset 0)
    const uint64_t* leaf_packed = d_packed + (size_t)leaf * packed_size_u64;

    __shared__ uint64_t shared[3 * AGL_SUMCHECK_BLOCK_SIZE * 2];

    AlmostGoldilocksExt2 acc0, acc1, acc2;

    for (size_t y = idx; y < half; y += grid_size) {
        size_t e_even = 4 * y;  // eq elements 2y (lo) and 2y+1 (hi)
        AlmostGoldilocksExt2 e0(leaf_eq[e_even],     leaf_eq[e_even + 1]);
        AlmostGoldilocksExt2 e1(leaf_eq[e_even + 2], leaf_eq[e_even + 3]);

        size_t bit_lo = 2 * y;
        uint64_t word = leaf_packed[bit_lo >> 6];
        uint64_t f0 = (word >> (bit_lo & 63)) & 1ULL;
        uint64_t f1 = (word >> ((bit_lo + 1) & 63)) & 1ULL;

        // c=0: e0*f0 ; c=1: e1*f1  (selective add).
        if (f0) acc0 = aext2_add(acc0, e0);
        if (f1) acc1 = aext2_add(acc1, e1);
        // c=2: (2e1-e0) * (2f1-f0), with 2f1-f0 ∈ {-1,0,1,2}.
        int f2 = 2 * (int)f1 - (int)f0;
        if (f2 != 0) {
            AlmostGoldilocksExt2 e2 = aext2_sub(aext2_add(e1, e1), e0);
            if (f2 == 1)       acc2 = aext2_add(acc2, e2);
            else if (f2 == 2)  acc2 = aext2_add(acc2, aext2_add(e2, e2));
            else /* f2 == -1 */ acc2 = aext2_sub(acc2, e2);
        }
    }

    int base0 = 0 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    int base1 = 1 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    int base2 = 2 * AGL_SUMCHECK_BLOCK_SIZE * 2 + tid * 2;
    shared[base0]     = acc0.c[0].value;  shared[base0 + 1] = acc0.c[1].value;
    shared[base1]     = acc1.c[0].value;  shared[base1 + 1] = acc1.c[1].value;
    shared[base2]     = acc2.c[0].value;  shared[base2 + 1] = acc2.c[1].value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < 3; c++) {
                int off = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
                AlmostGoldilocksExt2 a(shared[off + tid * 2], shared[off + tid * 2 + 1]);
                AlmostGoldilocksExt2 b(shared[off + (tid + s) * 2], shared[off + (tid + s) * 2 + 1]);
                AlmostGoldilocksExt2 sum = aext2_add(a, b);
                shared[off + tid * 2]     = sum.c[0].value;
                shared[off + tid * 2 + 1] = sum.c[1].value;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        size_t out_base = ((size_t)blockIdx.x * num_leaves + leaf) * 3 * 2;
        for (int c = 0; c < 3; c++) {
            int off = c * AGL_SUMCHECK_BLOCK_SIZE * 2;
            d_partial[out_base + c * 2]     = shared[off];
            d_partial[out_base + c * 2 + 1] = shared[off + 1];
        }
    }
}

__global__ void aext2_sumcheck_batched_round0_binary_fold_kernel(
    const uint64_t* __restrict__ d_polys,   // input eq-slot; f-slot ignored
    const uint64_t* __restrict__ d_packed,  // binary f bits
    uint64_t* __restrict__ d_output,        // eq' + f' (Ext2) at half size
    uint64_t challenge_c0,
    uint64_t challenge_c1,
    size_t original_size,
    size_t half,
    int num_leaves,
    size_t packed_size_u64
) {
    int leaf = blockIdx.y;
    if (leaf >= num_leaves) return;

    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    AlmostGoldilocksExt2 ch(challenge_c0, challenge_c1);
    AlmostGoldilocksExt2 one(1ULL, 0ULL);
    const size_t poly_stride_u64 = original_size * 2;
    const size_t leaf_stride_u64 = 2 * poly_stride_u64;
    const uint64_t* in_eq = d_polys + (size_t)leaf * leaf_stride_u64;
    const uint64_t* leaf_packed = d_packed + (size_t)leaf * packed_size_u64;
    uint64_t* out_leaf = d_output + (size_t)leaf * leaf_stride_u64;

    for (size_t y = idx; y < half; y += grid_size) {
        // eq fold (slot 0): eq' = e0 + ch*(e1 - e0).
        size_t e_off = 4 * y;
        AlmostGoldilocksExt2 a(in_eq[e_off],     in_eq[e_off + 1]);
        AlmostGoldilocksExt2 b(in_eq[e_off + 2], in_eq[e_off + 3]);
        AlmostGoldilocksExt2 re = aext2_add(a, aext2_mul(ch, aext2_sub(b, a)));
        out_leaf[y * 2]     = re.c[0].value;
        out_leaf[y * 2 + 1] = re.c[1].value;

        // f fold (slot poly_stride): f' = f0 + ch*(f1 - f0) ∈ {0, ch, 1-ch, 1}.
        size_t bit_lo = 2 * y;
        uint64_t word = leaf_packed[bit_lo >> 6];
        uint64_t f0 = (word >> (bit_lo & 63)) & 1ULL;
        uint64_t f1 = (word >> ((bit_lo + 1) & 63)) & 1ULL;
        AlmostGoldilocksExt2 rf;
        if (f0 && f1)        rf = one;                    // 1
        else if (!f0 && f1)  rf = ch;                     // ch
        else if (f0 && !f1)  rf = aext2_sub(one, ch);     // 1 - ch
        // else both 0 → rf stays 0 (default ctor)
        out_leaf[poly_stride_u64 + y * 2]     = rf.c[0].value;
        out_leaf[poly_stride_u64 + y * 2 + 1] = rf.c[1].value;
    }
}

// ============================================================================
// Batched binary → Ext2 lift, written into the f-slot of the batched
// same-point layout.
//
// Input:  d_packed = [leaf_0_packed | leaf_1_packed | ...]
//         each leaf is `packed_size_u64` u64s (= original_size / 64).
// Output: writes Ext2 values (bit ? 1 : 0, with c1 = 0) into
//         d_polys[leaf][f-slot] = d_polys + leaf*(2*original_size*2) + (original_size*2).
// ============================================================================

// Lift packed binary bits into a CONTIGUOUS per-leaf f layout (for the
// shared-eq state): f for leaf at `d_f + leaf*original_size*2`, element i at
// `[leaf*original_size*2 + i*2] = bit, +1 = 0`. (The non-contig variant below
// targets the interleaved [eq,f] f-slot.)
__global__ void aext2_lift_binary_contig_kernel(
    const uint64_t* __restrict__ d_packed,
    uint64_t* __restrict__ d_f,
    size_t original_size,
    int num_leaves,
    size_t packed_size_u64
) {
    int leaf = blockIdx.y;
    if (leaf >= num_leaves) return;
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;
    const uint64_t* leaf_packed = d_packed + (size_t)leaf * packed_size_u64;
    uint64_t* leaf_f = d_f + (size_t)leaf * original_size * 2;
    for (size_t i = idx; i < original_size; i += grid_size) {
        uint64_t word = leaf_packed[i >> 6];
        uint64_t bit = (word >> (i & 63)) & 1;
        leaf_f[i * 2]     = bit;
        leaf_f[i * 2 + 1] = 0;
    }
}

__global__ void aext2_batched_lift_binary_kernel(
    const uint64_t* __restrict__ d_packed,
    uint64_t* __restrict__ d_polys,
    size_t original_size,
    int num_leaves,
    size_t packed_size_u64
) {
    int leaf = blockIdx.y;
    if (leaf >= num_leaves) return;

    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    const uint64_t* leaf_packed = d_packed + (size_t)leaf * packed_size_u64;
    const size_t leaf_stride_u64 = 2 * original_size * 2;
    const size_t f_offset = (size_t)leaf * leaf_stride_u64 + original_size * 2;
    uint64_t* leaf_f = d_polys + f_offset;

    for (size_t i = idx; i < original_size; i += grid_size) {
        uint64_t word = leaf_packed[i >> 6];
        uint64_t bit = (word >> (i & 63)) & 1;
        leaf_f[i * 2]     = bit;
        leaf_f[i * 2 + 1] = 0;
    }
}

// ============================================================================
// Batched selective-add over a shared eq table.
//
// For each of `n_planes` binary witnesses, computes
//   eval_p = Σ_{i : plane_p[i] = 1} eq[i]
//
// One kernel launch handles ALL planes for a single edge (they share the
// same eq table). Output: per-(block, plane) partial Ext2 sums, reduced
// on host to one Ext2 per plane.
// ============================================================================

__global__ void aext2_selective_add_batched_planes_kernel(
    const uint64_t* __restrict__ d_eq,            // 2 * total u64
    const uint64_t* __restrict__ d_packed_planes, // n_planes * packed_size_u64
    uint64_t* __restrict__ d_partial,             // num_blocks_x * n_planes * 2 u64
    size_t total,
    int n_planes,
    size_t packed_size_u64
) {
    int plane = blockIdx.y;
    if (plane >= n_planes) return;

    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    const uint64_t* plane_packed = d_packed_planes + (size_t)plane * packed_size_u64;
    AlmostGoldilocksExt2 acc;

    for (size_t i = idx; i < total; i += grid_size) {
        uint64_t word = plane_packed[i >> 6];
        if ((word >> (i & 63)) & 1) {
            AlmostGoldilocksExt2 e(d_eq[i * 2], d_eq[i * 2 + 1]);
            acc = aext2_add(acc, e);
        }
    }

    __shared__ uint64_t shared[AGL_SUMCHECK_BLOCK_SIZE * 2];
    shared[tid * 2]     = acc.c[0].value;
    shared[tid * 2 + 1] = acc.c[1].value;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            AlmostGoldilocksExt2 a(shared[tid * 2], shared[tid * 2 + 1]);
            AlmostGoldilocksExt2 b(shared[(tid + s) * 2], shared[(tid + s) * 2 + 1]);
            AlmostGoldilocksExt2 sum = aext2_add(a, b);
            shared[tid * 2]     = sum.c[0].value;
            shared[tid * 2 + 1] = sum.c[1].value;
        }
        __syncthreads();
    }
    if (tid == 0) {
        size_t out_base = ((size_t)blockIdx.x * n_planes + plane) * 2;
        d_partial[out_base]     = shared[0];
        d_partial[out_base + 1] = shared[1];
    }
}

// Single-chunk ternary lift: per leaf, position value = pos_bit - neg_bit
// ∈ {-1, 0, +1}. Encoded as Ext2 with c0 = 0 / 1 / (q-1) and c1 = 0.
__global__ void aext2_batched_lift_ternary_single_kernel(
    const uint64_t* __restrict__ d_pos,
    const uint64_t* __restrict__ d_neg,
    uint64_t* __restrict__ d_polys,
    size_t original_size,
    int num_leaves,
    size_t packed_size_u64
) {
    int leaf = blockIdx.y;
    if (leaf >= num_leaves) return;

    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * (size_t)blockDim.x;

    const uint64_t* leaf_pos = d_pos + (size_t)leaf * packed_size_u64;
    const uint64_t* leaf_neg = d_neg + (size_t)leaf * packed_size_u64;
    const size_t leaf_stride_u64 = 2 * original_size * 2;
    const size_t f_offset = (size_t)leaf * leaf_stride_u64 + original_size * 2;
    uint64_t* leaf_f = d_polys + f_offset;

    const uint64_t neg_one = ALMOST_GOLDILOCKS_PRIME - 1ULL;
    for (size_t i = idx; i < original_size; i += grid_size) {
        uint64_t pw = leaf_pos[i >> 6];
        uint64_t nw = leaf_neg[i >> 6];
        uint64_t pos_bit = (pw >> (i & 63)) & 1ULL;
        uint64_t neg_bit = (nw >> (i & 63)) & 1ULL;
        // Branchless: c0 = pos_bit * 1 + neg_bit * (q-1)
        // (assumes pos & neg are disjoint by construction).
        uint64_t c0 = pos_bit + neg_bit * neg_one;
        // Wrap-reduce if both bits are set (shouldn't happen, but safe).
        if (c0 >= ALMOST_GOLDILOCKS_PRIME) c0 -= ALMOST_GOLDILOCKS_PRIME;
        leaf_f[i * 2]     = c0;
        leaf_f[i * 2 + 1] = 0ULL;
    }
}

#endif // ALMOST_SUMCHECK_PROVER_CUH

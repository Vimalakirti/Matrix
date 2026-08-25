/**
 * CPU reference for the Ajtai commitment.
 *
 * Parametric in the ring dimension `D` so we can validate at `D = 4` and
 * `D = 8` (where output is hand-checkable) before testing the production
 * `D = 64` GPU kernel against it.
 *
 * Provides:
 *   - cpu_ring_shift<D>: negacyclic X^ell · a
 *   - cpu_ring_mul<D>: naive polynomial multiplication mod (X^D + 1)
 *   - cpu_ring_binary_mul<D>: selected-rotation product for binary z
 *   - cpu_prg_ring_elem<D>: ChaCha8 + rejection sampling, scaled to D
 *   - cpu_ajtai_commit<D, KAPPA>: full commitment
 *
 * For D ≤ 64 a "ring element" is the first D entries of a 64-coefficient
 * field array; PRG reuses the same ChaCha8 stream and just truncates.
 */

#ifndef AJTAI_CPU_REFERENCE_CUH
#define AJTAI_CPU_REFERENCE_CUH

#include <stdint.h>
#include <vector>
#include <array>
#include <cassert>
#include "almost_goldilocks.cuh"
#include "ajtai_chacha8.cuh"

namespace ajtai_cpu {

// ============================================================================
// Field helpers (canonical inputs / outputs, host-only)
// ============================================================================

static inline uint64_t h_add(uint64_t a, uint64_t b) {
    __uint128_t s = (__uint128_t)a + (__uint128_t)b;
    return (uint64_t)(s % (__uint128_t)ALMOST_GOLDILOCKS_PRIME);
}
static inline uint64_t h_sub(uint64_t a, uint64_t b) {
    return a >= b ? a - b : ALMOST_GOLDILOCKS_PRIME - (b - a);
}
static inline uint64_t h_neg(uint64_t a) {
    return a == 0 ? 0 : ALMOST_GOLDILOCKS_PRIME - a;
}
static inline uint64_t h_mul(uint64_t a, uint64_t b) {
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)(p % (__uint128_t)ALMOST_GOLDILOCKS_PRIME);
}
static inline uint64_t h_canon(uint64_t v) {
    return v >= ALMOST_GOLDILOCKS_PRIME ? v - ALMOST_GOLDILOCKS_PRIME : v;
}

// ============================================================================
// Ring operations in F_q[X] / (X^D + 1)
// ============================================================================

template <int D>
using Ring = std::array<uint64_t, D>;

/**
 * (X^ell * a)[r]:
 *   = a[r - ell]            if r >= ell
 *   = -a[r - ell + D]       if r < ell
 */
template <int D>
inline Ring<D> cpu_ring_shift(const Ring<D>& a, int ell) {
    Ring<D> out{};
    for (int r = 0; r < D; r++) {
        int idx = r - ell;
        if (idx >= 0) {
            out[r] = a[idx];
        } else {
            out[r] = h_neg(a[idx + D]);
        }
    }
    return out;
}

/**
 * Naive polynomial multiplication mod (X^D + 1): for any two ring elements.
 * O(D^2). Used as ground truth in correctness tests against the
 * selected-rotation method below.
 */
template <int D>
inline Ring<D> cpu_ring_mul(const Ring<D>& a, const Ring<D>& b) {
    Ring<D> out{};
    for (int i = 0; i < D; i++) {
        for (int k = 0; k < D; k++) {
            int r = i + k;
            uint64_t v = h_mul(a[i], b[k]);
            if (r < D) {
                out[r] = h_add(out[r], v);
            } else {
                // X^D = -1: wraps with sign flip
                out[r - D] = h_sub(out[r - D], v);
            }
        }
    }
    return out;
}

/**
 * Selected-rotation product for a single ring element times a binary mask.
 *   a * z_R = Σ_{ell : z_bits has bit ell set} X^ell · a
 * Returns a ring element. For D <= 64 the bitmask uses the low D bits.
 */
template <int D>
inline Ring<D> cpu_ring_binary_mul(const Ring<D>& a, uint64_t z_bits) {
    static_assert(D <= 64, "D must be <= 64 to use a u64 bitmask");
    Ring<D> out{};
    uint64_t mask = z_bits & ((D == 64) ? ~0ULL : ((1ULL << D) - 1));
    while (mask) {
        int ell = __builtin_ctzll(mask);
        mask &= mask - 1;
        Ring<D> shifted = cpu_ring_shift<D>(a, ell);
        for (int r = 0; r < D; r++) {
            out[r] = h_add(out[r], shifted[r]);
        }
    }
    return out;
}

// ============================================================================
// PRG: same as device, just parameterized by D
// ============================================================================

/**
 * Generate M[row, j] for arbitrary D ≤ 64 by sampling from the ChaCha8
 * stream and taking the first D rejection-accepted coefficients.
 *
 * For D = 64 this matches the device-side PRG exactly. For D < 64 we
 * re-key the ChaCha8 stream identically but truncate after D values;
 * this gives a self-consistent reference for small-D tests but is not
 * meant to match what a hypothetical "D=8 GPU kernel" would do.
 */
template <int D>
inline Ring<D> cpu_prg_ring_elem(
    const uint32_t key[8],
    int      row,
    uint64_t j
) {
    Ring<D> out{};
    int written = 0;
    uint32_t block_idx = 0;
    while (written < D) {
        uint64_t buf[8];
        prg_ring_block_chacha8(key, (uint32_t)row, j, block_idx, buf);
        for (int k = 0; k < 8 && written < D; k++) {
            out[written++] = buf[k];
        }
        block_idx++;
    }
    return out;
}

// ============================================================================
// Full Ajtai commitment (CPU reference)
// ============================================================================

/**
 * Single-witness commit: c[i] = Σ_j M[i, j] * z[j] for i in [0, KAPPA).
 *
 * z_bits has length N, each u64 packing D binary coefficients (low bits).
 */
template <int D, int KAPPA>
inline std::vector<Ring<D>> cpu_ajtai_commit(
    const uint32_t key[8],
    const uint64_t* z_bits,
    uint64_t N
) {
    std::vector<Ring<D>> c(KAPPA);
    for (int i = 0; i < KAPPA; i++) c[i].fill(0);

    for (uint64_t j = 0; j < N; j++) {
        uint64_t bits = z_bits[j];
        if (bits == 0) continue;
        for (int i = 0; i < KAPPA; i++) {
            Ring<D> M_ij = cpu_prg_ring_elem<D>(key, i, j);
            Ring<D> contrib = cpu_ring_binary_mul<D>(M_ij, bits);
            for (int r = 0; r < D; r++) {
                c[i][r] = h_add(c[i][r], contrib[r]);
            }
        }
    }
    return c;
}

/**
 * Sparse commit: same as above but iterates a position list.
 * Each position p ∈ [0, N*D) decomposes as (j = p >> log2(D), ell = p & (D-1)).
 */
template <int D, int KAPPA>
inline std::vector<Ring<D>> cpu_ajtai_commit_sparse(
    const uint32_t key[8],
    const uint64_t* positions,
    uint64_t K
) {
    static_assert((D & (D - 1)) == 0, "D must be a power of 2");
    constexpr int LOG_D = (D == 64) ? 6 : ((D == 8) ? 3 : ((D == 4) ? 2 : -1));
    static_assert(LOG_D > 0, "Add LOG_D for this D");
    constexpr uint64_t D_MASK = (uint64_t)(D - 1);

    std::vector<Ring<D>> c(KAPPA);
    for (int i = 0; i < KAPPA; i++) c[i].fill(0);

    for (uint64_t k = 0; k < K; k++) {
        uint64_t p = positions[k];
        uint64_t j = p >> LOG_D;
        int ell = (int)(p & D_MASK);
        for (int i = 0; i < KAPPA; i++) {
            Ring<D> M_ij = cpu_prg_ring_elem<D>(key, i, j);
            Ring<D> shifted = cpu_ring_shift<D>(M_ij, ell);
            for (int r = 0; r < D; r++) {
                c[i][r] = h_add(c[i][r], shifted[r]);
            }
        }
    }
    return c;
}

/**
 * Batched commit reference: identical to running cpu_ajtai_commit B
 * times with B independent witnesses. Returned as a flat
 * [B][KAPPA][D] vector.
 */
template <int D, int KAPPA>
inline std::vector<Ring<D>> cpu_ajtai_commit_batched(
    const uint32_t key[8],
    const uint64_t* z_bits_packed,  // [B * N], row-major over batch
    uint64_t N,
    int B
) {
    std::vector<Ring<D>> result((size_t)B * KAPPA);
    for (int b = 0; b < B; b++) {
        auto sub = cpu_ajtai_commit<D, KAPPA>(key, z_bits_packed + (size_t)b * N, N);
        for (int i = 0; i < KAPPA; i++) {
            result[(size_t)b * KAPPA + i] = sub[i];
        }
    }
    return result;
}

} // namespace ajtai_cpu

#endif // AJTAI_CPU_REFERENCE_CUH

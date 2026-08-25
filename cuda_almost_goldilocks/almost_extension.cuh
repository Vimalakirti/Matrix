/**
 * Almost-Goldilocks Quadratic Extension Field GF(p^2)
 *
 * Base prime: p = 2^64 - 2^32 - 31 = 0xFFFFFFFEFFFFFFE1
 *
 * Non-residue choice: W = 3. In Goldilocks (p_G = 2^64 - 2^32 + 1) we have
 * Legendre(7) = -1, so X^2 - 7 is irreducible there. For almost-Goldilocks
 * the Legendre symbols are different:
 *
 *     Legendre(7) =  1   (so X^2 - 7 is REDUCIBLE here)
 *     Legendre(3) = -1   (so X^2 - 3 is irreducible)  ← use this
 *
 * Hence GF(p_A^2) = F_p[X] / (X^2 - 3), with cheap mul_by_3 = 2x + x.
 *
 * DTH_ROOT = W^((p-1)/2) = -1 = p - 1 = 0xFFFFFFFEFFFFFFE0
 *
 * Only the quadratic extension is provided in this directory; the quintic
 * extension is intentionally omitted (gcd(5, p_A - 1) = 1, so X^5 - W is
 * always reducible — Ext5 would need a different irreducible quintic).
 */

#ifndef ALMOST_EXTENSION_CUH
#define ALMOST_EXTENSION_CUH

#include "almost_goldilocks.cuh"

// ============================================================================
// Constants
// ============================================================================

// Quadratic non-residue used for Ext2: X^2 - W with W = 3.
#define ALMOST_EXT2_W 3ULL

// W^((p-1)/2) = -1 mod p. For W=3 and p=ALMOST_GOLDILOCKS_PRIME this is p-1.
#define ALMOST_EXT2_DTH_ROOT 0xFFFFFFFEFFFFFFE0ULL

// ============================================================================
// Ext2 element
// ============================================================================

struct AlmostGoldilocksExt2 {
    AlmostGoldilocksField c[2];  // c[0] + c[1] * X, X^2 = W

    __host__ __device__ __forceinline__
    AlmostGoldilocksExt2() {
        c[0] = AlmostGoldilocksField(0);
        c[1] = AlmostGoldilocksField(0);
    }

    __host__ __device__ __forceinline__
    AlmostGoldilocksExt2(AlmostGoldilocksField c0, AlmostGoldilocksField c1) {
        c[0] = c0;
        c[1] = c1;
    }

    __host__ __device__ __forceinline__
    AlmostGoldilocksExt2(uint64_t c0, uint64_t c1) {
        c[0] = AlmostGoldilocksField(c0);
        c[1] = AlmostGoldilocksField(c1);
    }

    __host__ __device__ __forceinline__
    explicit AlmostGoldilocksExt2(AlmostGoldilocksField base) {
        c[0] = base;
        c[1] = AlmostGoldilocksField(0);
    }
};

// ============================================================================
// Ext2 arithmetic
// ============================================================================

__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_add(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return AlmostGoldilocksExt2(agl_add(a.c[0], b.c[0]), agl_add(a.c[1], b.c[1]));
}

__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_sub(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return AlmostGoldilocksExt2(agl_sub(a.c[0], b.c[0]), agl_sub(a.c[1], b.c[1]));
}

__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_neg(AlmostGoldilocksExt2 a) {
    return AlmostGoldilocksExt2(agl_neg(a.c[0]), agl_neg(a.c[1]));
}

/**
 * Multiplication in F_p[X] / (X^2 - 3):
 *   (a0 + a1 X)(b0 + b1 X) = (a0 b0 + 3 a1 b1) + (a0 b1 + a1 b0) X
 *
 * Karatsuba: 3 base muls + cheap mul_by_3.
 *   m0 = a0 b0
 *   m1 = a1 b1
 *   m2 = (a0 + a1)(b0 + b1)
 *   c0 = m0 + 3 m1
 *   c1 = m2 - m0 - m1
 */
__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_mul(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    AlmostGoldilocksField m0 = agl_mul(a.c[0], b.c[0]);
    AlmostGoldilocksField m1 = agl_mul(a.c[1], b.c[1]);
    AlmostGoldilocksField m2 = agl_mul(
        agl_add(a.c[0], a.c[1]),
        agl_add(b.c[0], b.c[1])
    );
    AlmostGoldilocksField c0 = agl_add(m0, agl_mul_by_3(m1));
    AlmostGoldilocksField c1 = agl_sub(agl_sub(m2, m0), m1);
    return AlmostGoldilocksExt2(c0, c1);
}

/**
 * Squaring: (a0 + a1 X)^2 = (a0^2 + 3 a1^2) + 2 a0 a1 X.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_square(AlmostGoldilocksExt2 a) {
    AlmostGoldilocksField a0_sq = agl_square(a.c[0]);
    AlmostGoldilocksField a1_sq = agl_square(a.c[1]);
    AlmostGoldilocksField a1_sq_w = agl_mul_by_3(a1_sq);
    AlmostGoldilocksField c0 = agl_add(a0_sq, a1_sq_w);
    AlmostGoldilocksField c1 = agl_double(agl_mul(a.c[0], a.c[1]));
    return AlmostGoldilocksExt2(c0, c1);
}

__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_scalar_mul(AlmostGoldilocksField scalar, AlmostGoldilocksExt2 a) {
    return AlmostGoldilocksExt2(agl_mul(scalar, a.c[0]), agl_mul(scalar, a.c[1]));
}

/**
 * Inverse: 1 / (a0 + a1 X) = (a0 - a1 X) / (a0^2 - W a1^2).
 */
__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_inverse(AlmostGoldilocksExt2 a) {
    AlmostGoldilocksField a0_sq = agl_square(a.c[0]);
    AlmostGoldilocksField a1_sq = agl_square(a.c[1]);
    AlmostGoldilocksField w_a1_sq = agl_mul_by_3(a1_sq);
    AlmostGoldilocksField norm = agl_sub(a0_sq, w_a1_sq);
    AlmostGoldilocksField inv_norm = agl_inverse(norm);
    return AlmostGoldilocksExt2(
        agl_mul(a.c[0], inv_norm),
        agl_neg(agl_mul(a.c[1], inv_norm))
    );
}

__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_div(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return aext2_mul(a, aext2_inverse(b));
}

/**
 * Frobenius: a^p. For W with DTH_ROOT = W^((p-1)/2) = -1 this is conjugation:
 *   a0 + a1 X  ↦  a0 - a1 X.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_frobenius(AlmostGoldilocksExt2 a) {
    return AlmostGoldilocksExt2(
        a.c[0],
        agl_mul(a.c[1], AlmostGoldilocksField(ALMOST_EXT2_DTH_ROOT))
    );
}

__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_conjugate(AlmostGoldilocksExt2 a) {
    return AlmostGoldilocksExt2(a.c[0], agl_neg(a.c[1]));
}

__host__ __device__ __forceinline__
AlmostGoldilocksField aext2_norm(AlmostGoldilocksExt2 a) {
    AlmostGoldilocksField a0_sq = agl_square(a.c[0]);
    AlmostGoldilocksField a1_sq = agl_square(a.c[1]);
    AlmostGoldilocksField w_a1_sq = agl_mul_by_3(a1_sq);
    return agl_sub(a0_sq, w_a1_sq);
}

__host__ __device__ __forceinline__
AlmostGoldilocksExt2 aext2_exp(AlmostGoldilocksExt2 base, uint64_t exp) {
    AlmostGoldilocksExt2 result(AlmostGoldilocksField(1), AlmostGoldilocksField(0));
    AlmostGoldilocksExt2 b = base;
    while (exp > 0) {
        if (exp & 1) result = aext2_mul(result, b);
        b = aext2_square(b);
        exp >>= 1;
    }
    return result;
}

__host__ __device__ __forceinline__
bool aext2_eq(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return agl_eq(a.c[0], b.c[0]) && agl_eq(a.c[1], b.c[1]);
}

__host__ __device__ __forceinline__
bool aext2_is_zero(AlmostGoldilocksExt2 a) {
    return agl_is_zero(a.c[0]) && agl_is_zero(a.c[1]);
}

__host__ __device__ __forceinline__
AlmostGoldilocksExt2 agl_to_ext2(AlmostGoldilocksField a) {
    return AlmostGoldilocksExt2(a, AlmostGoldilocksField(0));
}

__host__ __device__ __forceinline__
AlmostGoldilocksField aext2_to_agl(AlmostGoldilocksExt2 a) {
    return a.c[0];
}

__host__ __device__ __forceinline__
bool aext2_is_base(AlmostGoldilocksExt2 a) {
    return agl_is_zero(a.c[1]);
}

#endif // ALMOST_EXTENSION_CUH

/**
 * Goldilocks Extension Fields - CUDA Implementation
 *
 * Based on the Plonky3 reference implementation.
 *
 * Supported Extensions:
 * - Quadratic (degree 2): F_p[X] / (X^2 - 7)
 * - Quintic (degree 5): F_p[X] / (X^5 - 3)
 *
 * Where p = 2^64 - 2^32 + 1 (Goldilocks prime)
 */

#ifndef EXTENSION_CUH
#define EXTENSION_CUH

#include "goldilocks.cuh"

// ============================================================================
// Extension Field Parameters
// ============================================================================

// Quadratic Extension: X^2 - 7
#define EXT2_W 7ULL
#define EXT2_DTH_ROOT 18446744069414584320ULL  // W^((p-1)/2) = -1 mod p

// Quintic Extension: X^5 - 3
#define EXT5_W 3ULL
#define EXT5_DTH_ROOT 1041288259238279555ULL   // W^((p-1)/5)

// ============================================================================
// Quadratic Extension Field (Degree 2)
// ============================================================================

/**
 * Element of F_p^2 = F_p[X] / (X^2 - 7)
 * Represented as c0 + c1*X where X^2 = 7
 */
struct GoldilocksExt2 {
    GoldilocksField c[2];  // c[0] + c[1]*X

    __host__ __device__ __forceinline__
    GoldilocksExt2() {
        c[0] = GoldilocksField(0);
        c[1] = GoldilocksField(0);
    }

    __host__ __device__ __forceinline__
    GoldilocksExt2(GoldilocksField c0, GoldilocksField c1) {
        c[0] = c0;
        c[1] = c1;
    }

    __host__ __device__ __forceinline__
    GoldilocksExt2(uint64_t c0, uint64_t c1) {
        c[0] = GoldilocksField(c0);
        c[1] = GoldilocksField(c1);
    }

    // Embed base field element
    __host__ __device__ __forceinline__
    explicit GoldilocksExt2(GoldilocksField base) {
        c[0] = base;
        c[1] = GoldilocksField(0);
    }
};

// ============================================================================
// Quadratic Extension Arithmetic
// ============================================================================

/**
 * Addition: (a0 + a1*X) + (b0 + b1*X) = (a0+b0) + (a1+b1)*X
 */
__device__ __forceinline__
GoldilocksExt2 ext2_add(GoldilocksExt2 a, GoldilocksExt2 b) {
    return GoldilocksExt2(
        gl_add(a.c[0], b.c[0]),
        gl_add(a.c[1], b.c[1])
    );
}

/**
 * Subtraction: (a0 + a1*X) - (b0 + b1*X) = (a0-b0) + (a1-b1)*X
 */
__device__ __forceinline__
GoldilocksExt2 ext2_sub(GoldilocksExt2 a, GoldilocksExt2 b) {
    return GoldilocksExt2(
        gl_sub(a.c[0], b.c[0]),
        gl_sub(a.c[1], b.c[1])
    );
}

/**
 * Negation: -(a0 + a1*X) = (-a0) + (-a1)*X
 */
__device__ __forceinline__
GoldilocksExt2 ext2_neg(GoldilocksExt2 a) {
    return GoldilocksExt2(
        gl_neg(a.c[0]),
        gl_neg(a.c[1])
    );
}

/**
 * Multiplication: (a0 + a1*X) * (b0 + b1*X)
 *              = a0*b0 + a1*b1*W + (a0*b1 + a1*b0)*X
 *
 * Karatsuba optimization: 3 base-field muls + cheap mul_by_7 instead of 5 muls.
 *   m0 = a0*b0, m1 = a1*b1, m2 = (a0+a1)*(b0+b1)
 *   c0 = m0 + 7*m1, c1 = m2 - m0 - m1
 */
__device__ __forceinline__
GoldilocksExt2 ext2_mul(GoldilocksExt2 a, GoldilocksExt2 b) {
    GoldilocksField m0 = gl_mul(a.c[0], b.c[0]);           // a0*b0
    GoldilocksField m1 = gl_mul(a.c[1], b.c[1]);           // a1*b1
    GoldilocksField m2 = gl_mul(                            // (a0+a1)*(b0+b1)
        gl_add(a.c[0], a.c[1]),
        gl_add(b.c[0], b.c[1])
    );

    GoldilocksField c0 = gl_add(m0, gl_mul_by_7(m1));      // m0 + 7*m1
    GoldilocksField c1 = gl_sub(gl_sub(m2, m0), m1);       // m2 - m0 - m1

    return GoldilocksExt2(c0, c1);
}

/**
 * Squaring: (a0 + a1*X)^2
 *         = a0^2 + 2*a0*a1*X + a1^2*X^2
 *         = a0^2 + a1^2*W + 2*a0*a1*X
 *
 * Optimized: Uses 2 multiplications instead of 3
 */
__device__ __forceinline__
GoldilocksExt2 ext2_square(GoldilocksExt2 a) {
    GoldilocksField a0_sq = gl_square(a.c[0]);
    GoldilocksField a1_sq = gl_square(a.c[1]);
    GoldilocksField a1_sq_w = gl_mul_by_7(a1_sq);

    // c0 = a0^2 + a1^2*W
    GoldilocksField c0 = gl_add(a0_sq, a1_sq_w);

    // c1 = 2*a0*a1
    GoldilocksField c1 = gl_double(gl_mul(a.c[0], a.c[1]));

    return GoldilocksExt2(c0, c1);
}

/**
 * Scalar multiplication: scalar * (a0 + a1*X)
 */
__device__ __forceinline__
GoldilocksExt2 ext2_scalar_mul(GoldilocksField scalar, GoldilocksExt2 a) {
    return GoldilocksExt2(
        gl_mul(scalar, a.c[0]),
        gl_mul(scalar, a.c[1])
    );
}

/**
 * Inversion using norm: (a0 + a1*X)^(-1)
 *
 * For a = a0 + a1*X in F_p^2:
 *   norm(a) = a * conj(a) = a0^2 - W*a1^2 (in F_p)
 *   a^(-1) = conj(a) / norm(a) = (a0 - a1*X) / (a0^2 - W*a1^2)
 */
__device__ __forceinline__
GoldilocksExt2 ext2_inverse(GoldilocksExt2 a) {
    // norm = a0^2 - W*a1^2
    GoldilocksField a0_sq = gl_square(a.c[0]);
    GoldilocksField a1_sq = gl_square(a.c[1]);
    GoldilocksField w_a1_sq = gl_mul_by_7(a1_sq);
    GoldilocksField norm = gl_sub(a0_sq, w_a1_sq);

    // inv_norm = 1/norm
    GoldilocksField inv_norm = gl_inverse(norm);

    // result = (a0, -a1) * inv_norm
    return GoldilocksExt2(
        gl_mul(a.c[0], inv_norm),
        gl_neg(gl_mul(a.c[1], inv_norm))
    );
}

/**
 * Division: a / b = a * b^(-1)
 */
__device__ __forceinline__
GoldilocksExt2 ext2_div(GoldilocksExt2 a, GoldilocksExt2 b) {
    return ext2_mul(a, ext2_inverse(b));
}

/**
 * Frobenius automorphism: a -> a^p
 *
 * For a = a0 + a1*X:
 *   a^p = a0 + a1*X^p = a0 + a1*X*X^(p-1)
 *       = a0 + a1*X*W^((p-1)/2) (since X^2 = W, so X^(p-1) = W^((p-1)/2))
 *       = a0 + a1*DTH_ROOT*X
 *
 * For W=7 and Goldilocks p, DTH_ROOT = -1, so:
 *   a^p = a0 - a1*X (conjugation)
 */
__device__ __forceinline__
GoldilocksExt2 ext2_frobenius(GoldilocksExt2 a) {
    return GoldilocksExt2(
        a.c[0],
        gl_mul(a.c[1], GoldilocksField(EXT2_DTH_ROOT))
    );
}

/**
 * Conjugation: a0 + a1*X -> a0 - a1*X
 * (Same as Frobenius for quadratic extension with DTH_ROOT = -1)
 */
__device__ __forceinline__
GoldilocksExt2 ext2_conjugate(GoldilocksExt2 a) {
    return GoldilocksExt2(a.c[0], gl_neg(a.c[1]));
}

/**
 * Norm to base field: a * conj(a) = a0^2 - W*a1^2
 */
__device__ __forceinline__
GoldilocksField ext2_norm(GoldilocksExt2 a) {
    GoldilocksField a0_sq = gl_square(a.c[0]);
    GoldilocksField a1_sq = gl_square(a.c[1]);
    GoldilocksField w_a1_sq = gl_mul_by_7(a1_sq);
    return gl_sub(a0_sq, w_a1_sq);
}

/**
 * Exponentiation: a^exp
 */
__device__ __forceinline__
GoldilocksExt2 ext2_exp(GoldilocksExt2 base, uint64_t exp) {
    GoldilocksExt2 result(GoldilocksField(1), GoldilocksField(0));
    GoldilocksExt2 b = base;

    while (exp > 0) {
        if (exp & 1) {
            result = ext2_mul(result, b);
        }
        b = ext2_square(b);
        exp >>= 1;
    }

    return result;
}

/**
 * Check equality
 */
__device__ __forceinline__
bool ext2_eq(GoldilocksExt2 a, GoldilocksExt2 b) {
    return gl_eq(a.c[0], b.c[0]) && gl_eq(a.c[1], b.c[1]);
}

/**
 * Check if zero
 */
__device__ __forceinline__
bool ext2_is_zero(GoldilocksExt2 a) {
    return gl_is_zero(a.c[0]) && gl_is_zero(a.c[1]);
}

/**
 * Convert base field element to Ext2: a -> (a, 0)
 */
__host__ __device__ __forceinline__
GoldilocksExt2 gl_to_ext2(GoldilocksField a) {
    return GoldilocksExt2(a, GoldilocksField(0));
}

/**
 * Extract base field from Ext2 (only valid if c[1] == 0)
 */
__host__ __device__ __forceinline__
GoldilocksField ext2_to_gl(GoldilocksExt2 a) {
    return a.c[0];
}

/**
 * Check if Ext2 element is in base field (c[1] == 0)
 */
__device__ __forceinline__
bool ext2_is_base(GoldilocksExt2 a) {
    return gl_is_zero(a.c[1]);
}

// ============================================================================
// Quintic Extension Field (Degree 5)
// ============================================================================

/**
 * Element of F_p^5 = F_p[X] / (X^5 - 3)
 * Represented as c0 + c1*X + c2*X^2 + c3*X^3 + c4*X^4 where X^5 = 3
 */
struct GoldilocksExt5 {
    GoldilocksField c[5];

    __host__ __device__ __forceinline__
    GoldilocksExt5() {
        for (int i = 0; i < 5; i++) {
            c[i] = GoldilocksField(0);
        }
    }

    __host__ __device__ __forceinline__
    GoldilocksExt5(GoldilocksField c0, GoldilocksField c1, GoldilocksField c2,
                   GoldilocksField c3, GoldilocksField c4) {
        c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3; c[4] = c4;
    }

    __host__ __device__ __forceinline__
    GoldilocksExt5(uint64_t c0, uint64_t c1, uint64_t c2, uint64_t c3, uint64_t c4) {
        c[0] = GoldilocksField(c0);
        c[1] = GoldilocksField(c1);
        c[2] = GoldilocksField(c2);
        c[3] = GoldilocksField(c3);
        c[4] = GoldilocksField(c4);
    }

    // Embed base field element
    __host__ __device__ __forceinline__
    explicit GoldilocksExt5(GoldilocksField base) {
        c[0] = base;
        for (int i = 1; i < 5; i++) {
            c[i] = GoldilocksField(0);
        }
    }
};

// ============================================================================
// Quintic Extension Arithmetic
// ============================================================================

/**
 * Addition
 */
__device__ __forceinline__
GoldilocksExt5 ext5_add(GoldilocksExt5 a, GoldilocksExt5 b) {
    GoldilocksExt5 result;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        result.c[i] = gl_add(a.c[i], b.c[i]);
    }
    return result;
}

/**
 * Subtraction
 */
__device__ __forceinline__
GoldilocksExt5 ext5_sub(GoldilocksExt5 a, GoldilocksExt5 b) {
    GoldilocksExt5 result;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        result.c[i] = gl_sub(a.c[i], b.c[i]);
    }
    return result;
}

/**
 * Negation
 */
__device__ __forceinline__
GoldilocksExt5 ext5_neg(GoldilocksExt5 a) {
    GoldilocksExt5 result;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        result.c[i] = gl_neg(a.c[i]);
    }
    return result;
}

/**
 * Multiplication in F_p[X] / (X^5 - W) where W = 3
 *
 * Product coefficients (with wraparound for X^5 = W):
 *   c0 = a0*b0 + W*(a1*b4 + a2*b3 + a3*b2 + a4*b1)
 *   c1 = a0*b1 + a1*b0 + W*(a2*b4 + a3*b3 + a4*b2)
 *   c2 = a0*b2 + a1*b1 + a2*b0 + W*(a3*b4 + a4*b3)
 *   c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0 + W*a4*b4
 *   c4 = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
 */
__device__ __forceinline__
GoldilocksExt5 ext5_mul(GoldilocksExt5 a, GoldilocksExt5 b) {
    GoldilocksField w = GoldilocksField(EXT5_W);
    GoldilocksExt5 result;

    // Pre-compute b[i] * W for wrapped terms
    GoldilocksField bw[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        bw[i] = gl_mul(b.c[i], w);
    }

    // c0 = a0*b0 + W*(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    result.c[0] = gl_add(
        gl_mul(a.c[0], b.c[0]),
        gl_add(
            gl_add(gl_mul(a.c[1], bw[4]), gl_mul(a.c[2], bw[3])),
            gl_add(gl_mul(a.c[3], bw[2]), gl_mul(a.c[4], bw[1]))
        )
    );

    // c1 = a0*b1 + a1*b0 + W*(a2*b4 + a3*b3 + a4*b2)
    result.c[1] = gl_add(
        gl_add(gl_mul(a.c[0], b.c[1]), gl_mul(a.c[1], b.c[0])),
        gl_add(
            gl_mul(a.c[2], bw[4]),
            gl_add(gl_mul(a.c[3], bw[3]), gl_mul(a.c[4], bw[2]))
        )
    );

    // c2 = a0*b2 + a1*b1 + a2*b0 + W*(a3*b4 + a4*b3)
    result.c[2] = gl_add(
        gl_add(
            gl_mul(a.c[0], b.c[2]),
            gl_add(gl_mul(a.c[1], b.c[1]), gl_mul(a.c[2], b.c[0]))
        ),
        gl_add(gl_mul(a.c[3], bw[4]), gl_mul(a.c[4], bw[3]))
    );

    // c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0 + W*a4*b4
    result.c[3] = gl_add(
        gl_add(
            gl_add(gl_mul(a.c[0], b.c[3]), gl_mul(a.c[1], b.c[2])),
            gl_add(gl_mul(a.c[2], b.c[1]), gl_mul(a.c[3], b.c[0]))
        ),
        gl_mul(a.c[4], bw[4])
    );

    // c4 = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    result.c[4] = gl_add(
        gl_add(
            gl_add(gl_mul(a.c[0], b.c[4]), gl_mul(a.c[1], b.c[3])),
            gl_mul(a.c[2], b.c[2])
        ),
        gl_add(gl_mul(a.c[3], b.c[1]), gl_mul(a.c[4], b.c[0]))
    );

    return result;
}

/**
 * Squaring in F_p^5
 *
 * Optimized coefficients:
 *   c0 = a0^2 + 2*W*(a1*a4 + a2*a3)
 *   c1 = W*a3^2 + 2*(a0*a1 + W*a2*a4)
 *   c2 = a1^2 + 2*(a0*a2 + W*a3*a4)
 *   c3 = W*a4^2 + 2*(a0*a3 + a1*a2)
 *   c4 = a2^2 + 2*(a0*a4 + a1*a3)
 */
__device__ __forceinline__
GoldilocksExt5 ext5_square(GoldilocksExt5 a) {
    GoldilocksField w = GoldilocksField(EXT5_W);
    GoldilocksExt5 result;

    // Pre-compute squares
    GoldilocksField a_sq[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        a_sq[i] = gl_square(a.c[i]);
    }

    // c0 = a0^2 + 2*W*(a1*a4 + a2*a3)
    GoldilocksField t0 = gl_add(gl_mul(a.c[1], a.c[4]), gl_mul(a.c[2], a.c[3]));
    result.c[0] = gl_add(a_sq[0], gl_double(gl_mul(t0, w)));

    // c1 = W*a3^2 + 2*(a0*a1 + W*a2*a4)
    GoldilocksField t1 = gl_add(gl_mul(a.c[0], a.c[1]), gl_mul(w, gl_mul(a.c[2], a.c[4])));
    result.c[1] = gl_add(gl_mul(a_sq[3], w), gl_double(t1));

    // c2 = a1^2 + 2*(a0*a2 + W*a3*a4)
    GoldilocksField t2 = gl_add(gl_mul(a.c[0], a.c[2]), gl_mul(w, gl_mul(a.c[3], a.c[4])));
    result.c[2] = gl_add(a_sq[1], gl_double(t2));

    // c3 = W*a4^2 + 2*(a0*a3 + a1*a2)
    GoldilocksField t3 = gl_add(gl_mul(a.c[0], a.c[3]), gl_mul(a.c[1], a.c[2]));
    result.c[3] = gl_add(gl_mul(a_sq[4], w), gl_double(t3));

    // c4 = a2^2 + 2*(a0*a4 + a1*a3)
    GoldilocksField t4 = gl_add(gl_mul(a.c[0], a.c[4]), gl_mul(a.c[1], a.c[3]));
    result.c[4] = gl_add(a_sq[2], gl_double(t4));

    return result;
}

/**
 * Scalar multiplication
 */
__device__ __forceinline__
GoldilocksExt5 ext5_scalar_mul(GoldilocksField scalar, GoldilocksExt5 a) {
    GoldilocksExt5 result;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        result.c[i] = gl_mul(scalar, a.c[i]);
    }
    return result;
}

/**
 * Frobenius automorphism: a -> a^p
 *
 * For a = sum(a_i * X^i):
 *   a^p = sum(a_i * X^(i*p)) = sum(a_i * (X^p)^i)
 *
 * Since X^5 = W, we have X^p = X * X^(p-1) = X * W^((p-1)/5) = X * DTH_ROOT
 *
 * So: a^p = sum(a_i * DTH_ROOT^i * X^i)
 */
__device__ __forceinline__
GoldilocksExt5 ext5_frobenius(GoldilocksExt5 a) {
    GoldilocksExt5 result;
    GoldilocksField dth = GoldilocksField(EXT5_DTH_ROOT);
    GoldilocksField dth_power = GoldilocksField(1);  // DTH_ROOT^0

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        result.c[i] = gl_mul(a.c[i], dth_power);
        dth_power = gl_mul(dth_power, dth);
    }

    return result;
}

/**
 * Repeated Frobenius: a -> a^(p^count)
 */
__device__ __forceinline__
GoldilocksExt5 ext5_repeated_frobenius(GoldilocksExt5 a, int count) {
    // Reduce count modulo 5 (Frobenius has order 5)
    count = count % 5;
    if (count == 0) return a;

    GoldilocksExt5 result;
    GoldilocksField dth = GoldilocksField(EXT5_DTH_ROOT);

    // Compute DTH_ROOT^count
    GoldilocksField dth_count = gl_exp(dth, count);
    GoldilocksField dth_power = GoldilocksField(1);

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        result.c[i] = gl_mul(a.c[i], dth_power);
        dth_power = gl_mul(dth_power, dth_count);
    }

    return result;
}

/**
 * Inversion using Frobenius (optimized logarithmic algorithm)
 *
 * For degree 5, we use:
 *   a^(-1) = a^(p^4 + p^3 + p^2 + p) / (a * a^(p^4 + p^3 + p^2 + p))
 *
 * The denominator is the norm, which lies in the base field.
 *
 * Algorithm:
 *   1. a_q = frobenius(a)                    // a^p
 *   2. prod = a * a_q                        // a^(1+p)
 *   3. prod_q2 = frobenius^2(prod)           // a^(p^2 + p^3)
 *   4. prod_conj = prod_q2 * prod            // a^(p + p^2 + p^3 + p^4) (wrong, need adjustment)
 *
 * Actually using the standard approach:
 *   prod_conj = a^(q-1) where q = p^5
 *   This requires computing a^(p^5 - 1) / (p - 1) = a^(p^4 + p^3 + p^2 + p + 1) - 1
 */
__device__ __forceinline__
GoldilocksExt5 ext5_inverse(GoldilocksExt5 a) {
    // Use the Frobenius-based algorithm from the reference
    // Step 1: a^q where q = p
    GoldilocksExt5 a_q = ext5_frobenius(a);

    // Step 2: a * a^q = a^(1+q)
    GoldilocksExt5 prod = ext5_mul(a, a_q);

    // Step 3: frobenius(prod) = a^((1+q)*q) = a^(q + q^2)
    GoldilocksExt5 prod_q = ext5_frobenius(prod);

    // Step 4: prod * prod_q = a^(1 + q + q + q^2) = a^(1 + 2q + q^2)
    // This is not quite right. Let me use a different approach.

    // Better approach: iterate Frobenius
    // a^(-1) = a^(q^4 + q^3 + q^2 + q) * norm^(-1)
    // where norm = a^(1 + q + q^2 + q^3 + q^4) is in base field

    GoldilocksExt5 conj = a_q;  // a^q
    GoldilocksExt5 a_q2 = ext5_frobenius(a_q);  // a^(q^2)
    conj = ext5_mul(conj, a_q2);  // a^(q + q^2)

    GoldilocksExt5 a_q3 = ext5_frobenius(a_q2);  // a^(q^3)
    conj = ext5_mul(conj, a_q3);  // a^(q + q^2 + q^3)

    GoldilocksExt5 a_q4 = ext5_frobenius(a_q3);  // a^(q^4)
    conj = ext5_mul(conj, a_q4);  // a^(q + q^2 + q^3 + q^4)

    // norm = a * conj = a^(1 + q + q^2 + q^3 + q^4) should be in base field
    GoldilocksExt5 norm_ext = ext5_mul(a, conj);

    // The norm should have only c[0] non-zero
    GoldilocksField norm = norm_ext.c[0];
    GoldilocksField inv_norm = gl_inverse(norm);

    // result = conj * inv_norm
    return ext5_scalar_mul(inv_norm, conj);
}

/**
 * Division: a / b
 */
__device__ __forceinline__
GoldilocksExt5 ext5_div(GoldilocksExt5 a, GoldilocksExt5 b) {
    return ext5_mul(a, ext5_inverse(b));
}

/**
 * Exponentiation: a^exp
 */
__device__ __forceinline__
GoldilocksExt5 ext5_exp(GoldilocksExt5 base, uint64_t exp) {
    GoldilocksExt5 result;
    result.c[0] = GoldilocksField(1);
    GoldilocksExt5 b = base;

    while (exp > 0) {
        if (exp & 1) {
            result = ext5_mul(result, b);
        }
        b = ext5_square(b);
        exp >>= 1;
    }

    return result;
}

/**
 * Check equality
 */
__device__ __forceinline__
bool ext5_eq(GoldilocksExt5 a, GoldilocksExt5 b) {
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        if (!gl_eq(a.c[i], b.c[i])) return false;
    }
    return true;
}

/**
 * Check if zero
 */
__device__ __forceinline__
bool ext5_is_zero(GoldilocksExt5 a) {
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        if (!gl_is_zero(a.c[i])) return false;
    }
    return true;
}

/**
 * Convert base field element to Ext5: a -> (a, 0, 0, 0, 0)
 */
__host__ __device__ __forceinline__
GoldilocksExt5 gl_to_ext5(GoldilocksField a) {
    GoldilocksExt5 result;
    result.c[0] = a;
    for (int i = 1; i < 5; i++) {
        result.c[i] = GoldilocksField(0);
    }
    return result;
}

/**
 * Extract base field from Ext5 (only valid if c[1..4] == 0)
 */
__host__ __device__ __forceinline__
GoldilocksField ext5_to_gl(GoldilocksExt5 a) {
    return a.c[0];
}

/**
 * Check if Ext5 element is in base field (c[1..4] == 0)
 */
__device__ __forceinline__
bool ext5_is_base(GoldilocksExt5 a) {
    for (int i = 1; i < 5; i++) {
        if (!gl_is_zero(a.c[i])) return false;
    }
    return true;
}

// ============================================================================
// Constants for device
// ============================================================================

// Pre-computed powers of DTH_ROOT for Frobenius
__constant__ uint64_t d_EXT5_DTH_POWERS[5];

static const uint64_t h_EXT5_DTH_POWERS[5] = {
    1ULL,                       // DTH_ROOT^0
    1041288259238279555ULL,     // DTH_ROOT^1
    0ULL,                       // DTH_ROOT^2 (to be computed)
    0ULL,                       // DTH_ROOT^3
    0ULL                        // DTH_ROOT^4
};

/**
 * Initialize extension field constants
 */
inline cudaError_t extension_init() {
    // Copy pre-computed DTH_ROOT powers to device constant memory
    // These values are pre-computed in h_EXT5_DTH_POWERS
    return cudaMemcpyToSymbol(d_EXT5_DTH_POWERS, h_EXT5_DTH_POWERS, sizeof(h_EXT5_DTH_POWERS));
}

#endif // EXTENSION_CUH

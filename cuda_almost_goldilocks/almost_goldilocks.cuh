/**
 * Almost-Goldilocks Field CUDA Implementation
 *
 * Prime: p = 2^64 - 2^32 + 1 - 32 = 2^64 - 2^32 - 31 = 0xFFFFFFFEFFFFFFE1
 *
 * Compared to Goldilocks (p_G = 2^64 - 2^32 + 1), this field has the same
 * 2^64 - 2^32 - <small constant> shape, but the small constant changes from
 * +1 to -31. Reduction is still Solinas-friendly because the wrap constant
 *
 *     c := 2^64 mod p  =  2^32 + 31  =  0x10000001F   (33 bits)
 *
 * is still small. The trade-off vs. Goldilocks:
 *   - lose the 2^96 ≡ -1 shortcut (here 2^96 ≡ 2^37 + 31, not -1)
 *   - lose the "c fits in 32 bits" property → c is now 33 bits, so the
 *     cheap "u32 × u32 → u64" pass is replaced by full u64 × u64 multiplies
 *
 * The reduction therefore uses a 2-pass Solinas iteration: replace 2^64
 * with c at each step until the value fits in u64.
 */

#ifndef ALMOST_GOLDILOCKS_CUH
#define ALMOST_GOLDILOCKS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Constants
// ============================================================================

// The almost-Goldilocks prime: P = 2^64 - 2^32 - 31
#define ALMOST_GOLDILOCKS_PRIME 0xFFFFFFFEFFFFFFE1ULL

// REDUCE_C = 2^64 mod P = 2^32 + 31    (33-bit wrap constant)
#define ALMOST_REDUCE_C 0x10000001FULL

// (P + 1) / 2, used for halving odd numbers
#define ALMOST_HALF_P_PLUS_ONE 0x7FFFFFFF7FFFFFF1ULL

// ============================================================================
// Field Element
// ============================================================================

struct AlmostGoldilocksField {
    uint64_t value;

    __host__ __device__ __forceinline__
    AlmostGoldilocksField() : value(0) {}

    __host__ __device__ __forceinline__
    explicit AlmostGoldilocksField(uint64_t v) : value(v) {}
};

// ============================================================================
// 128-bit Helpers
// ============================================================================

struct agl_uint128_t {
    uint64_t lo;
    uint64_t hi;

    __host__ __device__ __forceinline__
    agl_uint128_t() : lo(0), hi(0) {}

    __host__ __device__ __forceinline__
    agl_uint128_t(uint64_t l, uint64_t h) : lo(l), hi(h) {}
};

#ifdef __CUDA_ARCH__
__device__ __forceinline__
agl_uint128_t agl_mul_u64_u64(uint64_t a, uint64_t b) {
    agl_uint128_t result;
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(result.lo) : "l"(a), "l"(b));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(result.hi) : "l"(a), "l"(b));
    return result;
}
#else
inline agl_uint128_t agl_mul_u64_u64(uint64_t a, uint64_t b) {
    agl_uint128_t result;
#if defined(__SIZEOF_INT128__)
    __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
    result.lo = (uint64_t)prod;
    result.hi = (uint64_t)(prod >> 64);
#else
    uint64_t a_lo = a & 0xFFFFFFFFULL;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFULL;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint64_t mid = p1 + (p0 >> 32);
    uint64_t mid_lo = mid & 0xFFFFFFFFULL;
    uint64_t mid_hi = mid >> 32;

    mid_lo += p2;
    if (mid_lo < p2) mid_hi++;

    result.lo = (mid_lo << 32) | (p0 & 0xFFFFFFFFULL);
    result.hi = p3 + mid_hi + (mid_lo >> 32);
#endif
    return result;
}
#endif

// ============================================================================
// Core Arithmetic
// ============================================================================

/**
 * Canonicalize to [0, P).
 *
 * Input may be any u64. Because P = 2^64 - c and c ≤ 2^33, any u64 value v
 * satisfies v < 2P, so a single conditional subtract is enough.
 */
__host__ __device__ __forceinline__
uint64_t agl_canonicalize(uint64_t value) {
    if (value >= ALMOST_GOLDILOCKS_PRIME) {
        return value - ALMOST_GOLDILOCKS_PRIME;
    }
    return value;
}

/**
 * Addition without canonicalization.
 *
 * Result is in [0, 2^64), congruent to a + b mod P. On overflow, fold the
 * carry by adding c (= 2^64 mod P). A second overflow is theoretically
 * possible because c > 0, so we check and fold once more (this branch is
 * extremely rare in practice).
 */
__host__ __device__ __forceinline__
uint64_t agl_add_no_canonicalize(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    if (sum < a) {
        sum += ALMOST_REDUCE_C;
        if (sum < ALMOST_REDUCE_C) {
            sum += ALMOST_REDUCE_C;
        }
    }
    return sum;
}

/**
 * Subtraction without canonicalization.
 *
 * On underflow we have "added" 2^64; subtract c to compensate (since
 * 2^64 ≡ c, the corrective amount is c). Double-borrow check mirrors add.
 */
__host__ __device__ __forceinline__
uint64_t agl_sub_no_canonicalize(uint64_t a, uint64_t b) {
    uint64_t diff = a - b;
    if (a < b) {
        diff -= ALMOST_REDUCE_C;
        if (diff > (uint64_t)(-1) - ALMOST_REDUCE_C) {
            diff -= ALMOST_REDUCE_C;
        }
    }
    return diff;
}

/**
 * Reduce a 128-bit value modulo P using a 2-pass Solinas iteration.
 *
 * Identity: 2^64 ≡ c (mod P)  where  c = 2^32 + 31  = ALMOST_REDUCE_C.
 *
 * Pass 1: r1 = x.lo + x.hi * c                  ≤ 2^97
 * Pass 2: r2 = r1.lo + r1.hi * c                ≤ 2^66
 * Pass 3: r3 = r2.lo + r2.hi * c   (tiny mul)   ≤ 2^64 + O(2^37)
 *
 * Returned value is in [0, 2^64), congruent to x mod P (non-canonical).
 */
__host__ __device__ __forceinline__
uint64_t agl_reduce128(agl_uint128_t x) {
    // Pass 1
    agl_uint128_t prod1 = agl_mul_u64_u64(x.hi, ALMOST_REDUCE_C);
    uint64_t r1_lo = prod1.lo + x.lo;
    uint64_t carry1 = (r1_lo < prod1.lo) ? 1ULL : 0ULL;
    uint64_t r1_hi = prod1.hi + carry1;             // ≤ 2^33

    // Pass 2
    agl_uint128_t prod2 = agl_mul_u64_u64(r1_hi, ALMOST_REDUCE_C);
    uint64_t r2_lo = prod2.lo + r1_lo;
    uint64_t carry2 = (r2_lo < prod2.lo) ? 1ULL : 0ULL;
    uint64_t r2_hi = prod2.hi + carry2;             // very small (≤ a few bits)

    // Pass 3: r2_hi * c fits comfortably in u64 (≤ ~2^37)
    uint64_t corr = r2_hi * ALMOST_REDUCE_C;
    return agl_add_no_canonicalize(r2_lo, corr);
}

/**
 * Field addition.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_add(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return AlmostGoldilocksField(agl_add_no_canonicalize(a.value, b.value));
}

/**
 * Field subtraction.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_sub(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return AlmostGoldilocksField(agl_sub_no_canonicalize(a.value, b.value));
}

/**
 * Field negation.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_neg(AlmostGoldilocksField a) {
    uint64_t canonical = agl_canonicalize(a.value);
    if (canonical == 0) {
        return AlmostGoldilocksField(0);
    }
    return AlmostGoldilocksField(ALMOST_GOLDILOCKS_PRIME - canonical);
}

/**
 * Field multiplication.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_mul(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    agl_uint128_t prod = agl_mul_u64_u64(a.value, b.value);
    return AlmostGoldilocksField(agl_reduce128(prod));
}

/**
 * Field squaring.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_square(AlmostGoldilocksField a) {
    return agl_mul(a, a);
}

/**
 * Field doubling.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_double(AlmostGoldilocksField a) {
    return agl_add(a, a);
}

/**
 * Cheap multiplication by 3 (Ext2 non-residue): 3*a = 2*a + a.
 * Two field additions, no multiplication.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_mul_by_3(AlmostGoldilocksField a) {
    AlmostGoldilocksField two_a = agl_double(a);
    return agl_add(two_a, a);
}

/**
 * Field halving: a / 2.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_halve(AlmostGoldilocksField a) {
    uint64_t val = agl_canonicalize(a.value);
    if (val & 1) {
        return AlmostGoldilocksField((val >> 1) + ALMOST_HALF_P_PLUS_ONE);
    } else {
        return AlmostGoldilocksField(val >> 1);
    }
}

/**
 * Exponentiation by squaring.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_exp(AlmostGoldilocksField base, uint64_t exp) {
    AlmostGoldilocksField result(1);
    AlmostGoldilocksField b = base;
    while (exp > 0) {
        if (exp & 1) result = agl_mul(result, b);
        b = agl_square(b);
        exp >>= 1;
    }
    return result;
}

/**
 * Modular inverse via Fermat: a^(P-2).
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_inverse(AlmostGoldilocksField a) {
    return agl_exp(a, ALMOST_GOLDILOCKS_PRIME - 2);
}

/**
 * Field division.
 */
__host__ __device__ __forceinline__
AlmostGoldilocksField agl_div(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return agl_mul(a, agl_inverse(b));
}

/**
 * Equality (compares canonical forms).
 */
__host__ __device__ __forceinline__
bool agl_eq(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return agl_canonicalize(a.value) == agl_canonicalize(b.value);
}

__host__ __device__ __forceinline__
bool agl_is_zero(AlmostGoldilocksField a) {
    return agl_canonicalize(a.value) == 0;
}

__host__ __device__ __forceinline__
bool agl_is_one(AlmostGoldilocksField a) {
    return agl_canonicalize(a.value) == 1;
}

// ============================================================================
// Batch helpers (device-only)
// ============================================================================

__device__ __forceinline__
AlmostGoldilocksField agl_sum_array(const AlmostGoldilocksField* arr, int n) {
    if (n == 0) return AlmostGoldilocksField(0);
    if (n == 1) return arr[0];

    agl_uint128_t sum;
    sum.lo = arr[0].value;
    sum.hi = 0;
    for (int i = 1; i < n; i++) {
        uint64_t old_lo = sum.lo;
        sum.lo += arr[i].value;
        if (sum.lo < old_lo) sum.hi += 1;
    }
    return AlmostGoldilocksField(agl_reduce128(sum));
}

__device__ __forceinline__
AlmostGoldilocksField agl_dot_product(
    const AlmostGoldilocksField* a,
    const AlmostGoldilocksField* b,
    int n
) {
    AlmostGoldilocksField result(0);
    for (int i = 0; i < n; i++) {
        result = agl_add(result, agl_mul(a[i], b[i]));
    }
    return result;
}

// ============================================================================
// Initialization
// ============================================================================

/**
 * The Goldilocks port maintains a POWERS_OF_TWO / TWO_ADIC_GENERATORS table in
 * constant memory, but in the rest of this codebase those tables are never
 * read by any kernel (verified by grep). The almost-Goldilocks prime has
 * 2-adicity 5 (only enough for NTTs of size ≤ 32), so an FFT-style table
 * would not be useful here anyway. We expose a no-op init for API symmetry.
 */
inline cudaError_t almost_goldilocks_init() {
    return cudaSuccess;
}

#endif // ALMOST_GOLDILOCKS_CUH

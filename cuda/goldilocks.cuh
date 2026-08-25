/**
 * Goldilocks Field CUDA Implementation
 *
 * This implements arithmetic over the Goldilocks prime field:
 * P = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
 *
 * Based on the Plonky3 reference implementation.
 *
 * Key properties:
 * - 64-bit prime that fits in a single u64
 * - Non-canonical representation allowed (values can be in [0, 2^64))
 * - 2^96 ≡ -1 (mod P), enabling fast reduction
 * - 2^192 ≡ 1 (mod P)
 */

#ifndef GOLDILOCKS_CUH
#define GOLDILOCKS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Constants
// ============================================================================

// The Goldilocks prime: P = 2^64 - 2^32 + 1
#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL

// NEG_ORDER = 2^32 - 1 = P - 2^64 (used for reduction)
#define NEG_ORDER 0x00000000FFFFFFFFULL

// (P + 1) / 2, used for halving odd numbers
#define HALF_P_PLUS_ONE 0x7FFFFFFF80000001ULL

// ============================================================================
// Device Constants (stored in constant memory for fast access)
// ============================================================================

// Pre-computed powers of 2 modulo P for fast multiplication by powers of 2
// POWERS_OF_TWO[i] = 2^i mod P for i in [0, 96)
__constant__ uint64_t d_POWERS_OF_TWO[96];

// Two-adic generators for FFT operations
// TWO_ADIC_GENERATORS[i] is a primitive 2^i-th root of unity
__constant__ uint64_t d_TWO_ADIC_GENERATORS[33];

// ============================================================================
// Goldilocks Field Element Structure
// ============================================================================

struct GoldilocksField {
    uint64_t value;

    __host__ __device__ __forceinline__
    GoldilocksField() : value(0) {}

    __host__ __device__ __forceinline__
    explicit GoldilocksField(uint64_t v) : value(v) {}
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Check if a >= b for unsigned 64-bit integers
 */
__device__ __forceinline__
bool gte(uint64_t a, uint64_t b) {
    return a >= b;
}

/**
 * 128-bit unsigned integer representation using two 64-bit values
 */
struct uint128_t {
    uint64_t lo;  // Lower 64 bits
    uint64_t hi;  // Upper 64 bits

    __host__ __device__ __forceinline__
    uint128_t() : lo(0), hi(0) {}

    __host__ __device__ __forceinline__
    uint128_t(uint64_t l, uint64_t h) : lo(l), hi(h) {}
};

/**
 * Multiply two 64-bit unsigned integers to get a 128-bit result
 * Device version uses PTX assembly for optimal performance
 * Host version uses __uint128_t or manual decomposition
 */
#ifdef __CUDA_ARCH__
__device__ __forceinline__
uint128_t mul_u64_u64(uint64_t a, uint64_t b) {
    uint128_t result;
    // Use PTX mul.hi and mul.lo for 64-bit multiplication
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(result.lo) : "l"(a), "l"(b));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(result.hi) : "l"(a), "l"(b));
    return result;
}
#else
// Host-side implementation
inline uint128_t mul_u64_u64(uint64_t a, uint64_t b) {
    uint128_t result;
#if defined(__SIZEOF_INT128__)
    // Use compiler's 128-bit support if available (GCC, Clang)
    __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
    result.lo = (uint64_t)prod;
    result.hi = (uint64_t)(prod >> 64);
#else
    // Manual 64-bit multiplication using 32-bit parts
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

/**
 * Add two 64-bit integers with carry detection
 * Returns the sum and sets carry to 1 if overflow occurred
 */
__device__ __forceinline__
uint64_t add_with_carry(uint64_t a, uint64_t b, uint32_t* carry) {
    uint64_t sum = a + b;
    *carry = (sum < a) ? 1 : 0;
    return sum;
}

/**
 * Subtract two 64-bit integers with borrow detection
 * Returns the difference and sets borrow to 1 if underflow occurred
 */
__device__ __forceinline__
uint64_t sub_with_borrow(uint64_t a, uint64_t b, uint32_t* borrow) {
    uint64_t diff = a - b;
    *borrow = (a < b) ? 1 : 0;
    return diff;
}

// ============================================================================
// Core Arithmetic Operations
// ============================================================================

/**
 * Canonicalize a field element to [0, P)
 * Only needed for equality checks, hashing, or output
 */
__host__ __device__ __forceinline__
uint64_t canonicalize(uint64_t value) {
    // If value >= P, subtract P
    if (value >= GOLDILOCKS_PRIME) {
        return value - GOLDILOCKS_PRIME;
    }
    return value;
}

/**
 * Addition without canonicalization
 * Handles potential overflow by adding NEG_ORDER when carry occurs
 *
 * The result may be >= P but is guaranteed to be < 2^64
 */
__host__ __device__ __forceinline__
uint64_t add_no_canonicalize(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;

    // If overflow occurred (sum < a), we need to add NEG_ORDER
    // This is equivalent to subtracting P modulo 2^64
    if (sum < a) {
        sum += NEG_ORDER;
        // Check for second overflow (rare but possible)
        if (sum < NEG_ORDER) {
            sum += NEG_ORDER;
        }
    }

    return sum;
}

/**
 * Subtraction without canonicalization
 * Handles potential underflow by subtracting NEG_ORDER when borrow occurs
 */
__host__ __device__ __forceinline__
uint64_t sub_no_canonicalize(uint64_t a, uint64_t b) {
    uint64_t diff = a - b;

    // If underflow occurred (a < b), we need to subtract NEG_ORDER
    // This is equivalent to adding P modulo 2^64
    if (a < b) {
        diff -= NEG_ORDER;
        // Check for second underflow (rare but possible)
        if (diff > (uint64_t)(-1) - NEG_ORDER) {
            diff -= NEG_ORDER;
        }
    }

    return diff;
}

/**
 * Reduce a 128-bit value modulo P
 *
 * Key insight: 2^64 ≡ 2^32 - 1 (mod P)
 * And: 2^96 ≡ -1 (mod P)
 *
 * For x = x_lo + x_hi * 2^64:
 *   x_hi * 2^64 = (x_hi_hi * 2^32 + x_hi_lo) * 2^64
 *               = x_hi_hi * 2^96 + x_hi_lo * 2^64
 *               ≡ -x_hi_hi + x_hi_lo * (2^32 - 1) (mod P)
 */
__host__ __device__ __forceinline__
uint64_t reduce128(uint128_t x) {
    uint64_t x_lo = x.lo;
    uint64_t x_hi = x.hi;

    // Split x_hi into upper and lower 32-bit parts
    uint64_t x_hi_hi = x_hi >> 32;           // Upper 32 bits of x_hi
    uint64_t x_hi_lo = x_hi & 0xFFFFFFFFULL; // Lower 32 bits of x_hi

    // t0 = x_lo - x_hi_hi
    uint64_t t0 = x_lo - x_hi_hi;
    if (x_lo < x_hi_hi) {
        // Underflow: subtract NEG_ORDER (equivalent to adding P)
        t0 -= NEG_ORDER;
    }

    // t1 = x_hi_lo * NEG_ORDER
    uint64_t t1 = x_hi_lo * NEG_ORDER;

    // Return t0 + t1 without canonicalization
    return add_no_canonicalize(t0, t1);
}

/**
 * Field addition: a + b (mod P)
 */
__device__ __forceinline__
GoldilocksField gl_add(GoldilocksField a, GoldilocksField b) {
    return GoldilocksField(add_no_canonicalize(a.value, b.value));
}

/**
 * Field subtraction: a - b (mod P)
 */
__device__ __forceinline__
GoldilocksField gl_sub(GoldilocksField a, GoldilocksField b) {
    return GoldilocksField(sub_no_canonicalize(a.value, b.value));
}

/**
 * Field negation: -a (mod P)
 */
__device__ __forceinline__
GoldilocksField gl_neg(GoldilocksField a) {
    uint64_t canonical = canonicalize(a.value);
    if (canonical == 0) {
        return GoldilocksField(0);
    }
    return GoldilocksField(GOLDILOCKS_PRIME - canonical);
}

/**
 * Field multiplication: a * b (mod P)
 */
__device__ __forceinline__
GoldilocksField gl_mul(GoldilocksField a, GoldilocksField b) {
    uint128_t prod = mul_u64_u64(a.value, b.value);
    return GoldilocksField(reduce128(prod));
}

/**
 * Field squaring: a^2 (mod P)
 * Uses the same algorithm as multiplication
 */
__device__ __forceinline__
GoldilocksField gl_square(GoldilocksField a) {
    return gl_mul(a, a);
}

/**
 * Cheap multiplication by 7: 7*a = 8*a - a = (a << 3) - a (mod P)
 * Avoids full 64x64 multiply — uses only shifts, a tiny multiply, and adds.
 */
__device__ __forceinline__
GoldilocksField gl_mul_by_7(GoldilocksField a) {
    uint64_t x = a.value;
    uint64_t overflow = x >> 61;              // top 3 bits (0..7)
    uint64_t lo = x << 3;                     // low 64 bits of 8*x
    // 8*x mod p = lo + overflow * NEG_ORDER  (since 2^64 ≡ NEG_ORDER mod p)
    uint64_t eight_x = add_no_canonicalize(lo, overflow * NEG_ORDER);
    return GoldilocksField(sub_no_canonicalize(eight_x, x));
}

/**
 * Field doubling: 2*a (mod P)
 */
__device__ __forceinline__
GoldilocksField gl_double(GoldilocksField a) {
    return gl_add(a, a);
}

/**
 * Field halving: a/2 (mod P)
 * If a is even: a/2 = a >> 1
 * If a is odd:  a/2 = (a >> 1) + (P+1)/2
 */
__device__ __forceinline__
GoldilocksField gl_halve(GoldilocksField a) {
    uint64_t val = canonicalize(a.value);
    if (val & 1) {
        // Odd: (val + P) / 2 = (val >> 1) + ((P + 1) >> 1)
        return GoldilocksField((val >> 1) + HALF_P_PLUS_ONE);
    } else {
        // Even: val / 2
        return GoldilocksField(val >> 1);
    }
}

/**
 * Exponentiation by squaring: a^exp (mod P)
 */
__device__ __forceinline__
GoldilocksField gl_exp(GoldilocksField base, uint64_t exp) {
    GoldilocksField result(1);
    GoldilocksField b = base;

    while (exp > 0) {
        if (exp & 1) {
            result = gl_mul(result, b);
        }
        b = gl_square(b);
        exp >>= 1;
    }

    return result;
}

/**
 * Modular inverse using Fermat's little theorem: a^(-1) = a^(P-2) (mod P)
 *
 * This is simpler than the binary GCD approach but may be slower.
 * For production use, consider implementing the optimized GCD-based inverse.
 */
__device__ __forceinline__
GoldilocksField gl_inverse(GoldilocksField a) {
    // P - 2 = 0xFFFFFFFF00000001 - 2 = 0xFFFFFFFEFFFFFFFF
    return gl_exp(a, GOLDILOCKS_PRIME - 2);
}

/**
 * Field division: a / b (mod P)
 */
__device__ __forceinline__
GoldilocksField gl_div(GoldilocksField a, GoldilocksField b) {
    return gl_mul(a, gl_inverse(b));
}

/**
 * Equality check (compares canonical forms)
 */
__device__ __forceinline__
bool gl_eq(GoldilocksField a, GoldilocksField b) {
    return canonicalize(a.value) == canonicalize(b.value);
}

/**
 * Check if field element is zero
 */
__device__ __forceinline__
bool gl_is_zero(GoldilocksField a) {
    uint64_t val = canonicalize(a.value);
    return val == 0;
}

/**
 * Check if field element is one
 */
__device__ __forceinline__
bool gl_is_one(GoldilocksField a) {
    uint64_t val = canonicalize(a.value);
    return val == 1;
}

// ============================================================================
// Batch Operations (useful for parallelization)
// ============================================================================

/**
 * Sum of array elements
 * Uses 128-bit accumulation for better precision
 */
__device__ __forceinline__
GoldilocksField gl_sum_array(const GoldilocksField* arr, int n) {
    if (n == 0) return GoldilocksField(0);
    if (n == 1) return arr[0];
    if (n == 2) return gl_add(arr[0], arr[1]);
    if (n == 3) return gl_add(gl_add(arr[0], arr[1]), arr[2]);

    // For larger arrays, use 128-bit accumulation
    uint128_t sum;
    sum.lo = arr[0].value;
    sum.hi = 0;

    for (int i = 1; i < n; i++) {
        uint64_t old_lo = sum.lo;
        sum.lo += arr[i].value;
        if (sum.lo < old_lo) {
            sum.hi += 1;
        }
    }

    return GoldilocksField(reduce128(sum));
}

/**
 * Dot product of two arrays
 */
__device__ __forceinline__
GoldilocksField gl_dot_product(const GoldilocksField* a, const GoldilocksField* b, int n) {
    GoldilocksField result(0);
    for (int i = 0; i < n; i++) {
        result = gl_add(result, gl_mul(a[i], b[i]));
    }
    return result;
}

// ============================================================================
// Extension Field Operations (Quadratic Extension)
// ============================================================================

// Quadratic extension: F_{p^2} = F_p[x] / (x^2 - 7)
// W = 7 (non-residue)
#define QUADRATIC_NON_RESIDUE 7ULL

struct GoldilocksExtQuad {
    GoldilocksField c0;  // Coefficient of 1
    GoldilocksField c1;  // Coefficient of x

    __device__ __forceinline__
    GoldilocksExtQuad() : c0(0), c1(0) {}

    __device__ __forceinline__
    GoldilocksExtQuad(GoldilocksField a, GoldilocksField b) : c0(a), c1(b) {}

    __device__ __forceinline__
    GoldilocksExtQuad(uint64_t a, uint64_t b) : c0(a), c1(b) {}
};

/**
 * Quadratic extension addition
 */
__device__ __forceinline__
GoldilocksExtQuad gl_ext_add(GoldilocksExtQuad a, GoldilocksExtQuad b) {
    return GoldilocksExtQuad(
        gl_add(a.c0, b.c0),
        gl_add(a.c1, b.c1)
    );
}

/**
 * Quadratic extension subtraction
 */
__device__ __forceinline__
GoldilocksExtQuad gl_ext_sub(GoldilocksExtQuad a, GoldilocksExtQuad b) {
    return GoldilocksExtQuad(
        gl_sub(a.c0, b.c0),
        gl_sub(a.c1, b.c1)
    );
}

/**
 * Quadratic extension multiplication
 * (a0 + a1*x) * (b0 + b1*x) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*x
 */
__device__ __forceinline__
GoldilocksExtQuad gl_ext_mul(GoldilocksExtQuad a, GoldilocksExtQuad b) {
    GoldilocksField a0b0 = gl_mul(a.c0, b.c0);
    GoldilocksField a1b1 = gl_mul(a.c1, b.c1);
    GoldilocksField a0b1 = gl_mul(a.c0, b.c1);
    GoldilocksField a1b0 = gl_mul(a.c1, b.c0);

    // c0 = a0*b0 + 7*a1*b1
    GoldilocksField w_a1b1 = gl_mul(a1b1, GoldilocksField(QUADRATIC_NON_RESIDUE));
    GoldilocksField c0 = gl_add(a0b0, w_a1b1);

    // c1 = a0*b1 + a1*b0
    GoldilocksField c1 = gl_add(a0b1, a1b0);

    return GoldilocksExtQuad(c0, c1);
}

/**
 * Quadratic extension squaring
 * (a0 + a1*x)^2 = (a0^2 + 7*a1^2) + (2*a0*a1)*x
 */
__device__ __forceinline__
GoldilocksExtQuad gl_ext_square(GoldilocksExtQuad a) {
    GoldilocksField a0_sq = gl_square(a.c0);
    GoldilocksField a1_sq = gl_square(a.c1);
    GoldilocksField a0a1 = gl_mul(a.c0, a.c1);

    // c0 = a0^2 + 7*a1^2
    GoldilocksField w_a1_sq = gl_mul(a1_sq, GoldilocksField(QUADRATIC_NON_RESIDUE));
    GoldilocksField c0 = gl_add(a0_sq, w_a1_sq);

    // c1 = 2*a0*a1
    GoldilocksField c1 = gl_double(a0a1);

    return GoldilocksExtQuad(c0, c1);
}

// ============================================================================
// Initialization Functions (call from host)
// ============================================================================

// Powers of 2 table (to be copied to device constant memory)
static const uint64_t h_POWERS_OF_TWO[96] = {
    0x0000000000000001ULL, 0x0000000000000002ULL, 0x0000000000000004ULL, 0x0000000000000008ULL,
    0x0000000000000010ULL, 0x0000000000000020ULL, 0x0000000000000040ULL, 0x0000000000000080ULL,
    0x0000000000000100ULL, 0x0000000000000200ULL, 0x0000000000000400ULL, 0x0000000000000800ULL,
    0x0000000000001000ULL, 0x0000000000002000ULL, 0x0000000000004000ULL, 0x0000000000008000ULL,
    0x0000000000010000ULL, 0x0000000000020000ULL, 0x0000000000040000ULL, 0x0000000000080000ULL,
    0x0000000000100000ULL, 0x0000000000200000ULL, 0x0000000000400000ULL, 0x0000000000800000ULL,
    0x0000000001000000ULL, 0x0000000002000000ULL, 0x0000000004000000ULL, 0x0000000008000000ULL,
    0x0000000010000000ULL, 0x0000000020000000ULL, 0x0000000040000000ULL, 0x0000000080000000ULL,
    // 2^32 mod P = 2^32
    0x0000000100000000ULL, 0x0000000200000000ULL, 0x0000000400000000ULL, 0x0000000800000000ULL,
    0x0000001000000000ULL, 0x0000002000000000ULL, 0x0000004000000000ULL, 0x0000008000000000ULL,
    0x0000010000000000ULL, 0x0000020000000000ULL, 0x0000040000000000ULL, 0x0000080000000000ULL,
    0x0000100000000000ULL, 0x0000200000000000ULL, 0x0000400000000000ULL, 0x0000800000000000ULL,
    0x0001000000000000ULL, 0x0002000000000000ULL, 0x0004000000000000ULL, 0x0008000000000000ULL,
    0x0010000000000000ULL, 0x0020000000000000ULL, 0x0040000000000000ULL, 0x0080000000000000ULL,
    0x0100000000000000ULL, 0x0200000000000000ULL, 0x0400000000000000ULL, 0x0800000000000000ULL,
    0x1000000000000000ULL, 0x2000000000000000ULL, 0x4000000000000000ULL, 0x8000000000000000ULL,
    // 2^64 mod P = 2^32 - 1 = 0xFFFFFFFF
    0x00000000FFFFFFFFULL, 0x00000001FFFFFFFEULL, 0x00000003FFFFFFFCULL, 0x00000007FFFFFFF8ULL,
    0x0000000FFFFFFFF0ULL, 0x0000001FFFFFFFE0ULL, 0x0000003FFFFFFFC0ULL, 0x0000007FFFFFFF80ULL,
    0x000000FFFFFFFF00ULL, 0x000001FFFFFFFE00ULL, 0x000003FFFFFFFC00ULL, 0x000007FFFFFFF800ULL,
    0x00000FFFFFFFF000ULL, 0x00001FFFFFFFE000ULL, 0x00003FFFFFFFC000ULL, 0x00007FFFFFFF8000ULL,
    0x0000FFFFFFFF0000ULL, 0x0001FFFFFFFE0000ULL, 0x0003FFFFFFFC0000ULL, 0x0007FFFFFFF80000ULL,
    0x000FFFFFFFF00000ULL, 0x001FFFFFFFE00000ULL, 0x003FFFFFFFC00000ULL, 0x007FFFFFFF800000ULL,
    0x00FFFFFFFF000000ULL, 0x01FFFFFFFE000000ULL, 0x03FFFFFFFC000000ULL, 0x07FFFFFFF8000000ULL,
    0x0FFFFFFFF0000000ULL, 0x1FFFFFFFE0000000ULL, 0x3FFFFFFFC0000000ULL, 0x7FFFFFFF80000000ULL,
};

// Two-adic generators (primitive 2^i-th roots of unity)
static const uint64_t h_TWO_ADIC_GENERATORS[33] = {
    0x0000000000000001ULL,  // 2^0-th root = 1
    0xFFFFFFFF00000000ULL,  // 2^1-th root = -1 = P - 1
    0x0001000000000001ULL,  // 2^2-th root
    0x185629DCDA58878CULL,  // 2^3-th root
    0xFFC97F062A770992ULL,  // 2^4-th root
    0x10A0A1D088D53A51ULL,  // 2^5-th root
    0x54359E7B8BF1D7BCULL,  // 2^6-th root
    0xF2FD9E72E4A53A81ULL,  // 2^7-th root
    0x1B88AC00E0E47A3BULL,  // 2^8-th root
    0xB5360BD6D4A4B05BULL,  // 2^9-th root
    0xBE78E8BCA5B8B68CULL,  // 2^10-th root
    0x3D5F5E83CB3B5F0CULL,  // 2^11-th root
    0x0F3A91EA69D40F91ULL,  // 2^12-th root
    0xE00CE8E41B219A9EULL,  // 2^13-th root
    0x8C29AD304AE7F1F2ULL,  // 2^14-th root
    0x9B43A12A7EF8E9D8ULL,  // 2^15-th root
    0x4ABEF55E8A8A5BDDULL,  // 2^16-th root
    0xEECE3A65D4E0D4F6ULL,  // 2^17-th root
    0x5B8A0B0D4DC6D8E3ULL,  // 2^18-th root
    0x0E6C0ED1F5294378ULL,  // 2^19-th root
    0x61E82ACF85A99CEFULL,  // 2^20-th root
    0x2A62E18DF73BEFAFULL,  // 2^21-th root
    0x0AAB3B35CAB3A1BAULL,  // 2^22-th root
    0xC09D6EC47F780F8AULL,  // 2^23-th root
    0x95AAC9FD2508D44AULL,  // 2^24-th root
    0xAEDAABC5C03B8A1EULL,  // 2^25-th root
    0x3EBA2C39E32A72EAULL,  // 2^26-th root
    0x759FE45CB11E5E5AULL,  // 2^27-th root
    0x84CF26E29A0A8643ULL,  // 2^28-th root
    0xEC421D148F1E85B6ULL,  // 2^29-th root
    0x8E71BE77E2C51DB8ULL,  // 2^30-th root
    0xC7E1B6097F73B897ULL,  // 2^31-th root
    0x2B47F7A68FEFD787ULL,  // 2^32-th root (primitive)
};

/**
 * Initialize constant memory with pre-computed values
 * Call this once before using any kernels
 */
inline cudaError_t goldilocks_init() {
    cudaError_t err;

    err = cudaMemcpyToSymbol(d_POWERS_OF_TWO, h_POWERS_OF_TWO, sizeof(h_POWERS_OF_TWO));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(d_TWO_ADIC_GENERATORS, h_TWO_ADIC_GENERATORS, sizeof(h_TWO_ADIC_GENERATORS));
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

#endif // GOLDILOCKS_CUH

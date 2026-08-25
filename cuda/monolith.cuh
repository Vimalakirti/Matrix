/**
 * Monolith hash function for Goldilocks field — GPU implementation.
 *
 * Monolith uses a width-12 permutation with 6 rounds:
 *   Concrete(RC[0])                     // initial affine layer
 *   for round in 1..=6:
 *     Bars(state)                       // bitwise S-box on state[0..3]
 *     Bricks(state)                     // Feistel-squaring (reverse loop)
 *     Concrete(state, RC[round])        // MDS + round constants
 *
 * References:
 *   - https://eprint.iacr.org/2023/1025
 *   - research/monolith/ (Rust reference implementation)
 */

#pragma once

#include "goldilocks.cuh"

// ============================================================================
// Constants
// ============================================================================

#define MONOLITH_WIDTH     12
#define MONOLITH_RATE       8
#define MONOLITH_CAPACITY   4
#define MONOLITH_N_ROUNDS   6
#define MONOLITH_NUM_BARS   4
#define MONOLITH_DIGEST     4

// Round constants: 7 arrays of 12 u64 (initial + 6 rounds)
// LOOKUP_BITS = 8 variant for Goldilocks
__constant__ uint64_t d_MONO_RC[7][12];

// Circulant MDS matrix (12x12), first row = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]
__constant__ uint64_t d_MONO_MDS[12][12];

// Host-side constants for initialization
static const uint64_t h_MONO_RC[7][12] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {13596126580325903823ULL, 5676126986831820406ULL, 11349149288412960427ULL,
     3368797843020733411ULL, 16240671731749717664ULL, 9273190757374900239ULL,
     14446552112110239438ULL, 4033077683985131644ULL, 4291229347329361293ULL,
     13231607645683636062ULL, 1383651072186713277ULL, 8898815177417587567ULL},
    {2383619671172821638ULL, 6065528368924797662ULL, 16737578966352303081ULL,
     2661700069680749654ULL, 7414030722730336790ULL, 18124970299993404776ULL,
     9169923000283400738ULL, 15832813151034110977ULL, 16245117847613094506ULL,
     11056181639108379773ULL, 10546400734398052938ULL, 8443860941261719174ULL},
    {15799082741422909885ULL, 13421235861052008152ULL, 15448208253823605561ULL,
     2540286744040770964ULL, 2895626806801935918ULL, 8644593510196221619ULL,
     17722491003064835823ULL, 5166255496419771636ULL, 1015740739405252346ULL,
     4400043467547597488ULL, 5176473243271652644ULL, 4517904634837939508ULL},
    {18341030605319882173ULL, 13366339881666916534ULL, 6291492342503367536ULL,
     10004214885638819819ULL, 4748655089269860551ULL, 1520762444865670308ULL,
     8393589389936386108ULL, 11025183333304586284ULL, 5993305003203422738ULL,
     458912836931247573ULL, 5947003897778655410ULL, 17184667486285295106ULL},
    {15710528677110011358ULL, 8929476121507374707ULL, 2351989866172789037ULL,
     11264145846854799752ULL, 14924075362538455764ULL, 10107004551857451916ULL,
     18325221206052792232ULL, 16751515052585522105ULL, 15305034267720085905ULL,
     15639149412312342017ULL, 14624541102106656564ULL, 3542311898554959098ULL},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

static const uint64_t h_MONO_MDS[12][12] = {
    { 7, 23,  8, 26, 13, 10,  9,  7,  6, 22, 21,  8},
    { 8,  7, 23,  8, 26, 13, 10,  9,  7,  6, 22, 21},
    {21,  8,  7, 23,  8, 26, 13, 10,  9,  7,  6, 22},
    {22, 21,  8,  7, 23,  8, 26, 13, 10,  9,  7,  6},
    { 6, 22, 21,  8,  7, 23,  8, 26, 13, 10,  9,  7},
    { 7,  6, 22, 21,  8,  7, 23,  8, 26, 13, 10,  9},
    { 9,  7,  6, 22, 21,  8,  7, 23,  8, 26, 13, 10},
    {10,  9,  7,  6, 22, 21,  8,  7, 23,  8, 26, 13},
    {13, 10,  9,  7,  6, 22, 21,  8,  7, 23,  8, 26},
    {26, 13, 10,  9,  7,  6, 22, 21,  8,  7, 23,  8},
    { 8, 26, 13, 10,  9,  7,  6, 22, 21,  8,  7, 23},
    {23,  8, 26, 13, 10,  9,  7,  6, 22, 21,  8,  7},
};

inline cudaError_t monolith_init() {
    cudaError_t err;
    err = cudaMemcpyToSymbol(d_MONO_RC, h_MONO_RC, sizeof(h_MONO_RC));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_MONO_MDS, h_MONO_MDS, sizeof(h_MONO_MDS));
    return err;
}

// ============================================================================
// Goldilocks u96 reduction (from_noncanonical_u96)
// ============================================================================

// Reduce a 96-bit value (lo64, hi32) to Goldilocks field.
// p = 2^64 - 2^32 + 1, so 2^64 ≡ 2^32 - 1 (mod p).
// x = lo + hi * 2^64 ≡ lo + hi * (2^32 - 1) (mod p)
__device__ __forceinline__ GoldilocksField gl_from_u96(uint64_t lo, uint64_t hi) {
    // hi * (2^32 - 1) = (hi << 32) - hi
    // hi is at most ~40 bits (sum of 12 products of 64-bit × 5-bit),
    // so (hi << 32) fits in 96 bits, which we reduce via reduce128.
    uint128_t correction = mul_u64_u64(hi, 0xFFFFFFFFULL);  // hi * (2^32 - 1)
    // lo + correction, with potential overflow into hi word
    uint64_t sum_lo = lo + correction.lo;
    uint64_t carry = (sum_lo < lo) ? 1 : 0;
    uint64_t sum_hi = correction.hi + carry;
    return GoldilocksField(reduce128(uint128_t(sum_lo, sum_hi)));
}

// ============================================================================
// Bars layer: bitwise S-box on state[0..3] (LOOKUP_BITS = 8)
// ============================================================================

// Per-element bar function for 8-bit limbs.
// Splits 64-bit value into 8 bytes, applies y_i = x_i ^ ((~x_{i+1}) & x_{i+2} & x_{i+3})
// using byte-parallel rotations, then final rotation.
__device__ __forceinline__ uint64_t monolith_bar_64(uint64_t limb) {
    // Left-rotate each byte by 1 bit (with NOT for the +1 in (1 + x_{i+1}))
    uint64_t limbl1 = (((~limb) & 0x8080808080808080ULL) >> 7) |
                       (((~limb) & 0x7F7F7F7F7F7F7F7FULL) << 1);
    // Left-rotate each byte by 2 bits
    uint64_t limbl2 = ((limb & 0xC0C0C0C0C0C0C0C0ULL) >> 6) |
                       ((limb & 0x3F3F3F3F3F3F3F3FULL) << 2);
    // Left-rotate each byte by 3 bits
    uint64_t limbl3 = ((limb & 0xE0E0E0E0E0E0E0E0ULL) >> 5) |
                       ((limb & 0x1F1F1F1F1F1F1F1FULL) << 3);

    // y_i = x_i ^ ((1 + x_{i+1}) & x_{i+2} & x_{i+3})
    uint64_t tmp = limb ^ (limbl1 & limbl2 & limbl3);

    // Final left-rotate each byte by 1 bit
    return ((tmp & 0x8080808080808080ULL) >> 7) |
           ((tmp & 0x7F7F7F7F7F7F7F7FULL) << 1);
}

__device__ __forceinline__ void monolith_bars(GoldilocksField state[MONOLITH_WIDTH]) {
    // Bars operates on raw bits — state MUST be canonicalized first (< p).
    // gl_add may leave non-canonical values (>= p but < 2^64).
    state[0].value = monolith_bar_64(canonicalize(state[0].value));
    state[1].value = monolith_bar_64(canonicalize(state[1].value));
    state[2].value = monolith_bar_64(canonicalize(state[2].value));
    state[3].value = monolith_bar_64(canonicalize(state[3].value));
}

// ============================================================================
// Bricks layer: Feistel Type-3 (reverse iteration)
// ============================================================================

__device__ __forceinline__ void monolith_bricks(GoldilocksField state[MONOLITH_WIDTH]) {
    // state[i] += state[i-1]^2 for i = 11 down to 1
    #pragma unroll
    for (int i = MONOLITH_WIDTH - 1; i >= 1; i--) {
        GoldilocksField prev_sq = gl_mul(state[i - 1], state[i - 1]);
        state[i] = gl_add(state[i], prev_sq);
    }
}

// ============================================================================
// Concrete layer: MDS matrix multiply + round constants
// ============================================================================

// Direct 12x12 matrix multiply with u128 accumulation.
// MDS entries are at most 26 (5 bits). State values < 2^64 (may be non-canonical).
// Each product: u64 × 26 → at most 69 bits (hi ≤ 25).
// Sum of 12 products: at most 73 bits (hi ≤ 300).
// One reduce128 per row (not per product) — 12x fewer reductions than gl_mul approach.
__device__ void monolith_concrete(
    GoldilocksField state[MONOLITH_WIDTH],
    int round_idx
) {
    GoldilocksField result[MONOLITH_WIDTH];

    #pragma unroll
    for (int row = 0; row < MONOLITH_WIDTH; row++) {
        uint64_t acc_lo = 0;
        uint64_t acc_hi = 0;
        #pragma unroll
        for (int col = 0; col < MONOLITH_WIDTH; col++) {
            // state[col].value may be non-canonical (up to 2^64-1 from add_no_canonicalize).
            // MDS entry is at most 26. Product fits in 69 bits.
            uint128_t prod = mul_u64_u64(state[col].value, d_MONO_MDS[row][col]);
            // 128-bit add into accumulator
            uint64_t new_lo = acc_lo + prod.lo;
            uint64_t carry = (new_lo < acc_lo) ? 1ULL : 0ULL;
            acc_lo = new_lo;
            acc_hi += prod.hi + carry;
        }
        // Add round constant
        {
            uint64_t new_lo = acc_lo + d_MONO_RC[round_idx][row];
            uint64_t carry = (new_lo < acc_lo) ? 1ULL : 0ULL;
            acc_lo = new_lo;
            acc_hi += carry;
        }
        // Single reduce128 — acc_hi is at most ~300, so this is a light reduction
        result[row] = GoldilocksField(reduce128(uint128_t(acc_lo, acc_hi)));
    }

    #pragma unroll
    for (int i = 0; i < MONOLITH_WIDTH; i++) {
        state[i] = result[i];
    }
}

// ============================================================================
// Full Monolith permutation (width 12, 6 rounds)
// ============================================================================

__device__ void monolith_permute_12(GoldilocksField state[MONOLITH_WIDTH]) {
    // Initial Concrete layer (round 0)
    monolith_concrete(state, 0);

    // 6 full rounds
    #pragma unroll
    for (int r = 1; r <= MONOLITH_N_ROUNDS; r++) {
        monolith_bars(state);
        monolith_bricks(state);
        monolith_concrete(state, r);
    }
}

// ============================================================================
// 2-to-1 Compression (for Merkle trees)
// ============================================================================

/**
 * Compress two 4-element digests into one.
 * Rate = 8: load left[4] into positions [0..3], right[4] into [4..7].
 * Capacity = 4: positions [8..11] zeroed.
 * Output = state[0..3] after permutation.
 */
__device__ void monolith_compress(
    const GoldilocksField* left,
    const GoldilocksField* right,
    GoldilocksField* output
) {
    GoldilocksField state[MONOLITH_WIDTH];

    // Rate: left || right
    state[0] = left[0];
    state[1] = left[1];
    state[2] = left[2];
    state[3] = left[3];
    state[4] = right[0];
    state[5] = right[1];
    state[6] = right[2];
    state[7] = right[3];

    // Capacity: zeroed
    state[8]  = GoldilocksField(0);
    state[9]  = GoldilocksField(0);
    state[10] = GoldilocksField(0);
    state[11] = GoldilocksField(0);

    monolith_permute_12(state);

    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

/**
 * Compress two GF(p^2) elements into one (for Ext2 Merkle trees).
 * Left = (c0, c1), Right = (c0, c1) => 4 base field elements into rate.
 * Remaining 4 rate positions + 4 capacity positions zeroed.
 */
__device__ void monolith_compress_ext2(
    GoldilocksExt2 left,
    GoldilocksExt2 right,
    GoldilocksExt2* output
) {
    GoldilocksField state[MONOLITH_WIDTH];

    state[0] = left.c[0];
    state[1] = left.c[1];
    state[2] = right.c[0];
    state[3] = right.c[1];

    // Zero padding (remaining rate + capacity)
    state[4]  = GoldilocksField(0);
    state[5]  = GoldilocksField(0);
    state[6]  = GoldilocksField(0);
    state[7]  = GoldilocksField(0);
    state[8]  = GoldilocksField(0);
    state[9]  = GoldilocksField(0);
    state[10] = GoldilocksField(0);
    state[11] = GoldilocksField(0);

    monolith_permute_12(state);

    *output = GoldilocksExt2(state[0], state[1]);
}

/**
 * Hash two base-field elements into a 4-element digest.
 * Absorb into rate positions [0..1], zero rest, permute, output [0..3].
 */
__device__ void monolith_hash_gl_leaf(
    GoldilocksField a,
    GoldilocksField b,
    GoldilocksField* output
) {
    GoldilocksField state[MONOLITH_WIDTH];

    state[0] = a;
    state[1] = b;
    for (int i = 2; i < MONOLITH_WIDTH; i++) {
        state[i] = GoldilocksField(0);
    }

    monolith_permute_12(state);

    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

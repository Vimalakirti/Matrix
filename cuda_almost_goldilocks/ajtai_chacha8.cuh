/**
 * ChaCha8 PRG — host + device, deterministic across both.
 *
 * Used as the matrix PRG for the Ajtai commitment over almost-Goldilocks.
 * 8-round variant of ChaCha (vs the usual 20). Public differential
 * cryptanalysis breaks at most 6 rounds, so we keep a 2-round safety
 * margin for PRG-grade indistinguishability while paying 2.5× less
 * work per output block than ChaCha20.
 *
 * Each invocation of chacha8_block() produces 64 bytes (8 u64 / 16 u32)
 * keyed by (32-byte key, 12-byte nonce, 4-byte counter).
 *
 * For matrix sampling we use:
 *   key     = caller-supplied 32 bytes (typically a hash of the public seed)
 *   nonce   = (row, j_lo32, j_hi32)
 *   counter = block_idx ∈ [0, 8) for the eight 64-byte chunks
 *             that make up a 64-coefficient ring element. On rejection,
 *             the counter advances to fallback positions 8, 16, 24, ...
 *             unique per block_idx — see prg_ring_block_chacha8().
 */

#ifndef AJTAI_CHACHA8_CUH
#define AJTAI_CHACHA8_CUH

#include <stdint.h>
#include "almost_goldilocks.cuh"

#ifndef __host__
#define __host__
#define __device__
#define __forceinline__ inline
#endif

#define AJTAI_ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define AJTAI_CHACHA_QR(a, b, c, d)                       \
    do {                                                  \
        a += b; d ^= a; d = AJTAI_ROTL32(d, 16);          \
        c += d; b ^= c; b = AJTAI_ROTL32(b, 12);          \
        a += b; d ^= a; d = AJTAI_ROTL32(d, 8);           \
        c += d; b ^= c; b = AJTAI_ROTL32(b, 7);           \
    } while (0)

/**
 * One ChaCha8 block. Produces 64 bytes / 16 u32 of keystream.
 *
 * key      32 bytes (8 u32 little-endian)
 * counter  32-bit block counter
 * nonce    12 bytes (3 u32 little-endian)
 * out[16]  16 u32 output (8 u64 of keystream)
 *
 * Marked __noinline__ on device so the ChaCha working state (16 u32 init
 * + 16 u32 round state ≈ 32 32-bit registers) lives only inside this
 * function's frame, not across the kernel's whole lifetime. The kernel's
 * accumulator state and ChaCha state thus do not co-occupy the register
 * file, dropping per-thread register pressure substantially. Function-
 * call overhead is ~30 cycles, negligible vs the savings from higher
 * occupancy.
 */
__host__ __device__ __noinline__
void chacha8_block(
    const uint32_t key[8],
    uint32_t counter,
    const uint32_t nonce[3],
    uint32_t out[16]
) {
    uint32_t s[16];
    s[ 0] = 0x61707865u; s[ 1] = 0x3320646eu;
    s[ 2] = 0x79622d32u; s[ 3] = 0x6b206574u;
    s[ 4] = key[0];      s[ 5] = key[1];
    s[ 6] = key[2];      s[ 7] = key[3];
    s[ 8] = key[4];      s[ 9] = key[5];
    s[10] = key[6];      s[11] = key[7];
    s[12] = counter;
    s[13] = nonce[0];    s[14] = nonce[1];    s[15] = nonce[2];

    uint32_t x[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) x[i] = s[i];

    // 4 double rounds  ==  8 single rounds  ==  ChaCha8
    #pragma unroll
    for (int dr = 0; dr < 4; dr++) {
        // column rounds
        AJTAI_CHACHA_QR(x[0], x[4], x[ 8], x[12]);
        AJTAI_CHACHA_QR(x[1], x[5], x[ 9], x[13]);
        AJTAI_CHACHA_QR(x[2], x[6], x[10], x[14]);
        AJTAI_CHACHA_QR(x[3], x[7], x[11], x[15]);
        // diagonal rounds
        AJTAI_CHACHA_QR(x[0], x[5], x[10], x[15]);
        AJTAI_CHACHA_QR(x[1], x[6], x[11], x[12]);
        AJTAI_CHACHA_QR(x[2], x[7], x[ 8], x[13]);
        AJTAI_CHACHA_QR(x[3], x[4], x[ 9], x[14]);
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
}

/**
 * Generate 8 coefficients of a ring element (one ChaCha block's worth of
 * output bytes, interpreted as u64s), rejection-sampled to F_q.
 *
 * Rejection probability per u64 is ≈ 2^-31 (q is within 2^33 of 2^64), so
 * the loop almost never iterates more than once. When it does, the
 * counter advances to a deterministic fallback range:
 *
 *   primary:    counter = block_idx                  (uses up to 8 u64s)
 *   fallback k: counter = block_idx + 8 * k         (uses first 8 u64s)
 *
 * Different (row, j, block_idx) tuples use disjoint counter ranges
 * (the row and j are baked into the nonce), so rejection retries can
 * never collide and the output is deterministic given (key, row, j).
 */
__host__ __device__ __forceinline__
void prg_ring_block_chacha8(
    const uint32_t key[8],
    uint32_t       row,        // 0..14
    uint64_t       j,          // 0..N
    uint32_t       block_idx,  // 0..7
    uint64_t       out_coeffs[8]
) {
    uint32_t nonce[3];
    nonce[0] = row;
    nonce[1] = (uint32_t)(j & 0xFFFFFFFFu);
    nonce[2] = (uint32_t)(j >> 32);

    // Zero-init defensively: in the astronomically unlikely event that all
    // 16 retries reject (P ≈ 2^-496), the unfilled slots stay at 0 instead
    // of uninitialized — turns a latent UB into deterministic-but-biased
    // output for that single coefficient, which is still safe to use.
    #pragma unroll
    for (int k = 0; k < 8; k++) out_coeffs[k] = 0;

    int written = 0;
    // 16 retries is wildly more than ever needed (P[≥1 retry] ≈ 2^-31,
    // P[≥16 retries] ≈ 2^-(31*16) = 2^-496). Bounded loop keeps the
    // kernel from theoretically running forever on a pathological PRG state.
    for (uint32_t retry = 0; retry < 16 && written < 8; retry++) {
        uint32_t counter = block_idx + retry * 8u;
        uint32_t buf[16];
        chacha8_block(key, counter, nonce, buf);

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            uint64_t lo = (uint64_t)buf[2*k];
            uint64_t hi = (uint64_t)buf[2*k + 1];
            uint64_t s = (hi << 32) | lo;
            if (s < ALMOST_GOLDILOCKS_PRIME) {
                out_coeffs[written++] = s;
                if (written == 8) break;
            }
        }
    }
}

#endif // AJTAI_CHACHA8_CUH

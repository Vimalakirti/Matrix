// Shared production Keccak core for addressed public keys and private coins.
// The host implementation executes identical arithmetic for differential tests.
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define MSHAKE_INLINE __device__ __forceinline__
#define MSHAKE_CONSTANT __device__ __constant__
#else
#define MSHAKE_INLINE inline
#define MSHAKE_CONSTANT inline constexpr
#endif
namespace mshake {
typedef unsigned long long u64;
// ---------------------------------------------------------------- Keccak
MSHAKE_CONSTANT u64 KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

MSHAKE_INLINE u64 rotl64(u64 x, int s) { return (x << s) | (x >> (64 - s)); }

MSHAKE_INLINE void keccak_f(u64 a[25]) {
#pragma unroll 1
    for (int r = 0; r < 24; r++) {
        u64 c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
        u64 c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
        u64 c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
        u64 c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
        u64 c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
        u64 d0 = c4 ^ rotl64(c1, 1), d1 = c0 ^ rotl64(c2, 1), d2 = c1 ^ rotl64(c3, 1);
        u64 d3 = c2 ^ rotl64(c4, 1), d4 = c3 ^ rotl64(c0, 1);
#pragma unroll
        for (int j = 0; j < 25; j += 5) { a[j] ^= d0; a[j + 1] ^= d1; a[j + 2] ^= d2; a[j + 3] ^= d3; a[j + 4] ^= d4; }
        // rho + pi
        u64 b[25];
        b[0] = a[0];
        b[10] = rotl64(a[1], 1);   b[20] = rotl64(a[2], 62);  b[5] = rotl64(a[3], 28);   b[15] = rotl64(a[4], 27);
        b[16] = rotl64(a[5], 36);  b[1] = rotl64(a[6], 44);   b[11] = rotl64(a[7], 6);   b[21] = rotl64(a[8], 55);
        b[6] = rotl64(a[9], 20);   b[7] = rotl64(a[10], 3);   b[17] = rotl64(a[11], 10); b[2] = rotl64(a[12], 43);
        b[12] = rotl64(a[13], 25); b[22] = rotl64(a[14], 39); b[23] = rotl64(a[15], 41); b[8] = rotl64(a[16], 45);
        b[18] = rotl64(a[17], 15); b[3] = rotl64(a[18], 21);  b[13] = rotl64(a[19], 8);  b[14] = rotl64(a[20], 18);
        b[24] = rotl64(a[21], 2);  b[9] = rotl64(a[22], 61);  b[19] = rotl64(a[23], 56); b[4] = rotl64(a[24], 14);
        // chi
#pragma unroll
        for (int j = 0; j < 25; j += 5) {
            u64 t0 = b[j], t1 = b[j + 1], t2 = b[j + 2], t3 = b[j + 3], t4 = b[j + 4];
            a[j] = t0 ^ (~t1 & t2); a[j + 1] = t1 ^ (~t2 & t3); a[j + 2] = t2 ^ (~t3 & t4);
            a[j + 3] = t3 ^ (~t4 & t0); a[j + 4] = t4 ^ (~t0 & t1);
        }
        a[0] ^= KECCAK_RC[r];
    }
}


// SHAKE256 of the fixed 23-byte domain plus 48-byte address, first 64 bytes.
MSHAKE_INLINE void key_block(const uint32_t key[8], uint32_t counter,
                            const uint32_t nonce[3], uint32_t out[16]) {
    const unsigned char tag[]="MORPHEUS/KEY/SHAKE256/v2"; // include C terminator
    u64 st[25]={0};
    int offset=0;
    for (unsigned i=0;i<sizeof(tag);++i,++offset)
        st[offset/8]^=(u64)tag[i]<<(8*(offset%8));
    for (int i=0;i<12;++i) {
        uint32_t w=i<8?key[i]:(i==8?counter:nonce[i-9]);
        for (int j=0;j<4;++j,++offset)
            st[offset/8]^=(u64)((w>>(8*j))&255u)<<(8*(offset%8));
    }
    st[offset/8]^=0x1fULL<<(8*(offset%8)); // SHAKE suffix
    st[16]^=0x8000000000000000ULL; // 136-byte rate padding
    keccak_f(st);
    for (int i=0;i<16;++i) out[i]=(uint32_t)(st[i/2]>>(32*(i%2)));
}
} // namespace mshake
#undef MSHAKE_INLINE
#undef MSHAKE_CONSTANT

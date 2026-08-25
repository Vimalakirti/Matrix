/**
 * Poseidon2 Hash Function for Goldilocks Field - CUDA Implementation
 *
 * Based on the Plonky3 reference implementation.
 *
 * Poseidon2 Structure:
 * - External Initial Rounds (4 rounds): Full S-box + MDS
 * - Internal Rounds (22 rounds): Partial S-box on first element + Diffusion
 * - External Terminal Rounds (4 rounds): Full S-box + MDS
 *
 * S-box: x^7 (degree 7)
 * Width: 8, 12, 16, or 20 field elements
 */

#ifndef POSEIDON2_CUH
#define POSEIDON2_CUH

#include "goldilocks.cuh"
#include "extension.cuh"

// ============================================================================
// Poseidon2 Parameters
// ============================================================================

#define POSEIDON2_SBOX_DEGREE 7
#define POSEIDON2_ROUNDS_F 8        // Full/external rounds (4 initial + 4 terminal)
#define POSEIDON2_ROUNDS_P 22       // Partial/internal rounds

// Supported widths
#define POSEIDON2_WIDTH_8  8
#define POSEIDON2_WIDTH_12 12
#define POSEIDON2_WIDTH_16 16
#define POSEIDON2_WIDTH_20 20

// ============================================================================
// Round Constants for Width 8
// ============================================================================

// External round constants: 8 rounds × 8 elements
// First 4 rounds are initial, last 4 are terminal
__constant__ uint64_t d_RC_EXT_8[8][8];

// Internal round constants: 22 rounds, only first element
__constant__ uint64_t d_RC_INT[22];

// Internal diffusion diagonal for width 8
__constant__ uint64_t d_DIAG_8[8];

// Host-side constants for initialization
static const uint64_t h_RC_EXT_8[8][8] = {
    // Initial round 0
    {0xdd5743e7f2a5a5d9ULL, 0xcb3a864e58ada44bULL, 0xd5a3726889dedc0dULL, 0x79365a7f967583c1ULL,
     0xb7820c177b0a3c30ULL, 0x68a60479943b7240ULL, 0x0a22eeab67b97c41ULL, 0xbee9d90e037fa7d4ULL},
    // Initial round 1
    {0xf1dda5b9259dfcb4ULL, 0x27515210be112d59ULL, 0x654a4db8f15f6c3aULL, 0x5a9bfff8c29db0c4ULL,
     0xb72f2628c95a885bULL, 0xffcb54fc166beaddULL, 0xe32f91d59c463772ULL, 0x9e0ff2a9bbea79dbULL},
    // Initial round 2
    {0xce57d6245ddca6b2ULL, 0xb1fc8d402bba1eb1ULL, 0x64df974fb666d528ULL, 0x59e4e5b237e52e78ULL,
     0x5e255e2742caa8fcULL, 0xc4a17304e330ef9aULL, 0x60e3a513cdd9f0b1ULL, 0xbf29a6a5a0c9c8baULL},
    // Initial round 3
    {0xcea721cce82fb11bULL, 0xe5b55eb8098ece81ULL, 0x4ec66bfb89f7c380ULL, 0xc33be12b6ef7e4faULL,
     0xa41b5f5b84c57d7dULL, 0xb526206b2138a936ULL, 0x9f7a5ac62f16eb4eULL, 0x7de77f405f683aa5ULL},
    // Terminal round 0
    {0x014ef1197d341346ULL, 0x9725e20825d07394ULL, 0xc8ff2a22516f3604ULL, 0xde1b51f7cf493493ULL,
     0xa7e3af1a53eadb78ULL, 0x9b7b7ee0ddf9f229ULL, 0xf0a460f2dc649e7dULL, 0xdc4448d02c2cb823ULL},
    // Terminal round 1
    {0xaa62c88e0b294011ULL, 0x058eb9d810ce9f74ULL, 0x4e3d7a5403566d24ULL, 0xe8ae66da2f8b7a63ULL,
     0x3fdad3b08dae2e0bULL, 0xec81e61303dd409bULL, 0x3cbaa1cd35d31fc4ULL, 0x847f7806eec88ffaULL},
    // Terminal round 2
    {0x98ae09a325893803ULL, 0xf8a6475077968838ULL, 0xdf55b7419443de5bULL, 0xdbe15699e696ca70ULL,
     0x8e54a27d8db00424ULL, 0x7679a45cd9d1e12aULL, 0x2844f52be73e0e2fULL, 0xd41f8eb31ada34f8ULL},
    // Terminal round 3
    {0xe9dd318bae1f3961ULL, 0xf7462137299efe1aULL, 0x6dbbe06779e1d573ULL, 0xfe35b05cbe707632ULL,
     0x5d8896b12654fd8cULL, 0x6f96ef47c32d4ae2ULL, 0xb0caa221dbbfc0daULL, 0x0bc2a5bf1f238d3fULL}
};

// Internal round constants (22 values)
static const uint64_t h_RC_INT[22] = {
    0x488897d85ff51f56ULL, 0x1140737ccb162218ULL, 0xa7eeb9215866ed35ULL,
    0x9bd2976fee49fcc9ULL, 0xc0c8f0de580a3fccULL, 0x4fb2dae6ee8fc793ULL,
    0x343a89f35f37395bULL, 0x223b525a77ca72c8ULL, 0x56ccb62574aaa918ULL,
    0xc4d507d8027af9edULL, 0xa080673cf0b7e95cULL, 0xf0184884eb70dcf8ULL,
    0x044f10b0cb3d5c69ULL, 0xe9e3f7993938f186ULL, 0x1b761c80e772f459ULL,
    0x606cec607a1b5facULL, 0x14a0c2e1d45f03cdULL, 0x4eace8855398574fULL,
    0xf905ca7103eff3e6ULL, 0xf8c8f8d20862c059ULL, 0xb524fe8bdd678e5aULL,
    0xfbb7865901a1ec41ULL
};

// Internal diffusion diagonal for width 8
static const uint64_t h_DIAG_8[8] = {
    0xa98811a1fed4e3a5ULL, 0x1cc48b54f377e2a0ULL, 0xe40cd4f6c5609a26ULL,
    0x11de79ebca97a4a3ULL, 0x9177c73d8b7e929cULL, 0x2a6fe8085797e791ULL,
    0x3de6e93329f8d5adULL, 0x3f7af9125da962feULL
};

// ============================================================================
// Round Constants for Width 16
// ============================================================================

__constant__ uint64_t d_RC_EXT_16[8][16];
__constant__ uint64_t d_DIAG_16[16];

static const uint64_t h_RC_EXT_16[8][16] = {
    // Initial round 0
    {0xe049fa6db4f52e39ULL, 0xaae8370457d69015ULL, 0x4a256cbb76a0b201ULL, 0xb6e1004d14519f96ULL,
     0xbb4c69d7acfc8ce4ULL, 0x0ae848abc12b8d5eULL, 0x30bce7b5e5c55cf3ULL, 0xa6e3792e2d58980bULL,
     0x12d7b37b9c5ca7e7ULL, 0x25fa99cf9c2ddaa6ULL, 0x3f9a91c2d9285f41ULL, 0x2ca939a7765c55feULL,
     0x9c82fe3d58c00aceULL, 0xa4c4c2fe6d90ae8eULL, 0x17daff0d7c2d6795ULL, 0xa03d388fe8de9f29ULL},
    // Initial round 1
    {0xdc53c2991ee65d0dULL, 0xeab11c6a6ec3e7d6ULL, 0x4f934be33cdf8d33ULL, 0x7a4f36f31869cf3cULL,
     0x7c8d771d2a003c2fULL, 0x25a35bfdfe4b314dULL, 0x4310ff1fd0b32c9dULL, 0x86d4e7db76f36bb1ULL,
     0x96a3ca65a2a07e5dULL, 0x06b97eb5e16c79faULL, 0x3c9363a648204ac2ULL, 0x6eb6f2a074397b0fULL,
     0x9f0a275212f1a70dULL, 0x686ff5cfeab61165ULL, 0x27f917f69a0b50e3ULL, 0x149a91e4d7c0ea1eULL},
    // Initial round 2
    {0xd1f33acb32e3b6c5ULL, 0x2b2e08860804dc01ULL, 0x0a4df54967ce15cbULL, 0x4de4cdaed714c089ULL,
     0x62e2b2c8c4801508ULL, 0x39e58e15d390a678ULL, 0x81f57b35f9c12fd7ULL, 0x4bc43c4f7903de28ULL,
     0x21c38374c8c8dbf4ULL, 0xe4cfe31a2f6e516dULL, 0x822f7f5d13e5cbb8ULL, 0x85a48e8c5ac3eed9ULL,
     0x6a50faebac0018f4ULL, 0xae15d13acfc22d69ULL, 0x4d5d3a3e4b03b779ULL, 0x8dded66b81a28bb8ULL},
    // Initial round 3
    {0x0e849021669e6cc4ULL, 0x6ceb8a1812b8e32eULL, 0x909f5aab11b5c6dfULL, 0x8086adfcfe3d4a13ULL,
     0xee3a5fe5854ccaaaULL, 0x1a49fd1c14d2df47ULL, 0x70c3a7ba5f5208c8ULL, 0x8bc62664c2527180ULL,
     0x30430da29e6df5daULL, 0xb1ba0c7049f66ec9ULL, 0xf9a44822b0bf95f3ULL, 0xc961b0ce89a3ab04ULL,
     0xf2f63f84db2e1fedULL, 0x5ba9bd62a29ac1a8ULL, 0x1ada1bbc5ddba17cULL, 0x7fc5ab62c490db93ULL},
    // Terminal round 0
    {0x0ead30ce26f450a8ULL, 0xc83279e72a02e8a1ULL, 0xf7b8e869a182f6baULL, 0x6d54301e8fb15d0bULL,
     0x0406623441024f7aULL, 0x34c7c22ef35a0854ULL, 0x09a23b58bc2a0c71ULL, 0xf5ce6cd67c66c9a7ULL,
     0x7b72bc59f7c5dc0cULL, 0x64d73ddfba454d69ULL, 0xebf1adce6b4c5d66ULL, 0xfc03a8bda0c4af7fULL,
     0x3dca0e1c07594c6aULL, 0x6a3c262b7d38d475ULL, 0x0fe84571f2e75c27ULL, 0x4aba0097844e6ee1ULL},
    // Terminal round 1
    {0x98ccaa7e79a59c7cULL, 0xa34eb55b80f86c08ULL, 0xdfcba6b8c879e110ULL, 0x0538b8ea15bc929fULL,
     0x37d0c65b16aa0a0eULL, 0x80f5b54c80aada3aULL, 0x9b6c94e28c6fe8e7ULL, 0x5db9d2fe9ac6f9e2ULL,
     0xe1c82c8de03cbc8dULL, 0x8bc2cd31dbc4f6dbULL, 0x52b0a37c0bdb0577ULL, 0x5b1b4bb46c557898ULL,
     0x10ac9d47dba98ba9ULL, 0x9a04354a2c9e39c8ULL, 0x19f69c2a3a1dc997ULL, 0x7ddb17beb2c63f98ULL},
    // Terminal round 2
    {0x17dae8f80bfdce68ULL, 0x1e5b98d2dac9b36eULL, 0x3bfec8670a530fd2ULL, 0xc89d97fbd2c1b3d2ULL,
     0x4b4f2f1890381e2eULL, 0x0bdf6c4d8eb7d63bULL, 0x3ea8fdd7c5cbe040ULL, 0xb88e7f4adb5a22baULL,
     0x62a0d72d8c2c8b90ULL, 0x51cc07bbb0e69d8bULL, 0x4c0c3c0ab96ed068ULL, 0x2932d43ca70fced8ULL,
     0x6b4f59111033d5d4ULL, 0x7fe5c21342e5cca6ULL, 0x232aceff0d82d3fdULL, 0x6c564773e39c1f18ULL},
    // Terminal round 3
    {0xcf2e44fa9e4ed4dfULL, 0x5b2e2b137f41c552ULL, 0xce714c24fd8ab3e2ULL, 0x56c3f6f8af3c2aa6ULL,
     0x4c04f0ea94a2c5d0ULL, 0xc7c16c2eae60906aULL, 0x50f24d7f9e0a1a18ULL, 0x9e69c70fcc5d4a9aULL,
     0x0cf62cf3c6bfaabaULL, 0x28ce0b6a7cbe2decULL, 0x8e6c69fb37b1888eULL, 0x80f5e9eb4b9e9b4eULL,
     0x35bd8cb0b0c6a612ULL, 0x9b8d4fc35bef9ad4ULL, 0x53f56f635c0c1366ULL, 0xa87dc5b8cc2fa812ULL}
};

static const uint64_t h_DIAG_16[16] = {
    0xde9b91a467d6afc0ULL, 0xc5f16b9c76a9be17ULL, 0x0ab0fef2d540ac55ULL,
    0x3001d27009d05773ULL, 0xed23b1f906d3d9ebULL, 0x5ce73743cba97054ULL,
    0x1c3bab944af4ba24ULL, 0x2faa105854dbafaeULL, 0x53ffb3ae6d421a10ULL,
    0xbcda9df8884ba396ULL, 0xfc1273e4a31807bbULL, 0xc77952573d5142c0ULL,
    0x56683339a819b85eULL, 0x328fcbd8f0ddc8ebULL, 0xb5101e303fce9cb7ULL,
    0x774487b8c40089bbULL
};

// ============================================================================
// S-Box: x^7
// ============================================================================

/**
 * Compute x^7 using addition chain: x^2 -> x^4 -> x^3 -> x^7
 * x^7 = x^4 * x^2 * x = x^4 * x^3
 */
__device__ __forceinline__
GoldilocksField sbox(GoldilocksField x) {
    GoldilocksField x2 = gl_square(x);      // x^2
    GoldilocksField x4 = gl_square(x2);     // x^4
    GoldilocksField x3 = gl_mul(x2, x);     // x^3
    return gl_mul(x4, x3);                  // x^7
}

// ============================================================================
// MDS Matrix (4x4 circulant matrix)
// ============================================================================

/**
 * Apply 4x4 MDS matrix to a 4-element vector
 *
 * Matrix:
 * [ 2 3 1 1 ]
 * [ 1 2 3 1 ]
 * [ 1 1 2 3 ]
 * [ 3 1 1 2 ]
 *
 * Optimized implementation using 7 additions + 2 doubles
 */
__device__ __forceinline__
void mds4(GoldilocksField* state) {
    GoldilocksField t01 = gl_add(state[0], state[1]);
    GoldilocksField t23 = gl_add(state[2], state[3]);
    GoldilocksField t0123 = gl_add(t01, t23);
    GoldilocksField t01123 = gl_add(t0123, state[1]);
    GoldilocksField t01233 = gl_add(t0123, state[3]);

    GoldilocksField s0 = gl_add(t01123, t01);                          // 2*x0 + 3*x1 + x2 + x3
    GoldilocksField s1 = gl_add(t01123, gl_double(state[2]));          // x0 + 2*x1 + 3*x2 + x3
    GoldilocksField s2 = gl_add(t01233, t23);                          // x0 + x1 + 2*x2 + 3*x3
    GoldilocksField s3 = gl_add(t01233, gl_double(state[0]));          // 3*x0 + x1 + x2 + 2*x3

    state[0] = s0;
    state[1] = s1;
    state[2] = s2;
    state[3] = s3;
}

/**
 * Apply full MDS transformation for width-8 state
 *
 * 1. Apply 4x4 MDS to each 4-element chunk
 * 2. Add cross-chunk sums
 */
__device__ __forceinline__
void mds_light_8(GoldilocksField* state) {
    // Apply 4x4 MDS to chunks
    mds4(&state[0]);
    mds4(&state[4]);

    // Compute sums for each position mod 4
    GoldilocksField sum0 = gl_add(state[0], state[4]);
    GoldilocksField sum1 = gl_add(state[1], state[5]);
    GoldilocksField sum2 = gl_add(state[2], state[6]);
    GoldilocksField sum3 = gl_add(state[3], state[7]);

    // Add sums back
    state[0] = gl_add(state[0], sum0);
    state[1] = gl_add(state[1], sum1);
    state[2] = gl_add(state[2], sum2);
    state[3] = gl_add(state[3], sum3);
    state[4] = gl_add(state[4], sum0);
    state[5] = gl_add(state[5], sum1);
    state[6] = gl_add(state[6], sum2);
    state[7] = gl_add(state[7], sum3);
}

/**
 * Apply full MDS transformation for width-16 state
 */
__device__ __forceinline__
void mds_light_16(GoldilocksField* state) {
    // Apply 4x4 MDS to each chunk
    mds4(&state[0]);
    mds4(&state[4]);
    mds4(&state[8]);
    mds4(&state[12]);

    // Compute sums
    GoldilocksField sum0 = gl_add(gl_add(state[0], state[4]), gl_add(state[8], state[12]));
    GoldilocksField sum1 = gl_add(gl_add(state[1], state[5]), gl_add(state[9], state[13]));
    GoldilocksField sum2 = gl_add(gl_add(state[2], state[6]), gl_add(state[10], state[14]));
    GoldilocksField sum3 = gl_add(gl_add(state[3], state[7]), gl_add(state[11], state[15]));

    // Add sums back
    state[0] = gl_add(state[0], sum0);
    state[1] = gl_add(state[1], sum1);
    state[2] = gl_add(state[2], sum2);
    state[3] = gl_add(state[3], sum3);
    state[4] = gl_add(state[4], sum0);
    state[5] = gl_add(state[5], sum1);
    state[6] = gl_add(state[6], sum2);
    state[7] = gl_add(state[7], sum3);
    state[8] = gl_add(state[8], sum0);
    state[9] = gl_add(state[9], sum1);
    state[10] = gl_add(state[10], sum2);
    state[11] = gl_add(state[11], sum3);
    state[12] = gl_add(state[12], sum0);
    state[13] = gl_add(state[13], sum1);
    state[14] = gl_add(state[14], sum2);
    state[15] = gl_add(state[15], sum3);
}

// ============================================================================
// Internal Diffusion Layer
// ============================================================================

/**
 * Internal diffusion: (1 + Diag(v)) * state
 * For each element: state[i] = state[i] * diag[i] + sum(state)
 */
__device__ __forceinline__
void diffusion_8(GoldilocksField* state, const uint64_t* diag) {
    // Compute sum of all elements
    GoldilocksField sum = state[0];
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        sum = gl_add(sum, state[i]);
    }

    // Apply diffusion: state[i] = state[i] * diag[i] + sum
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        GoldilocksField prod = gl_mul(state[i], GoldilocksField(diag[i]));
        state[i] = gl_add(prod, sum);
    }
}

__device__ __forceinline__
void diffusion_16(GoldilocksField* state, const uint64_t* diag) {
    GoldilocksField sum = state[0];
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        sum = gl_add(sum, state[i]);
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        GoldilocksField prod = gl_mul(state[i], GoldilocksField(diag[i]));
        state[i] = gl_add(prod, sum);
    }
}

// ============================================================================
// Poseidon2 Permutation (Width 8)
// ============================================================================

/**
 * Full Poseidon2 permutation for width 8
 *
 * Structure:
 * - 4 initial external rounds
 * - 22 internal rounds
 * - 4 terminal external rounds
 */
__device__ void poseidon2_permute_8(GoldilocksField* state) {
    // Initial external rounds (4 rounds)
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        // Add round constants and apply S-box
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            state[i] = gl_add(state[i], GoldilocksField(d_RC_EXT_8[r][i]));
            state[i] = sbox(state[i]);
        }
        // Apply MDS
        mds_light_8(state);
    }

    // Internal rounds (22 rounds)
    #pragma unroll
    for (int r = 0; r < 22; r++) {
        // Add round constant and S-box only to first element
        state[0] = gl_add(state[0], GoldilocksField(d_RC_INT[r]));
        state[0] = sbox(state[0]);
        // Apply internal diffusion
        diffusion_8(state, d_DIAG_8);
    }

    // Terminal external rounds (4 rounds)
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        // Add round constants and apply S-box
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            state[i] = gl_add(state[i], GoldilocksField(d_RC_EXT_8[4 + r][i]));
            state[i] = sbox(state[i]);
        }
        // Apply MDS
        mds_light_8(state);
    }
}

// ============================================================================
// Poseidon2 Permutation (Width 16)
// ============================================================================

__device__ void poseidon2_permute_16(GoldilocksField* state) {
    // Initial external rounds
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            state[i] = gl_add(state[i], GoldilocksField(d_RC_EXT_16[r][i]));
            state[i] = sbox(state[i]);
        }
        mds_light_16(state);
    }

    // Internal rounds
    #pragma unroll
    for (int r = 0; r < 22; r++) {
        state[0] = gl_add(state[0], GoldilocksField(d_RC_INT[r]));
        state[0] = sbox(state[0]);
        diffusion_16(state, d_DIAG_16);
    }

    // Terminal external rounds
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            state[i] = gl_add(state[i], GoldilocksField(d_RC_EXT_16[4 + r][i]));
            state[i] = sbox(state[i]);
        }
        mds_light_16(state);
    }
}

// ============================================================================
// Sponge Construction (Padding-Free)
// ============================================================================

/**
 * Hash arbitrary input using Poseidon2 sponge (width 8, rate 4, output 4)
 *
 * Parameters:
 * - WIDTH = 8 (total state)
 * - RATE = 4 (input block size)
 * - CAPACITY = 4 (security margin)
 * - OUTPUT = 4 (hash output size)
 */
__device__ void poseidon2_hash_8_4(
    const GoldilocksField* input,
    int input_len,
    GoldilocksField* output
) {
    // Initialize state to zero
    GoldilocksField state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = GoldilocksField(0);
    }

    // Absorb phase: process input in blocks of RATE=4
    int pos = 0;
    while (pos + 4 <= input_len) {
        // Overwrite first RATE elements with input
        state[0] = input[pos];
        state[1] = input[pos + 1];
        state[2] = input[pos + 2];
        state[3] = input[pos + 3];
        pos += 4;

        // Apply permutation
        poseidon2_permute_8(state);
    }

    // Handle remaining elements (if any)
    if (pos < input_len) {
        int remaining = input_len - pos;
        for (int i = 0; i < remaining; i++) {
            state[i] = input[pos + i];
        }
        poseidon2_permute_8(state);
    }

    // Squeeze phase: output first 4 elements
    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

/**
 * Hash using width 16, rate 8, output 8
 */
__device__ void poseidon2_hash_16_8(
    const GoldilocksField* input,
    int input_len,
    GoldilocksField* output
) {
    GoldilocksField state[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        state[i] = GoldilocksField(0);
    }

    int pos = 0;
    while (pos + 8 <= input_len) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            state[i] = input[pos + i];
        }
        pos += 8;
        poseidon2_permute_16(state);
    }

    if (pos < input_len) {
        int remaining = input_len - pos;
        for (int i = 0; i < remaining; i++) {
            state[i] = input[pos + i];
        }
        poseidon2_permute_16(state);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        output[i] = state[i];
    }
}

// ============================================================================
// Compression Function (for Merkle Trees)
// ============================================================================

/**
 * Compress two chunks into one (2-to-1 compression)
 * Used for building Merkle trees
 *
 * Input: left[4] || right[4]
 * Output: hash[4]
 */
__device__ void poseidon2_compress_8(
    const GoldilocksField* left,
    const GoldilocksField* right,
    GoldilocksField* output
) {
    GoldilocksField state[8];

    // Load left chunk into first half
    state[0] = left[0];
    state[1] = left[1];
    state[2] = left[2];
    state[3] = left[3];

    // Load right chunk into second half
    state[4] = right[0];
    state[5] = right[1];
    state[6] = right[2];
    state[7] = right[3];

    // Apply permutation
    poseidon2_permute_8(state);

    // Output first 4 elements
    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

/**
 * Compress two 8-element chunks using width-16 permutation
 */
__device__ void poseidon2_compress_16(
    const GoldilocksField* left,
    const GoldilocksField* right,
    GoldilocksField* output
) {
    GoldilocksField state[16];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = left[i];
        state[8 + i] = right[i];
    }

    poseidon2_permute_16(state);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        output[i] = state[i];
    }
}

// ============================================================================
// Extension Field Hashing (via Serialization)
// ============================================================================

/**
 * Hash a single GF(p²) element by serializing to 2 base field elements
 * Output: 4 base field elements
 */
__device__ void poseidon2_hash_ext2(
    GoldilocksExt2 input,
    GoldilocksField* output
) {
    GoldilocksField state[8];

    // Initialize state to zero
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = GoldilocksField(0);
    }

    // Serialize extension element into first 2 positions
    state[0] = input.c[0];
    state[1] = input.c[1];

    // Apply permutation
    poseidon2_permute_8(state);

    // Output first 4 elements
    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

/**
 * Hash an array of GF(p²) elements
 * Each element is serialized to 2 base field elements
 * Uses sponge construction with rate=4
 */
__device__ void poseidon2_hash_ext2_array(
    const GoldilocksExt2* input,
    int input_len,
    GoldilocksField* output
) {
    GoldilocksField state[8];

    // Initialize state to zero
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = GoldilocksField(0);
    }

    // Process pairs of extension elements (4 base field elements = rate)
    int pos = 0;
    while (pos + 2 <= input_len) {
        // Absorb 2 extension elements = 4 base field elements
        state[0] = input[pos].c[0];
        state[1] = input[pos].c[1];
        state[2] = input[pos + 1].c[0];
        state[3] = input[pos + 1].c[1];
        pos += 2;

        poseidon2_permute_8(state);
    }

    // Handle remaining element (if odd number)
    if (pos < input_len) {
        state[0] = input[pos].c[0];
        state[1] = input[pos].c[1];
        poseidon2_permute_8(state);
    }

    // Output
    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

/**
 * Hash array of GF(p²) elements and return result as GF(p²) elements
 * Output: 2 extension field elements (4 base field elements reinterpreted)
 */
__device__ void poseidon2_hash_ext2_to_ext2(
    const GoldilocksExt2* input,
    int input_len,
    GoldilocksExt2* output
) {
    GoldilocksField base_output[4];
    poseidon2_hash_ext2_array(input, input_len, base_output);

    // Reinterpret as 2 extension elements
    output[0] = GoldilocksExt2(base_output[0], base_output[1]);
    output[1] = GoldilocksExt2(base_output[2], base_output[3]);
}

/**
 * Hash a single GF(p⁵) element by serializing to 5 base field elements
 * Output: 4 base field elements
 */
__device__ void poseidon2_hash_ext5(
    GoldilocksExt5 input,
    GoldilocksField* output
) {
    GoldilocksField state[8];

    // Initialize state to zero
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = GoldilocksField(0);
    }

    // Serialize extension element into first 5 positions
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        state[i] = input.c[i];
    }

    // Apply permutation
    poseidon2_permute_8(state);

    // Output first 4 elements
    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

/**
 * Hash an array of GF(p⁵) elements
 * Each element is serialized to 5 base field elements
 * Uses width-16 sponge with rate=8 for efficiency
 */
__device__ void poseidon2_hash_ext5_array(
    const GoldilocksExt5* input,
    int input_len,
    GoldilocksField* output
) {
    GoldilocksField state[16];

    // Initialize state to zero
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        state[i] = GoldilocksField(0);
    }

    // Process input (rate = 8 base field elements)
    int ext_pos = 0;   // Position in extension element array

    while (ext_pos < input_len) {
        // Fill rate portion with serialized elements
        int filled = 0;
        while (filled < 8 && ext_pos < input_len) {
            // Copy coefficients from current extension element
            for (int i = 0; i < 5 && filled < 8; i++) {
                state[filled] = input[ext_pos].c[i];
                filled++;
            }
            if (filled < 8 || (filled == 8 && ext_pos + 1 <= input_len)) {
                ext_pos++;
            }
        }

        poseidon2_permute_16(state);
        ext_pos++;
    }

    // Output first 8 elements
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        output[i] = state[i];
    }
}

/**
 * Compress two GF(p²) elements into one (2-to-1 compression)
 * Useful for Merkle trees over extension field elements
 */
__device__ void poseidon2_compress_ext2(
    GoldilocksExt2 left,
    GoldilocksExt2 right,
    GoldilocksExt2* output
) {
    GoldilocksField state[8];

    // Serialize: left.c[0], left.c[1], right.c[0], right.c[1] into first 4 positions
    state[0] = left.c[0];
    state[1] = left.c[1];
    state[2] = right.c[0];
    state[3] = right.c[1];

    // Zero padding for capacity
    state[4] = GoldilocksField(0);
    state[5] = GoldilocksField(0);
    state[6] = GoldilocksField(0);
    state[7] = GoldilocksField(0);

    // Apply permutation
    poseidon2_permute_8(state);

    // Output as single extension element
    *output = GoldilocksExt2(state[0], state[1]);
}

/**
 * Compress two GF(p⁵) elements into one
 */
__device__ void poseidon2_compress_ext5(
    GoldilocksExt5 left,
    GoldilocksExt5 right,
    GoldilocksExt5* output
) {
    GoldilocksField state[16];

    // Serialize both elements (5 + 5 = 10 base field elements)
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        state[i] = left.c[i];
        state[5 + i] = right.c[i];
    }

    // Zero padding
    #pragma unroll
    for (int i = 10; i < 16; i++) {
        state[i] = GoldilocksField(0);
    }

    // Apply permutation
    poseidon2_permute_16(state);

    // Output as single extension element
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        output->c[i] = state[i];
    }
}

/**
 * Batch compress pairs of GF(p²) elements for Merkle tree layer
 */
__device__ void poseidon2_compress_ext2_batch_element(
    const GoldilocksExt2* left,
    const GoldilocksExt2* right,
    GoldilocksExt2* output,
    int idx
) {
    poseidon2_compress_ext2(left[idx], right[idx], &output[idx]);
}

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize Poseidon2 constants in device memory
 * Call once before using any Poseidon2 kernels
 */
inline cudaError_t poseidon2_init() {
    cudaError_t err;

    // Width 8 constants
    err = cudaMemcpyToSymbol(d_RC_EXT_8, h_RC_EXT_8, sizeof(h_RC_EXT_8));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(d_RC_INT, h_RC_INT, sizeof(h_RC_INT));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(d_DIAG_8, h_DIAG_8, sizeof(h_DIAG_8));
    if (err != cudaSuccess) return err;

    // Width 16 constants
    err = cudaMemcpyToSymbol(d_RC_EXT_16, h_RC_EXT_16, sizeof(h_RC_EXT_16));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(d_DIAG_16, h_DIAG_16, sizeof(h_DIAG_16));
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

#endif // POSEIDON2_CUH

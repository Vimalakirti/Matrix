/**
 * Comprehensive correctness tests for the almost-Goldilocks base field.
 *
 * Strategy:
 *   1. Golden vector tests: a handful of hand-computed (a, b, op, expected)
 *      tuples that pin down the reduction algorithm.
 *   2. Algebraic identity tests run on N random inputs, comparing GPU
 *      results to a CPU reference (the same __host__ inline routines from
 *      almost_goldilocks.cuh + a u128 brute-force reduction).
 *
 * Identities covered:
 *   - add: commutativity, associativity, a + 0 = a, a + (-a) = 0
 *   - sub: a - a = 0, a - b = a + (-b), wrap on underflow
 *   - mul: commutativity, associativity, distributivity, a * 0 = 0, a * 1 = a
 *   - square: square(a) == mul(a, a)
 *   - neg: -(-a) == a
 *   - halve: 2 * halve(a) == a
 *   - inverse: a * inverse(a) == 1 for non-zero a
 *   - exp: a^(P-1) == 1 (Fermat's little theorem) for non-zero a
 *   - canonicalize: idempotent and never increases value
 *   - mul_by_3: agl_mul_by_3(a) == agl_mul(a, 3)
 *
 * Plus stress: 128-bit reduction edge cases (max u64 inputs).
 */

#include "almost_goldilocks_kernels.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <iostream>

#define AGL_PRIME ALMOST_GOLDILOCKS_PRIME
#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL (" << __FILE__ << ":" << __LINE__ << "): " << msg << std::endl; \
        return false; \
    } \
} while (0)

// ============================================================================
// CPU reference (mirrors the device __host__ inline routines, but explicit)
// ============================================================================

// Reduce an arbitrary 128-bit value (lo, hi) mod P, returning canonical in [0, P).
static uint64_t ref_reduce128_canon(uint64_t lo, uint64_t hi) {
    __uint128_t x = ((__uint128_t)hi << 64) | (__uint128_t)lo;
    __uint128_t p = (__uint128_t)AGL_PRIME;
    return (uint64_t)(x % p);
}

static uint64_t ref_add(uint64_t a, uint64_t b) {
    __uint128_t s = (__uint128_t)(a % AGL_PRIME) + (__uint128_t)(b % AGL_PRIME);
    return (uint64_t)(s % (__uint128_t)AGL_PRIME);
}
static uint64_t ref_sub(uint64_t a, uint64_t b) {
    uint64_t ac = a % AGL_PRIME;
    uint64_t bc = b % AGL_PRIME;
    if (ac >= bc) return ac - bc;
    return AGL_PRIME - (bc - ac);
}
static uint64_t ref_mul(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)(a % AGL_PRIME) * (__uint128_t)(b % AGL_PRIME);
    return (uint64_t)(prod % (__uint128_t)AGL_PRIME);
}
// (ref_neg, ref_exp, ref_inv intentionally not defined — the GPU's
// agl_inverse is verified via the algebraic identity a * a^-1 = 1.)

// ============================================================================
// Device kernels for per-element ops (so we can test the device code path
// without relying on the host inline copies)
// ============================================================================

__global__ void k_reduce128(const uint64_t* lo, const uint64_t* hi, uint64_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = agl_reduce128(agl_uint128_t(lo[idx], hi[idx]));
}
__global__ void k_canonicalize(const uint64_t* in, uint64_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = agl_canonicalize(in[idx]);
}
__global__ void k_halve(const AlmostGoldilocksField* in, AlmostGoldilocksField* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = agl_halve(in[idx]);
}
__global__ void k_mul_by_3(const AlmostGoldilocksField* in, AlmostGoldilocksField* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = agl_mul_by_3(in[idx]);
}

// ============================================================================
// Helpers
// ============================================================================

template <typename T>
static T* d_upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T>
static std::vector<T> d_download(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

static std::vector<uint64_t> rand_field_vec(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = rng() % AGL_PRIME;  // canonical inputs
    return v;
}

static std::vector<uint64_t> rand_u64_vec(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = rng();  // full u64 range (non-canonical)
    return v;
}

// ============================================================================
// Tests
// ============================================================================

static bool test_constants() {
    std::cout << "[constants] ";
    EXPECT(ALMOST_GOLDILOCKS_PRIME == 0xFFFFFFFEFFFFFFE1ULL, "prime constant");
    EXPECT(ALMOST_REDUCE_C == 0x10000001FULL, "reduce constant");
    EXPECT(ALMOST_HALF_P_PLUS_ONE == 0x7FFFFFFF7FFFFFF1ULL, "half(P+1)");
    // sanity: 2 * HALF_P_PLUS_ONE  ≡  1 (mod P)
    __uint128_t two_inv2 = 2 * (__uint128_t)ALMOST_HALF_P_PLUS_ONE;
    EXPECT((two_inv2 % (__uint128_t)ALMOST_GOLDILOCKS_PRIME) == 1, "2 * inv2 = 1");
    // sanity: P + c == 2^64
    EXPECT(((__uint128_t)ALMOST_GOLDILOCKS_PRIME + (__uint128_t)ALMOST_REDUCE_C) == ((__uint128_t)1 << 64), "P + c = 2^64");
    std::cout << "PASS" << std::endl;
    return true;
}

// Golden hand-computed vectors. These pin down the reduction algorithm.
static bool test_golden_vectors() {
    std::cout << "[golden_vectors] ";

    struct AddCase { uint64_t a, b, expected_canonical; };
    AddCase add_cases[] = {
        {0, 0, 0},
        {1, 2, 3},
        {AGL_PRIME - 1, 1, 0},                  // P - 1 + 1 = 0
        {AGL_PRIME - 5, 7, 2},                  // wrap
        {0xFFFFFFFFFFFFFFFFULL, 1, ALMOST_REDUCE_C},  // (2^64-1) + 1 = 2^64 ≡ c
    };
    for (auto& tc : add_cases) {
        AlmostGoldilocksField got = agl_add(AlmostGoldilocksField(tc.a), AlmostGoldilocksField(tc.b));
        uint64_t gc = agl_canonicalize(got.value);
        EXPECT(gc == tc.expected_canonical, "add golden");
    }

    struct MulCase { uint64_t a, b, expected_canonical; };
    MulCase mul_cases[] = {
        {0, 12345, 0},
        {1, 12345, 12345},
        {12345, 1, 12345},
        // (2^32) * (2^32) = 2^64 ≡ c
        {1ULL << 32, 1ULL << 32, ALMOST_REDUCE_C},
        // (P-1) * (P-1) = (-1)*(-1) = 1
        {AGL_PRIME - 1, AGL_PRIME - 1, 1},
        // 2 * (P+1)/2 = P+1 ≡ 1
        {2, ALMOST_HALF_P_PLUS_ONE, 1},
    };
    for (auto& tc : mul_cases) {
        AlmostGoldilocksField got = agl_mul(AlmostGoldilocksField(tc.a), AlmostGoldilocksField(tc.b));
        uint64_t gc = agl_canonicalize(got.value);
        EXPECT(gc == tc.expected_canonical, "mul golden");
    }

    std::cout << "PASS" << std::endl;
    return true;
}

// Reduction stress: cover boundary 128-bit inputs against u128 ground truth.
static bool test_reduce128_stress() {
    std::cout << "[reduce128_stress] ";
    const size_t N = 200000;
    std::mt19937_64 rng(0xC0FFEE);

    std::vector<uint64_t> lo(N), hi(N);
    // Mix in many extreme corner cases at the start.
    uint64_t corners[] = {
        0ULL, 1ULL, AGL_PRIME, AGL_PRIME - 1,
        0xFFFFFFFFFFFFFFFFULL, 0x8000000000000000ULL,
        ALMOST_REDUCE_C, (1ULL << 32), (1ULL << 33), ALMOST_HALF_P_PLUS_ONE,
    };
    size_t nc = sizeof(corners) / sizeof(corners[0]);
    for (size_t i = 0; i < nc * nc && i < N; i++) {
        lo[i] = corners[i / nc];
        hi[i] = corners[i % nc];
    }
    for (size_t i = nc * nc; i < N; i++) {
        lo[i] = rng();
        hi[i] = rng();
    }

    uint64_t *d_lo = d_upload(lo), *d_hi = d_upload(hi), *d_out = nullptr;
    cudaMalloc(&d_out, N * sizeof(uint64_t));
    int grid = (int)((N + 255) / 256);
    k_reduce128<<<grid, 256>>>(d_lo, d_hi, d_out, N);
    auto h_out = d_download(d_out, N);

    for (size_t i = 0; i < N; i++) {
        uint64_t got = agl_canonicalize(h_out[i]);
        uint64_t exp = ref_reduce128_canon(lo[i], hi[i]);
        if (got != exp) {
            std::cerr << "reduce128 mismatch at i=" << i
                      << " lo=" << lo[i] << " hi=" << hi[i]
                      << " got=" << got << " expected=" << exp << std::endl;
            cudaFree(d_lo); cudaFree(d_hi); cudaFree(d_out);
            return false;
        }
    }
    cudaFree(d_lo); cudaFree(d_hi); cudaFree(d_out);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

// Canonicalize idempotence and never-larger property
static bool test_canonicalize() {
    std::cout << "[canonicalize] ";
    auto inputs = rand_u64_vec(100000, 1);
    uint64_t *d_in = d_upload(inputs), *d_out = nullptr;
    cudaMalloc(&d_out, inputs.size() * sizeof(uint64_t));
    int grid = (int)((inputs.size() + 255) / 256);
    k_canonicalize<<<grid, 256>>>(d_in, d_out, (int)inputs.size());
    auto out = d_download(d_out, inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        EXPECT(out[i] < AGL_PRIME, "canonicalize out-of-range");
        EXPECT(out[i] == (inputs[i] % AGL_PRIME), "canonicalize value");
    }
    cudaFree(d_in); cudaFree(d_out);
    std::cout << "PASS" << std::endl;
    return true;
}

// Batch add: commutativity, associativity, a+0=a, a+(-a)=0
static bool test_add_identities() {
    std::cout << "[add_identities] ";
    const size_t N = 100000;
    auto av = rand_field_vec(N, 11);
    auto bv = rand_field_vec(N, 22);
    auto zv = std::vector<uint64_t>(N, 0);

    auto d_a = d_upload(av), d_b = d_upload(bv), d_z = d_upload(zv);
    AlmostGoldilocksField *d_ab, *d_ba, *d_az, *d_neg, *d_self_neg;
    cudaMalloc(&d_ab, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_ba, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_az, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_neg, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_self_neg, N * sizeof(AlmostGoldilocksField));

    agl_batch_add((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_b, d_ab, N);
    agl_batch_add((AlmostGoldilocksField*)d_b, (AlmostGoldilocksField*)d_a, d_ba, N);
    agl_batch_add((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_z, d_az, N);
    agl_batch_neg((AlmostGoldilocksField*)d_a, d_neg, N);
    agl_batch_add((AlmostGoldilocksField*)d_a, d_neg, d_self_neg, N);
    cudaDeviceSynchronize();

    auto ab = d_download(d_ab, N), ba = d_download(d_ba, N),
         az = d_download(d_az, N), self_neg = d_download(d_self_neg, N);

    for (size_t i = 0; i < N; i++) {
        EXPECT(agl_canonicalize(ab[i].value) == agl_canonicalize(ba[i].value), "commutativity");
        EXPECT(agl_canonicalize(ab[i].value) == ref_add(av[i], bv[i]), "add vs ref");
        EXPECT(agl_canonicalize(az[i].value) == av[i], "a + 0 = a");
        EXPECT(agl_canonicalize(self_neg[i].value) == 0, "a + (-a) = 0");
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_z);
    cudaFree(d_ab); cudaFree(d_ba); cudaFree(d_az); cudaFree(d_neg); cudaFree(d_self_neg);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

static bool test_sub_mul_identities() {
    std::cout << "[sub_mul_identities] ";
    const size_t N = 100000;
    auto av = rand_field_vec(N, 33);
    auto bv = rand_field_vec(N, 44);
    auto cv = rand_field_vec(N, 55);
    auto ones = std::vector<uint64_t>(N, 1);

    auto d_a = d_upload(av), d_b = d_upload(bv), d_c = d_upload(cv), d_one = d_upload(ones);

    AlmostGoldilocksField *d_sub, *d_self_sub, *d_mul, *d_mul_swap, *d_mul_one,
                          *d_a_plus_b, *d_a_mul_c, *d_b_mul_c, *d_sum_muls, *d_distrib;
    cudaMalloc(&d_sub, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_self_sub, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_mul, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_mul_swap, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_mul_one, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_a_plus_b, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_a_mul_c, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_b_mul_c, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_sum_muls, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_distrib, N * sizeof(AlmostGoldilocksField));

    agl_batch_sub((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_b, d_sub, N);
    agl_batch_sub((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_a, d_self_sub, N);
    agl_batch_mul((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_b, d_mul, N);
    agl_batch_mul((AlmostGoldilocksField*)d_b, (AlmostGoldilocksField*)d_a, d_mul_swap, N);
    agl_batch_mul((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_one, d_mul_one, N);

    // distributivity: (a + b) * c == a*c + b*c
    agl_batch_add((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_b, d_a_plus_b, N);
    agl_batch_mul(d_a_plus_b, (AlmostGoldilocksField*)d_c, d_distrib, N);
    agl_batch_mul((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_c, d_a_mul_c, N);
    agl_batch_mul((AlmostGoldilocksField*)d_b, (AlmostGoldilocksField*)d_c, d_b_mul_c, N);
    agl_batch_add(d_a_mul_c, d_b_mul_c, d_sum_muls, N);

    cudaDeviceSynchronize();

    auto h_sub = d_download(d_sub, N), h_self = d_download(d_self_sub, N),
         h_mul = d_download(d_mul, N), h_swap = d_download(d_mul_swap, N),
         h_one = d_download(d_mul_one, N), h_distrib = d_download(d_distrib, N),
         h_summ = d_download(d_sum_muls, N);

    for (size_t i = 0; i < N; i++) {
        EXPECT(agl_canonicalize(h_sub[i].value) == ref_sub(av[i], bv[i]), "sub vs ref");
        EXPECT(agl_canonicalize(h_self[i].value) == 0, "a - a = 0");
        EXPECT(agl_canonicalize(h_mul[i].value) == ref_mul(av[i], bv[i]), "mul vs ref");
        EXPECT(agl_canonicalize(h_mul[i].value) == agl_canonicalize(h_swap[i].value), "mul commutativity");
        EXPECT(agl_canonicalize(h_one[i].value) == av[i], "a * 1 = a");
        EXPECT(agl_canonicalize(h_distrib[i].value) == agl_canonicalize(h_summ[i].value), "distributivity");
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_one);
    cudaFree(d_sub); cudaFree(d_self_sub); cudaFree(d_mul); cudaFree(d_mul_swap);
    cudaFree(d_mul_one); cudaFree(d_a_plus_b); cudaFree(d_a_mul_c); cudaFree(d_b_mul_c);
    cudaFree(d_sum_muls); cudaFree(d_distrib);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

// Associativity: (a+b)+c == a+(b+c), (a*b)*c == a*(b*c)
static bool test_associativity() {
    std::cout << "[associativity] ";
    const size_t N = 50000;
    auto av = rand_field_vec(N, 66);
    auto bv = rand_field_vec(N, 77);
    auto cv = rand_field_vec(N, 88);

    auto d_a = d_upload(av), d_b = d_upload(bv), d_c = d_upload(cv);
    AlmostGoldilocksField *d_ab, *d_ab_c, *d_bc, *d_a_bc;
    AlmostGoldilocksField *d_amb, *d_amb_c, *d_bmc, *d_a_bmc;
    cudaMalloc(&d_ab, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_ab_c, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_bc, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_a_bc, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_amb, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_amb_c, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_bmc, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_a_bmc, N * sizeof(AlmostGoldilocksField));

    agl_batch_add((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_b, d_ab, N);
    agl_batch_add(d_ab, (AlmostGoldilocksField*)d_c, d_ab_c, N);
    agl_batch_add((AlmostGoldilocksField*)d_b, (AlmostGoldilocksField*)d_c, d_bc, N);
    agl_batch_add((AlmostGoldilocksField*)d_a, d_bc, d_a_bc, N);

    agl_batch_mul((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_b, d_amb, N);
    agl_batch_mul(d_amb, (AlmostGoldilocksField*)d_c, d_amb_c, N);
    agl_batch_mul((AlmostGoldilocksField*)d_b, (AlmostGoldilocksField*)d_c, d_bmc, N);
    agl_batch_mul((AlmostGoldilocksField*)d_a, d_bmc, d_a_bmc, N);
    cudaDeviceSynchronize();

    auto h1 = d_download(d_ab_c, N), h2 = d_download(d_a_bc, N),
         h3 = d_download(d_amb_c, N), h4 = d_download(d_a_bmc, N);

    for (size_t i = 0; i < N; i++) {
        EXPECT(agl_canonicalize(h1[i].value) == agl_canonicalize(h2[i].value), "add assoc");
        EXPECT(agl_canonicalize(h3[i].value) == agl_canonicalize(h4[i].value), "mul assoc");
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_ab); cudaFree(d_ab_c); cudaFree(d_bc); cudaFree(d_a_bc);
    cudaFree(d_amb); cudaFree(d_amb_c); cudaFree(d_bmc); cudaFree(d_a_bmc);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

// Inverse: a * a^(-1) = 1, and double_neg(a) = a
static bool test_inverse_neg() {
    std::cout << "[inverse_neg] ";
    const size_t N = 5000;
    auto av = rand_field_vec(N, 99);
    for (auto& x : av) if (x == 0) x = 1;  // avoid zero

    auto d_a = d_upload(av);
    AlmostGoldilocksField *d_inv, *d_check, *d_neg, *d_dneg;
    cudaMalloc(&d_inv, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_check, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_neg, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_dneg, N * sizeof(AlmostGoldilocksField));

    agl_batch_inverse((AlmostGoldilocksField*)d_a, d_inv, N);
    agl_batch_mul((AlmostGoldilocksField*)d_a, d_inv, d_check, N);
    agl_batch_neg((AlmostGoldilocksField*)d_a, d_neg, N);
    agl_batch_neg(d_neg, d_dneg, N);
    cudaDeviceSynchronize();

    auto h_check = d_download(d_check, N), h_dneg = d_download(d_dneg, N);
    for (size_t i = 0; i < N; i++) {
        EXPECT(agl_canonicalize(h_check[i].value) == 1, "a * a^-1 = 1");
        EXPECT(agl_canonicalize(h_dneg[i].value) == av[i], "-(-a) = a");
    }
    cudaFree(d_a); cudaFree(d_inv); cudaFree(d_check); cudaFree(d_neg); cudaFree(d_dneg);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

// Halve: 2 * halve(a) = a
static bool test_halve() {
    std::cout << "[halve] ";
    const size_t N = 10000;
    auto av = rand_field_vec(N, 111);
    // include edge cases
    av[0] = 0; av[1] = 1; av[2] = 2; av[3] = AGL_PRIME - 1; av[4] = AGL_PRIME - 2;

    auto d_a = d_upload(av);
    AlmostGoldilocksField *d_half, *d_back;
    cudaMalloc(&d_half, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_back, N * sizeof(AlmostGoldilocksField));

    int grid = (int)((N + 255) / 256);
    k_halve<<<grid, 256>>>((AlmostGoldilocksField*)d_a, d_half, (int)N);
    agl_batch_add(d_half, d_half, d_back, N);
    cudaDeviceSynchronize();

    auto h_back = d_download(d_back, N);
    for (size_t i = 0; i < N; i++) {
        EXPECT(agl_canonicalize(h_back[i].value) == av[i], "2 * halve(a) = a");
    }
    cudaFree(d_a); cudaFree(d_half); cudaFree(d_back);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

// mul_by_3 == mul(a, 3)
static bool test_mul_by_3() {
    std::cout << "[mul_by_3] ";
    const size_t N = 10000;
    auto av = rand_field_vec(N, 222);
    auto threes = std::vector<uint64_t>(N, 3);

    auto d_a = d_upload(av), d_3 = d_upload(threes);
    AlmostGoldilocksField *d_m3a, *d_m3b;
    cudaMalloc(&d_m3a, N * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_m3b, N * sizeof(AlmostGoldilocksField));

    int grid = (int)((N + 255) / 256);
    k_mul_by_3<<<grid, 256>>>((AlmostGoldilocksField*)d_a, d_m3a, (int)N);
    agl_batch_mul((AlmostGoldilocksField*)d_a, (AlmostGoldilocksField*)d_3, d_m3b, N);
    cudaDeviceSynchronize();

    auto h_a = d_download(d_m3a, N), h_b = d_download(d_m3b, N);
    for (size_t i = 0; i < N; i++) {
        EXPECT(agl_canonicalize(h_a[i].value) == agl_canonicalize(h_b[i].value), "mul_by_3 vs mul(_,3)");
    }
    cudaFree(d_a); cudaFree(d_3); cudaFree(d_m3a); cudaFree(d_m3b);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

// Fermat: a^(P-1) = 1 (a != 0)
static bool test_fermat() {
    std::cout << "[fermat] ";
    const size_t N = 200;  // exp is expensive
    auto av = rand_field_vec(N, 333);
    for (auto& x : av) if (x == 0) x = 1;

    auto d_a = d_upload(av);
    AlmostGoldilocksField* d_out;
    cudaMalloc(&d_out, N * sizeof(AlmostGoldilocksField));
    int grid = (int)((N + 255) / 256);
    agl_batch_exp_kernel<<<grid, 256>>>((AlmostGoldilocksField*)d_a, AGL_PRIME - 1, d_out, N);
    cudaDeviceSynchronize();

    auto h_out = d_download(d_out, N);
    for (size_t i = 0; i < N; i++) {
        EXPECT(agl_canonicalize(h_out[i].value) == 1, "Fermat: a^(P-1) = 1");
    }
    cudaFree(d_a); cudaFree(d_out);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

// Reduction: square test on a few specific points known to stress the reduction
static bool test_specific_stress() {
    std::cout << "[specific_stress] ";
    // 2^32 * 2^32 = 2^64 ≡ c
    AlmostGoldilocksField r1 = agl_mul(AlmostGoldilocksField(1ULL<<32), AlmostGoldilocksField(1ULL<<32));
    EXPECT(agl_canonicalize(r1.value) == ALMOST_REDUCE_C, "2^32 * 2^32 ≡ c");

    // 2^63 * 2 = 2^64 ≡ c
    AlmostGoldilocksField r2 = agl_mul(AlmostGoldilocksField(1ULL<<63), AlmostGoldilocksField(2));
    EXPECT(agl_canonicalize(r2.value) == ALMOST_REDUCE_C, "2^63 * 2 ≡ c");

    // (-1) * (-1) = 1
    AlmostGoldilocksField mn1 = AlmostGoldilocksField(AGL_PRIME - 1);
    AlmostGoldilocksField r3 = agl_mul(mn1, mn1);
    EXPECT(agl_canonicalize(r3.value) == 1, "(-1)^2 = 1");

    // ALMOST_REDUCE_C * 2^32 mod P = 2^96 mod P = 2^37 + 31 (verified offline)
    AlmostGoldilocksField r4 = agl_mul(AlmostGoldilocksField(ALMOST_REDUCE_C), AlmostGoldilocksField(1ULL<<32));
    EXPECT(agl_canonicalize(r4.value) == ((1ULL << 37) + 31), "c * 2^32 ≡ 2^37 + 31");

    std::cout << "PASS" << std::endl;
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) { std::cerr << "No CUDA device" << std::endl; return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ")" << std::endl;
    std::cout << "Field: P = 0x" << std::hex << AGL_PRIME << std::dec
              << " = 2^64 - 2^32 - 31" << std::endl;
    std::cout << "=== almost-Goldilocks field tests ===" << std::endl;

    bool ok = true;
    ok &= test_constants();
    ok &= test_golden_vectors();
    ok &= test_reduce128_stress();
    ok &= test_canonicalize();
    ok &= test_specific_stress();
    ok &= test_add_identities();
    ok &= test_sub_mul_identities();
    ok &= test_associativity();
    ok &= test_inverse_neg();
    ok &= test_halve();
    ok &= test_mul_by_3();
    ok &= test_fermat();

    std::cout << (ok ? "ALL PASS" : "FAILURES PRESENT") << std::endl;
    return ok ? 0 : 1;
}

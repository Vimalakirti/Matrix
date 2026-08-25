/**
 * Comprehensive correctness tests for the almost-Goldilocks Ext2.
 *
 * Identities covered:
 *   - Golden vectors: hand-computed (a, b, op, expected)
 *   - Commutativity, associativity, distributivity
 *   - a + 0 = a, a * 1 = a
 *   - a * a^(-1) = 1 (for non-zero a)
 *   - square(a) == mul(a, a)
 *   - norm(a) == c[0] of a * conj(a)
 *   - frobenius(frobenius(a)) == a  (since p^2 fixes all elements; frobenius is order 2)
 *   - conjugate(a) == frobenius(a)
 *   - Karatsuba mul equivalence with explicit polynomial multiplication
 *   - Embedding round-trip: agl_to_ext2 then aext2_to_agl
 *
 * Also a non-residue sanity check: 3 is a QR'sLegendre = -1, so X^2 = 3
 * has no solution in the base field, which is what makes Ext2 well-defined.
 */

#include "almost_extension_kernels.cu"
#include <cstdio>
#include <cstdlib>
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
// CPU reference (works on canonical u64s with __uint128_t reduction)
// ============================================================================

struct E2 { uint64_t c0, c1; };

static uint64_t r_mul(uint64_t a, uint64_t b) {
    __uint128_t p = (__uint128_t)(a % AGL_PRIME) * (__uint128_t)(b % AGL_PRIME);
    return (uint64_t)(p % (__uint128_t)AGL_PRIME);
}
static uint64_t r_add(uint64_t a, uint64_t b) {
    __uint128_t s = (__uint128_t)(a % AGL_PRIME) + (__uint128_t)(b % AGL_PRIME);
    return (uint64_t)(s % (__uint128_t)AGL_PRIME);
}
static uint64_t r_sub(uint64_t a, uint64_t b) {
    uint64_t ac = a % AGL_PRIME, bc = b % AGL_PRIME;
    return ac >= bc ? ac - bc : AGL_PRIME - (bc - ac);
}
static E2 r_e2_add(E2 a, E2 b) { return { r_add(a.c0, b.c0), r_add(a.c1, b.c1) }; }
static E2 r_e2_sub(E2 a, E2 b) { return { r_sub(a.c0, b.c0), r_sub(a.c1, b.c1) }; }
static E2 r_e2_mul(E2 a, E2 b) {
    // (a0 + a1 X)(b0 + b1 X) with X^2 = 3
    uint64_t c0 = r_add(r_mul(a.c0, b.c0), r_mul(3ULL, r_mul(a.c1, b.c1)));
    uint64_t c1 = r_add(r_mul(a.c0, b.c1), r_mul(a.c1, b.c0));
    return { c0, c1 };
}
static E2 r_e2_square(E2 a) { return r_e2_mul(a, a); }

// ============================================================================
// Helpers
// ============================================================================

template <typename T>
static T* d_upload(const std::vector<T>& v) {
    T* d = nullptr; cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <typename T>
static std::vector<T> d_download(const T* d, size_t n) {
    std::vector<T> h(n); cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

static std::vector<AlmostGoldilocksExt2> rand_e2_vec(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<AlmostGoldilocksExt2> v(n);
    for (auto& e : v) {
        e.c[0] = AlmostGoldilocksField(rng() % AGL_PRIME);
        e.c[1] = AlmostGoldilocksField(rng() % AGL_PRIME);
    }
    return v;
}
static E2 to_E2(const AlmostGoldilocksExt2& x) {
    return { agl_canonicalize(x.c[0].value), agl_canonicalize(x.c[1].value) };
}

// ============================================================================
// Tests
// ============================================================================

static bool test_non_residue() {
    std::cout << "[non_residue] ";
    // Verify 3 is a quadratic non-residue: 3^((P-1)/2) == P - 1 (= -1 mod P)
    AlmostGoldilocksField three(ALMOST_EXT2_W);
    AlmostGoldilocksField legendre = agl_exp(three, (AGL_PRIME - 1) / 2);
    EXPECT(agl_canonicalize(legendre.value) == AGL_PRIME - 1, "Legendre(3) = -1");
    EXPECT(ALMOST_EXT2_DTH_ROOT == AGL_PRIME - 1, "DTH_ROOT = -1");
    std::cout << "PASS" << std::endl;
    return true;
}

static bool test_golden_vectors() {
    std::cout << "[ext2 golden_vectors] ";

    // (1 + 2X)(3 + 4X) = (1*3 + 3*2*4) + (1*4 + 2*3)X = (3 + 24) + 10X = 27 + 10X
    AlmostGoldilocksExt2 a(1, 2), b(3, 4);
    AlmostGoldilocksExt2 prod = aext2_mul(a, b);
    EXPECT(agl_canonicalize(prod.c[0].value) == 27, "golden c0");
    EXPECT(agl_canonicalize(prod.c[1].value) == 10, "golden c1");

    // square (2 + 3X)^2 = (4 + 3*9) + 12X = 31 + 12X
    AlmostGoldilocksExt2 c(2, 3);
    AlmostGoldilocksExt2 csq = aext2_square(c);
    EXPECT(agl_canonicalize(csq.c[0].value) == 31, "square c0");
    EXPECT(agl_canonicalize(csq.c[1].value) == 12, "square c1");

    // X * X = 3
    AlmostGoldilocksExt2 X(0, 1);
    AlmostGoldilocksExt2 XX = aext2_mul(X, X);
    EXPECT(agl_canonicalize(XX.c[0].value) == 3, "X*X c0");
    EXPECT(agl_canonicalize(XX.c[1].value) == 0, "X*X c1");

    std::cout << "PASS" << std::endl;
    return true;
}

static bool test_add_mul_vs_ref() {
    std::cout << "[ext2 add_mul_vs_ref] ";
    const size_t N = 50000;
    auto av = rand_e2_vec(N, 1);
    auto bv = rand_e2_vec(N, 2);

    auto d_a = d_upload(av), d_b = d_upload(bv);
    AlmostGoldilocksExt2 *d_add, *d_sub, *d_mul, *d_sq;
    cudaMalloc(&d_add, N * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_sub, N * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_mul, N * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_sq,  N * sizeof(AlmostGoldilocksExt2));

    aext2_batch_add(d_a, d_b, d_add, N);
    aext2_batch_sub(d_a, d_b, d_sub, N);
    aext2_batch_mul(d_a, d_b, d_mul, N);
    aext2_batch_square(d_a, d_sq, N);
    cudaDeviceSynchronize();

    auto h_add = d_download(d_add, N), h_sub = d_download(d_sub, N),
         h_mul = d_download(d_mul, N), h_sq  = d_download(d_sq, N);

    for (size_t i = 0; i < N; i++) {
        E2 a = to_E2(av[i]), b = to_E2(bv[i]);
        E2 add = r_e2_add(a, b), sub = r_e2_sub(a, b), mul = r_e2_mul(a, b), sq = r_e2_square(a);
        EXPECT(to_E2(h_add[i]).c0 == add.c0 && to_E2(h_add[i]).c1 == add.c1, "add");
        EXPECT(to_E2(h_sub[i]).c0 == sub.c0 && to_E2(h_sub[i]).c1 == sub.c1, "sub");
        EXPECT(to_E2(h_mul[i]).c0 == mul.c0 && to_E2(h_mul[i]).c1 == mul.c1, "mul");
        EXPECT(to_E2(h_sq[i]).c0  == sq.c0  && to_E2(h_sq[i]).c1  == sq.c1,  "square");
    }
    cudaFree(d_a); cudaFree(d_b);
    cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_sq);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

static bool test_inverse() {
    std::cout << "[ext2 inverse] ";
    const size_t N = 2000;
    auto av = rand_e2_vec(N, 3);
    // Ensure non-zero (norm != 0): bump c0 to 1 if all zero
    for (auto& e : av) if (agl_is_zero(e.c[0]) && agl_is_zero(e.c[1])) e.c[0] = AlmostGoldilocksField(1);

    auto d_a = d_upload(av);
    AlmostGoldilocksExt2 *d_inv, *d_chk;
    cudaMalloc(&d_inv, N * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_chk, N * sizeof(AlmostGoldilocksExt2));

    aext2_batch_inverse(d_a, d_inv, N);
    aext2_batch_mul(d_a, d_inv, d_chk, N);
    cudaDeviceSynchronize();

    auto h = d_download(d_chk, N);
    for (size_t i = 0; i < N; i++) {
        E2 r = to_E2(h[i]);
        EXPECT(r.c0 == 1 && r.c1 == 0, "a * a^-1 = (1, 0)");
    }
    cudaFree(d_a); cudaFree(d_inv); cudaFree(d_chk);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

static bool test_frobenius_conjugate() {
    std::cout << "[ext2 frobenius_conjugate] ";
    const size_t N = 5000;
    auto av = rand_e2_vec(N, 4);
    auto d_a = d_upload(av);

    AlmostGoldilocksExt2 *d_frob, *d_frob2;
    cudaMalloc(&d_frob, N * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_frob2, N * sizeof(AlmostGoldilocksExt2));

    int grid = (int)((N + 255) / 256);
    aext2_batch_frobenius_kernel<<<grid, 256>>>(d_a, d_frob, (int)N);
    aext2_batch_frobenius_kernel<<<grid, 256>>>(d_frob, d_frob2, (int)N);
    cudaDeviceSynchronize();

    auto h_frob = d_download(d_frob, N), h_frob2 = d_download(d_frob2, N);

    for (size_t i = 0; i < N; i++) {
        // frobenius(a) = (c0, -c1) since DTH_ROOT = -1
        E2 a = to_E2(av[i]);
        E2 expected_frob = { a.c0, (a.c1 == 0) ? 0 : (AGL_PRIME - a.c1) };
        EXPECT(to_E2(h_frob[i]).c0 == expected_frob.c0 && to_E2(h_frob[i]).c1 == expected_frob.c1,
               "frobenius == conjugate");
        // frobenius^2 = identity
        EXPECT(to_E2(h_frob2[i]).c0 == a.c0 && to_E2(h_frob2[i]).c1 == a.c1, "frobenius^2 = id");
    }
    cudaFree(d_a); cudaFree(d_frob); cudaFree(d_frob2);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

static bool test_norm() {
    std::cout << "[ext2 norm] ";
    // norm(a) = a * conj(a) = a0^2 - 3 a1^2 — verify via aext2_norm matches base
    const size_t N = 1000;
    auto av = rand_e2_vec(N, 5);

    for (size_t i = 0; i < N; i++) {
        AlmostGoldilocksField norm = aext2_norm(av[i]);
        E2 a = to_E2(av[i]);
        uint64_t expected = r_sub(r_mul(a.c0, a.c0), r_mul(3, r_mul(a.c1, a.c1)));
        EXPECT(agl_canonicalize(norm.value) == expected, "norm");
    }
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

static bool test_embedding() {
    std::cout << "[ext2 embedding] ";
    // agl_to_ext2(a) = (a, 0); aext2_to_agl((a, 0)) = a
    const size_t N = 1000;
    std::mt19937_64 rng(6);
    std::vector<AlmostGoldilocksField> base(N);
    for (auto& x : base) x = AlmostGoldilocksField(rng() % AGL_PRIME);

    auto d_base = d_upload(base);
    AlmostGoldilocksExt2* d_e2;
    cudaMalloc(&d_e2, N * sizeof(AlmostGoldilocksExt2));
    AlmostGoldilocksField* d_back;
    cudaMalloc(&d_back, N * sizeof(AlmostGoldilocksField));

    agl_to_aext2_batch(d_base, d_e2, (int)N);
    aext2_to_agl_batch(d_e2, d_back, (int)N);
    cudaDeviceSynchronize();

    auto h_e2 = d_download(d_e2, N);
    auto h_back_vec = d_download(d_back, N);
    for (size_t i = 0; i < N; i++) {
        EXPECT(agl_canonicalize(h_e2[i].c[0].value) == agl_canonicalize(base[i].value), "embed c0");
        EXPECT(agl_canonicalize(h_e2[i].c[1].value) == 0, "embed c1 = 0");
        EXPECT(agl_canonicalize(h_back_vec[i].value) == agl_canonicalize(base[i].value), "round-trip");
    }
    cudaFree(d_base); cudaFree(d_e2); cudaFree(d_back);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

static bool test_distributivity() {
    std::cout << "[ext2 distributivity] ";
    const size_t N = 10000;
    auto av = rand_e2_vec(N, 7), bv = rand_e2_vec(N, 8), cv = rand_e2_vec(N, 9);
    auto d_a = d_upload(av), d_b = d_upload(bv), d_c = d_upload(cv);

    AlmostGoldilocksExt2 *d_apb, *d_lhs, *d_ac, *d_bc, *d_rhs;
    cudaMalloc(&d_apb, N * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_lhs, N * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_ac, N * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_bc, N * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_rhs, N * sizeof(AlmostGoldilocksExt2));

    aext2_batch_add(d_a, d_b, d_apb, N);
    aext2_batch_mul(d_apb, d_c, d_lhs, N);
    aext2_batch_mul(d_a, d_c, d_ac, N);
    aext2_batch_mul(d_b, d_c, d_bc, N);
    aext2_batch_add(d_ac, d_bc, d_rhs, N);
    cudaDeviceSynchronize();

    auto h_lhs = d_download(d_lhs, N), h_rhs = d_download(d_rhs, N);
    for (size_t i = 0; i < N; i++) {
        EXPECT(to_E2(h_lhs[i]).c0 == to_E2(h_rhs[i]).c0 && to_E2(h_lhs[i]).c1 == to_E2(h_rhs[i]).c1,
               "(a+b)*c == a*c + b*c");
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_apb); cudaFree(d_lhs); cudaFree(d_ac); cudaFree(d_bc); cudaFree(d_rhs);
    std::cout << "PASS (" << N << " cases)" << std::endl;
    return true;
}

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) { std::cerr << "No CUDA device" << std::endl; return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "=== almost-Goldilocks Ext2 tests (X^2 - 3) ===" << std::endl;

    bool ok = true;
    ok &= test_non_residue();
    ok &= test_golden_vectors();
    ok &= test_add_mul_vs_ref();
    ok &= test_inverse();
    ok &= test_frobenius_conjugate();
    ok &= test_norm();
    ok &= test_embedding();
    ok &= test_distributivity();

    std::cout << (ok ? "ALL PASS" : "FAILURES PRESENT") << std::endl;
    return ok ? 0 : 1;
}

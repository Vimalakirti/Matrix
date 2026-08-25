/**
 * Eq Lagrange CUDA tests for the almost-Goldilocks field.
 *
 * Verifies the DP and WHT implementations agree with each other and with
 * a CPU reference that computes eq(r, x) via the standard product formula:
 *
 *     eq(r, x) = prod_i ( r_i * x_i + (1 - r_i) * (1 - x_i) )
 *
 * Also exercises the Ext2 DP path against an Ext2 CPU reference.
 */

#include "almost_eq_lagrange.cuh"
#include <iostream>
#include <vector>
#include <random>

#define AGL_PRIME ALMOST_GOLDILOCKS_PRIME
#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL (" << __FILE__ << ":" << __LINE__ << "): " << msg << std::endl; \
        return false; \
    } \
} while (0)

// ============================================================
// Host-side helpers (mirroring the device __host__ inline ops)
// ============================================================

static inline AlmostGoldilocksField host_mul(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    agl_uint128_t prod = agl_mul_u64_u64(a.value, b.value);
    return AlmostGoldilocksField(agl_reduce128(prod));
}
static inline AlmostGoldilocksField host_sub(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return AlmostGoldilocksField(agl_sub_no_canonicalize(a.value, b.value));
}
static inline AlmostGoldilocksExt2 host_e2_mul(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    AlmostGoldilocksField m0 = host_mul(a.c[0], b.c[0]);
    AlmostGoldilocksField m1 = host_mul(a.c[1], b.c[1]);
    AlmostGoldilocksField m2 = host_mul(
        AlmostGoldilocksField(agl_add_no_canonicalize(a.c[0].value, a.c[1].value)),
        AlmostGoldilocksField(agl_add_no_canonicalize(b.c[0].value, b.c[1].value))
    );
    AlmostGoldilocksField m1_3 = AlmostGoldilocksField(agl_add_no_canonicalize(
        agl_add_no_canonicalize(m1.value, m1.value), m1.value));
    AlmostGoldilocksField c0 = AlmostGoldilocksField(agl_add_no_canonicalize(m0.value, m1_3.value));
    AlmostGoldilocksField c1 = AlmostGoldilocksField(agl_sub_no_canonicalize(
        agl_sub_no_canonicalize(m2.value, m0.value), m1.value));
    return AlmostGoldilocksExt2(c0, c1);
}
static inline AlmostGoldilocksExt2 host_e2_sub(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return AlmostGoldilocksExt2(host_sub(a.c[0], b.c[0]), host_sub(a.c[1], b.c[1]));
}

// ============================================================
// CPU reference: eq(r, x) over the full hypercube
// ============================================================

static void cpu_eq_base(const std::vector<AlmostGoldilocksField>& r,
                       std::vector<AlmostGoldilocksField>& out) {
    int log_n = (int)r.size();
    size_t n = 1ULL << log_n;
    out.resize(n);
    for (size_t x = 0; x < n; x++) {
        AlmostGoldilocksField acc(1);
        for (int i = 0; i < log_n; i++) {
            int bit = (x >> i) & 1;
            if (bit) acc = host_mul(acc, r[i]);
            else     acc = host_mul(acc, host_sub(AlmostGoldilocksField(1), r[i]));
        }
        out[x] = acc;
    }
}

static void cpu_eq_ext2(const std::vector<AlmostGoldilocksExt2>& r,
                       std::vector<AlmostGoldilocksExt2>& out) {
    int log_n = (int)r.size();
    size_t n = 1ULL << log_n;
    out.resize(n);
    AlmostGoldilocksExt2 one(AlmostGoldilocksField(1), AlmostGoldilocksField(0));
    for (size_t x = 0; x < n; x++) {
        AlmostGoldilocksExt2 acc = one;
        for (int i = 0; i < log_n; i++) {
            int bit = (x >> i) & 1;
            if (bit) acc = host_e2_mul(acc, r[i]);
            else     acc = host_e2_mul(acc, host_e2_sub(one, r[i]));
        }
        out[x] = acc;
    }
}

// ============================================================
// Tests
// ============================================================

static bool test_eq_base(int log_n) {
    std::cout << "  log_n = " << log_n << " (N = " << (1ULL << log_n) << "): ";
    size_t n = 1ULL << log_n;

    std::vector<AlmostGoldilocksField> h_r(log_n);
    std::mt19937_64 rng(42 + log_n);
    for (auto& x : h_r) x = AlmostGoldilocksField(rng() % AGL_PRIME);

    std::vector<AlmostGoldilocksField> cpu_out;
    cpu_eq_base(h_r, cpu_out);

    AlmostGoldilocksField *d_r, *d_a, *d_b, *d_wht;
    cudaMalloc(&d_r, log_n * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_a, n * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_b, n * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_wht, n * sizeof(AlmostGoldilocksField));
    cudaMemcpy(d_r, h_r.data(), log_n * sizeof(AlmostGoldilocksField), cudaMemcpyHostToDevice);

    AlmostGoldilocksField* d_dp = nullptr;
    agl_eq_dp_all(d_r, d_a, d_b, log_n, &d_dp);
    cudaDeviceSynchronize();

    std::vector<AlmostGoldilocksField> dp_out(n), wht_out(n);
    cudaMemcpy(dp_out.data(), d_dp, n * sizeof(AlmostGoldilocksField), cudaMemcpyDeviceToHost);

    agl_eq_wht_all(d_r, d_wht, log_n);
    cudaDeviceSynchronize();
    cudaMemcpy(wht_out.data(), d_wht, n * sizeof(AlmostGoldilocksField), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        uint64_t exp = agl_canonicalize(cpu_out[i].value);
        uint64_t got_dp = agl_canonicalize(dp_out[i].value);
        uint64_t got_wht = agl_canonicalize(wht_out[i].value);
        if (got_dp != exp || got_wht != exp) {
            std::cerr << "  mismatch at " << i << " expected " << exp
                      << " dp " << got_dp << " wht " << got_wht << std::endl;
            ok = false;
            break;
        }
    }
    cudaFree(d_r); cudaFree(d_a); cudaFree(d_b); cudaFree(d_wht);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

static bool test_eq_ext2(int log_n) {
    std::cout << "  ext2 log_n = " << log_n << " (N = " << (1ULL << log_n) << "): ";
    size_t n = 1ULL << log_n;

    std::vector<AlmostGoldilocksExt2> h_r(log_n);
    std::mt19937_64 rng(123 + log_n);
    for (auto& e : h_r) {
        e.c[0] = AlmostGoldilocksField(rng() % AGL_PRIME);
        e.c[1] = AlmostGoldilocksField(rng() % AGL_PRIME);
    }

    std::vector<AlmostGoldilocksExt2> cpu_out;
    cpu_eq_ext2(h_r, cpu_out);

    AlmostGoldilocksExt2 *d_r, *d_a, *d_b;
    cudaMalloc(&d_r, log_n * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_a, n * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_b, n * sizeof(AlmostGoldilocksExt2));
    cudaMemcpy(d_r, h_r.data(), log_n * sizeof(AlmostGoldilocksExt2), cudaMemcpyHostToDevice);

    AlmostGoldilocksExt2* d_dp = nullptr;
    aext2_eq_dp_all(d_r, d_a, d_b, log_n, &d_dp);
    cudaDeviceSynchronize();

    std::vector<AlmostGoldilocksExt2> dp_out(n);
    cudaMemcpy(dp_out.data(), d_dp, n * sizeof(AlmostGoldilocksExt2), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        uint64_t e0 = agl_canonicalize(cpu_out[i].c[0].value);
        uint64_t e1 = agl_canonicalize(cpu_out[i].c[1].value);
        uint64_t g0 = agl_canonicalize(dp_out[i].c[0].value);
        uint64_t g1 = agl_canonicalize(dp_out[i].c[1].value);
        if (g0 != e0 || g1 != e1) {
            std::cerr << "  ext2 mismatch at " << i << " expected (" << e0 << "," << e1
                      << ") got (" << g0 << "," << g1 << ")" << std::endl;
            ok = false;
            break;
        }
    }
    cudaFree(d_r); cudaFree(d_a); cudaFree(d_b);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) { std::cerr << "No CUDA device" << std::endl; return 1; }
    std::cout << "=== almost-Goldilocks eq_lagrange tests ===" << std::endl;
    bool ok = true;
    std::cout << "Base field:" << std::endl;
    for (int log_n : {4, 8, 12, 16}) ok &= test_eq_base(log_n);
    std::cout << "Ext2:" << std::endl;
    for (int log_n : {4, 8, 12, 16}) ok &= test_eq_ext2(log_n);
    std::cout << (ok ? "ALL PASS" : "FAILURES PRESENT") << std::endl;
    return ok ? 0 : 1;
}

/**
 * Sumcheck-prover correctness tests for the almost-Goldilocks field.
 *
 * Verifies:
 *   - sumcheck_round_message_kernel: for each c ∈ {0,...,d}, the GPU output
 *     matches the CPU's  g(c) = Σ_y Π_i p_i(c, y).
 *   - sumcheck_fold_kernel: the GPU fold matches the CPU
 *     p'_i(y) = p_i(2y) + r * (p_i(2y+1) - p_i(2y)).
 *
 * Tested with both base field and Ext2 paths.
 */

#include "almost_sumcheck_prover.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <cstring>

#define AGL_PRIME ALMOST_GOLDILOCKS_PRIME
#define EXPECT(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL (" << __FILE__ << ":" << __LINE__ << "): " << msg << std::endl; \
    return false; } } while (0)

// ------ host arithmetic ------
static inline AlmostGoldilocksField hmul(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    agl_uint128_t p = agl_mul_u64_u64(a.value, b.value);
    return AlmostGoldilocksField(agl_reduce128(p));
}
static inline AlmostGoldilocksField hadd(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return AlmostGoldilocksField(agl_add_no_canonicalize(a.value, b.value));
}
static inline AlmostGoldilocksField hsub(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return AlmostGoldilocksField(agl_sub_no_canonicalize(a.value, b.value));
}
static inline AlmostGoldilocksExt2 he2_mul(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    AlmostGoldilocksField m0 = hmul(a.c[0], b.c[0]);
    AlmostGoldilocksField m1 = hmul(a.c[1], b.c[1]);
    AlmostGoldilocksField m2 = hmul(hadd(a.c[0], a.c[1]), hadd(b.c[0], b.c[1]));
    AlmostGoldilocksField m1_3 = hadd(hadd(m1, m1), m1);
    return AlmostGoldilocksExt2(hadd(m0, m1_3), hsub(hsub(m2, m0), m1));
}
static inline AlmostGoldilocksExt2 he2_add(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return AlmostGoldilocksExt2(hadd(a.c[0], b.c[0]), hadd(a.c[1], b.c[1]));
}
static inline AlmostGoldilocksExt2 he2_sub(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return AlmostGoldilocksExt2(hsub(a.c[0], b.c[0]), hsub(a.c[1], b.c[1]));
}

// ============================================================================
// Base-field test
// ============================================================================

static bool test_base(int d, int log_n) {
    std::cout << "  base d=" << d << " log_n=" << log_n << ": ";
    size_t n = 1ULL << log_n;
    size_t half = n / 2;
    std::mt19937_64 rng(d * 9991 + log_n);

    // Pack d polynomials with stride n
    std::vector<uint64_t> polys(d * n);
    for (auto& v : polys) v = rng() % AGL_PRIME;

    // CPU round message
    std::vector<uint64_t> cpu_msg(d + 1, 0);
    for (size_t y = 0; y < half; y++) {
        std::vector<AlmostGoldilocksField> even(d), diff(d);
        for (int i = 0; i < d; i++) {
            size_t base = i * n;
            even[i] = AlmostGoldilocksField(polys[base + 2*y]);
            AlmostGoldilocksField odd(polys[base + 2*y + 1]);
            diff[i] = hsub(odd, even[i]);
        }
        for (int c = 0; c <= d; c++) {
            AlmostGoldilocksField cv((uint64_t)c);
            AlmostGoldilocksField product(1);
            for (int i = 0; i < d; i++) {
                AlmostGoldilocksField val = hadd(even[i], hmul(cv, diff[i]));
                product = hmul(product, val);
            }
            cpu_msg[c] = agl_canonicalize(hadd(AlmostGoldilocksField(cpu_msg[c]), product).value);
        }
    }

    // GPU
    uint64_t* d_polys = nullptr; cudaMalloc(&d_polys, polys.size() * sizeof(uint64_t));
    cudaMemcpy(d_polys, polys.data(), polys.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int blocks = (int)((half + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (blocks > 256) blocks = 256;  // grid-stride loop handles the rest
    uint64_t* d_partial = nullptr;
    cudaMalloc(&d_partial, blocks * (d + 1) * sizeof(uint64_t));

    agl_sumcheck_round_message_kernel<<<blocks, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_partial, d, n, half
    );
    cudaDeviceSynchronize();

    std::vector<uint64_t> h_partial(blocks * (d + 1));
    cudaMemcpy(h_partial.data(), d_partial, h_partial.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Sum across blocks on host
    std::vector<uint64_t> gpu_msg(d + 1, 0);
    for (int c = 0; c <= d; c++) {
        AlmostGoldilocksField acc(0);
        for (int b = 0; b < blocks; b++) {
            acc = hadd(acc, AlmostGoldilocksField(h_partial[b * (d + 1) + c]));
        }
        gpu_msg[c] = agl_canonicalize(acc.value);
    }

    for (int c = 0; c <= d; c++) {
        if (cpu_msg[c] != gpu_msg[c]) {
            std::cerr << "round_message mismatch at c=" << c
                      << " cpu=" << cpu_msg[c] << " gpu=" << gpu_msg[c] << std::endl;
            cudaFree(d_polys); cudaFree(d_partial);
            std::cout << "FAIL" << std::endl;
            return false;
        }
    }

    // Now test fold
    uint64_t challenge = rng() % AGL_PRIME;
    std::vector<uint64_t> cpu_folded(d * n, 0);
    AlmostGoldilocksField ch(challenge);
    for (size_t y = 0; y < half; y++) {
        for (int i = 0; i < d; i++) {
            size_t base = i * n;
            AlmostGoldilocksField a(polys[base + 2*y]);
            AlmostGoldilocksField b(polys[base + 2*y + 1]);
            cpu_folded[base + y] = agl_canonicalize(hadd(a, hmul(ch, hsub(b, a))).value);
        }
    }

    uint64_t* d_folded = nullptr;
    cudaMalloc(&d_folded, d * n * sizeof(uint64_t));
    int fold_blocks = (int)((half + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (fold_blocks > 256) fold_blocks = 256;
    agl_sumcheck_fold_kernel<<<fold_blocks, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_folded, challenge, d, n, half
    );
    cudaDeviceSynchronize();

    std::vector<uint64_t> gpu_folded(d * n);
    cudaMemcpy(gpu_folded.data(), d_folded, gpu_folded.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < d; i++) {
        for (size_t y = 0; y < half; y++) {
            uint64_t got = agl_canonicalize(gpu_folded[i * n + y]);
            if (got != cpu_folded[i * n + y]) {
                std::cerr << "fold mismatch poly=" << i << " y=" << y << std::endl;
                ok = false;
                goto out_base;
            }
        }
    }
out_base:
    cudaFree(d_polys); cudaFree(d_partial); cudaFree(d_folded);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

// ============================================================================
// Ext2 test (smaller because Ext2 mul is heavier on CPU reference)
// ============================================================================

static bool test_ext2(int d, int log_n) {
    std::cout << "  ext2 d=" << d << " log_n=" << log_n << ": ";
    size_t n = 1ULL << log_n;
    size_t half = n / 2;
    std::mt19937_64 rng(d * 19991 + log_n);

    // d polynomials, each n Ext2 elements (2 u64 each), packed back-to-back
    std::vector<uint64_t> polys(d * n * 2);
    for (auto& v : polys) v = rng() % AGL_PRIME;

    // CPU round message
    std::vector<AlmostGoldilocksExt2> cpu_msg(d + 1, AlmostGoldilocksExt2());
    for (size_t y = 0; y < half; y++) {
        std::vector<AlmostGoldilocksExt2> even(d), diff(d);
        for (int i = 0; i < d; i++) {
            size_t base = i * n * 2;
            size_t eo = base + 4 * y, oo = base + 4 * y + 2;
            even[i] = AlmostGoldilocksExt2(polys[eo], polys[eo + 1]);
            AlmostGoldilocksExt2 odd(polys[oo], polys[oo + 1]);
            diff[i] = he2_sub(odd, even[i]);
        }
        for (int c = 0; c <= d; c++) {
            AlmostGoldilocksExt2 cv(AlmostGoldilocksField((uint64_t)c));
            AlmostGoldilocksExt2 product(AlmostGoldilocksField(1));
            for (int i = 0; i < d; i++) {
                product = he2_mul(product, he2_add(even[i], he2_mul(cv, diff[i])));
            }
            cpu_msg[c] = he2_add(cpu_msg[c], product);
        }
    }

    // GPU
    uint64_t* d_polys = nullptr;
    cudaMalloc(&d_polys, polys.size() * sizeof(uint64_t));
    cudaMemcpy(d_polys, polys.data(), polys.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int blocks = (int)((half + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (blocks > 64) blocks = 64;
    uint64_t* d_partial = nullptr;
    cudaMalloc(&d_partial, blocks * (d + 1) * 2 * sizeof(uint64_t));

    aext2_sumcheck_round_message_kernel<<<blocks, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_partial, d, n, half
    );
    cudaDeviceSynchronize();

    std::vector<uint64_t> h_partial(blocks * (d + 1) * 2);
    cudaMemcpy(h_partial.data(), d_partial, h_partial.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int c = 0; c <= d; c++) {
        AlmostGoldilocksExt2 acc;
        for (int b = 0; b < blocks; b++) {
            AlmostGoldilocksExt2 p(h_partial[(b * (d + 1) + c) * 2],
                                   h_partial[(b * (d + 1) + c) * 2 + 1]);
            acc = he2_add(acc, p);
        }
        uint64_t e0 = agl_canonicalize(cpu_msg[c].c[0].value);
        uint64_t e1 = agl_canonicalize(cpu_msg[c].c[1].value);
        uint64_t g0 = agl_canonicalize(acc.c[0].value);
        uint64_t g1 = agl_canonicalize(acc.c[1].value);
        if (e0 != g0 || e1 != g1) {
            std::cerr << "ext2 round_message mismatch at c=" << c << std::endl;
            ok = false;
            break;
        }
    }
    cudaFree(d_polys); cudaFree(d_partial);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

int main() {
    int n = 0; cudaGetDeviceCount(&n);
    if (n == 0) { std::cerr << "No CUDA device" << std::endl; return 1; }
    std::cout << "=== almost-Goldilocks sumcheck tests ===" << std::endl;
    bool ok = true;
    for (int d : {1, 2, 3, 4})
        for (int log_n : {6, 10})
            ok &= test_base(d, log_n);
    for (int d : {1, 2, 3})
        for (int log_n : {6, 10})
            ok &= test_ext2(d, log_n);
    std::cout << (ok ? "ALL PASS" : "FAILURES PRESENT") << std::endl;
    return ok ? 0 : 1;
}

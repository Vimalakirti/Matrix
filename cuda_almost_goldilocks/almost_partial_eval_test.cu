/**
 * Partial-eval correctness tests for the almost-Goldilocks field.
 *
 * Checks both base-field and base→Ext2 folding paths against a CPU
 * reference. The reference uses the same identity each round:
 *
 *     out[j] = in[2j] + r_i * (in[2j+1] - in[2j])
 *
 * which reduces 2^N evaluations to 2^{N-m} by evaluating the first m vars.
 */

#include "almost_partial_eval.cuh"
#include <iostream>
#include <vector>
#include <random>

#define AGL_PRIME ALMOST_GOLDILOCKS_PRIME
#define EXPECT(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL (" << __FILE__ << ":" << __LINE__ << "): " << msg << std::endl; \
    return false; } } while (0)

// ------ host helpers ------
static inline AlmostGoldilocksField h_mul(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    agl_uint128_t p = agl_mul_u64_u64(a.value, b.value);
    return AlmostGoldilocksField(agl_reduce128(p));
}
static inline AlmostGoldilocksField h_sub(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return AlmostGoldilocksField(agl_sub_no_canonicalize(a.value, b.value));
}
static inline AlmostGoldilocksField h_add(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return AlmostGoldilocksField(agl_add_no_canonicalize(a.value, b.value));
}

static inline AlmostGoldilocksExt2 h_e2_mul(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    AlmostGoldilocksField m0 = h_mul(a.c[0], b.c[0]);
    AlmostGoldilocksField m1 = h_mul(a.c[1], b.c[1]);
    AlmostGoldilocksField m2 = h_mul(h_add(a.c[0], a.c[1]), h_add(b.c[0], b.c[1]));
    AlmostGoldilocksField m1_3 = h_add(h_add(m1, m1), m1);
    return AlmostGoldilocksExt2(h_add(m0, m1_3), h_sub(h_sub(m2, m0), m1));
}
static inline AlmostGoldilocksExt2 h_e2_sub(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return AlmostGoldilocksExt2(h_sub(a.c[0], b.c[0]), h_sub(a.c[1], b.c[1]));
}
static inline AlmostGoldilocksExt2 h_e2_add(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return AlmostGoldilocksExt2(h_add(a.c[0], b.c[0]), h_add(a.c[1], b.c[1]));
}

// CPU reference: partial eval (base field)
static std::vector<AlmostGoldilocksField> cpu_pe_base(
    std::vector<AlmostGoldilocksField> evals,
    const std::vector<AlmostGoldilocksField>& r
) {
    int m = (int)r.size();
    for (int i = 0; i < m; i++) {
        size_t pairs = evals.size() / 2;
        std::vector<AlmostGoldilocksField> next(pairs);
        for (size_t j = 0; j < pairs; j++) {
            AlmostGoldilocksField a = evals[2*j], b = evals[2*j+1];
            next[j] = h_add(a, h_mul(r[i], h_sub(b, a)));
        }
        evals = std::move(next);
    }
    return evals;
}
// CPU reference: partial eval (Ext2 r over base evals)
static std::vector<AlmostGoldilocksExt2> cpu_pe_ext2(
    const std::vector<AlmostGoldilocksField>& evals,
    const std::vector<AlmostGoldilocksExt2>& r
) {
    int m = (int)r.size();
    size_t pairs = evals.size() / 2;
    std::vector<AlmostGoldilocksExt2> cur(pairs);
    // Round 0: mixed
    for (size_t j = 0; j < pairs; j++) {
        AlmostGoldilocksField a = evals[2*j], b = evals[2*j+1];
        AlmostGoldilocksField diff = h_sub(b, a);
        cur[j] = AlmostGoldilocksExt2(
            h_add(a, h_mul(r[0].c[0], diff)),
            h_mul(r[0].c[1], diff)
        );
    }
    for (int i = 1; i < m; i++) {
        size_t p = cur.size() / 2;
        std::vector<AlmostGoldilocksExt2> next(p);
        for (size_t j = 0; j < p; j++) {
            auto a = cur[2*j], b = cur[2*j+1];
            next[j] = h_e2_add(a, h_e2_mul(r[i], h_e2_sub(b, a)));
        }
        cur = std::move(next);
    }
    return cur;
}

static bool test_base(int log_n, int m) {
    std::cout << "  base log_n=" << log_n << " m=" << m << ": ";
    size_t n = 1ULL << log_n;
    std::mt19937_64 rng(log_n * 1000 + m);
    std::vector<AlmostGoldilocksField> evals(n);
    for (auto& e : evals) e = AlmostGoldilocksField(rng() % AGL_PRIME);
    std::vector<AlmostGoldilocksField> r(m);
    for (auto& e : r) e = AlmostGoldilocksField(rng() % AGL_PRIME);

    auto cpu = cpu_pe_base(evals, r);

    AlmostGoldilocksField *d_data, *d_scratch, *d_r;
    cudaMalloc(&d_data, n * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_scratch, (n / 2) * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_r, m * sizeof(AlmostGoldilocksField));
    cudaMemcpy(d_data, evals.data(), n * sizeof(AlmostGoldilocksField), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r.data(), m * sizeof(AlmostGoldilocksField), cudaMemcpyHostToDevice);

    agl_partial_eval(d_data, d_scratch, d_r, log_n, m);
    cudaDeviceSynchronize();

    size_t out_size = 1ULL << (log_n - m);
    std::vector<AlmostGoldilocksField> gpu(out_size);
    cudaMemcpy(gpu.data(), d_data, out_size * sizeof(AlmostGoldilocksField), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < out_size; i++) {
        if (agl_canonicalize(gpu[i].value) != agl_canonicalize(cpu[i].value)) {
            std::cerr << "mismatch at " << i << " gpu=" << agl_canonicalize(gpu[i].value)
                      << " cpu=" << agl_canonicalize(cpu[i].value) << std::endl;
            ok = false;
            break;
        }
    }
    cudaFree(d_data); cudaFree(d_scratch); cudaFree(d_r);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

static bool test_ext2(int log_n, int m) {
    std::cout << "  ext2 log_n=" << log_n << " m=" << m << ": ";
    size_t n = 1ULL << log_n;
    std::mt19937_64 rng(log_n * 2000 + m);
    std::vector<AlmostGoldilocksField> evals(n);
    for (auto& e : evals) e = AlmostGoldilocksField(rng() % AGL_PRIME);
    std::vector<AlmostGoldilocksExt2> r(m);
    for (auto& e : r) {
        e.c[0] = AlmostGoldilocksField(rng() % AGL_PRIME);
        e.c[1] = AlmostGoldilocksField(rng() % AGL_PRIME);
    }
    auto cpu = cpu_pe_ext2(evals, r);

    AlmostGoldilocksField* d_input;
    AlmostGoldilocksExt2 *d_output, *d_scratch, *d_r;
    cudaMalloc(&d_input, n * sizeof(AlmostGoldilocksField));
    cudaMalloc(&d_output, (n/2) * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_scratch, (n/4 > 0 ? n/4 : 1) * sizeof(AlmostGoldilocksExt2));
    cudaMalloc(&d_r, m * sizeof(AlmostGoldilocksExt2));
    cudaMemcpy(d_input, evals.data(), n * sizeof(AlmostGoldilocksField), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r.data(), m * sizeof(AlmostGoldilocksExt2), cudaMemcpyHostToDevice);

    agl_partial_eval_ext2_from_base(d_input, d_output, d_scratch, d_r, log_n, m);
    cudaDeviceSynchronize();

    size_t out_size = 1ULL << (log_n - m);
    std::vector<AlmostGoldilocksExt2> gpu(out_size);
    cudaMemcpy(gpu.data(), d_output, out_size * sizeof(AlmostGoldilocksExt2), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < out_size; i++) {
        uint64_t g0 = agl_canonicalize(gpu[i].c[0].value), g1 = agl_canonicalize(gpu[i].c[1].value);
        uint64_t c0 = agl_canonicalize(cpu[i].c[0].value), c1 = agl_canonicalize(cpu[i].c[1].value);
        if (g0 != c0 || g1 != c1) {
            std::cerr << "ext2 mismatch at " << i << " gpu=(" << g0 << "," << g1
                      << ") cpu=(" << c0 << "," << c1 << ")" << std::endl;
            ok = false; break;
        }
    }
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_scratch); cudaFree(d_r);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

int main() {
    int n = 0; cudaGetDeviceCount(&n);
    if (n == 0) { std::cerr << "No CUDA device" << std::endl; return 1; }
    std::cout << "=== almost-Goldilocks partial_eval tests ===" << std::endl;
    bool ok = true;
    for (int log_n : {6, 10, 14}) {
        for (int m : {1, 3, log_n - 1, log_n}) {
            if (m < 0 || m > log_n) continue;
            ok &= test_base(log_n, m);
        }
    }
    for (int log_n : {6, 10, 14}) {
        for (int m : {1, 3, log_n - 1, log_n}) {
            if (m < 1 || m > log_n) continue;
            ok &= test_ext2(log_n, m);
        }
    }
    std::cout << (ok ? "ALL PASS" : "FAILURES PRESENT") << std::endl;
    return ok ? 0 : 1;
}

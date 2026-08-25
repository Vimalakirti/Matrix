/**
 * Eq Lagrange CUDA Kernels Test
 *
 * Tests that both DP and WHT implementations of eq(r, x) produce identical results.
 */

#include "eq_lagrange.cuh"
#include <stdio.h>

#ifdef EQ_LAGRANGE_TEST

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Host-side field operations (using the __host__ __device__ primitives)
inline GoldilocksField host_gl_mul(GoldilocksField a, GoldilocksField b) {
    uint128_t prod = mul_u64_u64(a.value, b.value);
    return GoldilocksField(reduce128(prod));
}

inline GoldilocksField host_gl_sub(GoldilocksField a, GoldilocksField b) {
    return GoldilocksField(sub_no_canonicalize(a.value, b.value));
}

/**
 * CPU reference implementation of eq(r, x)
 * eq(r, x) = prod_{i=0}^{n-1} (r_i * x_i + (1 - r_i) * (1 - x_i))
 */
void eq_cpu_reference(
    const std::vector<GoldilocksField>& r,
    std::vector<GoldilocksField>& result
) {
    int log_n = r.size();
    size_t n = 1ULL << log_n;
    result.resize(n);

    for (size_t x = 0; x < n; x++) {
        GoldilocksField acc(1);
        for (int i = 0; i < log_n; i++) {
            int x_i = (x >> i) & 1;
            if (x_i) {
                // r_i * 1 + (1 - r_i) * 0 = r_i
                acc = host_gl_mul(acc, r[i]);
            } else {
                // r_i * 0 + (1 - r_i) * 1 = 1 - r_i
                acc = host_gl_mul(acc, host_gl_sub(GoldilocksField(1), r[i]));
            }
        }
        result[x] = acc;
    }
}

bool test_eq_algorithms(int log_n) {
    std::cout << "Testing eq(r, x) with log_n = " << log_n << " (N = " << (1 << log_n) << ")" << std::endl;

    size_t n = 1ULL << log_n;

    // Generate random r values
    std::vector<GoldilocksField> h_r(log_n);
    std::mt19937_64 rng(42 + log_n);
    for (int i = 0; i < log_n; i++) {
        h_r[i] = GoldilocksField(rng() % GOLDILOCKS_PRIME);
    }

    // CPU reference
    std::vector<GoldilocksField> cpu_result;
    eq_cpu_reference(h_r, cpu_result);

    // Allocate device memory
    GoldilocksField *d_r, *d_buf_a, *d_buf_b, *d_wht_data;
    cudaMalloc(&d_r, sizeof(GoldilocksField) * log_n);
    cudaMalloc(&d_buf_a, sizeof(GoldilocksField) * n);
    cudaMalloc(&d_buf_b, sizeof(GoldilocksField) * n);
    cudaMalloc(&d_wht_data, sizeof(GoldilocksField) * n);

    // Copy r to device
    cudaMemcpy(d_r, h_r.data(), sizeof(GoldilocksField) * log_n, cudaMemcpyHostToDevice);

    // Test DP algorithm
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    GoldilocksField* d_dp_result = nullptr;
    cudaEventRecord(start);
    cudaError_t err = eq_dp_all(d_r, d_buf_a, d_buf_b, log_n, &d_dp_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (err != cudaSuccess) {
        std::cerr << "DP kernel error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    float dp_ms;
    cudaEventElapsedTime(&dp_ms, start, stop);

    std::vector<GoldilocksField> dp_result(n);
    cudaMemcpy(dp_result.data(), d_dp_result, sizeof(GoldilocksField) * n, cudaMemcpyDeviceToHost);

    // Test WHT algorithm
    cudaEventRecord(start);
    err = eq_wht_all(d_r, d_wht_data, log_n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (err != cudaSuccess) {
        std::cerr << "WHT kernel error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    float wht_ms;
    cudaEventElapsedTime(&wht_ms, start, stop);

    std::vector<GoldilocksField> wht_result(n);
    cudaMemcpy(wht_result.data(), d_wht_data, sizeof(GoldilocksField) * n, cudaMemcpyDeviceToHost);

    // Compare results
    bool dp_correct = true;
    bool wht_correct = true;
    bool dp_wht_match = true;

    for (size_t i = 0; i < n; i++) {
        uint64_t cpu_val = canonicalize(cpu_result[i].value);
        uint64_t dp_val = canonicalize(dp_result[i].value);
        uint64_t wht_val = canonicalize(wht_result[i].value);

        if (dp_val != cpu_val) {
            if (dp_correct) {
                std::cout << "DP mismatch at " << i << ": expected " << cpu_val << ", got " << dp_val << std::endl;
            }
            dp_correct = false;
        }

        if (wht_val != cpu_val) {
            if (wht_correct) {
                std::cout << "WHT mismatch at " << i << ": expected " << cpu_val << ", got " << wht_val << std::endl;
            }
            wht_correct = false;
        }

        if (dp_val != wht_val) {
            if (dp_wht_match) {
                std::cout << "DP vs WHT mismatch at " << i << ": DP=" << dp_val << ", WHT=" << wht_val << std::endl;
            }
            dp_wht_match = false;
        }
    }

    std::cout << "  DP algorithm:  " << (dp_correct ? "PASS" : "FAIL") << " (" << dp_ms << " ms)" << std::endl;
    std::cout << "  WHT algorithm: " << (wht_correct ? "PASS" : "FAIL") << " (" << wht_ms << " ms)" << std::endl;
    std::cout << "  DP == WHT:     " << (dp_wht_match ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    cudaFree(d_r);
    cudaFree(d_buf_a);
    cudaFree(d_buf_b);
    cudaFree(d_wht_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return dp_correct && wht_correct && dp_wht_match;
}

// ============================================================
// Ext2 Tests
// ============================================================

// Host-side ext2 operations
inline GoldilocksExt2 host_ext2_mul(GoldilocksExt2 a, GoldilocksExt2 b) {
    GoldilocksField b1_w = host_gl_mul(b.c[1], GoldilocksField(EXT2_W));
    GoldilocksField c0 = GoldilocksField(add_no_canonicalize(
        host_gl_mul(a.c[0], b.c[0]).value,
        host_gl_mul(a.c[1], b1_w).value
    ));
    GoldilocksField c1 = GoldilocksField(add_no_canonicalize(
        host_gl_mul(a.c[0], b.c[1]).value,
        host_gl_mul(a.c[1], b.c[0]).value
    ));
    return GoldilocksExt2(c0, c1);
}

inline GoldilocksExt2 host_ext2_sub(GoldilocksExt2 a, GoldilocksExt2 b) {
    return GoldilocksExt2(
        host_gl_sub(a.c[0], b.c[0]),
        host_gl_sub(a.c[1], b.c[1])
    );
}

/**
 * CPU reference implementation of eq(r, x) for Ext2
 * eq(r, x) = prod_{i=0}^{n-1} (r_i * x_i + (1 - r_i) * (1 - x_i))
 */
void ext2_eq_cpu_reference(
    const std::vector<GoldilocksExt2>& r,
    std::vector<GoldilocksExt2>& result
) {
    int log_n = r.size();
    size_t n = 1ULL << log_n;
    result.resize(n);

    GoldilocksExt2 one(GoldilocksField(1), GoldilocksField(0));

    for (size_t x = 0; x < n; x++) {
        GoldilocksExt2 acc = one;
        for (int i = 0; i < log_n; i++) {
            int x_i = (x >> i) & 1;
            if (x_i) {
                // r_i * 1 + (1 - r_i) * 0 = r_i
                acc = host_ext2_mul(acc, r[i]);
            } else {
                // r_i * 0 + (1 - r_i) * 1 = 1 - r_i
                acc = host_ext2_mul(acc, host_ext2_sub(one, r[i]));
            }
        }
        result[x] = acc;
    }
}

bool test_ext2_eq_dp(int log_n) {
    std::cout << "Testing ext2_eq_dp_all with log_n = " << log_n << " (N = " << (1 << log_n) << ")" << std::endl;

    size_t n = 1ULL << log_n;

    // Generate random ext2 r values
    std::vector<GoldilocksExt2> h_r(log_n);
    std::mt19937_64 rng(123 + log_n);
    for (int i = 0; i < log_n; i++) {
        h_r[i] = GoldilocksExt2(
            GoldilocksField(rng() % GOLDILOCKS_PRIME),
            GoldilocksField(rng() % GOLDILOCKS_PRIME)
        );
    }

    // CPU reference
    std::vector<GoldilocksExt2> cpu_result;
    ext2_eq_cpu_reference(h_r, cpu_result);

    // Allocate device memory
    GoldilocksExt2 *d_r, *d_buf_a, *d_buf_b;
    cudaMalloc(&d_r, sizeof(GoldilocksExt2) * log_n);
    cudaMalloc(&d_buf_a, sizeof(GoldilocksExt2) * n);
    cudaMalloc(&d_buf_b, sizeof(GoldilocksExt2) * n);

    // Copy r to device
    cudaMemcpy(d_r, h_r.data(), sizeof(GoldilocksExt2) * log_n, cudaMemcpyHostToDevice);

    // Test DP algorithm
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    GoldilocksExt2* d_dp_result = nullptr;
    cudaEventRecord(start);
    cudaError_t err = ext2_eq_dp_all(d_r, d_buf_a, d_buf_b, log_n, &d_dp_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (err != cudaSuccess) {
        std::cerr << "Ext2 DP kernel error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    float dp_ms;
    cudaEventElapsedTime(&dp_ms, start, stop);

    std::vector<GoldilocksExt2> dp_result(n);
    cudaMemcpy(dp_result.data(), d_dp_result, sizeof(GoldilocksExt2) * n, cudaMemcpyDeviceToHost);

    // Compare results
    bool dp_correct = true;
    int mismatch_count = 0;

    for (size_t i = 0; i < n; i++) {
        uint64_t cpu_c0 = canonicalize(cpu_result[i].c[0].value);
        uint64_t cpu_c1 = canonicalize(cpu_result[i].c[1].value);
        uint64_t dp_c0 = canonicalize(dp_result[i].c[0].value);
        uint64_t dp_c1 = canonicalize(dp_result[i].c[1].value);

        if (dp_c0 != cpu_c0 || dp_c1 != cpu_c1) {
            if (mismatch_count < 3) {
                std::cout << "  Ext2 DP mismatch at " << i << ": expected ("
                          << cpu_c0 << ", " << cpu_c1 << "), got ("
                          << dp_c0 << ", " << dp_c1 << ")" << std::endl;
            }
            dp_correct = false;
            mismatch_count++;
        }
    }

    if (mismatch_count > 3) {
        std::cout << "  ... and " << (mismatch_count - 3) << " more mismatches" << std::endl;
    }

    std::cout << "  Ext2 DP algorithm: " << (dp_correct ? "PASS" : "FAIL") << " (" << dp_ms << " ms)" << std::endl;

    // Cleanup
    cudaFree(d_r);
    cudaFree(d_buf_a);
    cudaFree(d_buf_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return dp_correct;
}

int main() {
    std::cout << "=== Eq Lagrange GPU Tests ===" << std::endl;
    std::cout << std::endl;

    bool all_pass = true;

    // Test base field (various sizes)
    std::cout << "--- Base Field Tests ---" << std::endl;
    for (int log_n = 4; log_n <= 20; log_n += 4) {
        if (!test_eq_algorithms(log_n)) {
            all_pass = false;
        }
        std::cout << std::endl;
    }

    // Test ext2 field
    std::cout << "--- Ext2 Field Tests ---" << std::endl;
    for (int log_n = 4; log_n <= 20; log_n += 4) {
        if (!test_ext2_eq_dp(log_n)) {
            all_pass = false;
        }
        std::cout << std::endl;
    }

    std::cout << "=== Summary ===" << std::endl;
    if (all_pass) {
        std::cout << "All tests PASSED!" << std::endl;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }

    return 0;
}

#endif // EQ_LAGRANGE_TEST

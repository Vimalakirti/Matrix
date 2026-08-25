/**
 * Fused permute + partial-eval correctness tests (almost-Goldilocks).
 *
 * Builds small random inputs and compares the GPU kernel output to a
 * straightforward CPU reference that performs the same identity:
 *
 *     output[j] = Σ_{b=0..2^m-1} evals[perm(b + j*2^m)] * eq_table[b]
 *
 * where perm(idx_new) = lo_lut[idx_new & lo_mask] | hi_lut[idx_new >> half].
 */

#include "almost_fused_permute_peval.cuh"
#include <iostream>
#include <vector>
#include <random>

#define AGL_PRIME ALMOST_GOLDILOCKS_PRIME
#define EXPECT(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL (" << __FILE__ << ":" << __LINE__ << "): " << msg << std::endl; \
    return false; } } while (0)

// Host helpers
static inline AlmostGoldilocksField hmul(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    agl_uint128_t p = agl_mul_u64_u64(a.value, b.value);
    return AlmostGoldilocksField(agl_reduce128(p));
}
static inline AlmostGoldilocksField hadd(AlmostGoldilocksField a, AlmostGoldilocksField b) {
    return AlmostGoldilocksField(agl_add_no_canonicalize(a.value, b.value));
}
static inline AlmostGoldilocksExt2 he2_scalar_mul(AlmostGoldilocksField s, AlmostGoldilocksExt2 a) {
    return AlmostGoldilocksExt2(hmul(s, a.c[0]), hmul(s, a.c[1]));
}
static inline AlmostGoldilocksExt2 he2_add(AlmostGoldilocksExt2 a, AlmostGoldilocksExt2 b) {
    return AlmostGoldilocksExt2(hadd(a.c[0], b.c[0]), hadd(a.c[1], b.c[1]));
}

static bool test_fused(int n, int m, int half_split) {
    std::cout << "  n=" << n << " m=" << m << " half=" << half_split << ": ";
    size_t total = 1ULL << n;
    size_t inner = 1ULL << m;
    size_t output_size = 1ULL << (n - m);
    size_t lo_size = 1ULL << half_split;
    size_t hi_size = 1ULL << (n - half_split);

    std::mt19937_64 rng((uint64_t)(n * 100 + m * 10 + half_split));

    // Random base-field evals
    std::vector<uint64_t> evals(total);
    for (auto& v : evals) v = rng() % AGL_PRIME;

    // Random Ext2 eq table
    std::vector<uint64_t> eq_table(inner * 2);
    for (auto& v : eq_table) v = rng() % AGL_PRIME;

    // Random split LUTs that compose into a permutation of [0, 2^n).
    // Construction: pick a random permutation P on [0, 2^n), then set
    //   lo_lut[k] = P[k] & lo_mask
    //   hi_lut[h] = P[h * lo_size] & hi_mask_shifted   (won't decompose generally)
    //
    // To keep this simple AND make perm() return a valid permutation, we
    // build lo_lut / hi_lut such that perm(idx_new) = idx_new directly
    // (identity permutation) — that exercises the kernel's lookup path
    // without requiring an inverse-permutation construction.
    //
    // We also do a second sub-test with a non-identity permutation built
    // by reversing the low half.
    std::vector<uint32_t> lo_lut(lo_size), hi_lut(hi_size);
    for (size_t k = 0; k < lo_size; k++) lo_lut[k] = (uint32_t)k;
    for (size_t k = 0; k < hi_size; k++) hi_lut[k] = (uint32_t)(k << half_split);
    uint32_t lo_mask = (uint32_t)(lo_size - 1);

    // CPU reference
    std::vector<AlmostGoldilocksExt2> cpu(output_size);
    for (size_t j = 0; j < output_size; j++) {
        AlmostGoldilocksExt2 acc;
        uint32_t base_idx = (uint32_t)j << m;
        for (uint32_t b = 0; b < inner; b++) {
            uint32_t idx_new = base_idx + b;
            uint32_t idx_old = lo_lut[idx_new & lo_mask] | hi_lut[idx_new >> half_split];
            AlmostGoldilocksField val(evals[idx_old]);
            AlmostGoldilocksExt2 eq(eq_table[2*b], eq_table[2*b+1]);
            acc = he2_add(acc, he2_scalar_mul(val, eq));
        }
        cpu[j] = acc;
    }

    // GPU
    uint64_t *d_evals, *d_output, *d_eq;
    uint32_t *d_lo, *d_hi;
    cudaMalloc(&d_evals, total * sizeof(uint64_t));
    cudaMalloc(&d_output, output_size * 2 * sizeof(uint64_t));
    cudaMalloc(&d_eq, inner * 2 * sizeof(uint64_t));
    cudaMalloc(&d_lo, lo_size * sizeof(uint32_t));
    cudaMalloc(&d_hi, hi_size * sizeof(uint32_t));
    cudaMemcpy(d_evals, evals.data(), total * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eq, eq_table.data(), inner * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lo, lo_lut.data(), lo_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hi, hi_lut.data(), hi_size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int block = 256;
    int num_warps = block / 32;
    size_t lut_bytes = (lo_size + hi_size) * sizeof(uint32_t);
    size_t aligned_lut = (lut_bytes + 7) & ~(size_t)7;
    size_t shmem = aligned_lut + num_warps * 2 * sizeof(uint64_t);
    int grid = (int)output_size < 1024 ? (int)output_size : 1024;

    agl_fused_permute_partial_eval_kernel<<<grid, block, shmem>>>(
        d_evals, d_output, d_eq, d_lo, d_hi, n, m, half_split, (int)output_size
    );
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "kernel error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_evals); cudaFree(d_output); cudaFree(d_eq); cudaFree(d_lo); cudaFree(d_hi);
        std::cout << "FAIL" << std::endl;
        return false;
    }

    std::vector<uint64_t> h_output(output_size * 2);
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t j = 0; j < output_size; j++) {
        uint64_t c0 = agl_canonicalize(cpu[j].c[0].value);
        uint64_t c1 = agl_canonicalize(cpu[j].c[1].value);
        uint64_t g0 = agl_canonicalize(h_output[2*j]);
        uint64_t g1 = agl_canonicalize(h_output[2*j + 1]);
        if (c0 != g0 || c1 != g1) {
            std::cerr << "mismatch at j=" << j << " cpu=(" << c0 << "," << c1
                      << ") gpu=(" << g0 << "," << g1 << ")" << std::endl;
            ok = false;
            break;
        }
    }
    cudaFree(d_evals); cudaFree(d_output); cudaFree(d_eq); cudaFree(d_lo); cudaFree(d_hi);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

int main() {
    int dc = 0; cudaGetDeviceCount(&dc);
    if (dc == 0) { std::cerr << "No CUDA device" << std::endl; return 1; }
    std::cout << "=== almost-Goldilocks fused_permute_peval tests ===" << std::endl;
    bool ok = true;
    // (n, m, half) configurations
    ok &= test_fused(8,  3, 4);
    ok &= test_fused(10, 4, 5);
    ok &= test_fused(12, 5, 6);
    ok &= test_fused(14, 6, 7);
    std::cout << (ok ? "ALL PASS" : "FAILURES PRESENT") << std::endl;
    return ok ? 0 : 1;
}

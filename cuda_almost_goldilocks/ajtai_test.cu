/**
 * Tests for the GPU Ajtai commitment.
 *
 *   Stage 1: CPU reference at D=4 and D=8, validated against naive
 *            polynomial multiplication (the binary-selected-rotation
 *            path must equal the full polynomial product).
 *   Stage 2: ChaCha8 PRG determinism, basic sanity (mean ≈ q/2).
 *   Stage 3: Negacyclic shift correctness vs CPU reference at D=64.
 *   Stage 4: GPU dense single commit vs CPU reference for several N.
 *   Stage 5: GPU dense batched commit equals B independent singles.
 *   Stage 6: GPU sparse commit vs GPU dense commit on the same z.
 *   Stage 7: GPU result invariant to CHUNK size.
 *   Stage 8: All-zero witness corner case.
 */

#include "ajtai.cuh"
#include "ajtai_cpu_reference.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <random>
#include <iostream>
#include <string>
#include <cmath>

using namespace ajtai;
using ajtai_cpu::Ring;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL (" << __FILE__ << ":" << __LINE__ << "): " << msg << std::endl; \
        return false; \
    } \
} while (0)

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        return false; \
    } \
} while (0)

// ============================================================================
// Helpers
// ============================================================================

static void make_key(std::mt19937_64& rng, uint32_t key[8]) {
    for (int i = 0; i < 8; i++) key[i] = (uint32_t)rng();
}

static std::vector<uint64_t> make_z_bits(std::mt19937_64& rng, uint64_t N, int D = 64) {
    std::vector<uint64_t> z(N);
    uint64_t mask = (D == 64) ? ~0ULL : ((1ULL << D) - 1);
    for (auto& b : z) b = rng() & mask;
    return z;
}

template <int D>
static bool rings_equal(const Ring<D>& a, const Ring<D>& b) {
    for (int r = 0; r < D; r++) {
        if (ajtai_cpu::h_canon(a[r]) != ajtai_cpu::h_canon(b[r])) return false;
    }
    return true;
}

// ============================================================================
// Stage 1: CPU reference at small D
// ============================================================================

template <int D>
static bool test_cpu_ref_at_D(int label_d) {
    std::cout << "  D=" << label_d << ": ";

    // 1. X^0 * a = a
    Ring<D> a; for (int r = 0; r < D; r++) a[r] = (uint64_t)(r * 0x9E3779B97F4A7C15ULL);
    Ring<D> s0 = ajtai_cpu::cpu_ring_shift<D>(a, 0);
    EXPECT(rings_equal<D>(s0, a), "X^0 * a == a");

    // 2. For each ell in [0, D), the formula matches the hand rule.
    for (int ell = 0; ell < D; ell++) {
        Ring<D> s = ajtai_cpu::cpu_ring_shift<D>(a, ell);
        for (int r = 0; r < D; r++) {
            int idx = r - ell;
            uint64_t expected;
            if (idx >= 0) {
                expected = a[idx];
            } else {
                expected = ajtai_cpu::h_neg(a[idx + D]);
            }
            EXPECT(ajtai_cpu::h_canon(s[r]) == ajtai_cpu::h_canon(expected),
                   "X^ell shift formula");
        }
    }

    // 3. Binary-selected-rotation == naive polynomial multiplication
    std::mt19937_64 rng(0xA17A1u + (unsigned)D);
    for (int trial = 0; trial < 20; trial++) {
        Ring<D> ar;
        for (int r = 0; r < D; r++) {
            ar[r] = rng() % ALMOST_GOLDILOCKS_PRIME;
        }
        uint64_t mask = rng() & ((D == 64) ? ~0ULL : ((1ULL << D) - 1));

        // Build the z polynomial: z[r] = bit r of mask.
        Ring<D> zr{};
        for (int r = 0; r < D; r++) {
            zr[r] = (mask >> r) & 1ULL;
        }
        Ring<D> naive_prod = ajtai_cpu::cpu_ring_mul<D>(ar, zr);
        Ring<D> bin_prod = ajtai_cpu::cpu_ring_binary_mul<D>(ar, mask);
        EXPECT(rings_equal<D>(naive_prod, bin_prod),
               "binary-selected-rotation == naive ring mul");
    }

    std::cout << "PASS" << std::endl;
    return true;
}

static bool test_cpu_reference() {
    std::cout << "[cpu_reference]\n";
    bool ok = true;
    ok &= test_cpu_ref_at_D<4>(4);
    ok &= test_cpu_ref_at_D<8>(8);
    ok &= test_cpu_ref_at_D<64>(64);
    return ok;
}

// ============================================================================
// Stage 2: ChaCha8 PRG determinism + sanity
// ============================================================================

static bool test_chacha8_determinism() {
    std::cout << "[chacha8_determinism] ";
    uint32_t key[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t nonce[3] = {42, 1000, 1};

    uint32_t out_a[16], out_b[16];
    chacha8_block(key, 0, nonce, out_a);
    chacha8_block(key, 0, nonce, out_b);
    for (int i = 0; i < 16; i++) {
        EXPECT(out_a[i] == out_b[i], "same input -> same output");
    }
    chacha8_block(key, 1, nonce, out_b);
    bool any_diff = false;
    for (int i = 0; i < 16; i++) {
        if (out_a[i] != out_b[i]) { any_diff = true; break; }
    }
    EXPECT(any_diff, "different counter -> different output");
    std::cout << "PASS" << std::endl;
    return true;
}

static bool test_prg_mean_sanity() {
    std::cout << "[prg_mean_sanity] ";
    // Sample 2^16 matrix entries and check the empirical mean is within 3σ of q/2.
    // σ_mean = (q / sqrt(12)) / sqrt(N) for uniform [0, q).
    uint32_t key[8] = {9, 8, 7, 6, 5, 4, 3, 2};
    const int N_SAMPLES = 1 << 14;  // 16384
    __int128 sum = 0;
    for (uint64_t j = 0; j < (uint64_t)N_SAMPLES / 8; j++) {
        uint64_t buf[8];
        prg_ring_block_chacha8(key, 0u, j, 0u, buf);
        for (int k = 0; k < 8; k++) sum += (__int128)buf[k];
    }
    // Empirical mean fits in uint64_t (it's < q < 2^64). q/2 ≈ 2^63 - 2^32
    // sits just below int64_t MAX, so a slightly-above-q/2 mean wraps to
    // negative under naive int64_t cast — work in unsigned then signed-diff.
    uint64_t mean_u = (uint64_t)(sum / N_SAMPLES);
    int64_t  dev    = (int64_t)mean_u - (int64_t)(ALMOST_GOLDILOCKS_PRIME / 2);
    double q = (double)ALMOST_GOLDILOCKS_PRIME;
    double sigma_mean = q / (sqrt(12.0) * sqrt((double)N_SAMPLES));
    double z = (double)dev / sigma_mean;
    EXPECT(z > -5.0 && z < 5.0,
           "empirical mean within 5σ of q/2 (gross bias check, not security check)");
    std::cout << "PASS  (mean dev = " << (double)dev << ", |z| = "
              << (z < 0 ? -z : z) << ")" << std::endl;
    return true;
}

// ============================================================================
// Stage 3-7: GPU vs CPU, dense / batched / sparse / chunk-invariance
// ============================================================================

// Run dense batched commit on GPU, return host-side B*KAPPA Ring<64>.
template <int B, int CHUNK>
static std::vector<Ring<64>> gpu_dense_batched(
    const uint32_t key[8],
    const std::vector<uint64_t>& z_bits_packed,
    uint64_t N
) {
    uint32_t* d_key;
    uint64_t* d_z;
    uint64_t* d_out;
    cudaMalloc(&d_key, 8 * sizeof(uint32_t));
    cudaMalloc(&d_z, z_bits_packed.size() * sizeof(uint64_t));
    cudaMalloc(&d_out, (size_t)B * KAPPA * D * sizeof(uint64_t));
    cudaMemcpy(d_key, key, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z_bits_packed.data(), z_bits_packed.size() * sizeof(uint64_t),
               cudaMemcpyHostToDevice);

    cudaError_t run_err = commit_dense_batched_run<B, CHUNK>(d_key, d_z, N, d_out);
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (run_err != cudaSuccess) {
        std::cerr << "  [launch err " << B << "/" << CHUNK << "]: "
                  << cudaGetErrorString(run_err) << std::endl;
    }
    if (sync_err != cudaSuccess) {
        std::cerr << "  [sync err " << B << "/" << CHUNK << "]: "
                  << cudaGetErrorString(sync_err) << std::endl;
    }

    std::vector<uint64_t> flat((size_t)B * KAPPA * D);
    cudaMemcpy(flat.data(), d_out, flat.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_key); cudaFree(d_z); cudaFree(d_out);

    std::vector<Ring<64>> out((size_t)B * KAPPA);
    for (size_t i = 0; i < out.size(); i++) {
        for (int r = 0; r < 64; r++) out[i][r] = flat[i * 64 + r];
    }
    return out;
}

template <int CHUNK>
static std::vector<Ring<64>> gpu_sparse(
    const uint32_t key[8],
    const std::vector<uint64_t>& positions
) {
    uint32_t* d_key;
    uint64_t* d_pos;
    uint64_t* d_out;
    cudaMalloc(&d_key, 8 * sizeof(uint32_t));
    cudaMalloc(&d_pos, positions.size() * sizeof(uint64_t));
    cudaMalloc(&d_out, KAPPA * D * sizeof(uint64_t));
    cudaMemcpy(d_key, key, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, positions.data(), positions.size() * sizeof(uint64_t),
               cudaMemcpyHostToDevice);

    commit_sparse_run<CHUNK>(d_key, d_pos, positions.size(), d_out);
    cudaDeviceSynchronize();

    std::vector<uint64_t> flat(KAPPA * D);
    cudaMemcpy(flat.data(), d_out, flat.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_key); cudaFree(d_pos); cudaFree(d_out);

    std::vector<Ring<64>> out(KAPPA);
    for (int i = 0; i < KAPPA; i++) {
        for (int r = 0; r < 64; r++) out[i][r] = flat[(size_t)i * 64 + r];
    }
    return out;
}

static bool test_gpu_single_vs_cpu() {
    std::cout << "[gpu_single_vs_cpu]\n";
    bool ok = true;
    for (uint64_t N : {(uint64_t)64, (uint64_t)256, (uint64_t)1024, (uint64_t)4096}) {
        std::cout << "  N=" << N << ": ";
        std::mt19937_64 rng(0x1234ULL + N);
        uint32_t key[8]; make_key(rng, key);
        auto z = make_z_bits(rng, N, 64);

        auto cpu_out = ajtai_cpu::cpu_ajtai_commit<64, 15>(key, z.data(), N);
        auto gpu_out = gpu_dense_batched<1, 1024>(key, z, N);

        bool match = true;
        for (int i = 0; i < KAPPA; i++) {
            if (!rings_equal<64>(cpu_out[i], gpu_out[i])) {
                std::cerr << "    row " << i << " mismatch\n";
                for (int r = 0; r < 64 && r < 4; r++) {
                    std::cerr << "      r=" << r
                              << " cpu=" << ajtai_cpu::h_canon(cpu_out[i][r])
                              << " gpu=" << ajtai_cpu::h_canon(gpu_out[i][r]) << "\n";
                }
                match = false;
                break;
            }
        }
        EXPECT(match, "single commit GPU vs CPU");
        std::cout << "PASS" << std::endl;
    }
    return ok;
}

template <int B>
static bool test_gpu_batched_at_B(uint64_t N) {
    std::cout << "  B=" << B << " N=" << N << ": ";
    std::mt19937_64 rng(0xBEEFULL + (uint64_t)B * 100 + N);
    uint32_t key[8]; make_key(rng, key);

    // Build B independent witnesses, pack as [b][j]
    std::vector<uint64_t> z_packed((size_t)B * N);
    for (int b = 0; b < B; b++) {
        for (uint64_t j = 0; j < N; j++) {
            z_packed[(size_t)b * N + j] = rng();
        }
    }

    auto cpu_out = ajtai_cpu::cpu_ajtai_commit_batched<64, 15>(key, z_packed.data(), N, B);
    auto gpu_out = gpu_dense_batched<B, 512>(key, z_packed, N);

    EXPECT(cpu_out.size() == gpu_out.size(), "size mismatch");
    bool match = true;
    for (size_t i = 0; i < cpu_out.size(); i++) {
        if (!rings_equal<64>(cpu_out[i], gpu_out[i])) {
            int b = (int)(i / KAPPA);
            int row = (int)(i % KAPPA);
            std::cerr << "    b=" << b << " row=" << row << " mismatch\n";
            match = false;
            break;
        }
    }
    EXPECT(match, "batched commit GPU vs CPU");
    std::cout << "PASS" << std::endl;
    return true;
}

static bool test_gpu_batched_vs_cpu() {
    std::cout << "[gpu_batched_vs_cpu]\n";
    bool ok = true;
    ok &= test_gpu_batched_at_B<2>(256);
    ok &= test_gpu_batched_at_B<4>(256);
    ok &= test_gpu_batched_at_B<8>(256);
    ok &= test_gpu_batched_at_B<16>(256);
    ok &= test_gpu_batched_at_B<4>(1024);
    return ok;
}

static bool test_chunk_invariance() {
    std::cout << "[chunk_invariance] ";
    uint64_t N = 2048;
    std::mt19937_64 rng(0xCC00CCULL);
    uint32_t key[8]; make_key(rng, key);
    auto z = make_z_bits(rng, N);

    auto out_256  = gpu_dense_batched<1, 256 >(key, z, N);
    auto out_1024 = gpu_dense_batched<1, 1024>(key, z, N);
    auto out_4096 = gpu_dense_batched<1, 4096>(key, z, N);

    for (int i = 0; i < KAPPA; i++) {
        EXPECT(rings_equal<64>(out_256[i], out_1024[i]), "256 vs 1024");
        EXPECT(rings_equal<64>(out_256[i], out_4096[i]), "256 vs 4096");
    }
    std::cout << "PASS" << std::endl;
    return true;
}

static bool test_zero_witness() {
    std::cout << "[zero_witness] ";
    uint64_t N = 256;
    std::mt19937_64 rng(0xDEAD7E57ULL);
    uint32_t key[8]; make_key(rng, key);
    std::vector<uint64_t> z(N, 0ULL);

    auto gpu_out = gpu_dense_batched<1, 1024>(key, z, N);
    for (int i = 0; i < KAPPA; i++) {
        for (int r = 0; r < 64; r++) {
            EXPECT(ajtai_cpu::h_canon(gpu_out[i][r]) == 0,
                   "zero witness should give zero commitment");
        }
    }
    std::cout << "PASS" << std::endl;
    return true;
}

// ============================================================================
// Stage 6: sparse vs dense on same z
// ============================================================================

static bool test_sparse_vs_dense() {
    std::cout << "[sparse_vs_dense] ";
    uint64_t N = 256;
    std::mt19937_64 rng(0x5DA75EULL);
    uint32_t key[8]; make_key(rng, key);

    // Sparse witness: ~10 set bits total across N blocks
    std::vector<uint64_t> z(N, 0ULL);
    std::vector<uint64_t> positions;
    for (int t = 0; t < 32; t++) {
        uint64_t p = rng() % (N * 64);
        positions.push_back(p);
        uint64_t j = p >> 6;
        int      ell = (int)(p & 63);
        z[j] |= (1ULL << ell);
    }
    // Deduplicate the bitmask (set bits at colliding positions don't change z;
    // but our "positions" list may contain duplicates which would double-count
    // in the sparse path). Manual insertion sort + dedup since <algorithm>
    // breaks under nvcc 11.5 + gcc 11.
    for (size_t i = 1; i < positions.size(); i++) {
        uint64_t v = positions[i];
        size_t j = i;
        while (j > 0 && positions[j - 1] > v) { positions[j] = positions[j - 1]; j--; }
        positions[j] = v;
    }
    size_t w = 0;
    for (size_t i = 0; i < positions.size(); i++) {
        if (i == 0 || positions[i] != positions[i - 1]) positions[w++] = positions[i];
    }
    positions.resize(w);

    auto dense_out  = gpu_dense_batched<1, 256>(key, z, N);
    auto sparse_out = gpu_sparse<256>(key, positions);

    for (int i = 0; i < KAPPA; i++) {
        if (!rings_equal<64>(dense_out[i], sparse_out[i])) {
            std::cerr << "  row " << i << " mismatch\n";
            return false;
        }
    }
    std::cout << "PASS (" << positions.size() << " set bits)" << std::endl;
    return true;
}

// ============================================================================
// Benchmark (informational, not pass/fail)
// ============================================================================

static void benchmark_dense(uint64_t N, int B_label = 1) {
    cudaDeviceSynchronize();
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    std::mt19937_64 rng(123);
    uint32_t key[8]; make_key(rng, key);
    std::vector<uint64_t> z((size_t)B_label * N);
    for (auto& v : z) v = rng();  // random dense binary

    uint32_t* d_key;
    uint64_t* d_z;
    uint64_t* d_out;
    cudaMalloc(&d_key, 8 * sizeof(uint32_t));
    cudaMalloc(&d_z, z.size() * sizeof(uint64_t));
    cudaMalloc(&d_out, (size_t)B_label * KAPPA * D * sizeof(uint64_t));
    cudaMemcpy(d_key, key, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z.data(), z.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    auto run = [&]() {
        if      (B_label == 1) commit_dense_batched_run<1,  4096>(d_key, d_z, N, d_out);
        else if (B_label == 4) commit_dense_batched_run<4,  4096>(d_key, d_z, N, d_out);
        else if (B_label == 8) commit_dense_batched_run<8,  4096>(d_key, d_z, N, d_out);
        else if (B_label == 16) commit_dense_batched_run<16, 4096>(d_key, d_z, N, d_out);
    };

    // warmup
    run();
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    run();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f; cudaEventElapsedTime(&ms, s, e);

    std::cout << "  N=" << N << " B=" << B_label << "  " << ms << " ms"
              << "  ( per-commit ≈ " << (ms / (float)B_label) << " ms )"
              << std::endl;

    cudaFree(d_key); cudaFree(d_z); cudaFree(d_out);
    cudaEventDestroy(s); cudaEventDestroy(e);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    std::cout << "GPU: " << p.name << "  sm_" << p.major << p.minor << "\n";
    std::cout << "=== Ajtai commit tests ===" << std::endl;

    bool ok = true;
    ok &= test_cpu_reference();
    ok &= test_chacha8_determinism();
    ok &= test_prg_mean_sanity();
    ok &= test_gpu_single_vs_cpu();
    ok &= test_gpu_batched_vs_cpu();
    ok &= test_chunk_invariance();
    ok &= test_zero_witness();
    ok &= test_sparse_vs_dense();

    std::cout << "\n=== Benchmark (optional) ===" << std::endl;
    benchmark_dense(1 << 14, 1);
    benchmark_dense(1 << 14, 4);
    benchmark_dense(1 << 14, 8);
    benchmark_dense(1 << 14, 16);
    benchmark_dense(1 << 18, 1);
    benchmark_dense(1 << 18, 8);

    std::cout << (ok ? "\nALL PASS" : "\nFAILURES PRESENT") << std::endl;
    return ok ? 0 : 1;
}

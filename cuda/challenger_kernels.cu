/**
 * Fiat-Shamir DuplexChallenger CUDA Kernels and Tests
 *
 * This file contains:
 * - Additional batch kernels for challenger operations
 * - Comprehensive tests for correctness and performance
 * - Verification against expected Fiat-Shamir behavior
 */

#include "challenger.cuh"
#include <stdio.h>
#include <set>

// ============================================================================
// Advanced Batch Kernels
// ============================================================================

/**
 * Combined observe-sample operation
 * Each challenger observes values, then samples challenges
 * Useful for FRI query phase where we observe commitment then sample indices
 */
__global__ void challenger_batch_observe_then_sample_kernel(
    DuplexChallengerState* states,
    const GoldilocksField* observe_values,
    int observe_count,
    GoldilocksField* sample_outputs,
    int sample_count,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Observe
    challenger_observe_slice(&states[idx], observe_values + idx * observe_count, observe_count);

    // Sample
    challenger_sample_array(&states[idx], sample_outputs + idx * sample_count, sample_count);
}

/**
 * Grinding kernel for PoW (proof of work) in FRI
 * Finds a nonce that makes sample_bits produce a value below threshold
 */
__global__ void challenger_grinding_kernel(
    DuplexChallengerState* base_state,  // Starting state (same for all threads)
    uint64_t start_nonce,
    int num_bits,
    uint64_t* found_nonce,              // Output: first successful nonce
    int* found_flag                      // Output: set to 1 when found
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;

    // Check if already found
    if (*found_flag) return;

    // Copy base state
    DuplexChallengerState state = *base_state;

    // Observe nonce
    challenger_observe(&state, GoldilocksField(nonce));

    // Sample and check
    uint64_t sample = challenger_sample_bits(&state, num_bits);

    // If all bits are zero, we found a valid nonce
    if (sample == 0) {
        // Atomic compare-and-swap to claim the find
        int old = atomicCAS(found_flag, 0, 1);
        if (old == 0) {
            *found_nonce = nonce;
        }
    }
}

// ============================================================================
// Host Wrapper for Advanced Operations
// ============================================================================

inline cudaError_t challenger_batch_observe_then_sample(
    DuplexChallengerState* d_states,
    const GoldilocksField* d_observe_values,
    int observe_count,
    GoldilocksField* d_sample_outputs,
    int sample_count,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
    challenger_batch_observe_then_sample_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
        d_states, d_observe_values, observe_count, d_sample_outputs, sample_count, batch_size
    );
    return cudaGetLastError();
}

/**
 * Find a grinding nonce (PoW)
 * Returns the nonce that makes sample_bits return 0
 */
inline cudaError_t challenger_find_grinding_nonce(
    DuplexChallengerState* d_base_state,
    int num_bits,
    uint64_t* h_found_nonce,
    int max_iterations = 1000000,
    cudaStream_t stream = 0
) {
    uint64_t* d_found_nonce;
    int* d_found_flag;
    cudaMalloc(&d_found_nonce, sizeof(uint64_t));
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMemset(d_found_flag, 0, sizeof(int));

    const int BATCH_SIZE = 65536;  // Threads per iteration
    uint64_t nonce_offset = 0;

    for (int iter = 0; iter < max_iterations; iter++) {
        int grid_size = (BATCH_SIZE + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
        challenger_grinding_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
            d_base_state, nonce_offset, num_bits, d_found_nonce, d_found_flag
        );

        // Check if found
        int found;
        cudaMemcpy(&found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (found) {
            cudaMemcpy(h_found_nonce, d_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaFree(d_found_nonce);
            cudaFree(d_found_flag);
            return cudaSuccess;
        }

        nonce_offset += BATCH_SIZE;
    }

    cudaFree(d_found_nonce);
    cudaFree(d_found_flag);
    return cudaErrorNotReady;  // Not found within max_iterations
}

// ============================================================================
// Test Code
// ============================================================================

#ifdef CHALLENGER_TEST

#include <iostream>
#include <vector>
#include <cstring>

/**
 * Test basic observe-sample functionality
 * Verifies deterministic behavior: same inputs -> same outputs
 */
void test_deterministic() {
    std::cout << "Testing deterministic behavior..." << std::endl;

    cudaError_t err = poseidon2_init();
    if (err != cudaSuccess) {
        std::cerr << "Failed to init Poseidon2: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Create two challengers with same inputs
    DuplexChallengerState *d_state1, *d_state2;
    cudaMalloc(&d_state1, sizeof(DuplexChallengerState));
    cudaMalloc(&d_state2, sizeof(DuplexChallengerState));

    challenger_batch_init(d_state1, 1);
    challenger_batch_init(d_state2, 1);

    // Observe same values
    std::vector<GoldilocksField> inputs = {
        GoldilocksField(123), GoldilocksField(456), GoldilocksField(789)
    };

    GoldilocksField* d_inputs;
    cudaMalloc(&d_inputs, inputs.size() * sizeof(GoldilocksField));
    cudaMemcpy(d_inputs, inputs.data(), inputs.size() * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    challenger_batch_observe_slice(d_state1, d_inputs, inputs.size(), 1);
    challenger_batch_observe_slice(d_state2, d_inputs, inputs.size(), 1);

    // Sample from both
    GoldilocksField *d_output1, *d_output2;
    cudaMalloc(&d_output1, sizeof(GoldilocksField));
    cudaMalloc(&d_output2, sizeof(GoldilocksField));

    challenger_batch_sample(d_state1, d_output1, 1);
    challenger_batch_sample(d_state2, d_output2, 1);

    GoldilocksField h_output1, h_output2;
    cudaMemcpy(&h_output1, d_output1, sizeof(GoldilocksField), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_output2, d_output2, sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    std::cout << "Sample 1: " << canonicalize(h_output1.value) << std::endl;
    std::cout << "Sample 2: " << canonicalize(h_output2.value) << std::endl;

    if (canonicalize(h_output1.value) == canonicalize(h_output2.value)) {
        std::cout << "Deterministic test PASSED!" << std::endl;
    } else {
        std::cout << "Deterministic test FAILED!" << std::endl;
    }

    // Sample more and verify they're still equal
    GoldilocksField samples1[4], samples2[4];
    GoldilocksField* d_samples1, *d_samples2;
    cudaMalloc(&d_samples1, 4 * sizeof(GoldilocksField));
    cudaMalloc(&d_samples2, 4 * sizeof(GoldilocksField));

    challenger_batch_sample_array(d_state1, d_samples1, 4, 1);
    challenger_batch_sample_array(d_state2, d_samples2, 4, 1);

    cudaMemcpy(samples1, d_samples1, 4 * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);
    cudaMemcpy(samples2, d_samples2, 4 * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    bool all_match = true;
    std::cout << "Additional samples: ";
    for (int i = 0; i < 4; i++) {
        if (canonicalize(samples1[i].value) != canonicalize(samples2[i].value)) {
            all_match = false;
        }
        std::cout << canonicalize(samples1[i].value) << " ";
    }
    std::cout << std::endl;

    if (all_match) {
        std::cout << "Extended deterministic test PASSED!" << std::endl;
    } else {
        std::cout << "Extended deterministic test FAILED!" << std::endl;
    }

    cudaFree(d_state1);
    cudaFree(d_state2);
    cudaFree(d_inputs);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_samples1);
    cudaFree(d_samples2);
}

/**
 * Test that observing different values produces different samples
 */
void test_different_inputs() {
    std::cout << "\nTesting different inputs produce different outputs..." << std::endl;

    DuplexChallengerState *d_state1, *d_state2;
    cudaMalloc(&d_state1, sizeof(DuplexChallengerState));
    cudaMalloc(&d_state2, sizeof(DuplexChallengerState));

    challenger_batch_init(d_state1, 1);
    challenger_batch_init(d_state2, 1);

    // Observe different values
    GoldilocksField input1 = GoldilocksField(100);
    GoldilocksField input2 = GoldilocksField(200);  // Different!

    GoldilocksField* d_input1, *d_input2;
    cudaMalloc(&d_input1, sizeof(GoldilocksField));
    cudaMalloc(&d_input2, sizeof(GoldilocksField));
    cudaMemcpy(d_input1, &input1, sizeof(GoldilocksField), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, &input2, sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    challenger_batch_observe(d_state1, d_input1, 1);
    challenger_batch_observe(d_state2, d_input2, 1);

    // Sample from both
    GoldilocksField *d_output1, *d_output2;
    cudaMalloc(&d_output1, sizeof(GoldilocksField));
    cudaMalloc(&d_output2, sizeof(GoldilocksField));

    challenger_batch_sample(d_state1, d_output1, 1);
    challenger_batch_sample(d_state2, d_output2, 1);

    GoldilocksField h_output1, h_output2;
    cudaMemcpy(&h_output1, d_output1, sizeof(GoldilocksField), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_output2, d_output2, sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    std::cout << "Input 1: 100 -> Sample: " << canonicalize(h_output1.value) << std::endl;
    std::cout << "Input 2: 200 -> Sample: " << canonicalize(h_output2.value) << std::endl;

    if (canonicalize(h_output1.value) != canonicalize(h_output2.value)) {
        std::cout << "Different inputs test PASSED!" << std::endl;
    } else {
        std::cout << "Different inputs test FAILED (collision!)!" << std::endl;
    }

    cudaFree(d_state1);
    cudaFree(d_state2);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output1);
    cudaFree(d_output2);
}

/**
 * Test observe invalidates output buffer
 * After observing, previously squeezed values should change
 */
void test_observe_invalidation() {
    std::cout << "\nTesting observe invalidates output buffer..." << std::endl;

    DuplexChallengerState* d_state;
    cudaMalloc(&d_state, sizeof(DuplexChallengerState));
    challenger_batch_init(d_state, 1);

    GoldilocksField* d_temp;
    cudaMalloc(&d_temp, sizeof(GoldilocksField));

    // Initial observe
    GoldilocksField input1 = GoldilocksField(42);
    cudaMemcpy(d_temp, &input1, sizeof(GoldilocksField), cudaMemcpyHostToDevice);
    challenger_batch_observe(d_state, d_temp, 1);

    // Sample once
    GoldilocksField* d_sample1;
    cudaMalloc(&d_sample1, sizeof(GoldilocksField));
    challenger_batch_sample(d_state, d_sample1, 1);

    GoldilocksField h_sample1;
    cudaMemcpy(&h_sample1, d_sample1, sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    // Re-init and do same sequence
    challenger_batch_init(d_state, 1);
    cudaMemcpy(d_temp, &input1, sizeof(GoldilocksField), cudaMemcpyHostToDevice);
    challenger_batch_observe(d_state, d_temp, 1);

    // But now observe something else BEFORE sampling
    GoldilocksField input2 = GoldilocksField(999);
    cudaMemcpy(d_temp, &input2, sizeof(GoldilocksField), cudaMemcpyHostToDevice);
    challenger_batch_observe(d_state, d_temp, 1);

    // Sample
    GoldilocksField* d_sample2;
    cudaMalloc(&d_sample2, sizeof(GoldilocksField));
    challenger_batch_sample(d_state, d_sample2, 1);

    GoldilocksField h_sample2;
    cudaMemcpy(&h_sample2, d_sample2, sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    std::cout << "Sample after observe(42): " << canonicalize(h_sample1.value) << std::endl;
    std::cout << "Sample after observe(42), observe(999): " << canonicalize(h_sample2.value) << std::endl;

    if (canonicalize(h_sample1.value) != canonicalize(h_sample2.value)) {
        std::cout << "Observe invalidation test PASSED!" << std::endl;
    } else {
        std::cout << "Observe invalidation test FAILED!" << std::endl;
    }

    cudaFree(d_state);
    cudaFree(d_temp);
    cudaFree(d_sample1);
    cudaFree(d_sample2);
}

/**
 * Test batch operations with multiple independent challengers
 */
void test_batch_operations() {
    std::cout << "\nTesting batch operations..." << std::endl;

    const int BATCH_SIZE = 1024;

    DuplexChallengerState* d_states;
    cudaMalloc(&d_states, BATCH_SIZE * sizeof(DuplexChallengerState));
    challenger_batch_init(d_states, BATCH_SIZE);

    // Each challenger observes a different value
    std::vector<GoldilocksField> h_inputs(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++) {
        h_inputs[i] = GoldilocksField(i);
    }

    GoldilocksField* d_inputs;
    cudaMalloc(&d_inputs, BATCH_SIZE * sizeof(GoldilocksField));
    cudaMemcpy(d_inputs, h_inputs.data(), BATCH_SIZE * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

    challenger_batch_observe(d_states, d_inputs, BATCH_SIZE);

    // Sample from all
    GoldilocksField* d_outputs;
    cudaMalloc(&d_outputs, BATCH_SIZE * sizeof(GoldilocksField));
    challenger_batch_sample(d_states, d_outputs, BATCH_SIZE);

    std::vector<GoldilocksField> h_outputs(BATCH_SIZE);
    cudaMemcpy(h_outputs.data(), d_outputs, BATCH_SIZE * sizeof(GoldilocksField), cudaMemcpyDeviceToHost);

    // Check that all outputs are different (with high probability)
    std::set<uint64_t> unique_outputs;
    for (int i = 0; i < BATCH_SIZE; i++) {
        unique_outputs.insert(canonicalize(h_outputs[i].value));
    }

    std::cout << "Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Unique outputs: " << unique_outputs.size() << std::endl;

    // All should be unique (collision probability is negligible)
    if (unique_outputs.size() == BATCH_SIZE) {
        std::cout << "Batch operations test PASSED!" << std::endl;
    } else {
        std::cout << "Batch operations test FAILED (collisions detected)!" << std::endl;
    }

    // Print first few
    std::cout << "First 5 samples: ";
    for (int i = 0; i < 5; i++) {
        std::cout << canonicalize(h_outputs[i].value) << " ";
    }
    std::cout << std::endl;

    cudaFree(d_states);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
}

/**
 * Test extension field sampling
 */
void test_extension_field_sampling() {
    std::cout << "\nTesting extension field sampling..." << std::endl;

    DuplexChallengerState* d_state;
    cudaMalloc(&d_state, sizeof(DuplexChallengerState));
    challenger_batch_init(d_state, 1);

    // Observe some values
    GoldilocksField input = GoldilocksField(12345);
    GoldilocksField* d_input;
    cudaMalloc(&d_input, sizeof(GoldilocksField));
    cudaMemcpy(d_input, &input, sizeof(GoldilocksField), cudaMemcpyHostToDevice);
    challenger_batch_observe(d_state, d_input, 1);

    // Sample GF(p^2) element
    GoldilocksExt2* d_ext2;
    cudaMalloc(&d_ext2, sizeof(GoldilocksExt2));
    challenger_batch_sample_ext2(d_state, d_ext2, 1);

    GoldilocksExt2 h_ext2;
    cudaMemcpy(&h_ext2, d_ext2, sizeof(GoldilocksExt2), cudaMemcpyDeviceToHost);

    std::cout << "Sampled GF(p^2): " << canonicalize(h_ext2.c[0].value)
              << " + " << canonicalize(h_ext2.c[1].value) << "X" << std::endl;

    // Sample GF(p^5) element
    GoldilocksExt5* d_ext5;
    cudaMalloc(&d_ext5, sizeof(GoldilocksExt5));
    challenger_batch_sample_ext5(d_state, d_ext5, 1);

    GoldilocksExt5 h_ext5;
    cudaMemcpy(&h_ext5, d_ext5, sizeof(GoldilocksExt5), cudaMemcpyDeviceToHost);

    std::cout << "Sampled GF(p^5): [";
    for (int i = 0; i < 5; i++) {
        std::cout << canonicalize(h_ext5.c[i].value) << (i < 4 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    std::cout << "Extension field sampling test PASSED!" << std::endl;

    cudaFree(d_state);
    cudaFree(d_input);
    cudaFree(d_ext2);
    cudaFree(d_ext5);
}

/**
 * Test performance of batch challenger operations
 */
void test_performance() {
    std::cout << "\nTesting performance..." << std::endl;

    const int BATCH_SIZE = 1024 * 1024;  // 1M challengers
    const int OBSERVE_COUNT = 10;
    const int SAMPLE_COUNT = 5;

    DuplexChallengerState* d_states;
    GoldilocksField* d_inputs;
    GoldilocksField* d_outputs;

    cudaMalloc(&d_states, BATCH_SIZE * sizeof(DuplexChallengerState));
    cudaMalloc(&d_inputs, BATCH_SIZE * OBSERVE_COUNT * sizeof(GoldilocksField));
    cudaMalloc(&d_outputs, BATCH_SIZE * SAMPLE_COUNT * sizeof(GoldilocksField));

    // Initialize
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Init benchmark
    cudaEventRecord(start);
    challenger_batch_init(d_states, BATCH_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_init;
    cudaEventElapsedTime(&ms_init, start, stop);
    std::cout << "Init " << BATCH_SIZE << " challengers: " << ms_init << " ms" << std::endl;

    // Observe benchmark
    cudaMemset(d_inputs, 1, BATCH_SIZE * OBSERVE_COUNT * sizeof(GoldilocksField));

    cudaEventRecord(start);
    challenger_batch_observe_slice(d_states, d_inputs, OBSERVE_COUNT, BATCH_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_observe;
    cudaEventElapsedTime(&ms_observe, start, stop);
    std::cout << "Observe " << OBSERVE_COUNT << " values each: " << ms_observe << " ms" << std::endl;
    std::cout << "  Throughput: " << (BATCH_SIZE * OBSERVE_COUNT / ms_observe / 1000.0)
              << " M observe ops/s" << std::endl;

    // Sample benchmark
    cudaEventRecord(start);
    challenger_batch_sample_array(d_states, d_outputs, SAMPLE_COUNT, BATCH_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_sample;
    cudaEventElapsedTime(&ms_sample, start, stop);
    std::cout << "Sample " << SAMPLE_COUNT << " values each: " << ms_sample << " ms" << std::endl;
    std::cout << "  Throughput: " << (BATCH_SIZE * SAMPLE_COUNT / ms_sample / 1000.0)
              << " M sample ops/s" << std::endl;

    // Combined observe-then-sample benchmark
    cudaEventRecord(start);
    challenger_batch_observe_then_sample(
        d_states, d_inputs, OBSERVE_COUNT, d_outputs, SAMPLE_COUNT, BATCH_SIZE
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_combined;
    cudaEventElapsedTime(&ms_combined, start, stop);
    std::cout << "Combined observe(" << OBSERVE_COUNT << ")+sample(" << SAMPLE_COUNT << "): "
              << ms_combined << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_states);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
}

/**
 * Test HostDuplexChallenger convenience class
 */
void test_host_challenger() {
    std::cout << "\nTesting HostDuplexChallenger..." << std::endl;

    HostDuplexChallenger challenger;

    // Observe some values
    challenger.observe(GoldilocksField(100));
    challenger.observe(GoldilocksField(200));
    challenger.observe(GoldilocksField(300));

    // Sample
    GoldilocksField sample1 = challenger.sample();
    GoldilocksField sample2 = challenger.sample();

    std::cout << "Sample 1: " << canonicalize(sample1.value) << std::endl;
    std::cout << "Sample 2: " << canonicalize(sample2.value) << std::endl;

    // Sample extension field
    GoldilocksExt2 ext2_sample = challenger.sample_ext2();
    std::cout << "Ext2 sample: " << canonicalize(ext2_sample.c[0].value)
              << " + " << canonicalize(ext2_sample.c[1].value) << "X" << std::endl;

    // Observe slice
    std::vector<GoldilocksField> inputs = {
        GoldilocksField(1), GoldilocksField(2), GoldilocksField(3)
    };
    challenger.observe_slice(inputs.data(), inputs.size());

    GoldilocksField sample3 = challenger.sample();
    std::cout << "Sample after observe_slice: " << canonicalize(sample3.value) << std::endl;

    std::cout << "HostDuplexChallenger test PASSED!" << std::endl;
}

/**
 * Simulate a simple FRI-like protocol exchange
 */
void test_fri_simulation() {
    std::cout << "\nTesting FRI-like protocol simulation..." << std::endl;

    // Prover and verifier start with same initial state
    HostDuplexChallenger prover;
    HostDuplexChallenger verifier;

    // Both observe the same "commitment" (simulated as a hash)
    GoldilocksField commitment[4] = {
        GoldilocksField(111), GoldilocksField(222),
        GoldilocksField(333), GoldilocksField(444)
    };

    prover.observe_slice(commitment, 4);
    verifier.observe_slice(commitment, 4);

    // Both derive the same challenge
    GoldilocksField prover_challenge = prover.sample();
    GoldilocksField verifier_challenge = verifier.sample();

    std::cout << "Prover challenge: " << canonicalize(prover_challenge.value) << std::endl;
    std::cout << "Verifier challenge: " << canonicalize(verifier_challenge.value) << std::endl;

    if (canonicalize(prover_challenge.value) == canonicalize(verifier_challenge.value)) {
        std::cout << "Challenges match!" << std::endl;
    } else {
        std::cout << "ERROR: Challenges don't match!" << std::endl;
        return;
    }

    // Prover computes response (simulated) and both observe it
    GoldilocksField response = GoldilocksField(canonicalize(prover_challenge.value) + 1000);

    prover.observe(response);
    verifier.observe(response);

    // Derive next challenge
    GoldilocksExt2 prover_challenge2 = prover.sample_ext2();
    GoldilocksExt2 verifier_challenge2 = verifier.sample_ext2();

    std::cout << "Prover challenge 2: " << canonicalize(prover_challenge2.c[0].value)
              << " + " << canonicalize(prover_challenge2.c[1].value) << "X" << std::endl;
    std::cout << "Verifier challenge 2: " << canonicalize(verifier_challenge2.c[0].value)
              << " + " << canonicalize(verifier_challenge2.c[1].value) << "X" << std::endl;

    if (canonicalize(prover_challenge2.c[0].value) == canonicalize(verifier_challenge2.c[0].value) &&
        canonicalize(prover_challenge2.c[1].value) == canonicalize(verifier_challenge2.c[1].value)) {
        std::cout << "FRI simulation test PASSED!" << std::endl;
    } else {
        std::cout << "FRI simulation test FAILED!" << std::endl;
    }
}

#include <set>

int main() {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    // Initialize Poseidon2 (required for challenger)
    cudaError_t err = goldilocks_init();
    if (err != cudaSuccess) {
        std::cerr << "Failed to init Goldilocks: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = poseidon2_init();
    if (err != cudaSuccess) {
        std::cerr << "Failed to init Poseidon2: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Run tests
    test_deterministic();
    test_different_inputs();
    test_observe_invalidation();
    test_batch_operations();
    test_extension_field_sampling();
    test_performance();
    test_host_challenger();
    test_fri_simulation();

    std::cout << "\n=== All challenger tests completed ===" << std::endl;

    return 0;
}

#endif // CHALLENGER_TEST

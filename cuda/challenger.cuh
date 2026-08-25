/**
 * Fiat-Shamir DuplexChallenger for Goldilocks Field - CUDA Implementation
 *
 * Based on the Plonky3 DuplexChallenger pattern.
 *
 * The DuplexChallenger implements a sponge-based Fiat-Shamir transform using
 * Poseidon2 as the underlying permutation. It provides:
 * - observe(): Absorb values into the sponge (for commitments, public inputs)
 * - sample(): Squeeze pseudorandom challenges from the sponge
 *
 * Parameters (WIDTH=8, RATE=4):
 * - State: 8 field elements (4 rate + 4 capacity)
 * - Input/output per duplexing: 4 field elements
 * - Security: 128-bit (4 * 64-bit capacity)
 */

#ifndef CHALLENGER_CUH
#define CHALLENGER_CUH

#include "poseidon2.cuh"

// ============================================================================
// DuplexChallenger Configuration
// ============================================================================

#define CHALLENGER_WIDTH 8
#define CHALLENGER_RATE 4
#define CHALLENGER_CAPACITY 4  // WIDTH - RATE

// ============================================================================
// DuplexChallenger State Structure
// ============================================================================

/**
 * DuplexChallenger state for a single Fiat-Shamir transcript
 *
 * The sponge state is divided into:
 * - Rate portion (indices 0..RATE-1): Public, used for input/output
 * - Capacity portion (indices RATE..WIDTH-1): Secret, provides security
 *
 * Buffers are managed as circular queues with head/count tracking.
 */
struct DuplexChallengerState {
    GoldilocksField sponge_state[CHALLENGER_WIDTH];  // Sponge internal state
    GoldilocksField input_buffer[CHALLENGER_RATE];   // Buffered inputs to absorb
    GoldilocksField output_buffer[CHALLENGER_RATE];  // Buffered outputs to squeeze
    int input_count;   // Number of elements in input buffer
    int output_count;  // Number of elements in output buffer
};

// ============================================================================
// Device-side DuplexChallenger Functions
// ============================================================================

/**
 * Initialize a DuplexChallenger state to default values
 * All state elements and buffers are zeroed
 */
__device__ __forceinline__
void challenger_init(DuplexChallengerState* state) {
    #pragma unroll
    for (int i = 0; i < CHALLENGER_WIDTH; i++) {
        state->sponge_state[i] = GoldilocksField(0);
    }
    #pragma unroll
    for (int i = 0; i < CHALLENGER_RATE; i++) {
        state->input_buffer[i] = GoldilocksField(0);
        state->output_buffer[i] = GoldilocksField(0);
    }
    state->input_count = 0;
    state->output_count = 0;
}

/**
 * Perform duplexing operation:
 * 1. Overwrite rate portion of sponge state with input buffer
 * 2. Apply Poseidon2 permutation
 * 3. Extract rate portion to output buffer
 * 4. Clear input buffer
 */
__device__ __forceinline__
void challenger_duplexing(DuplexChallengerState* state) {
    // Overwrite rate portion with buffered inputs
    #pragma unroll
    for (int i = 0; i < state->input_count; i++) {
        state->sponge_state[i] = state->input_buffer[i];
    }
    // Zero remaining rate positions if input buffer not full
    for (int i = state->input_count; i < CHALLENGER_RATE; i++) {
        state->sponge_state[i] = GoldilocksField(0);
    }

    // Clear input buffer
    state->input_count = 0;

    // Apply permutation
    poseidon2_permute_8(state->sponge_state);

    // Extract rate portion to output buffer (in reverse for LIFO access)
    #pragma unroll
    for (int i = 0; i < CHALLENGER_RATE; i++) {
        state->output_buffer[CHALLENGER_RATE - 1 - i] = state->sponge_state[i];
    }
    state->output_count = CHALLENGER_RATE;
}

/**
 * Observe (absorb) a single field element into the transcript
 *
 * This is used to commit values like:
 * - Polynomial commitments
 * - Public inputs
 * - Protocol messages
 *
 * Observing invalidates any unused outputs (ensures fresh randomness after new data)
 */
__device__ __forceinline__
void challenger_observe(DuplexChallengerState* state, GoldilocksField value) {
    // Invalidate output buffer - any new observation requires fresh randomness
    state->output_count = 0;

    // Add to input buffer
    state->input_buffer[state->input_count] = value;
    state->input_count++;

    // If input buffer is full, perform duplexing
    if (state->input_count == CHALLENGER_RATE) {
        challenger_duplexing(state);
    }
}

/**
 * Observe multiple field elements
 */
__device__ __forceinline__
void challenger_observe_slice(DuplexChallengerState* state,
                              const GoldilocksField* values,
                              int count) {
    for (int i = 0; i < count; i++) {
        challenger_observe(state, values[i]);
    }
}

/**
 * Sample (squeeze) a single pseudorandom field element
 *
 * Returns a deterministic challenge derived from all previously observed values.
 * Used for generating:
 * - Random evaluation points
 * - Combination coefficients
 * - Query indices
 */
__device__ __forceinline__
GoldilocksField challenger_sample(DuplexChallengerState* state) {
    // If output buffer is empty, perform duplexing to generate new outputs
    if (state->output_count == 0) {
        challenger_duplexing(state);
    }

    // Pop from output buffer
    state->output_count--;
    return state->output_buffer[state->output_count];
}

/**
 * Sample multiple field elements into an array
 */
__device__ __forceinline__
void challenger_sample_array(DuplexChallengerState* state,
                             GoldilocksField* output,
                             int count) {
    for (int i = 0; i < count; i++) {
        output[i] = challenger_sample(state);
    }
}

/**
 * Sample random bits by taking the low bits of a field element
 *
 * Used for sampling indices, binary choices, etc.
 * Note: For security, bits should be from a fresh sample, not reused
 */
__device__ __forceinline__
uint64_t challenger_sample_bits(DuplexChallengerState* state, int num_bits) {
    GoldilocksField sample = challenger_sample(state);
    uint64_t mask = (1ULL << num_bits) - 1;
    return canonicalize(sample.value) & mask;
}

/**
 * Sample a value in range [0, max) using rejection sampling
 *
 * This ensures uniform distribution, unlike simple modulo
 */
__device__ __forceinline__
uint64_t challenger_sample_range(DuplexChallengerState* state, uint64_t max) {
    // Find number of bits needed
    int bits = 64;
    uint64_t temp = max - 1;
    while (bits > 0 && (temp >> (bits - 1)) == 0) {
        bits--;
    }
    if (bits == 0) bits = 1;

    // Rejection sampling
    uint64_t mask = (1ULL << bits) - 1;
    while (true) {
        GoldilocksField sample = challenger_sample(state);
        uint64_t value = canonicalize(sample.value) & mask;
        if (value < max) {
            return value;
        }
    }
}

// ============================================================================
// Extension Field Sampling
// ============================================================================

/**
 * Sample a GF(p²) element by sampling 2 base field elements
 */
__device__ __forceinline__
GoldilocksExt2 challenger_sample_ext2(DuplexChallengerState* state) {
    GoldilocksField c0 = challenger_sample(state);
    GoldilocksField c1 = challenger_sample(state);
    return GoldilocksExt2(c0, c1);
}

/**
 * Sample a GF(p⁵) element by sampling 5 base field elements
 */
__device__ __forceinline__
GoldilocksExt5 challenger_sample_ext5(DuplexChallengerState* state) {
    GoldilocksExt5 result;
    for (int i = 0; i < 5; i++) {
        result.c[i] = challenger_sample(state);
    }
    return result;
}

/**
 * Observe a GF(p²) element by observing its 2 coefficients
 */
__device__ __forceinline__
void challenger_observe_ext2(DuplexChallengerState* state, GoldilocksExt2 value) {
    challenger_observe(state, value.c[0]);
    challenger_observe(state, value.c[1]);
}

/**
 * Observe a GF(p⁵) element by observing its 5 coefficients
 */
__device__ __forceinline__
void challenger_observe_ext5(DuplexChallengerState* state, GoldilocksExt5 value) {
    for (int i = 0; i < 5; i++) {
        challenger_observe(state, value.c[i]);
    }
}

// ============================================================================
// Batch DuplexChallenger for Parallel Proving
// ============================================================================

/**
 * Batch challenger state array
 * Each element represents an independent Fiat-Shamir transcript
 */

/**
 * Initialize multiple challenger states
 */
__global__ void challenger_batch_init_kernel(
    DuplexChallengerState* states,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    challenger_init(&states[idx]);
}

/**
 * Batch observe: each challenger observes one value from its corresponding input
 */
__global__ void challenger_batch_observe_kernel(
    DuplexChallengerState* states,
    const GoldilocksField* values,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    challenger_observe(&states[idx], values[idx]);
}

/**
 * Batch observe slice: each challenger observes `count` values
 * values layout: values[batch_idx * count + i] for batch_idx in [0, batch_size)
 */
__global__ void challenger_batch_observe_slice_kernel(
    DuplexChallengerState* states,
    const GoldilocksField* values,
    int count,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    challenger_observe_slice(&states[idx], values + idx * count, count);
}

/**
 * Batch sample: each challenger samples one value
 */
__global__ void challenger_batch_sample_kernel(
    DuplexChallengerState* states,
    GoldilocksField* outputs,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    outputs[idx] = challenger_sample(&states[idx]);
}

/**
 * Batch sample array: each challenger samples `count` values
 */
__global__ void challenger_batch_sample_array_kernel(
    DuplexChallengerState* states,
    GoldilocksField* outputs,
    int count,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    challenger_sample_array(&states[idx], outputs + idx * count, count);
}

/**
 * Batch sample extension field element (GF(p²))
 */
__global__ void challenger_batch_sample_ext2_kernel(
    DuplexChallengerState* states,
    GoldilocksExt2* outputs,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    outputs[idx] = challenger_sample_ext2(&states[idx]);
}

/**
 * Batch sample extension field element (GF(p⁵))
 */
__global__ void challenger_batch_sample_ext5_kernel(
    DuplexChallengerState* states,
    GoldilocksExt5* outputs,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    outputs[idx] = challenger_sample_ext5(&states[idx]);
}

/**
 * Batch observe extension field elements (GF(p²))
 */
__global__ void challenger_batch_observe_ext2_kernel(
    DuplexChallengerState* states,
    const GoldilocksExt2* values,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    challenger_observe_ext2(&states[idx], values[idx]);
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

#define CHALLENGER_BLOCK_SIZE 256

inline cudaError_t challenger_batch_init(
    DuplexChallengerState* d_states,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
    challenger_batch_init_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
        d_states, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t challenger_batch_observe(
    DuplexChallengerState* d_states,
    const GoldilocksField* d_values,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
    challenger_batch_observe_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
        d_states, d_values, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t challenger_batch_observe_slice(
    DuplexChallengerState* d_states,
    const GoldilocksField* d_values,
    int count,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
    challenger_batch_observe_slice_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
        d_states, d_values, count, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t challenger_batch_sample(
    DuplexChallengerState* d_states,
    GoldilocksField* d_outputs,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
    challenger_batch_sample_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
        d_states, d_outputs, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t challenger_batch_sample_array(
    DuplexChallengerState* d_states,
    GoldilocksField* d_outputs,
    int count,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
    challenger_batch_sample_array_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
        d_states, d_outputs, count, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t challenger_batch_sample_ext2(
    DuplexChallengerState* d_states,
    GoldilocksExt2* d_outputs,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
    challenger_batch_sample_ext2_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
        d_states, d_outputs, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t challenger_batch_sample_ext5(
    DuplexChallengerState* d_states,
    GoldilocksExt5* d_outputs,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
    challenger_batch_sample_ext5_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
        d_states, d_outputs, batch_size
    );
    return cudaGetLastError();
}

inline cudaError_t challenger_batch_observe_ext2(
    DuplexChallengerState* d_states,
    const GoldilocksExt2* d_values,
    int batch_size,
    cudaStream_t stream = 0
) {
    int grid_size = (batch_size + CHALLENGER_BLOCK_SIZE - 1) / CHALLENGER_BLOCK_SIZE;
    challenger_batch_observe_ext2_kernel<<<grid_size, CHALLENGER_BLOCK_SIZE, 0, stream>>>(
        d_states, d_values, batch_size
    );
    return cudaGetLastError();
}

// ============================================================================
// Single Challenger Host API (for single-proof scenarios)
// ============================================================================

/**
 * Host-side DuplexChallenger for single-proof scenarios
 * Keeps state on device but provides synchronous host API
 */
class HostDuplexChallenger {
private:
    DuplexChallengerState* d_state;
    GoldilocksField* d_temp;  // Temporary buffer for single values
    cudaStream_t stream;

public:
    HostDuplexChallenger(cudaStream_t s = 0) : stream(s) {
        cudaMalloc(&d_state, sizeof(DuplexChallengerState));
        cudaMalloc(&d_temp, sizeof(GoldilocksField) * CHALLENGER_RATE);
        challenger_batch_init(d_state, 1, stream);
        cudaStreamSynchronize(stream);
    }

    ~HostDuplexChallenger() {
        cudaFree(d_state);
        cudaFree(d_temp);
    }

    void observe(GoldilocksField value) {
        cudaMemcpyAsync(d_temp, &value, sizeof(GoldilocksField),
                        cudaMemcpyHostToDevice, stream);
        challenger_batch_observe(d_state, d_temp, 1, stream);
        cudaStreamSynchronize(stream);
    }

    void observe_slice(const GoldilocksField* values, int count) {
        GoldilocksField* d_values;
        cudaMalloc(&d_values, count * sizeof(GoldilocksField));
        cudaMemcpyAsync(d_values, values, count * sizeof(GoldilocksField),
                        cudaMemcpyHostToDevice, stream);
        challenger_batch_observe_slice(d_state, d_values, count, 1, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_values);
    }

    GoldilocksField sample() {
        challenger_batch_sample(d_state, d_temp, 1, stream);
        GoldilocksField result;
        cudaMemcpyAsync(&result, d_temp, sizeof(GoldilocksField),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return result;
    }

    void sample_array(GoldilocksField* output, int count) {
        GoldilocksField* d_output;
        cudaMalloc(&d_output, count * sizeof(GoldilocksField));
        challenger_batch_sample_array(d_state, d_output, count, 1, stream);
        cudaMemcpyAsync(output, d_output, count * sizeof(GoldilocksField),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_output);
    }

    GoldilocksExt2 sample_ext2() {
        GoldilocksExt2* d_output;
        cudaMalloc(&d_output, sizeof(GoldilocksExt2));
        challenger_batch_sample_ext2(d_state, d_output, 1, stream);
        GoldilocksExt2 result;
        cudaMemcpyAsync(&result, d_output, sizeof(GoldilocksExt2),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_output);
        return result;
    }

    GoldilocksExt5 sample_ext5() {
        GoldilocksExt5* d_output;
        cudaMalloc(&d_output, sizeof(GoldilocksExt5));
        challenger_batch_sample_ext5(d_state, d_output, 1, stream);
        GoldilocksExt5 result;
        cudaMemcpyAsync(&result, d_output, sizeof(GoldilocksExt5),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_output);
        return result;
    }

    // Get raw device state pointer (for integration with other GPU operations)
    DuplexChallengerState* get_device_state() { return d_state; }
};

#endif // CHALLENGER_CUH

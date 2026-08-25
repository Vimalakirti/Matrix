/**
 * C FFI wrapper for the almost-Goldilocks CUDA kernels.
 *
 * Exposes:
 *   - init + memory management
 *   - field batch ops (add/sub/mul/neg/square/double/inverse/exp/scalar-mul/div)
 *   - Ext2 batch ops (add/sub/mul/neg/square/inverse/frobenius/conjugate/exp/scalar-mul)
 *   - Ext2 scale-accumulate (acc[i] += scalar * src[i])
 *   - GL <-> Ext2 conversion batches
 *   - eq_lagrange DP (base + Ext2)
 *   - partial_eval (base in-place + base→Ext2 ping-pong)
 *   - fused permute + partial-eval
 *   - sumcheck round-message + fold (base + Ext2)
 *   - bit_permute (gather-based variable reordering, field-agnostic shuffle)
 */

#include "almost_goldilocks.cuh"
#include "almost_extension.cuh"
#include "almost_eq_lagrange.cuh"
#include "almost_partial_eval.cuh"
#include "almost_fused_permute_peval.cuh"
#include "almost_sumcheck_prover.cuh"
#include "link_sumcheck.cuh"
#include "ajtai.cuh"
#include "ajtai_tc.cuh"
#include "ajtai_premat.cuh"
#include "ajtai_tc_commit_probe.cuh"
// We also pull in the field/extension kernel TUs so the kernels are linked
// into this object file rather than left as external declarations.
#include "almost_goldilocks_kernels.cu"
#include "almost_extension_kernels.cu"

#include <cstdio>

#define WRAPPER_BLOCK_SIZE 256

extern "C" {

// ============================================================================
// Init / memory management
// ============================================================================

int almost_goldilocks_cuda_init() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return -1;
    if (cudaSetDevice(0) != cudaSuccess) return -1;
    return (almost_goldilocks_init() == cudaSuccess) ? 0 : -1;
}

int cuda_malloc(void** ptr, size_t size) {
    return (cudaMalloc(ptr, size) == cudaSuccess) ? 0 : -1;
}
int cuda_free(void* ptr) {
    return (cudaFree(ptr) == cudaSuccess) ? 0 : -1;
}
int cuda_memcpy_htod(void* dst, const void* src, size_t size) {
    return (cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess) ? 0 : -1;
}
int cuda_memcpy_dtoh(void* dst, const void* src, size_t size) {
    // Return the actual cudaError_t rather than -1. cudaSuccess is 0, so the
    // "!= 0 means failure" contract is unchanged, but the caller can now say
    // WHICH error -- and CUDA errors are sticky, so a failure reported here is
    // often a kernel launch that failed earlier and had nowhere to surface.
    return (int)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
const char* agl_cuda_error_string(int code) {
    return cudaGetErrorString((cudaError_t)code);
}
int cuda_memcpy_dtod(void* dst, const void* src, size_t size) {
    return (cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) == cudaSuccess) ? 0 : -1;
}
// Cross-device copy. cudaMemcpyPeer works with or without peer access
// (it stages through the host when P2P is unavailable) — unlike a plain
// cudaMemcpy DtoD across devices, which faults on this driver.
int cuda_memcpy_peer(void* dst, int dst_dev, const void* src, int src_dev, size_t size) {
    return (cudaMemcpyPeer(dst, dst_dev, src, src_dev, size) == cudaSuccess) ? 0 : -1;
}
// Best-effort peer-access enable from the CURRENT device to `peer`.
// Already-enabled and not-supported both report success=0 semantics for
// the caller's purposes (copies still work via cudaMemcpyPeer).
int cuda_enable_peer_access(int peer) {
    cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); return 0; }
    if (err != cudaSuccess) { cudaGetLastError(); return -1; }
    return 0;
}
int cuda_memset(void* dst, int value, size_t size) {
    return (cudaMemset(dst, value, size) == cudaSuccess) ? 0 : -1;
}
/// Trim the default memory pool for the current device. Releases up
/// to `min_bytes_to_keep`-worth of pool-held cached blocks back to the
/// OS. Use 0 to release everything possible. Requires CUDA 11.2+ and
/// a driver supporting cudaMallocAsync (most A100s do).
int cuda_pool_trim(size_t min_bytes_to_keep) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return -1;
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) return -1;
    return (cudaMemPoolTrimTo(pool, min_bytes_to_keep) == cudaSuccess) ? 0 : -1;
}

// Stream-aware variants for the eq-builder pipeline.
int cuda_stream_create(void** stream) {
    cudaStream_t s;
    if (cudaStreamCreate(&s) != cudaSuccess) return -1;
    *stream = (void*)s;
    return 0;
}
int cuda_stream_destroy(void* stream) {
    return (cudaStreamDestroy((cudaStream_t)stream) == cudaSuccess) ? 0 : -1;
}
int cuda_stream_synchronize(void* stream) {
    return (cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess) ? 0 : -1;
}
int cuda_memcpy_dtod_async(void* dst, const void* src, size_t size, void* stream) {
    return (cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
            (cudaStream_t)stream) == cudaSuccess) ? 0 : -1;
}
int cuda_memcpy_htod_async(void* dst, const void* src, size_t size, void* stream) {
    return (cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
            (cudaStream_t)stream) == cudaSuccess) ? 0 : -1;
}
int cuda_device_synchronize() {
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}
int cuda_get_last_error() { return (int)cudaGetLastError(); }
int cuda_peek_at_last_error() { return (int)cudaPeekAtLastError(); }
int cuda_mem_get_info(size_t* free, size_t* total) {
    return (cudaMemGetInfo(free, total) == cudaSuccess) ? 0 : -1;
}
int cuda_set_device(int device) {
    return (cudaSetDevice(device) == cudaSuccess) ? 0 : -1;
}
int cuda_get_device(int* device) {
    return (cudaGetDevice(device) == cudaSuccess) ? 0 : -1;
}
int cuda_get_device_count(int* count) {
    return (cudaGetDeviceCount(count) == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Field batch ops (operate on flat uint64_t arrays — AlmostGoldilocksField is
// repr(transparent) over u64)
// ============================================================================

int agl_batch_add_ffi(const uint64_t* a, const uint64_t* b, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_batch_add_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, (const AlmostGoldilocksField*)b,
        (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_batch_sub_ffi(const uint64_t* a, const uint64_t* b, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_batch_sub_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, (const AlmostGoldilocksField*)b,
        (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_batch_mul_ffi(const uint64_t* a, const uint64_t* b, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_batch_mul_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, (const AlmostGoldilocksField*)b,
        (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_batch_neg_ffi(const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_batch_neg_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_batch_square_ffi(const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_batch_square_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__global__ void agl_batch_double_kernel(const AlmostGoldilocksField* a,
                                        AlmostGoldilocksField* r, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) r[idx] = agl_double(a[idx]);
}
int agl_batch_double_ffi(const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_batch_double_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_batch_inverse_ffi(const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_batch_inverse_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_batch_exp_ffi(const uint64_t* a, uint64_t exp, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_batch_exp_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, exp, (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_batch_mul_scalar_ffi(uint64_t scalar, const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_scalar_mul_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, AlmostGoldilocksField(scalar),
        (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__global__ void agl_batch_div_kernel(const AlmostGoldilocksField* a,
                                     const AlmostGoldilocksField* b,
                                     AlmostGoldilocksField* r, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) r[idx] = agl_mul(a[idx], agl_inverse(b[idx]));
}
int agl_batch_div_ffi(const uint64_t* a, const uint64_t* b, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_batch_div_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)a, (const AlmostGoldilocksField*)b,
        (AlmostGoldilocksField*)r, (size_t)n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Ext2 batch ops (operate on interleaved [c0, c1, ...] u64 arrays)
// ============================================================================

int aext2_batch_add_ffi(const uint64_t* a, const uint64_t* b, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_add_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)a, (const AlmostGoldilocksExt2*)b,
        (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_batch_sub_ffi(const uint64_t* a, const uint64_t* b, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_sub_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)a, (const AlmostGoldilocksExt2*)b,
        (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_batch_mul_ffi(const uint64_t* a, const uint64_t* b, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_mul_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)a, (const AlmostGoldilocksExt2*)b,
        (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_batch_inverse_ffi(const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_inverse_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)a, (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__global__ void aext2_batch_neg_kernel(const AlmostGoldilocksExt2* a,
                                       AlmostGoldilocksExt2* r, int n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) r[idx] = aext2_neg(a[idx]);
}
int aext2_batch_neg_ffi(const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_neg_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)a, (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_batch_square_ffi(const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_square_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)a, (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_batch_frobenius_ffi(const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_frobenius_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)a, (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__global__ void aext2_batch_conjugate_kernel(const AlmostGoldilocksExt2* a,
                                             AlmostGoldilocksExt2* r, int n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) r[idx] = aext2_conjugate(a[idx]);
}
int aext2_batch_conjugate_ffi(const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_conjugate_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)a, (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_batch_exp_ffi(const uint64_t* a, uint64_t exp, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_exp_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)a, exp, (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_batch_mul_scalar_ffi(uint64_t scalar, const uint64_t* a, uint64_t* r, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_batch_scalar_mul_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        AlmostGoldilocksField(scalar), (const AlmostGoldilocksExt2*)a,
        (AlmostGoldilocksExt2*)r, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// acc[i] += scalar * src[i]  (Ext2 scalar, Ext2 src/acc)
__global__ void aext2_scale_accumulate_kernel(uint64_t scalar_c0, uint64_t scalar_c1,
                                              const uint64_t* src, uint64_t* acc, int n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    AlmostGoldilocksExt2 s(scalar_c0, scalar_c1);
    AlmostGoldilocksExt2 v(src[2*idx], src[2*idx+1]);
    AlmostGoldilocksExt2 a(acc[2*idx], acc[2*idx+1]);
    AlmostGoldilocksExt2 result = aext2_add(a, aext2_mul(s, v));
    acc[2*idx]   = result.c[0].value;
    acc[2*idx+1] = result.c[1].value;
}
int aext2_scale_accumulate_ffi(uint64_t scalar_c0, uint64_t scalar_c1,
                               const uint64_t* d_src, uint64_t* d_acc, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_scale_accumulate_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        scalar_c0, scalar_c1, d_src, d_acc, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_to_aext2_batch_ffi(const uint64_t* in, uint64_t* out, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_to_aext2_batch_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksField*)in, (AlmostGoldilocksExt2*)out, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_to_agl_batch_ffi(const uint64_t* in, uint64_t* out, int n) {
    int grid = (n + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    aext2_to_agl_batch_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        (const AlmostGoldilocksExt2*)in, (AlmostGoldilocksField*)out, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// eq_lagrange (DP)
// ============================================================================

int agl_eq_dp_all_ffi(const uint64_t* d_r, uint64_t* d_buf_a, uint64_t* d_buf_b,
                      int log_n, uint64_t** d_result) {
    AlmostGoldilocksField* result_ptr = nullptr;
    cudaError_t err = agl_eq_dp_all(
        (const AlmostGoldilocksField*)d_r,
        (AlmostGoldilocksField*)d_buf_a,
        (AlmostGoldilocksField*)d_buf_b,
        log_n,
        &result_ptr,
        0
    );
    if (err != cudaSuccess) return -1;
    *d_result = (uint64_t*)result_ptr;
    return 0;
}

int aext2_eq_dp_all_ffi(const uint64_t* d_r, uint64_t* d_buf_a, uint64_t* d_buf_b,
                        int log_n, uint64_t** d_result) {
    AlmostGoldilocksExt2* result_ptr = nullptr;
    cudaError_t err = aext2_eq_dp_all(
        (const AlmostGoldilocksExt2*)d_r,
        (AlmostGoldilocksExt2*)d_buf_a,
        (AlmostGoldilocksExt2*)d_buf_b,
        log_n,
        &result_ptr,
        0
    );
    if (err != cudaSuccess) return -1;
    *d_result = (uint64_t*)result_ptr;
    return 0;
}

int aext2_eq_dp_all_batched_ffi(
    const uint64_t* d_r_all, uint64_t* d_buf_a_all, uint64_t* d_buf_b_all,
    int log_n, int num_leaves, size_t leaf_stride,
    uint64_t** d_result, void* stream
) {
    AlmostGoldilocksExt2* result_ptr = nullptr;
    cudaError_t err = aext2_eq_dp_all_batched(
        (const AlmostGoldilocksExt2*)d_r_all,
        (AlmostGoldilocksExt2*)d_buf_a_all,
        (AlmostGoldilocksExt2*)d_buf_b_all,
        log_n, num_leaves, leaf_stride,
        &result_ptr,
        (cudaStream_t)stream
    );
    if (err != cudaSuccess) return -1;
    *d_result = (uint64_t*)result_ptr;
    return 0;
}

int aext2_eq_dp_all_stream_ffi(const uint64_t* d_r, uint64_t* d_buf_a, uint64_t* d_buf_b,
                               int log_n, uint64_t** d_result, void* stream) {
    AlmostGoldilocksExt2* result_ptr = nullptr;
    cudaError_t err = aext2_eq_dp_all(
        (const AlmostGoldilocksExt2*)d_r,
        (AlmostGoldilocksExt2*)d_buf_a,
        (AlmostGoldilocksExt2*)d_buf_b,
        log_n,
        &result_ptr,
        (cudaStream_t)stream
    );
    if (err != cudaSuccess) return -1;
    *d_result = (uint64_t*)result_ptr;
    return 0;
}

// ============================================================================
// partial_eval
// ============================================================================

int agl_partial_eval_ffi(uint64_t* d_data, const uint64_t* d_r, int log_n, int m) {
    size_t n = 1ULL << log_n;
    // Allocate scratch internally
    AlmostGoldilocksField* d_scratch = nullptr;
    cudaError_t err = cudaMalloc(&d_scratch, (n / 2) * sizeof(AlmostGoldilocksField));
    if (err != cudaSuccess) return -1;
    err = agl_partial_eval(
        (AlmostGoldilocksField*)d_data,
        d_scratch,
        (const AlmostGoldilocksField*)d_r,
        log_n,
        m,
        0
    );
    cudaFree(d_scratch);
    return (err == cudaSuccess) ? 0 : -1;
}

int agl_partial_eval_ext2_from_base_ffi(const uint64_t* d_input, uint64_t* d_output,
                                        const uint64_t* d_r, int log_n, int m) {
    size_t n = 1ULL << log_n;
    AlmostGoldilocksExt2* d_scratch = nullptr;
    size_t scratch_size = (n / 4) > 0 ? (n / 4) : 1;
    cudaError_t err = cudaMalloc(&d_scratch, scratch_size * sizeof(AlmostGoldilocksExt2));
    if (err != cudaSuccess) return -1;
    err = agl_partial_eval_ext2_from_base(
        (const AlmostGoldilocksField*)d_input,
        (AlmostGoldilocksExt2*)d_output,
        d_scratch,
        (const AlmostGoldilocksExt2*)d_r,
        log_n,
        m,
        0
    );
    cudaFree(d_scratch);
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// fused permute + partial eval
// ============================================================================

int agl_fused_permute_partial_eval_ffi(
    const uint64_t* d_evals,
    uint64_t* d_output,
    const uint64_t* d_eq_table,
    const uint32_t* d_lo_lut,
    const uint32_t* d_hi_lut,
    int n, int m, int half, int output_size,
    int smem_bytes
) {
    int block = 256;
    int grid = (output_size < 1024) ? output_size : 1024;
    // Default per-block dynamic shared memory is 48 KB. Large einsum operands
    // (e.g. Llama-2-7B's 4096 × 32000 logits head: n = 27, half = 13 →
    // lo_lut + hi_lut ≈ 96 KB; real Llama-3/GPT-J FFN at n = 26 → ≈ 64 KB)
    // need the opt-in via cudaFuncAttributeMaxDynamicSharedMemorySize.
    //
    // This MUST be set per-device: cudaFuncSetAttribute applies to the CURRENT
    // device only. A process-wide `static`-guarded one-shot left every GPU
    // except the first at the 48 KB default, so n>=26 einsums failed on 7/8
    // GPUs in a multi-GPU (NUM_PARTITIONS) run — surfacing as KernelFailed
    // ("OOM") on otherwise-idle H200s. Set it every launch (cheap) so whichever
    // device this call runs on has the raised cap. Request exactly what this
    // launch needs (Hopper allows up to ~227 KB; FUSED_MAX_N caps n at 28 →
    // <=128 KB). If the attribute can't be raised, the launch fails below and
    // the caller falls back to host.
    cudaFuncSetAttribute(
        (const void*)agl_fused_permute_partial_eval_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );
    agl_fused_permute_partial_eval_kernel<<<grid, block, (size_t)smem_bytes>>>(
        d_evals, d_output, d_eq_table, d_lo_lut, d_hi_lut,
        n, m, half, output_size
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Sumcheck (base + Ext2)
// ============================================================================

int agl_sumcheck_round_message_ffi(const uint64_t* d_polys, uint64_t* d_partial,
                                   int d, size_t original_size, size_t half,
                                   int num_blocks) {
    agl_sumcheck_round_message_kernel<<<num_blocks, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_partial, d, original_size, half
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_sumcheck_fold_ffi(const uint64_t* d_input, uint64_t* d_output, uint64_t challenge,
                          int d, size_t original_size, size_t half) {
    int grid = ((half + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (grid > 256) grid = 256;
    agl_sumcheck_fold_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_input, d_output, challenge, d, original_size, half
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_sumcheck_round_message_ffi(const uint64_t* d_polys, uint64_t* d_partial,
                                     int d, size_t original_size, size_t half,
                                     int num_blocks) {
    aext2_sumcheck_round_message_kernel<<<num_blocks, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_partial, d, original_size, half
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_sumcheck_fold_ffi(const uint64_t* d_input, uint64_t* d_output,
                            uint64_t challenge_c0, uint64_t challenge_c1,
                            int d, size_t original_size, size_t half) {
    int grid = ((half + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (grid > 256) grid = 256;
    aext2_sumcheck_fold_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_input, d_output, challenge_c0, challenge_c1, d, original_size, half
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Batched per-leaf same-point sumcheck: degree-2 with 2 polys (eq, f) per leaf.
int aext2_sumcheck_batched_round_message_ffi(
    const uint64_t* d_polys, uint64_t* d_partial,
    size_t original_size, size_t half, int num_leaves, int num_blocks_x
) {
    dim3 grid(num_blocks_x, num_leaves);
    aext2_sumcheck_batched_round_message_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_partial, original_size, half, num_leaves
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_batched_lift_ternary_single_ffi(
    const uint64_t* d_pos, const uint64_t* d_neg, uint64_t* d_polys,
    size_t original_size, int num_leaves, size_t packed_size_u64
) {
    int grid_x = ((original_size + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (grid_x > 256) grid_x = 256;
    dim3 grid(grid_x, num_leaves);
    aext2_batched_lift_ternary_single_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_pos, d_neg, d_polys, original_size, num_leaves, packed_size_u64
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_selective_add_batched_planes_ffi(
    const uint64_t* d_eq, const uint64_t* d_packed_planes,
    uint64_t* d_partial, size_t total, int n_planes,
    size_t packed_size_u64, int num_blocks_x
) {
    dim3 grid(num_blocks_x, n_planes);
    aext2_selective_add_batched_planes_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_eq, d_packed_planes, d_partial, total, n_planes, packed_size_u64
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_lift_binary_contig_ffi(
    const uint64_t* d_packed, uint64_t* d_f,
    size_t original_size, int num_leaves, size_t packed_size_u64
) {
    int grid_x = ((original_size + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (grid_x > 256) grid_x = 256;
    dim3 grid(grid_x, num_leaves);
    aext2_lift_binary_contig_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_packed, d_f, original_size, num_leaves, packed_size_u64
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_batched_lift_binary_ffi(
    const uint64_t* d_packed, uint64_t* d_polys,
    size_t original_size, int num_leaves, size_t packed_size_u64
) {
    int grid_x = ((original_size + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (grid_x > 256) grid_x = 256;
    dim3 grid(grid_x, num_leaves);
    aext2_batched_lift_binary_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_packed, d_polys, original_size, num_leaves, packed_size_u64
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_sumcheck_batched_fold_ffi(
    const uint64_t* d_input, uint64_t* d_output,
    uint64_t challenge_c0, uint64_t challenge_c1,
    size_t original_size, size_t half, int num_leaves
) {
    int grid_x = ((half + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (grid_x > 256) grid_x = 256;
    dim3 grid(grid_x, num_leaves);
    aext2_sumcheck_batched_fold_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_input, d_output, challenge_c0, challenge_c1, original_size, half, num_leaves
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Grid sizing: uncapped (was min(.., 256) — a 256-block cap undersubscribes
// large arities ~64x: at arity 24 it left 16M outputs to 65k threads, each
// serially looping 256 elements. Saturate the device instead; grid.x limit
// is 2^31-1 so no cap is needed at any realistic arity.
int aext2_build_fu_ffi(
    const uint64_t* d_packed, const uint64_t* d_alphas,
    const int* d_leaf_idx_sorted, const int* d_unique_offsets, uint64_t* d_Fu,
    size_t original_size, int num_unique, size_t packed_size_u64
) {
    int grid_x = ((original_size + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    dim3 grid(grid_x, num_unique);
    aext2_build_fu_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_packed, d_alphas, d_leaf_idx_sorted, d_unique_offsets, d_Fu,
        original_size, packed_size_u64
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_build_fu_ternary_ffi(
    const uint64_t* d_pos, const uint64_t* d_neg, const uint64_t* d_alphas,
    const int* d_leaf_idx_sorted, const int* d_unique_offsets, uint64_t* d_Fu,
    size_t original_size, int num_unique, size_t packed_size_u64
) {
    int grid_x = ((original_size + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    dim3 grid(grid_x, num_unique);
    aext2_build_fu_ternary_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_pos, d_neg, d_alphas, d_leaf_idx_sorted, d_unique_offsets, d_Fu,
        original_size, packed_size_u64
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_wide_to_ternary_ffi(
    const int16_t* d_wide, uint64_t* d_pos, uint64_t* d_neg, int* d_err,
    size_t n_ring, int k_chunks
) {
    if (k_chunks > 16) return -1;
    int grid_x = (int)((n_ring + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    aext2_wide_to_ternary_kernel<<<grid_x, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_wide, d_pos, d_neg, d_err, n_ring, k_chunks
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Build ALL eq suffix stages (eqsuf_n .. eqsuf_1) for num_unique points.
// eqsuf_t sits at element offset 2^{log_n - t} - 1 with 2^{log_n - t}
// elements inside each unique's eqsuf_stride_u64 region.
int aext2_eq_suffix_dp_ffi(
    const uint64_t* d_r_all, uint64_t* d_eqsuf,
    int log_n, int num_unique, size_t eqsuf_stride_u64
) {
    aext2_eq_suffix_init_kernel<<<num_unique, 1>>>(d_eqsuf, eqsuf_stride_u64);
    for (int t = log_n - 1; t >= 1; t--) {
        size_t in_size = 1ULL << (log_n - t - 1);
        size_t in_off  = in_size - 1;
        size_t out_off = (1ULL << (log_n - t)) - 1;
        int grid_x = (int)((in_size + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
        dim3 grid(grid_x, num_unique);
        aext2_eq_suffix_layer_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
            d_r_all, d_eqsuf, t, log_n, in_off, out_off, in_size, eqsuf_stride_u64
        );
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_sharedeq_factored_msg_ffi(
    const uint64_t* d_eqsuf, const uint64_t* d_fu, uint64_t* d_partial,
    size_t eqsuf_off_elems, size_t eqsuf_stride_u64, size_t poly_stride_u64,
    size_t half, int num_unique, int num_blocks_x
) {
    dim3 grid(num_blocks_x, num_unique);
    aext2_sharedeq_factored_msg_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_eqsuf, d_fu, d_partial, eqsuf_off_elems, eqsuf_stride_u64,
        poly_stride_u64, half, num_unique
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_sharedeq_msg_ffi(
    const uint64_t* d_eq, const uint64_t* d_f, const int* d_leaf_to_unique,
    uint64_t* d_partial, size_t original_size, size_t half, int num_leaves,
    int num_blocks_x
) {
    dim3 grid(num_blocks_x, num_leaves);
    aext2_sharedeq_msg_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_eq, d_f, d_leaf_to_unique, d_partial, original_size, half, num_leaves
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_fold_single_ffi(
    const uint64_t* d_in, uint64_t* d_out,
    uint64_t challenge_c0, uint64_t challenge_c1,
    size_t original_size, size_t half, int count
) {
    int grid_x = ((half + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    dim3 grid(grid_x, count);
    aext2_fold_single_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_in, d_out, challenge_c0, challenge_c1, original_size, half, count
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_sumcheck_batched_round0_binary_msg_ffi(
    const uint64_t* d_polys, const uint64_t* d_packed, uint64_t* d_partial,
    size_t original_size, size_t half, int num_leaves, int num_blocks_x,
    size_t packed_size_u64
) {
    dim3 grid(num_blocks_x, num_leaves);
    aext2_sumcheck_batched_round0_binary_msg_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_packed, d_partial, original_size, half, num_leaves, packed_size_u64
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aext2_sumcheck_batched_round0_binary_fold_ffi(
    const uint64_t* d_polys, const uint64_t* d_packed, uint64_t* d_output,
    uint64_t challenge_c0, uint64_t challenge_c1,
    size_t original_size, size_t half, int num_leaves, size_t packed_size_u64
) {
    int grid_x = ((half + AGL_SUMCHECK_BLOCK_SIZE - 1) / AGL_SUMCHECK_BLOCK_SIZE);
    if (grid_x > 256) grid_x = 256;
    dim3 grid(grid_x, num_leaves);
    aext2_sumcheck_batched_round0_binary_fold_kernel<<<grid, AGL_SUMCHECK_BLOCK_SIZE>>>(
        d_polys, d_packed, d_output, challenge_c0, challenge_c1,
        original_size, half, num_leaves, packed_size_u64
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Bit permute (field-agnostic gather)
// ============================================================================

__global__ void agl_bit_permute_kernel(
    const uint64_t* __restrict__ d_input,
    uint64_t* __restrict__ d_output,
    const int* __restrict__ d_inv_perm,
    int n_bits, int total
) {
    int new_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_idx >= total) return;
    int old_idx = 0;
    for (int bit = 0; bit < n_bits; bit++) {
        if (new_idx & (1 << bit)) {
            old_idx |= (1 << d_inv_perm[bit]);
        }
    }
    d_output[new_idx] = d_input[old_idx];
}
int agl_bit_permute_ffi(const uint64_t* d_input, uint64_t* d_output,
                        const int* d_perm_map, int n_bits, int total) {
    int grid = (total + WRAPPER_BLOCK_SIZE - 1) / WRAPPER_BLOCK_SIZE;
    agl_bit_permute_kernel<<<grid, WRAPPER_BLOCK_SIZE>>>(
        d_input, d_output, d_perm_map, n_bits, total
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Ajtai commitment FFI
// ============================================================================
//
// The kernel is templated on (B, CHUNK). Expose a small set of (B, CHUNK)
// instantiations and dispatch at the FFI boundary by integer values.
//
// Convention:
//   B in {1, 2, 4, 8, 16}
//   CHUNK in {64, 128, 256, 1024, 4096}
//     C64/C128 added for adaptive selection at small N — at log_n=18-22
//     they let us spread work across many more SMs vs C4096
//   chacha_key is 8 u32 little-endian on device
//   z_bits_packed is [B * N] u64 on device, row-major over batch
//   d_out is [B * KAPPA * D] u64 on device

// For B ≤ 8 use the halve-rows kernel (better occupancy via 2 threads per
// (b, r), each holding 8 u64 acc instead of 15). For B = 16 it doesn't fit
// (blockDim would be 2048 > 1024 hard limit), so we keep the standard kernel.
#define AJTAI_DISPATCH_HALVE(B_, CHUNK_)                                              \
    do {                                                                              \
        cudaError_t err = ajtai::commit_dense_batched_halve_run<B_, CHUNK_>(          \
            d_key, d_z, N, col_offset, d_out);                                        \
        return (err == cudaSuccess) ? 0 : -1;                                         \
    } while (0)
#define AJTAI_DISPATCH(B_, CHUNK_)                                                   \
    do {                                                                              \
        cudaError_t err = ajtai::commit_dense_batched_run<B_, CHUNK_>(                \
            d_key, d_z, N, col_offset, d_out);                                        \
        return (err == cudaSuccess) ? 0 : -1;                                         \
    } while (0)

#define AJTAI_DISPATCH_AUTO(B_, CHUNK_)                                              \
    do {                                                                              \
        cudaError_t err = ajtai::commit_dense_dispatch_run<B_, CHUNK_>(               \
            d_key, d_z, N, col_offset, d_out);                                        \
        return (err == cudaSuccess) ? 0 : -1;                                         \
    } while (0)

#define AJTAI_CHUNK_DISPATCH(CHUNK_)                                                  \
    switch (B) {                                                                       \
    case 1:  AJTAI_DISPATCH_AUTO(1,  CHUNK_);                                         \
    case 2:  AJTAI_DISPATCH_AUTO(2,  CHUNK_);                                         \
    case 4:  AJTAI_DISPATCH_AUTO(4,  CHUNK_);                                         \
    case 8:  AJTAI_DISPATCH_AUTO(8,  CHUNK_);                                         \
    case 16: AJTAI_DISPATCH(16, CHUNK_);                                              \
    default: return -2;                                                                \
    }

int ajtai_commit_dense_batched_at_ffi(
    const uint32_t* d_key, const uint64_t* d_z, uint64_t N, int B, int chunk,
    uint64_t col_offset, uint64_t* d_out);

int ajtai_commit_dense_batched_ffi(
    const uint32_t* d_key,
    const uint64_t* d_z,
    uint64_t        N,
    int             B,
    int             chunk,
    uint64_t*       d_out
) {
    const uint64_t col_offset = 0;
    return ajtai_commit_dense_batched_at_ffi(d_key, d_z, N, B, chunk, col_offset, d_out);
}

int ajtai_commit_dense_batched_at_ffi(
    const uint32_t* d_key,
    const uint64_t* d_z,
    uint64_t        N,
    int             B,
    int             chunk,
    uint64_t        col_offset,
    uint64_t*       d_out
) {
    switch (chunk) {
    case 64:   AJTAI_CHUNK_DISPATCH(64);
    case 128:  AJTAI_CHUNK_DISPATCH(128);
    case 256:  AJTAI_CHUNK_DISPATCH(256);
    case 1024: AJTAI_CHUNK_DISPATCH(1024);
    case 4096: AJTAI_CHUNK_DISPATCH(4096);
    default:
        return -3;
    }
}

#undef AJTAI_DISPATCH
#undef AJTAI_DISPATCH_HALVE
#undef AJTAI_CHUNK_DISPATCH


// --------------------------------------------------------------------------
// Wide commit (full-width field coefficients), with column offset.
// --------------------------------------------------------------------------

#define AJTAI_WIDE_DISPATCH(CHUNK_)                                              \
    do {                                                                          \
        cudaError_t err = ajtai::commit_wide_run<CHUNK_>(                         \
            d_key, d_z_wide, N, col_offset, d_out);                               \
        return (err == cudaSuccess) ? 0 : -1;                                     \
    } while (0)

extern "C"
int ajtai_commit_wide_ffi(
    const uint32_t* d_key,
    const uint64_t* d_z_wide,   // [N * 64] canonical field elements
    uint64_t        N,          // ring elements (coefficients / 64)
    uint64_t        col_offset, // column window start in M_max
    int             chunk,
    uint64_t*       d_out       // [KAPPA * 64]
) {
    switch (chunk) {
    case 64:   AJTAI_WIDE_DISPATCH(64);
    case 128:  AJTAI_WIDE_DISPATCH(128);
    case 256:  AJTAI_WIDE_DISPATCH(256);
    case 1024: AJTAI_WIDE_DISPATCH(1024);
    case 4096: AJTAI_WIDE_DISPATCH(4096);
    default:   return -3;
    }
}

#undef AJTAI_WIDE_DISPATCH


// --------------------------------------------------------------------------
// Link sumcheck (packed PCS)
// --------------------------------------------------------------------------

extern "C"
int link_round_message_ffi(
    const uint64_t* d_w,          // [n_commit][stride] Ext2 (2 u64 each)
    const uint64_t* d_omega,
    const uint64_t* d_eq_suffix,  // [half]
    const uint64_t* d_tags,       // [n_commit]
    uint64_t        stride,
    uint64_t        omega_stride,
    uint64_t        half,
    uint64_t        n_commit,
    int             first_round,
    uint64_t*       d_partial,    // scratch [n_commit * chunks * 7]
    uint64_t        chunks,
    uint64_t*       d_out         // [7] Ext2
) {
    using namespace link_sumcheck;
    dim3 grid1((unsigned)chunks, (unsigned)n_commit);
    link_round_kernel<<<grid1, BLOCK>>>(
        (const E2*)d_w, (const E2*)d_omega, (const E2*)d_eq_suffix,
        (const E2*)d_tags, stride, omega_stride, half, first_round, (E2*)d_partial);
    if (cudaGetLastError() != cudaSuccess) return -1;

    link_reduce_kernel<<<MSG_SLOTS, BLOCK>>>(
        (const E2*)d_partial, n_commit * chunks, (E2*)d_out);
    if (cudaGetLastError() != cudaSuccess) return -2;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -3;
}

extern "C"
int link_fold_ffi(
    uint64_t* d_w,
    uint64_t* d_omega,
    uint64_t  stride,
    uint64_t  omega_stride,
    uint64_t  half,
    uint64_t  n_commit,
    uint64_t  r_c0,
    uint64_t  r_c1
) {
    using namespace link_sumcheck;
    uint64_t total = half * n_commit;
    unsigned blocks = (unsigned)((total + BLOCK - 1) / BLOCK);
    if (blocks == 0) blocks = 1;
    if (blocks > 65535u) blocks = 65535u;
    link_fold_kernel<<<blocks, BLOCK>>>(
        (E2*)d_w, (E2*)d_omega, stride, omega_stride, half, n_commit, r_c0, r_c1);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -2;
}

extern "C"
int link_eq_expand_ffi(
    uint64_t* d_table,
    uint64_t  span,
    uint64_t  r_c0,
    uint64_t  r_c1
) {
    using namespace link_sumcheck;
    unsigned blocks = (unsigned)((span + BLOCK - 1) / BLOCK);
    if (blocks == 0) blocks = 1;
    if (blocks > 65535u) blocks = 65535u;
    link_eq_expand_kernel<<<blocks, BLOCK>>>((E2*)d_table, span, r_c0, r_c1);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -2;
}


extern "C"
int link_round0_bits_ffi(
    const uint64_t* d_bits, const uint64_t* d_omega, const uint64_t* d_eq,
    const uint64_t* d_tags, uint64_t stride, uint64_t half, uint64_t n_commit,
    uint64_t* d_partial, uint64_t chunks, uint64_t* d_out
) {
    using namespace link_sumcheck;
    dim3 grid1((unsigned)chunks, (unsigned)n_commit);
    link_round0_bits_kernel<<<grid1, BLOCK>>>(
        d_bits, (const E2*)d_omega, (const E2*)d_eq, (const E2*)d_tags,
        stride, half, (E2*)d_partial);
    if (cudaGetLastError() != cudaSuccess) return -1;
    link_reduce_kernel<<<MSG_SLOTS, BLOCK>>>(
        (const E2*)d_partial, n_commit * chunks, (E2*)d_out);
    if (cudaGetLastError() != cudaSuccess) return -2;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -3;
}

extern "C"
int link_round_interleaved_ffi(
    const uint64_t* d_bits, const uint64_t* d_w, const uint64_t* d_pts,
    const uint64_t* d_scale, const uint64_t* d_eq, const uint64_t* d_tags,
    const uint32_t* d_list, const uint64_t* d_list_off, const uint64_t* d_list_len,
    uint64_t w_stride, uint64_t bits_stride_words, uint64_t half, uint64_t n_commit,
    uint32_t block_mask, int block_bits, int leaf_arity, int round, int first_round,
    uint64_t* d_partial, uint64_t chunks, uint64_t* d_out
) {
    using namespace link_sumcheck;
    dim3 grid1((unsigned)chunks, (unsigned)n_commit);
    link_round_interleaved_kernel<<<grid1, BLOCK>>>(
        d_bits, (const E2*)d_w, (const E2*)d_pts, (const E2*)d_scale,
        (const E2*)d_eq, (const E2*)d_tags, d_list, d_list_off, d_list_len,
        w_stride, bits_stride_words, half, block_mask, block_bits, leaf_arity,
        round, first_round, (E2*)d_partial);
    if (cudaGetLastError() != cudaSuccess) return -1;
    link_reduce_kernel<<<MSG_SLOTS, BLOCK>>>(
        (const E2*)d_partial, n_commit * chunks, (E2*)d_out);
    if (cudaGetLastError() != cudaSuccess) return -2;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -3;
}

extern "C"
int link_fold_w_ffi(
    const uint64_t* d_bits, const uint64_t* d_w_in, uint64_t* d_w_out,
    uint64_t in_stride, uint64_t bits_stride_words, uint64_t out_stride,
    uint64_t half, uint64_t n_commit, int first_round, uint64_t r_c0, uint64_t r_c1
) {
    using namespace link_sumcheck;
    uint64_t total = half * n_commit;
    unsigned blocks = (unsigned)((total + BLOCK - 1) / BLOCK);
    if (blocks == 0) blocks = 1;
    if (blocks > 65535u) blocks = 65535u;
    link_fold_w_kernel<<<blocks, BLOCK>>>(
        d_bits, (const E2*)d_w_in, (E2*)d_w_out, in_stride, bits_stride_words,
        out_stride, half, n_commit, first_round, r_c0, r_c1);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -2;
}

extern "C"
int link_round0_bits_sparse_ffi(
    const uint64_t* d_bits, const uint64_t* d_omega, const uint64_t* d_eq,
    const uint64_t* d_tags, const uint32_t* d_list, const uint64_t* d_list_off,
    const uint64_t* d_list_len, uint64_t stride, uint64_t half, uint64_t n_commit,
    uint64_t* d_partial, uint64_t chunks, uint64_t* d_out
) {
    using namespace link_sumcheck;
    dim3 grid1((unsigned)chunks, (unsigned)n_commit);
    link_round0_bits_sparse_kernel<<<grid1, BLOCK>>>(
        d_bits, (const E2*)d_omega, (const E2*)d_eq, (const E2*)d_tags,
        d_list, d_list_off, d_list_len, stride, half, (E2*)d_partial);
    if (cudaGetLastError() != cudaSuccess) return -1;
    link_reduce_kernel<<<MSG_SLOTS, BLOCK>>>(
        (const E2*)d_partial, n_commit * chunks, (E2*)d_out);
    if (cudaGetLastError() != cudaSuccess) return -2;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -3;
}

extern "C"
int link_fold0_bits_ffi(
    const uint64_t* d_bits, uint64_t* d_w_out, uint64_t* d_omega,
    uint64_t stride, uint64_t half, uint64_t n_commit, uint64_t r_c0, uint64_t r_c1
) {
    using namespace link_sumcheck;
    uint64_t total = half * n_commit;
    unsigned blocks = (unsigned)((total + BLOCK - 1) / BLOCK);
    if (blocks == 0) blocks = 1;
    if (blocks > 65535u) blocks = 65535u;
    link_fold0_bits_kernel<<<blocks, BLOCK>>>(
        d_bits, (E2*)d_w_out, (E2*)d_omega, stride, half, n_commit, r_c0, r_c1);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -2;
}

extern "C"
int link_expand_bits_ffi(const uint64_t* d_bits, uint64_t* d_out, uint64_t n_words) {
    using namespace link_sumcheck;
    unsigned blocks = (unsigned)((n_words + BLOCK - 1) / BLOCK);
    if (blocks == 0) blocks = 1;
    if (blocks > 65535u) blocks = 65535u;
    link_expand_bits_kernel<<<blocks, BLOCK>>>(d_bits, (E2*)d_out, n_words);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -2;
}

extern "C"
int link_omega_expand_ffi(
    uint64_t*       d_omega,
    const uint64_t* d_bases,
    const uint64_t* d_rs,
    uint64_t        span,
    uint64_t        n_active
) {
    using namespace link_sumcheck;
    unsigned bx = (unsigned)((span + BLOCK - 1) / BLOCK);
    if (bx == 0) bx = 1;
    if (bx > 65535u) bx = 65535u;
    dim3 grid(bx, (unsigned)n_active);
    link_omega_expand_kernel<<<grid, BLOCK>>>(
        (E2*)d_omega, d_bases, d_rs, span);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -2;
}

extern "C"
int link_eq_gather_ffi(
    const uint64_t* d_alpha, uint64_t n_vars,
    const uint32_t* d_list, uint64_t* d_out, uint64_t total
) {
    using namespace link_sumcheck;
    unsigned blocks = (unsigned)((total + BLOCK - 1) / BLOCK);
    if (blocks == 0) blocks = 1;
    if (blocks > 65535u) blocks = 65535u;
    link_eq_gather_kernel<<<blocks, BLOCK>>>(d_alpha, n_vars, d_list, (E2*)d_out, total);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -2;
}

extern "C"
int link_round_message_sparse_ffi(
    const uint64_t* d_w, const uint64_t* d_omega, const uint64_t* d_eq_suffix,
    const uint64_t* d_tags, const uint32_t* d_list, const uint64_t* d_list_off,
    const uint64_t* d_list_len, uint64_t stride, uint64_t omega_stride,
    uint64_t half, uint64_t n_commit,
    int first_round, uint64_t* d_partial, uint64_t chunks, uint64_t* d_out
) {
    using namespace link_sumcheck;
    dim3 grid1((unsigned)chunks, (unsigned)n_commit);
    link_round_sparse_kernel<<<grid1, BLOCK>>>(
        (const E2*)d_w, (const E2*)d_omega, (const E2*)d_eq_suffix,
        (const E2*)d_tags, d_list, d_list_off, d_list_len,
        stride, omega_stride, half, first_round, (E2*)d_partial);
    if (cudaGetLastError() != cudaSuccess) return -1;
    link_reduce_kernel<<<MSG_SLOTS, BLOCK>>>(
        (const E2*)d_partial, n_commit * chunks, (E2*)d_out);
    if (cudaGetLastError() != cudaSuccess) return -2;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -3;
}

extern "C"
int link_fold_sparse_ffi(
    uint64_t* d_w, uint64_t* d_omega, const uint32_t* d_list,
    const uint64_t* d_list_off, const uint64_t* d_list_len,
    uint64_t stride, uint64_t half, uint64_t n_commit, uint64_t r_c0, uint64_t r_c1
) {
    using namespace link_sumcheck;
    dim3 grid(256u, (unsigned)n_commit);
    link_fold_sparse_kernel<<<grid, BLOCK>>>(
        (E2*)d_w, (E2*)d_omega, d_list, d_list_off, d_list_len,
        stride, half, n_commit, r_c0, r_c1);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -2;
}

// --------------------------------------------------------------------------
// Fold (additive homomorphism)
//
// Both functions take r as a pointer to 64 int8_t values (each in {-1, 0, 1, 2}).
// We copy r into a ChallengeR struct passed by value to the kernel.
// --------------------------------------------------------------------------

#define AJTAI_FOLD_WITNESS_DISPATCH(CHUNK_)                                          \
    do {                                                                              \
        cudaError_t err = ajtai::fold_witness_run<CHUNK_>(                            \
            d_z1, d_z2, r, N_ring, d_out);                                            \
        return (err == cudaSuccess) ? 0 : -1;                                         \
    } while (0)

int ajtai_fold_witness_ffi(
    const uint64_t* d_z1,
    const uint64_t* d_z2,
    const int8_t*   r_coeffs,    // 64 entries, each in {-1, 0, 1, 2}
    uint64_t        N_ring,
    int             chunk,
    uint64_t*       d_out        // [N_ring * 64]
) {
    ajtai::ChallengeR r;
    for (int k = 0; k < 64; k++) r.coeffs[k] = r_coeffs[k];

    switch (chunk) {
    case 64:   AJTAI_FOLD_WITNESS_DISPATCH(64);
    case 128:  AJTAI_FOLD_WITNESS_DISPATCH(128);
    case 256:  AJTAI_FOLD_WITNESS_DISPATCH(256);
    case 1024: AJTAI_FOLD_WITNESS_DISPATCH(1024);
    case 4096: AJTAI_FOLD_WITNESS_DISPATCH(4096);
    default:   return -3;
    }
}

#undef AJTAI_FOLD_WITNESS_DISPATCH

// Multi-fold (K + k binary instances). r_all is num_instances * 64 i8 device
// pointer; z_packed is num_instances * N_ring u64 device pointer; c_packed is
// num_instances * KAPPA * D u64 device pointer.

int ajtai_multifold_witness_ffi(
    const uint64_t* d_z_packed,
    const int8_t*   d_r_all,
    int16_t*        d_out,           // [N_ring * 64]
    int             num_instances,
    uint64_t        N_ring,
    uint64_t        chunk_size
) {
    cudaError_t err = ajtai::multifold_witness_run(
        d_z_packed, d_r_all, d_out, num_instances, N_ring, chunk_size
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Mixed-type multifold witness: K binary instances + T ternary chunks.
// r_all is [(K + T) * 64] int8 with r_all[0..64] = (1, 0, ..., 0) for the
// implicit-weight-1 binary[0]. d_out is [N_ring * 64] i16.
int ajtai_multifold_mixed_witness_ffi(
    const uint64_t* d_z_bin_packed,
    const uint64_t* d_pos_packed,
    const uint64_t* d_neg_packed,
    const int8_t*   d_r_all,
    int16_t*        d_out,
    int             num_binary,
    int             num_ternary,
    uint64_t        N_ring,
    uint64_t        chunk_size
) {
    cudaError_t err = ajtai::multifold_mixed_witness_run(
        d_z_bin_packed, d_pos_packed, d_neg_packed, d_r_all, d_out,
        num_binary, num_ternary, N_ring, chunk_size
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Fused tensor-core variant: skips the z_mat materialization.  The WMMA
// kernel unpacks z bitmasks on-the-fly into per-warp shared memory at
// each K iteration, then load_matrix_sync from shared.  Eliminates the
// 66 MB transient z_mat (at log_n=20) and the expand_z_kernel launch.
int ajtai_multifold_mixed_witness_tc_fused_ffi(
    const uint64_t* d_z_bin_packed,
    const uint64_t* d_pos_packed,
    const uint64_t* d_neg_packed,
    const int8_t*   d_r_all,
    int16_t*        d_out,
    int             num_binary,
    int             num_ternary,
    uint64_t        N_ring
) {
    int M_inst  = num_binary + num_ternary;
    int K_total = M_inst * 64;

    int8_t* d_R = nullptr;
    cudaError_t err = cudaMallocAsync(&d_R, (size_t)K_total * 64, 0);
    if (err != cudaSuccess) return -1;

    err = ajtai::build_R_run(d_r_all, d_R, M_inst);
    if (err != cudaSuccess) { cudaFreeAsync(d_R, 0); return -1; }

    err = ajtai::multifold_tc_fused_run(
        d_z_bin_packed, d_pos_packed, d_neg_packed,
        d_R, d_out, num_binary, num_ternary, N_ring
    );
    cudaFreeAsync(d_R, 0);
    return (err == cudaSuccess) ? 0 : -1;
}

// Tensor-core variant of the mixed multifold. The caller passes the same
// inputs as the scalar path; this routine internally builds the dense
// R[K, 64] (col-major) and z[N_ring, K] (row-major, padded to a multiple
// of 16 rows) and invokes the WMMA m16n16k16 INT8 kernel.
int ajtai_multifold_mixed_witness_tc_ffi(
    const uint64_t* d_z_bin_packed,
    const uint64_t* d_pos_packed,
    const uint64_t* d_neg_packed,
    const int8_t*   d_r_all,
    int16_t*        d_out,
    int             num_binary,
    int             num_ternary,
    uint64_t        N_ring
) {
    int      M_inst   = num_binary + num_ternary;
    int      K_total  = M_inst * 64;
    // Align to 64 to match multifold_tc_kernel's BLOCK_M (4 warps × 16 rows).
    uint64_t n_padded = (N_ring + 63) & ~(uint64_t)63;

    int8_t* d_R   = nullptr;
    int8_t* d_z   = nullptr;
    cudaError_t err;

    err = cudaMallocAsync(&d_R, (size_t)K_total * 64, 0);
    if (err != cudaSuccess) return -1;

    err = cudaMallocAsync(&d_z, (size_t)n_padded * (size_t)K_total, 0);
    if (err != cudaSuccess) { cudaFreeAsync(d_R, 0); return -1; }

    // Zero-pad the trailing rows of z so the WMMA kernel can safely read
    // a full 16-row tile at the end. (Padded rows contribute 0 and are
    // never written back to `output`.)
    if (n_padded > N_ring) {
        size_t pad_bytes = (size_t)(n_padded - N_ring) * (size_t)K_total;
        err = cudaMemsetAsync(d_z + (size_t)N_ring * K_total, 0, pad_bytes, 0);
        if (err != cudaSuccess) { cudaFreeAsync(d_R, 0); cudaFreeAsync(d_z, 0); return -1; }
    }

    err = ajtai::build_R_run(d_r_all, d_R, M_inst);
    if (err != cudaSuccess) { cudaFreeAsync(d_R, 0); cudaFreeAsync(d_z, 0); return -1; }

    err = ajtai::expand_z_run(
        d_z_bin_packed, d_pos_packed, d_neg_packed,
        d_z, num_binary, num_ternary, N_ring
    );
    if (err != cudaSuccess) { cudaFreeAsync(d_R, 0); cudaFreeAsync(d_z, 0); return -1; }

    err = ajtai::multifold_tc_run(d_z, d_R, d_out, M_inst, N_ring);

    cudaFreeAsync(d_z, 0);
    cudaFreeAsync(d_R, 0);
    return (err == cudaSuccess) ? 0 : -1;
}

// Split (decomposition): i16 wide witness → 13 ternary chunks as pos/neg pairs.
// d_z_wide is [N_ring * 64] i16 (signed). d_pos_chunks and d_neg_chunks each
// hold [SPLIT_K * N_ring] u64s (SPLIT_K = 13 in our build).
int ajtai_split_witness_ffi(
    const int16_t* d_z_wide,
    uint64_t*      d_pos_chunks,
    uint64_t*      d_neg_chunks,
    uint64_t       N_ring
) {
    cudaError_t err = ajtai::split_witness_run(
        d_z_wide, d_pos_chunks, d_neg_chunks, N_ring
    );
    return (err == cudaSuccess) ? 0 : -1;
}

// Ternary commit: shared-M batched commit over 13 ternary chunks.
//   pos / neg : device buffers of length [SPLIT_K * N_ring] (u64 bitmasks)
//   d_out     : device buffer of length [SPLIT_K * KAPPA * D]
// PRG cost is amortized — 13 commitments for the price of one matrix scan.
#define AJTAI_TERNARY_DISPATCH(CHUNK_)                                            \
    do {                                                                            \
        cudaError_t err = ajtai::commit_ternary_run<CHUNK_>(                       \
            d_key, d_pos, d_neg, N_ring, d_out);                                    \
        return (err == cudaSuccess) ? 0 : -1;                                       \
    } while (0)

int ajtai_commit_ternary_ffi(
    const uint32_t* d_key,
    const uint64_t* d_pos,
    const uint64_t* d_neg,
    uint64_t        N_ring,
    int             chunk,
    uint64_t*       d_out
) {
    switch (chunk) {
    case 64:   AJTAI_TERNARY_DISPATCH(64);
    case 128:  AJTAI_TERNARY_DISPATCH(128);
    case 256:  AJTAI_TERNARY_DISPATCH(256);
    case 1024: AJTAI_TERNARY_DISPATCH(1024);
    case 4096: AJTAI_TERNARY_DISPATCH(4096);
    default:   return -3;
    }
}
#undef AJTAI_TERNARY_DISPATCH

// Materialize the Ajtai matrix M into a [N * KAPPA * D] u64 device buffer.
// Runs ChaCha8 once, then the caller can re-use d_M across many commits.
int ajtai_materialize_m_ffi(
    const uint32_t* d_chacha_key,
    uint64_t*       d_M,
    uint64_t        N
) {
    cudaError_t err = ajtai::materialize_M_run(d_chacha_key, d_M, N);
    return (err == cudaSuccess) ? 0 : -1;
}

// Ternary commit against a pre-materialized M. Same semantics as
// ajtai_commit_ternary_ffi but reads M from global memory instead of
// regenerating it via ChaCha8 each call.
#define AJTAI_TERNARY_PREMAT_DISPATCH(CHUNK_)                                       \
    do {                                                                              \
        cudaError_t err = ajtai::commit_ternary_premat_run<CHUNK_>(                  \
            d_M, d_pos, d_neg, N_ring, d_out);                                        \
        return (err == cudaSuccess) ? 0 : -1;                                         \
    } while (0)

int ajtai_commit_ternary_premat_ffi(
    const uint64_t* d_M,
    const uint64_t* d_pos,
    const uint64_t* d_neg,
    uint64_t        N_ring,
    int             chunk,
    uint64_t*       d_out
) {
    switch (chunk) {
    case 64:   AJTAI_TERNARY_PREMAT_DISPATCH(64);
    case 128:  AJTAI_TERNARY_PREMAT_DISPATCH(128);
    case 256:  AJTAI_TERNARY_PREMAT_DISPATCH(256);
    case 1024: AJTAI_TERNARY_PREMAT_DISPATCH(1024);
    case 4096: AJTAI_TERNARY_PREMAT_DISPATCH(4096);
    default:   return -3;
    }
}
#undef AJTAI_TERNARY_PREMAT_DISPATCH

// MVP probe: time a single-limb tensor-core matmul at commit's shape.
// Output is meaningless (B and M are random INT8); this is a go/no-go
// throughput measurement before building the full 8-limb commit.
int ajtai_tc_commit_probe_ffi(
    const int8_t* d_z_int8,
    const int8_t* d_M_int8,
    int32_t*      d_partial,
    int           K_total,
    int           num_K_chunks
) {
    cudaError_t err = ajtai::tc_commit_probe_run(
        d_z_int8, d_M_int8, d_partial, K_total, num_K_chunks
    );
    return (err == cudaSuccess) ? 0 : -1;
}

int ajtai_multifold_commitment_ffi(
    const uint64_t* d_c_packed,
    const int8_t*   d_r_all,
    int             num_instances,
    uint64_t*       d_out            // [KAPPA * D]
) {
    cudaError_t err = ajtai::multifold_commitment_run(
        d_c_packed, d_r_all, num_instances, d_out
    );
    return (err == cudaSuccess) ? 0 : -1;
}

int ajtai_fold_commitment_ffi(
    const uint64_t* d_c1,        // [KAPPA * 64]
    const uint64_t* d_c2,        // [KAPPA * 64]
    const int8_t*   r_coeffs,    // 64 entries
    uint64_t*       d_out        // [KAPPA * 64]
) {
    ajtai::ChallengeR r;
    for (int k = 0; k < 64; k++) r.coeffs[k] = r_coeffs[k];
    cudaError_t err = ajtai::fold_commitment_run(d_c1, d_c2, r, d_out);
    return (err == cudaSuccess) ? 0 : -1;
}

int ajtai_commit_sparse_ffi(
    const uint32_t* d_key,
    const uint64_t* d_positions,
    uint64_t        K,
    int             chunk,
    uint64_t*       d_out
) {
    cudaError_t err;
    switch (chunk) {
    case 64:   err = ajtai::commit_sparse_run<64  >(d_key, d_positions, K, d_out); break;
    case 128:  err = ajtai::commit_sparse_run<128 >(d_key, d_positions, K, d_out); break;
    case 256:  err = ajtai::commit_sparse_run<256 >(d_key, d_positions, K, d_out); break;
    case 1024: err = ajtai::commit_sparse_run<1024>(d_key, d_positions, K, d_out); break;
    case 4096: err = ajtai::commit_sparse_run<4096>(d_key, d_positions, K, d_out); break;
    default:   return -3;
    }
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Einsum kernels (field-swapped from goldilocks-cuda gl_einsum1 / gl_einsum2).
//
// Operates on `EinsumDimSpec`: each dimension's size + per-input strides. A
// thread block evaluates one output cell by walking the sum dimensions with
// `agl_add(acc, a · b)` (binary einsum) or `agl_add(acc, a)` (unary einsum).
// ============================================================================

#define AGL_EINSUM_MAX_NDIM 8

struct AglEinsumDimSpec {
    int ndim;
    int dims[AGL_EINSUM_MAX_NDIM];
    int strides_a[AGL_EINSUM_MAX_NDIM];
    int strides_b[AGL_EINSUM_MAX_NDIM];
};

__global__ void agl_einsum2_kernel(
    const uint64_t* __restrict__ A,
    const uint64_t* __restrict__ B,
    uint64_t* __restrict__ C,
    int out_size,
    int sum_size,
    AglEinsumDimSpec out_spec,
    AglEinsumDimSpec sum_spec
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int base_a = 0;
    int base_b = 0;
    int rem = idx;
    for (int d = 0; d < out_spec.ndim; d++) {
        int c = rem % out_spec.dims[d];
        rem /= out_spec.dims[d];
        base_a += c * out_spec.strides_a[d];
        base_b += c * out_spec.strides_b[d];
    }

    AlmostGoldilocksField acc(0);
    for (int s = 0; s < sum_size; s++) {
        int sa = base_a;
        int sb = base_b;
        int sr = s;
        for (int d = 0; d < sum_spec.ndim; d++) {
            int c = sr % sum_spec.dims[d];
            sr /= sum_spec.dims[d];
            sa += c * sum_spec.strides_a[d];
            sb += c * sum_spec.strides_b[d];
        }
        AlmostGoldilocksField a_val(A[sa]);
        AlmostGoldilocksField b_val(B[sb]);
        acc = agl_add(acc, agl_mul(a_val, b_val));
    }
    C[idx] = acc.value;
}

__global__ void agl_einsum1_kernel(
    const uint64_t* __restrict__ A,
    uint64_t* __restrict__ C,
    int out_size,
    int sum_size,
    AglEinsumDimSpec out_spec,
    AglEinsumDimSpec sum_spec
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int base_a = 0;
    int rem = idx;
    for (int d = 0; d < out_spec.ndim; d++) {
        int c = rem % out_spec.dims[d];
        rem /= out_spec.dims[d];
        base_a += c * out_spec.strides_a[d];
    }

    AlmostGoldilocksField acc(0);
    for (int s = 0; s < sum_size; s++) {
        int sa = base_a;
        int sr = s;
        for (int d = 0; d < sum_spec.ndim; d++) {
            int c = sr % sum_spec.dims[d];
            sr /= sum_spec.dims[d];
            sa += c * sum_spec.strides_a[d];
        }
        AlmostGoldilocksField a_val(A[sa]);
        acc = agl_add(acc, a_val);
    }
    C[idx] = acc.value;
}

int agl_einsum2_ffi(
    const uint64_t* d_A,
    const uint64_t* d_B,
    uint64_t* d_C,
    int out_size,
    int sum_size,
    int out_ndim,
    const int* out_dims,
    const int* out_strides_a,
    const int* out_strides_b,
    int sum_ndim,
    const int* sum_dims,
    const int* sum_strides_a,
    const int* sum_strides_b
) {
    if (out_ndim > AGL_EINSUM_MAX_NDIM || sum_ndim > AGL_EINSUM_MAX_NDIM) return -1;

    AglEinsumDimSpec out_spec = {0};
    out_spec.ndim = out_ndim;
    for (int i = 0; i < out_ndim; i++) {
        out_spec.dims[i] = out_dims[i];
        out_spec.strides_a[i] = out_strides_a[i];
        out_spec.strides_b[i] = out_strides_b[i];
    }
    AglEinsumDimSpec sum_spec = {0};
    sum_spec.ndim = sum_ndim;
    for (int i = 0; i < sum_ndim; i++) {
        sum_spec.dims[i] = sum_dims[i];
        sum_spec.strides_a[i] = sum_strides_a[i];
        sum_spec.strides_b[i] = sum_strides_b[i];
    }

    int block = WRAPPER_BLOCK_SIZE;
    int grid = (out_size + block - 1) / block;
    agl_einsum2_kernel<<<grid, block>>>(d_A, d_B, d_C, out_size, sum_size, out_spec, sum_spec);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_einsum1_ffi(
    const uint64_t* d_A,
    uint64_t* d_C,
    int out_size,
    int sum_size,
    int out_ndim,
    const int* out_dims,
    const int* out_strides_a,
    int sum_ndim,
    const int* sum_dims,
    const int* sum_strides_a
) {
    if (out_ndim > AGL_EINSUM_MAX_NDIM || sum_ndim > AGL_EINSUM_MAX_NDIM) return -1;

    AglEinsumDimSpec out_spec = {0};
    out_spec.ndim = out_ndim;
    for (int i = 0; i < out_ndim; i++) {
        out_spec.dims[i] = out_dims[i];
        out_spec.strides_a[i] = out_strides_a[i];
    }
    AglEinsumDimSpec sum_spec = {0};
    sum_spec.ndim = sum_ndim;
    for (int i = 0; i < sum_ndim; i++) {
        sum_spec.dims[i] = sum_dims[i];
        sum_spec.strides_a[i] = sum_strides_a[i];
    }

    int block = WRAPPER_BLOCK_SIZE;
    int grid = (out_size + block - 1) / block;
    agl_einsum1_kernel<<<grid, block>>>(d_A, d_C, out_size, sum_size, out_spec, sum_spec);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// ReLUHelper / ProductZeroCheck small kernels
//
// `relu_helper`: neg[i] = (v > q/2) ? (q − v) : 0
// `zero_buffer`: cudaMemset wrapper, exposed for ProductZeroCheck certificates.
// ============================================================================

__global__ void agl_relu_helper_kernel(
    const uint64_t* __restrict__ x,
    uint64_t* __restrict__ neg,
    int n
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Canonicalize, then test against q/2. ALMOST_GOLDILOCKS_PRIME >> 1 is
    // the threshold above which the value is "negative" in signed-int rep.
    uint64_t v = x[i];
    if (v >= ALMOST_GOLDILOCKS_PRIME) v -= ALMOST_GOLDILOCKS_PRIME;
    neg[i] = (v > (ALMOST_GOLDILOCKS_PRIME >> 1)) ? (ALMOST_GOLDILOCKS_PRIME - v) : 0ULL;
}

int agl_relu_helper_ffi(const uint64_t* d_x, uint64_t* d_neg, int n) {
    int block = WRAPPER_BLOCK_SIZE;
    int grid = (n + block - 1) / block;
    agl_relu_helper_kernel<<<grid, block>>>(d_x, d_neg, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// n is size_t, NOT int. A witness buffer reaches 2^31 elements at 3D-UNet
// 128^3 (Y_full = c_out_pad 64 * s_full_pad 2^25), and as an int that wraps to
// -2147483648, which sign-extends to ~1.8e19 bytes and memsets off the end of
// memory. The symptom is a sticky illegal access surfacing in an unrelated
// later call, which is exactly as hard to trace as it sounds.
int agl_zero_buffer_ffi(uint64_t* d_buf, size_t n) {
    // Real code, not -1. A memset failing with "illegal memory access" means
    // the context was ALREADY corrupted by an earlier launch -- a completely
    // different diagnosis from the memset itself being wrong.
    cudaError_t err = cudaMemset(d_buf, 0, (size_t)n * sizeof(uint64_t));
    return (err == cudaSuccess) ? 0 : (int)err;
}

// ============================================================================
// Conv2D: direct 2D convolution with dilation + stride.
//
//   X     : [c_in,  h_in_pad,  w_in_pad]
//   W_flat: [c_out, c_in_pad,  s_kernel_pad]
//   Y     : [c_out, h_out_pad, w_out_pad]  (pre-zeroed)
//
// Output index `j` within W_flat encodes the dilated kernel position:
//   j = kh * dilation_h * stride_w_val + kw * dilation_w
// where `stride_w_val` is the padded W stride used by FlattenKernel2D.

__global__ void agl_conv2d_kernel(
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ W_flat,
    uint64_t* __restrict__ Y,
    int c_out, int h_out, int w_out,
    int c_in,  int kernel_h, int kernel_w,
    int conv_stride_h, int conv_stride_w,
    int dilation_h, int dilation_w,
    int w_in_pad, int h_in_pad,
    int c_in_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int stride_w_val,
    int batch, int x_stride, int y_stride
) {
    // The batch index is the most significant dimension of X and Y and the
    // weights are shared across it, so image b is an independent convolution
    // at offsets b*x_stride / b*y_stride. One grid covers all of them.
    long long per_img = (long long)c_out * h_out * w_out;
    long long total = (long long)batch * per_img;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int b = gid / per_img;
    long long flat_idx = gid - (long long)b * per_img;
    int wo = flat_idx % w_out;
    int tmp = flat_idx / w_out;
    int ho = tmp % h_out;
    int d  = tmp / h_out;

    AlmostGoldilocksField acc(0);
    for (int c = 0; c < c_in; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int ih = ho * conv_stride_h + kh * dilation_h;
                int iw = wo * conv_stride_w + kw * dilation_w;
                size_t x_idx = (size_t)b * (size_t)x_stride
                    + (size_t)iw + (size_t)ih * w_in_pad
                    + (size_t)c * w_in_pad * h_in_pad;
                int j = kh * dilation_h * stride_w_val + kw * dilation_w;
                int wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                acc = agl_add(acc, agl_mul(AlmostGoldilocksField(X[x_idx]),
                                           AlmostGoldilocksField(W_flat[wf_idx])));
            }
        }
    }
    size_t out_idx = (size_t)b * (size_t)y_stride
        + (size_t)wo + (size_t)ho * w_out_pad + (size_t)d * w_out_pad * h_out_pad;
    Y[out_idx] = acc.value;
}

int agl_conv2d_ffi(
    const uint64_t* d_X, const uint64_t* d_W_flat, uint64_t* d_Y,
    int c_out, int h_out, int w_out,
    int c_in,  int kernel_h, int kernel_w,
    int conv_stride_h, int conv_stride_w,
    int dilation_h, int dilation_w,
    int w_in_pad, int h_in_pad,
    int c_in_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int stride_w_val,
    int batch, int x_stride, int y_stride
) {
    int nb = batch > 0 ? batch : 1;
    long long total = (long long)nb * c_out * h_out * w_out;
    if (total <= 0) return 0;
    int block = WRAPPER_BLOCK_SIZE;
    int grid = (int)((total + block - 1) / block);
    agl_conv2d_kernel<<<grid, block>>>(
        d_X, d_W_flat, d_Y, c_out, h_out, w_out, c_in, kernel_h, kernel_w,
        conv_stride_h, conv_stride_w, dilation_h, dilation_w,
        w_in_pad, h_in_pad, c_in_pad, s_kernel_pad,
        w_out_pad, h_out_pad, stride_w_val,
        nb, x_stride, y_stride
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// ConvTranspose2D: gather form. One thread per output element.
//
//   X     : [c_in,  h_in_pad,  w_in_pad]
//   W_flat: [c_in,  c_out_pad, s_kernel_pad]
//   Y     : [c_out, h_out_pad, w_out_pad]   (pre-zeroed)
//
// The CPU path scatters (loops inputs, accumulates into outputs); on GPU that
// needs atomics, so this inverts it. Scatter writes oh = jh*stride_h + kh, so
// a thread owning oh gathers the taps with oh >= kh, (oh-kh) % stride_h == 0
// and jh = (oh-kh)/stride_h < input_h. No output bounds check is needed in
// either direction: h_out = (input_h-1)*stride_h + kernel_h by construction.

__global__ void agl_conv_transpose2d_kernel(
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ W_flat,
    uint64_t* __restrict__ Y,
    int c_out, int h_out, int w_out,
    int c_in, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int input_h, int input_w,
    int w_in_pad, int h_in_pad,
    int c_out_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int flat_stride
) {
    long long total = (long long)c_out * h_out * w_out;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int ow = (int)(gid % w_out);
    int oh = (int)((gid / w_out) % h_out);
    int d  = (int)(gid / ((long long)w_out * h_out));

    // See the 3D kernel: step by stride rather than test-and-skip.
    AlmostGoldilocksField acc(0);
    for (int c = 0; c < c_in; c++) {
        for (int kh = oh % stride_h; kh < kernel_h && kh <= oh; kh += stride_h) {
            int jh = (oh - kh) / stride_h;
            if (jh >= input_h) continue;
            for (int kw = ow % stride_w; kw < kernel_w && kw <= ow; kw += stride_w) {
                int jw = (ow - kw) / stride_w;
                if (jw >= input_w) continue;
                size_t x_idx = (size_t)jw + (size_t)jh * w_in_pad
                    + (size_t)c * w_in_pad * h_in_pad;
                int j = kh * flat_stride + kw;
                size_t wf_idx = (size_t)j + (size_t)d * s_kernel_pad
                    + (size_t)c * s_kernel_pad * c_out_pad;
                acc = agl_add(acc, agl_mul(AlmostGoldilocksField(X[x_idx]),
                                           AlmostGoldilocksField(W_flat[wf_idx])));
            }
        }
    }
    Y[(size_t)ow + (size_t)oh * w_out_pad + (size_t)d * w_out_pad * h_out_pad] = acc.value;
}

int agl_conv_transpose2d_ffi(
    const uint64_t* d_X, const uint64_t* d_W_flat, uint64_t* d_Y,
    int c_out, int h_out, int w_out,
    int c_in, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int input_h, int input_w,
    int w_in_pad, int h_in_pad,
    int c_out_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int flat_stride
) {
    long long total = (long long)c_out * h_out * w_out;
    if (total <= 0) return 0;
    int block = WRAPPER_BLOCK_SIZE;
    int grid = (int)((total + block - 1) / block);
    agl_conv_transpose2d_kernel<<<grid, block>>>(
        d_X, d_W_flat, d_Y, c_out, h_out, w_out, c_in, kernel_h, kernel_w,
        stride_h, stride_w, input_h, input_w, w_in_pad, h_in_pad,
        c_out_pad, s_kernel_pad, w_out_pad, h_out_pad, flat_stride
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// ConvTranspose3D: gather form. One thread per output element.
//
//   X     : [c_in,  d_in_pad,  h_in_pad,  w_in_pad]
//   W_flat: [c_in,  c_out_pad, s_kernel_pad]
//   Y     : [c_out, d_out_pad, h_out_pad, w_out_pad]   (pre-zeroed)
//
// Y[d,od,oh,ow] = sum_c sum_k X[c,jd,jh,jw] * W[c,d,j], where
// jd = (od-kd)/stride_d and the tap contributes only when od >= kd,
// (od-kd) % stride_d == 0 and jd < input_d (likewise h, w). Mirrors the CPU
// `ConvTranspose3D::run` index-for-index; the scatter form would need atomics.

__global__ void agl_conv_transpose3d_kernel(
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ W_flat,
    uint64_t* __restrict__ Y,
    int c_out, int d_out, int h_out, int w_out,
    int c_in, int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int input_d, int input_h, int input_w,
    int w_in_pad, int h_in_pad, int d_in_pad,
    int c_out_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad, int d_out_pad,
    int flat_stride_h, int flat_stride_w
) {
    long long total = (long long)c_out * d_out * h_out * w_out;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int ow = (int)(gid % w_out);
    int oh = (int)((gid / w_out) % h_out);
    int od = (int)((gid / ((long long)w_out * h_out)) % d_out);
    int d  = (int)(gid / ((long long)w_out * h_out * d_out));

    // Step the tap loops by `stride` from `o % stride` instead of testing every
    // tap and skipping. Only taps with (o-k) % stride == 0 contribute, so the
    // naive form evaluates stride^3 times more iterations than it uses -- 8x
    // at the decoder's kernel=2/stride=2, which measured SLOWER than the
    // rayon CPU path it was meant to replace (0.70x at c_in 64, 32^3).
    // The bound k <= o keeps j >= 0.
    AlmostGoldilocksField acc(0);
    for (int c = 0; c < c_in; c++) {
        for (int kd = od % stride_d; kd < kernel_d && kd <= od; kd += stride_d) {
            int jd = (od - kd) / stride_d;
            if (jd >= input_d) continue;
            for (int kh = oh % stride_h; kh < kernel_h && kh <= oh; kh += stride_h) {
                int jh = (oh - kh) / stride_h;
                if (jh >= input_h) continue;
                for (int kw = ow % stride_w; kw < kernel_w && kw <= ow; kw += stride_w) {
                    int jw = (ow - kw) / stride_w;
                    if (jw >= input_w) continue;
                    size_t x_idx = (size_t)jw + (size_t)jh * w_in_pad
                        + (size_t)jd * w_in_pad * h_in_pad
                        + (size_t)c * w_in_pad * h_in_pad * d_in_pad;
                    int j = kd * flat_stride_h + kh * flat_stride_w + kw;
                    size_t wf_idx = (size_t)j + (size_t)d * s_kernel_pad
                        + (size_t)c * s_kernel_pad * c_out_pad;
                    acc = agl_add(acc, agl_mul(AlmostGoldilocksField(X[x_idx]),
                                               AlmostGoldilocksField(W_flat[wf_idx])));
                }
            }
        }
    }
    size_t out_idx = (size_t)ow + (size_t)oh * w_out_pad
        + (size_t)od * w_out_pad * h_out_pad
        + (size_t)d * w_out_pad * h_out_pad * d_out_pad;
    Y[out_idx] = acc.value;
}

int agl_conv_transpose3d_ffi(
    const uint64_t* d_X, const uint64_t* d_W_flat, uint64_t* d_Y,
    int c_out, int d_out, int h_out, int w_out,
    int c_in, int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int input_d, int input_h, int input_w,
    int w_in_pad, int h_in_pad, int d_in_pad,
    int c_out_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad, int d_out_pad,
    int flat_stride_h, int flat_stride_w
) {
    long long total = (long long)c_out * d_out * h_out * w_out;
    if (total <= 0) return 0;
    int block = WRAPPER_BLOCK_SIZE;
    int grid = (int)((total + block - 1) / block);
    agl_conv_transpose3d_kernel<<<grid, block>>>(
        d_X, d_W_flat, d_Y, c_out, d_out, h_out, w_out,
        c_in, kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w, input_d, input_h, input_w,
        w_in_pad, h_in_pad, d_in_pad, c_out_pad, s_kernel_pad,
        w_out_pad, h_out_pad, d_out_pad, flat_stride_h, flat_stride_w
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// FlattenKernel2D: scatter W[C_out, C_in, kH, kW] → W_flat[C_out, C_in, S_pad].
// `j = kh * dilation_h * s_w + kw * dilation_w`. Output must be pre-zeroed
// since the scatter doesn't cover the dilation gaps.

__global__ void agl_flatten_kernel2d_kernel(
    const uint64_t* __restrict__ W,
    uint64_t* __restrict__ W_flat,
    int c_out, int c_in, int kh_size, int kw_size,
    int kw_pad, int kh_pad,
    int c_in_pad, int s_kernel_pad,
    int dilation_h, int dilation_w, int s_w
) {
    int total = c_out * c_in * kh_size * kw_size;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int kw = idx % kw_size;
    int t1 = idx / kw_size;
    int kh = t1 % kh_size;
    int t2 = t1 / kh_size;
    int c  = t2 % c_in;
    int d  = t2 / c_in;
    int w_idx = kw + kh * kw_pad + c * kw_pad * kh_pad
              + d * kw_pad * kh_pad * c_in_pad;
    int j = kh * dilation_h * s_w + kw * dilation_w;
    int wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
    W_flat[wf_idx] = W[w_idx];
}

int agl_flatten_kernel2d_ffi(
    const uint64_t* d_W, uint64_t* d_W_flat,
    int c_out, int c_in, int kh, int kw,
    int kw_pad, int kh_pad,
    int c_in_pad, int s_kernel_pad,
    int dilation_h, int dilation_w, int s_w
) {
    int total = c_out * c_in * kh * kw;
    if (total <= 0) return 0;
    int block = WRAPPER_BLOCK_SIZE;
    int grid = (total + block - 1) / block;
    agl_flatten_kernel2d_kernel<<<grid, block>>>(
        d_W, d_W_flat, c_out, c_in, kh, kw,
        kw_pad, kh_pad, c_in_pad, s_kernel_pad,
        dilation_h, dilation_w, s_w
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Conv3D + FlattenKernel3D
// ============================================================================

__global__ void agl_conv3d_kernel(
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ W_flat,
    uint64_t* __restrict__ Y,
    int c_out, int d_out, int h_out, int w_out,
    int c_in,  int kernel_d, int kernel_h, int kernel_w,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int w_in_pad, int h_in_pad, int d_in_pad,
    int c_in_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad, int d_out_pad,
    int stride_h_val, int stride_w_val,
    long long x_len, long long w_len, long long y_len, int* oob_flag
) {
    int total = c_out * d_out * h_out * w_out;
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total) return;
    int wo = flat_idx % w_out;
    int t1 = flat_idx / w_out;
    int ho = t1 % h_out;
    int t2 = t1 / h_out;
    int do_ = t2 % d_out;
    int d   = t2 / d_out;

    AlmostGoldilocksField acc(0);
    for (int c = 0; c < c_in; c++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int id = do_ * conv_stride_d + kd;
                    int ih = ho * conv_stride_h + kh;
                    int iw = wo * conv_stride_w + kw;
                    long long x_idx = (long long)iw + (long long)ih * w_in_pad
                              + (long long)id * w_in_pad * h_in_pad
                              + (long long)c * w_in_pad * h_in_pad * d_in_pad;
                    int j = kd * stride_h_val + kh * stride_w_val + kw;
                    long long wf_idx = (long long)j + (long long)c * s_kernel_pad
                              + (long long)d * s_kernel_pad * c_in_pad;
                    // Bounds guard. x_len/w_len are the REAL allocation sizes;
                    // pass 0 to disable. Reports the first offender instead of
                    // faulting, which is the only way to see the index on a
                    // box whose driver compute-sanitizer will not load.
                    if ((x_len > 0 && (x_idx < 0 || x_idx >= x_len)) ||
                        (w_len > 0 && (wf_idx < 0 || wf_idx >= w_len))) {
                        if (atomicCAS(oob_flag, 0, 1) == 0) {
                            printf("[conv3d OOB] x_idx=%lld/%lld wf_idx=%lld/%lld "
                                   "c=%d d=%d kd=%d kh=%d kw=%d id=%d ih=%d iw=%d\n",
                                   x_idx, (long long)x_len, wf_idx, (long long)w_len,
                                   c, d, kd, kh, kw, id, ih, iw);
                        }
                        return;
                    }
                    acc = agl_add(acc, agl_mul(AlmostGoldilocksField(X[x_idx]),
                                               AlmostGoldilocksField(W_flat[wf_idx])));
                }
            }
        }
    }
    long long out_idx = (long long)wo + (long long)ho * w_out_pad
                + (long long)do_ * w_out_pad * h_out_pad
                + (long long)d * w_out_pad * h_out_pad * d_out_pad;
    if (y_len > 0 && (out_idx < 0 || out_idx >= y_len)) {
        if (atomicCAS(oob_flag, 0, 1) == 0) {
            printf("[conv3d OOB-write] out_idx=%lld/%lld d=%d do_=%d ho=%d wo=%d\n",
                   out_idx, (long long)y_len, d, do_, ho, wo);
        }
        return;
    }
    Y[out_idx] = acc.value;
}

int agl_conv3d_ffi(
    const uint64_t* d_X, const uint64_t* d_W_flat, uint64_t* d_Y,
    int c_out, int d_out, int h_out, int w_out,
    int c_in,  int kernel_d, int kernel_h, int kernel_w,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int w_in_pad, int h_in_pad, int d_in_pad,
    int c_in_pad, int s_kernel_pad,
    int w_out_pad, int h_out_pad, int d_out_pad,
    int stride_h_val, int stride_w_val,
    long long x_len, long long w_len, long long y_len, int* oob_flag
) {
    int total = c_out * d_out * h_out * w_out;
    if (total <= 0) return 0;
    int block = WRAPPER_BLOCK_SIZE;
    int grid = (total + block - 1) / block;
    // Device flag so only the FIRST out-of-bounds thread prints. Allocated
    // only when the guard is armed (lengths non-zero): the readback below
    // needs a device sync, which would otherwise serialise every launch.
    int* d_oob = nullptr;
    const bool guard = (x_len > 0 || w_len > 0 || y_len > 0);
    if (guard) {
        cudaMalloc(&d_oob, sizeof(int));
        cudaMemset(d_oob, 0, sizeof(int));
    }
    agl_conv3d_kernel<<<grid, block>>>(
        d_X, d_W_flat, d_Y, c_out, d_out, h_out, w_out,
        c_in, kernel_d, kernel_h, kernel_w,
        conv_stride_d, conv_stride_h, conv_stride_w,
        w_in_pad, h_in_pad, d_in_pad,
        c_in_pad, s_kernel_pad,
        w_out_pad, h_out_pad, d_out_pad,
        stride_h_val, stride_w_val,
        x_len, w_len, y_len, d_oob
    );
    if (guard) {
        cudaError_t sync_err = cudaDeviceSynchronize();
        int oob = 0;
        cudaMemcpy(&oob, d_oob, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_oob);
        if (oob) return -2;
        if (sync_err != cudaSuccess) return (int)sync_err;
    }
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

__global__ void agl_flatten_kernel3d_kernel(
    const uint64_t* __restrict__ W,
    uint64_t* __restrict__ W_flat,
    int c_out, int c_in, int kd_size, int kh_size, int kw_size,
    int kw_pad, int kh_pad, int kd_pad,
    int c_in_pad, int s_kernel_pad,
    int stride_h, int stride_w
) {
    int total = c_out * c_in * kd_size * kh_size * kw_size;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int kw = idx % kw_size;
    int t1 = idx / kw_size;
    int kh = t1 % kh_size;
    int t2 = t1 / kh_size;
    int kd = t2 % kd_size;
    int t3 = t2 / kd_size;
    int c  = t3 % c_in;
    int d  = t3 / c_in;
    int w_idx = kw + kh * kw_pad + kd * kw_pad * kh_pad
              + c * kw_pad * kh_pad * kd_pad
              + d * kw_pad * kh_pad * kd_pad * c_in_pad;
    int j = kd * stride_h + kh * stride_w + kw;
    int wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
    W_flat[wf_idx] = W[w_idx];
}

int agl_flatten_kernel3d_ffi(
    const uint64_t* d_W, uint64_t* d_W_flat,
    int c_out, int c_in, int kd, int kh, int kw,
    int kw_pad, int kh_pad, int kd_pad,
    int c_in_pad, int s_kernel_pad,
    int stride_h, int stride_w
) {
    int total = c_out * c_in * kd * kh * kw;
    if (total <= 0) return 0;
    int block = WRAPPER_BLOCK_SIZE;
    int grid = (total + block - 1) / block;
    agl_flatten_kernel3d_kernel<<<grid, block>>>(
        d_W, d_W_flat, c_out, c_in, kd, kh, kw,
        kw_pad, kh_pad, kd_pad, c_in_pad, s_kernel_pad,
        stride_h, stride_w
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// DepthwiseConv2D
// ============================================================================

__global__ void agl_depthwise_conv2d_kernel(
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ W_flat,
    uint64_t* __restrict__ Y,
    int channels, int h_out, int w_out,
    int kernel_h, int kernel_w,
    int conv_stride_h, int conv_stride_w,
    int w_in_pad, int h_in_pad,
    int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int stride_w_val
) {
    int total = channels * h_out * w_out;
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total) return;
    int wo = flat_idx % w_out;
    int t1 = flat_idx / w_out;
    int ho = t1 % h_out;
    int c  = t1 / h_out;

    AlmostGoldilocksField acc(0);
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int ih = ho * conv_stride_h + kh;
            int iw = wo * conv_stride_w + kw;
            int x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
            int j = kh * stride_w_val + kw;
            int wf_idx = j + c * s_kernel_pad;
            acc = agl_add(acc, agl_mul(AlmostGoldilocksField(X[x_idx]),
                                       AlmostGoldilocksField(W_flat[wf_idx])));
        }
    }
    int out_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
    Y[out_idx] = acc.value;
}

int agl_depthwise_conv2d_ffi(
    const uint64_t* d_X, const uint64_t* d_W_flat, uint64_t* d_Y,
    int channels, int h_out, int w_out,
    int kernel_h, int kernel_w,
    int conv_stride_h, int conv_stride_w,
    int w_in_pad, int h_in_pad,
    int s_kernel_pad,
    int w_out_pad, int h_out_pad,
    int stride_w_val
) {
    int total = channels * h_out * w_out;
    if (total <= 0) return 0;
    int block = WRAPPER_BLOCK_SIZE;
    int grid = (total + block - 1) / block;
    agl_depthwise_conv2d_kernel<<<grid, block>>>(
        d_X, d_W_flat, d_Y, channels, h_out, w_out,
        kernel_h, kernel_w, conv_stride_h, conv_stride_w,
        w_in_pad, h_in_pad, s_kernel_pad,
        w_out_pad, h_out_pad, stride_w_val
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Full 1D flat convolution: the Y_full aux witness for Conv2D / Conv3D /
// DepthwiseConv2D.
//
//   Y_full[d, m] = Σ_c Σ_taps X[c, p] · W_flat[d, c, j],  p = (s_in − 1) − m + j
//
// Gather formulation, one thread per (d, m), m < s_full = s_in + s_kernel − 1.
// The tap index j = kd·tap_d + kh·tap_h + kw·tap_w unifies the variants:
//   Conv2D:    kernel_d = 1, tap_h = dilation_h·stride_w_val, tap_w = dilation_w
//   Conv3D:    tap_d = stride_h_val, tap_h = stride_w_val, tap_w = 1
//   Depthwise: kernel_d = 1, tap_h = stride_w_val, tap_w = 1, depthwise = 1
// X's channel stride is s_in and its padded buffer is zero at non-real
// positions, so the only masking needed is the bounds check 0 ≤ p < s_in
// (p must be computed in signed arithmetic: it goes negative for m near
// s_full). With `depthwise` set the channel loop collapses to c = d and
// W_flat is [C, s_kernel_pad]. Y_full is [c_out_pad, s_full_pad] and must be
// pre-zeroed (agl_zero_buffer_ffi) — threads only cover m < s_full, d < c_out.

__global__ void agl_conv_full_kernel(
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ W_flat,
    uint64_t* __restrict__ Y_full,
    int c_out, int c_in,
    int kernel_d, int kernel_h, int kernel_w,
    int tap_d, int tap_h, int tap_w,
    int s_in, int s_full, int s_full_pad,
    int c_in_pad, int s_kernel_pad,
    int depthwise,
    int batch, int x_stride, int yf_stride,
    long long x_len, long long w_len, long long y_len, int* oob_flag
) {
    long long per_img = (long long)c_out * s_full;
    long long total = (long long)batch * per_img;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int b = gid / per_img;
    long long flat_idx = gid - (long long)b * per_img;
    int m = flat_idx % s_full;
    int d = flat_idx / s_full;

    long long base = (long long)(s_in - 1) - (long long)m;
    int c_lo = depthwise ? d : 0;
    int c_hi = depthwise ? d + 1 : c_in;
    AlmostGoldilocksField acc(0);
    for (int c = c_lo; c < c_hi; c++) {
        const uint64_t* x_ch =
            X + (size_t)b * (size_t)x_stride + (size_t)c * (size_t)s_in;
        size_t wf_base = depthwise
            ? (size_t)d * s_kernel_pad
            : (size_t)c * s_kernel_pad + (size_t)d * s_kernel_pad * c_in_pad;
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int j = kd * tap_d + kh * tap_h + kw * tap_w;
                    long long p = base + (long long)j;
                    if (p < 0 || p >= (long long)s_in) continue;
                    // Guard the ABSOLUTE offsets, not the per-channel ones:
                    // x_ch already has b*x_stride + c*s_in folded in, and
                    // wf_base likewise, so a bad base is invisible to the
                    // p < s_in test above.
                    long long x_abs = (x_ch - X) + p;
                    long long w_abs = (long long)wf_base + j;
                    if ((x_len > 0 && (x_abs < 0 || x_abs >= x_len)) ||
                        (w_len > 0 && (w_abs < 0 || w_abs >= w_len))) {
                        if (atomicCAS(oob_flag, 0, 1) == 0) {
                            printf("[conv_full OOB] x_abs=%lld/%lld w_abs=%lld/%lld "
                                   "c=%d d=%d m=%d j=%d p=%lld s_in=%d\n",
                                   x_abs, (long long)x_len, w_abs, (long long)w_len,
                                   c, d, m, j, p, s_in);
                        }
                        return;
                    }
                    acc = agl_add(acc, agl_mul(AlmostGoldilocksField(x_ch[p]),
                                               AlmostGoldilocksField(W_flat[wf_base + j])));
                }
            }
        }
    }
    long long yf_abs = (long long)b * yf_stride
                     + (long long)d * s_full_pad + m;
    if (y_len > 0 && (yf_abs < 0 || yf_abs >= y_len)) {
        if (atomicCAS(oob_flag, 0, 1) == 0) {
            printf("[conv_full OOB-write] yf_abs=%lld/%lld b=%d d=%d m=%d "
                   "s_full_pad=%d yf_stride=%d\n",
                   yf_abs, (long long)y_len, b, d, m, s_full_pad, yf_stride);
        }
        return;
    }
    Y_full[yf_abs] = acc.value;
}

int agl_conv_full_ffi(
    const uint64_t* d_X, const uint64_t* d_W_flat, uint64_t* d_Y_full,
    int c_out, int c_in,
    int kernel_d, int kernel_h, int kernel_w,
    int tap_d, int tap_h, int tap_w,
    int s_in, int s_full, int s_full_pad,
    int c_in_pad, int s_kernel_pad,
    int depthwise,
    int batch, int x_stride, int yf_stride,
    long long x_len, long long w_len, long long y_len
) {
    int nb = batch > 0 ? batch : 1;
    long long total = (long long)nb * c_out * s_full;
    if (total <= 0) return 0;
    int* d_oob = nullptr;
    const bool guard = (x_len > 0 || w_len > 0 || y_len > 0);
    if (guard) { cudaMalloc(&d_oob, sizeof(int)); cudaMemset(d_oob, 0, sizeof(int)); }
    int block = WRAPPER_BLOCK_SIZE;
    int grid = (int)((total + block - 1) / block);
    agl_conv_full_kernel<<<grid, block>>>(
        d_X, d_W_flat, d_Y_full, c_out, c_in,
        kernel_d, kernel_h, kernel_w,
        tap_d, tap_h, tap_w,
        s_in, s_full, s_full_pad,
        c_in_pad, s_kernel_pad, depthwise,
        nb, x_stride, yf_stride,
        x_len, w_len, y_len, d_oob
    );
    if (guard) {
        cudaError_t sync_err = cudaDeviceSynchronize();
        int oob = 0;
        cudaMemcpy(&oob, d_oob, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_oob);
        if (oob) return -2;
        // Return the real code, not -1: a bare failure hides whether this was
        // an illegal address, a launch-configuration error, or an error
        // inherited from an earlier launch.
        if (sync_err != cudaSuccess) return (int)sync_err;
    }
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

}  // extern "C"

// ===========================================================================
// Sparse boolean-check sumcheck (zk-torch-4 `SparseBoolSumcheckProverExt2`).
//
// Each term is a selection polynomial: a sorted list of positions in a
// 2^arity cube, all values initially 1, sharing one dense eq table. The CPU
// prover walks each term's entries pairing (2r, 2r+1), gathers eq[2r], eq[2r+1],
// and accumulates a degree-3 round message. These kernels mirror that walk
// exactly; see `compute_round_message` / `receive_challenge`.
//
// A "group" is one pair slot: entry i starts a group unless it is the odd
// partner of i-1 within the same term. Every group emits exactly one folded
// entry, so an exclusive scan of the start flags gives output positions and no
// stream compaction is needed. Zero-valued folds are NOT dropped (the CPU
// drops them); this is transcript-identical because a zero entry contributes
// v(v-1)=0 to every later round message and to final_eval, and its presence
// cannot change a partner's (v0,v1) assignment.
// ===========================================================================

#define AGL_BOOL_BLOCK 256

// --- exclusive scan over uint32 -------------------------------------------
__global__ void agl_u32_blockscan_kernel(
    const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
    uint32_t* __restrict__ blocksums, size_t n
) {
    __shared__ uint32_t sh[AGL_BOOL_BLOCK];
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    uint32_t v = (i < n) ? in[i] : 0u;
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int off = 1; off < AGL_BOOL_BLOCK; off <<= 1) {
        uint32_t add = (threadIdx.x >= off) ? sh[threadIdx.x - off] : 0u;
        __syncthreads();
        sh[threadIdx.x] += add;
        __syncthreads();
    }
    if (i < n) out[i] = sh[threadIdx.x] - v;          // inclusive -> exclusive
    if (threadIdx.x == AGL_BOOL_BLOCK - 1 && blocksums)
        blocksums[blockIdx.x] = sh[threadIdx.x];
}

__global__ void agl_u32_addoffset_kernel(
    uint32_t* __restrict__ out, const uint32_t* __restrict__ offs, size_t n
) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] += offs[blockIdx.x];
}

// 2D launch: blockIdx.y IS the term, so no per-entry search for the owning
// term is needed. A first version binary-searched term_off per entry, which
// cost ~10 uncoalesced global reads per entry per kernel: 32.7G reads per
// sub-group over 24 rounds, and it made the GPU path SLOWER than the CPU.
// Terms are contiguous, so the term index can simply be the grid's y
// dimension. Each block grid-strides within its own term, so correctness does
// not depend on the chosen grid.x.

__global__ void agl_bool_mark_kernel(
    const uint32_t* __restrict__ idx, const uint32_t* __restrict__ term_off,
    uint32_t* __restrict__ flags
) {
    int t = blockIdx.y;
    size_t s = term_off[t], e = term_off[t + 1];
    size_t stride = gridDim.x * (size_t)blockDim.x;
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < e - s; k += stride) {
        size_t i = s + k;
        flags[i] = (k == 0) ? 1u
                 : (((idx[i] & 1u) && idx[i - 1] == idx[i] - 1u) ? 0u : 1u);
    }
}

__global__ void agl_bool_msg_kernel(
    const uint32_t* __restrict__ idx, const uint64_t* __restrict__ val,
    const uint32_t* __restrict__ flags, const uint32_t* __restrict__ term_off,
    const uint64_t* __restrict__ w, const uint64_t* __restrict__ eq,
    uint64_t* __restrict__ partial
) {
    __shared__ uint64_t sh[AGL_BOOL_BLOCK][8];
    int t = blockIdx.y;
    size_t s = term_off[t], e = term_off[t + 1];
    AlmostGoldilocksExt2 acc[4];
    for (int k = 0; k < 4; ++k) acc[k] = AlmostGoldilocksExt2((uint64_t)0, (uint64_t)0);
    const AlmostGoldilocksExt2 one((uint64_t)1, (uint64_t)0);
    const AlmostGoldilocksExt2 two((uint64_t)2, (uint64_t)0);
    const AlmostGoldilocksExt2 three((uint64_t)3, (uint64_t)0);
    AlmostGoldilocksExt2 wt(w[2 * t], w[2 * t + 1]);
    size_t stride = gridDim.x * (size_t)blockDim.x;

    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < e - s; k += stride) {
        size_t i = s + k;
        if (!flags[i]) continue;
        uint32_t pos = idx[i];
        uint32_t rest = pos >> 1;
        AlmostGoldilocksExt2 v0((uint64_t)0, (uint64_t)0), v1((uint64_t)0, (uint64_t)0);
        if ((pos & 1u) == 0u) {
            v0 = AlmostGoldilocksExt2(val[2 * i], val[2 * i + 1]);
            if (i + 1 < e && idx[i + 1] == 2u * rest + 1u)
                v1 = AlmostGoldilocksExt2(val[2 * (i + 1)], val[2 * (i + 1) + 1]);
        } else {
            v1 = AlmostGoldilocksExt2(val[2 * i], val[2 * i + 1]);
        }
        AlmostGoldilocksExt2 eq0(eq[4 * (size_t)rest],     eq[4 * (size_t)rest + 1]);
        AlmostGoldilocksExt2 eq1(eq[4 * (size_t)rest + 2], eq[4 * (size_t)rest + 3]);
        AlmostGoldilocksExt2 a2 = aext2_sub(aext2_mul(two, v1), v0);
        AlmostGoldilocksExt2 e2 = aext2_sub(aext2_mul(two, eq1), eq0);
        AlmostGoldilocksExt2 a3 = aext2_sub(aext2_mul(three, v1), aext2_mul(two, v0));
        AlmostGoldilocksExt2 e3 = aext2_sub(aext2_mul(three, eq1), aext2_mul(two, eq0));
        acc[0] = aext2_add(acc[0], aext2_mul(aext2_mul(wt, aext2_mul(v0, aext2_sub(v0, one))), eq0));
        acc[1] = aext2_add(acc[1], aext2_mul(aext2_mul(wt, aext2_mul(v1, aext2_sub(v1, one))), eq1));
        acc[2] = aext2_add(acc[2], aext2_mul(aext2_mul(wt, aext2_mul(a2, aext2_sub(a2, one))), e2));
        acc[3] = aext2_add(acc[3], aext2_mul(aext2_mul(wt, aext2_mul(a3, aext2_sub(a3, one))), e3));
    }
    for (int k = 0; k < 4; ++k) {
        sh[threadIdx.x][2 * k]     = acc[k].c[0].value;
        sh[threadIdx.x][2 * k + 1] = acc[k].c[1].value;
    }
    __syncthreads();
    for (int st = AGL_BOOL_BLOCK / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            for (int k = 0; k < 4; ++k) {
                AlmostGoldilocksExt2 a(sh[threadIdx.x][2*k],      sh[threadIdx.x][2*k+1]);
                AlmostGoldilocksExt2 b(sh[threadIdx.x+st][2*k],   sh[threadIdx.x+st][2*k+1]);
                AlmostGoldilocksExt2 r = aext2_add(a, b);
                sh[threadIdx.x][2*k]   = r.c[0].value;
                sh[threadIdx.x][2*k+1] = r.c[1].value;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        size_t b = (size_t)blockIdx.y * gridDim.x + blockIdx.x;
        for (int k = 0; k < 8; ++k) partial[b * 8 + k] = sh[0][k];
    }
}

__global__ void agl_bool_fold_kernel(
    const uint32_t* __restrict__ idx, const uint64_t* __restrict__ val,
    const uint32_t* __restrict__ flags, const uint32_t* __restrict__ gid,
    const uint32_t* __restrict__ term_off, uint64_t r0, uint64_t r1,
    uint32_t* __restrict__ out_idx, uint64_t* __restrict__ out_val
) {
    int t = blockIdx.y;
    size_t s = term_off[t], e = term_off[t + 1];
    AlmostGoldilocksExt2 ch(r0, r1);
    size_t stride = gridDim.x * (size_t)blockDim.x;
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < e - s; k += stride) {
        size_t i = s + k;
        if (!flags[i]) continue;
        uint32_t pos = idx[i];
        uint32_t rest = pos >> 1;
        AlmostGoldilocksExt2 v0((uint64_t)0, (uint64_t)0), v1((uint64_t)0, (uint64_t)0);
        if ((pos & 1u) == 0u) {
            v0 = AlmostGoldilocksExt2(val[2 * i], val[2 * i + 1]);
            if (i + 1 < e && idx[i + 1] == 2u * rest + 1u)
                v1 = AlmostGoldilocksExt2(val[2 * (i + 1)], val[2 * (i + 1) + 1]);
        } else {
            v1 = AlmostGoldilocksExt2(val[2 * i], val[2 * i + 1]);
        }
        AlmostGoldilocksExt2 nv = aext2_add(v0, aext2_mul(ch, aext2_sub(v1, v0)));
        uint32_t g = gid[i];
        out_idx[g] = rest;
        out_val[2 * (size_t)g]     = nv.c[0].value;
        out_val[2 * (size_t)g + 1] = nv.c[1].value;
    }
}

__global__ void agl_bool_newoff_kernel(
    const uint32_t* __restrict__ gid, const uint32_t* __restrict__ term_off,
    int n_terms, size_t n, uint32_t total, uint32_t* __restrict__ new_off
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t > n_terms) return;
    size_t s = (size_t)term_off[t];
    new_off[t] = (t == n_terms || s >= n) ? total : gid[s];
}

// Gather the per-term final value: after all rounds each term holds at most
// one live entry, and final_evaluation only needs the one at position 0.
// Downloading the whole idx/val buffers to find it moved ~900 MB per sub-group
// across PCIe to read a few hundred field elements.
__global__ void agl_bool_finish_kernel(
    const uint32_t* __restrict__ idx, const uint64_t* __restrict__ val,
    const uint32_t* __restrict__ term_off, int n_terms, size_t n,
    uint64_t* __restrict__ out
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_terms) return;
    size_t s = term_off[t], e = term_off[t + 1];
    uint64_t c0 = 0, c1 = 0;
    if (s < e && s < n && idx[s] == 0u) { c0 = val[2 * s]; c1 = val[2 * s + 1]; }
    out[2 * t] = c0;
    out[2 * t + 1] = c1;
}

__global__ void agl_bool_init_val_kernel(uint64_t* __restrict__ val, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    val[2 * i] = 1u;          // every selection entry starts at Ext2(1, 0)
    val[2 * i + 1] = 0u;
}

extern "C" {

// Initialize the selection values on device. Building this host-side and
// uploading it cost 726 MB per sub-group at arity 22 to transfer a buffer
// whose every entry is the constant 1.
int agl_bool_finish_ffi(
    const uint32_t* d_idx, const uint64_t* d_val, const uint32_t* d_term_off,
    int n_terms, size_t n, uint64_t* d_out
) {
    if (n_terms <= 0) return 0;
    int nb = (n_terms + 255) / 256;
    agl_bool_finish_kernel<<<nb, 256>>>(d_idx, d_val, d_term_off, n_terms, n, d_out);
    return (int)cudaDeviceSynchronize();
}

int agl_bool_init_val_ffi(uint64_t* d_val, size_t n) {
    if (n == 0) return 0;
    size_t nb = (n + AGL_BOOL_BLOCK - 1) / AGL_BOOL_BLOCK;
    agl_bool_init_val_kernel<<<nb, AGL_BOOL_BLOCK>>>(d_val, n);
    return (int)cudaDeviceSynchronize();
}

}

extern "C" {

// One round: mark groups, scan, degree-3 message, reduce. `d_scan_scratch` is
// caller-owned and must hold >= 2*ceil(n/AGL_BOOL_BLOCK) u32; allocating it per
// round cost ~96 cudaMallocs per sub-group and was part of why the first
// version lost to the CPU.
static int agl_u32_exscan_scratch(
    const uint32_t* d_in, uint32_t* d_out, size_t n,
    uint32_t* scratch, size_t scratch_len
) {
    if (n == 0) return 0;
    size_t nb = (n + AGL_BOOL_BLOCK - 1) / AGL_BOOL_BLOCK;
    if (nb > scratch_len) return -1;
    uint32_t* d_bs = scratch;
    agl_u32_blockscan_kernel<<<nb, AGL_BOOL_BLOCK>>>(d_in, d_out, d_bs, n);
    if (nb > 1) {
        // `rest` is the recursion's OUTPUT and occupies nb entries, so its own
        // scratch must start at rest + nb. Starting it at rest + ceil(nb/256)
        // aliased the output: the recursive block-sum write landed on rest[1]
        // and corrupted the scan, which showed up as garbage term offsets and
        // then an illegal address one round later.
        uint32_t* rest = scratch + nb;
        size_t rest_len = scratch_len - nb;
        if (rest_len < 2 * nb) return -1;
        if (agl_u32_exscan_scratch(d_bs, rest, nb, rest + nb, rest_len - nb) != 0) return -1;
        agl_u32_addoffset_kernel<<<nb, AGL_BOOL_BLOCK>>>(d_out, rest, n);
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int agl_bool_round_msg_ffi(
    const uint32_t* d_idx, const uint64_t* d_val, const uint32_t* d_term_off,
    const uint64_t* d_w, const uint64_t* d_eq, int n_terms, size_t n,
    uint32_t* d_flags, uint32_t* d_gid, uint64_t* d_partial, int grid_x,
    uint32_t* d_scan_scratch, size_t scan_scratch_len,
    uint64_t* h_msg, uint32_t* h_total
) {
    if (n == 0) { for (int k = 0; k < 8; ++k) h_msg[k] = 0; *h_total = 0; return 0; }
    // Per-kernel timing. Three rounds of guessing at this path's cost were all
    // wrong; the only way to attribute it is to time each launch.
    static int t_dbg = -1;
    if (t_dbg < 0) t_dbg = getenv("ZK4_BOOL_KTIME") ? 1 : 0;
    static double acc_mark = 0, acc_scan = 0, acc_msg = 0, acc_red = 0;
    static long acc_rounds = 0;
    cudaEvent_t e0, e1, e2, e3;
    if (t_dbg) {
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventCreate(&e2); cudaEventCreate(&e3);
        cudaEventRecord(e0);
    }
    dim3 g(grid_x, n_terms);
    agl_bool_mark_kernel<<<g, AGL_BOOL_BLOCK>>>(d_idx, d_term_off, d_flags);
    if (t_dbg) cudaEventRecord(e1);
    if (getenv("ZK4_GPU_BOOL_DBG")) {
        cudaError_t me = cudaDeviceSynchronize();
        if (me != cudaSuccess) { fprintf(stderr, "[bool] mark failed: %d (n=%zu terms=%d gx=%d)\n", (int)me, n, n_terms, grid_x); return (int)me; }
    }
    if (agl_u32_exscan_scratch(d_flags, d_gid, n, d_scan_scratch, scan_scratch_len) != 0) return -1;
    if (t_dbg) cudaEventRecord(e2);
    if (getenv("ZK4_GPU_BOOL_DBG")) {
        cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) { fprintf(stderr, "[bool] scan failed: %d (n=%zu scratch=%zu)\n", (int)se, n, scan_scratch_len); return (int)se; }
    }
    uint32_t two_last[2] = {0, 0};
    cudaMemcpy(&two_last[0], d_gid + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&two_last[1], d_flags + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    *h_total = two_last[0] + two_last[1];

    agl_bool_msg_kernel<<<g, AGL_BOOL_BLOCK>>>(
        d_idx, d_val, d_flags, d_term_off, d_w, d_eq, d_partial);
    if (t_dbg) cudaEventRecord(e3);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return (int)e;
    double t_red0 = 0;
    if (t_dbg) {
        float m1, m2, m3;
        cudaEventElapsedTime(&m1, e0, e1);
        cudaEventElapsedTime(&m2, e1, e2);
        cudaEventElapsedTime(&m3, e2, e3);
        acc_mark += m1; acc_scan += m2; acc_msg += m3; acc_rounds++;
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaEventDestroy(e2); cudaEventDestroy(e3);
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        t_red0 = ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
    }
    // grid_x * n_terms partials, not one per 256 entries: at arity 22 that is
    // a few thousand instead of 65535, so the copy and the serial fold below
    // are small.
    size_t mb = (size_t)grid_x * (size_t)n_terms;
    uint64_t* h = (uint64_t*)malloc(mb * 8 * sizeof(uint64_t));
    if (!h) return -1;
    cudaMemcpy(h, d_partial, mb * 8 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    AlmostGoldilocksExt2 sacc[4];
    for (int k = 0; k < 4; ++k) sacc[k] = AlmostGoldilocksExt2((uint64_t)0, (uint64_t)0);
    for (size_t b = 0; b < mb; ++b)
        for (int k = 0; k < 4; ++k)
            sacc[k] = aext2_add(sacc[k], AlmostGoldilocksExt2(h[b*8 + 2*k], h[b*8 + 2*k + 1]));
    for (int k = 0; k < 4; ++k) { h_msg[2*k] = sacc[k].c[0].value; h_msg[2*k+1] = sacc[k].c[1].value; }
    free(h);
    if (t_dbg) {
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        acc_red += (ts.tv_sec * 1e3 + ts.tv_nsec / 1e6) - t_red0;
        if ((acc_rounds % 24) == 0)
            fprintf(stderr, "[bool_ktime] rounds=%ld n=%zu terms=%d gx=%d | mark %.1fms scan %.1fms msg %.1fms hostreduce %.1fms (blocks=%d)\n",
                    acc_rounds, n, n_terms, grid_x, acc_mark, acc_scan, acc_msg, acc_red,
                    grid_x * n_terms);
    }
    return 0;
}

int agl_bool_fold_ffi(
    const uint32_t* d_idx, const uint64_t* d_val, const uint32_t* d_flags,
    const uint32_t* d_gid, const uint32_t* d_term_off, int n_terms, size_t n,
    uint64_t r0, uint64_t r1, uint32_t total, int grid_x,
    uint32_t* d_out_idx, uint64_t* d_out_val, uint32_t* d_new_off
) {
    if (n == 0) return 0;
    dim3 g(grid_x, n_terms);
    agl_bool_fold_kernel<<<g, AGL_BOOL_BLOCK>>>(
        d_idx, d_val, d_flags, d_gid, d_term_off, r0, r1, d_out_idx, d_out_val);
    int tb = (n_terms + 1 + 255) / 256;
    agl_bool_newoff_kernel<<<tb, 256>>>(d_gid, d_term_off, n_terms, n, total, d_new_off);
    return (int)cudaDeviceSynchronize();
}

} // extern "C"

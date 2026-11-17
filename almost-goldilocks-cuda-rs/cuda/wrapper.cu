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
#include "morpheus_gauss.cuh"
#include "morpheus_commit.cuh"
#include "morpheus_auth.cuh"
#include "morpheus_link.cuh"
#include "morpheus_link2.cuh"
#include "morpheus_piop.cuh"
// We also pull in the field/extension kernel TUs so the kernels are linked
// into this object file rather than left as external declarations.
#include "almost_goldilocks_kernels.cu"
#include "almost_extension_kernels.cu"

#include <cstdio>
#include <chrono>

#define WRAPPER_BLOCK_SIZE 256

#include <mutex>
#include <nvtx3/nvToolsExt.h>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdlib>

namespace {
struct AglBlock { void* p; size_t sz; cudaEvent_t ev; };
struct AglLive { int dev; size_t sz; };
constexpr int AGL_MAX_DEV = 64;
std::mutex g_agl_mx;
std::unordered_map<void*, AglLive> g_agl_live;
std::multimap<size_t, AglBlock> g_agl_free[AGL_MAX_DEV];
size_t g_agl_cached[AGL_MAX_DEV] = {0};
std::vector<cudaEvent_t> g_agl_evpool[AGL_MAX_DEV];

int agl_cache_on() {
    static int on = -1;
    if (on < 0) {
        const char* e = std::getenv("AGL_ALLOC_CACHE");
        on = (e && e[0] == '0') ? 0 : 1;
    }
    return on;
}
size_t agl_round(size_t n) {
    // 512 B granules below 1 MiB, 2 MiB (the driver's own page) above.
    if (n <= (size_t(1) << 20)) return (n + 511) & ~size_t(511);
    const size_t g = size_t(2) << 20;
    return (n + g - 1) / g * g;
}
// Release every cached block of `dev`. Caller holds g_agl_mx and has `dev`
// current. cudaFree synchronizes, so pending events are complete afterwards.
void agl_flush_locked(int dev) {
    for (auto& kv : g_agl_free[dev]) {
        cudaEventSynchronize(kv.second.ev);
        cudaFree(kv.second.p);
        g_agl_evpool[dev].push_back(kv.second.ev);
    }
    g_agl_free[dev].clear();
    g_agl_cached[dev] = 0;
}
} // namespace

// Wait for THIS thread's GPU work, not the whole device's. Every kernel FFI
// entry point below used to end in agl_sync_mine(), which waits for
// every thread's work on the device: with 6+ fold-tree workers plus the bool
// and node provers sharing a GPU, each return waited for all of them (3,475
// calls, 120 s of thread time in a GPT-2 2-layer profile). The library is built
// with --default-stream per-thread, so each thread's kernels and synchronous
// copies are on cudaStreamPerThread and this is the sync the callers need.
// AGL_DEVICE_SYNC=1 restores device-wide syncs.
static inline cudaError_t agl_sync_mine() {
    static int dev_sync = -1;
    if (dev_sync < 0) {
        const char* e = std::getenv("AGL_DEVICE_SYNC");
        dev_sync = (e && e[0] == '1') ? 1 : 0;
    }
    return dev_sync ? cudaDeviceSynchronize() : cudaStreamSynchronize(cudaStreamPerThread);
}

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

// ----------------------------------------------------------------------------
// Caching device allocator behind cuda_malloc / cuda_free (every DeviceBuffer).
//
// Why: an nsys profile of GPT-2 (2 layers, seq 128, 2 GPUs) showed 12,092
// cudaMalloc + 12,092 cudaFree per run costing 228 s + 144 s of thread time,
// with single calls up to 1.5 s, against ~16 s of kernels per GPU. cudaMalloc
// and cudaFree take a device-wide lock and cudaFree synchronizes the device, so
// the fold-tree workers, bool provers and node provers serialized behind each
// other (gpu_sp "setup" jumped from ~20 ms to ~1 s on whichever call lost).
//
// What: freed blocks go to a per-device free list and are handed out again to
// a request that fits (best fit, at most 25% + 2 MiB larger). On a cudaMalloc
// failure the device's cache is released and the allocation retried, so peak
// memory is no worse than before when the device is actually full.
//
// Ordering: cudaFree used to synchronize the device, so a buffer was never
// reused while a kernel still read it. Every kernel FFI entry point here
// synchronizes its own stream before returning (agl_sync_mine), so by the time
// Rust drops a DeviceBuffer the kernels that touched it have completed. As a
// second guard each freed block records an event on the freeing thread's
// stream, synchronized before the block is handed out again. (Recording on the
// legacy stream instead would also cover other threads' streams, but measured
// as a blocking call: 170 s of thread time, up to 0.87 s per record.)
//
// AGL_ALLOC_CACHE=0 restores plain cudaMalloc / cudaFree.
// ----------------------------------------------------------------------------

int cuda_malloc(void** ptr, size_t size) {
    if (!agl_cache_on()) return (cudaMalloc(ptr, size) == cudaSuccess) ? 0 : -1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= AGL_MAX_DEV)
        return (cudaMalloc(ptr, size) == cudaSuccess) ? 0 : -1;
    const size_t want = agl_round(size == 0 ? 1 : size);
    // The stream-ordered pool behind cudaMallocAsync defaults to a release
    // threshold of 0, so it hands memory back to the driver at every
    // synchronization and the next cudaMallocAsync pays the driver again
    // (1,462 such calls cost 53 s of thread time in the same profile). Let it
    // keep what it has; cuda_pool_trim still releases it explicitly.
    {
        static bool thr_set[AGL_MAX_DEV] = {false};
        std::lock_guard<std::mutex> lk(g_agl_mx);
        if (!thr_set[dev]) {
            thr_set[dev] = true;
            cudaMemPool_t pool;
            if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
                uint64_t thr = UINT64_MAX;
                cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
            }
        }
    }
    {
        std::unique_lock<std::mutex> lk(g_agl_mx);
        auto& fl = g_agl_free[dev];
        auto it = fl.lower_bound(want);
        if (it != fl.end() && it->first <= want + want / 4 + (size_t(2) << 20)) {
            AglBlock b = it->second;
            fl.erase(it);
            g_agl_cached[dev] -= b.sz;
            g_agl_live[b.p] = AglLive{dev, b.sz};
            lk.unlock();
            // Previous owner's work on this block must be finished. Almost
            // always already true, so this rarely blocks.
            cudaEventSynchronize(b.ev);
            std::lock_guard<std::mutex> lk2(g_agl_mx);
            g_agl_evpool[dev].push_back(b.ev);
            *ptr = b.p;
            return 0;
        }
    }
    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, want);
    if (err != cudaSuccess) {
        cudaGetLastError();  // clear the sticky OOM before retrying
        std::lock_guard<std::mutex> lk(g_agl_mx);
        agl_flush_locked(dev);
        err = cudaMalloc(&p, want);
        if (err != cudaSuccess) { cudaGetLastError(); return -1; }
    }
    std::lock_guard<std::mutex> lk(g_agl_mx);
    g_agl_live[p] = AglLive{dev, want};
    *ptr = p;
    return 0;
}

int cuda_free(void* ptr) {
    if (ptr == nullptr) return 0;
    if (!agl_cache_on()) return (cudaFree(ptr) == cudaSuccess) ? 0 : -1;
    AglLive info;
    cudaEvent_t ev = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_agl_mx);
        auto it = g_agl_live.find(ptr);
        if (it == g_agl_live.end()) {
            // Not ours (allocated before the cache existed, or elsewhere).
            return (cudaFree(ptr) == cudaSuccess) ? 0 : -1;
        }
        info = it->second;
        g_agl_live.erase(it);
        auto& ep = g_agl_evpool[info.dev];
        if (!ep.empty()) { ev = ep.back(); ep.pop_back(); }
    }
    int cur = 0;
    cudaGetDevice(&cur);
    if (cur != info.dev) cudaSetDevice(info.dev);
    if (ev == nullptr && cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        if (cur != info.dev) cudaSetDevice(cur);
        return (cudaFree(ptr) == cudaSuccess) ? 0 : -1;
    }
    cudaEventRecord(ev, cudaStreamPerThread);
    if (cur != info.dev) cudaSetDevice(cur);
    std::lock_guard<std::mutex> lk(g_agl_mx);
    g_agl_free[info.dev].emplace(info.sz, AglBlock{ptr, info.sz, ev});
    g_agl_cached[info.dev] += info.sz;
    return 0;
}

// Release the current device's cached blocks (called by cuda_pool_trim).
int agl_alloc_cache_flush() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= AGL_MAX_DEV) return -1;
    std::lock_guard<std::mutex> lk(g_agl_mx);
    agl_flush_locked(dev);
    return 0;
}
// Bytes currently held in the current device's cache (for diagnostics).
size_t agl_alloc_cache_bytes() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= AGL_MAX_DEV) return 0;
    std::lock_guard<std::mutex> lk(g_agl_mx);
    return g_agl_cached[dev];
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
int agl_alloc_cache_flush();
int cuda_pool_trim(size_t min_bytes_to_keep) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return -1;
    // Also release the DeviceBuffer cache (cuda_malloc above): callers trim to
    // hand memory back before a memory-heavy phase, and that holds for both.
    agl_alloc_cache_flush();
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
// Per-thread-stream sync for the Rust wrappers' own "my kernels are done"
// waits (memory::sync_mine). See agl_sync_mine.
int cuda_sync_mine() {
    return (agl_sync_mine() == cudaSuccess) ? 0 : -1;
}
// NVTX ranges for nsys timelines (zk-torch-4 phase markers). Header-only
// NVTX v3: a no-op unless a tool (nsys) is attached.
void agl_nvtx_push(const char* name) { nvtxRangePushA(name); }
void agl_nvtx_pop() { nvtxRangePop(); }
int cuda_device_synchronize() {
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}
int cuda_get_last_error() { return (int)cudaGetLastError(); }
int cuda_peek_at_last_error() { return (int)cudaPeekAtLastError(); }
/// Device-to-host copy through two reusable pinned 64 MB buffers per device:
/// the DMA of chunk c+1 overlaps the host copy of chunk c (pageable copies of
/// GB-scale buffers run at a few GB/s).
int agl_d2h_staged(void* h_dst, const void* d_src, size_t bytes) {
    if (bytes == 0) return 0;
    const size_t CH = (size_t)64 << 20;
    static std::mutex mx[AGL_MAX_DEV];   // per device: devices copy concurrently
    static void* buf[AGL_MAX_DEV][2] = {};
    static cudaStream_t st[AGL_MAX_DEV] = {};
    static cudaEvent_t ev[AGL_MAX_DEV][2] = {};
    int dev = 0; cudaGetDevice(&dev);
    if (dev < 0 || dev >= AGL_MAX_DEV) return (cudaMemcpy(h_dst, d_src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 0 : -1;
    std::lock_guard<std::mutex> lk(mx[dev]);
    if (!buf[dev][0]) {
        for (int k = 0; k < 2; k++) {
            if (cudaMallocHost(&buf[dev][k], CH) != cudaSuccess) return -1;
            cudaEventCreateWithFlags(&ev[dev][k], cudaEventDisableTiming);
        }
        cudaStreamCreateWithFlags(&st[dev], cudaStreamNonBlocking);
    }
    const size_t n = (bytes + CH - 1) / CH;
    auto issue = [&](size_t c) {
        const size_t off = c * CH, len = bytes - off < CH ? bytes - off : CH;
        cudaMemcpyAsync(buf[dev][c & 1], (const char*)d_src + off, len, cudaMemcpyDeviceToHost, st[dev]);
        cudaEventRecord(ev[dev][c & 1], st[dev]);
    };
    // The caller's prior kernels on its own stream must be complete.
    agl_sync_mine();
    issue(0);
    for (size_t c = 0; c < n; c++) {
        if (c + 1 < n) issue(c + 1);
        if (cudaEventSynchronize(ev[dev][c & 1]) != cudaSuccess) return -1;
        const size_t off = c * CH, len = bytes - off < CH ? bytes - off : CH;
        memcpy((char*)h_dst + off, buf[dev][c & 1], len);
    }
    return 0;
}

/// Host-to-device copy through two pinned 64 MB staging buffers (pageable
/// sources otherwise copy at a fraction of the bus rate). Returns once the
/// data is on the device.
int agl_h2d_staged(void* d_dst, const void* h_src, size_t bytes) {
    if (bytes == 0) return 0;
    const size_t CH = (size_t)64 << 20;
    static std::mutex mx[AGL_MAX_DEV];
    static void* buf[AGL_MAX_DEV][2] = {};
    static cudaStream_t st[AGL_MAX_DEV] = {};
    static cudaEvent_t ev[AGL_MAX_DEV][2] = {};
    int dev = 0; cudaGetDevice(&dev);
    if (dev < 0 || dev >= AGL_MAX_DEV) return (cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice) == cudaSuccess) ? 0 : -1;
    std::lock_guard<std::mutex> lk(mx[dev]);
    if (!buf[dev][0]) {
        for (int k = 0; k < 2; k++) {
            if (cudaMallocHost(&buf[dev][k], CH) != cudaSuccess) return -1;
            cudaEventCreateWithFlags(&ev[dev][k], cudaEventDisableTiming);
        }
        cudaStreamCreateWithFlags(&st[dev], cudaStreamNonBlocking);
    }
    const size_t n = (bytes + CH - 1) / CH;
    agl_sync_mine();
    for (size_t c = 0; c < n; c++) {
        const size_t off = c * CH, len = bytes - off < CH ? bytes - off : CH;
        // Buffer c & 1 is free once its previous copy (chunk c − 2) is done.
        if (c >= 2 && cudaEventSynchronize(ev[dev][c & 1]) != cudaSuccess) return -1;
        memcpy(buf[dev][c & 1], (const char*)h_src + off, len);
        cudaMemcpyAsync((char*)d_dst + off, buf[dev][c & 1], len, cudaMemcpyHostToDevice, st[dev]);
        cudaEventRecord(ev[dev][c & 1], st[dev]);
    }
    return cudaStreamSynchronize(st[dev]) == cudaSuccess ? 0 : -1;
}

/// Bytes held in the caching allocator's free lists for the current device.
size_t agl_cached_bytes() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= AGL_MAX_DEV) return 0;
    std::lock_guard<std::mutex> lk(g_agl_mx);
    return g_agl_cached[dev];
}
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
    int dev = 0, max_optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (smem_bytes <= max_optin) {
        cudaFuncSetAttribute(
            (const void*)agl_fused_permute_partial_eval_kernel_t<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        agl_fused_permute_partial_eval_kernel_t<true><<<grid, block, (size_t)smem_bytes>>>(
            d_evals, d_output, d_eq_table, d_lo_lut, d_hi_lut,
            n, m, half, output_size
        );
    } else {
        // LUTs too large for this device's shared memory: read them from
        // global memory instead of failing the launch (which sent the whole
        // einsum to the host prover: ~20 s for Llama-3's 128K-vocab
        // embedding on an A100). Only the per-warp partials need smem.
        size_t warp_bytes = (size_t)(block / 32) * 2 * sizeof(uint64_t);
        agl_fused_permute_partial_eval_kernel_t<false><<<grid, block, warp_bytes>>>(
            d_evals, d_output, d_eq_table, d_lo_lut, d_hi_lut,
            n, m, half, output_size
        );
    }
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -3;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -2;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -2;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -3;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -3;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -2;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -3;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -2;
}

extern "C"
int link_expand_bits_ffi(const uint64_t* d_bits, uint64_t* d_out, uint64_t n_words) {
    using namespace link_sumcheck;
    unsigned blocks = (unsigned)((n_words + BLOCK - 1) / BLOCK);
    if (blocks == 0) blocks = 1;
    if (blocks > 65535u) blocks = 65535u;
    link_expand_bits_kernel<<<blocks, BLOCK>>>(d_bits, (E2*)d_out, n_words);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (agl_sync_mine() == cudaSuccess) ? 0 : -2;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -2;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -2;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -3;
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
    return (agl_sync_mine() == cudaSuccess) ? 0 : -2;
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
        cudaError_t sync_err = agl_sync_mine();
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
        cudaError_t sync_err = agl_sync_mine();
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
    return (int)agl_sync_mine();
}

int agl_bool_init_val_ffi(uint64_t* d_val, size_t n) {
    if (n == 0) return 0;
    size_t nb = (n + AGL_BOOL_BLOCK - 1) / AGL_BOOL_BLOCK;
    agl_bool_init_val_kernel<<<nb, AGL_BOOL_BLOCK>>>(d_val, n);
    return (int)agl_sync_mine();
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
        cudaError_t me = agl_sync_mine();
        if (me != cudaSuccess) { fprintf(stderr, "[bool] mark failed: %d (n=%zu terms=%d gx=%d)\n", (int)me, n, n_terms, grid_x); return (int)me; }
    }
    if (agl_u32_exscan_scratch(d_flags, d_gid, n, d_scan_scratch, scan_scratch_len) != 0) return -1;
    if (t_dbg) cudaEventRecord(e2);
    if (getenv("ZK4_GPU_BOOL_DBG")) {
        cudaError_t se = agl_sync_mine();
        if (se != cudaSuccess) { fprintf(stderr, "[bool] scan failed: %d (n=%zu scratch=%zu)\n", (int)se, n, scan_scratch_len); return (int)se; }
    }
    uint32_t two_last[2] = {0, 0};
    cudaMemcpy(&two_last[0], d_gid + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&two_last[1], d_flags + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    *h_total = two_last[0] + two_last[1];

    agl_bool_msg_kernel<<<g, AGL_BOOL_BLOCK>>>(
        d_idx, d_val, d_flags, d_term_off, d_w, d_eq, d_partial);
    if (t_dbg) cudaEventRecord(e3);
    cudaError_t e = agl_sync_mine();
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
    return (int)agl_sync_mine();
}

// --------------------------------------------------------------------------
// Morpheus: GPU first pass of the certified Gaussian sampler
// --------------------------------------------------------------------------
// consts: sizeof(mgauss::Consts)/8 words; base: 17 Keccak rate lanes.
// d_out/d_flag are device buffers of n entries. Returns 0 on success.
int agl_morpheus_gauss_words() { return (int)(sizeof(mgauss::Consts) / 8); }

// Samples coordinates [coord0, coord0+n) into device buffer d_out. Indices of
// coordinates the host must replay go to h_flag_idx (capacity n); their count
// to *h_nflag. d_out entries of flagged coordinates are 0 until scattered.
// Optional low-priority stream for the calling thread's Gaussian sampling
// (background presampling must not delay latency-bound work on the device).
static thread_local cudaStream_t t_gauss_stream = 0;
int agl_gauss_low_priority(int on) {
    if (on && !t_gauss_stream) {
        int lo = 0, hi = 0;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        if (cudaStreamCreateWithPriority(&t_gauss_stream, cudaStreamNonBlocking, lo) != cudaSuccess) { t_gauss_stream = 0; return -1; }
    } else if (!on && t_gauss_stream) { cudaStreamDestroy(t_gauss_stream); t_gauss_stream = 0; }
    return 0;
}

int agl_morpheus_gauss_dev(const uint64_t* h_consts, const uint64_t* h_base, uint32_t coord0, uint32_t n,
                           int64_t* d_out, uint32_t* h_flag_idx, uint32_t* h_nflag) {
    cudaStream_t st = t_gauss_stream;
    *h_nflag = 0;
    if (n == 0) return 0;
    void *d_c = nullptr, *d_q0 = nullptr, *d_q1 = nullptr, *d_cnt = nullptr, *d_fl = nullptr;
    size_t qbytes = (size_t)n * 4;
    if (cuda_malloc(&d_c, sizeof(mgauss::Consts)) || cuda_malloc(&d_q0, qbytes) || cuda_malloc(&d_q1, qbytes) ||
        cuda_malloc(&d_fl, qbytes) || cuda_malloc(&d_cnt, 8)) return -1;
    uint32_t* cnt = (uint32_t*)d_cnt;  // [0] queue length, [1] flagged count
    cudaMemcpyAsync(d_c, h_consts, sizeof(mgauss::Consts), cudaMemcpyHostToDevice, st);
    cudaMemsetAsync(cnt, 0, 8, st);
    mgauss::CoinBase base;
    memcpy(base.lane, h_base, sizeof(base.lane));
    int dev = 0; cudaGetDevice(&dev);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const int threads = 128;
    uint32_t in_n = n;
    const uint32_t* in_idx = nullptr;
    uint32_t* out_q = (uint32_t*)d_q0;
    for (uint32_t tr = 0; in_n > 0 && tr < 8192; tr++) {
        cudaMemsetAsync(cnt, 0, 4, st);
        long long need = ((long long)in_n + threads - 1) / threads;
        // Background (low-priority) sampling uses one short block per 128
        // coordinates so that other work's blocks are scheduled in between.
        // MORPHEUS_GAUSS_BG_BLOCKS caps the background grid (grid-stride
        // loop) so that the other stream's kernels keep free SM slots.
        static long long bg_cap = []{ const char* e = getenv("MORPHEUS_GAUSS_BG_BLOCKS"); return e ? atoll(e) : (1LL << 30); }();
        long long cap = st ? bg_cap : (long long)sms * 16;
        int blocks = (int)(need < cap ? need : cap);
        mgauss::trial_kernel<<<blocks, threads, 0, st>>>((const mgauss::Consts*)d_c, base, coord0, tr, in_idx, in_n, out_q, cnt,
                                                  (long long*)d_out, (uint32_t*)d_fl, cnt + 1);
        uint32_t next = 0;
        cudaMemcpyAsync(&next, cnt, 4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        in_n = next;
        in_idx = out_q;
        out_q = (out_q == (uint32_t*)d_q0) ? (uint32_t*)d_q1 : (uint32_t*)d_q0;
    }
    if (in_n > 0) mgauss::flag_rest_kernel<<<sms * 4, 256, 0, st>>>(in_idx, in_n, (long long*)d_out, (uint32_t*)d_fl, cnt + 1);
    cudaMemcpyAsync(h_nflag, cnt + 1, 4, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    if (*h_nflag) { cudaMemcpyAsync(h_flag_idx, d_fl, (size_t)*h_nflag * 4, cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); }
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = st ? cudaStreamSynchronize(st) : agl_sync_mine();
    cuda_free(d_c); cuda_free(d_q0); cuda_free(d_q1); cuda_free(d_fl); cuda_free(d_cnt);
    return (int)e;
}

int agl_morpheus_gauss_scatter(int64_t* d_out, const uint32_t* h_idx, const int64_t* h_val, uint32_t m) {
    if (m == 0) return 0;
    void *di = nullptr, *dv = nullptr;
    if (cuda_malloc(&di, (size_t)m * 4) || cuda_malloc(&dv, (size_t)m * 8)) return -1;
    cudaMemcpy(di, h_idx, (size_t)m * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, h_val, (size_t)m * 8, cudaMemcpyHostToDevice);
    mgauss::scatter_kernel<<<(m + 255) / 256, 256>>>((long long*)d_out, (const uint32_t*)di, (const long long*)dv, m);
    cudaError_t e = agl_sync_mine();
    cuda_free(di); cuda_free(dv);
    return (int)e;
}

// --------------------------------------------------------------------------
// Morpheus: exact NTT-CRT Ajtai commitment over R = F_q[Y]/(Y^4096+1)
// --------------------------------------------------------------------------
static uint64_t mc_pow(uint64_t b, uint64_t e, uint64_t p) {
    uint64_t r = 1; b %= p;
    while (e) { if (e & 1) r = (unsigned __int128)r * b % p; b = (unsigned __int128)b * b % p; e >>= 1; }
    return r;
}

// Device tables, built once per device.
const mcommit::Tables* agl_mcommit_tables() {
    static std::mutex mx;
    static const mcommit::Tables* per_dev[AGL_MAX_DEV] = {nullptr};
    int dev = 0; cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(mx);
    if (per_dev[dev]) return per_dev[dev];
    static const uint64_t PR[5][2] = {{2147377153ULL, 1202012506ULL}, {2147352577ULL, 760495213ULL},
                                      {2147295233ULL, 920713129ULL}, {2147205121ULL, 2061834542ULL},
                                      {2147196929ULL, 1321772815ULL}};
    mcommit::Tables* h = (mcommit::Tables*)calloc(1, sizeof(mcommit::Tables));
    const int N = mcommit::N;
    for (int i = 0; i < 5; i++) {
        uint64_t p = PR[i][0], psi = PR[i][1], psi_inv = mc_pow(psi, p - 2, p);
        h->p[i] = (uint32_t)p;
        h->barrett[i] = UINT64_MAX / p;
        uint64_t ni = mc_pow(N, p - 2, p);
        h->ninv[i] = (uint32_t)ni; h->ninv_sh[i] = (uint32_t)((ni << 32) / p);
        for (int k = 0; k < N; k++) {
            int br = 0; for (int b = 0; b < 12; b++) br |= ((k >> b) & 1) << (11 - b);
            uint64_t z = mc_pow(psi, br, p), zi = mc_pow(psi_inv, br, p);
            h->zeta[i][k] = (uint32_t)z; h->zeta_sh[i][k] = (uint32_t)((z << 32) / p);
            h->izeta[i][k] = (uint32_t)zi; h->izeta_sh[i][k] = (uint32_t)((zi << 32) / p);
        }
        for (int j = 0; j < i; j++) h->ginv[i][j] = (uint32_t)mc_pow(PR[j][0] % p, p - 2, p);
    }
    const uint64_t Qv = mcommit::Q;
    for (int k = 1; k <= 5; k++) {
        // digits of (P-1)/2 by halving the mixed-radix digits (p_i - 1) of P-1.
        uint64_t dg[5], rem = 0;
        for (int i = 0; i < k; i++) dg[i] = PR[i][0] - 1;
        for (int i = k - 1; i >= 0; i--) { uint64_t cur = rem * PR[i][0] + dg[i]; dg[i] = cur / 2; rem = cur % 2; }
        for (int i = 0; i < k; i++) h->half[k - 1][i] = (uint32_t)dg[i];
        unsigned __int128 pm = 1;
        for (int i = 0; i < k; i++) pm = pm * PR[i][0] % Qv;
        h->pmodq[k - 1] = (uint64_t)pm;
    }
    mcommit::Tables* d = nullptr;
    if (cudaMalloc(&d, sizeof(mcommit::Tables)) != cudaSuccess) { free(h); return nullptr; }
    cudaMemcpy(d, h, sizeof(mcommit::Tables), cudaMemcpyHostToDevice);
    free(h);
    per_dev[dev] = d;
    return d;
}

static int mc_grid(size_t work, int threads) {
    int dev = 0; cudaGetDevice(&dev);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    size_t need = (work + threads - 1) / threads;
    size_t cap = (size_t)sms * 32;
    return (int)(need < cap ? (need ? need : 1) : cap);
}

// NTT-domain key cache key[prime][c][row][4096] for columns [col0, col0+ncols).
int agl_mcommit_keygen(const uint32_t* h_seed, uint32_t kind, int rows, uint64_t col0, int ncols, int k, uint32_t* d_key) {
    const mcommit::Tables* T = agl_mcommit_tables();
    if (!T || rows > mcommit::MAXROWS || k > mcommit::MAXP) return -1;
    void* d_seed = nullptr;
    if (cuda_malloc(&d_seed, 32)) return -1;
    cudaMemcpy(d_seed, h_seed, 32, cudaMemcpyHostToDevice);
    size_t work = (size_t)ncols * rows * (mcommit::N / 8);
    mcommit::keygen_kernel<<<mc_grid(work, 128), 128>>>((const uint32_t*)d_seed, kind, rows, col0, ncols, k, T, d_key);
    size_t transforms = (size_t)k * ncols * rows;
    if (transforms > 0x7fffffffULL) return -2;
    mcommit::ntt_fwd_kernel<<<(unsigned)transforms, 512>>>(d_key, ncols * rows, k, T);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(d_seed);
    return (int)e;
}

// NTT-domain partial commitment acc[v][prime][row][4096] = sum over the C
// input columns of key[prime][colidx[c]][row] * NTT(x[v][c]). x_bytes = 1
// (int8) or 8 (int64). colidx may be null (identity). V in {1, 2}. Vector v
// starts at element v * x_vstride (0 = packed, C * 4096).
int agl_mcommit_partial(const uint32_t* d_key, int kcols, int rows, int k, const void* d_x, int x_bytes, int V, int C,
                        uint64_t x_vstride, const uint32_t* d_colidx, uint32_t* d_acc) {
    const mcommit::Tables* T = agl_mcommit_tables();
    if (!T || V < 1 || V > mcommit::MV || rows > mcommit::MAXROWS) return -1;
    const int N = mcommit::N;
    if (C == 0) { cudaMemset(d_acc, 0, (size_t)V * k * rows * N * 4); return (int)agl_sync_mine(); }
    void* d_xh = nullptr;
    size_t xh_elems = (size_t)V * k * C * N;
    if (cuda_malloc(&d_xh, xh_elems * 4)) return -1;
    size_t xw = (size_t)V * C * N;
    size_t vs = x_vstride ? (size_t)x_vstride : (size_t)C * N;
    if (x_bytes == 1) mcommit::residues_kernel<int8_t><<<mc_grid(xw, 256), 256>>>((const int8_t*)d_x, vs, V, C, k, T, (uint32_t*)d_xh);
    else mcommit::residues_kernel<int64_t><<<mc_grid(xw, 256), 256>>>((const int64_t*)d_x, vs, V, C, k, T, (uint32_t*)d_xh);
    mcommit::ntt_fwd_kernel<<<(unsigned)((size_t)V * k * C), 512>>>((uint32_t*)d_xh, C, k, T);
    // Column chunks so that k*N/256 x chunks blocks fill the GPU.
    int dev = 0; cudaGetDevice(&dev);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int xblocks = (k * N + 255) / 256;
    int chunks = (sms * 8 + xblocks - 1) / xblocks;
    if (chunks > C) chunks = C;
    if (chunks < 1) chunks = 1;
    int chunk_cols = (C + chunks - 1) / chunks;
    chunks = (C + chunk_cols - 1) / chunk_cols;
    size_t stride = (size_t)V * k * rows * N;
    void* d_part = nullptr;
    if (cuda_malloc(&d_part, (size_t)chunks * stride * 4)) { cuda_free(d_xh); return -1; }
    dim3 g(xblocks, chunks);
    if (V == 1 && !getenv("MORPHEUS_MAC_V")) mcommit::mac_kernel<1><<<g, 256>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part);
    else if (V == 2 && !getenv("MORPHEUS_MAC_V")) mcommit::mac_kernel<2><<<g, 256>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part);
    else {
        // Row groups of RPT rows; 64 positions per block.
        dim3 gv((unsigned)(k * N / 64), (unsigned)chunks);
        const int th = 64 * ((rows + mcommit::RPT - 1) / mcommit::RPT);
        switch (V) {
            case 1: mcommit::mac_v_kernel<1><<<gv, th>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part); break;
            case 2: mcommit::mac_v_kernel<2><<<gv, th>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part); break;
            case 3: mcommit::mac_v_kernel<3><<<gv, th>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part); break;
            case 4: mcommit::mac_v_kernel<4><<<gv, th>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part); break;
            case 5: mcommit::mac_v_kernel<5><<<gv, th>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part); break;
            case 6: mcommit::mac_v_kernel<6><<<gv, th>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part); break;
            case 7: mcommit::mac_v_kernel<7><<<gv, th>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part); break;
            default: mcommit::mac_v_kernel<8><<<gv, th>>>(d_key, kcols, rows, (const uint32_t*)d_xh, d_colidx, C, k, chunk_cols, T, (uint32_t*)d_part); break;
        }
    }
    mcommit::sum_partials_kernel<<<mc_grid(stride, 256), 256>>>((const uint32_t*)d_part, chunks, stride, k, rows, T, d_acc, stride);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(d_xh); cuda_free(d_part);
    return (int)e;
}

// dst += src (NTT-domain accumulators, same shape), mod p.
int agl_mcommit_add(uint32_t* d_dst, const uint32_t* d_src, int V, int k, int rows) {
    const mcommit::Tables* T = agl_mcommit_tables();
    size_t n = (size_t)V * k * rows * mcommit::N;
    // Reuse sum_partials: two blocks [dst, src] are not contiguous, so copy-add.
    void* tmp = nullptr;
    if (cuda_malloc(&tmp, 2 * n * 4)) return -1;
    cudaMemcpy(tmp, d_dst, n * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy((uint32_t*)tmp + n, d_src, n * 4, cudaMemcpyDeviceToDevice);
    mcommit::sum_partials_kernel<<<mc_grid(n, 256), 256>>>((const uint32_t*)tmp, 2, n, k, rows, T, d_dst, n);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(tmp);
    return (int)e;
}

// out[v][row][4096] = CRT(INTT(acc)) mod q + err. acc is consumed (inverse NTT in place).
int agl_mcommit_finish(uint32_t* d_acc, int V, int k, int rows, const int64_t* d_err, uint64_t* d_out) {
    const mcommit::Tables* T = agl_mcommit_tables();
    if (!T) return -1;
    mcommit::ntt_inv_kernel<<<(unsigned)((size_t)V * k * rows), 512>>>(d_acc, rows, k, T);
    size_t n = (size_t)V * rows * mcommit::N;
    mcommit::crt_kernel<<<mc_grid(n, 256), 256>>>(d_acc, V, k, rows, T, d_err, (unsigned long long*)d_out);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return (int)le;
    return (int)agl_sync_mine();
}

// --------------------------------------------------------------------------
// Morpheus: authentication data path (Ev_raw, decode, combine, norms)
// --------------------------------------------------------------------------
static int ma_sms() { int dev = 0; cudaGetDevice(&dev); int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); return sms; }

// w[l][j] = e_l e_j (K elements, 8 canonical words each).
int agl_mauth_wtable(const uint64_t* h_el, int cols, const uint64_t* h_ej, uint64_t* d_w) {
    void *del = nullptr, *dej = nullptr;
    if (cuda_malloc(&del, (size_t)cols * 64) || cuda_malloc(&dej, 64 * 64)) return -1;
    cudaMemcpy(del, h_el, (size_t)cols * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(dej, h_ej, 64 * 64, cudaMemcpyHostToDevice);
    size_t n = (size_t)cols * 64;
    mauth::wtable_kernel<<<(unsigned)((n + 255) / 256), 256>>>((const mauth::K8d*)del, (const mauth::K8d*)dej, cols, (mauth::K8d*)d_w);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(del); cuda_free(dej);
    return (int)e;
}

// h_out[64][8] = P_m = Σ_{l,j} w[l][j] W_l[j + 64 m]; W is int8 (w_bytes 1) or int64 (8).
int agl_mauth_ev(const void* d_W, int w_bytes, int cols, const uint64_t* d_w, uint64_t* h_out) {
    int blocks = ma_sms() * 4;
    if (blocks > cols) blocks = cols;
    void *part = nullptr, *out = nullptr;
    if (cuda_malloc(&part, (size_t)blocks * 64 * 64) || cuda_malloc(&out, 64 * 64)) return -1;
    if (w_bytes == 0) mauth::ev_partial_kernel<int8_t, true><<<blocks, 256>>>((const int8_t*)d_W, cols, (const mauth::K8d*)d_w, (mauth::K8d*)part);
    else if (w_bytes == 1) mauth::ev_partial_kernel<int8_t><<<blocks, 256>>>((const int8_t*)d_W, cols, (const mauth::K8d*)d_w, (mauth::K8d*)part);
    else mauth::ev_partial_kernel<int64_t><<<blocks, 256>>>((const int64_t*)d_W, cols, (const mauth::K8d*)d_w, (mauth::K8d*)part);
    mauth::ev_reduce_kernel<<<1, 512>>>((const mauth::K8d*)part, blocks, (mauth::K8d*)out);
    cudaMemcpy(h_out, out, 64 * 64, cudaMemcpyDeviceToHost);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(part); cuda_free(out);
    return (int)e;
}

/// agl_mauth_ev for nsrc packed sources at d_W + s · stride bytes (cols
/// columns each): h_out[s][64][8].
int agl_mauth_ev_packed_batch(const uint8_t* d_W, uint64_t stride, int nsrc, int cols, const uint64_t* d_w, uint64_t* h_out) {
    if (nsrc <= 0) return 0;
    int blocks = (ma_sms() * 4 + nsrc - 1) / nsrc;
    if (blocks < 1) blocks = 1;
    if (blocks > cols) blocks = cols;
    void *part = nullptr, *out = nullptr;
    if (cuda_malloc(&part, (size_t)nsrc * blocks * 64 * 64) || cuda_malloc(&out, (size_t)nsrc * 64 * 64)) return -1;
    dim3 grid(blocks, nsrc);
    mauth::ev_partial_kernel<int8_t, true><<<grid, 256>>>((const int8_t*)d_W, cols, (const mauth::K8d*)d_w, (mauth::K8d*)part, (size_t)stride);
    mauth::ev_reduce_kernel<<<nsrc, 512>>>((const mauth::K8d*)part, blocks, (mauth::K8d*)out);
    cudaMemcpy(h_out, out, (size_t)nsrc * 64 * 64, cudaMemcpyDeviceToHost);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(part); cuda_free(out);
    return (int)e;
}

// h_out[c][m] = Σ_h 2^h col_c[h + 64 m] mod q.
int agl_mauth_decode(const int64_t* d_cols, int ncols, uint64_t* h_out) {
    void* out = nullptr;
    size_t n = (size_t)ncols * 64;
    if (cuda_malloc(&out, n * 8)) return -1;
    mauth::decode_kernel<<<(unsigned)((n + 255) / 256), 256>>>(d_cols, ncols, (unsigned long long*)out);
    cudaMemcpy(h_out, out, n * 8, cudaMemcpyDeviceToHost);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(out);
    return (int)e;
}

// Z0 += β0·V, Z1 += β1·V over `cols` ring columns (V int8).
int agl_mauth_combine_multi(const int8_t* const* h_V, int nsrc, int cols, const int8_t* h_beta, int64_t* d_Z0, int64_t* d_Z1) {
    if (nsrc <= 0 || cols <= 0) return 0;
    void* db = nullptr;
    const size_t pb = (size_t)nsrc * sizeof(void*), bb = (size_t)nsrc * 128;
    if (cuda_malloc(&db, pb + bb)) return -1;
    std::vector<uint8_t> tmp(pb + bb);
    memcpy(tmp.data(), h_V, pb); memcpy(tmp.data() + pb, h_beta, bb);
    cudaMemcpy(db, tmp.data(), pb + bb, cudaMemcpyHostToDevice);
    int blocks = ma_sms() * 4;
    if (blocks > cols) blocks = cols;
    if (getenv("MORPHEUS_COMBINE_SCALAR")) mauth::combine_multi_kernel<<<blocks, 512>>>((const int8_t* const*)db, nsrc, cols, (const int8_t*)((char*)db + pb), d_Z0, d_Z1);
    else mauth::combine_dp4a_kernel<false><<<blocks, 512>>>((const int8_t* const*)db, nsrc, cols, (const int8_t*)((char*)db + pb), d_Z0, d_Z1);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(db);
    return (int)e;
}

/// agl_mauth_combine_multi over packed physical columns (1024 bytes each).
int agl_mauth_combine_packed(const uint8_t* const* h_V, int nsrc, int cols, const int8_t* h_beta, int64_t* d_Z0, int64_t* d_Z1) {
    if (nsrc <= 0 || cols <= 0) return 0;
    void* db = nullptr;
    const size_t pb = (size_t)nsrc * sizeof(void*), bb = (size_t)nsrc * 128;
    if (cuda_malloc(&db, pb + bb)) return -1;
    std::vector<uint8_t> tmp(pb + bb);
    memcpy(tmp.data(), h_V, pb); memcpy(tmp.data() + pb, h_beta, bb);
    cudaMemcpy(db, tmp.data(), pb + bb, cudaMemcpyHostToDevice);
    int blocks = ma_sms() * 4;
    if (blocks > cols) blocks = cols;
    mauth::combine_dp4a_kernel<true><<<blocks, 512>>>((const int8_t* const*)db, nsrc, cols, (const int8_t*)((char*)db + pb), d_Z0, d_Z1);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(db);
    return (int)e;
}

/// Tensor-core combine (mauth::combine_tc_kernel): packed (1024-byte) or
/// int8 (4096-byte) physical columns; same result as agl_mauth_combine_multi.
int agl_mauth_combine_tc(const uint8_t* const* h_V, int packed, int nsrc, int cols, const int8_t* h_beta, int64_t* d_Z0, int64_t* d_Z1) {
    if (nsrc <= 0 || cols <= 0) return 0;
    void *dp = nullptr, *db = nullptr, *dc = nullptr;
    const size_t pb = (size_t)nsrc * sizeof(void*), bb = (size_t)nsrc * 128, cb = (size_t)nsrc * 8192;
    if (cuda_malloc(&dp, pb) || cuda_malloc(&db, bb) || cuda_malloc(&dc, cb)) return -1;
    cudaMemcpy(dp, h_V, pb, cudaMemcpyHostToDevice);
    cudaMemcpy(db, h_beta, bb, cudaMemcpyHostToDevice);
    mauth::build_c_kernel<<<ma_sms() * 8, 256>>>((const int8_t*)db, nsrc, (int8_t*)dc);
    int blocks = ma_sms() * 4;
    int need = (cols + mauth::TC_COLS - 1) / mauth::TC_COLS;
    if (blocks > need) blocks = need;
    if (packed) mauth::combine_tc_kernel<true><<<blocks, 256>>>((const uint8_t* const*)dp, nsrc, cols, (const int8_t*)dc, d_Z0, d_Z1);
    else mauth::combine_tc_kernel<false><<<blocks, 256>>>((const uint8_t* const*)dp, nsrc, cols, (const int8_t*)dc, d_Z0, d_Z1);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(dp); cuda_free(db); cuda_free(dc);
    return (int)e;
}

/// Packed logical columns -> packed physical columns.
int agl_morpheus_permute2(const uint8_t* d_src, uint8_t* d_dst, uint64_t ncols) {
    int grid = ma_sms() * 32;
    mauth::permute2_kernel<<<grid, 256>>>(d_src, d_dst, ncols);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

int agl_mauth_combine(const int8_t* d_V, int cols, const int8_t* h_beta2, int64_t* d_Z0, int64_t* d_Z1) {
    void* db = nullptr;
    if (cuda_malloc(&db, 128)) return -1;
    cudaMemcpy(db, h_beta2, 128, cudaMemcpyHostToDevice);
    int blocks = ma_sms() * 8;
    if (blocks > cols) blocks = cols;
    mauth::combine_kernel<<<blocks, 512>>>(d_V, cols, (const int8_t*)db, d_Z0, d_Z1);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(db);
    return (int)e;
}

// 37-bit response wire codec (n a multiple of 8; out holds n/8·37 bytes).
int agl_mauth_pack37(const int64_t* d_z, uint64_t n, int64_t off, uint8_t* d_out) {
    if (n % 8) return -2;
    mauth::pack37_kernel<<<ma_sms() * 8, 256>>>(d_z, n / 8, off, d_out);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}
// Returns 1 if some code exceeds umax, 0 if all decode, < 0 on error.
int agl_mauth_unpack37(const uint8_t* d_in, uint64_t n, int64_t off, uint64_t umax, int64_t* d_z) {
    if (n % 8) return -2;
    void* bad = nullptr;
    if (cuda_malloc(&bad, sizeof(int))) return -1;
    cudaMemset(bad, 0, sizeof(int));
    mauth::unpack37_kernel<<<ma_sms() * 8, 256>>>(d_in, n / 8, off, umax, d_z, (int*)bad);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    int hb = 0;
    cudaMemcpy(&hb, bad, sizeof(int), cudaMemcpyDeviceToHost);
    cuda_free(bad);
    return e != cudaSuccess ? -(int)e : hb;
}

// h_out = {sum of squares lo, hi, max |z|}.
int agl_mauth_norm(const int64_t* d_z, uint64_t n, uint64_t* h_out) {
    int blocks = ma_sms() * 4;
    void* part = nullptr;
    if (cuda_malloc(&part, (size_t)blocks * 24)) return -1;
    mauth::norm_kernel<<<blocks, 256>>>(d_z, n, (unsigned long long*)part);
    uint64_t* h = (uint64_t*)malloc((size_t)blocks * 24);
    cudaMemcpy(h, part, (size_t)blocks * 24, cudaMemcpyDeviceToHost);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    uint64_t lo = 0, hi = 0, mx = 0;
    for (int b = 0; b < blocks; b++) { uint64_t nl = lo + h[3 * b]; hi += h[3 * b + 1] + (nl < lo); lo = nl; if (h[3 * b + 2] > mx) mx = h[3 * b + 2]; }
    h_out[0] = lo; h_out[1] = hi; h_out[2] = mx;
    free(h); cuda_free(part);
    return (int)e;
}

// --------------------------------------------------------------------------
// Morpheus: structured Π_link, block-local rounds and block folds
// --------------------------------------------------------------------------
int agl_mlink_phase1(const int8_t* d_raw, uint64_t src_len, int b0, int i, int d,
                     const void* d_blocks, uint32_t nblocks, const void* d_bterms, const void* d_ttab,
                     const uint64_t* d_A, const uint64_t* d_B, const uint64_t* d_epre, const uint64_t* d_ealpha,
                     const uint64_t* d_qd, uint64_t* h_out) {
    if (d > mlink::MAXD) return -3;
    int sms = ma_sms();
    int grid = sms * 16;
    if ((uint32_t)grid > nblocks) grid = nblocks ? (int)nblocks : 1;
    size_t smem = i > 0 ? ((size_t)1 << (b0 - i)) * sizeof(mlink::K8d) : 0;
    // Per device (and cheap): set on every call.
    cudaFuncSetAttribute(mlink::phase1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    void* part = nullptr;
    if (cuda_malloc(&part, (size_t)grid * mlink::NACC * sizeof(mlink::K8d))) return -1;
    if (cudaMemset(part, 0, (size_t)grid * mlink::NACC * sizeof(mlink::K8d)) != cudaSuccess) { cuda_free(part); return -1; }
    mlink::phase1_kernel<<<grid, 32 * (d + 1), smem>>>(d_raw, src_len, b0, i, d, (const mlink::BlockDesc*)d_blocks, nblocks,
        (const mlink::BlockTerm*)d_bterms, (const mlink::TermTab*)d_ttab, (const mlink::K8d*)d_A, (const mlink::K8d*)d_B,
        (const mlink::K8d*)d_epre, (const mlink::K8d*)d_ealpha, (const mlink::K8d*)d_qd, (mlink::K8d*)part);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    if (e == cudaSuccess) cudaMemcpy(h_out, part, (size_t)grid * mlink::NACC * sizeof(mlink::K8d), cudaMemcpyDeviceToHost);
    cuda_free(part);
    return e == cudaSuccess ? grid : -(int)e;
}

int agl_mlink_fold(const int8_t* d_raw, uint64_t src_len, int b0, const void* d_blocks, uint32_t nblocks,
                   const uint64_t* d_efull, uint64_t* d_out) {
    int grid = ma_sms() * 8;
    if ((uint32_t)grid > nblocks) grid = nblocks ? (int)nblocks : 1;
    mlink::fold_kernel<<<grid, 256>>>(d_raw, src_len, b0, (const mlink::BlockDesc*)d_blocks, nblocks,
        (const mlink::K8d*)d_efull, (mlink::K8d*)d_out);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

// --------------------------------------------------------------------------
// Morpheus: addressed private coins (ternary / uniform-q) on the GPU
// --------------------------------------------------------------------------
// mode 0: ternary (int8 out), mode 1: uniform-q (u64 out). Flags (one byte
// per coordinate) mark coordinates the host must finish on the CPU.
int agl_morpheus_coins(const uint64_t* h_base, int purpose, uint32_t coord0, uint32_t n, int mode,
                       void* d_out, uint8_t* d_flag) {
    if (n == 0) return 0;
    mgauss::CoinBase base;
    memcpy(base.lane, h_base, sizeof(base.lane));
    int grid = ma_sms() * 16;
    long long need = ((long long)n + 255) / 256;
    if (need < grid) grid = (int)need;
    if (mode == 0) mgauss::ternary_kernel<<<grid, 256>>>(base, purpose, coord0, n, (int8_t*)d_out, d_flag);
    else mgauss::uniform_q_kernel<<<grid, 256>>>(base, purpose, coord0, n, (unsigned long long*)d_out, d_flag);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

// --------------------------------------------------------------------------
// Morpheus: structured Π_link v2 (region-organised, one-hot fast path)
// --------------------------------------------------------------------------
// h_consts: mlink2::Consts with device pointers. Each call returns the
// summed NACC partial K values in h_out (8·NACC words).
static int ml2_sum(void* part, int grid, uint64_t* h_out) {
    std::vector<uint64_t> h((size_t)grid * mlink2::NACC * 8);
    cudaMemcpy(h.data(), part, h.size() * 8, cudaMemcpyDeviceToHost);
    for (int k = 0; k < mlink2::NACC * 8; k++) h_out[k] = 0;
    // Host accumulation of canonical words as field elements.
    for (int g = 0; g < grid; g++)
        for (int k = 0; k < mlink2::NACC * 8; k++) {
            unsigned __int128 s = (unsigned __int128)h_out[k] + h[(size_t)g * mlink2::NACC * 8 + k];
            h_out[k] = (uint64_t)(s % 18446744069414584289ULL);
        }
    return 0;
}

static int mp_grid(uint64_t n, int threads);
static int mp_ret();
int agl_mlink2_consts(const void* d_blks, uint32_t nblk, const void* h_consts, uint64_t* d_bc) {
    mlink2::Consts C; memcpy(&C, h_consts, sizeof(C));
    mlink2::block_consts_kernel<<<mp_grid(nblk, 128), 128>>>((const mlink2::Blk*)d_blks, nblk, C, (mauth::K8d*)d_bc);
    return mp_ret();
}

int agl_mlink2_round(const uint8_t* d_raw, uint64_t src_len, int b0, int i, int d, int reduced, const void* h_consts,
                     const void* d_onehot, uint32_t n_onehot, const void* d_generic, const uint32_t* const* d_lists, const uint32_t* n_lists, const uint64_t* d_bc,
                     const uint64_t* d_e1, const uint64_t* d_e2, const uint64_t* d_e3, const uint64_t* d_ealpha,
                     int skip_range, uint64_t* h_out) {
    if (d > mlink2::MAXD) return -3;
    mlink2::Consts C; memcpy(&C, h_consts, sizeof(C));
    int sms = ma_sms();
    for (int k = 0; k < mlink2::NACC * 8; k++) h_out[k] = 0;
    std::vector<uint64_t> tmp(mlink2::NACC * 8);
    auto add_into = [&](void) { for (int k = 0; k < mlink2::NACC * 8; k++) h_out[k] = (uint64_t)(((unsigned __int128)h_out[k] + tmp[k]) % 18446744069414584289ULL); };
    if (n_onehot > 0) {
        int grid = sms * 8; if ((uint32_t)grid > n_onehot) grid = (int)n_onehot;
        void* part = nullptr; if (cuda_malloc(&part, (size_t)grid * mlink2::NACC * 64)) return -1;
        mlink2::onehot_kernel<<<grid, 256>>>(d_raw, src_len, b0, i, d, (const mlink2::Blk*)d_onehot, n_onehot, C,
            (const mauth::K8d*)d_e1, (const mauth::K8d*)d_e2, (const mauth::K8d*)d_e3, (const mauth::K8d*)d_ealpha, (mauth::K8d*)part);
        cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
        if (e != cudaSuccess) { cuda_free(part); return -(int)e; }
        ml2_sum(part, grid, tmp.data()); add_into(); cuda_free(part);
    }
    for (int mode = 0; mode < 4; mode++) {
        const uint32_t n_generic = n_lists[mode];
        if (n_generic == 0) continue;
        int grid = sms * 16; if ((uint32_t)grid > n_generic) grid = (int)n_generic;
        void* part = nullptr; if (cuda_malloc(&part, (size_t)grid * mlink2::NACC * 64)) return -1;
        cudaMemset(part, 0, (size_t)grid * mlink2::NACC * 64);
        const auto tm0 = std::chrono::steady_clock::now();
        const char* gd = getenv("MORPHEUS_DBG_GEN");
        const int dbg = (gd ? atoi(gd) : 0) | (skip_range ? 1 : 0);
        auto gk = mode == 0 ? mlink2::generic_kernel<2, 0> : mode == 1 ? mlink2::generic_kernel<2, 1> : mode == 2 ? mlink2::generic_kernel<2, 2> : mlink2::generic_kernel<2, 3>;
        gk<<<grid, reduced ? 96 : 32 * (d + 1)>>>(d_raw, src_len, b0, i, d, (reduced ? 1 : 0) | (dbg << 1), (const mlink2::Blk*)d_generic, d_lists[mode], n_generic, C, (const mauth::K8d*)d_bc,
            (const mauth::K8d*)d_e1, (const mauth::K8d*)d_ealpha, (mauth::K8d*)part);
        cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
        if (e != cudaSuccess) { cuda_free(part); return -(int)e; }
        ml2_sum(part, grid, tmp.data()); add_into(); cuda_free(part);
        if (getenv("MORPHEUS_GEN_TIMES")) fprintf(stderr, "[gen] round %d mode %d: %u blocks %.1f ms\n", i, mode, n_generic, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tm0).count());
    }
    return 0;
}

int agl_mlink2_coarse_init(const uint8_t* d_raw, uint64_t src_len, int b0, const void* h_consts,
                           const void* d_blks, uint32_t nblk, const uint32_t* d_rid, const uint32_t* d_k, uint64_t nrb,
                           const uint64_t* d_tfold, const uint64_t* d_efull, uint64_t coarse, uint32_t nsrc,
                           uint64_t* d_w, uint64_t* d_q, uint64_t* d_chi) {
    mlink2::Consts C; memcpy(&C, h_consts, sizeof(C));
    size_t bytes = (size_t)nsrc * coarse * 64;
    cudaMemset(d_w, 0, bytes); cudaMemset(d_q, 0, bytes); cudaMemset(d_chi, 0, bytes);
    int sms = ma_sms();
    if (nblk) mlink2::fold_kernel<<<sms * 8, 256>>>(d_raw, src_len, b0, (const mlink2::Blk*)d_blks, nblk, (const mauth::K8d*)d_efull, coarse, (mauth::K8d*)d_w);
    if (nrb) mlink2::coarse_q_kernel<<<sms * 8, 128>>>(C, d_rid, d_k, nrb, (const mauth::K8d*)d_tfold, (const mauth::K8d*)d_efull, b0, coarse, (mauth::K8d*)d_q, (mauth::K8d*)d_chi);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

// Coarse init of the sources [src_base, src_base + nsrc) into local arrays
// (indexing stays global: the kernels see base-shifted pointers).
int agl_mlink2_coarse_init2(const uint8_t* d_raw_virtual, uint64_t src_len, int b0, const void* h_consts,
                            const void* d_blks, uint32_t nblk, const uint32_t* d_rid, const uint32_t* d_k, uint64_t nrb,
                            const uint64_t* d_tfold, const uint64_t* d_efull, int qbits, uint64_t coarse, uint32_t src_base, uint32_t nsrc,
                            uint64_t* d_w, uint64_t* d_q, uint64_t* d_chi) {
    mlink2::Consts C; memcpy(&C, h_consts, sizeof(C));
    size_t bytes = (size_t)nsrc * coarse * 64;
    cudaMemset(d_w, 0, bytes); cudaMemset(d_q, 0, bytes); cudaMemset(d_chi, 0, bytes);
    mauth::K8d* w = (mauth::K8d*)d_w - (size_t)src_base * coarse;
    mauth::K8d* q = (mauth::K8d*)d_q - (size_t)src_base * coarse;
    mauth::K8d* x = (mauth::K8d*)d_chi - (size_t)src_base * coarse;
    int sms = ma_sms();
    if (nblk) mlink2::fold_kernel<<<sms * 8, 256>>>(d_raw_virtual, src_len, b0, (const mlink2::Blk*)d_blks, nblk, (const mauth::K8d*)d_efull, coarse, w);
    if (nrb) mlink2::coarse_q_kernel<<<sms * 8, 128>>>(C, d_rid, d_k, nrb, (const mauth::K8d*)d_tfold, (const mauth::K8d*)d_efull, qbits, coarse, q, x);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

int agl_mlink2_dense_expand(const uint64_t* d_af, int afn, const uint64_t* d_pf, const uint32_t* d_pf_off, int pbits, int ybits,
                            uint64_t ntab, uint64_t* d_out) {
    if (ntab == 0) return 0;
    mlink2::dense_expand_kernel<<<ma_sms() * 16, 256>>>((const mauth::K8d*)d_af, afn, (const mauth::K8d*)d_pf, d_pf_off, pbits, ybits, ntab, (mauth::K8d*)d_out);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

int agl_mlink2_range_bin1(const uint8_t* d_raw, uint64_t src_len, int b0, const void* h_consts,
                          const void* d_blks, uint32_t nblk, const uint64_t* d_bc, const uint64_t* d_ea1, uint64_t* d_R) {
    if (nblk == 0) return 0;
    mlink2::Consts C; memcpy(&C, h_consts, sizeof(C));
    const size_t sh = 4 * (size_t)(mlink2::RB1_C * 16 + 2 * mlink2::RB1_C * 8) * 8;
    cudaFuncSetAttribute(mlink2::range_bin1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    mlink2::range_bin1_kernel<<<ma_sms() * 2, 128, sh>>>(d_raw, src_len, b0, (const mlink2::Blk*)d_blks, nblk, C,
        (const mauth::K8d*)d_bc, (const mauth::K8d*)d_ea1, (unsigned long long*)d_R);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

int agl_mlink2_b_build(const uint64_t* h_coords, uint64_t ncoords, const uint32_t* h_co, const uint8_t* h_nv, const uint32_t* h_off,
                       const uint64_t* h_pre, uint32_t nseg, uint64_t total, uint64_t* d_out, uint64_t out_k) {
    cudaMemset(d_out, 0, out_k * 64);
    if (nseg == 0 || total == 0) return (int)agl_sync_mine();
    void* buf = nullptr;
    const size_t cb = (ncoords ? ncoords : 1) * 64, sb = (size_t)nseg * (4 + 4 + 8) + nseg;
    if (cuda_malloc(&buf, cb + sb + 64)) return -1;
    char* p = (char*)buf;
    cudaMemcpy(p, h_coords, ncoords * 64, cudaMemcpyHostToDevice);
    uint32_t* d_co = (uint32_t*)(p + cb); cudaMemcpy(d_co, h_co, (size_t)nseg * 4, cudaMemcpyHostToDevice);
    uint32_t* d_off = d_co + nseg; cudaMemcpy(d_off, h_off, (size_t)nseg * 4, cudaMemcpyHostToDevice);
    uint64_t* d_pre = (uint64_t*)(((uintptr_t)(d_off + nseg) + 7) & ~(uintptr_t)7); cudaMemcpy(d_pre, h_pre, (size_t)nseg * 8, cudaMemcpyHostToDevice);
    uint8_t* d_nv = (uint8_t*)(d_pre + nseg); cudaMemcpy(d_nv, h_nv, nseg, cudaMemcpyHostToDevice);
    mlink2::b_build_kernel<<<ma_sms() * 16, 256>>>((const mauth::K8d*)p, d_co, d_nv, d_off, d_pre, nseg, total, (mauth::K8d*)d_out);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    cuda_free(buf);
    return (int)e;
}

int agl_mlink2_b_halve(const uint64_t* d_src, uint64_t* d_dst, uint64_t nk, const uint32_t* d_off, const uint64_t* d_pre, uint32_t nseg, uint64_t total) {
    cudaMemcpyAsync(d_dst, d_src, nk * 64, cudaMemcpyDeviceToDevice, 0);
    if (nseg && total) mlink2::b_halve_kernel<<<ma_sms() * 16, 256>>>((const mauth::K8d*)d_src, (mauth::K8d*)d_dst, d_off, d_pre, nseg, total);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

int agl_mlink2_coarse_round(const uint64_t* d_w, const uint64_t* d_q, const uint64_t* d_chi, uint32_t nsrc, uint64_t len,
                            const uint64_t* d_grange, const uint64_t* d_gbit, const uint64_t* d_ealpha, int d, uint64_t* h_out) {
    int sms = ma_sms();
    uint64_t pairs = (len / 2) * nsrc;
    int grid = sms * 16; if ((uint64_t)grid * 32 > pairs) grid = (int)((pairs + 31) / 32); if (grid < 1) grid = 1;
    void* part = nullptr; if (cuda_malloc(&part, (size_t)grid * mlink2::NACC * 64)) return -1;
    cudaMemset(part, 0, (size_t)grid * mlink2::NACC * 64);
    mlink2::coarse_round_kernel<<<grid, 32 * (d + 1)>>>((const mauth::K8d*)d_w, (const mauth::K8d*)d_q, (const mauth::K8d*)d_chi, nsrc, len,
        (const mauth::K8d*)d_grange, (const mauth::K8d*)d_gbit, (const mauth::K8d*)d_ealpha, d, (mauth::K8d*)part);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    if (e == cudaSuccess) ml2_sum(part, grid, h_out);
    cuda_free(part);
    return (int)e;
}

int agl_mlink2_coarse_fold(const uint64_t* d_in, uint64_t* d_out, uint32_t nsrc, uint64_t len, const uint64_t* h_r) {
    mauth::K8d r; memcpy(r.c, h_r, 64);
    uint64_t total = (len / 2) * nsrc;
    int grid = (int)((total + 255) / 256); if (grid > ma_sms() * 32) grid = ma_sms() * 32; if (grid < 1) grid = 1;
    mlink2::coarse_fold_kernel<<<grid, 256>>>((const mauth::K8d*)d_in, (mauth::K8d*)d_out, nsrc, len, r);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

// One-hot plane round (v3): returns 7 K coefficients (range c0..c3, query c0..c2).
int agl_mlink2_onehot3(const uint8_t* d_raw, uint64_t src_len, int b0, int i, const void* h_consts,
                       const void* d_blks, uint32_t nblk, const uint64_t* d_e1, const uint64_t* d_e2, const uint64_t* d_e3,
                       const uint64_t* d_eai, const uint32_t* d_eai_off, const uint64_t* d_eapr, const uint32_t* d_eapr_off,
                       uint64_t* h_out) {
    for (int k = 0; k < 7 * 8; k++) h_out[k] = 0;
    if (nblk == 0) return 0;
    mlink2::Consts C; memcpy(&C, h_consts, sizeof(C));
    int grid = ma_sms() * 16;
    if ((uint32_t)grid * 4 > nblk) grid = (int)((nblk + 3) / 4);
    void* part = nullptr; if (cuda_malloc(&part, (size_t)grid * 7 * 64)) return -1;
    mlink2::onehot3_kernel<<<grid, 128>>>(d_raw, src_len, b0, i, (const mlink2::Blk*)d_blks, nblk, C,
        (const mauth::K8d*)d_e1, (const mauth::K8d*)d_e2, (const mauth::K8d*)d_e3,
        (const mauth::K8d*)d_eai, d_eai_off, (const mauth::K8d*)d_eapr, d_eapr_off, (mauth::K8d*)part);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    if (e == cudaSuccess) {
        std::vector<uint64_t> h((size_t)grid * 7 * 8);
        cudaMemcpy(h.data(), part, h.size() * 8, cudaMemcpyDeviceToHost);
        for (int g = 0; g < grid; g++) for (int k = 0; k < 7 * 8; k++)
            h_out[k] = (uint64_t)(((unsigned __int128)h_out[k] + h[(size_t)g * 56 + k]) % 18446744069414584289ULL);
    }
    cuda_free(part);
    return (int)e;
}

int agl_mlink2_onehot_pre(const uint8_t* d_raw, uint64_t src_len, int b0, const void* h_consts,
                          const void* d_blks, uint32_t nblk, const uint64_t* d_bc_oh,
                          const uint64_t* d_eai, const uint32_t* d_eai_off, uint64_t* d_G, uint64_t* d_W) {
    if (nblk == 0) return 0;
    mlink2::Consts C; memcpy(&C, h_consts, sizeof(C));
    int grid = ma_sms() * 16;
    uint32_t warps = (uint32_t)grid * 4;
    uint32_t per_warp = (nblk + warps - 1) / warps;
    mlink2::onehot_pre_kernel<<<grid, 128>>>(d_raw, src_len, b0, (const mlink2::Blk*)d_blks, nblk, C, (const mauth::K8d*)d_bc_oh,
        (const mauth::K8d*)d_eai, d_eai_off, per_warp, (unsigned long long*)d_G, (unsigned long long*)d_W);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

// Sum `rep` replicas of n split (lo, hi) K8 accumulators [rep][n][16] into
// canonical K8 words [n][8] (mod q).
__global__ void mlink2_rep_reduce_kernel(const uint64_t* in, int rep, uint64_t n, uint64_t* out) {
    const unsigned __int128 Q = 18446744069414584289ULL;
    for (uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; t < n * 8; t += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t c = t >> 3, w = t & 7;
        unsigned __int128 acc = 0;
        for (int r = 0; r < rep; r++) {
            const uint64_t* e = in + ((uint64_t)r * n + c) * 16;
            unsigned __int128 v = ((unsigned __int128)e[2 * w] + (((unsigned __int128)(e[2 * w + 1] % 18446744069414584289ULL)) << 32)) % Q;
            acc += v;
        }
        out[t] = (uint64_t)(acc % Q);
    }
}
int agl_mlink2_rep_reduce(const uint64_t* d_in, int rep, uint64_t n, uint64_t* d_out) {
    mlink2_rep_reduce_kernel<<<(unsigned)((n * 8 + 255) / 256 < 4096 ? (n * 8 + 255) / 256 : 4096), 256>>>(d_in, rep, n, d_out);
    return (int)agl_sync_mine();
}

int agl_mlink2_onehot_bin(const uint8_t* d_raw, uint64_t src_len, int b0, const void* h_consts,
                          const void* d_blks, uint32_t nblk, const uint64_t* d_bc_oh,
                          const uint64_t* d_ea1, const uint64_t* d_ea5, const uint64_t* d_B5,
                          uint64_t* d_GV, uint64_t* d_U, uint64_t* d_J, uint64_t* d_J5) {
    if (nblk == 0) return 0;
    if (b0 != 11) return -1;
    mlink2::Consts C; memcpy(&C, h_consts, sizeof(C));
    const size_t sh = 4 * (size_t)(mlink2::BIN_NP * mlink2::BIN_NP + 64 + 2) * sizeof(mauth::K8d);
    cudaFuncSetAttribute(mlink2::onehot_bin_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    int grid = ma_sms() * 2;
    const char* ch = getenv("MORPHEUS_BIN_CHUNK");
    uint32_t chunk = ch ? (uint32_t)atoi(ch) : 32;
    const bool prof = getenv("MORPHEUS_BIN_PROF") != nullptr;
    if (prof) { unsigned long long z[8] = {0}; cudaMemcpyToSymbol(mlink2::g_bin_prof, z, sizeof(z)); }
    mlink2::onehot_bin_kernel<<<grid, 128, sh>>>(d_raw, src_len, b0, (const mlink2::Blk*)d_blks, nblk, C, (const mauth::K8d*)d_bc_oh,
        (const mauth::K8d*)d_ea1, (const mauth::K8d*)d_ea5, (const mauth::K8d*)d_B5, chunk, (prof ? 1 : 0) | ((getenv("MORPHEUS_DBG_BIN") ? atoi(getenv("MORPHEUS_DBG_BIN")) : 0) << 1),
        (unsigned long long*)d_GV, (unsigned long long*)d_U, (unsigned long long*)d_J, (unsigned long long*)d_J5);
    // Part B: query moments and round-5 bins.
    const size_t sh5 = 4 * (size_t)(64 + 128 + 2) * sizeof(mauth::K8d);
    cudaFuncSetAttribute(mlink2::onehot_q5_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh5);
    mlink2::onehot_q5_kernel<<<ma_sms() * 3, 128, sh5>>>(d_raw, src_len, b0, (const mlink2::Blk*)d_blks, nblk, C, (const mauth::K8d*)d_bc_oh,
        (const mauth::K8d*)d_ea5, (const mauth::K8d*)d_B5, chunk, (unsigned long long*)d_U, (unsigned long long*)d_J5);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    if (prof) {
        unsigned long long h[8]; cudaMemcpyFromSymbol(h, mlink2::g_bin_prof, sizeof(h));
        fprintf(stderr, "[binprof] head %.3g sel %.3g ballot %.3g loop %.3g tail %.3g skip %.3g (Gcycles, lane 0 of each warp); lane max iters %llu, non-dominant max %llu, blocks %u\n", h[0]/1e9, h[1]/1e9, h[2]/1e9, h[3]/1e9, h[4]/1e9, h[7]/1e9, h[5], h[6], nblk);
    }
    return (int)e;
}

/// Packed (2-bit) raw digits -> int8 (phys = 1: physical column order).
int agl_morpheus_unpack2(const uint8_t* d_src, int8_t* d_dst, uint64_t n, int phys) {
    int grid = ma_sms() * 32;
    mauth::unpack2_kernel<<<grid, 256>>>(d_src, d_dst, n, phys);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

int agl_morpheus_permute(const int8_t* d_src, int8_t* d_dst, uint64_t n) {
    int grid = ma_sms() * 32;
    mauth::permute_kernel<<<grid, 256>>>(d_src, d_dst, n);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

// --------------------------------------------------------------------------
// Morpheus: masked graph PIOP tables on the device
// --------------------------------------------------------------------------
static mpiop::VarMap mp_map(const int* pos, int n) { mpiop::VarMap m; m.n = n; for (int j = 0; j < n && j < mpiop::MAXV; j++) m.pos[j] = pos[j]; return m; }
static int mp_grid(uint64_t n, int threads) { uint64_t g = (n + threads - 1) / threads; uint64_t cap = (uint64_t)ma_sms() * 32; if (g > cap) g = cap; return g < 1 ? 1 : (int)g; }
static int mp_ret() { cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine(); return (int)e; }

int agl_mpiop_pe(const uint64_t* d_vals, const int* rest_pos, int nrest_v, const int* fix_pos, int nfix_v,
                 const uint64_t* d_e, uint64_t* d_out) {
    mpiop::VarMap rest = mp_map(rest_pos, nrest_v), fix = mp_map(fix_pos, nfix_v);
    uint64_t nrest = 1ULL << nrest_v, nfix = 1ULL << nfix_v;
    if (nrest >= 4096 || nfix <= 256) {
        mpiop::pe_kernel<<<mp_grid(nrest, 32), 256>>>((const unsigned long long*)d_vals, rest, fix, (const mauth::K8d*)d_e, nrest, nfix, (mauth::K8d*)d_out);
        return mp_ret();
    }
    int gx = (int)((nfix + 255) / 256); if (gx > 64) gx = 64;
    int gy = (int)nrest;
    void* part = nullptr; if (cuda_malloc(&part, (size_t)gx * gy * 64)) return -1;
    dim3 g(gx, gy);
    mpiop::pe_wide_kernel<<<g, 256>>>((const unsigned long long*)d_vals, rest, fix, (const mauth::K8d*)d_e, nrest, nfix, (mauth::K8d*)part);
    int rc = mp_ret();
    if (rc == 0) {
        std::vector<uint64_t> h((size_t)gx * gy * 8), o(gy * 8, 0);
        cudaMemcpy(h.data(), part, h.size() * 8, cudaMemcpyDeviceToHost);
        for (int r = 0; r < gy; r++) for (int x = 0; x < gx; x++) for (int a = 0; a < 8; a++)
            o[r * 8 + a] = (uint64_t)(((unsigned __int128)o[r * 8 + a] + h[((size_t)r * gx + x) * 8 + a]) % 18446744069414584289ULL);
        cudaMemcpy(d_out, o.data(), o.size() * 8, cudaMemcpyHostToDevice);
    }
    cuda_free(part);
    return rc;
}

int agl_mpiop_mle(const uint64_t* d_vals, const uint64_t* h_point, int n, uint64_t* h_out) {
    const int nlo = n < 8 ? n : 8, nmid = n - nlo < 8 ? n - nlo : 8, nhi = n - nlo - nmid;
    if (nhi > 8) return -2;
    const uint64_t nrows = 1ULL << (nmid + nhi);
    uint64_t g = (nrows + 7) / 8; if (g > (uint64_t)ma_sms() * 4) g = (uint64_t)ma_sms() * 4;
    const int grid = (int)g;
    void* buf = nullptr;
    const size_t eb = (size_t)(24 + 768) * 64, pb = (size_t)grid * 64;
    if (cuda_malloc(&buf, eb + pb)) return -1;
    if (n > 0) cudaMemcpy(buf, h_point, (size_t)n * 64, cudaMemcpyHostToDevice);
    mauth::K8d* tab = (mauth::K8d*)buf + 24;
    mauth::K8d* part = (mauth::K8d*)((char*)buf + eb);
    mpiop::eq3_kernel<<<3, 256>>>((const mauth::K8d*)buf, nlo, nmid, nhi, tab);
    mpiop::mle2_kernel<<<grid, 256>>>((const unsigned long long*)d_vals, tab, nlo, nmid, nhi, part);
    int rc = mp_ret();
    if (rc == 0) {
        std::vector<uint64_t> h((size_t)grid * 8);
        cudaMemcpy(h.data(), part, h.size() * 8, cudaMemcpyDeviceToHost);
        for (int a = 0; a < 8; a++) h_out[a] = 0;
        for (int k = 0; k < grid; k++) for (int a = 0; a < 8; a++)
            h_out[a] = (uint64_t)(((unsigned __int128)h_out[a] + h[(size_t)k * 8 + a]) % 18446744069414584289ULL);
    }
    cuda_free(buf);
    return rc;
}

int agl_mpiop_sel_lookup(const uint64_t* d_sel, uint64_t n, int shift, uint64_t mask, const uint64_t* d_table, uint64_t* d_out) {
    mpiop::sel_lookup_kernel<<<mp_grid(n, 256), 256>>>((const mpiop::u64*)d_sel, n, shift, mask, (const mauth::K8d*)d_table, (mauth::K8d*)d_out);
    return mp_ret();
}

int agl_mpiop_eq_outer(const uint64_t* d_lo, int nlo, const uint64_t* d_hi, int nhi, uint64_t* d_out) {
    uint64_t n = (uint64_t)1 << (nlo + nhi);
    mpiop::eq_outer_kernel<<<mp_grid(n, 256), 256>>>((const mauth::K8d*)d_lo, nlo, (const mauth::K8d*)d_hi, n, (mauth::K8d*)d_out);
    return mp_ret();
}

int agl_mpiop_eq(const uint64_t* h_point, int n, uint64_t* d_out) {
    if (n == 0) { uint64_t one[8] = {1,0,0,0,0,0,0,0}; cudaMemcpy(d_out, one, 64, cudaMemcpyHostToDevice); return 0; }
    void* tmp = nullptr; if (cuda_malloc(&tmp, ((size_t)1 << n) * 64)) return -1;
    uint64_t one[8] = {1,0,0,0,0,0,0,0};
    // Ping-pong so the final level lands in d_out.
    mauth::K8d* bufs[2] = {(mauth::K8d*)d_out, (mauth::K8d*)tmp};
    int cur = (n % 2 == 0) ? 0 : 1;
    cudaMemcpy(bufs[cur], one, 64, cudaMemcpyHostToDevice);
    uint64_t half = 1;
    for (int j = n - 1; j >= 0; j--) {
        mauth::K8d r; memcpy(r.c, h_point + 8 * j, 64);
        mpiop::eq_double_kernel<<<mp_grid(half, 256), 256>>>(bufs[cur], bufs[cur ^ 1], half, r);
        cur ^= 1; half <<= 1;
    }
    int rc = mp_ret();
    cuda_free(tmp);
    return rc;
}

int agl_mpiop_gather(const uint64_t* d_in, const int* pos, int nv, uint64_t n, uint64_t* d_out) {
    mpiop::gather_kernel<<<mp_grid(n, 256), 256>>>((const mauth::K8d*)d_in, mp_map(pos, nv), n, (mauth::K8d*)d_out);
    return mp_ret();
}

int agl_mpiop_axpy(uint64_t* d_out, const uint64_t* d_in, uint64_t n, const uint64_t* h_c) {
    mauth::K8d c; memcpy(c.c, h_c, 64);
    mpiop::axpy_kernel<<<mp_grid(n, 256), 256>>>((mauth::K8d*)d_out, (const mauth::K8d*)d_in, n, c);
    return mp_ret();
}

int agl_mpiop_lift(const uint64_t* d_in, uint64_t n, uint64_t* d_out) {
    mpiop::lift_kernel<<<mp_grid(n, 256), 256>>>((const unsigned long long*)d_in, n, (mauth::K8d*)d_out);
    return mp_ret();
}

int agl_mpiop_sc_round(const uint64_t* const* h_tabs, int ntab, uint64_t len, int d, uint64_t* h_out) {
    if (ntab > mpiop::MAXTAB || d > mpiop::MAXTAB) return -3;
    mpiop::Tabs T; T.ntab = ntab; for (int j = 0; j < ntab; j++) T.t[j] = (const mauth::K8d*)h_tabs[j];
    int grid = mp_grid(len / 2, 256);
    void* part = nullptr; if (cuda_malloc(&part, (size_t)grid * (mpiop::MAXTAB + 1) * 64)) return -1;
    mpiop::sc_round_kernel<<<grid, 256>>>(T, len, d, (mauth::K8d*)part);
    int rc = mp_ret();
    if (rc == 0) {
        std::vector<uint64_t> h((size_t)grid * (mpiop::MAXTAB + 1) * 8);
        cudaMemcpy(h.data(), part, h.size() * 8, cudaMemcpyDeviceToHost);
        for (int k = 0; k < (d + 1) * 8; k++) h_out[k] = 0;
        for (int g = 0; g < grid; g++) for (int t = 0; t <= d; t++) for (int a = 0; a < 8; a++)
            h_out[t * 8 + a] = (uint64_t)(((unsigned __int128)h_out[t * 8 + a] + h[((size_t)g * (mpiop::MAXTAB + 1) + t) * 8 + a]) % 18446744069414584289ULL);
    }
    cuda_free(part);
    return rc;
}

// Fused round: fold the ntab tables with r into d_outs, and return the next
// round's d+1 sums (reduced on the device; one small copy, one sync).
int agl_mpiop_sc_fold_round(const uint64_t* const* h_tabs, uint64_t* const* h_outs, int ntab, uint64_t len,
                            const uint64_t* h_r, int d, uint64_t* h_out) {
    if (ntab > mpiop::MAXTAB || d > mpiop::MAXTAB || len < 4) return -3;
    mpiop::Tabs T; T.ntab = ntab; mpiop::OutTabs O;
    for (int j = 0; j < ntab; j++) { T.t[j] = (const mauth::K8d*)h_tabs[j]; O.t[j] = (mauth::K8d*)h_outs[j]; }
    mauth::K8d r; memcpy(r.c, h_r, 64);
    int grid = mp_grid(len / 4, 256);
    // Persistent per-device scratch (partials of the largest grid, the sums,
    // pinned host landing): no allocation per round.
    static void* part_buf[64] = {nullptr}; static void* sum_buf[64] = {nullptr}; static uint64_t* host_buf[64] = {nullptr};
    static std::mutex mx;
    int dev = 0; cudaGetDevice(&dev); if (dev < 0 || dev >= 64) return -4;
    std::lock_guard<std::mutex> lk(mx);   // scratch is shared per device
    {
        if (!part_buf[dev]) {
            const size_t cap = (size_t)ma_sms() * 32 * (mpiop::MAXTAB + 1) * 64;
            if (cudaMalloc(&part_buf[dev], cap) != cudaSuccess) return -1;
            if (cudaMalloc(&sum_buf[dev], (size_t)(mpiop::MAXTAB + 1) * 64) != cudaSuccess) return -1;
            if (cudaMallocHost((void**)&host_buf[dev], (size_t)(mpiop::MAXTAB + 1) * 64) != cudaSuccess) return -1;
        }
    }
    mpiop::sc_fold_round_kernel<<<grid, 256>>>(T, O, len, r, d, (mauth::K8d*)part_buf[dev]);
    mpiop::sc_partial_sum_kernel<<<1, 256>>>((const mauth::K8d*)part_buf[dev], grid, d, (mauth::K8d*)sum_buf[dev]);
    int rc = (int)cudaGetLastError();
    if (rc == 0) rc = (int)cudaMemcpy(host_buf[dev], sum_buf[dev], (size_t)(d + 1) * 64, cudaMemcpyDeviceToHost);
    if (rc == 0) memcpy(h_out, host_buf[dev], (size_t)(d + 1) * 64);
    return rc;
}

int agl_mpiop_sc_fold(const uint64_t* d_in, uint64_t* d_out, uint64_t len, const uint64_t* h_r) {
    mauth::K8d r; memcpy(r.c, h_r, 64);
    mpiop::sc_fold_kernel<<<mp_grid(len / 2, 256), 256>>>((const mauth::K8d*)d_in, (mauth::K8d*)d_out, len, r);
    return mp_ret();
}

// Late block-local round (v2 index kernel): 7 K values (range t=0..3, query t=0..2).
int agl_mlink2_index(const uint8_t* d_raw, uint64_t src_len, int b0, int i, const void* h_consts,
                     const void* d_blks, uint32_t nblk, const uint64_t* d_bc, const uint64_t* d_epre,
                     const uint64_t* d_ealpha, uint64_t* h_out) {
    for (int k = 0; k < 7 * 8; k++) h_out[k] = 0;
    if (nblk == 0) return 0;
    if ((1 << (b0 - i)) > mlink2::IDX_MAXY || b0 < 9 || i < 4) return -3;
    mlink2::Consts C; memcpy(&C, h_consts, sizeof(C));
    int grid = ma_sms() * 16;
    if ((uint32_t)grid * mlink2::IDX_WARPS > nblk) grid = (int)((nblk + mlink2::IDX_WARPS - 1) / mlink2::IDX_WARPS);
    void* part = nullptr; if (cuda_malloc(&part, (size_t)grid * 7 * 64)) return -1;
    // Per device (and cheap): set on every call.
    const char* im = getenv("MORPHEUS_IDX_MINB");
    const int iminb = im ? atoi(im) : 4;
    auto ik = iminb == 1 ? mlink2::index_kernel<1> : iminb == 2 ? mlink2::index_kernel<2> : iminb == 3 ? mlink2::index_kernel<3> : mlink2::index_kernel<4>;
    cudaFuncSetAttribute(ik, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    size_t smem = (size_t)mlink2::IDX_WARPS * 2 * ((size_t)1 << (b0 - i)) * sizeof(mauth::K8d);
    if (smem < (size_t)mlink2::IDX_WARPS * 32 * 2 * sizeof(mauth::K8d)) smem = (size_t)mlink2::IDX_WARPS * 32 * 2 * sizeof(mauth::K8d);
    if ((1 << (b0 - i)) <= 8 && !getenv("MORPHEUS_NO_IBATCH"))
        mlink2::index_batch_kernel<<<grid, 32 * mlink2::IDX_WARPS, smem>>>(d_raw, src_len, b0, i, (const mlink2::Blk*)d_blks, nblk, C,
            (const mauth::K8d*)d_bc, (const mauth::K8d*)d_epre, (const mauth::K8d*)d_ealpha, (mauth::K8d*)part);
    else
    ik<<<grid, 32 * mlink2::IDX_WARPS, smem>>>(d_raw, src_len, b0, i, (const mlink2::Blk*)d_blks, nblk, C,
        (const mauth::K8d*)d_bc, (const mauth::K8d*)d_epre, (const mauth::K8d*)d_ealpha, (mauth::K8d*)part);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    if (e == cudaSuccess) {
        std::vector<uint64_t> h((size_t)grid * 7 * 8);
        cudaMemcpy(h.data(), part, h.size() * 8, cudaMemcpyDeviceToHost);
        for (int g = 0; g < grid; g++) for (int k = 0; k < 7 * 8; k++)
            h_out[k] = (uint64_t)(((unsigned __int128)h_out[k] + h[(size_t)g * 56 + k]) % 18446744069414584289ULL);
    }
    cuda_free(part);
    return (int)e;
}

} // extern "C"

// Narrow / accumulate partial responses of the sharded authentication.
__global__ void k_i64_to_i32(const int64_t* __restrict__ s, int32_t* __restrict__ d, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) d[i] = (int32_t)s[i];
}
__global__ void k_add_i32_i64(int64_t* __restrict__ d, const int32_t* __restrict__ s, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) d[i] += (int64_t)s[i];
}
extern "C" int agl_i64_to_i32(const int64_t* s, int32_t* d, uint64_t n) {
    k_i64_to_i32<<<ma_sms() * 8, 256>>>(s, d, n);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}
extern "C" int agl_add_i32_i64(int64_t* d, const int32_t* s, uint64_t n) {
    k_add_i32_i64<<<ma_sms() * 8, 256>>>(d, s, n);
    cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = agl_sync_mine();
    return (int)e;
}

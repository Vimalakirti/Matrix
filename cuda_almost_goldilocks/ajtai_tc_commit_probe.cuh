// ============================================================================
// MVP probe kernel: just bench the pure WMMA throughput at the commit
// matmul's shape, *without* the negacyclic-Toeplitz structure or
// field-element reconstruction.  Used as a go/no-go gate before building
// the full 8-limb tensor-core commit.
//
// Layout matches what the real kernel would have at log_n=22 (B=13
// ternary chunks padded to 16, output dim κ·D = 960, K = N_ring·D):
//
//     c[B=16, OUT=960]  =  z_unpacked[B, K]  ·  M_dummy[K, OUT]   (col-major)
//
// where both A and B operands are pre-existing INT8 device buffers
// (random content — output is meaningless, only the kernel runtime
// matters here).  Compute = matmul body without Toeplitz unpacking
// or limb decomposition; gives a *lower bound* on commit kernel time
// if we built the full thing.
// ============================================================================

#pragma once

#include <cstdint>
#include <mma.h>
#include "almost_goldilocks.cuh"
#include "ajtai.cuh"

namespace ajtai {

constexpr int COMMIT_OUT_DIM = KAPPA * D;        // = 960

// One warp per output tile (16 × 16). 60 N-tiles (= 960/16), so 60 warps
// per output's full N axis. K-axis split across blocks: each block does
// CHUNK_K K-tiles of (1×CHUNK_K×16) work, writes a partial INT32 [16, 16]
// tile into a partial buffer. A reduce kernel sums partials across the
// K-axis for the final output.
//
// gridDim  = (60, num_K_chunks, 1)
// blockDim = 32                 — single warp per block
template <int CHUNK_K_TILES>
__global__ void
__launch_bounds__(32, 8)
tc_commit_probe_partial_kernel(
    const int8_t* __restrict__ z_int8,          // [16,  K]    row-major
    const int8_t* __restrict__ M_int8,          // [K, 960]    col-major
    int32_t*      __restrict__ partial,         // [num_K_chunks][60][16*16]  row-major within tile
    int                         K_total         // = N_ring · D
) {
    using namespace nvcuda::wmma;
    int n_tile = blockIdx.x;          // 0..59
    int k_blk  = blockIdx.y;          // 0..num_K_chunks-1
    int lane   = threadIdx.x & 31;

    int k_lo   = k_blk * CHUNK_K_TILES * 16;
    int k_end  = k_lo + CHUNK_K_TILES * 16;
    if (k_end > K_total) k_end = K_total;

    fragment<accumulator, 16, 16, 16, int32_t> c_frag;
    fill_fragment(c_frag, 0);

    for (int k = k_lo; k < k_end; k += 16) {
        fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
        load_matrix_sync(a_frag, z_int8 + k, K_total);

        fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
        load_matrix_sync(b_frag,
                         M_int8 + (uint64_t)(n_tile * 16) * (uint64_t)K_total + k,
                         K_total);

        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Write the partial [16, 16] tile.
    __shared__ int32_t tile_smem[16 * 16];
    store_matrix_sync(tile_smem, c_frag, 16, mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < 256; idx += 32) {
        int dst_off = ((uint64_t)k_blk * 60 + (uint64_t)n_tile) * 256 + idx;
        partial[dst_off] = tile_smem[idx];
    }
}

inline cudaError_t tc_commit_probe_run(
    const int8_t* d_z_int8,
    const int8_t* d_M_int8,
    int32_t*      d_partial,
    int           K_total,
    int           num_K_chunks,
    cudaStream_t  stream = 0
) {
    // CHUNK_K_TILES = K_total / 16 / num_K_chunks  (no rounding for the probe).
    dim3 grid(60, (unsigned)num_K_chunks, 1);
    dim3 block(32);
    // Use a fixed CHUNK_K_TILES template parameter — pick one that fits
    // our chosen num_K_chunks at this N.
    constexpr int CK = 64;            // each block does 64 K-tiles = 1024 K-elements
    tc_commit_probe_partial_kernel<CK><<<grid, block, 0, stream>>>(
        d_z_int8, d_M_int8, d_partial, K_total
    );
    return cudaGetLastError();
}

} // namespace ajtai

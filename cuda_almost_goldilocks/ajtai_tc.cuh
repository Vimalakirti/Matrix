// ============================================================================
// Tensor-core variant of the mixed multifold (binary + ternary inputs).
//
// The scalar multifold_mixed_witness_kernel in ajtai.cuh exploits the
// small-coefficient structure of (r, z) with int8 lookups + int32 adds —
// optimal at the scalar level, but it does not use the SM's tensor cores.
//
// This file reformulates the multifold as one INT8 matrix multiply:
//
//     out[N_ring, 64]  =  z_mat[N_ring, M*64]  @  R_mat[M*64, 64]
//
// where R_mat stacks the negacyclic-Toeplitz matrices of all M ring
// challenges (entries ∈ {-2, -1, 0, 1, 2}) and z_mat unpacks the binary
// bitmasks + ternary (pos, neg) pairs to int8 ∈ {-1, 0, +1}.
//
// We then call mma.sync.aligned.m16n16k16.row.col.s32.s8.s8.s32 via the
// CUDA WMMA API (nvcuda::wmma). A100 INT8 tensor cores peak at 312 TOPS
// dense; the scalar path tops out at ~10 G int-ops/s effective, so this
// kernel targets the order-of-magnitude speedup mentioned in the design
// review.
//
// Both paths produce bit-exact identical i16 output. The scalar kernel
// is retained in ajtai.cuh for direct A/B comparison.
// ============================================================================

#pragma once

#include <cstdint>
#include <mma.h>

#include "almost_goldilocks.cuh"

namespace ajtai {

// ---------------------------------------------------------------------------
// build_R_kernel: expand `r_all[M*64]` (int8 ring challenges) into the
// stacked negacyclic-Toeplitz matrix R[M*64, 64] in COLUMN-MAJOR layout.
//
//   R[i*64 + ell, k]  =  sign(wrap) * r_i[(k - ell) mod 64]
//
// Column-major storage is required by the m16n16k16 INT8 mma layout
// (B operand is col_major). One thread per (row, col) entry.
// ---------------------------------------------------------------------------
__global__ void build_R_kernel(
    const int8_t* __restrict__ r_all,           // [num_instances * 64]
    int8_t*       __restrict__ R_mat,           // [K * 64]  col-major, K = num_instances * 64
    int                         num_instances
) {
    int K   = num_instances * 64;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * 64;
    if (tid >= total) return;

    int k       = tid / K;                       // 0..63   (output coefficient)
    int row     = tid % K;                       // 0..K-1
    int i_inst  = row >> 6;                      // row / 64
    int ell     = row & 63;                      // row % 64

    int  signed_idx = k - ell;
    bool wrap       = signed_idx < 0;
    int  idx        = (signed_idx + 64) & 63;
    int8_t rv       = r_all[i_inst * 64 + idx];
    if (wrap) rv = -rv;

    // Column-major: column k starts at offset k * K.
    R_mat[row + (uint64_t)k * (uint64_t)K] = rv;
}

inline cudaError_t build_R_run(
    const int8_t* d_r_all,
    int8_t*       d_R_mat,
    int           num_instances,
    cudaStream_t  stream = 0
) {
    int K = num_instances * 64;
    int total = K * 64;
    int block = 256;
    int grid  = (total + block - 1) / block;
    build_R_kernel<<<grid, block, 0, stream>>>(d_r_all, d_R_mat, num_instances);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// expand_z_kernel: unpack the binary bitmasks (`z_bin_packed`) and ternary
// (pos, neg) chunks into a dense int8 matrix `z_mat[N_ring, M*64]` in
// row-major layout. Entries are in {-1, 0, +1}.
//
// One thread per output element (j, i, ell). Memory layout:
//   z_mat[j * K + i * 64 + ell]  where K = (K_bin + K_tern) * 64
// ---------------------------------------------------------------------------
__global__ void expand_z_kernel(
    const uint64_t* __restrict__ z_bin_packed,   // [K_bin  * N_ring]
    const uint64_t* __restrict__ pos_packed,     // [K_tern * N_ring]
    const uint64_t* __restrict__ neg_packed,     // [K_tern * N_ring]
    int8_t*         __restrict__ z_mat,          // [N_ring * K]   K = (K_bin + K_tern) * 64
    int                          num_binary,
    int                          num_ternary,
    uint64_t                     N_ring
) {
    int      M_inst = num_binary + num_ternary;
    int      K      = M_inst * 64;
    uint64_t total  = N_ring * (uint64_t)K;
    uint64_t tid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    uint64_t j      = tid / (uint64_t)K;
    int      flat   = (int)(tid % (uint64_t)K);
    int      i_inst = flat >> 6;
    int      ell    = flat & 63;

    int8_t v;
    if (i_inst < num_binary) {
        uint64_t bits = z_bin_packed[(uint64_t)i_inst * N_ring + j];
        v = (int8_t)((bits >> ell) & 1ULL);
    } else {
        int      t   = i_inst - num_binary;
        uint64_t p   = pos_packed[(uint64_t)t * N_ring + j];
        uint64_t n   = neg_packed[(uint64_t)t * N_ring + j];
        int8_t   pb  = (int8_t)((p >> ell) & 1ULL);
        int8_t   nb  = (int8_t)((n >> ell) & 1ULL);
        v = pb - nb;                            // ∈ {-1, 0, 1}
    }
    z_mat[tid] = v;
}

inline cudaError_t expand_z_run(
    const uint64_t* d_z_bin_packed,
    const uint64_t* d_pos_packed,
    const uint64_t* d_neg_packed,
    int8_t*         d_z_mat,
    int             num_binary,
    int             num_ternary,
    uint64_t        N_ring,
    cudaStream_t    stream = 0
) {
    int      M_inst = num_binary + num_ternary;
    uint64_t total  = N_ring * (uint64_t)M_inst * 64;
    int      block  = 256;
    uint64_t grid64 = (total + block - 1) / block;
    int      grid   = (int)grid64;
    expand_z_kernel<<<grid, block, 0, stream>>>(
        d_z_bin_packed, d_pos_packed, d_neg_packed,
        d_z_mat, num_binary, num_ternary, N_ring
    );
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// multifold_tc_kernel: WMMA INT8 matmul `out = z_mat @ R_mat`.
//
// Layout (4 warps per block, BLOCK_M = 64 output rows):
//   gridDim  = (N_ring_padded / 64,)
//   blockDim = 128                              4 warps
//
// Each warp owns a 16-row × 64-col output sub-tile. The 4 warps share the
// R-matrix load via shared memory (1 KB R-strip cached once per K-iter,
// then reused across all 4 warps' 4 N-fragments × 4 warps = 16 reads).
//
// Output: each warp writes its 16×64 tile through a separate slice of
// `tile_smem` (4 KB total). The int32 accumulator is cast to int16 on
// write-out.
//
// Assumes A100 (sm_80). The kernel requires N_ring_padded % 64 == 0;
// the host-side wrapper pads z_mat trailing rows with zeros.
// ---------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 4)
multifold_tc_kernel(
    const int8_t* __restrict__ z_mat,         // [N_ring, K]  row-major (K = M*64)
    const int8_t* __restrict__ R_mat,         // [K, 64]      col-major
    int16_t*      __restrict__ output,        // [N_ring, 64] row-major
    int                          num_instances,
    uint64_t                     N_ring
) {
    using namespace nvcuda::wmma;
    int K = num_instances * 64;

    int warp_id = threadIdx.x >> 5;          // 0..3
    int lane    = threadIdx.x & 31;
    int j_lo    = blockIdx.x * 64 + warp_id * 16;

    fragment<accumulator, 16, 16, 16, int32_t> c_frag[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) fill_fragment(c_frag[n], 0);

    // Each warp scans the entire K axis independently — no cross-warp
    // synchronization in the inner loop. R is read straight from global;
    // since all 4 warps in the block load the *same* b_frag at each (k, n)
    // step, L2 absorbs the redundancy (R is ~258 KB ≪ L2 = 40 MB).
    for (int k_lo = 0; k_lo < K; k_lo += 16) {
        fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
        load_matrix_sync(a_frag, z_mat + (uint64_t)j_lo * K + k_lo, K);

        #pragma unroll
        for (int n = 0; n < 4; n++) {
            fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
            load_matrix_sync(
                b_frag,
                R_mat + k_lo + (uint64_t)(n * 16) * (uint64_t)K,
                K
            );
            mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
        }
    }

    // Write-out: each warp owns its own 256-int32 slice of tile_smem.
    __shared__ int32_t tile_smem[4 * 16 * 16];      // 4 warps × 256 ints
    int32_t* warp_tile = tile_smem + warp_id * 256;

    #pragma unroll
    for (int n = 0; n < 4; n++) {
        store_matrix_sync(warp_tile, c_frag[n], 16, mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 256; idx += 32) {
            int row  = idx >> 4;                    // 0..15
            int col  = idx & 15;                    // 0..15
            uint64_t out_row = (uint64_t)j_lo + (uint64_t)row;
            if (out_row < N_ring) {
                output[out_row * 64 + (uint64_t)n * 16 + col]
                    = (int16_t)warp_tile[idx];
            }
        }
        __syncwarp();
    }
}

inline cudaError_t multifold_tc_run(
    const int8_t* d_z_mat,           // [N_ring_padded, M*64] row-major
    const int8_t* d_R_mat,           // [M*64, 64]            col-major
    int16_t*      d_output,          // [N_ring, 64]          row-major
    int           num_instances,
    uint64_t      N_ring,
    cudaStream_t  stream = 0
) {
    // Caller is responsible for padding z_mat rows to a multiple of 64
    // (= BLOCK_M, 4 warps × 16 rows each).
    uint64_t n_padded = (N_ring + 63) & ~(uint64_t)63;
    int      blocks   = (int)(n_padded / 64);

    dim3 grid((unsigned)blocks);
    dim3 block(128);
    multifold_tc_kernel<<<grid, block, 0, stream>>>(
        d_z_mat, d_R_mat, d_output, num_instances, N_ring
    );
    return cudaGetLastError();
}

// ===========================================================================
// FUSED variant: skip the z_mat materialization, and batch the unpack +
// MMA work in K-chunks of 64 (= one full instance's ring-element width).
//
// Outer K loop: one iteration per instance i_inst ∈ [0, M).
//   (a) Unpack: each warp's 16 lanes read one u64 of z (binary) or two
//       u64s (ternary, pos+neg) for instance i_inst at their row j_my,
//       expand all 64 bits to 64 int8 ∈ {-1, 0, +1}, and store the
//       resulting 64 bytes to a per-warp 1 KB shared slot (16 rows × 64).
//   (b) MMA: 4 inner k-tiles (ell_off ∈ {0, 16, 32, 48}) × 4 N-tiles =
//       16 wmma operations per outer iter, all reading A from shared.
//
// Wins vs the BLOCK_K=16 variant:
//   * 4× fewer unpacks: the unpack work per row (read u64 + spread to
//     64 int8) is paid once per instance instead of four times.
//   * 4× fewer __syncwarp() barriers in the K-loop.
//   * The 16 MMAs per outer iter pipeline cleanly across the 4
//     independent c_frag[n] accumulator chains.
//
// HBM traffic:  M·N_ring u64 reads for binary, +M_tern·N_ring×2 u64 reads
// for ternary; R is read M·4·16 bytes per warp per outer iter from global
// (L2 cached, R is only ~258 KB at M=63). Output is 2·N_ring·64 bytes.
// No z_mat materialization.
// ===========================================================================

__global__ void
__launch_bounds__(128, 4)
multifold_tc_fused_kernel(
    const uint64_t* __restrict__ z_bin_packed,    // [K_bin  * N_ring]
    const uint64_t* __restrict__ pos_packed,      // [K_tern * N_ring]
    const uint64_t* __restrict__ neg_packed,      // [K_tern * N_ring]
    const int8_t*   __restrict__ R_mat,           // [K, 64] col-major (K = M·64)
    int16_t*        __restrict__ output,          // [N_ring, 64] row-major
    int                          num_binary,
    int                          num_ternary,
    uint64_t                     N_ring
) {
    using namespace nvcuda::wmma;
    int M_inst = num_binary + num_ternary;
    int K      = M_inst * 64;

    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int j_lo    = blockIdx.x * 64 + warp_id * 16;

    fragment<accumulator, 16, 16, 16, int32_t> c_frag[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) fill_fragment(c_frag[n], 0);

    // 4 warps × 1024 bytes = 4 KB for A staging (per-warp).
    __shared__ int8_t a_smem_buf[4 * 16 * 64];
    int8_t* my_a = a_smem_buf + warp_id * 16 * 64;

    // Shared R-instance cache: 64 rows × 64 cols col-major = 4 KB. Loaded
    // once per outer iter (one instance's full ring element), then all 4
    // warps' 16 inner mma's read b_frag from shared instead of L2.
    __shared__ int8_t R_smem[64 * 64];

    int       row        = lane;
    bool      row_active = row < 16;
    uint64_t  j_my       = (uint64_t)j_lo + (uint64_t)row;
    bool      row_valid  = row_active && (j_my < N_ring);

    int tid = threadIdx.x;

    for (int i_inst = 0; i_inst < M_inst; i_inst++) {

        // (a-1) Unpack one full instance (64 bits per row) → 64 int8 / row.
        if (row_active) {
            uint64_t p_word, n_word;
            if (i_inst < num_binary) {
                p_word = row_valid
                    ? z_bin_packed[(uint64_t)i_inst * N_ring + j_my] : 0;
                n_word = 0;
            } else {
                int      t = i_inst - num_binary;
                p_word = row_valid
                    ? pos_packed[(uint64_t)t * N_ring + j_my] : 0;
                n_word = row_valid
                    ? neg_packed[(uint64_t)t * N_ring + j_my] : 0;
            }

            uint64_t* my_a_u64 = reinterpret_cast<uint64_t*>(my_a + row * 64);
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                uint64_t packed = 0;
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    int    bit_idx = g * 8 + c;
                    int8_t v = (int8_t)((p_word >> bit_idx) & 1ULL)
                             - (int8_t)((n_word >> bit_idx) & 1ULL);
                    packed |= ((uint64_t)(uint8_t)v) << (c * 8);
                }
                my_a_u64[g] = packed;
            }
        }

        // (a-2) Cooperatively cache R[i_inst*64 : (i_inst+1)*64, 0:64] into
        // R_smem (col-major). 4096 bytes / 128 threads × 8 bytes = 4 int64
        // stores per thread.
        //
        // Layout in shared (col-major): R_smem[col*64 + row]  for col∈[0,64),
        // row∈[0,64). Each int64 covers 8 consecutive rows of one column.
        {
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int idx        = q * 128 + tid;      // 0..511
                int col        = idx >> 3;           // 0..63
                int row_chunk  = idx & 7;            // 0..7
                int byte_off   = row_chunk * 8;
                const int64_t* src = reinterpret_cast<const int64_t*>(
                    R_mat + (uint64_t)col * (uint64_t)K + i_inst * 64 + byte_off
                );
                int64_t* dst = reinterpret_cast<int64_t*>(
                    R_smem + col * 64 + byte_off
                );
                *dst = *src;
            }
        }
        __syncthreads();

        // (b) 4 inner k-tiles × 4 N-tiles = 16 MMAs per outer iter. Both
        // A and B fragments come from shared memory now.
        #pragma unroll
        for (int inner_k = 0; inner_k < 4; inner_k++) {
            int ell_off = inner_k * 16;
            fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
            load_matrix_sync(a_frag, my_a + ell_off, 64);

            #pragma unroll
            for (int n = 0; n < 4; n++) {
                fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
                // R_smem stride = 64 (col-major, 64 rows per col).
                load_matrix_sync(
                    b_frag,
                    R_smem + ell_off + (uint64_t)(n * 16) * 64,
                    64
                );
                mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
            }
        }
        __syncthreads();
    }

    // Write-out: identical to v1.
    __shared__ int32_t tile_smem[4 * 16 * 16];
    int32_t* warp_tile = tile_smem + warp_id * 256;

    #pragma unroll
    for (int n = 0; n < 4; n++) {
        store_matrix_sync(warp_tile, c_frag[n], 16, mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 256; idx += 32) {
            int row  = idx >> 4;
            int col  = idx & 15;
            uint64_t out_row = (uint64_t)j_lo + (uint64_t)row;
            if (out_row < N_ring) {
                output[out_row * 64 + (uint64_t)n * 16 + col]
                    = (int16_t)warp_tile[idx];
            }
        }
        __syncwarp();
    }
}

inline cudaError_t multifold_tc_fused_run(
    const uint64_t* d_z_bin_packed,
    const uint64_t* d_pos_packed,
    const uint64_t* d_neg_packed,
    const int8_t*   d_R_mat,
    int16_t*        d_output,
    int             num_binary,
    int             num_ternary,
    uint64_t        N_ring,
    cudaStream_t    stream = 0
) {
    uint64_t n_padded = (N_ring + 63) & ~(uint64_t)63;
    int      blocks   = (int)(n_padded / 64);

    dim3 grid((unsigned)blocks);
    dim3 block(128);
    multifold_tc_fused_kernel<<<grid, block, 0, stream>>>(
        d_z_bin_packed, d_pos_packed, d_neg_packed, d_R_mat, d_output,
        num_binary, num_ternary, N_ring
    );
    return cudaGetLastError();
}

} // namespace ajtai

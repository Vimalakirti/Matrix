/**
 * GPU Ajtai commitment c = M·z over R = F_q[X] / (X^64 + 1).
 *
 * Layout (production):
 *   q       = 2^64 - 2^32 - 31   (almost-Goldilocks)
 *   D       = 64                 (ring dimension)
 *   KAPPA   = 15                 (output rows)
 *
 * Three execution paths:
 *   commit_dense_batched_kernel<B, CHUNK>  -- dense + batched, the main path
 *   commit_sparse_partial_kernel<CHUNK>    -- sparse, single witness only
 *   reduce_partials_kernel                  -- shared stage-2 reduction
 *
 * Field arithmetic delegates to almost_goldilocks.cuh; the PRG is in
 * ajtai_chacha8.cuh.
 *
 * See ajtai.md for the full design rationale.
 */

#ifndef ALMOST_AJTAI_CUH
#define ALMOST_AJTAI_CUH

#include "almost_goldilocks.cuh"
#include "ajtai_chacha8.cuh"
#include <cstdlib>

namespace ajtai {

constexpr int D     = 64;
constexpr int KAPPA = 42;

// Row-group split used by the "halve-rows" dense kernel: each thread owns
// ROWS_PER_H accumulators instead of KAPPA, so per-thread register pressure is
// independent of KAPPA. NSPLIT groups cover all KAPPA rows (the last group is
// partial when KAPPA % ROWS_PER_H != 0).
constexpr int ROWS_PER_H = 8;
constexpr int NSPLIT     = (KAPPA + ROWS_PER_H - 1) / ROWS_PER_H;

// ============================================================================
// Per-thread helper: signed-coefficient add (negacyclic wrap-aware)
// ============================================================================

__device__ __forceinline__
uint64_t add_signed(uint64_t acc, uint64_t v, bool sub) {
    return sub
        ? agl_sub_no_canonicalize(acc, v)
        : agl_add_no_canonicalize(acc, v);
}

// ============================================================================
// Cooperative PRG: fill M_shared[KAPPA * D] for column j.
//
//   Total tasks  : KAPPA * 8 = 120 ChaCha blocks
//   Distribution : grid-stride over the n_threads in the block
// ============================================================================

__device__ __forceinline__
void prg_fill_column(
    const uint32_t key[8],
    uint64_t       j,
    uint64_t*      M_shared,   // KAPPA * D u64
    int            tid,
    int            n_threads
) {
    constexpr int TOTAL_TASKS = KAPPA * 8;  // 120

    for (int task = tid; task < TOTAL_TASKS; task += n_threads) {
        int row       = task / 8;
        int block_idx = task & 7;

        uint64_t buf[8];
        prg_ring_block_chacha8(key, (uint32_t)row, j, (uint32_t)block_idx, buf);

        int base = row * D + 8 * block_idx;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            M_shared[base + k] = buf[k];
        }
    }
}

// ============================================================================
// Dense batched commit kernel
//
//   gridDim  = (num_chunks,)
//   blockDim = D * B   (= 64*B, must be <= 1024 → B <= 16)
//
// Each thread t  ↔  (b = t/D, r = t%D)
// Each thread owns KAPPA u64 accumulators in registers (one per output row).
//
// Layout of `partial` is [chunk][b][i][r] flattened row-major.
// ============================================================================

template <int B, int CHUNK>
__global__ void
__launch_bounds__(B * D, 1)
commit_dense_batched_kernel(
    const uint32_t* __restrict__ chacha_key,        // 8 u32
    const uint64_t* __restrict__ z_bits_packed,     // [B * N]
    uint64_t*       __restrict__ partial,           // [num_chunks][B][KAPPA][D]
    uint64_t                     N,
    uint64_t                     col_offset         // column window start in M_max
) {
    int chunk = blockIdx.x;
    int t     = threadIdx.x;
    int b     = t / D;
    int r     = t & (D - 1);

    __shared__ uint32_t key_sh[8];
    __shared__ uint64_t M_sh[KAPPA * D];
    __shared__ uint64_t bits_sh[(B > 0) ? B : 1];

    if (t < 8) key_sh[t] = chacha_key[t];

    uint64_t acc[KAPPA];
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) acc[i] = 0;

    uint64_t j_lo = (uint64_t)chunk * CHUNK;
    uint64_t N_   = N;
    uint64_t j_hi = (j_lo + CHUNK < N_) ? (j_lo + CHUNK) : N_;

    __syncthreads();

    for (uint64_t j = j_lo; j < j_hi; j++) {

        // (a) load the B batch bitmasks for this j into shared
        if (t < B) {
            bits_sh[t] = z_bits_packed[(uint64_t)t * N_ + j];
        }

        // (b) cooperatively fill M_shared with M[*, j + col_offset]
        prg_fill_column(key_sh, j + col_offset, M_sh, t, (int)blockDim.x);
        __syncthreads();

        // (c) set-bit loop for this thread's batch index
        uint64_t mask = bits_sh[b];
        while (mask) {
            int  ell  = __ffsll((long long)mask) - 1;
            mask     &= mask - 1;

            int  idx_  = r - ell;
            bool wrap  = idx_ < 0;
            idx_      += wrap ? D : 0;

            #pragma unroll
            for (int i = 0; i < KAPPA; i++) {
                uint64_t v = M_sh[i * D + idx_];
                acc[i] = add_signed(acc[i], v, wrap);
            }
        }
        __syncthreads();
    }

    // (d) write partial[chunk][b][i][r]
    uint64_t base = ((uint64_t)chunk * B + b) * (KAPPA * D) + r;
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) {
        partial[base + i * D] = acc[i];
    }
}


// ============================================================================
// WIDE commit: C = sum_j M_j * z_j  for a witness with FULL-WIDTH field
// coefficients (as opposed to the binary / ternary set-bit kernels above).
//
// Motivation: the masked-RLC mask commitment D_l = L(U_l) has Gaussian
// coefficients of ~36 bits. Doing it as 36 binary plane-commits re-runs the
// ChaCha8 matrix PRG 36 times over the same columns; this kernel pays the PRG
// once and replaces the shifted-add inner loop with a modular multiply.
//
// Row tiling: thread t <-> (i_local = t / D, r = t % D) owns exactly ONE
// accumulator, for global output row `i = tile * ROW_TILE + i_local` and
// coefficient r. That is what makes the kernel independent of KAPPA -- the
// `acc[KAPPA]` register array in commit_dense_batched_kernel is why raising
// KAPPA to 42 spills, and this layout has no such array.
//
//   gridDim  = (num_chunks, ceil(KAPPA / ROW_TILE))
//   blockDim = ROW_TILE * D
//   shared   = ROW_TILE * D  (matrix rows) + D (witness column) u64
//
// `col_offset` shifts which columns of M_max are used, so a witness can be
// committed against an arbitrary aligned column window. That is what lets
// several polynomials be packed into one commitment: commit each block against
// its own window and ring-sum the results (the Ajtai map is linear).
// ============================================================================

__device__ __forceinline__
void prg_fill_column_rows(
    const uint32_t key[8],
    uint64_t       j,          // GLOBAL column index (already offset)
    int            row_lo,
    int            rows,       // fill rows [row_lo, row_lo + rows)
    uint64_t*      M_shared,   // [rows * D]
    int            tid,
    int            n_threads
) {
    const int total_tasks = rows * 8;
    for (int task = tid; task < total_tasks; task += n_threads) {
        int i_local   = task >> 3;
        int block_idx = task & 7;
        int row       = row_lo + i_local;
        if (row >= KAPPA) continue;

        uint64_t buf[8];
        prg_ring_block_chacha8(key, (uint32_t)row, j, (uint32_t)block_idx, buf);

        int base = i_local * D + 8 * block_idx;
        #pragma unroll
        for (int k = 0; k < 8; k++) M_shared[base + k] = buf[k];
    }
}

template <int ROW_TILE, int CHUNK>
__global__ void
__launch_bounds__(ROW_TILE * D, 1)
commit_wide_kernel(
    const uint32_t* __restrict__ chacha_key,   // 8 u32
    const uint64_t* __restrict__ z_wide,       // [N * D] canonical field elems
    uint64_t*       __restrict__ partial,      // [num_chunks][KAPPA][D]
    uint64_t                     N,            // columns (ring elements)
    uint64_t                     col_offset
) {
    const int chunk   = blockIdx.x;
    const int tile    = blockIdx.y;
    const int t       = threadIdx.x;
    const int i_local = t / D;
    const int r       = t & (D - 1);
    const int row     = tile * ROW_TILE + i_local;

    __shared__ uint32_t key_sh[8];
    __shared__ uint64_t M_sh[ROW_TILE * D];
    __shared__ uint64_t z_sh[D];

    if (t < 8) key_sh[t] = chacha_key[t];

    uint64_t acc = 0;

    const uint64_t j_lo = (uint64_t)chunk * CHUNK;
    const uint64_t j_hi = (j_lo + CHUNK < N) ? (j_lo + CHUNK) : N;

    __syncthreads();

    for (uint64_t j = j_lo; j < j_hi; j++) {
        // (a) this column's D witness coefficients
        if (t < D) z_sh[t] = z_wide[j * D + (uint64_t)t];

        // (b) matrix rows for this tile at the offset column
        prg_fill_column_rows(key_sh, j + col_offset, tile * ROW_TILE, ROW_TILE,
                             M_sh, t, (int)blockDim.x);
        __syncthreads();

        if (row < KAPPA) {
            const uint64_t* Mrow = &M_sh[i_local * D];
            // negacyclic: out[r] = sum_l M[r-l] * z[l], sign-flipped on wrap.
            // Split at l = r so the wrap test leaves the inner loop.
            #pragma unroll 8
            for (int l = 0; l <= r; l++) {
                uint64_t prod = agl_reduce128(agl_mul_u64_u64(Mrow[r - l], z_sh[l]));
                acc = agl_add_no_canonicalize(acc, prod);
            }
            #pragma unroll 8
            for (int l = r + 1; l < D; l++) {
                uint64_t prod = agl_reduce128(agl_mul_u64_u64(Mrow[r - l + D], z_sh[l]));
                acc = agl_sub_no_canonicalize(acc, prod);
            }
        }
        __syncthreads();
    }

    if (row < KAPPA) {
        partial[((uint64_t)chunk * KAPPA + (uint64_t)row) * D + (uint64_t)r] = acc;
    }
}

// ============================================================================
// Halve-rows variant for B ∈ {1, 2, 4, 8}
//
//   gridDim  = (num_chunks,)
//   blockDim = 2 * B * D   (= 128..1024)
//
// Thread layout: t  ↔  (h, b, r) with
//   h = t / (B * D)              -- which half of the output rows
//   b = (t / D) % B               -- which batched witness
//   r = t & (D - 1)               -- which coefficient slot
//
// Each thread holds 8 u64 accumulators (half the 15 rows). h=0 handles
// rows 0..7, h=1 handles rows 8..14 (with the 8th slot unused).
//
// vs the standard kernel:
//   • per-thread accumulator state: 8 u64 (16 regs) vs 15 u64 (30 regs)
//   • blockDim doubles, total work unchanged
//   • occupancy ≈ 2× the standard kernel at the same B
//
// Compiles only for B ∈ {1, 2, 4, 8}: at B = 16, blockDim would be 2048
// which exceeds the SM_80 hard limit of 1024 threads/block.
// ============================================================================

template <int B, int CHUNK>
__global__ void
__launch_bounds__(NSPLIT * B * D, 1)
commit_dense_batched_halve_kernel(
    const uint32_t* __restrict__ chacha_key,
    const uint64_t* __restrict__ z_bits_packed,
    uint64_t*       __restrict__ partial,
    uint64_t                     N,
    uint64_t                     col_offset
) {
    static_assert(B >= 1 && B <= 8, "halve-rows kernel requires B <= 8");
    static_assert(NSPLIT * B * D <= 1024, "blockDim exceeds 1024");

    int chunk = blockIdx.x;
    int t     = threadIdx.x;
    int h     = t / (B * D);
    int b     = (t / D) % B;
    int r     = t & (D - 1);
    int row_base  = h * ROWS_PER_H;

    __shared__ uint32_t key_sh[8];
    __shared__ uint64_t M_sh[KAPPA * D];
    __shared__ uint64_t bits_sh[(B > 0) ? B : 1];

    if (t < 8) key_sh[t] = chacha_key[t];

    uint64_t acc[ROWS_PER_H];          // last group may use fewer than all slots
    #pragma unroll
    for (int k = 0; k < ROWS_PER_H; k++) acc[k] = 0;

    uint64_t j_lo = (uint64_t)chunk * CHUNK;
    uint64_t N_   = N;
    uint64_t j_hi = (j_lo + CHUNK < N_) ? (j_lo + CHUNK) : N_;

    __syncthreads();

    for (uint64_t j = j_lo; j < j_hi; j++) {

        // (a) cooperatively load B bitmasks (only threads with t < B do work)
        if (t < B) {
            bits_sh[t] = z_bits_packed[(uint64_t)t * N_ + j];
        }

        // (b) cooperatively fill M_shared across all NSPLIT*B*D threads
        prg_fill_column(key_sh, j + col_offset, M_sh, t, (int)blockDim.x);
        __syncthreads();

        // (c) set-bit loop on this thread's batch bitmask
        uint64_t mask = bits_sh[b];
        while (mask) {
            int  ell  = __ffsll((long long)mask) - 1;
            mask     &= mask - 1;

            int  idx_  = r - ell;
            bool wrap  = idx_ < 0;
            idx_      += wrap ? D : 0;

            // ROWS_PER_H unrolled iterations; the final group is partial
            #pragma unroll
            for (int i_local = 0; i_local < ROWS_PER_H; i_local++) {
                int i = row_base + i_local;
                if (i < KAPPA) {
                    uint64_t v = M_sh[i * D + idx_];
                    acc[i_local] = add_signed(acc[i_local], v, wrap);
                }
            }
        }
        __syncthreads();
    }

    // (d) write partial[chunk][b][i][r] for this thread's row range
    uint64_t base = ((uint64_t)chunk * B + b) * (KAPPA * D) + r;
    #pragma unroll
    for (int i_local = 0; i_local < ROWS_PER_H; i_local++) {
        int i = row_base + i_local;
        if (i < KAPPA) {
            partial[base + i * D] = acc[i_local];
        }
    }
}

// (commit_dense_batched_halve_run is defined alongside the other host
// wrappers below, after reduce_partials_kernel is in scope.)

// ============================================================================
// Sparse single-commit kernel
//
//   gridDim  = (num_chunks,)
//   blockDim = D = 64
//
// Each thread t ∈ [0, 64) owns coefficient r = t of all KAPPA output rows.
// Each chunk handles CHUNK consecutive positions from `positions[]`.
// `partial` shape: [num_chunks][KAPPA][D].
// ============================================================================

template <int CHUNK>
__global__ void commit_sparse_partial_kernel(
    const uint32_t* __restrict__ chacha_key,
    const uint64_t* __restrict__ positions,
    uint64_t                     K,
    uint64_t*       __restrict__ partial
) {
    int chunk = blockIdx.x;
    int t     = threadIdx.x;            // 0..63 == r

    __shared__ uint32_t key_sh[8];
    __shared__ uint64_t M_sh[KAPPA * D];

    if (t < 8) key_sh[t] = chacha_key[t];

    // Lazy accumulation (lo + 2^64 * hi, one reduction at the end; a negacyclic
    // wrap adds q - v), as in commit_ternary_rows_kernel. With sorted positions
    // (the online commit sorts them) a one-hot 2^20-row chunk went 13.7 ->
    // 7.9 ms; the canonicalized commitment is identical. hi counts carries:
    // at most CHUNK adds per accumulator.
    uint64_t lo[KAPPA];
    uint32_t hi[KAPPA];
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) { lo[i] = 0; hi[i] = 0; }

    uint64_t p_lo = (uint64_t)chunk * CHUNK;
    uint64_t p_hi = (p_lo + CHUNK < K) ? (p_lo + CHUNK) : K;

    __syncthreads();

    // Regenerate column j only when it changes. Positions are sorted, so all
    // set bits of one ring column arrive together, and the previous version
    // re-ran the 336-block ChaCha8 fill for every one of them. One-hot lookup
    // chunks are the extreme case: a column is 64 consecutive rows of ONE
    // table entry, and in the high chunks of real activations most rows share
    // an entry, so a column can hold up to 64 set bits. The accumulation order
    // is unchanged, so the commitment is bit-identical.
    uint64_t prev_j = ~(uint64_t)0;
    for (uint64_t k = p_lo; k < p_hi; k++) {
        uint64_t p   = positions[k];
        uint64_t j   = p >> 6;
        int      ell = (int)(p & 63);

        if (j != prev_j) {          // uniform across the block (same k)
            __syncthreads();        // previous column's reads are done
            prg_fill_column(key_sh, j, M_sh, t, (int)blockDim.x);
            __syncthreads();
            prev_j = j;
        }

        int  idx_ = t - ell;
        bool wrap = idx_ < 0;
        idx_     += wrap ? D : 0;

        #pragma unroll
        for (int i = 0; i < KAPPA; i++) {
            const uint64_t v = M_sh[i * D + idx_];
            const uint64_t x = wrap ? (ALMOST_GOLDILOCKS_PRIME - v) : v;
            const uint64_t s2 = lo[i] + x;
            hi[i] += (s2 < x) ? 1u : 0u;
            lo[i] = s2;
        }
    }

    uint64_t base = ((uint64_t)chunk * KAPPA) * D + t;
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) {
        agl_uint128_t x; x.lo = lo[i]; x.hi = hi[i];
        partial[base + i * D] = agl_reduce128(x);
    }
}

// ============================================================================
// Reduce kernel: sum over `num_chunks` partials, canonicalize at the end.
//
// Dense batched form (B-aware):
//   gridDim = (B, KAPPA, D), blockDim = 256, shared = blockDim * sizeof(u64)
//
// Sparse form (single witness):
//   Use B = 1 below.
// ============================================================================

__global__ void reduce_partials_kernel(
    const uint64_t* __restrict__ partial,
    uint64_t*       __restrict__ c,        // [B][KAPPA][D]
    int                          B,
    uint64_t                     num_chunks
) {
    int b   = blockIdx.x;
    int i   = blockIdx.y;
    int r   = blockIdx.z;
    int tid = threadIdx.x;

    extern __shared__ uint64_t smem[];

    uint64_t acc = 0;
    for (uint64_t k = tid; k < num_chunks; k += blockDim.x) {
        uint64_t off = ((k * (uint64_t)B + (uint64_t)b) * KAPPA + (uint64_t)i) * D + (uint64_t)r;
        acc = agl_add_no_canonicalize(acc, partial[off]);
    }
    smem[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = agl_add_no_canonicalize(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        c[((uint64_t)b * KAPPA + (uint64_t)i) * D + (uint64_t)r] = agl_canonicalize(smem[0]);
    }
}

// ============================================================================
// Host wrappers: high-level commit() entry points that allocate scratch,
// launch the right (B, CHUNK) instantiation, and run stage 2.
//
// `out` must be a device buffer of size B * KAPPA * D u64s.
// `d_z_bits_packed` is a device buffer of size B * N u64s in row-major
// [b][j] order.
// ============================================================================


// Wide commit host wrapper. `d_z_wide` is [N * D] canonical field elements;
// `d_out` is [KAPPA * D]. ROW_TILE = 8 keeps blockDim at 512 and shared at
// 4.5 KB regardless of KAPPA.
template <int CHUNK>
inline cudaError_t commit_wide_run(
    const uint32_t* d_chacha_key,
    const uint64_t* d_z_wide,
    uint64_t        N,
    uint64_t        col_offset,
    uint64_t*       d_out,
    cudaStream_t    stream = 0
) {
    constexpr int ROW_TILE = 8;
    const int num_tiles = (KAPPA + ROW_TILE - 1) / ROW_TILE;

    uint64_t num_chunks = (N + CHUNK - 1) / CHUNK;
    uint64_t partial_count = num_chunks * (uint64_t)KAPPA * D;
    uint64_t* d_partial = nullptr;
    cudaError_t err = cudaMallocAsync(&d_partial, partial_count * sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;
    // Tiles whose rows are all >= KAPPA never write; zero so the reduction is
    // still well-defined for every (chunk, row, r).
    err = cudaMemsetAsync(d_partial, 0, partial_count * sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    dim3 grid1((unsigned)num_chunks, (unsigned)num_tiles);
    dim3 block1((unsigned)(ROW_TILE * D));
    commit_wide_kernel<ROW_TILE, CHUNK><<<grid1, block1, 0, stream>>>(
        d_chacha_key, d_z_wide, d_partial, N, col_offset
    );

    dim3 grid2(1u, (unsigned)KAPPA, (unsigned)D);
    int block2 = 256;
    size_t shared = (size_t)block2 * sizeof(uint64_t);
    reduce_partials_kernel<<<grid2, block2, shared, stream>>>(
        d_partial, d_out, 1, num_chunks
    );

    cudaError_t free_err = cudaFreeAsync(d_partial, stream);
    if (free_err != cudaSuccess) return free_err;
    return cudaGetLastError();
}

template <int B, int CHUNK>
inline cudaError_t commit_dense_batched_halve_run(
    const uint32_t* d_chacha_key,
    const uint64_t* d_z_bits_packed,
    uint64_t        N,
    uint64_t        col_offset,
    uint64_t*       d_out,
    cudaStream_t    stream = 0
) {
    static_assert(B >= 1 && B <= 8, "halve kernel supports B in {1,2,4,8}");

    uint64_t num_chunks = (N + CHUNK - 1) / CHUNK;
    uint64_t partial_count = num_chunks * (uint64_t)B * KAPPA * D;
    uint64_t* d_partial = nullptr;
    cudaError_t err = cudaMallocAsync(&d_partial, partial_count * sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    dim3 grid1((unsigned)num_chunks);
    dim3 block1((unsigned)(NSPLIT * B * D));
    commit_dense_batched_halve_kernel<B, CHUNK><<<grid1, block1, 0, stream>>>(
        d_chacha_key, d_z_bits_packed, d_partial, N, col_offset
    );

    dim3 grid2((unsigned)B, (unsigned)KAPPA, (unsigned)D);
    int block2 = 256;
    size_t shared = (size_t)block2 * sizeof(uint64_t);
    reduce_partials_kernel<<<grid2, block2, shared, stream>>>(
        d_partial, d_out, B, num_chunks
    );

    cudaError_t free_err = cudaFreeAsync(d_partial, stream);
    if (free_err != cudaSuccess) return free_err;
    return cudaGetLastError();
}

template <int B, int CHUNK>
inline cudaError_t commit_dense_batched_run(
    const uint32_t* d_chacha_key,
    const uint64_t* d_z_bits_packed,
    uint64_t        N,
    uint64_t        col_offset,
    uint64_t*       d_out,
    cudaStream_t    stream = 0
) {
    static_assert(B >= 1 && B <= 16, "B must be in [1, 16] for single-block kernel");
    static_assert(B * D <= 1024,     "B * D must be <= 1024 (A100 limit)");

    uint64_t num_chunks = (N + CHUNK - 1) / CHUNK;

    uint64_t partial_count = num_chunks * (uint64_t)B * KAPPA * D;
    uint64_t* d_partial = nullptr;
    cudaError_t err = cudaMallocAsync(&d_partial, partial_count * sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    dim3 grid1((unsigned)num_chunks);
    dim3 block1((unsigned)(B * D));
    commit_dense_batched_kernel<B, CHUNK><<<grid1, block1, 0, stream>>>(
        d_chacha_key, d_z_bits_packed, d_partial, N, col_offset
    );

    dim3 grid2((unsigned)B, (unsigned)KAPPA, (unsigned)D);
    int block2 = 256;
    size_t shared = (size_t)block2 * sizeof(uint64_t);
    reduce_partials_kernel<<<grid2, block2, shared, stream>>>(
        d_partial, d_out, B, num_chunks
    );

    cudaError_t free_err = cudaFreeAsync(d_partial, stream);
    if (free_err != cudaSuccess) return free_err;
    return cudaGetLastError();
}

// Pick the dense kernel that fits: the halve-rows variant needs
// NSPLIT * B * D threads per block, which at large KAPPA can exceed the 1024
// hard limit. When it does, fall back to the single-block kernel (which holds
// KAPPA accumulators per thread and therefore has higher register pressure but
// no blockDim constraint).
template <int B, int CHUNK>
inline cudaError_t commit_dense_dispatch_run(
    const uint32_t* d_chacha_key,
    const uint64_t* d_z_bits_packed,
    uint64_t        N,
    uint64_t        col_offset,
    uint64_t*       d_out,
    cudaStream_t    stream = 0
) {
    if constexpr (NSPLIT * B * D <= 1024) {
        return commit_dense_batched_halve_run<B, CHUNK>(
            d_chacha_key, d_z_bits_packed, N, col_offset, d_out, stream);
    } else {
        return commit_dense_batched_run<B, CHUNK>(
            d_chacha_key, d_z_bits_packed, N, col_offset, d_out, stream);
    }
}

template <int CHUNK>
inline cudaError_t commit_sparse_run(
    const uint32_t* d_chacha_key,
    const uint64_t* d_positions,
    uint64_t        K,
    uint64_t*       d_out,
    cudaStream_t    stream = 0
) {
    uint64_t num_chunks = (K + CHUNK - 1) / CHUNK;

    uint64_t partial_count = num_chunks * KAPPA * D;
    uint64_t* d_partial = nullptr;
    cudaError_t err = cudaMallocAsync(&d_partial, partial_count * sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    dim3 grid1((unsigned)num_chunks);
    dim3 block1((unsigned)D);
    commit_sparse_partial_kernel<CHUNK><<<grid1, block1, 0, stream>>>(
        d_chacha_key, d_positions, K, d_partial
    );

    dim3 grid2(1u, (unsigned)KAPPA, (unsigned)D);
    int block2 = 256;
    size_t shared = (size_t)block2 * sizeof(uint64_t);
    reduce_partials_kernel<<<grid2, block2, shared, stream>>>(
        d_partial, d_out, 1, num_chunks
    );

    cudaError_t free_err = cudaFreeAsync(d_partial, stream);
    if (free_err != cudaSuccess) return free_err;
    return cudaGetLastError();
}

// ============================================================================
// Folding primitives (additive homomorphism of Ajtai commitments)
//
// For a small-coefficient ring challenge r ∈ {-1, 0, 1, 2}^64:
//   • fold_witness   :  out[j]  = z1[j] + r · z2[j]   for j ∈ [0, N_ring)
//     z1, z2 are binary (one u64 packs 64 binary coefficients).
//     Output is N_ring · 64 u64s of F_q values; per-coefficient range is
//     small (|·| ≤ 65) but we store as F_q for downstream consumption.
//   • fold_commitment:  out[i] = c1[i] + r · c2[i]   for i ∈ [0, KAPPA)
//     c1, c2 ∈ R^KAPPA with arbitrary F_q coefficients.
//
// Both use selected negacyclic rotations (the same trick the commit kernel
// uses) — for fold_witness we iterate set bits of z2[j] and look up r[idx];
// for fold_commitment we iterate nonzero entries of r and look up c2[i][idx].
// ============================================================================

struct ChallengeR {
    // r[k] ∈ {-1, 0, 1, 2}. Passed by value as a kernel arg (64 bytes).
    int8_t coeffs[64];
};

// ---------------- fold_witness ----------------

template <int CHUNK>
__global__ void
__launch_bounds__(D, 16)
fold_witness_kernel(
    const uint64_t* __restrict__ z1_bits,   // [N_ring]
    const uint64_t* __restrict__ z2_bits,   // [N_ring]
    ChallengeR                   r,
    uint64_t*       __restrict__ output,    // [N_ring * D]
    uint64_t                     N_ring
) {
    int k = threadIdx.x;                    // 0..63 = coefficient slot

    // Load r into shared once per block (broadcast-friendly).
    __shared__ int8_t r_sh[D];
    r_sh[k] = r.coeffs[k];
    __syncthreads();

    uint64_t j_lo = (uint64_t)blockIdx.x * CHUNK;
    uint64_t j_hi = (j_lo + CHUNK < N_ring) ? (j_lo + CHUNK) : N_ring;

    for (uint64_t j = j_lo; j < j_hi; j++) {
        uint64_t z2 = z2_bits[j];
        uint64_t z1 = z1_bits[j];

        // Accumulate in i32 since the final coefficient is in [-64, +65].
        int32_t acc = (int32_t)((z1 >> k) & 1ULL);

        uint64_t mask = z2;
        while (mask) {
            int ell = __ffsll((long long)mask) - 1;
            mask &= mask - 1;
            int  idx_signed = k - ell;
            bool wrap       = idx_signed < 0;
            int  idx        = idx_signed + (wrap ? D : 0);
            int  rv         = (int)r_sh[idx];     // sign-extends to int
            if (wrap) rv = -rv;
            acc += rv;
        }

        // Convert i32 → F_q canonical
        uint64_t out_val;
        if (acc >= 0) {
            out_val = (uint64_t)acc;
        } else {
            out_val = ALMOST_GOLDILOCKS_PRIME - (uint64_t)(-acc);
        }
        output[j * D + k] = out_val;
    }
}

template <int CHUNK>
inline cudaError_t fold_witness_run(
    const uint64_t*   d_z1_bits,
    const uint64_t*   d_z2_bits,
    const ChallengeR& r,
    uint64_t          N_ring,
    uint64_t*         d_output,            // [N_ring * D]
    cudaStream_t      stream = 0
) {
    uint64_t num_chunks = (N_ring + CHUNK - 1) / CHUNK;
    dim3 grid((unsigned)num_chunks);
    dim3 block((unsigned)D);
    fold_witness_kernel<CHUNK><<<grid, block, 0, stream>>>(
        d_z1_bits, d_z2_bits, r, d_output, N_ring
    );
    return cudaGetLastError();
}

// ---------------- fold_commitment ----------------

__global__ void
fold_commitment_kernel(
    const uint64_t* __restrict__ c1,        // [KAPPA * D]
    const uint64_t* __restrict__ c2,        // [KAPPA * D]
    ChallengeR                   r,
    uint64_t*       __restrict__ output     // [KAPPA * D]
) {
    int i = blockIdx.x;                     // 0..KAPPA-1 = output row
    int k = threadIdx.x;                    // 0..D-1     = coefficient slot

    __shared__ int8_t  r_sh [D];
    __shared__ uint64_t c2_sh[D];
    r_sh [k] = r.coeffs[k];
    c2_sh[k] = c2[i * D + k];
    __syncthreads();

    uint64_t acc = c1[i * D + k];

    // r is sparse; iterate ell=0..63 and skip zeros (compiler unrolls
    // the small-loop body; #pragma unroll keeps the if/else stable).
    #pragma unroll
    for (int ell = 0; ell < D; ell++) {
        int8_t rv = r_sh[ell];
        if (rv == 0) continue;
        int  idx_signed = k - ell;
        bool wrap       = idx_signed < 0;
        int  idx        = idx_signed + (wrap ? D : 0);
        uint64_t c2_val = c2_sh[idx];
        int rv_signed = wrap ? -(int)rv : (int)rv;

        // rv_signed ∈ {-2, -1, 0, 1, 2}
        if (rv_signed == 1) {
            acc = agl_add_no_canonicalize(acc, c2_val);
        } else if (rv_signed == -1) {
            acc = agl_sub_no_canonicalize(acc, c2_val);
        } else if (rv_signed == 2) {
            uint64_t two = agl_add_no_canonicalize(c2_val, c2_val);
            acc = agl_add_no_canonicalize(acc, two);
        } else if (rv_signed == -2) {
            uint64_t two = agl_add_no_canonicalize(c2_val, c2_val);
            acc = agl_sub_no_canonicalize(acc, two);
        }
    }

    output[i * D + k] = agl_canonicalize(acc);
}

inline cudaError_t fold_commitment_run(
    const uint64_t*   d_c1,                 // [KAPPA * D]
    const uint64_t*   d_c2,                 // [KAPPA * D]
    const ChallengeR& r,
    uint64_t*         d_output,             // [KAPPA * D]
    cudaStream_t      stream = 0
) {
    dim3 grid((unsigned)KAPPA);
    dim3 block((unsigned)D);
    fold_commitment_kernel<<<grid, block, 0, stream>>>(
        d_c1, d_c2, r, d_output
    );
    return cudaGetLastError();
}

// ============================================================================
// Multi-fold: K + k binary instances in one pass
//
// Following SuperNeo's Almost-Goldilocks parameters (K = 50 fresh + k = 13
// accumulator), we fold up to ~63 binary instances z_1, ..., z_M with
// matching challenges r_1, ..., r_M into a single non-binary witness
//
//     z' = sum_i r_i · z_i
//
// and likewise for commitments. With M = 63, T = 128, b = 2:
//   ||z'||_inf  <=  M * T * (b - 1)  =  63 * 128 * 1  =  8064  <  2^13 = B
//
// so each output coefficient fits in an i16 (range [-8192, 8191]).
//
// Two kernels:
//   multifold_witness_kernel    -- M binary inputs, i16 output (wide witness)
//   multifold_commitment_kernel -- M ring inputs, F_q output (R^KAPPA)
//
// Both pass the challenges as a [M * 64] int8 device array (loaded into
// shared mem once per block), and the inputs as a single contiguous device
// buffer (witnesses as [M * N_ring] packed u64; commitments as [M * KAPPA * D]
// flat u64).
// ============================================================================

__global__ void
multifold_witness_kernel(
    const uint64_t* __restrict__ z_packed,   // [num_instances * N_ring]
    const int8_t*   __restrict__ r_all,      // [num_instances * 64]
    int16_t*        __restrict__ output,     // [N_ring * D]
    int                          num_instances,
    uint64_t                     N_ring,
    uint64_t                     chunk_size
) {
    int k = threadIdx.x;                     // 0..63 — output coefficient slot

    extern __shared__ unsigned char dynsmem[];
    int8_t*   r_sh        = (int8_t*)dynsmem;
    // Align z_bits_sh to 8 bytes after r_sh.
    size_t    r_sh_bytes  = ((size_t)num_instances * 64 + 7) & ~(size_t)7;
    uint64_t* z_bits_sh   = (uint64_t*)(dynsmem + r_sh_bytes);

    // Cooperatively load all challenges into shared (once per block).
    for (int idx = threadIdx.x; idx < num_instances * 64; idx += (int)blockDim.x) {
        r_sh[idx] = r_all[idx];
    }
    __syncthreads();

    uint64_t j_lo = (uint64_t)blockIdx.x * chunk_size;
    uint64_t j_hi = (j_lo + chunk_size < N_ring) ? (j_lo + chunk_size) : N_ring;

    for (uint64_t j = j_lo; j < j_hi; j++) {

        // Cooperatively load num_instances u64 bitmasks (one per instance).
        for (int i = threadIdx.x; i < num_instances; i += (int)blockDim.x) {
            z_bits_sh[i] = z_packed[(uint64_t)i * N_ring + j];
        }
        __syncthreads();

        int32_t acc = 0;

        // Sum over instances; within each instance use the selected-rotation
        // identity (X^ell · z_i contributes r_i[(k-ell)%64] with sign).
        for (int i = 0; i < num_instances; i++) {
            uint64_t bits = z_bits_sh[i];
            while (bits) {
                int  ell        = __ffsll((long long)bits) - 1;
                bits           &= bits - 1;
                int  idx_signed = k - ell;
                bool wrap       = idx_signed < 0;
                int  idx        = idx_signed + (wrap ? D : 0);
                int  rv         = (int)r_sh[i * 64 + idx];
                if (wrap) rv = -rv;
                acc += rv;
            }
        }

        // |acc| <= num_instances * 128 ≤ ~8000 for typical M=63, fits in i16.
        output[j * D + k] = (int16_t)acc;

        __syncthreads();
    }
}

inline cudaError_t multifold_witness_run(
    const uint64_t* d_z_packed,
    const int8_t*   d_r_all,
    int16_t*        d_output,
    int             num_instances,
    uint64_t        N_ring,
    uint64_t        chunk_size,
    cudaStream_t    stream = 0
) {
    uint64_t num_chunks = (N_ring + chunk_size - 1) / chunk_size;
    size_t r_bytes = ((size_t)num_instances * 64 + 7) & ~(size_t)7;
    size_t z_bytes = (size_t)num_instances * sizeof(uint64_t);
    size_t shared  = r_bytes + z_bytes;

    dim3 grid((unsigned)num_chunks);
    dim3 block((unsigned)D);
    multifold_witness_kernel<<<grid, block, shared, stream>>>(
        d_z_packed, d_r_all, d_output, num_instances, N_ring, chunk_size
    );
    return cudaGetLastError();
}

// ============================================================================
// Mixed-type multifold witness: K binary instances + T ternary chunks.
//
// Closes the prover loop: after split, the accumulator is 13 ternary chunks
// stored as (pos, neg) bitmask pairs. The next round folds those alongside
// K=50 fresh binary instances using K+T-1 ring challenges (binary[0] has
// implicit weight 1, encoded as a constant-1 challenge in r_all[0..64]).
//
// Output coefficient at (j, k) is:
//   acc(j, k) = Σ_{b<K} Σ_{ℓ : bin[b][j][ℓ]=1}   sign(b, k, ℓ) · r_bin_b[(k-ℓ) mod 64]
//             + Σ_{t<T} Σ_{ℓ : pos[t][j][ℓ]=1}   sign(t, k, ℓ) · r_tern_t[(k-ℓ) mod 64]
//             − Σ_{t<T} Σ_{ℓ : neg[t][j][ℓ]=1}   sign(t, k, ℓ) · r_tern_t[(k-ℓ) mod 64]
// where sign() applies negacyclic wrap negation.
//
// r_all layout: [bin_0, bin_1, ..., bin_{K-1}, tern_0, ..., tern_{T-1}],
// each block is 64 int8 coefficients ∈ {-1, 0, 1, 2}. r_all[0..64] must be
// the constant-1 challenge (1, 0, 0, …) to give bin[0] weight 1.
//
// |acc| bound at K=50, T=13, b=2, T_norm=128 (SuperNeo Almost-Goldilocks):
//   per instance ≤ popcount · 2 ≤ 128
//   total ≤ (K + T) · 128 = 63·128 = 8064 < 2^13, fits in i16.
// ============================================================================

__global__ void
multifold_mixed_witness_kernel(
    const uint64_t* __restrict__ z_bin_packed,   // [num_binary  * N_ring]
    const uint64_t* __restrict__ pos_packed,     // [num_ternary * N_ring]
    const uint64_t* __restrict__ neg_packed,     // [num_ternary * N_ring]
    const int8_t*   __restrict__ r_all,          // [(num_binary + num_ternary) * 64]
    int16_t*        __restrict__ output,         // [N_ring * D]
    int                          num_binary,
    int                          num_ternary,
    uint64_t                     N_ring,
    uint64_t                     chunk_size
) {
    int k = threadIdx.x;                         // 0..63 — output coefficient slot
    int num_instances = num_binary + num_ternary;

    extern __shared__ unsigned char dynsmem[];
    int8_t*   r_sh        = (int8_t*)dynsmem;
    size_t    r_sh_bytes  = ((size_t)num_instances * 64 + 7) & ~(size_t)7;
    // z_sh layout: [num_binary bin masks] || [num_ternary pos] || [num_ternary neg].
    uint64_t* z_sh        = (uint64_t*)(dynsmem + r_sh_bytes);

    // Cooperatively load all challenges (constants for the block lifetime).
    for (int idx = threadIdx.x; idx < num_instances * 64; idx += (int)blockDim.x) {
        r_sh[idx] = r_all[idx];
    }
    __syncthreads();

    uint64_t j_lo = (uint64_t)blockIdx.x * chunk_size;
    uint64_t j_hi = (j_lo + chunk_size < N_ring) ? (j_lo + chunk_size) : N_ring;

    for (uint64_t j = j_lo; j < j_hi; j++) {

        // (a) Cooperatively load all bitmasks for this j.
        for (int i = threadIdx.x; i < num_binary; i += (int)blockDim.x) {
            z_sh[i] = z_bin_packed[(uint64_t)i * N_ring + j];
        }
        for (int t = threadIdx.x; t < num_ternary; t += (int)blockDim.x) {
            z_sh[num_binary + t]               = pos_packed[(uint64_t)t * N_ring + j];
            z_sh[num_binary + num_ternary + t] = neg_packed[(uint64_t)t * N_ring + j];
        }
        __syncthreads();

        int32_t acc = 0;

        // (b) Binary instances: selected-rotation accumulation (same as
        // multifold_witness_kernel).
        for (int i = 0; i < num_binary; i++) {
            uint64_t bits = z_sh[i];
            int r_off = i * 64;
            while (bits) {
                int  ell        = __ffsll((long long)bits) - 1;
                bits           &= bits - 1;
                int  idx_signed = k - ell;
                bool wrap       = idx_signed < 0;
                int  idx        = idx_signed + (wrap ? D : 0);
                int  rv         = (int)r_sh[r_off + idx];
                if (wrap) rv = -rv;
                acc += rv;
            }
        }

        // (c) Ternary instances: pos bits contribute +r, neg bits contribute -r.
        for (int t = 0; t < num_ternary; t++) {
            int      r_off = (num_binary + t) * 64;
            uint64_t mp    = z_sh[num_binary + t];
            uint64_t mn    = z_sh[num_binary + num_ternary + t];

            while (mp) {
                int  ell        = __ffsll((long long)mp) - 1;
                mp             &= mp - 1;
                int  idx_signed = k - ell;
                bool wrap       = idx_signed < 0;
                int  idx        = idx_signed + (wrap ? D : 0);
                int  rv         = (int)r_sh[r_off + idx];
                if (wrap) rv = -rv;
                acc += rv;
            }
            while (mn) {
                int  ell        = __ffsll((long long)mn) - 1;
                mn             &= mn - 1;
                int  idx_signed = k - ell;
                bool wrap       = idx_signed < 0;
                int  idx        = idx_signed + (wrap ? D : 0);
                int  rv         = (int)r_sh[r_off + idx];
                if (wrap) rv = -rv;
                acc -= rv;
            }
        }

        output[j * D + k] = (int16_t)acc;
        __syncthreads();
    }
}

inline cudaError_t multifold_mixed_witness_run(
    const uint64_t* d_z_bin_packed,
    const uint64_t* d_pos_packed,
    const uint64_t* d_neg_packed,
    const int8_t*   d_r_all,
    int16_t*        d_output,
    int             num_binary,
    int             num_ternary,
    uint64_t        N_ring,
    uint64_t        chunk_size,
    cudaStream_t    stream = 0
) {
    int      num_instances = num_binary + num_ternary;
    uint64_t num_chunks    = (N_ring + chunk_size - 1) / chunk_size;
    size_t   r_bytes       = ((size_t)num_instances * 64 + 7) & ~(size_t)7;
    size_t   z_bytes       = (size_t)(num_binary + 2 * num_ternary) * sizeof(uint64_t);
    size_t   shared        = r_bytes + z_bytes;

    dim3 grid((unsigned)num_chunks);
    dim3 block((unsigned)D);
    multifold_mixed_witness_kernel<<<grid, block, shared, stream>>>(
        d_z_bin_packed, d_pos_packed, d_neg_packed, d_r_all, d_output,
        num_binary, num_ternary, N_ring, chunk_size
    );
    return cudaGetLastError();
}

__global__ void
multifold_commitment_kernel(
    const uint64_t* __restrict__ c_all,      // [num_instances * KAPPA * D]
    const int8_t*   __restrict__ r_all,      // [num_instances * 64]
    int                          num_instances,
    uint64_t*       __restrict__ output      // [KAPPA * D]
) {
    int i_row = blockIdx.x;                  // 0..KAPPA-1
    int k     = threadIdx.x;                 // 0..D-1

    extern __shared__ unsigned char dynsmem[];
    int8_t* r_sh = (int8_t*)dynsmem;

    // Cooperatively load r_all into shared (once per block).
    for (int idx = threadIdx.x; idx < num_instances * 64; idx += (int)blockDim.x) {
        r_sh[idx] = r_all[idx];
    }
    __syncthreads();

    uint64_t acc = 0;

    for (int i = 0; i < num_instances; i++) {
        const uint64_t* c2 = &c_all[((uint64_t)i * KAPPA + i_row) * D];

        // Iterate ell = 0..63 of r_i — sparse (skip zeros). Same selected-
        // rotation pattern as fold_commitment_kernel, just summed over i.
        for (int ell = 0; ell < D; ell++) {
            int8_t rv = r_sh[i * 64 + ell];
            if (rv == 0) continue;

            int  idx_signed = k - ell;
            bool wrap       = idx_signed < 0;
            int  idx        = idx_signed + (wrap ? D : 0);
            uint64_t c2_val = c2[idx];
            int rv_signed = wrap ? -(int)rv : (int)rv;

            if (rv_signed == 1) {
                acc = agl_add_no_canonicalize(acc, c2_val);
            } else if (rv_signed == -1) {
                acc = agl_sub_no_canonicalize(acc, c2_val);
            } else if (rv_signed == 2) {
                uint64_t two = agl_add_no_canonicalize(c2_val, c2_val);
                acc = agl_add_no_canonicalize(acc, two);
            } else if (rv_signed == -2) {
                uint64_t two = agl_add_no_canonicalize(c2_val, c2_val);
                acc = agl_sub_no_canonicalize(acc, two);
            }
        }
    }

    output[i_row * D + k] = agl_canonicalize(acc);
}

inline cudaError_t multifold_commitment_run(
    const uint64_t* d_c_all,
    const int8_t*   d_r_all,
    int             num_instances,
    uint64_t*       d_output,
    cudaStream_t    stream = 0
) {
    dim3 grid((unsigned)KAPPA);
    dim3 block((unsigned)D);
    size_t shared = (size_t)num_instances * 64;
    multifold_commitment_kernel<<<grid, block, shared, stream>>>(
        d_c_all, d_r_all, num_instances, d_output
    );
    return cudaGetLastError();
}

// ============================================================================
// Split: i16 wide witness → 13 ternary chunks (pos / neg bitmask pairs)
//
// Implements SuperNeo's split_b for b = 2, k = 13 on the Almost-Goldilocks
// parameter set. Each i16 coefficient v of the wide witness is decomposed as
//
//     v  =  Σ_{i=0..12}  2^i · (pos_bit_i  −  neg_bit_i)
//
// where (pos_bit_i, neg_bit_i) ∈ {0, 1}² with NO simultaneous 1s. Encoding:
//   |v| in binary gives the magnitudes of the 13 bits.
//   sign(v) routes each set bit to either pos_chunks[i][j] or neg_chunks[i][j].
//
// Output layout for downstream commit / multifold kernels:
//   pos_chunks[i * N_ring + j]  — u64, bit k set iff bit_i(|v_jk|)=1 AND v_jk > 0
//   neg_chunks[i * N_ring + j]  — u64, bit k set iff bit_i(|v_jk|)=1 AND v_jk < 0
// (same packing as the binary witness format, just doubled for the sign axis)
//
// Thread layout: one thread per ring element j (so one thread processes all
// 64 coefficients of one j). Block size 256.
// ============================================================================

constexpr int SPLIT_K = 13;

// __launch_bounds__(256, 2): allow up to 128 regs/thread (2 blocks/SM at
// 256 threads/block). The 26 u64 acc registers + temporaries fit comfortably
// without spills, and the memory-bound kernel doesn't care about higher
// occupancy than ~2 blocks/SM.
__global__ void
__launch_bounds__(256, 2)
split_witness_kernel(
    const int16_t* __restrict__ z_wide,         // [N_ring * 64]
    uint64_t*       __restrict__ pos_chunks,    // [SPLIT_K * N_ring]
    uint64_t*       __restrict__ neg_chunks,    // [SPLIT_K * N_ring]
    uint64_t                     N_ring
) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_ring) return;

    // 13 + 13 = 26 u64 register accumulators per thread.
    uint64_t pos[SPLIT_K];
    uint64_t neg[SPLIT_K];
    #pragma unroll
    for (int i = 0; i < SPLIT_K; i++) { pos[i] = 0; neg[i] = 0; }

    const int16_t* z_ptr = &z_wide[j * 64];

    // For each coefficient k, extract |v_k| bits and route via sign.
    #pragma unroll
    for (int k = 0; k < 64; k++) {
        int v       = (int)z_ptr[k];
        int v_abs   = (v >= 0) ? v : -v;
        bool is_neg = (v < 0);
        uint64_t bit_at_k = 1ULL << k;

        // Branch is OUTSIDE the chunk loop: predictable + non-divergent
        // when sign distribution is locally similar across coefficients
        // (mild divergence at worst — k loops are unrolled anyway).
        if (is_neg) {
            #pragma unroll
            for (int i = 0; i < SPLIT_K; i++) {
                uint64_t b = (uint64_t)((v_abs >> i) & 1);
                neg[i] |= b * bit_at_k;          // branchless: 0 or bit_at_k
            }
        } else {
            #pragma unroll
            for (int i = 0; i < SPLIT_K; i++) {
                uint64_t b = (uint64_t)((v_abs >> i) & 1);
                pos[i] |= b * bit_at_k;
            }
        }
    }

    // Coalesced writes: warps of 32 consecutive j threads each write a
    // 256-byte contiguous region within pos_chunks[i*N_ring..] / neg_chunks[i*N_ring..].
    #pragma unroll
    for (int i = 0; i < SPLIT_K; i++) {
        pos_chunks[(uint64_t)i * N_ring + j] = pos[i];
        neg_chunks[(uint64_t)i * N_ring + j] = neg[i];
    }
}

inline cudaError_t split_witness_run(
    const int16_t* d_z_wide,
    uint64_t*      d_pos_chunks,
    uint64_t*      d_neg_chunks,
    uint64_t       N_ring,
    cudaStream_t   stream = 0
) {
    int block = 256;
    int grid  = (int)((N_ring + block - 1) / block);
    split_witness_kernel<<<grid, block, 0, stream>>>(
        d_z_wide, d_pos_chunks, d_neg_chunks, N_ring
    );
    return cudaGetLastError();
}

// ============================================================================
// Ternary commit kernel (13 chunks, shared M, additive + subtractive passes)
//
// Inputs (device-resident):
//   pos_packed[i * N_ring + j] : bitmask of {ℓ : z_i[j][ℓ] = +1}
//   neg_packed[i * N_ring + j] : bitmask of {ℓ : z_i[j][ℓ] = -1}
//   (split_witness_kernel guarantees pos & neg == 0 — disjoint by construction)
//
// Output: 13 ring commitments c_0..c_12 where
//   c_i = Σ_j M[*,j] · z_i[j],     z_i ∈ {-1, 0, +1}^{64}
// PRG cost is identical to one binary commit — the M_shared fill is amortized
// across all 13 chunks. Total inner-loop work per j is
//   Σ_i (popcount(pos[i][j]) + popcount(neg[i][j])).
// Since each (j, ℓ) coefficient appears in at most popcount(2^13 - 1) = 13
// of the 13 chunks (in fact exactly once per nonzero coefficient by the
// digit decomposition), the average ternary loop count is the same as
// the binary loop count of the wide witness — no extra add work.
//
// Layout:
//   gridDim  = (num_chunks_j,)
//   blockDim = SPLIT_K * D = 13 * 64 = 832    (under A100's 1024/block cap)
// Each thread t = (b, r) with b = t/D ∈ [0, 13), r = t & 63.
// ============================================================================

template <int CHUNK>
__global__ void
__launch_bounds__(SPLIT_K * D, 1)
commit_ternary_kernel(
    const uint32_t* __restrict__ chacha_key,        // 8 u32
    const uint64_t* __restrict__ pos_packed,        // [SPLIT_K * N_ring]
    const uint64_t* __restrict__ neg_packed,        // [SPLIT_K * N_ring]
    uint64_t*       __restrict__ partial,           // [num_chunks_j][SPLIT_K][KAPPA][D]
    uint64_t                     N_ring
) {
    int chunk = blockIdx.x;
    int t     = threadIdx.x;
    int b     = t / D;                              // 0..12 (chunk index)
    int r     = t & (D - 1);                        // 0..63 (coef slot)

    __shared__ uint32_t key_sh[8];
    __shared__ uint64_t M_sh[KAPPA * D];
    __shared__ uint64_t pos_sh[SPLIT_K];
    __shared__ uint64_t neg_sh[SPLIT_K];

    if (t < 8) key_sh[t] = chacha_key[t];

    uint64_t acc[KAPPA];
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) acc[i] = 0;

    uint64_t j_lo = (uint64_t)chunk * CHUNK;
    uint64_t j_hi = (j_lo + CHUNK < N_ring) ? (j_lo + CHUNK) : N_ring;

    __syncthreads();

    for (uint64_t j = j_lo; j < j_hi; j++) {

        // (a) load 13 pos + 13 neg bitmasks for this j (cooperatively).
        if (t < SPLIT_K) {
            pos_sh[t] = pos_packed[(uint64_t)t * N_ring + j];
            neg_sh[t] = neg_packed[(uint64_t)t * N_ring + j];
        }

        // (b) cooperatively fill M_sh[*, j] across all 832 threads.
        prg_fill_column(key_sh, j, M_sh, t, (int)blockDim.x);
        __syncthreads();

        // (c) additive pass: bits in pos_sh[b] each contribute +X^ℓ · M.
        uint64_t mp = pos_sh[b];
        while (mp) {
            int  ell  = __ffsll((long long)mp) - 1;
            mp       &= mp - 1;

            int  idx_  = r - ell;
            bool wrap  = idx_ < 0;
            idx_      += wrap ? D : 0;

            #pragma unroll
            for (int i = 0; i < KAPPA; i++) {
                uint64_t v = M_sh[i * D + idx_];
                acc[i] = add_signed(acc[i], v, wrap);
            }
        }

        // (d) subtractive pass: bits in neg_sh[b] each contribute -X^ℓ · M.
        //     The negacyclic sign flips, so feed `!wrap` to add_signed.
        uint64_t mn = neg_sh[b];
        while (mn) {
            int  ell  = __ffsll((long long)mn) - 1;
            mn       &= mn - 1;

            int  idx_  = r - ell;
            bool wrap  = idx_ < 0;
            idx_      += wrap ? D : 0;

            #pragma unroll
            for (int i = 0; i < KAPPA; i++) {
                uint64_t v = M_sh[i * D + idx_];
                acc[i] = add_signed(acc[i], v, !wrap);
            }
        }
        __syncthreads();
    }

    // (e) write partial[chunk][b][i][r]
    uint64_t base = ((uint64_t)chunk * SPLIT_K + b) * (KAPPA * D) + r;
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) {
        partial[base + i * D] = acc[i];
    }
}

// ============================================================================
// Row-split ternary commit kernel (default; AGL_TERNARY_ROWS=0 for the above).
//
// commit_ternary_kernel keeps all KAPPA = 42 u64 accumulators per thread at
// 832 threads per block, i.e. 84 registers of accumulators against a 72-register
// cap: ptxas spills 920 B per thread and every add round-trips local memory.
// Measured 1017 ms for N_ring = 2^18 (log_n 24), ~0.16 T modular adds/s.
//
// Here blockIdx.y selects a group of TROWS output rows, so a thread holds
// TROWS lazy accumulators (no spill) and a block fills only its rows of the PRG
// column (TROWS * 8 ChaCha8 blocks instead of KAPPA * 8: total PRG work is
// KAPPA rounded up to TROWS, ~14% more). The 4 KB tile lets several blocks
// share an SM, which also hides the idle warps of digits with few set bits.
//
// Output layout and the reduction are unchanged. The reduction canonicalizes,
// so the different association of the (mod q) additions gives a bit-identical
// commitment.
// ============================================================================
constexpr int TROWS   = 8;
constexpr int TGROUPS = (KAPPA + TROWS - 1) / TROWS;

template <int CHUNK>
__global__ void
__launch_bounds__(SPLIT_K * D, 1)
commit_ternary_rows_kernel(
    const uint32_t* __restrict__ chacha_key,
    const uint64_t* __restrict__ pos_packed,        // [SPLIT_K * N_ring]
    const uint64_t* __restrict__ neg_packed,        // [SPLIT_K * N_ring]
    uint64_t*       __restrict__ partial,           // [num_chunks_j][SPLIT_K][KAPPA][D]
    uint64_t                     N_ring
) {
    const int chunk = blockIdx.x;
    const int i0    = blockIdx.y * TROWS;
    const int nrows = (KAPPA - i0 < TROWS) ? (KAPPA - i0) : TROWS;
    const int t     = threadIdx.x;
    const int b     = t / D;
    const int r     = t & (D - 1);

    __shared__ uint32_t key_sh[8];
    __shared__ uint64_t M_sh[TROWS * D];
    __shared__ uint64_t pos_sh[SPLIT_K];
    __shared__ uint64_t neg_sh[SPLIT_K];

    if (t < 8) key_sh[t] = chacha_key[t];

    // Lazy accumulation: lo + 2^64 * hi, reduced once at the end. A signed
    // term -v is added as q - v (M entries are canonical, so q - v is in
    // (0, q]), which removes both the per-add modular corrections and the
    // lane-divergent add/sub branch of add_signed. hi counts carries out of
    // lo; a thread performs at most CHUNK * 64 adds per row, far below 2^32.
    // Microbenchmark (N_ring = 2^18, 128 nonzeros/column): 294 -> 210 ms, and
    // the canonicalized partials are identical.
    uint64_t lo[TROWS];
    uint32_t hi[TROWS];
    #pragma unroll
    for (int i = 0; i < TROWS; i++) { lo[i] = 0; hi[i] = 0; }

    const uint64_t j_lo = (uint64_t)chunk * CHUNK;
    const uint64_t j_hi = (j_lo + CHUNK < N_ring) ? (j_lo + CHUNK) : N_ring;

    __syncthreads();

    for (uint64_t j = j_lo; j < j_hi; j++) {
        if (t < SPLIT_K) {
            pos_sh[t] = pos_packed[(uint64_t)t * N_ring + j];
            neg_sh[t] = neg_packed[(uint64_t)t * N_ring + j];
        }
        for (int task = t; task < nrows * 8; task += (int)blockDim.x) {
            const int lr = task >> 3;
            const int bi = task & 7;
            uint64_t buf[8];
            prg_ring_block_chacha8(key_sh, (uint32_t)(i0 + lr), j, (uint32_t)bi, buf);
            #pragma unroll
            for (int k = 0; k < 8; k++) M_sh[lr * D + 8 * bi + k] = buf[k];
        }
        __syncthreads();

        // One pass over pos | neg (disjoint): the sign of a term is its
        // negative-digit bit XOR the negacyclic wrap.
        const uint64_t mn = neg_sh[b];
        uint64_t m = pos_sh[b] | mn;
        while (m) {
            const int ell = __ffsll((long long)m) - 1;
            m &= m - 1;
            int  idx_ = r - ell;
            const bool wrap = idx_ < 0;
            idx_ += wrap ? D : 0;
            const bool neg = (((mn >> ell) & 1ull) != 0) ^ wrap;
            #pragma unroll
            for (int i = 0; i < TROWS; i++) {
                if (i < nrows) {
                    const uint64_t v = M_sh[i * D + idx_];
                    const uint64_t x = neg ? (ALMOST_GOLDILOCKS_PRIME - v) : v;
                    const uint64_t s2 = lo[i] + x;
                    hi[i] += (s2 < x) ? 1u : 0u;
                    lo[i] = s2;
                }
            }
        }
        __syncthreads();
    }

    const uint64_t base = ((uint64_t)chunk * SPLIT_K + b) * (KAPPA * D) + r;
    #pragma unroll
    for (int i = 0; i < TROWS; i++) {
        if (i < nrows) {
            agl_uint128_t x; x.lo = lo[i]; x.hi = hi[i];
            partial[base + (uint64_t)(i0 + i) * D] = agl_reduce128(x);
        }
    }
}

// ============================================================================
// Tensor-core ternary commit (default; AGL_TERNARY_TC=0 restores the lazy
// row-split kernel above).
//
// For one matrix column j, the 13 digit commitments are one integer GEMM:
//   C[(i, r), b] += sum_ell A_j[(i, r), ell] * Z_j[ell, b],
// A_j = the negacyclic Toeplitz expansion of the column's KAPPA ring elements
// (A[r][ell] = M_i[r - ell], negated on wrap), Z_j = the ternary digits as s8.
// A's 64-bit field entries are split into 8 byte planes; each plane is an
// exact u8 x s8 -> s32 m16n8k32 MMA, and the planes recombine exactly as
// sum_p 2^(8p) C_p before one reduction mod q. Exact integer arithmetic, so
// the commitment is bit-identical after the final canonicalization.
// s32 accumulators: |C_p| <= 255 * 64 * cols < 2^31 for cols <= 131072 columns per block.
//
// Microbenchmark (A100, N_ring = 2^18): 50.6 ms vs 210-331 ms for the lazy
// kernel, independent of digit density.
// ============================================================================
constexpr int TC_ROWS = 2;                 // KAPPA rows per block (42 = 21 groups)
static_assert(KAPPA % TC_ROWS == 0, "tensor-core commit assumes KAPPA divisible by TC_ROWS");
constexpr int TC_WARPS = TC_ROWS * 4;      // one m-tile (16 coeffs) per warp
constexpr int TC_FLEN = 136;               // bytes per plane row of the Toeplitz array (127 used + pad)

__device__ __forceinline__ void mma_u8s8(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// 4 consecutive bytes starting at byte offset o (any alignment) of a 4-byte-aligned buffer.
__device__ __forceinline__ uint32_t ld4u(const uint32_t* w, int o) {
    uint32_t lo = w[o >> 2], hi = w[(o >> 2) + 1];
    return __funnelshift_r(lo, hi, (o & 3) * 8);
}
// 4 signed digits {-1,0,1} as s8 bytes from 4 bits of pos/neg.
__device__ __forceinline__ uint32_t spread4(uint32_t x) {
    // bit k of a nibble -> byte k (0x01). The shifted copies x << 7m land on
    // distinct bit positions, so the multiply has no carries.
    return (x * 0x00204081u) & 0x01010101u;
}
__device__ __forceinline__ uint32_t digits4(uint64_t pos, uint64_t neg, int ell) {
    const uint32_t p = (uint32_t)(pos >> ell) & 0xF, n = (uint32_t)(neg >> ell) & 0xF;
    return spread4(p) | (spread4(n) * 0xFFu);   // +1 -> 0x01, -1 -> 0xFF (s8)
}

// v2: warp = (local row li, plane pair pp); covers all 4 m-tiles of its row.
// A lane's Toeplitz offsets across m-tiles differ by exactly 16 bytes, so one
// window of WINW words per plane serves 4 m-tiles x 2 k-steps, with
// compile-time word indices (only the funnel shift amount is per-lane).
constexpr int TC2_PPW = 2;                     // planes per warp
constexpr int TC2_WARPS = TC_ROWS * (8 / TC2_PPW);
constexpr int TC2_FLEN = 192;                  // bytes per plane row (window may start below 0: offset 56)
constexpr int TC2_OFF = 56;                    // F index u' is stored at byte u' + TC2_OFF
template <int CB>
__global__ void __launch_bounds__(TC2_WARPS * 32)
commit_ternary_tc2_kernel(const uint32_t* __restrict__ chacha_key, const uint64_t* __restrict__ pos_packed,
                          const uint64_t* __restrict__ neg_packed, uint64_t* __restrict__ partial, uint64_t N_ring,
                          uint64_t cols) {
    const int chunk = blockIdx.x, i0 = blockIdx.y * TC_ROWS;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, gid = lane >> 2, tig = lane & 3;
    const int li = warp / (8 / TC2_PPW);
    const int p0 = (warp % (8 / TC2_PPW)) * TC2_PPW;
    constexpr int NT = TC2_WARPS * 32;
    __shared__ uint32_t key_sh[8];
    __shared__ uint64_t M_sh[CB][TC_ROWS * D];
    __shared__ uint64_t pos_sh[CB][SPLIT_K], neg_sh[CB][SPLIT_K];
    __shared__ __align__(16) uint32_t F_sh[CB][TC_ROWS][8][TC2_FLEN / 4];
    if (t < 8) key_sh[t] = chacha_key[t];
    int acc[TC2_PPW][4][2][4];   // [plane][m-tile][n-tile][c]
    #pragma unroll
    for (int q = 0; q < TC2_PPW; q++)
      #pragma unroll
      for (int mt = 0; mt < 4; mt++)
        #pragma unroll
        for (int nt = 0; nt < 2; nt++)
          #pragma unroll
          for (int c = 0; c < 4; c++) acc[q][mt][nt][c] = 0;
    const uint64_t j_lo = (uint64_t)chunk * cols, j_hi = (j_lo + cols < N_ring) ? (j_lo + cols) : N_ring;
    // Lane window: o_mt,ks,delta = ks*32 + tig*4 - (16mt + gid) + 63 + delta, delta in {-8,0,8,16};
    // smallest = tig*4 - gid + 63 - 48 - 8 = tig*4 - gid + 7 (>= 0). Stored at +TC2_OFF.
    const int wbase_b = tig * 4 - gid + 7 + TC2_OFF;      // byte offset of window start
    const int sh = (wbase_b & 3) * 8;
    const int w0 = wbase_b >> 2;
    // zero the whole F (pad region) once
    for (int x = t; x < CB * TC_ROWS * 8 * (TC2_FLEN / 4); x += NT) ((uint32_t*)F_sh)[x] = 0;
    __syncthreads();
    for (uint64_t jb = j_lo; jb < j_hi; jb += CB) {
        const int nb = (int)((j_hi - jb < (uint64_t)CB) ? (j_hi - jb) : CB);
        for (int x = t; x < nb * SPLIT_K; x += NT) {
            const int c = x / SPLIT_K, d = x % SPLIT_K;
            pos_sh[c][d] = pos_packed[(uint64_t)d * N_ring + jb + c];
            neg_sh[c][d] = neg_packed[(uint64_t)d * N_ring + jb + c];
        }
        for (int x = t; x < nb * TC_ROWS * 8; x += NT) {
            const int c = x / (TC_ROWS * 8), lr = (x >> 3) % TC_ROWS, bi = x & 7; uint64_t buf[8];
            prg_ring_block_chacha8(key_sh, (uint32_t)(i0 + lr), jb + c, (uint32_t)bi, buf);
            #pragma unroll
            for (int k = 0; k < 8; k++) M_sh[c][lr * D + 8 * bi + k] = buf[k];
        }
        __syncthreads();
        for (int x = t; x < nb * TC_ROWS * 128; x += NT) {
            const int c = x / (TC_ROWS * 128), lr = (x >> 7) % TC_ROWS, u = x & 127;
            uint64_t v = 0;
            if (u < 127) {
                const int k = 63 - u;
                if (k >= 0) v = M_sh[c][lr * D + k];
                else { uint64_t m = M_sh[c][lr * D + k + 64]; v = m ? (ALMOST_GOLDILOCKS_PRIME - m) : 0; }
            }
            unsigned char* Fb = (unsigned char*)&F_sh[c][lr][0][0];
            #pragma unroll
            for (int p = 0; p < 8; p++) Fb[p * TC2_FLEN + TC2_OFF + u] = (unsigned char)(v >> (8 * p));
        }
        __syncthreads();
        for (int c = 0; c < nb; c++) {
            uint32_t bf[2][2][2];
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                const int b = nt * 8 + gid;
                const uint64_t P = (b < SPLIT_K) ? pos_sh[c][b] : 0, N = (b < SPLIT_K) ? neg_sh[c][b] : 0;
                #pragma unroll
                for (int ks = 0; ks < 2; ks++) {
                    bf[nt][ks][0] = digits4(P, N, ks * 32 + tig * 4);
                    bf[nt][ks][1] = digits4(P, N, ks * 32 + 16 + tig * 4);
                }
            }
            #pragma unroll
            for (int q = 0; q < TC2_PPW; q++) {
                const uint32_t* Fw = F_sh[c][li][p0 + q];
                // window words w0 .. w0+28 (byte span 107 + funnel lookahead)
                uint32_t win[29];
                #pragma unroll
                for (int k = 0; k < 29; k++) win[k] = Fw[w0 + k];
                #pragma unroll
                for (int mt = 0; mt < 4; mt++) {
                    #pragma unroll
                    for (int ks = 0; ks < 2; ks++) {
                        // o_mt,ks (delta 0) sits 56 - 16mt + 32ks bytes past the window
                        // start (plus the lane's sub-word shift `sh`): word `base`.
                        const int base = (56 - 16 * mt + 32 * ks) / 4;   // compile-time
                        uint32_t a[4];
                        a[0] = __funnelshift_r(win[base + 0], win[base + 1], sh);   // row r,   cols ell..   (delta 0)
                        a[1] = __funnelshift_r(win[base - 2], win[base - 1], sh);   // row r+8           (delta -8)
                        a[2] = __funnelshift_r(win[base + 4], win[base + 5], sh);   // row r,   cols +16 (delta +16)
                        a[3] = __funnelshift_r(win[base + 2], win[base + 3], sh);   // row r+8, cols +16 (delta +8)
                        mma_u8s8(acc[q][mt][0], a, bf[0][ks]);
                        mma_u8s8(acc[q][mt][1], a, bf[1][ks]);
                    }
                }
            }
        }
        __syncthreads();
    }
    // Each warp writes its plane group's signed partial (mod q) as its own
    // "chunk" (index chunk * 4 + pg); reduce_partials sums them with the
    // column chunks, which is just more terms of the same sum mod q.
    const int pg = p0 / TC2_PPW;
    const int i = i0 + li;
    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
      #pragma unroll
      for (int nt = 0; nt < 2; nt++)
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const int rr = 16 * mt + gid + ((c >= 2) ? 8 : 0);
            const int b = nt * 8 + tig * 2 + (c & 1);
            if (b >= SPLIT_K || i >= KAPPA) continue;
            uint64_t ulo = 0; int64_t uhi = 0;
            #pragma unroll
            for (int q = 0; q < TC2_PPW; q++) {
                const int p = p0 + q;
                const int64_t a = acc[q][mt][nt][c];
                uint64_t tlo; int64_t thi;
                if (p == 0) { tlo = (uint64_t)a; thi = (a < 0) ? -1 : 0; }
                else { tlo = (uint64_t)a << (8 * p); thi = a >> (64 - 8 * p); }
                const uint64_t s2 = ulo + tlo; uhi += thi + (s2 < ulo ? 1 : 0); ulo = s2;
            }
            uint64_t v;
            if (uhi >= 0) { agl_uint128_t X; X.lo = ulo; X.hi = (uint64_t)uhi; v = agl_reduce128(X); }
            else {
                const uint64_t nlo = ~ulo + 1; const uint64_t nhi = ~(uint64_t)uhi + (nlo == 0 ? 1 : 0);
                agl_uint128_t X; X.lo = nlo; X.hi = nhi;
                v = agl_sub_no_canonicalize(0, agl_reduce128(X));
            }
            partial[((((uint64_t)chunk * 4 + pg) * SPLIT_K + b) * KAPPA + i) * D + rr] = v;
        }
}

inline bool agl_ternary_tc_on() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("AGL_TERNARY_TC");
        on = (e && e[0] == '0') ? 0 : 1;
    }
    return on == 1;
}

inline bool agl_ternary_rows_on() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("AGL_TERNARY_ROWS");
        on = (e && e[0] == '0') ? 0 : 1;
    }
    return on == 1;
}

template <int CHUNK>
inline cudaError_t commit_ternary_run(
    const uint32_t* d_chacha_key,
    const uint64_t* d_pos_packed,
    const uint64_t* d_neg_packed,
    uint64_t        N_ring,
    uint64_t*       d_out,               // [SPLIT_K][KAPPA][D]
    cudaStream_t    stream = 0
) {
    static_assert(SPLIT_K * D <= 1024, "blockDim exceeds 1024");

    uint64_t num_chunks = (N_ring + CHUNK - 1) / CHUNK;

    if (agl_ternary_tc_on()) {
        // Column blocks: at most 256 (4 partial slots each, one per
        // plane-pair warp group), so partials stay <= ~290 MB even at
        // arity 29 (N_ring 2^23), where one slot set per caller CHUNK came to
        // 2.3 GB and failed to allocate in the memory-heavy finalization.
        // Each block may then take more columns, up to 131072 (the s32
        // accumulator bound).
        uint64_t gx = num_chunks < 256 ? num_chunks : 256;
        const uint64_t need = (N_ring + 131071) / 131072;
        if (gx < need) gx = need;
        const uint64_t cols = (N_ring + gx - 1) / gx;
        const uint64_t slots = gx * 4;
        uint64_t* d_partial = nullptr;
        cudaError_t err = cudaMallocAsync(&d_partial, slots * (uint64_t)SPLIT_K * KAPPA * D * sizeof(uint64_t), stream);
        if (err != cudaSuccess) {
            cudaGetLastError();   // clear, and fall through to the lazy kernel
            goto lazy_path;
        }
        dim3 g((unsigned)gx, (unsigned)(KAPPA / TC_ROWS));
        commit_ternary_tc2_kernel<4><<<g, TC2_WARPS * 32, 0, stream>>>(
            d_chacha_key, d_pos_packed, d_neg_packed, d_partial, N_ring, cols);
        dim3 grid2((unsigned)SPLIT_K, (unsigned)KAPPA, (unsigned)D);
        reduce_partials_kernel<<<grid2, 256, 256 * sizeof(uint64_t), stream>>>(
            d_partial, d_out, SPLIT_K, slots);
        cudaError_t free_err = cudaFreeAsync(d_partial, stream);
        if (free_err != cudaSuccess) return free_err;
        return cudaGetLastError();
    }
lazy_path:
    uint64_t partial_count = num_chunks * (uint64_t)SPLIT_K * KAPPA * D;
    uint64_t* d_partial = nullptr;
    cudaError_t err = cudaMallocAsync(&d_partial, partial_count * sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    dim3 block1((unsigned)(SPLIT_K * D));
    if (agl_ternary_rows_on()) {
        dim3 grid1((unsigned)num_chunks, (unsigned)TGROUPS);
        commit_ternary_rows_kernel<CHUNK><<<grid1, block1, 0, stream>>>(
            d_chacha_key, d_pos_packed, d_neg_packed, d_partial, N_ring
        );
    } else {
        dim3 grid1((unsigned)num_chunks);
        commit_ternary_kernel<CHUNK><<<grid1, block1, 0, stream>>>(
            d_chacha_key, d_pos_packed, d_neg_packed, d_partial, N_ring
        );
    }

    dim3 grid2((unsigned)SPLIT_K, (unsigned)KAPPA, (unsigned)D);
    int block2 = 256;
    size_t shared = (size_t)block2 * sizeof(uint64_t);
    reduce_partials_kernel<<<grid2, block2, shared, stream>>>(
        d_partial, d_out, SPLIT_K, num_chunks
    );

    cudaError_t free_err = cudaFreeAsync(d_partial, stream);
    if (free_err != cudaSuccess) return free_err;
    return cudaGetLastError();
}

} // namespace ajtai

#endif // ALMOST_AJTAI_CUH

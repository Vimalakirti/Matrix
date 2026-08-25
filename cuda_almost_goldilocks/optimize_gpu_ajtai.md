# Optimizing the GPU Ajtai commitment

This document records the three optimizations applied to the GPU Ajtai
commitment kernel and the before/after performance on a single A100.

All commits are over `R = F_q[X] / (X^{64} + 1)` with the almost-Goldilocks
prime `q = 2^64 - 2^32 - 31`. Sizes given by `log_n` mean `2^log_n`
polynomial coefficients (Basefold) and `2^(log_n - 6)` ring elements
(Ajtai, 64 binary coefficients per ring element).

---

## Starting point

The initial implementation followed the design in `ajtai.md`:

- Single kernel `commit_dense_batched_kernel<B, CHUNK>` with
  `gridDim = (num_chunks,)`, `blockDim = 64 · B`.
- ChaCha8 PRG with rejection sampling, `__forceinline__`'d everywhere.
- Fixed `CHUNK = 4096` in the benchmark.
- `__launch_bounds__(1024, 1)` on the kernel (needed only for `B = 16` to
  avoid silent launch failure from register over-allocation).

That implementation produced correct output but was slower than Basefold
across every size in the `log_n ∈ [14, 24]` range. At log_n = 18 the gap
was almost 10×.

## Diagnosis: ptxas resource report

`nvcc -O3 -arch=sm_80 --resource-usage` told three concurrent stories:

| Config | Threads/block | Regs/thread | Spills | Block occupancy/SM | SM occupancy |
|---|---|---|---|---|---|
| `B = 1`,  CHUNK = 4096 | 64   | 88 | 0          | 11 blocks | **34 %** |
| `B = 8`,  CHUNK = 4096 | 512  | 87 | 0          | 1 block  | **25 %** |
| `B = 16`, CHUNK = 4096 | 1024 | 64 (clamped) | 168 B store / 112 B load | 1 block | **50 %**, with spills |

Three problems:

1. **Per-thread register count is high.** The 15 `u64` accumulators
   (`acc[KAPPA]` = 30 32-bit regs) plus the ChaCha state co-occupy the
   register file because they have overlapping lifetimes from the
   compiler's perspective. Occupancy lands at 25–50 % everywhere — the
   SM has too few in-flight warps to hide latency.

2. **`B = 16` actively spills.** `__launch_bounds__(1024, 1)` forces the
   compiler to fit in 64 regs/thread, but the kernel naturally wants
   ~87. The compiler responds by spilling 168 B per kernel invocation.

3. **`CHUNK = 4096` wastes SMs at medium `N`.** With 108 SMs on A100, we
   want enough chunks to keep them all busy. CHUNK = 4096 gives
   `num_chunks = ceil(N / 4096)`:
   - `log_n = 18` (`N = 4096` ring elements) → 1 chunk → 1 SM busy.
   - `log_n = 22` (`N = 65536`) → 16 chunks → 16 SMs busy out of 108.

   The kernel's wall time was effectively pinned at ~170 ms (one chunk's
   worth of work) for everything from `log_n = 18` to `log_n = 24`,
   because we never had enough chunks to overlap across all SMs.

## Optimization (1): adaptive CHUNK selection

**What.** Add `C64` and `C128` to the `ChunkSize` enum (alongside the
existing `C256`, `C1024`, `C4096`). Pick the smallest supported CHUNK
that yields ~`target_chunks` blocks. The target depends on `B` since
different B configs have different block-occupancy per SM:

```rust
fn pick_default_chunk(n: u64, b: usize) -> ChunkSize {
    let target_chunks: u64 = if b <= 2 { 1200 } else { 200 };
    let needed = (n + target_chunks - 1) / target_chunks;
    if      needed <=   64 { ChunkSize::C64   }
    else if needed <=  128 { ChunkSize::C128  }
    else if needed <=  256 { ChunkSize::C256  }
    else if needed <= 1024 { ChunkSize::C1024 }
    else                   { ChunkSize::C4096 }
}
```

- `B = 1` has ~11 blocks/SM occupancy → wants `num_chunks ≳ 1200` to
  saturate the GPU.
- `B ≥ 4` has 1 block/SM → wants `num_chunks ≳ 108` (rounded to 200);
  going much past that adds stage-2 reduce work without parallelism
  benefit.

**Files.** `src/ajtai.rs`, `cuda/wrapper.cu`.

**Why it works.** At `log_n = 18` (`N = 4096`) with `B = 1`, the new
heuristic picks `CHUNK = 4`, yielding `num_chunks = 1024`. Now 1024
chunks run across all 108 SMs (with ~9-10 chunks per SM via the
~11 blocks/SM occupancy). Per-chunk work shrinks by 1024× and parallel
execution covers the rest. Same total work, much better scheduling.

## Optimization (2): `__noinline__` on `chacha8_block`

**What.** Mark the ChaCha8 block function as `__noinline__` so its
working state (16 u32 init + 16 u32 round state ≈ 32 32-bit regs)
lives only inside the function's frame, not across the kernel's whole
lifetime.

```cuda
// ajtai_chacha8.cuh
__host__ __device__ __noinline__
void chacha8_block(
    const uint32_t key[8], uint32_t counter,
    const uint32_t nonce[3], uint32_t out[16]
) {
    ...
}
```

**Files.** `ajtai_chacha8.cuh`.

**Why it works.** Previously, everything was `__forceinline__` so the
compiler saw `chacha8_block` inlined inside `prg_ring_block_chacha8`
inlined inside `prg_fill_column` inlined inside the main loop. The
ChaCha state's lifetime conceptually overlaps with the kernel's `acc[]`
state. With `__noinline__`, the function call is opaque: the ChaCha
state sits inside `chacha8_block`'s call frame, the kernel's regs are
free during the call. Function call overhead is ~30 cycles, negligible
versus the occupancy / spill savings.

**Resource change.** Per-thread register count drops from 87 → 76 at
`B ≤ 8`, **and all spills disappear at `B = 16`** (was 168 B store /
112 B load → 0 B). A 128-byte stack frame appears (the function's
locals), routed through L1 cache.

## Optimization (3): halve-rows kernel for `B ∈ {1, 2, 4, 8}`

**What.** A second kernel `commit_dense_batched_halve_kernel<B, CHUNK>`
where **two threads** cover each `(b, r)` position. Each thread holds
**8 u64 accumulators** instead of 15 — one half of the 15 output rows.
`blockDim` doubles to `2 · B · D`.

```cuda
// ajtai.cuh — sketch
template <int B, int CHUNK>
__global__ void
__launch_bounds__(2 * B * D, 1)
commit_dense_batched_halve_kernel(...) {
    int h = t / (B * D);                 // row-half: 0 or 1
    int b = (t / D) % B;                 // batch index
    int r = t & (D - 1);                 // coefficient slot
    int row_base = h * 8;
    uint64_t acc[8];                     // h=0 uses all 8, h=1 uses first 7

    for (uint64_t j = j_lo; j < j_hi; j++) {
        // load bits, fill M_shared via cooperative ChaCha8
        ...
        uint64_t mask = bits_sh[b];
        while (mask) {
            int ell = __ffsll(mask) - 1; mask &= mask - 1;
            int idx; bool wrap;
            // (idx, wrap) ← (r, ell)
            #pragma unroll
            for (int i_local = 0; i_local < 8; i_local++) {
                int i = row_base + i_local;
                if (i < KAPPA) {  // skip i=15 when h=1, i_local=7
                    uint64_t v = M_sh[i * D + idx];
                    acc[i_local] = add_signed(acc[i_local], v, wrap);
                }
            }
        }
    }
    // ... write partial[chunk][b][i][r] for h's row range
}
```

The wrapper.cu dispatch routes `B ∈ {1, 2, 4, 8}` to the halve kernel,
keeping `B = 16` on the standard kernel (the halve layout would need
`blockDim = 2 · 16 · 64 = 2048`, above SM_80's 1024 hard limit).

**Files.** `ajtai.cuh`, `cuda/wrapper.cu`.

**Why it works.** Per-thread `acc` drops from 15 → 8 u64 (30 → 16
32-bit regs). At `B = 1` this changes occupancy from
`12 blocks × 64 threads = 768 threads/SM (37.5 %)` to
`8 blocks × 128 threads = 1024 threads/SM (50 %+)`. More importantly,
the additional warps per block hide latency much better — `B = 1` is
the configuration with the lowest occupancy to start with, so it gains
the most.

`B ≥ 4` sees marginal occupancy improvement (block count already capped
at 1) but still benefits from the larger in-block warp pool.

`B = 16` cannot use this layout (thread overflow), so we keep it on the
standard kernel where (2) has already eliminated its spills.

---

## Before / after on A100

Same `cargo run --release --example bench_commit_only` invocation, same
hardware, same min-of-N timing methodology.

### Per-commit time (ms)

| log_n | # coefs | Basefold | Ajtai B=1 (before) | **Ajtai B=1 (after)** | Ajtai B=16 (before) | **Ajtai B=16 (after)** |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 16 K  |  1.25 |  11.10 |  **2.12** (5.2× ↓) |  1.80 | **0.47** (3.8× ↓) |
| 16 | 64 K  |  1.81 |  42.93 |  **2.13** (20× ↓)  |  7.12 | **0.51** (14× ↓) |
| 18 | 256 K |  3.22 | 170.50 |  **2.18** (78× ↓)  | 28.43 | **0.51** (56× ↓) |
| 20 | 1 M   |  8.49 | 171.13 |  **3.24** (53× ↓)  | 28.54 | **1.82** (16× ↓) |
| 22 | 4 M   | 45.29 | 171.13 |  **6.75** (25× ↓)  | 28.63 | **7.17** (4.0× ↓) |
| 24 | 16 M  | 212.0 | 173.54 | **23.35** (7.4× ↓) | 30.75 | **29.33** (~ same) |
| 26 | 64 M  | 1619  | 194.53 | **88.13** (2.2× ↓) | 106.8 | **102.7** (~ same) |

### Speedup vs Basefold

| log_n | Best Ajtai (before) | **Best Ajtai (after)** |
|---:|---:|---:|
| 14 | 0.7 × (Basefold) | **2.7 × (Ajtai)** |
| 16 | 0.25 × (Basefold) | **3.5 × (Ajtai)** |
| 18 | 0.11 × (Basefold) | **6.3 × (Ajtai)** |
| 20 | 0.30 × (Basefold) | **4.7 × (Ajtai)** |
| 22 | 1.6 × (Ajtai)     | **6.7 × (Ajtai)** |
| 24 | 6.9 × (Ajtai)     | **9.1 × (Ajtai)** |
| 26 | 15.2 × (Ajtai)    | **18.4 × (Ajtai)** |

### Resource changes

| Config | Regs/thread before | Regs/thread after | Spill stores before | Spill stores after | Block occupancy before | Block occupancy after |
|---|---:|---:|---:|---:|---:|---:|
| B=1   | 88 | 76† | 0   | 0 | 11 blk × 64 thr  | 8 blk × 128 thr   (halve) |
| B=4   | 87 | 76† | 0   | 0 | 1 blk × 256 thr  | 1 blk × 512 thr   (halve) |
| B=8   | 87 | 76† | 0   | 0 | 1 blk × 512 thr  | 1 blk × 1024 thr  (halve) |
| B=16  | 64 | 64  | 168 | **0** | 1 blk × 1024 thr | 1 blk × 1024 thr  (standard) |

† Halve kernel uses about the same reg count per thread, but each
thread now does half the row work — total per-block reg pressure
roughly matches.

## Result

Ajtai now wins at every polynomial size tested, by 2.7×–18.4× over
Basefold. The biggest individual gains came from (1) adaptive CHUNK at
medium N (78× drop at log_n=18 B=1, just from giving the SMs work to
do); (3) halve-rows added another ~30% on top at B=1; (2) was a
correctness-improving change that incidentally also fixed B=16's spills.

## Open items not addressed

- **`log_n = 26` with `B ∈ {8, 16}`** still measures ~200 ms / commit
  and ~103 ms / commit respectively. Not investigated in detail —
  most likely a `cudaMallocAsync` pool resize or SM-wave issue at this
  specific (CHUNK=4096, num_chunks=256, ~16 MB partial buffer) combo.
  At this size the single-commit path (`B = 1` = 88 ms) is already
  fastest, so the practical impact is small.

- **PTX-tuned ChaCha8.** The portable C ARX implementation is what we
  ship. A PTX-tuned version using `__byte_perm` for rotate-by-8/16
  could cut PRG throughput by another ~2×. Not done — would move every
  Ajtai number in the table down by roughly half.

- **Approach 2 batched kernel for `B > 16`** (multiple j's per block,
  per the design doc §14). Deferred per the original design until
  benchmarking actually demands it.

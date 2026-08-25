# GPU Ajtai Commitment over Almost-Goldilocks for Binary Witnesses

This document specifies an efficient and correct GPU implementation of the
Ajtai commitment

$$ c = Mz $$

over the ring

$$ R = \mathbb F_q[X] / (X^{64} + 1), \qquad q = 2^{64} - 2^{32} - 31. $$

`q` is the **almost-Goldilocks** prime; field arithmetic is reused verbatim
from `almost_goldilocks.cuh` (no re-implementation — see §4).

Target sizes:

| | |
|---|---|
| witness `z` | binary, length `2^33` |
| packed `z_R` | `R^{2^27}`, 64 binary coefficients per ring element |
| matrix `M` | `R^{15 × 2^27}`, never materialized |
| output `c` | `R^15` (960 field elements) |

Three execution paths are exposed:

* **Dense single**: scan packed `z_bits[2^27]`, one commit.
* **Dense batched**: scan `B` packed witnesses together against the same
  public `M`. PRG cost is shared across the batch (single-commit PRG cost,
  `B`× rotation cost). This is the recommended high-throughput path.
* **Sparse single**: scan a list of set-bit positions. Used when within-block
  density is very low (see §9 for the precise crossover).

All paths produce bit-exact identical `c` for the same `(seed, z)`.

---

## 1. Mathematical structure

For `R = F_q[X] / (X^{64} + 1)`, since `X^{64} = -1`, multiplication by `X^ℓ`
is a **negacyclic** rotation:

$$ (X^\ell \cdot a)_r = \begin{cases} a_{r-\ell}, & r \ge \ell \\ -a_{r-\ell+64}, & r < \ell. \end{cases} $$

A ring element is stored as a length-64 array of `u64` field elements.

The witness is binary, so each ring element `z_j` is packed as a single
`uint64_t` bitmask: bit `ℓ` of `z_bits[j]` holds `z_{j,ℓ} ∈ {0, 1}`.

**Why this prime + this ring.** Goldilocks (`p_G = 2^64 - 2^32 + 1`) has
2-adicity 32 in `p_G - 1`, so `128 | p_G - 1`, and `X^{64} + 1 = Φ_128(X)`
splits *fully* into 64 linear factors over `F_{p_G}` — that collapses ring
SIS to component-wise base-field SIS and weakens the security argument.
For almost-Goldilocks: `(q − 1)` has 2-adicity only 5, and a direct
computation gives `q mod 128 = 97` with `ord_128(97) = 4`. Hence
`X^{64} + 1` splits into **16 irreducible factors of degree 4** over `F_q`
— enough non-trivial ring structure to preserve the SIS hardness argument.

## 2. The selected-rotation identity

Because each `z_{j,ℓ}` is 0 or 1, the ring product collapses:

$$ M_{i,j} \cdot z_j = \sum_{\ell \,:\, z_{j,\ell}=1} X^\ell \cdot M_{i,j}. $$

So

$$ c_i = \sum_{j=0}^{2^{27}-1} \sum_{\ell \,:\, z_{j,\ell}=1} X^\ell \cdot M_{i,j}. $$

This is the kernel-level objective: **every field multiplication is
replaced by a signed add of a shifted coefficient**. The actual cost
breakdown shifts as a result — see §13.

## 3. Negacyclic shift

For coefficient `r ∈ [0, 64)` of `X^ℓ · a` (with `ℓ ∈ [0, 64)`), the
implementation computes `(idx, wrap)` from `(r, ℓ)` and dispatches:

```cuda
int  idx  = r - ell;          // signed
bool wrap = idx < 0;
idx      += wrap ? 64 : 0;    // now in [0, 64)
uint64_t v = M_shared[i * 64 + idx];
acc[i]  = wrap
        ? agl_sub_no_canonicalize(acc[i], v)
        : agl_add_no_canonicalize(acc[i], v);
```

Both `(idx, wrap)` depend only on `(r, ℓ)` — not on `i` or batch index
`b` — so when multiple output rows (and multiple batch items) are
accumulated together (§7, §8) the index/sign is computed once per
`(r, ℓ)` and reused across all `15 × B` accumulator slots.

**Worked sanity check (d = 4):**
`X · [a₀, a₁, a₂, a₃] = [−a₃, a₀, a₁, a₂]`. Read off: r=0 ⇒ r<ℓ wraps from
index 4−1=3, signed; r=1,2,3 ⇒ no wrap, take `a[r-1]`. ✓

## 4. Field arithmetic — reuse existing kernels

**Do not re-derive** `add_mod_q`. The naive

```cuda
uint64_t s = a + b;
if (s < a)        s += C;        // C = 2^32 + 31
if (s >= Q)       s -= Q;
return s;
```

is *incorrect* on roughly `2^{-31}` of random inputs: after the carry
fold, `s + C` can itself overflow `u64` and the single conditional
subtract doesn't recover. Over `2^{37}` adds (a full dense commitment)
that's hundreds of corrupted coefficients.

The crate already exposes correct routines in `almost_goldilocks.cuh`:

| Function | Use |
|---|---|
| `agl_add(a, b)` | mod-q add, double-fold safe |
| `agl_sub(a, b)` | mod-q sub |
| `agl_neg(a)` | mod-q negation, canonicalized output |
| `agl_canonicalize(v)` | final reduction to `[0, q)` |
| `agl_add_no_canonicalize`, `agl_sub_no_canonicalize` | inner-loop variants on raw u64 |

Use these directly. The only commit-specific helper needed is the
signed-coefficient add (just dispatches by the wrap flag):

```cuda
__device__ __forceinline__
uint64_t add_signed(uint64_t acc, uint64_t v, bool sub) {
    return sub
        ? agl_sub_no_canonicalize(acc, v)
        : agl_add_no_canonicalize(acc, v);
}
```

Final canonicalization happens once per `c[b][i][r]` at the end of stage 2.

## 5. Matrix PRG: ChaCha8

`M` would require `15 · 2^{33} · 8 ≈ 960 GiB` to materialize. Generate
each `M_{i,j}` on the fly from a public seed using **ChaCha8**, keyed
deterministically by `(seed, i, j)`.

### 5.1 Why ChaCha8 (not ChaCha20, not AES)

For SIS-commitment security we need a **cryptographic PRG**: output
must be computationally indistinguishable from uniform over `F_q^{κ×N}`
to an adversary who chose `z` before seeing the seed. We do **not** need
AEAD-grade security — there are no chosen-input attacks, no integrity,
no collision resistance.

ChaCha20 has 20 rounds to defend against differential attacks under
adversarial chosen plaintexts. We have no chosen plaintexts. The best
public differential cryptanalysis breaks ~6 rounds of ChaCha;
**ChaCha8 retains a >2 round security margin against any known attack
on a PRG-grade primitive**, and is exactly 2.5× faster than ChaCha20.
It is the standard fast-PRG variant of the ChaCha family — used in
the Rust `rand_chacha` crate, in BoringSSL's random-bytes path, and
in many MPC libraries.

**AES is not faster on our hardware.** NVIDIA datacenter GPUs (V100 /
A100 / H100 / B200) have **no hardware AES instructions** exposed to
CUDA. Software AES on A100 lands at 400–500 GB/s with careful T-table
tuning; ChaCha8 lands at 700–1000 GB/s with simpler code and no
S-box bank-conflict concerns. Even the most aggressive 4-round-reduced
AES variant doesn't beat ChaCha8 on A100. If we ever ship a CPU PRG
pipeline, AES-NI is a clear win — but for GPU it is not.

### 5.2 Keying and bias

Key derivation:

```
chacha_key  = SHA-256(seed || "agl_ajtai_matrix")     // 32 bytes
nonce_96    = (i << 64) | j                            // 96-bit nonce
counter_32  = block_idx ∈ [0, 8)                      // 8 ChaCha blocks per ring element
output      = ChaCha8(chacha_key, nonce_96, counter_32) → 64 bytes = 8 u64
```

So one ring element (64 u64) = 8 ChaCha8 calls; all 15 rows for column `j` =
120 ChaCha8 calls. ChaCha8 is keyed once per `(i, j)` pair, never re-keyed
across blocks (the counter advances).

**Bias.** Reducing a uniform 64-bit value `s` to `F_q` via `s mod q` is
biased by `(2^{64} - q · ⌊2^{64}/q⌋) / 2^{64} ≈ 2^{-31}`. Over `2^{37}`
samples this is *not* negligible — it skews the matrix distribution
measurably and the worst-case-to-average-case SIS reduction stops being
clean. Fix:

```cuda
// rejection sample: cost is negligible (~1 retry per 2^31 samples)
const uint64_t REJ = Q * (UINT64_MAX / Q);   // largest u64 multiple of q
uint64_t s;
do { s = chacha8_next_u64(); } while (s >= REJ);
return s % Q;
```

Rejection probability ≈ 2⁻³¹ per coefficient ⇒ ~`64 retries` for the
whole `2^{37}`-sample commit ⇒ essentially zero divergence cost. **Use
rejection sampling, not wider-draw.** (Wider-draw needs ≥128 bits per
coefficient for negligible bias, which doubles PRG cost — strictly worse.)

The seed for `M` must be fixed *before* the prover chooses the witness
(standard Fiat-Shamir requirement).

## 6. Output reduction strategy

The output is tiny (`c ∈ R^15`, or `B × 15 × 64` u64s for batched). Global
atomics into 960·B locations across millions of blocks will be
contention-bound. Use **two-stage reduction**:

1. **Stage 1** — `commit_dense_batched_kernel`: each block processes a
   `CHUNK_SIZE` slice of the `j` axis and writes `B × 15` ring elements
   to `partial[num_chunks][B][15][64]`.
2. **Stage 2** — `reduce_partials_kernel`: one block per `(b, i, r)`,
   sums `partial[*][b][i][r]` to `c[b][i][r]` and canonicalizes.

If `partial` is too large for the GPU (see §11), insert one or more
intermediate reduction levels.

## 7. Block layout — one block per chunk, all batches and rows fused

**The recommended layout.** A single block handles one `j`-chunk and
all `B` batched witnesses simultaneously, sharing one cooperatively
generated `M_shared[15 × 64]` across the entire batch.

```
gridDim  = (num_chunks,)
blockDim = 64 · B                  // one thread per (b, r) pair
```

Thread `t` maps to `(b, r) = (t / 64, t mod 64)`:

* `b` ∈ [0, B): which batched witness this thread accumulates for
* `r` ∈ [0, 64): which coefficient slot this thread owns

Per-thread state:

* `acc[15]` — fifteen `u64` accumulators (one per row `i`). About 30
  32-bit registers; fits comfortably in the 64-register budget that
  preserves full occupancy on SM_80.

Per-block shared:

* `M_shared[15 × 64]` = 7.5 KiB — the cooperatively generated matrix
  column `M[*, j]`.
* `bits_shared[B]` = `8B` bytes — the witness bitmasks `z_bits[b][j]`
  for the current `j`.

Why this layout:

* `z_bits[b][j]` is read **once** per `j` per batch (not 15× as a
  per-row layout would force).
* PRG output for `M[*, j]` is generated **once** per `j` across the
  entire batch — this is the source of the batching speedup.
* `(idx, wrap)` is computed once per `(r, ℓ)` and reused across
  `15 × ?` accumulator updates (only the rows; batches use different
  `ℓ`s coming from different bitmasks).

For `B ≤ 16`, `blockDim ≤ 1024`, which is the A100 hard ceiling. For
larger batch sizes, split into multiple blocks per chunk (see §14).

## 8. Dense batched kernel

```cuda
constexpr int  D      = 64;
constexpr int  KAPPA  = 15;
constexpr uint64_t Q  = 0xFFFFFFFEFFFFFFE1ULL;

template <int B, int CHUNK>
__global__ void commit_dense_batched_kernel(
    Seed                       seed,
    const uint64_t* const* __restrict__ z_bits,   // [B][N]
    uint64_t*       __restrict__       partial,   // [num_chunks][B][KAPPA][D]
    uint64_t                            N
) {
    int chunk = blockIdx.x;
    int t     = threadIdx.x;                       // 0 .. 64*B - 1
    int b     = t / D;
    int r     = t % D;

    __shared__ uint64_t M_shared   [KAPPA * D];    // 7.5 KiB
    __shared__ uint64_t bits_shared[B];

    uint64_t acc[KAPPA];
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) acc[i] = 0;

    uint64_t j_lo = (uint64_t)chunk * CHUNK;
    uint64_t j_hi = min(j_lo + CHUNK, N);

    for (uint64_t j = j_lo; j < j_hi; j++) {

        // (a) Load all B bitmasks for this j cooperatively.
        if (t < B) {
            bits_shared[t] = z_bits[t][j];
        }

        // (b) Cooperatively fill M_shared[15*64] with ChaCha8 keyed (seed, *, j).
        //     120 ChaCha8 blocks total, distributed across 64*B threads.
        prg_fill_column_chacha8(seed, j, M_shared, t, blockDim.x);
        __syncthreads();

        // (c) This thread's batch bitmask. All 64 threads with the same b
        //     execute the same set-bit iteration count — non-divergent within
        //     a warp when warps align to b (which they do, since b = t / 64).
        uint64_t mask = bits_shared[b];

        while (mask) {
            int  ell  = __ffsll(mask) - 1;
            mask     &= mask - 1;

            int  idx_  = r - ell;
            bool wrap  = idx_ < 0;
            idx_      += wrap ? D : 0;

            #pragma unroll
            for (int i = 0; i < KAPPA; i++) {
                uint64_t v = M_shared[i * D + idx_];
                acc[i] = add_signed(acc[i], v, wrap);
            }
        }
        __syncthreads();
    }

    // (d) Write partial[chunk][b][i][r] for i = 0..14, coalesced over r in r-warps.
    uint64_t base = ((uint64_t)chunk * B + b) * (KAPPA * D) + r;
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) {
        partial[base + i * D] = acc[i];
    }
}
```

Notes:

* `__ffsll(mask) + mask &= mask - 1` makes the inner loop iterate exactly
  `popcount(bits_shared[b])` times. Within a warp, all 32 threads share
  the same `b` (since `b = t / 64`, and warps are aligned on 32-thread
  boundaries) and therefore the **same `mask`** → the loop is fully
  non-divergent within a warp.
* `(idx, wrap)` computation depends only on `(r, ℓ)`, so each warp
  computes 32 distinct `(idx, wrap)` pairs in lock-step and uses them
  to update each thread's 15 accumulators in parallel.
* Single-commit is just `B = 1` — no separate code path required.

### Skipping all-zero blocks

When the witness is partially sparse, `bits_shared[b] == 0` for some `b`
on some `j`. The natural optimization "`if (bits == 0) continue;`" is
**wrong inside a batched kernel**: different batches might have different
bitmasks, so the block can't skip the `j` entirely. Instead, each thread
just falls through the empty `while (mask)` loop — zero cost — and the
shared `M_shared` generation still runs (necessary for the other batch
items). If you have telemetry that *all* batches have `bits == 0` for
a given `j`, you can broadcast that via a warp vote and `continue` the
whole `j` iteration. Optional micro-optimization.

## 9. Sparse single-commit kernel

For witnesses with very low within-block density (per-non-zero-block
`E[popcount] < ~2`), iterate over a position list `positions[K]` instead
of scanning `z_bits[N]`. Batching is **not** supported in the sparse
path — different witnesses have different position lists with no
alignment, so PRG cannot be amortized across batches. If you need to
commit to multiple sparse witnesses, run sequential sparse commits or
densify and use the batched kernel.

```cuda
template <int CHUNK>
__global__ void commit_sparse_partial_kernel(
    Seed                              seed,
    const uint64_t* __restrict__      positions,   // length K
    uint64_t                          K,
    uint64_t*       __restrict__      partial,     // [num_chunks][15][64]
    uint64_t                          num_chunks
) {
    int chunk = blockIdx.x;
    int t     = threadIdx.x;                       // 0..63 = coefficient index r

    __shared__ uint64_t M_shared[KAPPA * D];
    uint64_t acc[KAPPA];
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) acc[i] = 0;

    uint64_t p_lo = (uint64_t)chunk * CHUNK;
    uint64_t p_hi = min(p_lo + CHUNK, K);

    for (uint64_t k = p_lo; k < p_hi; k++) {
        uint64_t p   = positions[k];
        uint64_t j   = p >> 6;
        int      ell = (int)(p & 63);

        prg_fill_column_chacha8(seed, j, M_shared, t, blockDim.x);
        __syncthreads();

        int  idx_ = t - ell;
        bool wrap = idx_ < 0;
        idx_     += wrap ? D : 0;

        #pragma unroll
        for (int i = 0; i < KAPPA; i++) {
            uint64_t v = M_shared[i * D + idx_];
            acc[i] = add_signed(acc[i], v, wrap);
        }
        __syncthreads();
    }

    uint64_t base = ((uint64_t)chunk * KAPPA) * D + t;
    #pragma unroll
    for (int i = 0; i < KAPPA; i++) {
        partial[base + i * D] = acc[i];
    }
}
```

**When does sparse actually win?** PRG dominates total cost. Dense
regenerates `M[*, j]` once per non-zero `j`-block; sparse regenerates
once per set bit. Let `B_block` = expected set bits per non-zero
`z_bits[j]` block. Then:

* Dense PRG cost ∝ `#(non-zero blocks)`
* Sparse PRG cost ∝ `K = B_block · #(non-zero blocks)`

So sparse only wins when **`B_block < ~2`** — i.e., on average fewer
than 2 set bits per non-zero block, which is `< 3%` per-block density.
For random uniform binary witnesses (`E[B_block] = 32`), dense is
**~16× cheaper** even without batching. Recommendation: route to
sparse only when an upstream density check indicates `B_block < 2`.

## 10. Reduce-partials kernel (batched-aware)

```cuda
__global__ void reduce_partials_kernel(
    const uint64_t* __restrict__ partial,          // [num_chunks][B][15][64]
    uint64_t*       __restrict__ c,                // [B][15][64]
    int                          B,
    uint64_t                     num_chunks
) {
    int b   = blockIdx.x;                          // 0..B-1
    int i   = blockIdx.y;                          // 0..14
    int r   = blockIdx.z;                          // 0..63
    int tid = threadIdx.x;

    extern __shared__ uint64_t smem[];

    uint64_t acc = 0;
    for (uint64_t k = tid; k < num_chunks; k += blockDim.x) {
        uint64_t off = ((k * B + b) * KAPPA + i) * D + r;
        acc = agl_add_no_canonicalize(acc, partial[off]);
    }
    smem[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = agl_add_no_canonicalize(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    if (tid == 0) c[(b * KAPPA + i) * D + r] = agl_canonicalize(smem[0]);
}
```

Launch:

```
gridDim  = (B, 15, 64)
blockDim = 256
shared   = 256 * sizeof(uint64_t)
```

Only the final write is canonicalized — within-block accumulation runs
on raw u64 representatives, matching the rest of the crate's convention.

## 11. Memory budget

`partial[num_chunks][B][15][64]` is the only sizable transient buffer.

| `CHUNK` | `num_chunks` | partial size (B=1) | partial size (B=8) | partial size (B=32) |
|--------:|------------:|----:|----:|----:|
|     256 |        2^19 | 3.75 GiB | 30 GiB | 120 GiB |
|    1024 |        2^17 | 960 MiB  | 7.5 GiB | 30 GiB |
|    4096 |        2^15 | 240 MiB  | 1.9 GiB | 7.5 GiB |
|    8192 |        2^14 | 120 MiB  | 960 MiB | 3.75 GiB |
|   16384 |        2^13 | 60 MiB   | 480 MiB | 1.9 GiB |

For batched commits on an 80 GB A100, prefer `CHUNK ≥ 4096`. For smaller
GPUs or large batches, see **hierarchical reduction**: stage 1 writes a
smaller per-chunk-group partial, an intermediate kernel collapses
chunk-groups, stage 2 collapses the result. Atomic adds into a small
partial buffer are an alternative but add modular-arithmetic complexity
to the atomic path; tree reduction is preferred.

## 12. Multi-GPU

The `j`-axis splits embarrassingly. For `G` GPUs:

```
GPU g  computes   c^(g)_{b,i} = Σ_{j ∈ range_g} M_{i,j} z_{b,j}
final            c_{b,i} = Σ_g c^(g)_{b,i}        // 15·B ring elements all-reduce
```

Communication: `G · B · 15 · 64 · 8 B` ≈ a few KiB even for large `B`.
Trivially overlap with compute.

**Determinism requirement.** Every GPU must derive identical
`M_{i,j}` for shared `(i, j)`. This is satisfied iff the PRG is a
pure function of `(seed, i, j, block_idx)` with no GPU-local state.
The ChaCha8-with-nonce-`(i, j)` keying in §5.2 satisfies this — each
nonce is independent, no shared counter, no launch-order dependence.

Batching composes cleanly with multi-GPU: each GPU processes the same
batch of `B` witnesses over its `j`-range; all-reduce produces `B`
final commitments.

## 13. Performance model

Single A100, ChaCha8 PRG, no batching, dense random binary witness
(`E[wt] = 2^{32}`):

| | work | A100 throughput | wall time |
|---|---|---|---|
| ChaCha8 PRG | 2³⁴ blocks (1 TiB output) | ~800 GB/s | **~1.3 s** |
| signed mod-q adds | 2³⁸ ops | ~10 G ops/s (64-bit) | ~30 ms |
| witness reads (`z_bits`) | 1 GiB | 2 TB/s | <1 ms |
| stage-1 partial writes | up to 960 MiB | HBM | <1 ms |
| stage-2 reduce | 960 elements × 2¹⁷ blocks | HBM | a few ms |
| **total** | | | **~1.4 s** |

PRG dominates ~40:1 over the add loop. This sets the floor for
single-commit latency.

### Batched, single A100

PRG cost is independent of batch size; add-loop cost scales linearly.

| `B` | PRG | adds | total | per-commit |
|---:|---:|---:|---:|---:|
|   1 | 1.3 s | 0.03 s | **1.35 s** | 1.35 s |
|   4 | 1.3 s | 0.12 s | **1.42 s** | 0.36 s |
|   8 | 1.3 s | 0.24 s | **1.54 s** | 0.19 s |
|  16 | 1.3 s | 0.48 s | **1.78 s** | 0.11 s |
|  32 | 1.3 s | 0.96 s | **2.26 s** | 0.071 s |
|  64 | 1.3 s | 1.92 s | **3.22 s** | 0.050 s |
| 128 | 1.3 s | 3.84 s | **5.14 s** | 0.040 s |

Crossover where adds = PRG: **`B ≈ 43`**. Diminishing returns past `B ≈ 32`.
Recommended default if memory permits: `B = 16` or `B = 32`.

### Multi-GPU, batched

For `G` GPUs and batch `B`, wall time ≈ `(1.3 s + B · 0.03 s) / G`.
On 8× A100 with `B = 32`: `~0.28 s` total ⇒ **~9 ms per commit**.

## 14. Batched-kernel design details

Block-size budget: `blockDim = 64 · B` must be ≤ 1024 on A100, so the
single-block batched kernel works for **`B ∈ {1, 2, 4, 8, 16}`**. For
larger batches:

* **Approach 1 (split batch into multiple blocks per chunk):**
  `gridDim = (num_chunks, ceil(B / B_block))`, each block handles
  `B_block` batches of the same chunk. PRG is generated once per
  `(chunk, batch_group)`, so this regenerates `M[*, j]` ⌈B / B_block⌉
  times per `j` — **partially defeating the amortization**. Use only
  if necessary.
* **Approach 2 (multiple j's per block):** Keep one block per chunk
  but make each block handle 2-4 consecutive j's in parallel. PRG
  generation is fully amortized within a block, accumulators are
  per-(b, r, j-slot). Block size = `64 · B`; j-slots add a small
  per-thread state factor. Best choice for `B ∈ [16, 64]`.

Recommended path: implement `B ∈ {1, 2, 4, 8, 16}` with the
single-block kernel from §8; defer Approach 2 to a tuning pass after
benchmarking shows it's needed.

### Batch API

```rust
fn ajtai_commit_batched(
    seed: &Seed,
    witnesses: &[&[u64]],          // B witnesses, each length N = 2^27
) -> Result<Vec<[[u64; 64]; 15]>>; // B commitments
```

Single commit:

```rust
fn ajtai_commit_single(seed: &Seed, z_bits: &[u64]) -> Result<[[u64; 64]; 15]> {
    let out = ajtai_commit_batched(seed, &[z_bits])?;
    Ok(out.into_iter().next().unwrap())
}
```

For commitments that share `(seed, M)` across many calls, the batched
API is strictly preferred — it amortizes the dominant PRG cost.

## 15. Testing

Implement and verify in order:

1. **CPU reference** for `d = 4` and `d = 8`: dense polynomial mul,
   negacyclic shift, binary-selected-rotation product, full
   commit-then-compare against (`d`, `2^N`, `κ`) = (4, 16, 3) and
   (8, 64, 3). Exact equality required.
2. **GPU field arithmetic** golden vectors:
   `add`, `sub`, `neg`, `agl_canonicalize` against the existing
   `cuda_almost_goldilocks/almost_field_test`. (Already covered.)
3. **ChaCha8 + rejection-sample PRG** determinism: same `(seed, i, j)`
   ⇒ identical `M[i, j, *]` across launches and across GPUs.
4. **Shift correctness:** for `ℓ ∈ [0, 64)` and a random ring element
   `a`, verify `(X^ℓ · a)[r]` matches the CPU reference for all `r`.
5. **Dense single commit** vs CPU reference for `N ∈ {64, 256, 1024, 4096}`.
   Exact equality.
6. **Dense batched commit** (`B ∈ {2, 4, 8, 16}`) vs `B` independent
   single commits. Exact equality. Confirms amortization is correctness-
   preserving.
7. **Sparse single commit** vs dense single commit on the same `z`.
   Exact equality.
8. **Chunk-size invariance:** result identical for `CHUNK ∈
   {256, 512, 1024, 2048, 4096, 8192, 16384}`.
9. **Multi-GPU sum** equals single-GPU result for `G ∈ {2, 4}` splits,
   both single and batched.
10. **Bias check** (sanity, not a hard pass/fail): sample `2^{25}` matrix
   coefficients and confirm empirical mean ≈ `q/2` within ≈ 3σ. Detects
   gross PRG implementation bugs.

## 16. Pitfalls

| Pitfall | Mitigation |
|---|---|
| Reimplementing `add_mod_q` with a single carry-fold (rare double-overflow path) | Use `agl_add_*` from `almost_goldilocks.cuh` |
| `s mod q` on raw 64-bit PRG output (bias ≈ 2⁻³¹) | Rejection sampling (negligible cost, exact uniformity) |
| Wider-draw instead of rejection (needs 128 bits for safety, doubles PRG cost) | Use rejection — strictly cheaper |
| Materializing `M` | Generate `M[*, j]` on the fly per block |
| Global atomics into `c` from millions of blocks | Two-stage tree reduction |
| Treating binary mul as field mul | Selected negacyclic rotation only |
| Per-row block layout `(i, chunk)` (15× redundant witness traffic) | Per-chunk block layout, all batches and 15 rows fused (§7) |
| `if (bits == 0) continue;` inside batched kernel (wrong: other batches' bits) | Each thread falls through empty `while (mask)`; optional all-batch vote |
| Inferring sparse vs dense from total weight | Compare *within-non-zero-block density* `B_block` against 2 (§9) |
| Different PRG keying logic per GPU (breaks determinism) | PRG is pure function of `(seed, i, j, block_idx)` |
| ChaCha20 by default for "safety margin" | ChaCha8 is the correct PRG-grade choice; ChaCha20 is 2.5× wasted work |

## 17. Recommended implementation path

| Step | Output | Validation |
|---|---|---|
| 1 | CPU reference for `d = 4`, `d = 8` (single + batched) | Hand-computed vectors |
| 2 | `prg_fill_column_chacha8` (ChaCha8 + rejection sampling) | PRG determinism, bias sanity check |
| 3 | `commit_dense_batched_kernel` with `B ∈ {1, 2, 4, 8, 16}` (§8) | GPU vs CPU exact match for `N ≤ 4096`, all `B` |
| 4 | `reduce_partials_kernel` (§10) | Stage-2 output canonicalized correctly |
| 5 | Sweep `CHUNK` ∈ {256..16384}, pick fastest at target `B` | Result invariant to `CHUNK` |
| 6 | `commit_sparse_partial_kernel` (§9) | Sparse single = dense single on same `z` |
| 7 | Full `N = 2^{27}` dense commit benchmark, single and `B = 16` | Time within §13 model |
| 8 | Multi-GPU split (§12) | Per-GPU sum = single-GPU `c`, single and batched |
| 9 | Approach-2 large-batch kernel (`B ∈ {32, 64}`) if §13 batched perf warrants | Result identical to multiple `B = 16` calls |
| 10 | Rust bindings in `almost-goldilocks-cuda-rs`: `ajtai_commit_single`, `ajtai_commit_batched` | `cargo test` integration test against CPU |

## 18. Final design summary

```
Ring:                F_q[X] / (X^64 + 1)   with q = 2^64 - 2^32 - 31
Witness storage:     z_bits[2^27], one u64 = 64 binary coefficients
Sparse storage:      positions[K], each u64 ∈ [0, 2^33)
Matrix PRG:          ChaCha8 keyed by (seed, i, j), nonce = (i, j),
                     counter = block_idx ∈ [0, 8) per ring element,
                     rejection-sampled to F_q for unbiased output
Product:             selected negacyclic rotations (X^ℓ · M[i,j])
Output reduction:    two-stage (commit_dense_batched → reduce_partials)
Block layout:        gridDim = (num_chunks,), blockDim = 64 · B
                     each thread owns (b, r) = (batch, coefficient slot)
                     each thread holds 15 u64 accumulators (one per row i)
Batching:            first-class up to B = 16 single-block, B ≤ 64 with
                     Approach-2 multi-j-per-block. Amortizes PRG cost
                     linearly with B until adds dominate (~B = 43).
Field arithmetic:    agl_add / agl_sub / agl_neg from almost_goldilocks.cuh
                     (correct under double-fold)
Multi-GPU split:     partition j-axis; PRG is pure in (seed, i, j, block_idx)
```

The core formula to evaluate, with every implementation choice oriented
around it:

$$ \boxed{ c_{b,i} = \sum_{j=0}^{2^{27}-1} \sum_{\ell \,:\, z_{b,j,\ell}=1} X^\ell \cdot M_{i,j} } $$

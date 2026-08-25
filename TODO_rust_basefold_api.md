# Rust Basefold API — Comprehensive TODO

## Design Principle: GPU-Resident Data

**All intermediate data stays on GPU.** Host↔device transfers happen only at:
- **Input boundary**: polynomial evaluations enter GPU once
- **Output boundary**: proof struct (oracles, merkle roots, query data) exits GPU once
- **Fiat-Shamir sync points**: 3 scalars per round (c0, c1, c2) to host for transcript,
  1 scalar (challenge) back to GPU — unavoidable, ~24 bytes/round

Everything else — codewords, eq arrays, bh_evals, merkle trees, folded intermediates —
lives entirely in `DeviceBuffer` and never touches host memory.

---

## Current State

### What already exists

**Low-level kernel wrappers** (`src/basefold.rs` → `BasefoldBatch`):
All operate on `DeviceBuffer` — zero host copies internally.
- `bit_reverse_gl` / `bit_reverse_ext2` — in-place bit-reversal permutation
- `bhc_interpolate` — evaluations → coefficients + Type1 bh_evals
- `encode` — coefficients → codeword (Type1)
- `fold_gl` / `fold_mixed` / `fold_ext2` — one codeword folding step
- `sumcheck_interp_gl` / `sumcheck_interp_ext2` — [a, b] → [a, b-a] (in-place)
- `sumcheck_eval_gl` / `sumcheck_eval_mixed` / `sumcheck_eval_ext2` — evaluate at challenge, compact
- `sumcheck_product_gl` — compute degree-2 sum-check polynomial (partial block sums)
- `dot_product_gl` / `dot_product_mixed` — inner product (partial block sums)

**Supporting infrastructure** (other modules):
- `eq_lagrange::eq_dp_all_device` / `ext2_eq_dp_all_device` — eq(r, x) **on GPU** ✓
- `eq_lagrange::eq_dp_all` / `ext2_eq_dp_all` — eq(r, x) copies to host ✗ (don't use)
- `Poseidon2Batch::merkle_layer` — one merkle layer on GPU `DeviceBuffer` ✓
- `Poseidon2Ops::build_merkle_tree` — **per-layer host round-trip** ✗ (must replace)
- `DeviceBuffer<T>` — GPU memory (alloc, from_slice, to_vec, as_ptr/as_mut_ptr)
- `BasefoldTable` — random table generation + upload to GPU
- `FoldingEntry` — `{point, weight}` struct

### Data movement audit — problems to fix

| Problem | Where | Impact | Fix |
|---------|-------|--------|-----|
| Merkle tree per-layer round-trip | `Poseidon2Ops::build_merkle_tree` | ~2×log₂(N) transfers | Phase 0a: GPU-resident merkle tree |
| No device-to-device copy | `DeviceBuffer` | Can't copy codeword for bit-reverse | Phase 0b: Add `copy_from_device` |
| No sub-buffer pointer | `DeviceBuffer` | Need offset pointers for merkle layers | Phase 0b: Add `offset_ptr` |
| Partial-sum reduce on host | Not yet implemented | ~256 values per reduce | Phase 2: acceptable — only 3×256 scalars |

### What's missing for a usable commit→open→verify API

1. `DeviceBuffer` enhancements (d2d copy, offset pointer, partial read)
2. GPU-resident Merkle tree builder
3. Leaf-hashing FFI (raw codeword elements → Poseidon2 digests on GPU)
4. `BasefoldCommitment` struct
5. `basefold_commit()` orchestration
6. `BasefoldProof` / `BasefoldProofExt2` structs
7. Partial-sum reduction (block sums → 3 coefficients)
8. `basefold_open()` orchestration (sum-check + folding loop)
9. `basefold_open_ext2()` orchestration (mixed → ext2 transition)
10. Query phase (gather values + merkle paths from device)
11. Fiat-Shamir integration
12. Verifier
13. Integration tests

---

## Phase 0: Infrastructure Fixes

### 0a. `DeviceBuffer` Enhancements (`src/memory.rs`)

```rust
impl<T> DeviceBuffer<T> {
    /// Device-to-device copy: create a new buffer with same contents.
    /// Uses cudaMemcpyDeviceToDevice — no host involvement.
    pub fn clone_on_device(&self) -> Result<Self>;

    /// Copy from another device buffer (same size).
    /// Uses cudaMemcpyDeviceToDevice — no host involvement.
    pub fn copy_from_device(&mut self, src: &DeviceBuffer<T>) -> Result<()>;

    /// Get a raw pointer offset by `n` elements.
    /// Used for accessing sub-regions (e.g., merkle tree layers).
    /// SAFETY: caller must ensure offset < self.len.
    pub unsafe fn offset_ptr(&self, n: usize) -> *const T;
    pub unsafe fn offset_mut_ptr(&mut self, n: usize) -> *mut T;

    /// Read a single element from device (copies 1 element).
    /// Useful for reading reduction results without downloading full buffer.
    pub fn read_element(&self, index: usize) -> Result<T> where T: Default;

    /// Read a small contiguous slice from device (copies `len` elements).
    /// For extracting e.g. 4-element merkle root without downloading whole tree.
    pub fn read_slice(&self, offset: usize, len: usize) -> Result<Vec<T>>
        where T: Clone + Default;
}
```

**Required new FFI** (`wrapper.cu` + `ffi.rs`):
```c
int cuda_memcpy_dtod(void* dst, const void* src, size_t size);
```

### 0b. GPU-Resident Merkle Tree (`src/merkle.rs` — new module)

The current `Poseidon2Ops::build_merkle_tree` does **per-layer host round-trips**
(~40 transfers for 2^20 leaves). Replace with a fully GPU-resident builder.

```rust
/// Merkle tree stored entirely on GPU.
/// Layout: single contiguous DeviceBuffer with layers packed sequentially.
///
/// For N leaves: [layer0: N digests] [layer1: N/2 digests] ... [root: 1 digest]
/// Each digest = 4 × u64 (Poseidon2Hash).
/// Total size = (2N - 1) × 4 u64.
pub struct DeviceMerkleTree {
    d_tree: DeviceBuffer<u64>,     // All layers packed
    num_leaves: usize,
    layer_offsets: Vec<usize>,     // Byte offsets for each layer (computed, not stored on GPU)
}

impl DeviceMerkleTree {
    /// Build from leaf digests already on GPU.
    /// Input: DeviceBuffer of N × 4 u64 (leaf hashes).
    /// Only data movement: none (all on GPU).
    pub fn build(d_leaf_digests: &DeviceBuffer<u64>, num_leaves: usize) -> Result<Self>;

    /// Get the root hash. Copies exactly 4 u64 (32 bytes) from device.
    pub fn root(&self) -> Result<Poseidon2Hash>;

    /// Extract authentication path for one leaf.
    /// Copies exactly log₂(N) × 4 u64 from device (one sibling per layer).
    pub fn auth_path(&self, leaf_index: usize) -> Result<Vec<Poseidon2Hash>>;

    /// Extract authentication paths for multiple leaves (batched).
    /// More efficient than calling auth_path() in a loop.
    pub fn auth_paths_batch(&self, leaf_indices: &[usize]) -> Result<Vec<Vec<Poseidon2Hash>>>;

    /// Get raw device pointer to a specific layer (for kernel consumption).
    pub fn layer_ptr(&self, layer: usize) -> *const u64;
}
```

**Implementation**:
- Allocate single `DeviceBuffer` of size `(2N - 1) * 4` u64
- Copy leaf digests into first N×4 region (device-to-device)
- For each layer: call `Poseidon2Batch::merkle_layer` with offset pointers
  into the same contiguous buffer
- **Total host transfers: 0** during construction

### 0c. Leaf Hashing FFI

Codeword elements are raw `GoldilocksField` (1 u64 each) or `GoldilocksExt2` (2 u64).
Merkle trees need Poseidon2 digests (4 u64 each). Need a GPU kernel to hash leaves.

The CUDA side already has `poseidon2_build_merkle_tree_8` which does this internally,
but it's not exposed as a separate "hash leaves" step.

**Option A**: Expose `poseidon2_hash_leaves_ffi` — pads each element to width-8, permutes,
extracts first 4 elements as digest. New FFI function.

**Option B**: Group elements into chunks matching Poseidon2 rate and hash. E.g., group
4 consecutive `GoldilocksField` values as one Poseidon2 input, hash to 4-element digest.

**Option C**: Expose the full `poseidon2_build_merkle_tree_8` from C++ as one FFI call
that takes raw codeword data and returns a full merkle tree buffer.

→ Recommend **Option C** — it already exists in CUDA, avoids multi-step coordination,
and the C++ side already handles leaf padding + tree construction in one pass.

```c
// New FFI in wrapper.cu:
int poseidon2_build_merkle_tree_gl_ffi(
    const uint64_t* d_codeword,    // Raw codeword (1 u64 per element)
    uint64_t* d_tree,              // Output: full tree (pre-allocated, (2N-1)*4 u64)
    int num_leaves                  // Number of leaves
);

int poseidon2_build_merkle_tree_ext2_ffi(
    const uint64_t* d_codeword,    // Raw ext2 codeword (2 u64 per element)
    uint64_t* d_tree,              // Output: full tree
    int num_leaves
);
```

---

## Phase 1: Data Structures

### 1a. `BasefoldCommitment`
```rust
/// Commitment to a multilinear polynomial. All heavy data lives on GPU.
pub struct BasefoldCommitment {
    pub root: Poseidon2Hash,                   // Only data on host (32 bytes)
    d_codeword: DeviceBuffer<u64>,             // Codeword (Type1) — stays on GPU
    d_bh_evals: DeviceBuffer<u64>,             // BH evals (Type1) — stays on GPU
    d_merkle_tree: DeviceMerkleTree,           // Full merkle tree — stays on GPU
    pub num_vars: usize,
    pub log_rate: usize,
}
```

### 1b. `SumcheckOracle`
```rust
/// One round's degree-2 polynomial: p(X) = c0 + c1·X + c2·X²
/// These are small (3 scalars) — OK to live on host for Fiat-Shamir.
#[derive(Clone, Debug)]
pub struct SumcheckOracle<F: Copy> {
    pub c0: F,
    pub c1: F,
    pub c2: F,
}
```

### 1c. `BasefoldProof` (base field)
```rust
/// Opening proof. This is the final output — all data on host.
/// Serialized for transmission to verifier.
pub struct BasefoldProof {
    pub eval: GoldilocksField,
    pub sumcheck_oracles: Vec<SumcheckOracle<GoldilocksField>>,
    pub folded_roots: Vec<Poseidon2Hash>,
    pub final_codeword: Vec<GoldilocksField>,
    pub query_proofs: Vec<QueryProof<GoldilocksField>>,
}
```

### 1d. `BasefoldProofExt2`
```rust
pub struct BasefoldProofExt2 {
    pub eval: GoldilocksExt2,
    pub sumcheck_oracles: Vec<SumcheckOracle<GoldilocksExt2>>,
    pub folded_roots: Vec<Poseidon2Hash>,
    pub final_codeword: Vec<GoldilocksExt2>,
    pub query_proofs: Vec<QueryProof<GoldilocksExt2>>,
}
```

### 1e. `QueryProof`
```rust
pub struct QueryProof<F: Copy> {
    pub index: usize,
    /// (left, right) codeword pair for each round (initial + folded rounds).
    pub values: Vec<(F, F)>,
    /// Merkle authentication path for each round.
    pub merkle_paths: Vec<Vec<Poseidon2Hash>>,
}
```

---

## Phase 2: Partial-Sum Reduction

The GPU `sumcheck_product_*` kernels produce per-block partial sums (~256 blocks).
The reduction to 3 final coefficients is O(256) additions — trivially fast on CPU.

**Why this is acceptable on host**: The 3 coefficients MUST go to host anyway for
Fiat-Shamir (the challenger needs them to produce the next challenge). So the reduction
piggybacks on a transfer that's already required.

```rust
impl BasefoldBatch {
    /// Reduce GPU partial block sums to 3 final sum-check coefficients.
    /// Downloads ~256 partial sums per coefficient (768 u64 total ≈ 6 KB).
    /// Returns host-side coefficients for Fiat-Shamir consumption.
    pub fn reduce_sumcheck_partials_gl(
        partial_c0: &DeviceBuffer<u64>,
        partial_c1: &DeviceBuffer<u64>,
        partial_c2: &DeviceBuffer<u64>,
        num_blocks: usize,
    ) -> Result<SumcheckOracle<GoldilocksField>>;

    /// Same for extension field (~1536 u64 total ≈ 12 KB).
    pub fn reduce_sumcheck_partials_ext2(
        partial_c0: &DeviceBuffer<u64>,
        partial_c1: &DeviceBuffer<u64>,
        partial_c2: &DeviceBuffer<u64>,
        num_blocks: usize,
    ) -> Result<SumcheckOracle<GoldilocksExt2>>;

    /// Reduce dot-product partial sums to a single scalar.
    /// Downloads ~256 u64 (2 KB).
    pub fn reduce_dot_product_gl(
        partial: &DeviceBuffer<u64>,
        num_blocks: usize,
    ) -> Result<GoldilocksField>;

    /// Mixed dot-product reduction (ext2 result).
    pub fn reduce_dot_product_mixed(
        partial: &DeviceBuffer<u64>,
        num_blocks: usize,
    ) -> Result<GoldilocksExt2>;
}
```

---

## Phase 3: Commit

```rust
impl BasefoldCommitment {
    /// Commit to polynomial evaluations already on GPU.
    /// Primary API — avoids uploading evals from host.
    ///
    /// Data flow (all on GPU):
    ///   d_evals → bhc_interp → d_coeffs + d_bh_evals(Type1)
    ///   d_coeffs → encode → d_codeword(Type2) → bit_reverse → d_codeword(Type1)
    ///   d_codeword → merkle_tree → root (only root comes to host: 32 bytes)
    ///
    /// Transfers: 0 H→D, 32 bytes D→H (merkle root only).
    pub fn commit_device(
        d_evals: &DeviceBuffer<u64>,
        num_vars: usize,
        log_rate: usize,
    ) -> Result<Self>;

    /// Convenience: commit from host data.
    /// Transfers: 2^num_vars × 8 bytes H→D (one-time upload), 32 bytes D→H.
    pub fn commit(
        evals: &[GoldilocksField],
        num_vars: usize,
        log_rate: usize,
    ) -> Result<Self>;
}
```

**Steps for `commit_device`**:
1. Allocate `d_coeffs`, `d_bh_evals`, `d_codeword` — all on GPU
2. `bhc_interpolate(d_evals, d_coeffs, d_bh_evals, num_vars)` — GPU only
3. `encode(d_coeffs, d_codeword, num_vars, log_rate)` — GPU only
4. `bit_reverse_gl(d_codeword, num_vars + log_rate)` — GPU only
5. Build `DeviceMerkleTree` from `d_codeword` — GPU only
6. `root = merkle_tree.root()` — 32 bytes D→H
7. Drop `d_coeffs` (no longer needed)
8. Return `BasefoldCommitment { root, d_codeword, d_bh_evals, d_merkle_tree, ... }`

---

## Phase 4: Open (Base Field)

```rust
impl BasefoldCommitment {
    /// Generate opening proof for a base-field evaluation point.
    ///
    /// Per-round transfers (unavoidable for Fiat-Shamir):
    ///   D→H: 3 scalars (sumcheck oracle) + 32 bytes (merkle root) = 56 bytes
    ///   H→D: 1 scalar (challenge) = 8 bytes
    /// Total: ~64 bytes/round × num_rounds (e.g., 20 rounds = 1.3 KB)
    ///
    /// End transfers:
    ///   D→H: final codeword (small) + query pairs + merkle paths
    pub fn open(
        &self,
        point: &[GoldilocksField],
        table: &BasefoldTable,        // Must already be uploaded
        transcript: &mut impl BasefoldTranscript,
        num_queries: usize,
    ) -> Result<BasefoldProof>;
}
```

**Steps — data stays on GPU unless marked**:

1. Upload `point` → `d_point` (num_vars × 8 bytes H→D)
2. `eq_dp_all_device(d_point, num_vars)` → `d_eq` — **GPU only**
3. `d_bh_work = d_bh_evals.clone_on_device()` — **GPU→GPU** (d2d copy)
4. `bit_reverse_gl(d_bh_work)` — **GPU only** (Type1 → Type2)
5. `dot_product_gl(d_bh_work, d_eq)` → `d_partial` — **GPU only**
6. `eval = reduce_dot_product_gl(d_partial)` — **~2 KB D→H** (unavoidable: goes into proof)
7. `sumcheck_interp_gl(d_eq)` + `sumcheck_interp_gl(d_bh_work)` — **GPU only**
8. `sumcheck_product_gl(d_eq, d_bh_work)` → reduce → `oracle[0]` — **~6 KB D→H** (Fiat-Shamir)
9. `transcript.observe(oracle[0])` — host Fiat-Shamir
10. Store intermediate folded codewords on GPU for query extraction:
    `folded_codewords: Vec<DeviceBuffer<u64>>` and `folded_trees: Vec<DeviceMerkleTree>`
11. **For each round** (all heavy computation on GPU):
    a. `challenge = transcript.sample()` — host → 8 bytes (scalar)
    b. Allocate `d_eq_half`, `d_bh_half` — GPU
    c. `sumcheck_eval_gl(d_eq, challenge, d_eq_half)` — **GPU only**
    d. `sumcheck_eval_gl(d_bh_work, challenge, d_bh_half)` — **GPU only**
    e. `sumcheck_interp_gl(d_eq_half)` + `sumcheck_interp_gl(d_bh_half)` — **GPU only**
    f. `sumcheck_product_gl(d_eq_half, d_bh_half)` → reduce → `oracle[round+1]` — **~6 KB D→H**
    g. `fold_gl(d_codeword, table_level, challenge, d_folded)` — **GPU only**
    h. Build `DeviceMerkleTree` from `d_folded` — **GPU only**
    i. `root = tree.root()` — **32 bytes D→H**
    j. `transcript.observe(oracle[round+1])` + `transcript.observe(root)`
    k. Store `d_folded` and tree in `folded_codewords` / `folded_trees`
    l. Swap buffers: `d_eq = d_eq_half`, `d_bh_work = d_bh_half`, `d_codeword = d_folded`
12. **Final codeword**: `d_codeword.to_vec()` — small, **D→H** (goes into proof)
13. **Query phase** (Phase 5):
    - Sample indices from transcript
    - For each round's codeword + tree: extract pairs + merkle paths (**D→H**, into proof)
14. Drop all intermediate `DeviceBuffer`s

**Transfer summary for basefold_open** (num_vars = 20, log_rate = 1):
- H→D: point (160B) + challenges (20 × 8B = 160B) = **320 bytes**
- D→H: eval (8B) + oracles (21 × 24B = 504B) + roots (20 × 32B = 640B)
       + final_codeword (~small) + query data = **~2 KB + query data**
- GPU→GPU: all heavy work (codewords, eq, bh, merkle trees)

---

## Phase 5: Query Phase

All folded codewords and merkle trees are already on GPU (stored in Phase 4 step 10).
Query extraction does targeted small reads from GPU.

```rust
/// Extract query proof data from on-device codewords and merkle trees.
fn extract_queries(
    initial_tree: &DeviceMerkleTree,
    initial_codeword: &DeviceBuffer<u64>,
    folded_trees: &[DeviceMerkleTree],
    folded_codewords: &[DeviceBuffer<u64>],
    query_indices: &[usize],
    elem_size: usize,  // 1 for Fp, 2 for Fp²
) -> Result<Vec<QueryProof<...>>>;
```

**Per query, per round**: reads 2 codeword elements + 1 merkle auth path.
- Codeword pair: `read_slice(pair_idx * 2 * elem_size, 2 * elem_size)` — 16 bytes (Fp) or 32 bytes (Fp²)
- Auth path: `tree.auth_path(query_idx)` — log₂(N) × 32 bytes

**Optimization**: Batch all reads per round into a single `cudaMemcpy` using a gather
kernel or by downloading the full small codeword when it's small enough.

**Threshold strategy**:
- If codeword is large (>1K elements): use targeted reads via `read_slice`
- If codeword is small (≤1K elements): download entire codeword via `to_vec`, extract on host

---

## Phase 6: Open (Extension Field)

```rust
impl BasefoldCommitment {
    /// Generate opening proof for an extension-field evaluation point.
    /// Same transfer characteristics as base field open.
    pub fn open_ext2(
        &self,
        point: &[GoldilocksExt2],
        table: &BasefoldTable,
        transcript: &mut impl BasefoldTranscript,
        num_queries: usize,
    ) -> Result<BasefoldProofExt2>;
}
```

**Key difference — round 0 is "mixed" (F_p data, F_{p²} challenge)**:

Steps 1-6: same as base field but with ext2 eq and mixed dot product

Round 0 (mixed):
- `sumcheck_eval_mixed(d_bh_fp, challenge_ext2)` → `d_bh_ext2` — **GPU only** (F_p → F_{p²})
- `sumcheck_eval_ext2(d_eq_ext2, challenge_ext2)` → `d_eq_half` — **GPU only**
- `fold_mixed(d_codeword_fp, challenge_ext2)` → `d_codeword_ext2` — **GPU only**

Rounds 1+: all pure ext2 on GPU
- `sumcheck_interp_ext2`, `sumcheck_product_ext2`, `sumcheck_eval_ext2`, `fold_ext2`

**No additional host transfers compared to base field** — the mixed→ext2 transition
is entirely on-device.

---

## Phase 7: Fiat-Shamir Integration

```rust
/// Trait for challenge generation during basefold opening.
/// Implementations may use GPU-accelerated or CPU Poseidon2.
pub trait BasefoldTranscript {
    fn observe_field(&mut self, value: GoldilocksField) -> Result<()>;
    fn observe_ext2(&mut self, value: GoldilocksExt2) -> Result<()>;
    fn observe_hash(&mut self, hash: &Poseidon2Hash) -> Result<()>;
    fn sample_challenge(&mut self) -> Result<GoldilocksField>;
    fn sample_challenge_ext2(&mut self) -> Result<GoldilocksExt2>;
}

/// Implement for existing DuplexChallenger (GPU-backed Poseidon2 state).
impl BasefoldTranscript for DuplexChallenger { ... }

/// Simple deterministic transcript for testing (no Poseidon2, just xorshift).
pub struct TestTranscript { state: u64 }
impl BasefoldTranscript for TestTranscript { ... }
```

**Transcript protocol** (must match between prover and verifier):
1. Observe commitment root (4 field elements)
2. Observe evaluation point (num_vars field elements)
3. For each round:
   a. Observe sum-check oracle (c0, c1, c2)
   b. Sample challenge
   c. Observe folded codeword root (4 field elements)
4. Sample query indices

---

## Phase 8: Verifier

The verifier is CPU-side (small work: O(num_vars + num_queries × log N)).
No GPU needed.

```rust
pub struct BasefoldVerifier;

impl BasefoldVerifier {
    /// Verify a base-field opening proof. All CPU.
    pub fn verify(
        root: &Poseidon2Hash,
        point: &[GoldilocksField],
        proof: &BasefoldProof,
        table: &BasefoldTable,
        transcript: &mut impl BasefoldTranscript,
    ) -> Result<bool>;

    /// Verify an extension-field opening proof. All CPU.
    pub fn verify_ext2(
        root: &Poseidon2Hash,
        point: &[GoldilocksExt2],
        proof: &BasefoldProofExt2,
        table: &BasefoldTable,
        transcript: &mut impl BasefoldTranscript,
    ) -> Result<bool>;
}
```

**Verification steps** (all host-side, O(num_vars + num_queries)):
1. Re-derive challenges from transcript (same protocol as prover)
2. **Sum-check consistency**: `oracle[i].c0 + oracle[i].c1 + oracle[i].c2` checks
3. **Query checks**: For each query at each round, verify Lagrange interpolation
4. **Merkle path checks**: Verify each auth path against committed/folded roots
5. **Final codeword check**: Verify consistency with last sum-check output

---

## Phase 9: Integration Tests

### 9a. Base field round-trip
```rust
#[test]
fn test_basefold_commit_open_verify() {
    // 1. Generate random polynomial (num_vars = 10)
    // 2. Generate + upload folding table
    // 3. commit() → BasefoldCommitment (only root on host)
    // 4. Choose random evaluation point
    // 5. open() → BasefoldProof
    // 6. verify() → assert true
}
```

### 9b. Extension field round-trip
```rust
#[test]
fn test_basefold_ext2_commit_open_verify() {
    // Same but with GoldilocksExt2 evaluation point
}
```

### 9c. Soundness test
```rust
#[test]
fn test_basefold_reject_bad_eval() {
    // Commit, open, tamper with proof.eval, verify → assert false
}
```

### 9d. Various sizes
```rust
#[test]
fn test_basefold_various_sizes() {
    // num_vars = 4, 8, 12, 16, 20
}
```

### 9e. GPU memory test
```rust
#[test]
fn test_no_unnecessary_transfers() {
    // Instrument DeviceBuffer to count H↔D transfers
    // Verify commit does exactly 1 H→D (evals) + 1 D→H (root)
    // Verify open does O(num_rounds) small transfers only
}
```

---

## Implementation Order

| Step | Phase | Description | Depends On | Complexity |
|------|-------|-------------|------------|-----------|
| 1 | 0a | `DeviceBuffer` enhancements (d2d copy, offset_ptr, read_slice) | — | Low |
| 2 | 0b | GPU-resident `DeviceMerkleTree` | Phase 0a | Medium |
| 3 | 0c | Leaf-hashing FFI (poseidon2_build_merkle_tree_*_ffi) | Phase 0b | Medium |
| 4 | 1 | Data structures (Commitment, Proof, Oracle, QueryProof) | — | Low |
| 5 | 2 | Partial-sum reduction helpers | Phase 1 | Low |
| 6 | 3 | `basefold_commit` / `commit_device` | Phases 0-2 | Medium |
| 7 | 7 | Fiat-Shamir transcript trait + test impl | — | Low |
| 8 | 4 | `basefold_open` (base field) | Phases 2-3, 7 | High |
| 9 | 5 | Query phase extraction | Phase 0b | Medium |
| 10 | 6 | `basefold_open_ext2` | Phases 4, 7 | High |
| 11 | 8 | Verifier | Phases 1, 7 | Medium |
| 12 | 9 | Integration tests | All | Medium |

---

## Transfer Budget Summary

For a typical instance (num_vars=20, log_rate=1, 20 rounds, 32 queries):

| Operation | H→D | D→H | D→D (GPU internal) |
|-----------|------|------|---------------------|
| **commit** | 8 MB (evals) | 32 B (root) | ~24 MB (interp, encode, merkle) |
| **open per round** | 8 B (challenge) | 56 B (oracle + root) | ~megabytes (fold, merkle, sumcheck) |
| **open total** | 320 B | ~2 KB | ~all heavy work |
| **query extraction** | 0 | ~50 KB (pairs + paths) | 0 |
| **TOTAL** | ~8 MB | ~52 KB | Everything else |

The only significant H→D transfer is the initial polynomial upload.
Everything else is sub-kilobyte per round.

---

## Open Design Questions

1. **Leaf hashing strategy**: Expose C++ `poseidon2_build_merkle_tree_8` as monolithic FFI
   (Option C) or build in Rust layer-by-layer using `Poseidon2Batch::merkle_layer` with
   offset pointers (Option B)?
   → Recommend Option C for simplicity and to match C++ behavior exactly.

2. **Memory pool / arena**: For the folding loop, we allocate + free many DeviceBuffers
   per round. A GPU memory pool/arena could reduce `cudaMalloc` overhead.
   → Defer. `cudaMalloc` is typically fast enough. Optimize only if profiling shows issue.

3. **Async streams**: Currently all on default stream (synchronous). Could pipeline
   rounds using CUDA streams (e.g., start merkle tree build while doing next sumcheck).
   → Defer. Correctness first.

4. **Batch opening**: Multiple polynomials at same point.
   → Defer. Single-polynomial API first.

## Completion Criteria

- [ ] `DeviceBuffer` has d2d copy, offset_ptr, read_slice
- [ ] `DeviceMerkleTree` builds on GPU with 0 intermediate host transfers
- [ ] `BasefoldCommitment::commit()` does exactly 1 H→D + 1 D→H
- [ ] `BasefoldCommitment::open()` does O(num_rounds) tiny transfers only
- [ ] `BasefoldCommitment::open_ext2()` works for extension field points
- [ ] `BasefoldVerifier::verify()` accepts valid proofs, rejects tampered
- [ ] Integration tests pass for num_vars = 4, 10, 16
- [ ] All `cargo test` passes

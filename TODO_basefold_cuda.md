# Basefold CUDA Kernel Implementation Plan

## Overview

Implement GPU-accelerated kernels for the Basefold polynomial commitment scheme
over the Goldilocks field, with extension field opening support (GoldilocksExt2).

**Files to create:**
- `cuda/basefold.cuh` — device functions + kernel declarations
- `cuda/basefold_kernels.cu` — kernel implementations + host wrappers + tests
- `goldilocks-cuda-rs/` updates — Rust FFI bindings + high-level API

**Existing primitives we reuse (no reimplementation needed):**
- `goldilocks.cuh` / `extension.cuh` — field arithmetic
- `poseidon2_build_merkle_tree_8` — Merkle tree over F_p
- `poseidon2_build_merkle_tree_ext2` — Merkle tree over F_{p^2}
- `eq_dp_all` / `ext2_eq_dp_all` — eq(r,x) polynomial construction
- `gl_dot_product_kernel` — inner product / sum reduction

---

## Phase 1: Bit-Reversal Permutation

**Kernel:** `bit_reverse_permute_kernel<T>(T* data, int log_n)`

Converts between Type1 (folding-pair-adjacent) and Type2 (encoding-friendly)
orderings. Each thread at index `i` swaps `data[i]` with `data[bit_reverse(i)]`
(only when `i < bit_reverse(i)` to avoid double-swaps).

Needed for both F_p and F_{p^2} variants (template on element type).

---

## Phase 2: Boolean Hypercube Interpolation (Evals → Coefficients)

Converts evaluation-form polynomial to coefficient-form over the Boolean
hypercube. This is the inverse multilinear transform.

**Algorithm** (from `interpolate_over_boolean_hypercube_with_copy`):
1. First pass (pairs): `coeffs[2i] = evals[2i]`, `coeffs[2i+1] = evals[2i+1] - evals[2i]`
2. For levels k = 2..log_n: for each chunk of size 2^k, second half elements:
   `chunk[j] = chunk[j] - chunk[j - half_chunk]`
3. Also produce a copy of evals bit-reversed to Type1 order (for the commitment).

**Kernels:**
- `bhc_interp_first_pass_kernel(evals, coeffs, n)` — pairwise differences
- `bhc_interp_layer_kernel(coeffs, level, n)` — one butterfly level
- Combined with Phase 1 bit-reversal on the evals copy

Total: 1 kernel launch for first pass + (log_n - 1) launches for layers + 1 bit-reversal.

Only F_p variant needed (commit is always over base field).

---

## Phase 3: Foldable Domain Encoding

Encodes the coefficient vector into a codeword over the foldable domain.
Two modes supported:

### Mode A: Repetition + Random Fold (AES-based table)

**Step 1 — Repetition code:**
- `repetition_encode_kernel(coeffs, output, rate, k)`
- Each coefficient replicated `rate = 2^log_rate` times.
- Thread i writes: `output[i] = coeffs[i / rate]`

**Step 2 — Iterative folding layers:**
- `foldable_domain_layer_kernel(data, d_table, level, chunk_size, n)`
- For log_n levels, each level applies butterfly on pairs using table values:
  ```
  lhs = -data[j]  (or twiddle factor for binary_rs)
  data[j]          = data[j - half] + lhs
  data[j - half]  += data[j_original]
  ```
- Table values pre-uploaded to device memory.

### Mode B: RS Basecode + Fold

**Step 1 — RS basecode:**
- `rs_basecode_encode_kernel(coeffs, output, chunk_size, rate_times_chunk)`
- Each thread evaluates one chunk polynomial at one domain point using Horner's method.
- Domain points: 1, 2, 3, ..., chunk_size * rate (generated in-kernel or pre-uploaded).

**Step 2 — Folding layers:**
- Same `foldable_domain_layer_kernel` but fewer levels (starts from larger chunks).

**Step 3 — Bit-reversal** to convert Type2 → Type1 (reuse Phase 1).

Only F_p variants needed (encoding is always over base field per Remark 1).

---

## Phase 4: Commit (Orchestration)

This is a host-side function that chains Phases 2–3 + existing Merkle tree:

```
basefold_commit(d_evals, num_vars, log_rate, d_table, ...) {
    1. bhc_interp(d_evals, d_coeffs, d_bh_evals_type1)     // Phase 2
    2. encode(d_coeffs, d_codeword, log_rate, d_table)      // Phase 3
    3. bit_reverse(d_codeword)                               // Phase 1
    4. poseidon2_build_merkle_tree_8(d_tree, num_leaves)     // existing
}
```

Returns: d_codeword (Type1), d_bh_evals (Type1), d_tree (Merkle).

---

## Phase 5: Sum-Check Kernels (Base Field)

The sum-check proves `eval = Σ_x f(x) · eq(x, r)` by reducing one variable
per round. Each round produces a degree-2 univariate polynomial (3 coefficients).

### Kernel 5a: `sumcheck_interp_kernel(data, n)`
Single-level Boolean hypercube interpolation (convert eval-form to coeff-form).
For each pair `[a, b]`: write `[a, b - a]`.
This is `one_level_interp_hc`.

### Kernel 5b: `sumcheck_eval_kernel(data, challenge, n)`
Evaluate at challenge and compact. For each pair `[c, d]`:
write `c + challenge * d` into position `i` (half the size).
This is `one_level_eval_hc`.

### Kernel 5c: `sumcheck_product_kernel(eq_coeffs, bh_coeffs, partial_sums, n)`
Compute the 3 coefficients of the degree-2 sum-check polynomial.
For each pair index i, compute:
```
c0_i = eq_even[i] * bh_even[i]
c1_i = eq_even[i] * bh_odd[i] + eq_odd[i] * bh_even[i]
c2_i = eq_odd[i] * bh_odd[i]
```
Then parallel-reduce all `(c0_i, c1_i, c2_i)` into 3 global sums.
This is `parallel_pi`.

### Sum-check round flow:
- **First round:** interp(eq) + interp(bh) → product → 3 coefficients
- **Challenge round:** eval(eq, α) + eval(bh, α) → interp(eq) + interp(bh) → product → 3 coefficients
- Repeat for num_rounds.

---

## Phase 6: Basefold Codeword Folding (Base Field)

**Kernel:** `basefold_fold_kernel(d_codeword, d_table_w_weights, challenge, d_output, level, n)`

For each pair index i:
```
x0 = table[i].point
w  = table[i].weight       // precomputed 1/(x1 - x0)
val0 = codeword[2*i]
val1 = codeword[2*i + 1]
output[i] = val0 + (challenge - x0) * (val1 - val0) * w
```

This is `basefold_one_round_by_interpolation_weights` — Lagrange interpolation
of the line through `(x0, val0), (x1, val1)` evaluated at `challenge`.

Output is half the size of input. One launch per round.

---

## Phase 7: Extension Field Opening Support

Per Remark 1 & 2 from remark.tex: when the evaluation point z is in F_{p^m},
the protocols run over the extension field. The encoding lifts component-wise
and retains its minimum distance.

### Data flow with extension field:

```
Round 0 (initial state):
  bh_evals  ∈ F_p^{2^d}          (polynomial evals, base field)
  eq        ∈ F_{p^2}^{2^d}      (eq polynomial at extension point)
  codeword  ∈ F_p^{N}            (base field encoding)
  challenge ∈ F_{p^2}            (extension field Fiat-Shamir)

After 1st challenge round:
  bh_evals  ∈ F_{p^2}^{2^{d-1}}  (evaluated at ext challenge → promoted)
  eq        ∈ F_{p^2}^{2^{d-1}}
  codeword  ∈ F_{p^2}^{N/2}      (folded with ext challenge → promoted)

Rounds 2+:
  Everything in F_{p^2}            (standard extension field operations)
```

### Kernel 7a: `sumcheck_product_mixed_kernel(eq_ext2, bh_Fp, partial_sums_ext2, n)`
First sum-check round with mixed types: bh in F_p, eq in F_{p^2}.
Multiplication is scalar_mul (F_p × F_{p^2} → F_{p^2}).

### Kernel 7b: `sumcheck_eval_mixed_kernel(data_Fp, challenge_ext2, output_ext2, n)`
First evaluation round: F_p data evaluated at F_{p^2} challenge.
Result: `output[i] = gl_to_ext2(c) + challenge_ext2 * gl_to_ext2(d)`, i.e.,
promote base field values then evaluate.

### Kernel 7c: `basefold_fold_mixed_kernel(codeword_Fp, d_table, challenge_ext2, output_ext2, n)`
First codeword fold: F_p codeword folded with F_{p^2} challenge.
Lagrange interpolation: table points are F_p, values are F_p, challenge is F_{p^2}.
```
x0, w ∈ F_p;   val0, val1 ∈ F_p;   challenge ∈ F_{p^2}
result = gl_to_ext2(val0) + ext2_mul_scalar(challenge - gl_to_ext2(x0), (val1 - val0) * w)
```
Output: F_{p^2} codeword.

### Kernel 7d: `sumcheck_interp_ext2_kernel(data_ext2, n)`
Same as 5a but over F_{p^2}.

### Kernel 7e: `sumcheck_eval_ext2_kernel(data_ext2, challenge_ext2, n)`
Same as 5b but over F_{p^2}.

### Kernel 7f: `sumcheck_product_ext2_kernel(eq_ext2, bh_ext2, partial_sums_ext2, n)`
Same as 5c but over F_{p^2}.

### Kernel 7g: `basefold_fold_ext2_kernel(codeword_ext2, d_table, challenge_ext2, output_ext2, n)`
Same as Phase 6 but over F_{p^2}. Table points/weights stay F_p (they come from
the base-field encoding), but codeword values and challenge are F_{p^2}.

### Merkle tree for extension field:
Already have `poseidon2_build_merkle_tree_ext2` — reuse directly.

### EQ polynomial for extension field:
Already have `ext2_eq_dp_all` — reuse directly.

---

## Phase 8: Open (Prover-Side Orchestration)

Host function that chains the above kernels:

### Base field opening: `basefold_open(...)`
```
1. eq = eq_dp_all(point)                              // existing
2. eval = dot_product(bh_evals, eq)                    // existing
3. interp(eq); interp(bh_evals)                        // 5a
4. sumcheck_oracle[0] = product(eq, bh_evals)          // 5c
5. for round in 0..num_rounds:
     a. write root + sumcheck_oracle to transcript     // host
     b. challenge = transcript.squeeze()               // host
     c. eval(eq, challenge); eval(bh_evals, challenge) // 5b
     d. interp(eq); interp(bh_evals)                   // 5a
     e. sumcheck_oracle[round+1] = product(eq, bh)     // 5c
     f. fold codeword at challenge                     // Phase 6
     g. merkle tree on new codeword                    // existing
6. query phase (extract values at random indices)      // largely host-side
```

### Extension field opening: `basefold_open_ext2(...)`
```
1. eq = ext2_eq_dp_all(point_ext2)                     // existing
2. eval = ext2_dot_product(bh_evals_Fp_as_ext2, eq)    // new: mixed dot product
3. interp(eq_ext2); interp(bh_evals_Fp)                // 5a + 7d
4. sumcheck_oracle[0] = product_mixed(eq, bh)          // 7a
5. round 0:
     a. write to transcript                            // host
     b. challenge_ext2 = transcript.squeeze_ext2()     // host
     c. eval_mixed(bh_evals_Fp, challenge_ext2)        // 7b  → bh now F_{p^2}
     d. eval_ext2(eq, challenge_ext2)                  // 7e
     e. interp(eq_ext2); interp(bh_ext2)               // 7d
     f. sumcheck_oracle[1] = product_ext2(eq, bh)      // 7f
     g. fold_mixed(codeword_Fp, challenge_ext2)        // 7c  → codeword now F_{p^2}
     h. merkle tree ext2 on new codeword               // existing
6. rounds 1..num_rounds:
     a–g. same as base field but all ext2 variants     // 7d,7e,7f,7g
7. query phase                                         // host-side
```

---

## Phase 9: Table Generation

The folding table defines the foldable code structure. Two options:

### Option A: AES-CTR random table (host-generated, uploaded to GPU)
- Generate on host using ChaCha8/AES-CTR (matching Rust reference).
- Compute weights: `w[i] = inverse(x1[i] - x0[i])`.
- Upload flat array of `(point, weight)` pairs to device.
- **No CUDA kernel needed** — CPU generation + cudaMemcpy.

### Option B: Binary RS twiddle table
- Generate from binary subspace structure.
- Can also be host-generated and uploaded.

**Device-side table layout:**
```
struct FoldingEntry {
    GoldilocksField point;   // x0 value
    GoldilocksField weight;  // 1 / (x1 - x0)
};
// d_table[level] → array of FoldingEntry, length 2^level
// Flattened: d_table_flat[offset_for_level + i]
```

---

## Phase 10: Query Phase & Merkle Path Extraction

Mostly host-side: given random query indices, extract codeword values and
Merkle authentication paths.

### Kernel 10a: `basefold_extract_queries_kernel(d_codeword, d_query_indices, d_output, num_queries)`
For each query index q, extract the pair `(codeword[2*(q/2)], codeword[2*(q/2)+1])`.
Parallel over queries. Run once per oracle (initial codeword + each folded oracle).

### Merkle path extraction:
- `merkle_extract_path_kernel(d_tree, query_index, d_path, tree_depth)`
- For each level, extract the sibling hash.
- Parallel over queries.

F_p and F_{p^2} variants needed.

---

## Phase 11: Verify (Verifier-Side)

The verifier is typically CPU-side (small work), but for batch verification
with many queries, GPU acceleration helps.

### Kernel 11a: `basefold_verify_query_kernel(...)`
For each query, verify interpolation consistency across all rounds:
```
For each round i:
  interpolate2(x0, val0, x1, val1, fold_challenge[i]) == next_oracle_val
```
Parallel over queries.

### Kernel 11b: `merkle_verify_path_kernel(...)`
Verify Merkle authentication paths. Parallel over queries × rounds.

### Host-side verification:
- Sum-check consistency: `degree_2_zero_plus_one(oracle[0]) == eval`, etc.
- Final oracle check / virtual_open.
- These are O(num_vars) work — stay on CPU.

---

## Phase 12: Rust Wrapper (`goldilocks-cuda-rs/`)

### FFI bindings (`ffi.rs` additions):
```rust
extern "C" {
    // Commit
    fn basefold_commit(d_evals: *mut u64, num_vars: i32, log_rate: i32,
                       d_table: *const u64, d_tree: *mut u64, ...) -> i32;
    // Open (base field)
    fn basefold_open(d_codeword: *const u64, d_bh_evals: *const u64,
                     d_point: *const u64, d_table: *const u64, ...) -> i32;
    // Open (extension field)
    fn basefold_open_ext2(d_codeword: *const u64, d_bh_evals: *const u64,
                          d_point_ext2: *const u64, d_table: *const u64, ...) -> i32;
    // Individual kernels exposed for flexibility
    fn basefold_fold(...) -> i32;
    fn basefold_fold_ext2(...) -> i32;
    // etc.
}
```

### High-level API (`basefold.rs`):
```rust
pub struct BasefoldProver { ... }
impl BasefoldProver {
    fn commit(&self, poly: &[GoldilocksField]) -> BasefoldCommitment;
    fn open(&self, comm: &BasefoldCommitment, point: &[GoldilocksField],
            eval: GoldilocksField) -> BasefoldProof;
    fn open_ext2(&self, comm: &BasefoldCommitment, point: &[GoldilocksExt2],
                 eval: GoldilocksExt2) -> BasefoldProof;
}
pub struct BasefoldVerifier { ... }
impl BasefoldVerifier {
    fn verify(&self, ...) -> Result<(), Error>;
    fn verify_ext2(&self, ...) -> Result<(), Error>;
}
```

---

## Implementation Order

1. **Phase 1** — Bit-reversal permutation
2. **Phase 2** — Boolean hypercube interpolation
3. **Phase 3** — Foldable domain encoding (Mode A first, Mode B later)
4. **Phase 4** — Commit orchestration + test against Rust reference
5. **Phase 5** — Sum-check kernels (base field)
6. **Phase 6** — Codeword folding (base field)
7. **Phase 8** — Open orchestration (base field) + test against Rust reference
8. **Phase 9** — Table generation
9. **Phase 7** — Extension field kernels (mixed + pure ext2)
10. **Phase 8** — Open orchestration (ext2) + test
11. **Phase 10** — Query phase kernels
12. **Phase 11** — Verifier kernels
13. **Phase 12** — Rust FFI + API

---

## Testing Strategy

- **Unit tests** for each kernel against Rust reference (same inputs → same outputs).
- **Round-trip test**: commit → open → verify for small instances (num_vars = 10–15).
- **Extension field test**: commit (F_p) → open_ext2 (F_{p^2} point) → verify_ext2.
- **Performance benchmarks**: num_vars = 20–25, compare wall-clock vs Rust CPU.

## Completion
Only output <promise>PHASE 1 to 12 are all completed</promise> when:
1. all the above tasks are completed
2. all the tests are passed
3. all compile and run successfully

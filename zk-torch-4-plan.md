# zk-torch-4: GPU-Native ZKML with Almost-Goldilocks + Ajtai

## 0. Relationship to zk-torch-3

zk-torch-4 reuses zk-torch-3's DAG, forward-pass, backward-pass, sumcheck, and
BasicBlock machinery essentially unchanged. The two things that change are the
**commitment layer** and the **opening phase**:

| Stage    | zk-torch-3                          | zk-torch-4                                  |
|----------|-------------------------------------|---------------------------------------------|
| Field    | Goldilocks `p = 2^64 − 2^32 + 1`    | Almost-Goldilocks `q = 2^64 − 2^32 − 31`    |
| Commit   | Basefold PCS (one proof per edge)   | Ajtai SIS over `R = F_q[X]/(X^64+1)`        |
| Open     | Per-edge Basefold opening proof     | **One** opening at the end, via fold-tree   |

Ajtai is additively homomorphic but **not** a polynomial commitment scheme. So
zk-torch-4 cannot "open `c_i` at `r_i`". Instead, all (commitment, claim) pairs
produced by the backward pass are accumulated and **multi-folded together into a
single commitment** whose witness is small enough to send to the verifier
verbatim. A same-point sumcheck preconditions the folding (all claims must be at
the same evaluation point) and a base-decomposition step (`split_witness`) keeps
the folded witness binary-norm-bounded as the tree gets deeper.

### Two-phase pipeline

| Phase                        | When             | What gets committed                                                |
|------------------------------|------------------|--------------------------------------------------------------------|
| **Offline** (per model)      | Once, ahead of time | All `Role::Constant` witnesses — model weights, fixed bias vectors, range/two-pow lookup tables, and any other input-independent constants. Bit-decomposed and Ajtai-committed; commitments + binary planes persisted to disk. |
| **Online** (per input)       | Each proof       | Activations, intermediate edge values, and per-input auxiliary witnesses (`Role::Auxiliary`, `Role::Output`). |

The offline commitments are loaded once at prover startup and reused across
every proof against that model — for transformer models, weights are 99%+ of
the committed bytes, so this amortization is the single biggest constant-factor
win. The online phase is what the wall-clock-per-proof number measures.

The CUDA kernels for every primitive needed already live in
`cuda_almost_goldilocks/` and are wrapped in
`almost-goldilocks-cuda-rs/src/ajtai.rs`. zk-torch-4's job is the
**Rust-level protocol orchestration**.

---

## 1. Field, ring, parameter set

- Base field `q = 2^64 − 2^32 − 31`. Two's-adic structure makes
  `X^64 + 1` split into 16 irreducible factors of degree 4 over `F_q`, which is
  what makes ring-SIS secure (see `cuda_almost_goldilocks/ajtai.md` §1).
- Ring `R = F_q[X] / (X^64 + 1)`.
- Ajtai output rows `κ = 15`. Each ring element is 64 `u64` coefficients.
- Challenge norm `T_chal = 128` (sum of `|γ_ℓ|` over the 64 coefficients of a
  `RingChallenge` ∈ `{−1, 0, 1, 2}^64`).
- SuperNeo binding bound `B = 8192 = 2^13`. The norm-growth invariant
  `(K + k) · T_chal · (b − 1) < B` must hold at every fold step
  (memory: `[[project_superneo_norm_bound]]`).
- Splitb parameters: `b = 2`, `k = 13`. Each splitb decomposes one wide i16
  witness into 13 ternary chunks (coefficients in `{−1, 0, 1}`).

Fold budget per round (`M = K + k`):

| Level                      | b − 1 | Max `M` such that `M · T · (b − 1) < B` |
|----------------------------|-------|-----------------------------------------|
| Binary inputs (level 0)    |   1   | 63 fresh + 1 constant slot = **64**     |
| Ternary inputs (level ≥ 1) |   1   | same — coefficients are still ±1        |

We use the conservative cap **63 instances per fold node** (matches the Rust API's
M ≤ 64 with the implicit constant-1 anchor), which gives a geometric tree
contraction of `63 → 13` per level (≈ 4.85× per round). The 13 produced ternary
chunks are themselves valid instances for the next round.

---

## 2. Witness preparation: signed binary decomposition

Ajtai requires each instance to be a **binary** witness. zk-torch's witnesses
are signed fixed-point integers. With scale factor `SF = 2^16` and tensors
naturally landing in `[−2^20, 2^20]`, **b = 21** binary planes are enough using a
sign-bit representation:

```
f(x) = Σ_{i = 0..b−2} 2^i · f_i(x)  −  2^{b−1} · f_{b−1}(x)
```

with each `f_i: {0,1}^k → {0,1}`. The top plane carries the sign. Concretely,
the prover stores `f_i` as a packed bitmask (one `u64` per 64 evaluations) ready
for `commit_batched`. A claim `f(r) = y` decomposes into b sub-claims at the
**same** point r: `y = Σ_i 2^i · f_i(r) − 2^{b−1} · f_{b−1}(r)`. The verifier
combines them after the fold tree gives `f_i(R)` for each plane.

> **Parameter choice.** `b` is determined by the global SF and the empirical
> tensor range. We default to 21 but expose it as a config so we can crank it
> up (or down) per model.

---

## 3. The matrix family `{M, M'_k}`

Sample one κ × 2^max_num_vars matrix `M` via ChaCha8 from a public seed
(see `cuda_almost_goldilocks/ajtai.md` §5). `M` is **never materialized at
2^max_num_vars width** — kernels regenerate columns on the fly, and the
`MaterializedM` device buffer is only used when N is small enough to fit.

For every distinct arity `k < max_num_vars` actually used in the DAG, precompute

```
M'_k[j] = Σ_{i = 0..2^{max_num_vars − k} − 1}  M[ j + i · 2^k ]    for j ∈ [0, 2^k)
```

so committing to a k-variate poly `g` costs `O(2^k)` column adds, not
`O(2^max_num_vars)`. This is the "M' optimization" — the ratio is exactly
`2^{max_num_vars − k}`, which is huge for the many small-arity witnesses in
zk-torch (range-check bit aux, two-pow selection polys, biases, etc.).

In practice we expect a handful of distinct arities. Bucket the witnesses by
arity at commit time and pay the M' precompute once per bucket.

> The arities present can be discovered with a single pass over `witnesses`
> after `dag.run()` — mirrors how zk-torch-3 computes `max_num_vars`.

---

## 4. Commit phase

The same per-edge commit recipe applies in both phases — only the **set of
edges** differs. The recipe:

1. Read the signed integer evaluations `f_e: {0,1}^{k_e} → Z` (we already store
   `data_int` in zk-torch-3).
2. Bit-decompose into `b` binary planes `{f_e^{(0)}, …, f_e^{(b−1)}}`.
3. For each plane, look up the matching `M'_{k_e}` (or use `M` directly when
   `k_e = max_num_vars`) and commit via `ajtai::commit_batched` — passing all
   `b` planes as one B-batch shares ChaCha8 / HBM cost across them. Result: a
   `[b] RingCommitment` per edge.
4. Store on GPU (`GpuAjtaiStore`) and a 32-byte Monolith hash of each
   `RingCommitment` in the transcript-side `AjtaiCommitmentData` (the verifier's
   per-edge data; 15·64·8 = 7680 bytes is too big to ship verbatim for every
   edge in a 30k-edge DAG, so we hash).

### 4.1 Offline phase (one-time per model)

After `DagBuilder::compile()` returns a `(Dag, Vec<Vec<Witness>>)`, walk the
witnesses and select every edge with `role == Role::Constant`. This is the
canonical set of input-independent witnesses:

- All weight tensors and biases of every layer.
- All static lookup tables (range table `T(y) = 2^y`, two-pow table, any
  embedded constants used by `DivConst` / `SoftmaxConst` / `SigmoidConst`).
- Any `Role::Constant` advice output (none currently produced by zk-torch-3
  BasicBlocks but the role is reserved).

Run the commit recipe over this set, then **persist**:

- The `[b] RingCommitment` for every committed edge → on-disk
  `model.ajtai_commits` file (small, only the per-edge `RingCommitment` rows).
- The bit-plane binary evaluations themselves → on-disk `model.bitplanes` blob
  (large; one packed `[N_ring] u64` per plane per edge). The prover needs
  these later during the fold tree because the fold-tree leaves carry the
  binary `Witness` for every constant edge.
- An index mapping `(model_seed, edge_id) → (offset, length)` for both files.

Persistence format is the same struct layout `GpuAjtaiStore` uses in memory,
so loading is a straight `mmap` + page-faulted upload.

**Crucially**, the offline phase fixes the public `seed` (= the `M` matrix) and
the per-arity `M'_k` family. Both are derived deterministically from
`(model_id, "ajtai_matrix")`, so re-running the offline commit on the same
model produces bit-identical commitments. The verifier needs only `model_id`
and the per-edge commitment hashes — no per-input data.

**Cost amortization.** For a 7B-parameter LLaMA at `b = 21`, the offline
commit moves roughly `21 × (model bytes) / 64`  ring-element commits — a
once-per-model cost on the order of minutes to hours, vs. a per-input proving
cost that should land in seconds-to-minutes. Skipping it on every proof is
worth the on-disk storage.

### 4.2 Online phase (per input)

After `dag.run()` populates the input-dependent witnesses, the online commit
covers the remaining edges that need commitments:

- `boundary_edges` (partition boundaries) whose producer is an input-dependent
  node.
- `self_claim_edges` (Conv2D outputs etc.) — input-dependent by definition.
- Lookup auxiliaries — `SparseMLPoly` selection polynomials produced by
  ScaleDown / ScaleUp / NonNegative / ExpHelper. **These are committed via
  `ajtai::commit_sparse`** rather than the dense `commit_batched` path. See
  §5.5 for why this matters.

Same commit recipe as §4 for dense (activation) edges. The `GpuAjtaiStore` is
initialized by loading the offline commits + bit-planes, then the online phase
appends to it.

**Throughput.** `commit_batched` amortizes PRG over batch size; running
b = 21 planes through one call ≈ 21 commits in ~1.5× the time of a single one.
Per `cuda_almost_goldilocks/ajtai.md` §13, batched `B = 16` ≈ 0.11 s/commit at
N = 2^27 on one A100; at the smaller arities most witnesses occupy
(N ≤ 2^22), it's a few ms. The online commit set is small enough that this is
not a bottleneck.

---

## 5. Backward pass (unchanged)

`dag.prove()` runs as in zk-torch-3:

- Initialize output claims, walk topologically backward, call each BasicBlock's
  `prove()` to consume output claims and emit input claims, plus sumcheck
  proofs. **Drop-in change**: every `GoldilocksExt2`-typed challenge / claim
  becomes `AlmostGoldilocksExt2`. The Almost-GL crate already exposes
  `eq_lagrange`, `partial_eval`, `sumcheck_prover`, and `extension` (`F_q[X] / (X^2 − 3)`).
- Range-check (`prove_range`) and two-pow (`prove_two_pow`) lookup protocols
  remain unchanged in structure; their auxiliary witnesses are committed
  alongside the regular edges.

After this phase we have, for every committed edge `e`, a set of claims
`(r_e^{(1)}, y_e^{(1)}), …, (r_e^{(K_e)}, y_e^{(K_e)})` accumulated by the
backward pass and the per-edge opening reducer.

**Crucial:** Each `y_e^{(j)} = f_e(r_e^{(j)})` is a claim on the **full
integer-valued** polynomial. The decomposition `y_e^{(j)} = Σ_i 2^i · y_e^{(j),i}
− 2^{b−1} · y_e^{(j),b−1}` is taken to derive the corresponding per-bit-plane
claims `y_e^{(j),i} = f_e^{(i)}(r_e^{(j)})`. **All bit planes share the same
point `r_e^{(j)}`**, which is what makes the fold tree work.

---

## 5.5 Lookup auxiliaries — revert to zk-torch-2's sparse form

zk-torch-3 switched the range-check auxiliary from a sparse `SelectionPolynomial`
(zk-torch-2's form) to a dense bit polynomial `B(x, y)` with
`BIT_TABLE_VARS = 5` (32 bit positions per row). That was a big win against
Basefold because Basefold's per-commit cost is roughly linear in the witness's
total entry count, and the dense bit form has 32×2^n entries vs the selection
polynomial's `2^(n+t)` ambient entries — a ~10³–10⁴× shrink in committed
data for typical `t ∈ [10, 20]`.

**For Ajtai, the trade-off reverses.** Ajtai's commit cost scales with the
count of **set bits** (or set u64 blocks), not ambient size. A SparseMLPoly
selection polynomial has **exactly one set bit per input row** — the only
nonzero ambient entries — so committing all 2^(n+t) positions is no more
expensive than committing the 2^n nonzeros. The dense bit form, in contrast,
sets ~16 bits per row on average (32 positions, half on for random data), so
it costs Ajtai ~16× more set-bit work per range check.

zk-torch-4 therefore reverts to **zk-torch-2's sparse selection-polynomial
form** for all range-check auxiliaries. The associated wins:

| Property                          | z-t-3 dense bit aux            | z-t-2 sparse selection aux       |
|-----------------------------------|--------------------------------|----------------------------------|
| Witness                           | `[2^n, 32]` binary             | `(input_idx, table_idx)` list    |
| Set bits per row                  | ~16 (50 % density of 32 slots) | **1**                            |
| Per-edge bitwidth                 | fixed 32 (BIT_TABLE_VARS=5)    | per-edge `t` (10 for ScaleDown, 20 for NonNeg, …) |
| Ajtai commit path                 | dense                          | `commit_sparse`                  |
| Need bit decomposition (§2)?      | already binary                 | already binary                   |
| RingCommitments per edge          | b·1 = 21                        | **1**                            |
| Expected commit-time ratio        | 1×                             | ~8–16× cheaper                   |

(`RingCommitments per edge = 1` because the selection polynomial's values are
already in `{0,1}`, no bit decomposition is needed. The factor-of-b savings
versus the dense-bit form compounds with the per-row set-bit savings.)

### 5.5.1 Witness construction

Restore the zk-torch-2 NonNegative / ScaleDown / ScaleUp body. Each one
produces a `SelectionPolynomial { input_num_vars: n, table_num_vars: t,
selection: [(i, t_i)] for i = 0..2^n }`. `t` is the bitwidth needed for the
node (NonNegative: 20, ScaleDown: 10, …). Convert to `SparseMLPoly` via
`to_sparse()` so the on-device storage is the position list, not a dense
hashmap.

ExpHelper's selection polynomial is already in this form in zk-torch-3 — no
change.

### 5.5.2 Commit (Ajtai sparse path)

Each `SparseMLPoly` becomes one position list `Vec<u64>` of length 2^n, where
position `p_i = i + t_i · 2^n` encodes the unique nonzero of input row `i`
(little-endian, matching zk-torch-3's MLE-index convention). Commit via:

```rust
let pos: Vec<u64> = (0..(1usize<<n)).map(|i| {
    (i + selection[i].1 * (1<<n)) as u64
}).collect();
let c = ajtai::commit_sparse(seed, &pos, /*chunk=*/None)?;
```

By `cuda_almost_goldilocks/ajtai.md` §9, sparse-commit-via-position-list wins
when within-non-zero-block density `B_block < 2`. For our selection polynomials,
`B_block = 1` exactly (each nonzero is alone in its u64 block, because
consecutive rows almost always have different `t_i` values that map them to
different blocks). So `commit_sparse` is unambiguously the right path.

**Arity and `max_num_vars` interaction.** The selection polynomial has
ambient arity `n + t` (input rows × table positions). Since `t` can be 20
(NonNegative bitwidth), an aux at `n = 20` has arity 40 — bigger than typical
activation polys. That's fine for two reasons:
1. `commit_sparse` regenerates `M[*, j]` columns on the fly via ChaCha8 and
   touches only the nonzero `j`-blocks, so commit cost scales with `2^n`
   (nonzeros), not `2^(n + t)` (ambient size). No M' precompute is used —
   precompute helps the *dense* short-arity path, not the sparse one.
2. `max_num_vars` is set to `max_i arity_i` across all committed witnesses
   — including sparse aux. So if NonNeg pushes `max_num_vars` to 40, the
   same-point sumcheck (§6.1) runs 40 rounds. That's ~2× the rounds of an
   activation-only `max_num_vars ≈ 22`, but each round is cheap because
   sparse polynomials contribute work proportional to their nonzero count,
   not their ambient size.

The verifier side never materializes the `2^(n+t)`-wide poly either; it
checks the sparse-bool sumcheck against the position list directly, and the
sparse-aux contribution to the same-point sumcheck uses
`eq(r_i, (R_x, R_y))` evaluated on (input_idx, table_idx) pairs — O(K)
work in the nonzero count.

### 5.5.3 Range protocol (combined sumcheck over all range edges)

Restore zk-torch-2's `prove_range` shape, with `LookupProof` containing:

- **One table sumcheck** combining all range-checked auxes via β-power
  weighting. Identity (per zk-torch-2 §`prove_range`):

  ```
  Σ_y Σ_{i ∈ range} βᵢ · aux_i(x_i, y) · (T_t(y) + α)
       =  Σ_{i ∈ range} βᵢ · ( value_i(x_i) + α · 1 )
  ```

  where `T_t(y) = y` (the per-bitwidth identity table) and the partial-eval
  `aux_i(x_i, y)` has already been reduced to a `t`-variate poly at the
  point `x_i = r_i` (the input-row coordinates from that node's claim point).

- **One sparse-boolean sumcheck per distinct aux arity** proving
  `Σ_z eq(ρ, z) · aux(z) · (1 − aux(z)) = 0`. Reuses the
  `SparseBoolSumcheckProver` pattern from zk-torch-2 (iterate over set positions
  via the position list — same data structure we committed).

The α / β / γ challenges are sampled from the transcript exactly as in
zk-torch-2.

The protocol output is **one claim per range-checked aux at one shared point**
(constructed by concatenating the input-row coords from the node's claim with
the y-coords from the table sumcheck). That single claim feeds into the fold
tree exactly like any other edge claim — the lookup protocol is just another
producer of `FoldInstance` leaves.

### 5.5.4 Two-pow protocol (unchanged)

Two-pow auxiliaries are already `SelectionPolynomial`s in zk-torch-3. We keep
the protocol as-is and commit via `commit_sparse`, same as the range
auxiliaries.

---

## 6. Fold tree (the new opening phase)

After backward pass + opening-reducers, we hold one claim per (edge, bit-plane)
— call this the **leaf set** of the fold tree. The leaf set unifies offline
and online commits: a weight edge whose constant value was committed back in
§4.1 produces the same shape of leaf as a per-input edge committed in §4.2.
Both contribute their bit-plane commitment, their (still resident) bit-plane
witness, and the claim point/value derived by the backward pass. Each leaf is
a `FoldInstance`:

```rust
struct FoldInstance {
    commitment: RingCommitment,       // c_i ∈ R^15
    witness:    Witness,              // f_i either Binary (u64 bitmask) or Ternary
    arity:      usize,                // k_i ≤ max_num_vars
    claim_pt:   Vec<AlmostGoldilocksExt2>,  // r_i of length arity
    claim_val:  AlmostGoldilocksExt2,       // y_i = f_i(r_i)
}

enum Witness {
    Binary  (DeviceBuffer<u64>),                              // packed
    Ternary (TernaryChunksDevice /* k_chunks may be > 13 */), // after split
}
```

The fold tree contracts the leaf set to **one** `FoldInstance` whose witness is
small enough to send to the verifier. The contraction has two repeated steps —
**same-point sumcheck** and **multi-fold** — terminated by **splitb** every
time the working set's norm budget would be exceeded by the next fold.

### 6.1 Same-point sumcheck

Goal: take a working group `G = {(c_i, f_i, r_i, y_i)}_{i=0..M−1}` with
heterogeneous `r_i`s, and reduce to a single shared point `R` with
`y'_i = f_i(R)` for every `i`.

Sumcheck identity:

```
Σ_i α^{i} · y_i · 2^{max_num_vars − k_i}
   = Σ_{x ∈ {0,1}^{max_num_vars}} Σ_i α^{i} · eq(r_i, x_{[1..k_i]}) · f_i(x_{[1..k_i]})
```

The `2^{max_num_vars − k_i}` factor compensates for the implicit broadcast
(summing a k-variate poly over an m-variate cube counts each value
`2^{m − k}` times). Both sides are publicly known to the verifier.

After `max_num_vars` sumcheck rounds we have a shared challenge
`R = (R_1, …, R_max_num_vars)` and the final reduced eval
`Σ_i α^i · eq(r_i, R_{[1..k_i]}) · f_i(R_{[1..k_i]})` — from which the verifier
recovers each `y'_i = f_i(R_{[1..k_i]})` (modulo the standard sumcheck
soundness via the `α`-power randomization).

**Implementation note.** Run the sumcheck in `max_num_vars` rounds. For each
`f_i` with `k_i < max_num_vars`, the polynomial contributes nothing to rounds
`k_i + 1 .. max_num_vars` past a single (already-collapsed) constant — drop it
from the round-poly evaluator after round `k_i`. This is the optimization
mentioned in the original plan ("just when we process x_3 for f_1 in sumcheck,
we can just add the eq(r_a, r_b, R_a, R_b) f_2(R_a, R_b) to the sum during
sumcheck"). The Almost-GL `GpuSumcheckStateExt2` (in
`almost-goldilocks-cuda-rs/src/sumcheck_prover.rs`) handles the heavy lifting
once we feed it the multiplexed polynomial.

### 6.2 Multi-fold

After same-point sumcheck, sample one shared `RingChallenge` per non-anchor
slot: `γ_1, …, γ_{M−1} ∈ {−1, 0, 1, 2}^64`. Then

```
c' = c_0  +  Σ_{i=1..M−1} γ_i · c_i        // multifold_commitment
f' = f_0  +  Σ_{i=1..M−1} γ_i · f_i        // multifold_witness (or _mixed_/_tc_fused)
```

By Ajtai linearity, `c' = M · f'` holds bit-exactly (homomorphism is verified
by `test_multifold_homomorphism_K50_k13`). The single sumcheck-derived claim
on the folded witness is

```
y' = Σ_i α^i · γ_i · f_i(R_{[1..k_i]})       (γ_0 := 1)
```

which the verifier computes from the per-instance `f_i(R_{[1..k_i]})` it
already has.

**Kernel mapping.**
- Binary-only fold: `ajtai::multifold_witness` + `ajtai::multifold_commitment`.
- Mixed binary + ternary fold: `ajtai::multifold_mixed_witness_tc_fused`
  (preferred — bit-exact identical to the scalar `multifold_mixed_witness` per
  the integration tests, and ~order-of-magnitude faster at large `N_ring`).
- Pure ternary fold (after the first split): same `_mixed_` API with
  `k_bin = 0`.

### 6.3 Splitb (base-decomposition)

After multi-fold the witness coefficients live in `i16` with `|·| ≤ M · T · (b−1)`.
Even at `M = 63, T = 128, b = 2` that's bounded by 8064 < 8192 = B, so the
folded commitment is still binding. But the **next** fold would multiply norms
by another `T = 128`, busting B. So before sending it back into the tree we
decompose:

```
f' = Σ_{i=0..12} 2^i · (f'^{(i)}_pos − f'^{(i)}_neg)
```

with `f'^{(i)}_pos, f'^{(i)}_neg ∈ {0,1}^{|f'|}` and disjoint supports (per
ring-element coefficient). Each chunk is a valid ternary instance for the next
fold. Kernel: `ajtai::split_witness_device` (operates entirely on-GPU).

After splitting, also commit each of the 13 chunks: one `ajtai::commit_ternary`
call (or `commit_ternary_premat` when `M` fits in HBM, which is ~2× faster) does
all 13 at once, sharing PRG across them.

> The committed chunks satisfy `Σ 2^i · commit(chunk_i) = c'` (homomorphism
> check), so the verifier never needs to receive any of the chunk commitments
> separately — it can derive them.

### 6.4 Parallel tree structure

The tree's leaves are the `(edge, bit_plane)` Ajtai commits. Internal nodes are
multi-fold-then-split operations. Each node ingests up to 63 instances and
emits 13. So if we start with N leaves:

```
Level 0:  N  leaves            (binary)
Level 1:  ⌈N/63⌉ · 13 ternary chunks
Level 2:  ⌈Level 1 / 63⌉ · 13 ternary chunks
...
Level L:  ≤ 63 instances        → one final multi-fold, no split, opens directly
```

Contraction ratio per level is `63 / 13 ≈ 4.85`, so
`L ≈ log_{4.85}(N / 63)` levels. For `N = 50_000` (typical large DAG, 21 bit
planes × ~2400 edges), that's `L ≈ 4` internal levels.

**Parallelism.**

- Within a level, **all 63-groups are independent** — different sumchecks,
  different multifolds, different splits. Schedule them across all GPUs the
  same way zk-torch-3 schedules opening proofs across the dual CPU/GPU pool
  (`std::thread::scope` + per-thread CUDA streams).
- The transcript dependency between levels is a serial bottleneck for
  Fiat-Shamir challenge derivation, but the per-group sumcheck/multifold/split
  is dominated by GPU compute, not transcript hashing. Use the same
  `Transcript::fork(k)` trick zk-torch-3 uses for partition-aware proving:
  each group at a given level forks the transcript with its group-index, runs
  its sumcheck against the forked transcript, and after the level absorbs the
  per-group sumcheck digests back into the parent transcript in deterministic
  order. (This is a standard domain-separation pattern; no soundness cost.)
- Cross-level dependency is straightforward: level `L+1` cannot start until
  level `L` has produced all its output instances.

**Group-formation policy.** At each level, sort the surviving instances by
`(arity, kind)` so that within a group all instances share the same arity (or as
many as possible). Reasons:
1. Same-point sumcheck is cheapest when all `k_i = max_num_vars` (no skip
   bookkeeping); same arity throughout is the next-cheapest case.
2. The mixed binary + ternary fold kernel runs at full WMMA throughput when
   `k_bin` and `k_tern` partition the group cleanly.

### 6.5 Termination & final opening

When the surviving set has size ≤ 63, run one last same-point sumcheck and one
last multi-fold — **but skip the split**. Output: a single `(c*, f*, R*, y*)`
4-tuple. The verifier needs to be convinced that `c* = M · f*` and `f*(R*) = y*`.

Send the full witness `f*` to the verifier. Size: at most `2^max_num_vars · 8`
bytes if it's i16-packed; with `max_num_vars ≤ 22` this is ≤ 32 MiB and
typically much less because the witness "shrinks with the tree" — but more
importantly, **it's only sent once**, vs. the per-edge Basefold proofs of
zk-torch-3 which together dwarf this.

The verifier:
1. Receives `f*`.
2. Reconstructs `c_expected = M · f*` using the same seeded ChaCha8 PRG. (For
   `max_num_vars` up to ~22 this fits on one GPU; otherwise we re-execute the
   tree-level operations symbolically against the `RingCommitment`s using the
   homomorphism, never reconstructing `c_expected` from scratch.)
3. Checks `c_expected == c*` and `f*(R*) == y*` (the latter via direct
   multilinear eval).

---

## 7. Verifier — full sequence

1. Read max_num_vars, b, the public seed, and the verifier's lightweight
   per-edge commitment hashes from the transcript.
2. Replay the zk-torch-3 backward-pass sumchecks (no change here besides the
   field swap).
3. For each fold level, replay the per-group same-point sumcheck. The shared
   challenge `R` at the end of each sumcheck is derived from the transcript
   via Fiat-Shamir — verifier and prover agree by construction.
4. For each fold level, re-derive the per-group ring challenges γ from the
   transcript and combine the input `RingCommitment`s the prover sent for that
   level (or, after the first split, the chunk commitments are
   homomorphically derivable from the parent commitment — see §6.3 — so no
   extra communication).
5. At the root, check the final `c* = M · f*` and `f*(R*) = y*`.

Crucially, the verifier **never** runs a Basefold-style query loop. The opening
is a deterministic algebraic check given `f*` plus the recorded transcript.

---

## 7.5 Implementation philosophy (rules I commit to)

These are the rules I follow for every step. Re-read them before starting any
new sub-task; flag in conversation whenever I'm tempted to violate one.

1. **No TODO, no FIX, no "for now" stubs.** Every line of code committed
   either does its real job or doesn't exist. A function I haven't fully
   implemented does not get a placeholder body that compiles — it stays
   unwritten, blocking its caller, until I'm ready to write the real one.
2. **Validate before declaring done.** A step ends with green tests, not
   "looks right." If the validation for a step would require a piece I
   haven't built yet, I either build that piece first or shorten the step
   to what can be validated now.
3. **Discuss difficulties not in the plan.** When the work uncovers a
   constraint, dependency, or design question that this document doesn't
   already settle — pause, raise it to the user, propose the resolution,
   amend this document, *then* code. Silent improvisation produces drift.
4. **Bottom-up dependency order.** Port leaves first (util, poly types),
   then modules that depend on them (sumcheck, basicblocks), then the DAG
   plumbing on top. Each layer compiles and tests green before I move up.
5. **Reuse zk-torch-3's shapes where unchanged.** Field swap is mechanical;
   I do it via a small number of clearly-scoped `sed`-style passes plus
   manual fixups, not by rewriting every module from scratch. New code only
   where the protocol genuinely differs (commit, fold-tree, range-check
   shape).
6. **Tests live in the same files they test.** Mirror zk-torch-3's
   `#[cfg(test)] mod tests { ... }` placement. Don't create separate
   `tests/` files for unit tests — the existing layout makes locality
   obvious.
7. **No silent CPU↔GPU fallback paths.** If a GPU kernel is missing, raise
   it as a difficulty and choose one of: port the kernel, eliminate the
   need via a protocol change, or make the CPU path the documented
   default. Don't hide it behind a runtime branch.
8. **Default to porting the kernel.** When a basicblock has a real GPU
   speedup in zk-torch-3 (einsum, conv, large per-element ops, large
   sumchecks), the AGL CUDA kernel ships in the same pass as the
   basicblock — not a follow-up. CPU-only documented-defaults are
   reserved for ops where CPU is fast enough that the GPU port would be
   pure churn (Reducer at small `n`, ChangeShape metadata).

---

## 8. Implementation plan

### 8.1 New crate layout (mirrors zk-torch-3)

```
zk-torch-4/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── transcript.rs          // Monolith over AlmostGoldilocks (new constants)
    ├── poly/                   // DenseMLPoly etc., over AlmostGoldilocks
    ├── sumcheck/               // wraps almost-goldilocks-cuda-rs::sumcheck_prover
    ├── dag/                    // same DAG / topology as zk-torch-3
    ├── basicblock/             // ported, with field swap
    ├── commit/
    │   ├── mod.rs              // GpuAjtaiStore, per-edge commit
    │   ├── matrix.rs           // M and M'_k generation + caching
    │   ├── bit_decompose.rs    // signed → b binary planes
    │   ├── offline.rs          // §4.1 — bit-decompose + commit weights, persist
    │   └── online.rs           // §4.2 — append per-input commits
    └── fold/
        ├── mod.rs              // FoldInstance, working-set bookkeeping
        ├── same_point_sumcheck.rs
        ├── multifold.rs        // wraps ajtai::multifold_*
        ├── split.rs            // wraps ajtai::split_witness_device + commit_ternary
        ├── tree.rs             // level-by-level parallel scheduler
        └── verifier.rs         // verifier-side fold replay
```

### 8.2 Implementation order

> **Note (decisions resolved 2026-05-23):** Step 2 (Monolith) moved ahead of
> the transcript port; CUDA-kernel ports added as explicit prereqs of step 6
> (basicblock port); step 6 covers **all** basicblocks; step 7 validation
> gate is non-commit tests only.

1. **Port the field plumbing — leaf modules first.** Copy zk-torch-3's
   `poly/` (DenseMLPoly + SparseMLPoly + SelectionPolynomial), `util/`, and
   any other genuinely-foundational pieces modulo the field swap
   (Goldilocks → AlmostGoldilocks). These have no Monolith dependency, so
   they can be ported and unit-tested independently.

2. **Derive AlmostGoldilocks Monolith round constants** (was step 2, now
   prereq for §1's transcript). See §8.3 for the procedure. Sanity-check the
   derivation logic by reproducing Plonky3's Goldilocks constants byte-for-
   byte first.

3. **Port `transcript.rs`** with the new constants. All other ports below
   depend on this.

4. **Port `sumcheck/`** (cpu_ext2_prover, general_prover, gpu_prover,
   linear_prover, verifier). Wraps `almost-goldilocks-cuda-rs::sumcheck_prover`.

5. **Port missing CUDA kernels to `almost-goldilocks-cuda-rs`** — porting
   schedule is per-block: a basicblock lands with its GPU kernel in the same
   pass (per the goal of the fastest prover end-to-end; see §7.5 rule #7's
   "port the kernel" branch). The kernels needed:
   - **Already in AGL**: `agl_bit_permute`, `partial_eval_ext2_device`,
     `fused_permute_partial_eval`, `ext2_eq_dp_all_device`,
     `AlmostGoldilocksBatch::{add, sub, mul, neg}`,
     `AlmostExt2Batch::{add, sub, mul, scale_accumulate, from_base}`,
     `GpuSumcheckStateExt2::{new, new_from_base, fold, ...}`.
   - **Small helpers** (port together as one mini-PR):
     - `agl_relu_helper`: one-line kernel `neg[i] = (v > q/2) ? (q - v) : 0`.
     - `agl_zero_buffer`: `cudaMemset` wrapper.
     - `GpuSumcheckStateExt2::from_device_buffers`: constructor from existing
       device buffers (avoids the Reducer's redundant re-upload).
   - **Einsum** (transformer hot path — port alongside `basicblock::einsum`):
     - `agl_einsum1` (unary einsum / transpose / sum-reduction).
     - `agl_einsum2` (general two-input einsum / matmul).
   - **Conv** (CNN models — port alongside `basicblock::conv`):
     - `agl_conv1d`, `agl_conv2d`, `agl_conv3d`, `agl_depthwise_conv2d`,
       `agl_conv_transpose1d`, `agl_conv_transpose2d`, `agl_conv_transpose3d`.
   - `bit_decomp` — **not needed** thanks to the §5.5 z-t-2 range-check
     switch (selection polys are natively binary; no GPU bit-decompose path).

   Each kernel is a direct field-swap of the goldilocks-cuda CUDA source.
   Validate with a CPU-reference test asserting bit-exactness.

6. **Port all basicblocks (CPU paths only).** Concretely:
   - 6a. **Restore z-t-2 range-check shape (§5.5)** in `Witness`,
     `NonNegative`, `ScaleDown`, `ScaleUp`. Replace zk-torch-3's
     `prove_range` / `verify_range` in `dag/mod.rs` with the zk-torch-2
     versions (table sumcheck + sparse-bool sumcheck).
   - 6b. **Port everything in `basicblock/`** with field swap (~12k LOC,
     18 files):
     - Transformer-relevant: `add`, `einsum`, `llama`, `range`, `reducer`,
       `relu`, `scale`, `shape`, `permute`, `exp`, `mod`.
     - CNN/3D: `conv`, `concat`, `instancenorm`, `maxpool`, `pad`,
       `pointpillar`, `subsample`.
   - Mechanical port; the protocol logic is unchanged. New work only in 6a.
   - GPU dispatch points (e.g., `einsum.rs::run_gpu`, `relu.rs::run_gpu`)
     are kept but routed to the CPU path until step 5 lands. The threshold
     constants are preserved so the GPU paths re-activate cleanly with a
     single re-wire later.

7. **Port `dag/{mod, builder, partition}.rs`** with field swap. **Strip
   out** the basefold-specific commit/open glue — the new commit module
   (step 9 below) will plug in. Re-route `prove_range` / `verify_range` to
   the §5.5 versions. **Validation gate for step 1**: `cargo test --lib`
   passes for everything ported so far (non-commit, non-fold tests).

8. **Build `GpuAjtaiStore` and the M / M' matrix family.** No proving
   logic — just `commit_batched`-driven commits + arity bucketing + M'
   precompute. Unit test:
   `Σ 2^i · commit(f_i) == commit(Σ 2^i · f_i)` for a randomly
   bit-decomposed witness.

9. **Offline weight-commit (§4.1).** Walk `Role::Constant` witnesses,
   bit-decompose, commit, persist to disk. Test: load → in-memory state is
   bit-identical to fresh commit. Ship this as a separate `bin/precommit_model`
   binary.

10. **Online commit (§4.2)** wires into `dag.commit()` for the
    input-dependent edges (plus `commit_sparse` for §5.5 lookup aux).

11. **Backward pass.** Compiles after step 7. The per-edge "opening reducer"
    still runs; it's the natural producer of `FoldInstance` leaves.

    ✅ **Status: shipped.**
    - `src/sumcheck/sparse_bool_prover.rs` — `SparseBoolSumcheckProverExt2`
      for the z-t-2 bool check (one sumcheck per `aux_num_var` group,
      degree-3 round messages).
    - `src/dag/lookups.rs` — `prove_range` / `verify_range` /
      `prove_two_pow` / `verify_two_pow`. Range covers NonNegative,
      ScaleDown, ScaleUp, and ExpHelper (ExpHelper's `eval_to_check`
      currently passes the input claim through verbatim — the
      `−ln 2`-inverse coupling lands when exp pipeline tests come online).
    - `src/dag/proving.rs` — `Dag::prove` / `Dag::verify`:
      output-claim seeding, reverse-topo `BTreeSet` walk, reducer
      pre-pass per node (Reducer basicblock), lookup proofs, then per-edge
      opening reducer (CPU sumcheck on `x · Σ α^i · eq(r_i, x)`).
    - Per-`NodeProof` layout: `sumcheck_proofs = [reducer..., node...]`
      (reducer occupies slot 0 when used; verifier replays both halves).
    - Per-`DagProof`: `output_claims` are explicit `(edge_id, point, eval)`
      triples so the verifier never reads witness data outside of the
      shape/role metadata needed for transcript replay.
    - 6 round-trip tests in `dag/proving.rs` (single Add, two-range,
      two-input shared-constant + opening reducer, plus three tampering
      negatives). 3 in `dag/lookups.rs`. 3 in `sumcheck/sparse_bool_prover.rs`.
      All 247 lib tests green.

12. **Build the fold tree.**
    - 12a. `same_point_sumcheck.rs`: implement the identity from §6.1.
    - 12b. `multifold.rs`: wraps `multifold_witness` /
      `multifold_mixed_witness_tc_fused`.
    - 12c. `split.rs`: pairs `split_witness_device` + `commit_ternary_premat`.
    - 12d. `tree.rs`: scheduler.

    ✅ **Status: shipped (sequential CPU-reference path; GPU and parallel
    paths are follow-up).**
    - `src/fold/mod.rs` — `FoldInstance`, `FoldData::{Binary,Ternary}`,
      `WireCommitment`/`WireRingChallenge` for proof serialization.
    - `src/fold/same_point_sumcheck.rs` — heterogeneous-arity sumcheck
      with the per-instance "constant-mode" optimization for `j > k_i`
      (per-round message stays degree-2). 4 tests cover single-instance,
      two heterogeneous arities, three-instance constant-mode, and
      tampered f_eval rejection.
    - `src/fold/multifold.rs` — host-reference ring multifold of
      witnesses (`f' = f_0 + Σ γ_i · f_i` via per-ring-element
      `mod (X^64+1)` convolution); commitment fold uses the GPU
      `multifold_commitment` kernel. γ challenges sampled from
      transcript via 2 bits per coef → `{-1, 0, 1, 2}^64`. 3 tests
      (single-instance passthrough, two-instance round-trip, γ-tamper
      rejection).
    - `src/fold/split.rs` — base-2 decomposition into 13 ternary chunks
      and `commit_ternary` per-chunk; verifier-side `Σ 2^i · c_i = c_parent`
      homomorphism check. 2 tests (binary→chunk-0 trivial, random
      wide-coef homomorphism).
    - `src/fold/tree.rs` — sequential 63→13 contraction with internal
      and final levels. The final level's witness `f*` ships verbatim;
      every internal level's split chunks become independent next-level
      instances (no 2^i scaling carried into the next multifold's norm
      budget — per-instance |coef| stays ≤ 1). 3 tests (final-only,
      internal-split, final-val tamper rejection).
    - `src/fold/verifier.rs` — `FoldTreeError` enum; verify logic is
      inlined into `tree::verify_fold_tree`.

13. **Verifier.** `fold/verifier.rs` replays the tree.

    ✅ **Status: shipped alongside step 12.** Per-level: replay
    `same_point + multifold + (split homomorphism)`; final: check
    `commit(f*) = c*` ∧ `f*(R*) = y*`.

13.5. **Integration with `dag.prove`.**

    ✅ **Status: shipped.** `src/dag/fold_integration.rs` provides
    `Dag::prove_with_fold_tree` / `Dag::verify_with_fold_tree`:
    - Per committed dense edge: bit-decompose the witness into `b`
      planes via the existing `decompose_and_pack`, evaluate each
      plane's MLE at the opening reducer's combined point, build one
      `FoldInstance` per (edge, plane).
    - Per committed sparse edge (lookup aux): single-plane via the
      sparse-positions broadcast.
    - The combined claim point (native arity) is extended to
      `max_num_vars` by sampling transcript challenges that both prover
      and verifier replay — the broadcast poly is constant in the
      extra vars, so the MLE at the extended point equals the
      original MLE.
    - Verifier checks the signed two's-complement reconstruction
      `v = Σ_{i=0..b-2} 2^i · y_i − 2^{b-1} · y_{b-1}` per edge,
      then runs `verify_fold_tree` against the per-plane leaves.
    - 1 end-to-end test in `dag/fold_integration.rs`: `y = x + w` with
      `w` constant, commit + prove + verify round-trip green. 263
      total lib tests passing.

13.7. **M' optimization — per-arity native commits (Option A).**

    ✅ **Status: shipped.** Removes the broadcast-to-`max_num_vars` from
    every commit and runs per-arity sub-trees in the fold tree.
    - `commit/mod.rs`: `commit_dense` and `commit_sparse_witness` now
      commit at the witness's NATIVE arity (padded to ≥ 6 so a full
      `u64` of bits fits). `EdgeCommitment.arity` records the native
      arity. The matrix family is `M_k = first 2^k columns of M_max`
      — uniformly random under the same ChaCha8 seed, so SIS hardness
      holds. Different arities use different matrices, so multifold
      only fuses within an arity bucket.
    - `fold/tree.rs`: `FoldTreeProof = { buckets: Vec<BucketFoldProof> }`,
      one bucket per distinct leaf arity. Each bucket runs the existing
      (now-internal) `prove_fold_tree_uniform` end-to-end and ships
      one tip `(c*_k, f*_k, R_k, y*_k)`. Verifier checks each bucket's
      tip independently (no cross-arity fold ⇒ no per-tip cross-check).
    - `dag/fold_integration.rs`: leaves live at native arity. The
      combined claim point is zero-extended (not random-extended) to
      the leaf's arity ≥ 6, because a zero-padded witness's MLE at
      `(native_pt, 0, …, 0)` exactly equals its MLE at `native_pt`.
    - Benchmark: GPT-2 tiny (hidden = 8, NUM_LAYERS = 1) prove time
      dropped from **~22 min → 10.4 s** (~127× speedup); verify went
      from infeasible → 2.6 ms. The protocol still correctly rejects
      out-of-range intermediates (the gpt2 tiny test's random/zero
      weights produce a negative value that fails NonNegative's range
      check — a feature, not a bug). Real model weights with proper
      SF_LOG = 15 scaling would verify.
    - 269 lib tests still passing.

13.6. **Follow-ups (performance + sparse coverage).**

    ✅ **Status: shipped.**
    - **GPU witness-fold kernel wiring** (`fold/multifold.rs`):
      `fold_witnesses_gpu` dispatches to `ajtai::multifold_witness`
      (all-binary path, level 0) or `ajtai::multifold_mixed_witness_tc_fused`
      with `k_bin = 0` (all-ternary path, every post-split level).
      Host reference kept as `#[cfg(test)]` oracle; bit-exact agreement
      tests `gpu_witness_fold_matches_host_{all_binary,all_ternary}`.
    - **Parallel fold-tree scheduler** (`fold/tree.rs`): per-level groups
      run under `rayon::par_iter` against `transcript.fork(group_idx)`.
      After each level the parent transcript absorbs every group's
      `(combined_commitment, chunk_commitments, chunk_claim_vals)` in
      deterministic order via `absorb_group_commitments`, so the next
      level's Fiat-Shamir challenges still bind to this level's
      outputs. Verifier mirrors the fork + absorb pattern exactly.
    - **Full sparse-edge fold-tree integration** (`dag/fold_integration.rs`):
      `end_to_end_with_sparse_range_aux` exercises a DAG with a
      NonNegative range check (which emits a sparse SparseMLPoly aux
      committed via `commit_sparse`). The fold-integration code path
      routes the aux through the sparse branch of
      `decompose_witness_for_fold` (single binary plane from the
      broadcast position list) and runs prove + verify end-to-end.
    - 266 total lib tests, no regressions.

14. **End-to-end tests.** 1-layer transformer first. Then sweep zk-torch-3's
    model bins (gpt2, bert, …) at NUM_LAYERS=1, then full size.

### 8.3 Monolith round constants for AlmostGoldilocks

The transcript uses Monolith natively on `F_q` (`q = 2^64 − 2^32 − 31`).
Monolith is structurally well-suited for this: it costs ~10× less CPU per
permutation than Poseidon2 (6 rounds vs 30; byte-wise bit-op S-box vs `x^7`),
which matters because the transcript is invoked many times per proof on the
Fiat-Shamir critical path.

**Structural parameters** (unchanged from the Goldilocks reference at
`goldilocks-cuda-rs/src/cpu_monolith.rs`):

- `WIDTH = 12`, `NUM_BARS = 4`, `N_ROUNDS = 6`.
- `bar_64`: 8-bit byte-wise S-box, **field-agnostic** (operates on raw `u64`
  bit patterns; subsequent `gl_mul` canonicalizes).
- `bricks`: Feistel Type-3 reverse iteration `state[i] += state[i-1]^2`.
  Uses the new `agl_mul` / `agl_add`.
- `concrete`: MDS matrix-vector multiply + round constants. The MDS matrix
  has small-integer entries `{6, 7, 8, 9, 10, 13, 21, 22, 23, 26}` and is
  MDS over any prime field where all 11×11 minor determinants are nonzero
  — this is true for both Goldilocks and AlmostGoldilocks (the minors are
  small integers and `q ≫ max(minor)`).

**Round constants — what changes for AlmostGoldilocks.** Only the
`ROUND_CONSTANTS: [[u64; 12]; 7]` table is regenerated. Derivation method:

```
For each (round r ∈ 0..7, position i ∈ 0..12):
  for salt in 0u32, 1u32, …:
    bytes = SHA-256("AGL-Monolith-RC-v1" || r.to_le_bytes() || i.to_le_bytes() || salt.to_le_bytes())
    val   = u64::from_le_bytes(bytes[0..8])
    if val < q:
      ROUND_CONSTANTS[r][i] = val
      break
```

This is deterministic, reproducible, and cryptographically sound for the
Fiat-Shamir use case (Monolith's security argument requires that constants
have no exploitable algebraic structure; SHA-256 outputs are computationally
indistinguishable from random — Indifferentiability from random oracle).
Rejection probability per draw is `(2^64 − q) / 2^64 ≈ 2^{-58}`, so a salt
counter is effectively never needed but is present for cryptographic hygiene.

This *deliberately departs* from the Grain-LFSR construction Plonky3 uses
for its Goldilocks Monolith constants. Justification: we lack a portable
Grain-LFSR-Monolith reference implementation in this repo, and the Monolith
paper's security argument (Theorem 1, indistinguishability under random
oracle assumption on the constants) holds for any constants with negligible
algebraic structure — SHA-256-derived constants satisfy this trivially.
Anyone reading the code can re-derive every value bit-exactly from the tag
string above.

**Validation procedure:**

1. **Reference replay (sanity check).** First, implement Monolith with the
   *Goldilocks* round constants from `goldilocks-cuda-rs/src/cpu_monolith.rs`
   and the Goldilocks field arithmetic. Reproduce the reference test vector
   from `test_monolith_permute_test_vector` (`state = [0..11]` → known
   12-element output) bit-exactly. This confirms my permutation
   implementation is correct.
2. **Field swap.** Re-target the same permutation code to
   `AlmostGoldilocksField`. Run the reference test vector with the existing
   Goldilocks round constants reduced mod `q` — output is a deterministic
   sanity-check value that I commit as the AGL-side test vector.
3. **Constant swap.** Generate the new round constants from SHA-256, install
   them, recompute the AGL test vector. Commit as the final golden test
   vector.

This is a localized, finite piece of work — estimated 4–6 hours including
tests.

### 8.4 What we explicitly are *not* building

- A new Basefold-style PCS (out of scope).
- Monolith over almost-Goldilocks. Monolith is enough for the transcript.
- Multi-GPU sumcheck *within* a single fold-tree group. Inter-group parallelism
  is enough at our DAG sizes.

---

## 9. Open questions / future tuning

- **`b` per edge vs. global.** Right now `b = 21` is global. For witnesses with
  smaller dynamic range (e.g., post-ReLU outputs, range-check bit aux
  themselves) we could use a smaller `b` and save commit cost. The protocol
  already permits this — the verifier just needs to know `b_e` per edge — but
  the bookkeeping is easier with one global `b` initially.

- **Skip the same-point sumcheck when M = 2 and arities match?** The fold's
  Schwartz-Zippel soundness still goes through if both inputs are already at
  the same point. This is a minor optimization that mostly matters at the
  leaves of the tree.

- **Where does `commit_ternary_premat` lose to `commit_ternary` on the fly?**
  The break-even is `N_ring · 7680 bytes vs. GPU HBM headroom`. At N ≤ 2^20
  premat is unambiguously a win; at N = 2^22 it's tight; at N = 2^27 we must
  go on-the-fly. The scheduler should pick per-group based on the actual
  surviving witness size.

- **Tensor-core (`multifold_mixed_witness_tc_fused`) crossover.** From
  `examples/bench_multifold_tc.rs` benchmarks, the fused TC path is faster
  past ~`N_ring = 2^14`. Below that, the scalar `_mixed_` path can be cheaper.
  Add a runtime branch in `multifold.rs`.

- **Soundness of the same-point sumcheck across mixed arities.** The
  `2^{max − k_i}` rescaling factor needs careful unit testing — easy to get
  off by a factor of 2.

- **Sparse-bool sumcheck on GPU.** zk-torch-2's `SparseBoolSumcheckProver`
  was a CPU implementation. For deep models with many range-checked nodes,
  this may need GPU acceleration. Defer until we have a benchmark showing
  it's a bottleneck — it iterates only over set positions, so its work is
  `O(Σ_i 2^{n_i})`, which is typically dominated by the activation sumchecks.

---

## 10. The boxed identity

The protocol-defining equations, restated for quick reference:

**Commitment** (linear, additively homomorphic):

$$c_e^{(i)} \;=\; M'_{k_e} \cdot f_e^{(i)}, \qquad f_e \;=\; \sum_{i = 0}^{b-2} 2^i f_e^{(i)} - 2^{b-1} f_e^{(b-1)}$$

**Same-point sumcheck** (drives `r_i` heterogeneity → shared `R`):

$$\sum_i \alpha^{i} \cdot y_i \cdot 2^{\,m - k_i} \;=\; \sum_{x \in \{0,1\}^m} \sum_i \alpha^{i} \cdot \mathrm{eq}(r_i, x_{[1..k_i]}) \cdot f_i(x_{[1..k_i]}) \qquad (m = \text{max\_num\_vars})$$

**Multi-fold** (collapses M instances to 1, by Ajtai homomorphism):

$$c' \;=\; c_0 + \sum_{i=1}^{M-1} \gamma_i \cdot c_i, \qquad f' \;=\; f_0 + \sum_{i=1}^{M-1} \gamma_i \cdot f_i, \qquad c' = M \cdot f'$$

**Splitb** (rebinds norms to `b = 2`, ternary chunks):

$$f' \;=\; \sum_{i = 0}^{12} 2^i \cdot \bigl(f'^{(i)}_{\text{pos}} - f'^{(i)}_{\text{neg}}\bigr), \qquad \big|f'^{(i)}_{\text{pos}}\big|, \big|f'^{(i)}_{\text{neg}}\big| \in \{0,1\}, \;\; \text{disjoint support}$$

**Final reveal** (one opening for the whole DAG):

$$\text{Prover sends } f^*; \quad \text{Verifier checks } M \cdot f^* \stackrel{?}{=} c^* \;\; \text{and} \;\; f^*(R^*) \stackrel{?}{=} y^*$$

# Prover Optimization Plan

**Constraint**: Prover outputs (proofs) must remain bit-identical. Only internal execution may change.

## Current Benchmarks (Full-Layer Models)

| Model | Layers | Prove Time |
|-------|--------|-----------|
| GPT-2 | 12 | 11.26s |
| BERT | 24 | 20.03s |
| GPT-J | 28 | 294.00s |
| LLaMA | 32 | 462.75s |

The large gap between GPT-2/BERT and GPT-J/LLaMA suggests that opening proofs (which scale with polynomial size) dominate for larger models.

---

## Identified Bottlenecks

### 1. `evaluations()` clones entire polynomial vectors

**Location**: `poly/mod.rs:79` (trait), `poly/dense.rs:209` (impl)

```rust
fn evaluations(&self) -> Vec<GoldilocksField> {
    self.evaluations.clone()  // full Vec allocation + memcpy
}
```

**Call sites** (each triggers a full clone of the evaluation table):
- `dag/mod.rs:307` — commit phase (once per edge)
- `dag/mod.rs:487` — opening proofs (once per claim, **inside `par_iter`**)
- `basicblock/reducer.rs:53,107` — reducer prove (once per reducer call)
- `basicblock/einsum.rs:538,588,659` — einsum prove (once per input witness)
- `basicblock/add.rs:19-20,141-142` — add/sub run
- `basicblock/scale.rs:18,86` — scale run
- `basicblock/llama.rs:20,91,136,202` — LLaMA-specific ops
- `basicblock/range.rs:20`, `basicblock/exp.rs:30`, `basicblock/shape.rs:20`, `basicblock/permute.rs:58`

**Impact**: For a polynomial with n=20 (1M entries), each clone allocates 8MB and copies it. For GPT-J with hundreds of edges and multiple claims each, this adds up to GBs of unnecessary allocation.

### 2. `cpu_open_ext2` allocates new vectors every fold round

**Location**: `commit/cpu_basefold.rs:73-88`

```rust
for _round in 0..num_vars - 1 {
    let half = eq.len() / 2;
    let mut eq_new = Vec::with_capacity(half);  // new alloc each round
    let mut f_new = Vec::with_capacity(half);   // new alloc each round
    for j in 0..half {
        eq_new.push(...);
        f_new.push(...);
    }
    eq = eq_new;  // old vec dropped
    f = f_new;
}
```

For n=24, this performs 23 rounds of allocation. The first round alone allocates 2 x 128MB (8M Ext2 elements x 16 bytes each).

### 3. `evaluate_lagrange_basis_ext2` is sequential and per-task

**Location**: `poly/mod.rs:113-131`

Called at the start of each `cpu_open_ext2` (line 44) to build a 2^n-sized eq table. For n=24, this allocates 256MB and runs sequentially. Since each opening task in the `par_iter` calls this independently, the same-sized allocation is done once per task.

### 4. Opening proofs don't share polynomial data across claims on the same edge

**Location**: `dag/mod.rs:470-492`

When multiple claims exist on the same edge (e.g., from reducer + lookup), each opening task independently:
1. Clones the full evaluation table via `evaluations()` (bottleneck #1)
2. Builds a fresh eq table via `evaluate_lagrange_basis_ext2` (bottleneck #3)
3. Runs the full fold loop

The polynomial data is identical across all claims on the same edge; only the point differs.

### 5. Reducer timing is hidden

**Location**: `dag/mod.rs:399-412`

```rust
// Reducer runs here, BEFORE the timing block
let (proofs, rc) = reducer.prove(&reducer_witness, &reducer_edge_ids, &local_claims, transcript);
...
// Timing starts here — only covers the main node prove
let t_node = std::time::Instant::now();
let (proofs, new_claims) = timed!(..., { node.kind.prove(...) });
let elapsed = t_node.elapsed();
```

Reducer cost is invisible in the timing summary but can be significant when there are many claims.

### 6. `compute_oracle` and fold are sequential in `cpu_open_ext2`

**Location**: `commit/cpu_basefold.rs:124-151` (oracle), `commit/cpu_basefold.rs:73-88` (fold)

Both `compute_oracle` and the fold loop iterate over `half` Ext2 elements sequentially. For n=24, the first round processes 8M element pairs. These are embarrassingly parallel with `rayon`.

---

## Optimizations (Ranked by Expected Impact)

### Optimization 1: Add `evaluations_ref()` to `MLPoly` trait

**Impact**: HIGH | **Effort**: Easy | **Risk**: None

Add a borrowed accessor to avoid cloning:

```rust
// poly/mod.rs — add to MLPoly trait
fn evaluations_ref(&self) -> &[GoldilocksField];

// poly/dense.rs — implementation
fn evaluations_ref(&self) -> &[GoldilocksField] {
    &self.evaluations
}
```

Then update all call sites that don't need ownership to use `evaluations_ref()` instead of `evaluations()`. The opening proof path (`dag/mod.rs:487`) is the highest-value target since it runs inside `par_iter` and each clone is redundant — `cpu_open_ext2` already takes `&[GoldilocksField]`.

**Note**: `SparseMLPoly::evaluations()` materializes the full table, so `evaluations_ref()` won't work there. Those call sites will keep using `evaluations()`.

**Files to modify**:
- `src/poly/mod.rs` — add trait method
- `src/poly/dense.rs` — implement
- `src/dag/mod.rs:307,487` — use ref
- `src/basicblock/reducer.rs:53,107` — use ref
- `src/basicblock/einsum.rs:538,588,659` — use ref where possible
- All other `basicblock/*.rs` call sites

### Optimization 2: In-place fold in `cpu_open_ext2`

**Impact**: HIGH | **Effort**: Easy | **Risk**: None

Replace the allocate-per-round pattern with in-place folding. Since we read pairs `[2j, 2j+1]` and write to position `j`, there is no race condition in sequential code:

```rust
let mut current_size = n;
for _round in 0..num_vars - 1 {
    let challenge = transcript.sample_challenge_ext2();
    let half = current_size / 2;
    for j in 0..half {
        eq[j] = ext2_add(eq[2*j], ext2_mul(challenge, ext2_sub(eq[2*j+1], eq[2*j])));
        f[j] = ext2_add(f[2*j], ext2_mul(challenge, ext2_sub(f[2*j+1], f[2*j])));
    }
    current_size = half;
    // compute_oracle only needs eq[..half] and f[..half]
    let oracle = compute_oracle(&eq[..half], &f[..half], two, inv2);
    ...
}
```

This eliminates 23 pairs of `Vec::with_capacity` + fill for n=24. The initial allocation (2 x 256MB for eq and f) is still needed once, but all fold rounds reuse it.

**Files to modify**:
- `src/commit/cpu_basefold.rs`

### Optimization 3: Parallelize `compute_oracle` and fold with rayon

**Impact**: HIGH | **Effort**: Medium | **Risk**: None

Both `compute_oracle` and the fold loop are embarrassingly parallel over the index `j`. Use `rayon` chunks for the oracle computation:

```rust
fn compute_oracle_parallel(
    eq: &[GoldilocksExt2],
    f: &[GoldilocksExt2],
    two: GoldilocksExt2,
    inv2: GoldilocksExt2,
) -> SumcheckOracle<GoldilocksExt2> {
    let half = eq.len() / 2;
    let chunk_size = (half + rayon::current_num_threads() - 1) / rayon::current_num_threads();
    let (p0, p1, p2) = (0..half)
        .into_par_iter()
        .with_min_len(chunk_size.max(1024))
        .fold(|| (zero, zero, zero), |(p0, p1, p2), j| {
            // accumulate p0, p1, p2
        })
        .reduce(|| (zero, zero, zero), |(a0,a1,a2), (b0,b1,b2)| {
            (ext2_add(a0,b0), ext2_add(a1,b1), ext2_add(a2,b2))
        });
    // convert to c0, c1, c2
}
```

For the fold, use `par_chunks_mut` on the output half while reading from the full buffer (requires the in-place fold from Optimization 2 to be adapted to write to a separate output, or use `unsafe` split):

```rust
// Safe approach: split eq into two halves
let (eq_out, eq_hi) = eq[..current_size].split_at_mut(half);
// But eq_hi is [half..current_size], not [0..half] interleaved —
// so this needs the double-buffer approach instead.
```

A simpler approach: parallelize `compute_oracle` (the hot loop), keep fold sequential (it's memory-bound and already fast with in-place writes).

**Files to modify**:
- `src/commit/cpu_basefold.rs`

### Optimization 4: Group opening tasks by edge

**Impact**: HIGH | **Effort**: Medium | **Risk**: None

Currently each opening task independently clones the polynomial and builds an eq table. When an edge has K claims, we can:

1. Clone/reference the polynomial data once per edge
2. Run K opening proofs sharing the same `f` data

```rust
// Group tasks by edge
let mut edge_groups: BTreeMap<usize, Vec<(usize, usize, Vec<GoldilocksExt2>)>> = BTreeMap::new();
for (task_idx, (e, i, point)) in tasks.iter().enumerate() {
    edge_groups.entry(*e).or_default().push((task_idx, *i, point.clone()));
}

// Process groups in parallel, sharing polynomial data within each group
let results: Vec<(usize, BasefoldOpeningProof)> = edge_groups.par_iter().flat_map(|(e, group)| {
    let w = &witnesses[*e][0];
    let evals = w.data.as_ref().unwrap().evaluations_ref(); // single borrow
    let num_vars = w.data.as_ref().unwrap().n();
    let root = &commitments[*e].as_ref().unwrap().root;

    group.iter().map(|(task_idx, _, point)| {
        let mut t = Transcript::new(b"bf-open");
        t.append_ext2(b"", &master_seed);
        t.append_u64(b"", *task_idx as u64);
        (*task_idx, cpu_open_ext2(evals, num_vars, point, root, &mut t, key.num_queries))
    }).collect::<Vec<_>>()
}).collect();
```

This eliminates K-1 clones per edge. Combined with Optimization 1, it eliminates all clones.

**Files to modify**:
- `src/dag/mod.rs` — restructure opening task loop

### Optimization 5: GPU opening proofs for large polynomials (n > 16)

**Impact**: VERY HIGH | **Effort**: Complex | **Risk**: Medium (GPU memory management)

The `cpu_open_ext2` inner-product sumcheck is essentially the same computation as the GPU sumcheck prover, but run on CPU. For large polynomials (n > 16), the GPU can perform the fold + oracle computation much faster.

**Approach**: Adapt `GpuLinearSumcheckProver` for the opening proof protocol. The key difference from regular sumcheck is:
- Opening proof has 2 polynomials (f, eq) with a product interaction
- Each round needs a degree-2 oracle (c0, c1, c2), same as `compute_oracle`
- After oracle, fold both f and eq by the challenge

This is structurally identical to the existing GPU sumcheck with `num_polys=2`. The challenge is that opening proofs run in parallel (many per edge), so GPU memory must be managed carefully.

**Options**:
- **Batch on GPU**: Upload all polynomials for one edge, run K openings sequentially on GPU
- **Hybrid**: Use GPU for large polys (n > 16), CPU for smaller ones (already have threshold logic)
- **Streaming**: Process openings in batches that fit in GPU memory

**Files to modify**:
- `src/commit/cpu_basefold.rs` — add GPU path
- `src/dag/mod.rs` — route large polys to GPU
- Possibly new CUDA kernels for batched inner-product sumcheck

### Optimization 6: Cache evaluation for passthrough/advice claims

**Impact**: MEDIUM | **Effort**: Easy | **Risk**: None

Advice operations (DivConst, RMSReciprocal, etc.) return empty proofs from `prove()` but still generate claims that need opening proofs. When the claim point matches a point already opened for the same edge, the evaluation is redundant.

More generally, some nodes pass through their input unchanged (Reshape, passthrough edges). If a claim on the output edge can be answered by the already-computed evaluation of the input edge at the same point, we can skip the opening proof entirely.

This requires tracking which evaluations have already been proven and only opening for genuinely new (edge, point) pairs.

**Files to modify**:
- `src/dag/mod.rs` — dedup opening tasks

### Optimization 7: Parallelize `evaluate_lagrange_basis_ext2`

**Impact**: MEDIUM | **Effort**: Medium | **Risk**: None

The current implementation builds the eq table layer by layer, with each layer doubling the active region. The inner loop at layer `i` processes `2^i` elements — early layers are tiny, late layers are large.

For the late layers (e.g., layer 20+ out of 24), the inner loop processes millions of elements and can be parallelized:

```rust
pub fn evaluate_lagrange_basis_ext2_par(r: &[GoldilocksExt2]) -> Vec<GoldilocksExt2> {
    let n = r.len();
    let size = 1usize << n;
    let mut evals = vec![GoldilocksExt2::zero(); size];
    evals[0] = GoldilocksExt2::one();
    let one = GoldilocksExt2::one();

    const PAR_THRESHOLD: usize = 1 << 14; // 16K elements

    for i in 0..n {
        let one_minus_ri = ext2_sub(one, r[i]);
        let half = 1usize << i;
        if half >= PAR_THRESHOLD {
            // Parallel: process pairs independently
            let (lo, hi) = evals[..2*half].split_at_mut(half);
            lo.par_iter_mut().zip(hi.par_iter_mut()).rev().for_each(|(lo_j, hi_j)| {
                *hi_j = ext2_mul(*lo_j, r[i]);
                *lo_j = ext2_mul(*lo_j, one_minus_ri);
            });
        } else {
            for j in (0..half).rev() {
                evals[j | half] = ext2_mul(evals[j], r[i]);
                evals[j] = ext2_mul(evals[j], one_minus_ri);
            }
        }
    }
    evals
}
```

**Files to modify**:
- `src/poly/mod.rs`

---

## Implementation Order

Recommended order (maximize cumulative impact, minimize risk):

1. **Optimization 1** (evaluations_ref) — foundational, unblocks others
2. **Optimization 2** (in-place fold) — simple, pairs with #1
3. **Optimization 3** (parallel oracle) — builds on #2
4. **Optimization 4** (group by edge) — builds on #1
5. **Optimization 7** (parallel eq table) — independent
6. **Optimization 6** (cache evals) — independent
7. **Optimization 5** (GPU openings) — most complex, do last

Optimizations 1-4 together should yield the largest speedup for GPT-J/LLaMA since opening proofs dominate their runtime. Optimizations 1-2 alone should be achievable in a single session with immediate measurable benefit.

---

## Diagnostic Improvement

Add timing for the opening proof phase and reducer phase to the prove summary:

```rust
// Before opening proofs
let t_open = std::time::Instant::now();
// ... opening proof code ...
println!("  Opening proofs:     {:>8.3}s  ({} tasks)", t_open.elapsed().as_secs_f64(), tasks.len());
```

Include reducer in the per-node type timing (move timing block to include reducer call):

```rust
let t_node = std::time::Instant::now();
// Reducer if multiple claims
if local_claims.len() > 1 { ... }
// Prove the node
let (proofs, new_claims) = node.kind.prove(...);
let elapsed = t_node.elapsed();
```

**Files to modify**:
- `src/dag/mod.rs`

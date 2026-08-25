# Parallel Proving via DAG Partitioning

## Overview

Split the DAG at boundary edges into N partitions that can be proved independently and in parallel on separate GPUs. Each partition is a self-contained sub-proof connected by polynomial commitments at the boundaries.

**Expected speedup**: ~N× on the sumcheck-dominated prover phase (currently 300s for GPT-J 28L, 480s for LLaMA 32L).

---

## 1. Protocol Design

### Current Protocol (Sequential)

```
Commit all polynomials
Draw output challenges: r_out
Backward pass: output → ... → inputs (single transcript, sequential)
Lookup proofs
Opening proofs (parallel)
```

### Parallel Protocol

```
1. Commit all polynomials INCLUDING boundary edges B_1, ..., B_{N-1}
2. Draw output challenges: r_out from transcript
3. Draw boundary challenges: r_bnd_k from transcript (one per boundary edge)
4. Prover evaluates: v_bnd_k = B_k(r_bnd_k)
5. Fork transcript into N partition transcripts: T_k = seed(master_state, k)
6. Prove N partitions in parallel:
   - Partition N:  starts from r_out,         proves backward to B_{N-1}
   - Partition k:  starts from r_bnd_k,       proves backward to B_{k-1}
   - Partition 1:  starts from r_bnd_1,       proves backward to inputs
7. Collect all results
8. Lookup proofs (after all partitions complete)
9. Opening proofs (parallel, across all edges)
```

### Soundness Argument

Each partition k proves: "the computation from B_{k-1} to B_k is correct."

- Partition k's sumcheck chain reduces B_k's claim to claims on B_{k-1} and other edges
- Opening proofs for B_{k-1} verify these claims against B_{k-1}'s commitment
- Opening proof for B_k at r_bnd_k verifies Part k's starting claim
- Commitment binding ensures all partitions reference the same polynomials

Since each partition independently proves a segment, and commitments cryptographically glue them together, the full computation from inputs to outputs is verified. Soundness error is additive: N × ε (negligible for any reasonable N).

---

## 2. DAG Partitioning

### Split Point Selection

For transformer models, the natural split point is between layers. Each layer's output is a single hidden-state edge (e.g., shape `[1, 1, 4096]`). This gives:

- **1 boundary edge per split point** (the hidden state tensor)
- **No cross-partition weight sharing** (each layer has its own weights)
- **Clean topological separation** (no edges skip across split points)

### Partition Descriptor

```rust
struct PartitionDesc {
    partition_id: usize,
    node_ids: Vec<NodeId>,           // Nodes in this partition
    input_edges: Vec<EdgeId>,        // Edges entering from previous partition (or DAG inputs)
    output_edges: Vec<EdgeId>,       // Edges leaving to next partition (or DAG outputs)
    boundary_input_edges: Vec<EdgeId>,  // Subset of input_edges that are boundaries
    boundary_output_edges: Vec<EdgeId>, // Subset of output_edges that are boundaries
}
```

### Partitioning Algorithm

```
partition_dag(dag, split_edges: Vec<EdgeId>) -> Vec<PartitionDesc>:
    1. Mark split_edges as boundary edges
    2. For each node, assign to a partition:
       - Walk backward from each split edge to find partition boundaries
       - Nodes between split_edge[k-1] and split_edge[k] belong to partition k
    3. Compute input_edges and output_edges for each partition
    4. Verify no edges cross more than one partition boundary
```

For transformers, a simpler approach works: split by topo_levels.

```
partition_by_levels(dag, cuts: Vec<usize>) -> Vec<PartitionDesc>:
    Partition k gets topo_levels[cuts[k-1]..cuts[k]]
    Boundary edges = edges produced by last level of partition k
                     and consumed by first level of partition k+1
```

### Identifying Split Edges in Transformer Binaries

Each binary (gpt2, bert, gptj, llama) builds the DAG layer by layer. The hidden state edge between layers is the natural split point. We need to record these edge IDs during DAG construction.

```rust
// In gpt2.rs:
let mut layer_boundary_edges = Vec::new();
for i in 0..num_layers {
    x = g.pipe(&[x], gpt2_layer(weights[i]));
    layer_boundary_edges.push(x);  // Record the boundary edge
}
```

Then: `partition_dag(dag, layer_boundary_edges[split_indices])`

---

## 3. Boundary Edge Commitment

### Current Behavior

`should_commit()` in `dag/mod.rs` only commits:
- Constants (weights, biases)
- Auxiliaries (lookup selection polynomials)
- Inputs (external inputs)
- Final outputs (edges with no consumers)

Intermediate outputs (hidden states between layers) are **NOT committed**.

### Required Change

Boundary edges must be committed. Two approaches:

**A. Mark during partitioning** (recommended):
```rust
// After partitioning, before commit():
for &edge_id in &boundary_edges {
    witnesses[edge_id][0].role = Role::Input;  // Force commitment
}
```

**B. Override should_commit**:
```rust
fn should_commit(&self, witness: &Witness, edge_id: EdgeId, boundary_edges: &HashSet<EdgeId>) -> bool {
    if boundary_edges.contains(&edge_id) { return true; }
    // ... existing logic
}
```

### Cost

One Basefold commitment per boundary edge. For a hidden state of shape `[1, 1, 4096]` (n=12), this is negligible (~1ms commit, ~1ms opening proof).

---

## 4. Transcript Management

### Forking Strategy

After drawing boundary challenges, fork the main transcript into per-partition transcripts with domain separation:

```rust
// Main transcript: absorb commitments, draw output + boundary challenges
let master_state = transcript.get_state();  // Need to expose this

// Per-partition transcript: seeded from master state + partition index
for k in 0..num_partitions {
    let mut t_k = Transcript::new(b"partition");
    t_k.append_ext2(b"seed", &master_state_as_ext2);
    t_k.append_scalar(b"idx", &GoldilocksField(k as u64));
    partition_transcripts.push(t_k);
}
```

### Required Transcript Changes

Add to `transcript.rs`:
```rust
impl Transcript {
    /// Get current sponge state for forking.
    pub fn get_state(&self) -> Vec<u64> {
        self.state.clone()
    }

    /// Create a transcript seeded from a parent state + domain separator.
    pub fn fork(parent_state: &[u64], partition_id: usize) -> Self {
        let mut t = Self::new(b"partition");
        for &s in parent_state {
            t.append_scalar(b"s", &GoldilocksField(s));
        }
        t.append_scalar(b"id", &GoldilocksField(partition_id as u64));
        t
    }
}
```

### Verification Transcript

The verifier performs the same forking:
1. Absorb commitments (same as prover)
2. Draw output + boundary challenges (same as prover)
3. Fork into per-partition transcripts (same seeding)
4. Verify each partition with its own transcript

---

## 5. Parallel Prove

### Modified `prove()` Signature

```rust
pub fn prove_parallel(
    &self,
    witnesses: &[Vec<Witness>],
    commitments: &[Option<BasefoldCommitmentData>],
    transcript: &mut Transcript,
    partitions: &[PartitionDesc],
    num_queries: usize,
) -> ParallelProof
```

### Algorithm

```
prove_parallel(dag, witnesses, commitments, transcript, partitions):

    // Phase 1: Output + boundary challenges (sequential, single transcript)
    for e in dag.output_ports:
        output_claims[e] = (transcript.challenge_ext2(), witnesses[e].evaluate(point))

    boundary_claims = {}
    for k in 1..partitions.len():
        for e in partitions[k].boundary_input_edges:
            r = transcript.challenge_ext2(num_vars times)
            v = witnesses[e].evaluate(r)
            boundary_claims[e] = Claim { edge_id: e, point: r, eval: v }

    // Phase 2: Fork transcript
    master_state = transcript.get_state()
    partition_transcripts = [Transcript::fork(master_state, k) for k in 0..N]

    // Phase 3: Prove partitions in parallel
    partition_proofs = partitions.par_iter().enumerate().map(|(k, part)| {
        let mut t_k = partition_transcripts[k].clone()

        // Determine starting claims for this partition
        starting_claims = if k == N-1 {
            output_claims  // Last partition starts from output
        } else {
            // Partition k's output = partition (k+1)'s boundary input
            // But partition k starts from its OWN boundary output claims
            boundary_claims for part.boundary_output_edges
            // Wait — this is the claim on partition k's OUTPUT edge
            // Which is the boundary edge B_k
            // The claim is: B_k(r_bnd_k) = v_bnd_k
        }

        // Run standard backward pass on this partition's nodes
        prove_partition(dag, part, witnesses, starting_claims, commitments, &mut t_k)
    }).collect()

    // Phase 4: Lookup proofs (sequential, using main transcript)
    // All lookup nodes across all partitions
    lookup_proofs = prove_lookups(dag, witnesses, &all_claims, transcript)

    // Phase 5: Opening proofs (parallel, master_seed from main transcript)
    master_seed = transcript.challenge_ext2()
    opening_proofs = prove_openings(witnesses, commitments, &all_claims, master_seed)

    return ParallelProof { partition_proofs, lookup_proofs, opening_proofs }
```

### `prove_partition()` — Standard backward pass on a subset

```rust
fn prove_partition(
    dag: &Dag,
    partition: &PartitionDesc,
    witnesses: &[Vec<Witness>],
    starting_claims: HashMap<EdgeId, Vec<Claim>>,
    commitments: &[Option<BasefoldCommitmentData>],
    transcript: &mut Transcript,
) -> PartitionProof {
    let mut claims: HashMap<EdgeId, Vec<Claim>> = starting_claims;
    let mut nodes_to_prove: BTreeSet<NodeId> = /* producers of starting claim edges, filtered to this partition */;

    // Standard backward pass, but only process nodes in partition.node_ids
    while !nodes_to_prove.is_empty() {
        let node_id = nodes_to_prove.pop_last();
        // ... same logic as current prove() ...
        // When a new claim is on a boundary_input_edge: DON'T add producer to nodes_to_prove
        // (that producer belongs to another partition)
        // Instead: collect the claim for opening proof verification
    }

    PartitionProof { node_proofs, edge_claims, reducer_proofs }
}
```

### Key Difference from Standard Prove

When the backward pass produces a claim on a **boundary input edge** (an edge entering this partition from a previous partition), the claim is NOT propagated to a producer node. Instead:
- The claim is collected for opening proof verification
- The opening proof will verify the claim against the committed polynomial

This is the "cut" that allows independence.

---

## 6. Parallel Verify

### Algorithm

```
verify_parallel(dag, proof, commitments, transcript, partitions):

    // Phase 1: Re-derive output + boundary challenges
    // (Same transcript ops as prover)
    output_challenges = re_derive(transcript)
    boundary_challenges = re_derive(transcript)

    // Phase 2: Fork transcript
    master_state = transcript.get_state()
    partition_transcripts = [Transcript::fork(master_state, k) for k in 0..N]

    // Phase 3: Verify partitions in parallel
    for k in 0..N (parallel):
        verify_partition(dag, proof.partition_proofs[k], partitions[k],
                        starting_claims[k], commitments, &mut partition_transcripts[k])

    // Phase 4: Verify lookup proofs
    verify_lookups(proof.lookup_proofs, transcript)

    // Phase 5: Verify opening proofs (parallel)
    master_seed = transcript.challenge_ext2()
    verify_openings(proof.opening_proofs, commitments, master_seed)
```

---

## 7. Multi-GPU Dispatch

### Phase 1: Single-GPU Parallelism (threads only)

Use rayon to parallelize partition proofs on CPU. Each partition's sumcheck uses the same GPU (sequentially) but overall the work is split:

```rust
// Prove partitions in parallel using rayon
let results: Vec<_> = partitions.par_iter().enumerate().map(|(k, part)| {
    prove_partition(dag, part, witnesses, claims[k], commitments, &mut transcripts[k])
}).collect();
```

The sumcheck within each partition still uses the GPU (for Einsum). With rayon parallelism, different partitions' CPU work (claim propagation, reducer) overlaps with GPU work.

### Phase 2: Multi-GPU Parallelism

Each partition gets its own CUDA device:

```rust
// Each thread sets its own CUDA device
std::thread::scope(|s| {
    for (k, part) in partitions.iter().enumerate() {
        s.spawn(move || {
            cuda_set_device(k);  // GPU k for partition k
            prove_partition(dag, part, witnesses, claims[k], commitments, &mut transcripts[k])
        });
    }
});
```

Required changes to `goldilocks-cuda-rs`:
1. Add `cuda_set_device(device_id)` FFI wrapper
2. Each GPU needs its own `BasefoldTable` (allocated on that device)
3. GPU memory management per device
4. `GpuSumcheckStateExt2` already uses device-local buffers

### CUDA Multi-Device Changes

```c
// In wrapper.cu:
extern "C" int cuda_set_device(int device) {
    return cudaSetDevice(device);
}
```

```rust
// In goldilocks-cuda-rs:
pub fn set_device(device: i32) -> Result<()> {
    let ret = unsafe { ffi::cuda_set_device(device) };
    if ret != 0 { return Err(CudaError::NoDevice); }
    Ok(())
}
```

---

## 8. Data Structures

### ParallelProof

```rust
pub struct ParallelProof {
    pub num_partitions: usize,
    pub boundary_evals: Vec<(EdgeId, Vec<GoldilocksExt2>, GoldilocksExt2)>, // (edge, point, eval)
    pub partition_proofs: Vec<PartitionProof>,
    pub range_proof: Option<RangeProof>,
    pub two_pow_proof: Option<TwoPowProof>,
    pub edge_proofs: Vec<EdgeProof>,  // Opening proofs, globally indexed by edge_id
}

pub struct PartitionProof {
    pub partition_id: usize,
    pub node_proofs: HashMap<NodeId, NodeProof>,
    pub reducer_proofs: HashMap<NodeId, ReducerProof>,
    pub boundary_claims: Vec<Claim>,  // Claims on boundary input edges (for opening proofs)
}
```

---

## 9. Implementation Phases

### Phase A: DAG Partitioning Infrastructure

**Files**: `dag/partition.rs` (new), `dag/mod.rs`

1. Define `PartitionDesc` struct
2. Implement `partition_by_layer()`: given split edge IDs, produce partition descriptors
3. Add `boundary_edges: Vec<EdgeId>` field to `Dag`
4. Modify binaries to record layer boundary edges during construction
5. Unit test: partition a simple 2-node DAG

### Phase B: Boundary Commitment

**Files**: `dag/mod.rs`

1. Modify `should_commit()` to accept boundary edge set
2. Force-commit boundary edges
3. Verify boundary edges get proper Basefold commitments
4. Test: commit + open boundary edge in a 2-partition DAG

### Phase C: Transcript Forking

**Files**: `transcript.rs`

1. Add `get_state()` method
2. Add `fork(parent_state, partition_id)` constructor
3. Test: fork produces different challenges for different partition IDs
4. Test: fork is deterministic (same inputs → same transcript)

### Phase D: Partition-Aware Prove

**Files**: `dag/mod.rs`

1. Implement `prove_partition()` function
2. Handle boundary input edges: collect claims without propagating to producers
3. Handle boundary output edges: use as starting claims
4. Implement `prove_parallel()` orchestrator
5. Test with 2 partitions on a small DAG (unit test)

### Phase E: Partition-Aware Verify

**Files**: `dag/mod.rs`

1. Implement `verify_partition()` function
2. Implement `verify_parallel()` orchestrator
3. Verify boundary opening proofs
4. Test: prove_parallel + verify_parallel on small DAG

### Phase F: Integration with Model Binaries

**Files**: `bin/gpt2.rs`, `bin/bert.rs`, `bin/gptj.rs`, `bin/llama.rs`

1. Record layer boundary edges during DAG construction
2. Accept `NUM_PARTITIONS` env var
3. Compute split points: evenly divide layers across partitions
4. Call `prove_parallel()` instead of `prove()`
5. Test: GPT-2 12L with 2 partitions verifies correctly

### Phase G: Multi-GPU Support

**Files**: `goldilocks-cuda-rs/src/ffi.rs`, `goldilocks-cuda-rs/cuda/wrapper.cu`

1. Add `cuda_set_device()` FFI
2. Per-device `BasefoldTable` allocation
3. Thread-local GPU context management
4. Test: 2 partitions on 2 GPUs
5. Benchmark: GPT-J 28L with 2 GPUs vs 1 GPU

---

## 10. Expected Performance

### Current Bottleneck Analysis (GPT-J 28L)

| Phase | Time | % |
|-------|------|---|
| Einsum sumcheck | ~280s | 93% |
| Other node proofs | ~15s | 5% |
| Opening proofs | ~4s | 1% |
| Lookup proofs | ~1s | <1% |

### With 2 Partitions (14 layers each)

| Phase | Time | Notes |
|-------|------|-------|
| Commit boundary | ~0.001s | n=12 polynomial |
| Draw boundary challenges | ~0s | Negligible |
| Partition 1 prove | ~150s | Half the nodes |
| Partition 2 prove | ~150s | Half the nodes, parallel |
| Lookup proofs | ~1s | Sequential, unchanged |
| Opening proofs | ~4s | +1 boundary opening, negligible |
| **Total** | **~155s** | **~2× speedup** |

### With 4 Partitions (7 layers each)

| Phase | Time | Notes |
|-------|------|-------|
| Partition proofs (×4) | ~75s each | Quarter the nodes, parallel |
| Overhead | ~5s | Lookups + openings |
| **Total** | **~80s** | **~3.75× speedup** |

### Scaling Limits

- N = number of layers: diminishing returns (overhead from boundary commitments, lookup serialization)
- Practical limit: N ≤ 4-8 (one partition per GPU)
- Each GPU needs ~16GB for sumcheck state of its partition

---

## 11. Correctness Testing Strategy

1. **Bit-identical oracles**: For N=1 (single partition), `prove_parallel` must produce identical proofs to `prove`
2. **Cross-verify**: Proof from `prove_parallel` verified by standard `verify`; proof from `prove` verified by `verify_parallel`
3. **Random split points**: Test with various partition counts (1, 2, 4, odd splits)
4. **All models**: GPT-2 (12L), BERT (24L), GPT-J (28L), LLaMA (32L) with 2 and 4 partitions
5. **Edge cases**: Partition with 1 layer, partition with lookup nodes spanning boundary

# ZK-Torch Prover Architecture Book

A comprehensive guide to the ZK-Torch proving system: architecture, protocols, BasicBlocks, GPU acceleration, and model catalog.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Data Structures](#2-core-data-structures)
3. [Polynomial Representations](#3-polynomial-representations)
4. [Transcript](#4-transcript)
5. [Forward Pass](#5-forward-pass)
6. [Commitment Scheme](#6-commitment-scheme)
7. [Sumcheck Protocol](#7-sumcheck-protocol)
8. [Backward Pass & Prove](#8-backward-pass--prove)
9. [BasicBlocks Reference](#9-basicblocks-reference) — A: Zero-Sumcheck, B: H-Polynomial, C: Convolution, D: Einsum, E: Lookup Aux, F: Advice
10. [Lookup Protocols](#10-lookup-protocols)
11. [Opening Proofs](#11-opening-proofs)
12. [Partition-Aware Parallel Proving](#12-partition-aware-parallel-proving)
13. [Model Catalog & Running Guide](#13-model-catalog--running-guide)
14. [Prover Optimizations](#14-prover-optimizations)

---

## 1. System Overview

### Pipeline

Every ZK-Torch proof follows a five-stage pipeline:

```
Build  -->  Run  -->  Commit  -->  Prove  -->  Verify
```

| Stage    | What happens |
|----------|-------------|
| **Build** | Construct the computation DAG via `DagBuilder`. Wire inputs, weights, and operations. Call `compile()` to produce a `Dag` + initial `Witness` arrays. |
| **Run**   | Execute the forward pass. `dag.run()` traverses topological levels in parallel, calling each node's `BasicBlock::run()` to populate output witnesses. |
| **Commit** | Commit edge polynomials using the Basefold PCS on GPU. Produces Merkle roots (`BasefoldCommitmentData`) for the verifier and GPU-resident commitments for the prover. |
| **Prove**  | Backward pass through the DAG. Propagate claims from outputs to inputs via sumcheck proofs. Then perform lookup proofs (range, two-pow) and polynomial opening proofs. |
| **Verify** | Replay the transcript, check every sumcheck, verify lookup arguments, and validate opening proofs against Merkle commitments. |

### Goldilocks Field

All arithmetic is over the **Goldilocks prime field** `p = 2^64 - 2^32 + 1`. This field is chosen for:
- Efficient 64-bit modular arithmetic
- NTT-friendly structure (large 2-adic subgroup)
- GPU-friendly word size

The extension field `GoldilocksExt2 = GF(p^2)` is used during the proving phase, constructed as `GF(p)[x] / (x^2 - 7)`. All challenges and evaluation points live in Ext2, while witness data remains in the base field.

### Little-Endian Convention

ZK-Torch uses a **little-endian polynomial convention** throughout:

- **Variable 0 = bit 0 = LSB** of the evaluation index
- `DenseMLPoly::fix_variables` fixes variable 0 first, operating on pairs `[2j, 2j+1]`
- For a tensor with shape `[C, H, W]`, the MLE index is `w + h * W_pad + c * W_pad * H_pad`
  - W bits are lowest (variable 0, 1, ...)
  - H bits are middle
  - C bits are highest
- `evaluate_lagrange_basis` uses the `evals[j | half]` pattern for the eq polynomial table

This convention must be consistent across `char_to_range` (einsum), `einsum_compute`, polynomial evaluation, and all BasicBlock indexing.

---

## 2. Core Data Structures

### DAG (`src/dag/mod.rs`)

The `Dag` struct represents the computation graph:

```rust
pub struct Dag {
    pub nodes: Vec<Node>,           // All computation nodes
    pub num_edges: usize,           // Total number of edges
    pub topo: Vec<NodeId>,          // Topological order (flat)
    pub topo_levels: Vec<Vec<NodeId>>, // Level-grouped topo order (for parallel run)
    pub range: Vec<NodeId>,         // Nodes requiring range checks
    pub two_pow: Vec<NodeId>,       // Nodes requiring two-pow lookups
    pub consumers: Vec<Vec<NodeId>>,  // edge -> list of consuming nodes
    pub producers: Vec<Option<NodeId>>, // edge -> producing node (None for inputs)
    pub input_ports: Vec<EdgeId>,   // External input edges
    pub output_ports: Vec<EdgeId>,  // Terminal output edges
    pub layer_boundaries: Vec<usize>, // Layer boundary node indices
    pub boundary_edges: Vec<EdgeId>,  // Partition boundary edges (committed)
    pub self_claim_edges: HashSet<EdgeId>, // Edges claimed via PCS (Conv2D outputs)
    // Alias system for multiple consumers of the same edge
    pub edge_aliases: Vec<Vec<AliasId>>,
    pub alias_to_edge: Vec<EdgeId>,
    pub alias_to_consumer: Vec<NodeId>,
    pub alias_input_slot: Vec<usize>,
}
```

### Node

```rust
pub struct Node {
    pub id: NodeId,
    pub kind: BasicBlockType,   // The operation (Add, Einsum, Conv2D, etc.)
    pub inputs: Vec<EdgeId>,    // Input edge IDs
    pub outputs: Vec<EdgeId>,   // Output edge IDs
}
```

### Edge

Edges are identified by `EdgeId` (a `usize` index). They carry data (witnesses) between nodes. An edge can have multiple consumers but at most one producer.

### Witness

```rust
pub struct Witness {
    pub shape: Vec<usize>,           // Logical tensor shape
    pub data: Option<DenseMLPoly>,   // MLE polynomial (evaluations)
    pub data_int: Option<Vec<i128>>, // Integer interpretation (for range checks)
    pub data_type: DataType,         // Uint or Int
    pub sf: u32,                     // Scale factor (fixed-point)
    pub role: Role,                  // Input, Constant, Output, Auxiliary
    pub selection_poly: Option<SparseMLPoly>, // For two-pow lookups
    pub bit_aux: Option<DenseMLPoly>,         // Dense bit polynomial (range checks)
}
```

### Claim

```rust
pub struct Claim {
    pub edge_id: EdgeId,
    pub sparse_id: usize,              // 0 for dense claims
    pub point: Vec<GoldilocksExt2>,    // Evaluation point
    pub eval: GoldilocksExt2,          // Claimed evaluation
}
```

### EdgeProof

```rust
pub struct EdgeProof {
    pub claims: Vec<Claim>,
    pub dense_opening_proof: Vec<BasefoldOpeningProof>,
}
```

Each edge accumulates claims during the backward pass. After all node proofs are generated, opening proofs bind each claim to the committed polynomial.

### DagBuilder (`src/dag/builder.rs`)

The builder API for constructing computation graphs:

```rust
let mut g = DagBuilder::new();
let x = g.input(vec![768, 128], DataType::Uint);  // Input tensor
let w = g.param(weight_witness);                    // Weight parameter
let y = g.einsum("ab,bc->ac".to_string(), vec![x, w], false);
let z = g.add(y[0], bias);
let (dag, witnesses) = g.compile();
```

Key builder methods: `input()`, `param()`, `add()`, `sub()`, `einsum()`, `conv2d()`, `conv2d_strided()`, `conv2d_dilated()`, `conv3d_strided()`, `conv1d_strided()`, `depthwise_conv2d()`, `pad()`, `pad_asym()`, `pad3d()`, `pad1d()`, `maxpool2d()`, `maxpool_general()`, `relu()`, `concat()`, `general_concat()`, `multi_concat()`, `channel_slice()`, `subsample2d_sized()`, `reduce_mean()`, `change_shape()`, `scale_down()`, `scale_up()`.

`compile()` performs Kahn's topological sort with level tracking, builds the alias system for edges with multiple consumers, and returns `(Dag, Vec<Vec<Witness>>)`.

---

## 3. Polynomial Representations

### DenseMLPoly (`src/poly/dense.rs`)

The core polynomial type: a multilinear extension (MLE) stored as a vector of `2^n` evaluations over the Boolean hypercube.

```rust
pub struct DenseMLPoly {
    num_var: usize,                    // n (number of variables)
    evaluations: Vec<GoldilocksField>, // 2^n evaluations
}
```

Key methods:

- **`fix_variables(point)`** -- Partially evaluates the polynomial by fixing variables starting from variable 0 (LSB). For each variable value `r`, updates pairs: `evals[j] = evals[2j] * (1-r) + evals[2j+1] * r`.
- **`evaluate_at_point_ext2(point)`** -- Full evaluation at an Ext2 point. Calls `fix_variables` with all coordinates.
- **`evaluate_ext2_gpu(point)`** -- GPU-accelerated evaluation using `partial_eval_ext2_device_u64` for polynomials with n > 14.
- **`index(i)`** -- Direct access to evaluation at Boolean point `i`.
- **`n()`** -- Returns `num_var`.

### SparseMLPoly (`src/poly/sparse.rs`)

A sparse representation for polynomials with few nonzero evaluations:

```rust
pub struct SparseMLPoly {
    pub num_var: usize,
    pub entries: Vec<(usize, GoldilocksField)>, // (index, value) pairs
}
```

- **`evaluate_at_point_ext2(point)`** -- O(k*n) evaluation: for each entry, compute `eq(point, entry_index)` via per-bit product, then accumulate `value * eq_val`. Avoids materializing the full `2^n` table.

### SelectionPolynomial

```rust
pub struct SelectionPolynomial {
    pub entries: Vec<(usize, usize)>, // (input_index, table_index) pairs
    pub num_input_vars: usize,
    pub num_table_vars: usize,
}
```

Used for two-pow lookup proofs. Converts to `SparseMLPoly` with `n = num_input_vars + num_table_vars`.

### evaluate_lagrange_basis / evaluate_lagrange_basis_ext2

Computes the eq polynomial table: `eq(r, x) = prod_j (r_j * x_j + (1-r_j)(1-x_j))` for all `x in {0,1}^n`.

Uses the `evals[j | half]` pattern (little-endian):
```
For each variable r_j:
  half = 1 << j
  for j in 0..half:
    evals[j | half] = evals[j] * r_j
    evals[j] = evals[j] - evals[j | half]
```

Parallelized with `split_at_mut(half)` + `par_iter_mut` when `half >= 8192`.

---

## 4. Transcript

### Poseidon2 Sponge (`src/transcript.rs`)

The `Transcript` struct implements the Fiat-Shamir transform using a Poseidon2-like algebraic sponge hash:

```rust
pub struct Transcript {
    state: [GoldilocksField; 12],  // Poseidon2 state (width 12)
    pending: Vec<GoldilocksField>, // Buffer for absorption
}
```

Key methods:

- **`new(label)`** -- Creates a new transcript. The label is absorbed into the initial state.
- **`append_scalar(label, value)`** -- Absorbs a base-field element. Labels are absorbed but functionally ignored (they contribute to the sponge state but don't affect security since the sponge is collision-resistant).
- **`append_ext2(label, value)`** -- Absorbs an Ext2 element (two base-field elements).
- **`append_u64(label, value)`** -- Absorbs a u64 as a field element.
- **`challenge_scalar(label)`** -- Squeezes a base-field challenge.
- **`challenge_ext2(label)`** -- Squeezes an Ext2 challenge (two base-field squeezes).
- **`fork(k)`** -- Creates a copy of the transcript with partition index `k` absorbed, used for domain separation in parallel proving.
- **`fingerprint()`** -- Returns a hash of the current state for debugging (not used in proofs).

### BasefoldTranscript Trait

The Basefold PCS uses a `BasefoldTranscript` trait, which `Transcript` implements:

```rust
impl BasefoldTranscript for Transcript {
    fn observe_field(&mut self, value: GoldilocksField);
    fn observe_ext2(&mut self, value: GoldilocksExt2);
    fn observe_hash(&mut self, hash: &Poseidon2Hash);
    fn sample_challenge(&mut self) -> GoldilocksField;
    fn sample_challenge_ext2(&mut self) -> GoldilocksExt2;
}
```

---

## 5. Forward Pass

### `dag.run()` (`src/dag/mod.rs`)

The forward pass executes the computation graph to produce all intermediate tensors:

```rust
pub fn run(&self, witnesses: &mut [Vec<Witness>], inputs: &[(usize, Witness)])
```

**Algorithm:**

1. **Load inputs**: Set witness data for input edges from the provided `inputs` list.
2. **Level-parallel execution**: Iterate through `topo_levels` (computed during `compile()`). Each level contains nodes with no dependencies on other nodes in the same level.
3. **Per-node execution**: For each node in a level (processed in parallel via `par_iter`):
   - Gather input witnesses
   - Call `BasicBlock::run()` to compute outputs
   - Store output witnesses
4. **Auxiliary generation**: Nodes in `self.range` and `self.two_pow` may produce additional auxiliary witnesses (bit decomposition polynomials, selection polynomials).

The `topo_levels` structure enables natural parallelism: all nodes within a level can execute concurrently since they have no inter-dependencies.

### Parallelization

- `einsum_compute` uses `(0..out_size).into_par_iter()` with pre-computed strides
- Level processing uses `par_iter()` over nodes within each level
- For GPT-2 12-layer: run time reduced from 26.2s to 1.5s (17x speedup)

---

## 6. Commitment Scheme

### Basefold PCS

ZK-Torch uses the **Basefold** polynomial commitment scheme, implemented primarily on GPU. Basefold combines:
- **Reed-Solomon encoding**: Polynomial evaluations are encoded into a codeword
- **Merkle tree**: Codeword hashed into a Merkle tree for binding
- **FRI-like folding**: Opening proofs use interactive folding with query proofs

### Key Structures

#### BasefoldTable (`goldilocks-cuda-rs/src/basefold.rs`)

```rust
pub struct BasefoldTable {
    pub entries: Vec<FoldingEntry>,  // Random folding points + weights
    pub level_offsets: Vec<usize>,   // Per-level offset into entries
    pub level_sizes: Vec<usize>,     // Per-level size
    pub num_rounds: usize,
    d_entries: Option<DeviceBuffer<u64>>, // GPU copy
}
```

Generated deterministically from a seed. Each entry contains a random evaluation point and its precomputed inverse weight for the folding step.

#### BasefoldCommitment

```rust
pub struct BasefoldCommitment {
    pub root: Poseidon2Hash,              // Merkle root
    d_codeword: DeviceBuffer<u64>,        // GPU: RS codeword
    d_bh_evals: DeviceBuffer<u64>,        // GPU: BHC evaluations
    merkle_tree: DeviceMerkleTree,        // GPU: full Merkle tree
    pub num_vars: usize,
    pub log_rate: usize,
}
```

Heavy data lives on GPU. The commitment process:
1. Upload evaluations to GPU
2. BHC interpolation (evaluation -> coefficient basis)
3. Reed-Solomon encoding (coefficients -> codeword)
4. Merkle tree construction over the codeword
5. Read back only the 32-byte root to host

#### BasefoldCommitmentData (Verifier-side)

```rust
pub struct BasefoldCommitmentData {
    pub root: Poseidon2Hash,
    pub num_vars: usize,
}
```

Lightweight verifier-side commitment (just the Merkle root).

#### HostCommitmentCache

```rust
pub struct HostCommitmentCache {
    pub root: Poseidon2Hash,
    pub codeword: Vec<u64>,
    pub bh_evals: Vec<u64>,
    pub num_vars: usize,
    pub log_rate: usize,
}
```

CPU-side cache for partition-aware GPU placement. Used when GPU memory is insufficient to hold all commitments simultaneously.

### GpuCommitmentStore (`src/commit/basefold.rs`)

Manages all commitments across the proving pipeline:

```rust
pub struct GpuCommitmentStore {
    pub commitments: Vec<Option<BasefoldCommitment>>,  // GPU commitments
    pub host_caches: Vec<Option<HostCommitmentCache>>, // CPU fallback
    pub table: BasefoldTable,                           // Folding table
    pub per_device_tables: Vec<BasefoldTable>,          // Per-GPU table copies
    pub device_ids: Vec<Option<i32>>,                   // GPU device per edge
}
```

### Commitment Constraints

| Constant | Value | Reason |
|----------|-------|--------|
| `MIN_BASEFOLD_VARS` | 2 | GPU kernel fails on tiny polynomials |
| `MAX_BASEFOLD_VARS` | 22 | GPU OOM on polynomials larger than 2^22 |

### Commit Flow

```rust
dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, edge_partition_map)
```

1. Iterate over all edges
2. Skip edges that don't need commitment (outputs without consumers, non-committed roles)
3. For edges in `self_claim_edges` or `boundary_edges`: always commit
4. Upload evaluations, commit on GPU, store `BasefoldCommitmentData` for verifier
5. Optionally download to `HostCommitmentCache` for partition-aware memory management

---

## 7. Sumcheck Protocol

The sumcheck protocol is the core building block of ZK-Torch proofs. It reduces a claim about a sum over the Boolean hypercube to a claim about a single evaluation point.

### Protocol Overview

**Claim**: `H = sum_{x in {0,1}^n} g(x)` where `g` is a low-degree polynomial.

**Round i** (for i = 0, 1, ..., n-1):
1. Prover sends univariate polynomial `s_i(X) = sum_{x_{i+1},...,x_{n-1}} g(r_0,...,r_{i-1}, X, x_{i+1},...)`
2. Verifier checks `s_i(0) + s_i(1) = current_sum`
3. Verifier sends random challenge `r_i`
4. Update `current_sum = s_i(r_i)`

**Final check**: Verifier checks `current_sum == g(r_0, ..., r_{n-1})` (the final evaluation).

### SumcheckProof

```rust
pub struct SumcheckProof {
    pub final_eval: GoldilocksExt2,
    pub round_messages: Vec<Vec<GoldilocksExt2>>,  // Per-round univariate evaluations
}
```

### Linear Sumcheck

Most ZK-Torch sumchecks are **linear**: `g(x) = f_0(x) * f_1(x)` (degree-2 product of two MLEs). The round polynomial is degree-2, so 3 evaluation points suffice per round: `s_i(0), s_i(1), s_i(2)`.

### GPU Sumcheck Prover (`src/sumcheck/gpu_prover.rs`)

```rust
pub struct GpuLinearSumcheckProver {
    pub num_var: usize,
    pub num_poly: usize,
    pub challenges: Vec<GoldilocksExt2>,
}
```

**Architecture:**
- Polynomials packed contiguously on GPU with `stride = original_size`
- Double-buffered fold: separate input/output GPU buffers, swapped after each round
- Block reduction: grid-stride loop + shared memory tree reduction + host sums block partials
- `prove_gpu_resident()`: accepts pre-built `GpuSumcheckStateExt2` (polynomials already on GPU)

**Critical design choice**: Double buffering was necessary because the original in-place fold had a cross-warp race condition (thread y=k writes to position k while thread y=k/2 reads from position k).

### CPU Ext2 Sumcheck Prover (`src/sumcheck/cpu_ext2_prover.rs`)

```rust
pub struct CpuLinearSumcheckProverExt2 {
    pub num_var: usize,
    pub num_poly: usize,
    pub challenges: Vec<GoldilocksExt2>,
}
```

Drop-in replacement for GPU prover on small polynomials. Operates directly on `Vec<GoldilocksExt2>` arrays.

### General Sumcheck Prover (`src/sumcheck/general_prover.rs`)

For higher-degree sumchecks (e.g., degree-3 in DepthwiseConv2D):

```rust
pub struct GeneralLinearSumcheckProver {
    pub num_var: usize,
    pub num_poly: usize,           // Degree of the product
    pub eq: DenseMLPoly,
    pub a_scalars: Vec<GoldilocksField>,
    pub a_arrays: Vec<Vec<DenseMLPoly>>,
}
```

Computes `H = sum_x eq(r, x) * sum_j (a_j * prod_i polys[j][i](x))`. Each round message has `num_poly + 1` evaluation points.

### Sumcheck Verifier (`src/sumcheck/verifier.rs`)

```rust
impl SumcheckVerifier {
    pub fn verify(
        proof: &SumcheckProof,
        claimed_sum: GoldilocksExt2,
        num_var: usize,
        num_poly: usize,
        transcript: &mut Transcript,
    ) -> (bool, Vec<GoldilocksExt2>)
}
```

Checks each round: `s_i(0) + s_i(1) = current_sum`, then Lagrange-interpolates `s_i` at the challenge point. Returns `(verified, challenges)`.

### Thresholds

| Threshold | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| GPU Sumcheck | `ZK_GPU_SUMCHECK_THRESHOLD` | 14 | Use GPU prover when n > threshold |
| GPU Partial Eval | `ZK_GPU_PARTIAL_EVAL_THRESHOLD` | 16 | Use GPU for partial evaluation in CPU sumcheck path |
| GPU Fused Permute | `ZK_GPU_FUSED_THRESHOLD` | 16 | Use fused permute+partial_eval GPU kernel |

---

## 8. Backward Pass & Prove

### `dag.prove()` (`src/dag/mod.rs`)

The prove function performs the backward pass through the DAG:

```rust
pub fn prove(
    &self,
    key: &BasefoldCommitKey,
    witnesses: &mut [Vec<Witness>],
    commitments: &[Option<BasefoldCommitmentData>],
    gpu_store: &GpuCommitmentStore,
    table: &BasefoldTable,
    transcript: &mut Transcript,
    timing: &mut TimingTree,
) -> (Vec<NodeProof>, Vec<EdgeProof>, RangeProof, TwoPowProof, Vec<ReducerProof>)
```

**Algorithm:**

1. **Initialize output claims**: For each output edge, generate a random evaluation point and evaluate the witness polynomial at that point.

2. **Backward traversal**: Process nodes in reverse topological order. For each node:
   - Collect all claims on its output edges
   - If multiple claims exist on the same edge, insert a **Reducer** node to combine them via random linear combination
   - Call `BasicBlock::prove()` which runs sumcheck(s) and produces claims on input edges
   - Route resulting claims to the appropriate input edges

3. **Lookup proofs**: After the backward pass:
   - `prove_range()`: Range check proofs using bit decomposition
   - `prove_two_pow()`: Two-pow lookup proofs using sparse selection polynomials

4. **Witness freeing**: After the backward pass, free non-essential witness data to reclaim memory:
   - Keep: committed edges (needed for openings), range aux outputs (bit polys), two_pow inputs (selection polys)
   - Free: all other edges (`w.data = None; w.data_int = None`)

5. **Opening reducers**: For each committed edge with K > 1 claims, run a Reducer sumcheck to combine them into 1 claim. (See Chapter 11.)

6. **Opening proofs**: Generate one Basefold opening proof per committed edge. (See Chapter 11.)

### Claim Propagation

Claims flow backward through the DAG:

```
Output claim on Y  -->  Node prove()  -->  Input claims on X, W
```

Each `BasicBlock::prove()` receives:
- `witnesses`: References to input/output witness data
- `edge_ids`: The edge IDs for constructing claims
- `out_claims`: Claims on the output edges
- `transcript`: For Fiat-Shamir challenges

And returns:
- `Vec<SumcheckProof>`: One or more sumcheck proofs
- `Vec<Claim>`: Claims on input edges (to be propagated further)

### Reducer (`src/basicblock/reducer.rs`)

When an edge has multiple consumers, each generating a claim at a different point, the Reducer combines them:

**Claim**: Given claims `(r_i, v_i)` for `i = 0..K`, prove `sum_i alpha^i * eq(r_i, x) * f(x) = sum_i alpha^i * v_i`.

**Implementation**: Single sumcheck with `eq_combined = sum_i alpha^i * eq(r_i, x)` and the polynomial `f(x)`.

Uses GPU path when `n > GPU_SUMCHECK_THRESHOLD`, CPU path otherwise.

---

## 9. BasicBlocks Reference

BasicBlocks are grouped by prover technique. Each category header describes the common proof mechanism shared by all blocks in the group.

---

### Category A: Zero-Sumcheck (Claim Transform Only)

**Common technique**: No sumcheck is needed. The prover directly transforms the output claim into input claims by extracting or adjusting evaluation points. The verifier checks a simple algebraic relation between claimed evaluations. These are the cheapest operations — O(1) verifier work, no round messages.

#### A.1 Add / Sub (`src/basicblock/add.rs`)

**Operation**: Element-wise addition/subtraction with NumPy-style broadcasting.

- **Run**: Broadcast indexing via stride tables. For each output flat index, decomposes into per-dimension indices (little-endian: dim 0 has stride 1), then maps to each input's flat index using `broadcast_strides()`. Broadcast dimensions (input size 1) have stride 0, so all output positions along that dimension read the same input element.
  - Example: `A[4] + B[2,4] -> C[2,4]`. A broadcasts along dim 0. For flat index `idx`, dim0 = `idx % 2`, dim1 = `idx / 2`. A index = dim1 (only matched dim). B index = dim0 + dim1*2 (both dims).
- **Prove**: Uses `matched_axes()` to identify which output dimensions each input maps to, then `extract_broadcast_point()` extracts the correct bit ranges from the output claim's evaluation point. Iterates through ALL output dimensions tracking bit offsets, only including bits for matched (non-broadcast) dimensions.
  - Example: `A[4] + B[2,4] -> C[2,4]`. Output point has 3 variables: 1 bit (dim 0) + 2 bits (dim 1). A matches dim 1 only, so A's point = `point[1..3]` (skipping dim 0's bit). B matches both dims, so B's point = full `point[0..3]`.
  - Claim arithmetic: `a_eval + b_eval = c_eval` for Add, `a_eval - b_eval = c_eval` for Sub.
- **Verify**: Reconstructs the same broadcast point extraction, checks `a_eval +/- b_eval == c_eval` and that claimed points match the extracted points.
- **Edges**: 2 inputs (A, B), 1 output

#### A.2 ChangeShape (`src/basicblock/change_shape.rs`)

**Operation**: Reshape/view change without modifying data.

- **Prove**: Direct claim transform — the MLE data is identical, only the shape interpretation changes. When `output_n > input_n`, evaluation divided by `prod(1 - r_j)` for extra variables (they are fixed to 0 in the padding region).
- **Edges**: 1 input, 1 output

#### A.3 Concat (`src/basicblock/concat.rs`)

**Operation**: Equal-size channel concatenation. `A[C, spatial], B[C, spatial] -> Y[2C, spatial]`

- **Prove**: `Y(r) = (1 - r_c_top) * A(r_ab) + r_c_top * B(r_ab)` where `r_c_top` is the highest channel bit. The prover evaluates A and B at `r_ab` (the remaining point), then distributes the claim.
- **Builder**: `concat(a, b)`, `general_concat(inputs)`, `multi_concat(inputs)`

#### A.4 ChannelSlice (`src/basicblock/concat.rs`)

**Operation**: Extract a contiguous slice of channels. `X[C_in, spatial] -> Y[C_out, spatial]` starting at `channel_start`.

- **Prove**: Extends the evaluation point with selector bits encoding `channel_start / C_out` in the upper channel bit positions. The output MLE is a slice of the input MLE.
- **Builder**: `channel_slice(x, channel_start, channels_out)`

---

### Category B: H-Polynomial Sumcheck (Embedding/Subsampling)

**Common technique**: A sumcheck over the *input* domain proving that the output MLE evaluation equals a sum of input values weighted by an "H-polynomial" that encodes the geometric embedding. The H-polynomial is `H(x) = eq(r_output, embed(x))` where `embed` maps input coordinates to output coordinates. This is the standard technique for operations where the output is a sparse embedding of the input (padding, subsampling).

#### B.1 ZeroPad (`src/basicblock/pad.rs`)

**Operation**: Symmetric zero-padding. `X[C, H, W] -> Y[C, H+2p, W+2p]`

- **H-polynomial**: `H[c,h,w] = eq(r_Y, (c, h+pad, w+pad))` evaluated over input dimensions
- **Sumcheck**: `sum_x X(x) * H(x) = Y(r_Y)` over the input domain
- **Builder**: `pad(x, pad_h, pad_w)`

#### B.2 ZeroPadAsym (`src/basicblock/pad.rs`)

**Operation**: Asymmetric padding with different top/bottom/left/right amounts.

- Same H-polynomial sumcheck with per-side offsets: `embed(c, h, w) = (c, h + pad_top, w + pad_left)`
- **Builder**: `pad_asym(x, pad_top, pad_bottom, pad_left, pad_right)`

#### B.3 ZeroPad3D (`src/basicblock/pad.rs`)

**Operation**: Symmetric 3D padding for `[C, D, H, W]` tensors.

- Same H-polynomial sumcheck extended to 4D
- **Builder**: `pad3d(x, pad_d, pad_h, pad_w)`

#### B.4 SubSample2D (`src/basicblock/subsample.rs`)

**Operation**: Spatial subsampling (stride > 1 extraction). `X[C, H, W] -> Y[C, H/s, W/s]`

- **H-polynomial**: `H[c,ho,wo] = eq(r_X, (c, ho*stride + offset_h, wo*stride + offset_w))` over the *output* domain
- Uses the same sumcheck structure as ZeroPad but in the reverse direction (output is smaller than input)
- Original carry-chain claim transform was incorrect for `offset >= stride` due to non-linear carry propagation; fixed to H-poly sumcheck
- **Builder**: `subsample2d_sized(x, stride_h, stride_w, out_h, out_w)`

#### B.5 Replicate2x2 (`src/basicblock/maxpool.rs`)

**Operation**: Proves that `Y_rep[C, H, W]` correctly replicates `Y[C, H/2, W/2]` on the downsampled grid.

- **Sumcheck**: Verifies the replication constraint — each 2x2 block in `Y_rep` contains the same value as the corresponding entry in `Y`

---

### Category C: Cascaded Convolution Sumchecks (Alpha-Power Trick)

Convolution is the most expensive and mathematically involved operation to prove. A 2D convolution computes:

```
Y[d, ho, wo] = Σ_c Σ_kh Σ_kw X[c, ho*stride+kh, wo*stride+kw] · W[d, c, kh, kw]
```

The challenge is that the kernel-position relationship (`ho*stride+kh`) is **non-algebraic** — it couples the output position with the kernel offset in a way that MLE sumchecks can't directly handle. The alpha-power trick solves this.

#### The Alpha-Power Trick

The key insight: `alpha^{i+j} = alpha^i · alpha^j`. If we weight each position by a power of a random `alpha`, we can **separate** the spatial coupling:

```
Σ_m alpha^m · Y_conv[m] = (Σ_i alpha^i · X_rev[i]) · (Σ_j alpha^j · W_flat[j])
```

where `Y_conv[m]` is the 1D convolution at position `m`, `X_rev` is the time-reversed input, and `W_flat` is the flattened kernel. The product on the right separates input from weight, enabling independent sumchecks on each factor.

#### 1D Flattening

All N-D convolutions are first reduced to 1D. For Conv2D with input `X[C, H, W]` and kernel `(kh, kw)`:

- **Flatten spatial**: position `(h, w)` maps to 1D index `i = h * W_pad + w`
- **Flatten kernel**: position `(kh, kw)` maps to 1D index `j = kh * W_pad + kw`
- **FlattenKernel BasicBlock** performs this scatter: `W[d, c, kh, kw] -> W_flat[d, c, j]`
- **Reverse indexing**: `X_rev[c, i] = X[c, S_in - 1 - i]` aligns the convolution sum

After flattening, 1D convolution is: `Y_conv[m] = Σ_c Σ_j X_rev[c, m+j] · W_flat[c, j]`, with `alpha^{m+j} = alpha^m · alpha^j`.

#### The 4 Cascaded Sumchecks

Starting from a claim `Y(r) = v` at point `r = (r_spatial, r_d)`:

**Sumcheck 1 — Output spatial reduction** (reduces `h_out, w_out` to a point):

```
Σ_k eq(r_spatial, k) · YP[k] = v
```

where `YP[k] = Σ_d Y[d, k] · eq(r_d, d)` is Y partially evaluated over channels. This is a standard Lagrange-basis sumcheck that "opens" the output spatial dimensions. After this sumcheck, the verifier has `YP` evaluated at a random spatial point, which equals `s_alpha_conv` — the alpha-weighted convolution sum the prover sends via the transcript.

**Sumcheck 2 — Channel F*G factorization** (reduces `c_in` to a point):

Define per-channel polynomials:
- `F[c] = Σ_i X_rev[c, i] · alpha^i` (alpha-weighted reversed input)
- `G[c] = Σ_j WP[c, j] · alpha^j` (alpha-weighted weight, with WP partially evaluated over `r_d`)

Then the alpha-power trick gives:

```
Σ_c F[c] · G[c] = s_alpha_conv
```

This degree-2 sumcheck reduces the channel sum to a single point `r_c`, yielding evals `F(r_c)` and `G(r_c)`.

**Sumcheck 3 — F-to-X reduction** (reduces spatial input dimension to a point):

The F polynomial is still an alpha-weighted sum over spatial positions. Define `XP[i] = Σ_c X_rev[c, i] · eq(r_c, c)` (X partially evaluated over channels). Then:

```
Σ_i alpha^i · XP[i] = F(r_c)
```

This sumcheck uses the **alpha-table MLE** as one polynomial. After reducing to random point `r_i`, we get a claim on `X_rev(r_c, r_i) = X(r_c, 1 - r_i)` (the reversal flips the point).

**Alpha-table MLE**: The polynomial `[1, alpha, alpha^2, ..., alpha^{2^n-1}]` has MLE `alpha_table(r) = Π_j(1 + r_j · (alpha^{2^j} - 1))`, computable in O(n) by the verifier.

**Sumcheck 4 — G-to-W reduction** (reduces kernel dimension to a point):

Analogous to sumcheck 3. Define `WPP[j] = Σ_c eq(r_c, c) · WP[c, j]`. Then:

```
Σ_j alpha^j · WPP[j] = G(r_c)
```

After reducing to random point `r_j`, we get a claim on `W_flat(r_d, r_c, r_j)`.

**Cross-check**: The verifier confirms `F(r_c) · G(r_c) = final_eval(sumcheck 2)`, ensuring the channel factorization is consistent.

**Final output**: Claims on X (from sumcheck 3) and W_flat (from sumcheck 4), which propagate backward through the DAG. The Conv output is committed via PCS (`self_claim_edges`) so sumcheck 1's claim is resolved by an opening proof.

#### C.1 Conv1D (`src/basicblock/conv.rs`)

**Operation**: `X[C_in, L_in], W[C_out, C_in, K] -> Y[C_out, L_out]`

- Same 4-sumcheck structure. No 2D-to-1D flattening needed.
- **Builder**: `conv1d_strided(x, w, kernel_size, stride)`

#### C.2 Conv2D (`src/basicblock/conv.rs`)

**Operation**: `X[C_in, H, W], W_flat[C_out, C_in, S_kernel] -> Y[C_out, H_out, W_out]`

- **Parameters**: `c_in, c_out, kh, kw, input_h, input_w, conv_stride_h/w, dilation_h/w`
- **1x1 kernel fix**: When `l_kernel = 0`, sumcheck 4 has 0 rounds; use `final_eval` as `inferred_sum`
- **Builder**: `conv2d(x, w, kernel_size)`, `conv2d_strided(...)`, `conv2d_dilated(...)`

#### C.3 Conv3D (`src/basicblock/conv.rs`)

**Operation**: `X[C_in, D, H, W], W_flat[C_out, C_in, S_kernel] -> Y[C_out, D_out, H_out, W_out]`

- Same structure. 3D flattening: `i = d * H_pad * W_pad + h * W_pad + w`.
- **Builder**: `conv3d_strided(x, w, kernel_size, stride)`

#### C.4 ConvTranspose1D/2D/3D (`src/basicblock/conv.rs`)

**Operation**: Transposed convolution (deconvolution).

- Prove/verify structure mirrors the corresponding Conv, with transposed indexing
- Same 1x1 kernel fix applies

#### C.5 DepthwiseConv2D (`src/basicblock/conv.rs`)

**Operation**: Depthwise separable 2D convolution. Each channel convolved independently.

- **Key difference**: Sumcheck 2 is **degree-3** (`eq_C(c) · F[c] · G[c]`) because input and output channels must be the same (no cross-channel summation). The extra `eq_C` polynomial constrains channel identity.
- Cross-check becomes: `eq_C(r_c) · F(r_c) · G(r_c) = final_eval(sumcheck 2)`.
- **Builder**: `depthwise_conv2d_strided(x, w, kernel_size, stride)`

#### C.6 FlattenKernel (`src/basicblock/conv.rs`)

**Operation**: Scatter `W[C_out, C_in, kH, kW] -> W_flat[C_out, C_in, S_kernel]` where `j = kh * dilation_h * W_pad + kw * dilation_w`.

- **Prove**: Small sumcheck (l_kh + l_kw rounds): `W_flat(r_d, r_c, r_j) = Σ_{kh,kw} W(r_d, r_c, kh, kw) · eq(r_j, kh*S_w + kw)`
- **Weight zero-padding**: Conv kernels MUST have zeros at padding positions (kw >= actual_kw, kh >= actual_kh). FlattenKernel's proof assumes this.
- Always paired with a Conv node in the builder

---

### Category D: Linear Sumcheck (Einsum / Tensor Contraction)

**Common technique**: A single linear sumcheck over summation dimensions. The prover constructs per-input polynomials via permutation and eq-table weighting, then runs a GPU or CPU sumcheck. This is the workhorse for matrix multiplication, attention, and general tensor contractions.

#### D.1 Einsum (`src/basicblock/einsum.rs`)

**Operation**: Einstein summation notation (matrix multiply, tensor contraction, etc.).

- **Notation**: `"ab,bc->ac"` for matmul, `"abc->abc"` for identity, etc.
- **Run**: `einsum_compute()` with little-endian indexing (first dim has stride 1). Uses `% padded_shape[i]` for dimension broadcasting.
- **Prove**: Single linear sumcheck over the summation dimensions. Constructs per-input polynomials with permutation + eq tables.
  - GPU path: Fused permute + partial_eval kernel when `n > ZK_GPU_FUSED_THRESHOLD`
  - CPU path: LUT-based permutation + CPU/GPU partial eval
- **Key fields**: `term_strings`, `out_string`, `permute_vecs`, `summation_round`
- **Precomputation**: `compute_permute_vecs()` + `compute_einsum_challenges()` refactored from `einsum_helper`

---

### Category E: Range & Lookup Auxiliaries

**Common technique**: These blocks produce auxiliary witness polynomials (bit decompositions or selection polynomials) that are later verified by the DAG-level lookup protocols (`prove_range`, `prove_two_pow`). The blocks themselves return empty proofs from `prove()` — their correctness is guaranteed by the lookup arguments in Chapter 10.

#### E.1 ScaleDown / ScaleUp (`src/basicblock/scale.rs`)

**Operation**: Fixed-point scaling. ScaleDown divides by `2^shift`, ScaleUp multiplies.

- **Run**: `out[i] = floor(in[i] / 2^shift)` (ScaleDown) or `out[i] = in[i] * 2^shift` (ScaleUp)
- **Auxiliary**: Produces bit decomposition polynomial `B(x,y)` with `BIT_TABLE_VARS=5` (32 bit positions) for range proofs
- **Builder**: `scale_down(x, shift)`, `scale_up(x, shift)`

#### E.2 NonNegative (`src/basicblock/scale.rs`)

**Operation**: Asserts all values are non-negative (used after ReLU, max operations).

- **Run**: `out[i] = max(0, in[i])` (clamp negatives to zero)
- **Auxiliary**: Produces bit decomposition polynomial for proving non-negativity
- **Builder**: `relu(x)` (wraps NonNegative)

#### E.3 ExpHelper (`src/basicblock/scale.rs`)

**Operation**: Exponent extraction for two-pow lookups.

- **Auxiliary**: Produces `SelectionPolynomial` mapping input indices to powers-of-two table
- Correctness proven by `prove_two_pow` (not by range checks)

---

### Category F: Advice Operations

**Common technique**: These operations produce auxiliary witness values that the prover claims are correct, returning empty `(vec![], vec![])` from `prove()`. Soundness comes from **downstream constraints**: the DAG wires these outputs into Sub + NonNegative chains, Einsum identity checks, or other verified blocks that implicitly constrain the advice values. If the advice is wrong, those downstream checks will fail.

#### F.1 ReLUHelper + ProductZeroCheck (`src/basicblock/relu.rs`)

**Operation**: ReLU decomposition. `neg = max(0, -x)` (advice), `y = x + neg` (Add).

- **Soundness**: Three constraints ensure `y = max(0, x)`:
  1. **NonNeg(y)**: `y ≥ 0`
  2. **NonNeg(neg)**: `neg ≥ 0`
  3. **ProductZeroCheck(neg, y)**: `neg · y = 0` pointwise (complementary slackness)
- **ProductZeroCheck**: Degree-3 sumcheck proving `Σ_x eq(r, x) · A(x) · B(x) = 0`. Uses CpuLinearSumcheckProverExt2 with 3 polynomials. The certificate output (all zeros) is committed and opened via PCS, confirming the claimed evaluation.
- **Builder**: `relu(x)`

#### F.2 MaxPoolHelper (`src/basicblock/maxpool.rs`)

**Operation**: 2x2 max pooling. Produces `Y[C, H/2, W/2]`.

- **Soundness**: Dominance proven: `Y ≥ X` at all pool positions via Replicate2x2 + Sub + NonNeg. This proves Y upper-bounds all inputs but does not prove achievability (that Y equals some actual input). True achievability requires a lookup/selection argument — future work.
- **Builder**: `maxpool2d(x, 2, 2)`

#### F.3 GeneralMaxPoolHelper (`src/basicblock/maxpool.rs`)

**Operation**: Arbitrary kernel/stride max pooling. Produces `Y[C, H_out, W_out]`.

- **Soundness**: Dominance via SubSample2D + Sub + NonNeg per kernel position. Each kernel position is extracted by SubSample2D, then `Y - X_kh_kw ≥ 0` is proven.
- **Builder**: `maxpool_general(x, pool_h, pool_w, stride_h, stride_w)`

#### F.4 InstanceNormHelper (`src/basicblock/instancenorm.rs`)

**Operation**: Computes per-channel `scale[C]` and `offset[C]` for instance normalization. Output is a single packed polynomial `[2, C]` (group 0 = scale, group 1 = offset), unpacked via ChannelSlice. The actual computation `Y = scale * X + offset` is decomposed into proven Einsum + Add nodes.

- **Soundness**: The prover can only choose scale/offset freely; the resulting Y is deterministically constrained by `Y = Einsum("a,abcd->abcd", scale, X) + offset`, both fully proven. Same trust model as DivConst.
- **Packed output**: Single output avoids the multi-output reducer issue (reducer expects all claims on the same polynomial). ChannelSlice is a zero-cost claim transform.
- **Builder**: `instancenorm3d(x, gamma, beta, eps)`

#### F.5 RMSReciprocal, DivConst, SoftmaxConst, SigmoidConst (`src/basicblock/llama.rs`)

- **RMSReciprocal**: Computes `1/RMS(x)` for RMSNorm
- **DivConst**: Divides each element by a constant
- **SoftmaxConst**: Piecewise linear softmax approximation
- **SigmoidConst**: Piecewise linear sigmoid approximation (same cancellation technique as SoftmaxConst)
- **Downstream**: Verified by Einsum product checks (`x * (1/RMS(x)) = normalized`)

#### F.6 PillarMaxPool (`src/basicblock/pointpillar.rs`)

**Operation**: Max-pools pillars along the points dimension. Produces `Y[N, D]`.

- **Soundness**: Dominance via SubSample2D time-slice extraction + Sub + NonNeg for each time step. For each `t` in `0..max_points`, `Y - X[:, t, :] ≥ 0` is proven.
- **Builder**: `pillar_maxpool(x, n_pillars, max_points, features)`

#### F.7 ScatterToBEV (`src/basicblock/pointpillar.rs`)

**Operation**: Scatters pillar features to a bird's-eye-view grid using coordinate indices.

- **SOUNDNESS GAP**: This is a coordinate-indexed scatter (non-algebraic operation). A full proof requires a lookup/permutation argument to verify the mapping from pillars to grid cells. Currently unsound — the prover's output is trusted. Future work: lookup-based scatter verification.

---

## 10. Lookup Protocols

Lookup protocols prove that auxiliary values lie in prescribed tables. ZK-Torch uses two lookup types:

### 10.1 Range Check (Bit Decomposition)

**Purpose**: Prove that ScaleDown/ScaleUp/NonNegative auxiliary values are valid (within range).

**Old approach**: `SelectionPolynomial` S(x,y) as `SparseMLPoly` with n+t vars (t=10-20). Very expensive.

**New approach (bit decomposition)**: Dense bit polynomial `B(x,y)` with `y in {0,1}^5` (32 bit positions). `BIT_TABLE_VARS = 5`.

**B(x,y)** = bit y of the value at position x. The MLE index is `x + y * 2^n`. The value is recovered as `value(x) = sum_y B(x,y) * 2^y`.

**prove_range** (`src/dag/mod.rs`):

1. Collect all bit auxiliary polynomials from range-checked nodes
2. Random combine with beta challenges: `combined_aux = sum_i beta^i * B_i`
3. Partial evaluate `B` at each claim point to get 32-element vectors
4. Combine with beta weights
5. Single sumcheck over 5 variables with table `T(y) = 2^y`
6. Final claim relates to the committed bit polynomial

**Size reduction**: ScaleDown n+10 -> n+5 (32x smaller), NonNegative n+20 -> n+5 (32768x smaller).

**Impact**: GPT-2 12L auxiliary commits: 2439M -> 18.2M elements (~130x reduction).

### 10.2 Two-Pow Lookup

**Purpose**: Prove that ExpHelper selection polynomials map to the powers-of-two table `[2^15, 2^14, ..., 2^0]`.

**prove_two_pow** (`src/dag/mod.rs`):

1. Collect all `SelectionPolynomial` entries from ExpHelper nodes
2. Convert to `SparseMLPoly` with `n_input + n_table` variables
3. Random-combine with beta challenges
4. Table sumcheck: `sum_y combined_aux(y) * (table(y) + alpha) = sum_i bg_i * (mc_i + alpha * sa_i)`
5. Sparse evaluation optimization: `SparseMLPoly::evaluate_at_point_ext2` uses O(k*n) per-entry eq computation, not O(2^n) full table

**Critical**: The expected sum must include `alpha * sum_aux` (not just `alpha`), since `sum_aux != 1` when not all inputs are in range.

---

## 11. Opening Proofs

Opening proofs bind sumcheck-derived claims to committed polynomials via Basefold.

### Opening Reducers

After the backward pass and lookup proofs, a committed edge may have K > 1 claims at K different evaluation points (from multiple consumers, the backward-pass reducer, or lookup proofs). Instead of generating K separate opening proofs per edge, a Reducer sumcheck combines all K claims into a single claim at one point.

**Why this works**: The Reducer proves `Σ_i α^i · eq(r_i, x) · f(x) = Σ_i α^i · v_i` where `(r_i, v_i)` are the K original claims. Since `Σ_x eq(r, x) · f(x) = f(r)`, the sumcheck equivalently proves `Σ_i α^i · f(r_i) = Σ_i α^i · v_i`. By Schwartz-Zippel over the random choice of α, this proves `f(r_i) = v_i` for all i simultaneously. The sumcheck reduces to a claim at a new random point u with `f(u) = w`, which is the only point that needs a PCS opening proof.

**Result**: At most 1 opening proof per committed edge, regardless of how many claims accumulated.

The opening reducer proof is stored in `EdgeProof.opening_reducer`. When present, `claims[0..K-1]` are the original claims and `claims[K-1]` is the combined claim from the reducer. Only the combined claim gets a Basefold opening proof.

### Task Collection

After the opening reducers, exactly one opening task is collected per committed edge:
```
For each committed edge with non-empty claims:
    Take the last non-empty claim (combined if reducer ran, or the single original)
    → 1 task per edge
```

### Dual-Pool Architecture

Opening proofs are split between CPU and GPU based on polynomial size:

| Path | Condition | Implementation |
|------|-----------|---------------|
| CPU | `n <= CPU_OPEN_THRESHOLD` (default 14) | `cpu_full_open_ext2()` in `src/commit/cpu_basefold.rs` |
| GPU | `n > CPU_OPEN_THRESHOLD` | `BasefoldCommitment::open_ext2()` on GPU |

**Concurrent execution**: CPU and GPU tasks run simultaneously via `std::thread::scope`:
- CPU pool: `num_cpus - gpu_pool_size` threads
- GPU pool: `num_devices * 12` threads (12 per GPU for best throughput)
- Per-thread CUDA streams (`--default-stream per-thread`) enable concurrent kernel execution

### GPU Opening (`goldilocks-cuda-rs/src/basefold.rs`)

`BasefoldCommitment::open_ext2()`:

1. Compute eq table on GPU: `ext2_eq_dp_all_device(point)`
2. For each round:
   - Sumcheck oracle: `sumcheck_product_and_reduce_ext2(eq, bh_evals, pair_count)`
   - Fold eq and bh_evals with challenge
   - Record oracle and folded Merkle root
3. Generate query proofs with Merkle authentication paths
4. Pre-allocated double buffers: `eq`/`f` pairs with `std::mem::swap` to eliminate per-round allocation

### CPU Opening (`src/commit/cpu_basefold.rs`)

`cpu_full_open_ext2()`:

1. Download codeword and bh_evals from GPU commitment
2. Normalize field values (fix non-canonical representation from GPU)
3. Inner-product sumcheck on CPU
4. Build Merkle tree on CPU for query proofs
5. Full `BasefoldProofExt2` with Merkle auth paths

### Re-Upload Mode

For partition-aware GPU placement when GPU memory is insufficient:

1. During commit: `to_host_cache()` downloads codeword/bh_evals
2. During opening: Group tasks by edge, re-upload each edge to GPU, open all its tasks, drop commitment (freeing GPU memory for next edge)

### Stale CUDA Error Handling

After the sumcheck prover phase, `cudaErrorMemoryAllocation` (code 2) may be left pending. Must call `cudaGetLastError()` (via `goldilocks_cuda::get_last_error()`) to clear before GPU opening proofs. `cudaDeviceSynchronize()` alone does NOT clear stale errors.

### Constants

| Constant | Env Var | Default | Description |
|----------|---------|---------|-------------|
| CPU open threshold | `CPU_OPEN_THRESHOLD` | 14 | n <= this: CPU path |

---

## 12. Partition-Aware Parallel Proving

### Motivation

Large models benefit from multi-GPU parallelism. The DAG is partitioned into independent segments that can be proven concurrently on different GPUs.

### Partitioning (`src/dag/partition.rs`)

#### PartitionDesc

```rust
pub struct PartitionDesc {
    pub node_ids: Vec<NodeId>,
    pub boundary_input_edges: Vec<EdgeId>,   // Edges entering this partition
    pub boundary_output_edges: Vec<EdgeId>,  // Edges leaving this partition
}
```

#### `set_partition_boundaries(num_partitions)`

Selects evenly-spaced layer boundaries from `topo_levels`:
```
boundary_level_indices = [L/P, 2L/P, ..., (P-1)L/P]
```
All edges crossing these boundaries are added to `dag.boundary_edges` and force-committed.

#### `partition_dag(dag, boundary_edges)`

Assigns each node to a partition based on its topological position relative to boundary levels. Returns `Vec<PartitionDesc>`.

### Parallel Proving (`dag.prove_parallel()`)

```rust
pub fn prove_parallel(
    &self, key, witnesses, commitments, gpu_store, table, transcript, partitions, timing
) -> ParallelProof
```

**Algorithm:**

1. **Output claims**: Generate random evaluation points for output edges
2. **Boundary claim routing**: Propagate claims to identify which partition each claim belongs to (route by producer node's partition)
3. **Transcript forking**: `transcript.fork(k)` for each partition k (absorbs partition_id for domain separation)
4. **Parallel partition proving**: `par_iter` over partitions, each assigned to GPU `k % num_devices`:
   ```rust
   set_device(k as i32 % num_devices as i32);
   prove_partition(partition_k, forked_transcript_k)
   ```
5. **Merge claims**: Collect all claims from partition boundaries
6. **Lookup proofs**: Range and two-pow proofs (sequential, after all partitions)
7. **Opening proofs**: Multi-GPU parallel opening (see Chapter 11)

### ParallelProof

```rust
pub struct ParallelProof {
    pub boundary_evals: Vec<(EdgeId, GoldilocksExt2, Vec<GoldilocksExt2>)>,
    pub partition_proofs: Vec<PartitionProof>,
    pub edge_proofs: Vec<EdgeProof>,
    pub range_proof: RangeProof,
    pub two_pow_proof: TwoPowProof,
}
```

### Multi-GPU Support

- **FFI**: `goldilocks_cuda::set_device(i32)`, `goldilocks_cuda::get_device()`
- **Per-device tables**: `GpuCommitmentStore.per_device_tables` pre-cloned at init
- **Stale error clearing**: All devices cleared before GPU opening phase
- **GPU placement**: Each partition assigned to `k % num_devices`

---

## 13. Model Catalog & Running Guide

### Prerequisites

All models require:
- CUDA-capable GPUs (tested on 4x A100-80GB)
- Rust toolchain with `cargo`
- The `goldilocks-cuda-rs` crate compiled with CUDA support

### Build

```bash
cd /scratch/bjchen4/goldilocks/zk-torch-3
cargo build --release
```

### Environment Variables (Common)

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | all |
| `NUM_PARTITIONS` | Number of partitions for parallel proving | 1 |
| `ZK_GPU_SUMCHECK_THRESHOLD` | GPU sumcheck threshold | 14 |
| `ZK_GPU_PARTIAL_EVAL_THRESHOLD` | GPU partial eval threshold | 16 |
| `ZK_GPU_FUSED_THRESHOLD` | GPU fused permute+peval threshold | 16 |
| `CPU_OPEN_THRESHOLD` | CPU opening proof threshold | 14 |

---

### 13.1 GPT-2

**Architecture**: 12-layer GPT-2 transformer. Each layer: LayerNorm + Multi-Head Self-Attention (12 heads) + LayerNorm + MLP (768 -> 3072 -> 768). Embedding dim 768, sequence length 128.

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 12 | Number of transformer layers |

**Full-size command (4x A100):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_LAYERS=12 \
  cargo run --release --bin gpt2
```

---

### 13.2 BERT

**Architecture**: 24-layer BERT-Large encoder. Each layer: LayerNorm + Multi-Head Self-Attention (16 heads) + LayerNorm + MLP (1024 -> 4096 -> 1024). Embedding dim 1024, sequence length 128.

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 24 | Number of transformer layers |

**Full-size command (4x A100):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_LAYERS=24 \
  cargo run --release --bin bert
```

---

### 13.3 GPT-J

**Architecture**: 28-layer GPT-J transformer. Each layer: LayerNorm + Multi-Head Self-Attention (16 heads, rotary PE) + MLP (4096 -> 16384 -> 4096). Embedding dim 4096, sequence length 128.

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 28 | Number of transformer layers |

**Full-size command (4x A100):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_LAYERS=28 \
  cargo run --release --bin gptj
```

---

### 13.4 LLaMA 2

**Architecture**: 32-layer LLaMA-2 transformer. Each layer: RMSNorm + Multi-Head Attention (32 heads) + RMSNorm + SwiGLU MLP (4096 -> 11008 -> 4096). Embedding dim 4096, sequence length 128.

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 32 | Number of transformer layers |

**Full-size command (4x A100):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_LAYERS=32 \
  cargo run --release --bin llama
```

---

### 13.4b LLaMA 3.1 8B

**Architecture**: LLaMA 3.1 8B with Grouped-Query Attention (GQA). 32 transformer layers. Each layer: RMSNorm + GQA (32 Q heads, 8 KV heads, head_dim=128) + RMSNorm + SwiGLU MLP (4096 -> 14336 -> 4096). Uses LLaMA 3 RoPE with adjusted frequencies. Vocab size 128256.

**Inputs**: `[1, SEQ_LEN, 4096]` token embeddings. Default `SEQ_LEN=1` (single-token inference). Note: seq_len > 1 computes non-causal attention (no causal mask implemented).

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 1 | Number of transformer layers (full model: 32) |
| `SEQ_LEN` | 1 | Sequence length |
| `NUM_PARTITIONS` | 1 | Number of partitions |

**Full-size command (4x A100):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_LAYERS=32 \
  cargo run --release --bin llama3
```

---

### 13.5 VGG-16

**Architecture**: VGG-16 convolutional network. 13 conv layers (3x3 kernels) organized in 5 blocks with 2x2 max pooling between blocks. Channel progression: 64 -> 128 -> 256 -> 512 -> 512. Input 224x224x3.

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 16 | Number of layers (all 13 convs at default) |

**Full-size command (4x A100):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_LAYERS=16 \
  cargo run --release --bin vgg
```

---

### 13.6 ResNet-50

**Architecture**: ResNet-50 with 53 convolution layers. 4 stages of bottleneck blocks (3-4-6-3 blocks). Each bottleneck: 1x1 conv (reduce) + 3x3 conv + 1x1 conv (expand) + skip connection. Channel progression: 256 -> 512 -> 1024 -> 2048. Input 224x224x3.

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 1 | Number of conv layers (full model: 53) |
| `NUM_PARTITIONS` | 1 | Number of partitions |

**Full-size command (4x A100):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_LAYERS=53 \
  cargo run --release --bin resnet
```

---

### 13.7 3D UNet

**Architecture**: 3D UNet with encoder-decoder structure and skip connections. 6 encoder levels (Conv3D + InstanceNorm3D + ReLU, channel doubling) + 5 decoder levels (upsample + concat + Conv3D). Input: volumetric 3D data.

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 6 | Number of encoder levels |
| `NUM_PARTITIONS` | 1 | Number of partitions |
| `INPUT_D` | 32 | Input depth |
| `INPUT_H` | 32 | Input height |
| `INPUT_W` | 32 | Input width |

**Full-size command (4x A100):**
```bash
# 128³ input (full size, bottleneck 4×4×4)
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_LAYERS=6 \
  INPUT_D=128 INPUT_H=128 INPUT_W=128 \
  cargo run --release --bin unet3d

# 32³ input (fast test, max 5 levels to avoid 1×1×1 spatial bottleneck)
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_LAYERS=5 \
  INPUT_D=32 INPUT_H=32 INPUT_W=32 \
  cargo run --release --bin unet3d
```

---

### 13.8 YOLOv11

**Architecture**: YOLOv11n object detection. 8 stages: backbone (CBS blocks, C3k2 blocks with depthwise convolutions, SPPF) + neck (upsampling, C3k2 fusion) + 3 detection heads. Uses depthwise separable convolutions, SiLU activation, and multi-scale feature fusion.

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_STAGES` | 8 | Number of stages (backbone + neck + heads) |
| `NUM_PARTITIONS` | 1 | Number of partitions |
| `INPUT_SIZE` | 640 | Input image size (square) |

**Full-size command (4x A100):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 NUM_STAGES=8 INPUT_SIZE=640 \
  cargo run --release --bin yolo
```

---

### 13.9 Whisper

**Architecture**: Whisper speech recognition model (OpenAI). Audio encoder: 2 Conv1D layers (stride 1, stride 2) + sinusoidal positional embedding + N transformer encoder layers. Text decoder: learned positional embedding + N transformer decoder layers with cross-attention to encoder outputs. MLP expansion factor: 4x.

**Inputs**:
- Encoder: `[N_MELS, 2 * N_AUDIO_CTX]` mel spectrogram (80 bins × 3000 time frames for 30s audio)
- Decoder: `[1, N_TEXT_CTX, N_STATE]` token embeddings (1 × 448 tokens × hidden dim)

**OpenAI Whisper model variants:**

| Variant | Enc Layers | Dec Layers | N_STATE | N_HEAD | Parameters |
|---------|-----------|-----------|---------|--------|------------|
| Tiny | 4 | 4 | 384 | 6 | 39M |
| Base | 6 | 6 | 512 | 8 | 74M |
| Small | 12 | 12 | 768 | 12 | 244M |
| Medium | 24 | 24 | 1024 | 16 | 769M |
| Large | 32 | 32 | 1280 | 20 | 1550M |

All variants share: `N_MELS=80`, `N_AUDIO_CTX=1500`, `N_TEXT_CTX=448`.

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_ENC_LAYERS` | 4 | Number of encoder transformer layers |
| `NUM_DEC_LAYERS` | 4 | Number of decoder transformer layers |
| `N_STATE` | 384 | Model hidden dimension |
| `N_HEAD` | 6 | Number of attention heads |
| `N_MELS` | 80 | Number of mel spectrogram channels |
| `N_AUDIO_CTX` | 1500 | Audio context length (frames) |
| `N_TEXT_CTX` | 448 | Text context length (tokens) |
| `NUM_PARTITIONS` | 1 | Number of partitions |

**Commands (4x A100):**
```bash
# Whisper-Tiny (default)
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 \
  cargo run --release --bin whisper

# Whisper-Base
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 \
  NUM_ENC_LAYERS=6 NUM_DEC_LAYERS=6 N_STATE=512 N_HEAD=8 \
  cargo run --release --bin whisper

# Whisper-Small
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 \
  NUM_ENC_LAYERS=12 NUM_DEC_LAYERS=12 N_STATE=768 N_HEAD=12 \
  cargo run --release --bin whisper

# Whisper-Medium
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 \
  NUM_ENC_LAYERS=24 NUM_DEC_LAYERS=24 N_STATE=1024 N_HEAD=16 \
  cargo run --release --bin whisper

# Whisper-Large
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 \
  NUM_ENC_LAYERS=32 NUM_DEC_LAYERS=32 N_STATE=1280 N_HEAD=20 \
  cargo run --release --bin whisper
```

---

### 13.10 PointPainting (DeepLabV3+ + PointPillar)

**Architecture**: Two-stage autonomous driving pipeline.

**Stage 1 -- DeepLabV3+**: ResNet-101 backbone (output_stride=8) + ASPP module (5 parallel branches with dilations 1/12/24/36) + lightweight decoder. Produces per-pixel semantic segmentation.

**Stage 2 -- PointPillar**: Voxel Feature Encoding (VFE: linear 11->64 + pillar max pool) + BEV backbone (3 blocks x 6 convs each + 3 deblock ConvTranspose2d) + 3 detection heads (classification, bounding box, direction).

| Env Var | Default | Description |
|---------|---------|-------------|
| `STAGE` | `both` | `deeplabv3`, `pointpillar`, or `both` |
| `NUM_LAYERS` | 33 | Number of bottleneck conv layers (DeepLabV3+) |
| `NUM_PARTITIONS` | 1 | Number of partitions |
| `INPUT_H` | 512 | Image height (DeepLabV3+) |
| `INPUT_W` | 512 | Image width (DeepLabV3+) |
| `NY` | 496 | BEV grid Y dimension |
| `NX` | 432 | BEV grid X dimension |
| `N_PILLARS` | 12000 | Number of pillars |
| `MAX_POINTS` | 32 | Max points per pillar |
| `NUM_CLASSES` | 5 | Number of object classes |

**Full-size command (4x A100):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PARTITIONS=4 STAGE=both \
  cargo run --release --bin pointpainting
```

---

### Quick Reference: All Full-Size Commands

```bash
# Set common environment for multi-GPU proving
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_PARTITIONS=4

# Transformers
NUM_LAYERS=12 cargo run --release --bin gpt2
NUM_LAYERS=24 cargo run --release --bin bert
NUM_LAYERS=28 cargo run --release --bin gptj
NUM_LAYERS=32 cargo run --release --bin llama
NUM_LAYERS=32 cargo run --release --bin llama3   # LLaMA 3.1 8B (GQA)

# CNNs
NUM_LAYERS=16 cargo run --release --bin vgg
cargo run --release --bin resnet

# 3D / Volumetric
NUM_LAYERS=6 INPUT_D=32 INPUT_H=32 INPUT_W=32 cargo run --release --bin unet3d

# Detection
NUM_STAGES=8 INPUT_SIZE=640 cargo run --release --bin yolo

# Speech
NUM_ENC_LAYERS=4 NUM_DEC_LAYERS=4 cargo run --release --bin whisper

# Autonomous Driving
STAGE=both cargo run --release --bin pointpainting
```

**Hash function selection**: By default, Monolith hash is used (faster on both GPU and CPU). To use Poseidon2 instead:

```bash
cargo run --release --no-default-features --bin llama3
```

**Multi-GPU tips for optimal prover time**:

- Set `NUM_PARTITIONS` equal to the number of GPUs (e.g., 4 for 4× A100).
- `CUDA_VISIBLE_DEVICES` controls which GPUs are used. Partitions are assigned round-robin.
- Opening proofs are parallelized across all GPUs with `GPU_OPEN_THREADS_PER_DEVICE=12` threads per device (configurable via env var).
- Sumcheck proving runs one partition per GPU in parallel. Each partition gets its own CUDA stream via `--default-stream per-thread`.
- For models with many layers (LLaMA 32L, BERT 24L), partitioning gives near-linear multi-GPU scaling on the sumcheck phase.

---

### Performance Summary (4x A100-80GB, Monolith hash)

| Model | Nodes | Edges | Prove Time | Verify Time |
|-------|-------|-------|-----------|-------------|
| GPT-2 (12L) | ~150 | ~200 | ~5s | ~40ms |
| BERT (24L) | ~300 | ~400 | ~10s | ~80ms |
| VGG-16 | ~60 | ~100 | ~2.3s | ~3ms |
| ResNet-50 | 368 | ~500 | ~20s | ~10ms |
| 3D UNet (6L, 32^3) | 194 | ~300 | ~2.4s | ~35ms |
| YOLOv11 (640x640) | 2138 | ~3000 | ~69s | ~233ms |
| Whisper (4+4L) | 1699 | ~2400 | ~191s | ~169ms |
| LLaMA 3.1 8B (32L) | 4573 | 6636 | ~12s | ~240ms |
| PointPainting (both) | ~1000+ | varies | varies | varies |

*Times are approximate and depend on GPU utilization, memory pressure, and partition count.*

---

## 14. Prover Optimizations

This section catalogs every optimization in the prover pipeline, organized by the proving stage they affect. Each entry explains what the optimization does, where it lives in the code, what thresholds or configuration it uses, and what speedup it provides.

---

### 14.1 Forward Pass Optimizations

#### Level-Parallel DAG Execution

**Problem.** A naive topological-order forward pass executes nodes one at a time. In a 12-layer GPT-2, many nodes at the same "depth" are independent (e.g., Q/K/V projections within one layer).

**Solution.** During `compile()`, the DAG computes `topo_levels: Vec<Vec<NodeId>>` — a level-grouped topological sort where all nodes in a level are independent. `dag.run()` processes each level:

- Single-node levels: run sequentially (no rayon overhead).
- Multi-node levels: `level.par_iter().map(|nid| node.run(...))` via rayon, then collect results and write back.

**File:** `src/dag/mod.rs`, `Dag::run()` (lines 326-365).

**Speedup:** GPT-2 12L forward pass: 26.2s -> 1.5s (17x).

#### Parallelized `einsum_compute`

The inner loop of `einsum_compute` iterates over all output elements. For large tensor contractions (e.g., 4096x4096 matmul), this is millions of elements.

The loop uses `(0..out_size).into_par_iter()` with pre-computed `term_chars` and `padded_shapes` to avoid per-iteration allocation. Little-endian indexing: first dimension has stride 1.

**File:** `src/basicblock/einsum.rs`, `einsum_compute()`.

---

### 14.2 Commitment Optimizations

#### Multi-GPU Commitment

Commitment tasks are distributed across all available GPUs. Each edge is assigned to a device (either by partition affinity or round-robin), and Basefold commitments (BHC interpolation + RS encoding + Merkle tree) are built in parallel using rayon + `set_device()`.

**File:** `src/dag/mod.rs`, `Dag::commit()`.

#### Commitment Size Bounds

Two constants prevent pathological cases:

| Constant | Value | Reason |
|----------|-------|--------|
| `MIN_BASEFOLD_VARS` | 2 | GPU Basefold kernels fail on trivially small polynomials |
| `MAX_BASEFOLD_VARS` | 22 | GPU OOM for polynomials with 2^23+ evaluations × rate expansion |

Polynomials outside `[2, 22]` vars are not GPU-committed. They use evaluation-only proofs (prover sends the evaluation, verifier checks sumcheck consistency).

#### Per-Device Table Caching

The `BasefoldTable` (precomputed folding coefficients) is cloned to every GPU device once during `GpuCommitmentStore::new()`, stored in `per_device_tables: Vec<BasefoldTable>`. This avoids re-cloning during every opening proof call.

**File:** `src/commit/basefold.rs`, `GpuCommitmentStore::new()`.

#### Bit Decomposition for Range Checks

**Problem.** The old range-check approach used `SelectionPolynomial` — a sparse MLE with `n + t` variables (where `t = 10–20` for the range table). For `ScaleDown`, this meant `n + 10` vars; for `NonNegative`, `n + 20` vars. Committing these auxiliary polynomials dominated witness size.

**Solution.** Replace with a dense bit polynomial `B(x, y)` where `y in {0,1}^5` indexes 32 bit positions (`BIT_TABLE_VARS = 5`):

```
B(x, y) = bit y of value at x
Index = x + y * 2^n
```

The range proof does a partial evaluation of B at the claim point to get a 32-element vector, then runs a single 5-variable sumcheck against the table `T(y) = 2^y`.

| BasicBlock | Old (vars) | New (vars) | Reduction |
|-----------|-----------|-----------|-----------|
| ScaleDown | n + 10 | n + 5 | 32x |
| NonNegative | n + 20 | n + 5 | 32,768x |

**Speedup:** GPT-2 12L auxiliary commits: 2,439M -> 18.2M elements (~130x). Prove: 2.67s -> 0.95s (2.8x). VGG-16 prove: 3.15s -> 0.30s (10x).

**Files:** `src/basicblock/scale.rs`, `src/basicblock/range.rs`, `src/dag/mod.rs` (prove_range).

---

### 14.3 Sumcheck Optimizations

#### GPU Sumcheck Prover

The core sumcheck prover packs all polynomials contiguously on GPU memory (stride = original polynomial size) and runs GPU kernels for round-message computation and polynomial folding.

**Critical design:** Double-buffered fold with separate input/output buffers and swap after each round. An early in-place fold had a cross-warp race condition (thread `y=k` writes to position `k`, thread `y=k/2` reads from position `k`). Small tests (<=32 elements) passed because all threads fit in one warp.

**Files:** `cuda/sumcheck_prover.cuh`, `goldilocks-cuda-rs/src/sumcheck_prover.rs`, `src/sumcheck/gpu_prover.rs`.

#### CPU Ext2 Sumcheck Fallback

For small polynomials (few rounds), GPU kernel launch overhead dominates. The CPU fallback `CpuLinearSumcheckProverExt2` is a drop-in replacement for `GpuLinearSumcheckProver`.

**Threshold:** `ZK_GPU_SUMCHECK_THRESHOLD` (default `14`, env var). When `total_rounds <= 14`, Einsum and Reducer use the CPU path.

**File:** `src/sumcheck/cpu_ext2_prover.rs`.

#### Fused GPU Permute + Partial Evaluation

**Problem.** Einsum requires variable reordering (permutation) before partial evaluation. The baseline approach is: CPU permute -> GPU upload -> GPU partial_eval. For large weight matrices, the CPU permute is the bottleneck.

**Solution.** A single GPU kernel that reads from the original (unpermuted) layout using split lookup tables in shared memory:

```
output[j] = sum_{b in {0,1}^m} evals[perm(b | j << m)] * eq(r, b)
```

The split-LUT design stores `lo_lut[2^(n/2)]` and `hi_lut[2^(n - n/2)]` in shared memory. For any new-index `idx`, the old-index is `lo_lut[idx & lo_mask] | hi_lut[idx >> half]` — O(1) per element.

**Thresholds:**

| Check | Value | Reason |
|-------|-------|--------|
| `n > ZK_GPU_FUSED_THRESHOLD` | default 16 | Skip for small polys |
| `n <= FUSED_MAX_N` | 28 | Shared memory overflow for n >= 29 (LUTs exceed 164KB on A100); u32 index overflow for n > 32 |
| `needs_permute` | true | Skip identity permutations |
| `m > 0` | true | Skip when no partial eval needed |

**Speedup:**

| Model | Without Fused | With Fused | Speedup |
|-------|--------------|-----------|---------|
| GPT-2 12L | 5.48s | 5.15s | 1.06x |
| BERT 24L | 9.82s | 8.85s | 1.11x |
| LLaMA 4L | 5.87s | 3.72s | 1.58x |
| LLaMA 8L | 10.03s | 5.91s | 1.70x |

The fused kernel helps most when the permutation is "expensive" (many variables reordered) and the polynomial is large but not too large for shared memory.

**Files:** `cuda/fused_permute_peval.cuh`, `goldilocks-cuda-rs/src/partial_eval.rs` (`fused_permute_partial_eval`), `src/basicblock/einsum.rs` (`prepare_input_poly`).

#### LUT-Based CPU Permutation

For the CPU fallback path (when fused kernel is not used), variable permutation uses a 2-half LUT approach for `n > 16`:

- Precompute `lo_lut[2^(n/2)]` and `hi_lut[2^(n - n/2)]`
- Parallel gather via `into_par_iter()` for arrays >= 256K elements (`PAR_THRESHOLD = 1 << 18`)

For `n <= 16`, direct bit manipulation (sequential gather).

**Speedup:** ~2x on weight matrices vs per-element O(n) bit operations.

**File:** `src/basicblock/einsum.rs`, `permute_evals_by_ranges()`.

#### GPU Partial Eval in CPU Sumcheck Path

**Problem.** For batch=1, total_rounds often falls to 12-14 (below `GPU_SUMCHECK_THRESHOLD`), so the CPU sumcheck path is selected. But the weight matrices can be 128M elements, and CPU `partial_eval_ext2_cpu` took 72% of Einsum prove time.

**Solution.** In the CPU sumcheck path, use `goldilocks_cuda::partial_eval_ext2()` (the high-level GPU API) when `n > GPU_PARTIAL_EVAL_THRESHOLD`:

```
if n > gpu_partial_eval_threshold() {
    partial_eval_ext2(permuted, challenges)       // GPU
} else {
    partial_eval_ext2_cpu(permuted, challenges)   // CPU
}
```

Must use the high-level API (which handles buffer sizing: needs `2^(n-1)` for first round output), not the low-level `partial_eval_ext2_device_u64`.

**Threshold:** `ZK_GPU_PARTIAL_EVAL_THRESHOLD` (default `16`, env var).

**Speedup:** GPT-J: 3.02x. LLaMA: 3.11x. BERT: 1.54x.

**File:** `src/basicblock/einsum.rs`, `prepare_input_poly()`.

#### Parallel Lagrange Basis Evaluation

The eq polynomial table `eq(r, x)` is computed iteratively, doubling in size each round. For large challenge vectors (high-dimensional polynomials), the later rounds operate on millions of Ext2 elements.

When `half >= 8192`, the update loop uses `split_at_mut(half)` + `par_iter_mut().zip()` for parallel processing.

**File:** `src/poly/mod.rs`, `evaluate_lagrange_basis_ext2()`.

---

### 14.4 Opening Proof Optimizations

Opening proofs are the single most expensive phase for large models (65% of prove time for LLaMA 3.1 8B). Multiple optimizations target this phase.

#### GPU Opening Proofs (Basefold)

For polynomials with `n > CPU_OPEN_THRESHOLD` (default 14), opening proofs run on GPU. Each `open_ext2` call performs:

1. Ext2 eq polynomial on GPU
2. Bit-reverse + mixed dot product for evaluation
3. Iterative sumcheck rounds (interp + product + reduce)
4. Codeword folding using Basefold table
5. Merkle tree construction per round
6. Query proof extraction (Merkle auth paths)

**Pre-allocated buffers** (avoid per-round `cudaMalloc`/`cudaFree`):

| Buffer | Purpose | Allocation |
|--------|---------|------------|
| `d_eq_a`, `d_eq_b` | Double-buffered eq polynomial | Once at max size (n * 2 u64) |
| `d_bh_a`, `d_bh_b` | Double-buffered bh evaluations | Once at max size |
| `pc0`, `pc1`, `pc2` | Partial reduction temporaries | Once at 256 * 2 u64 each |
| `fold_buffers[0..n]` | All codeword fold outputs | Pre-allocated at exact sizes |

**File:** `goldilocks-cuda-rs/src/basefold.rs`, `BasefoldCommitment::open_ext2()`.

#### Dual CPU+GPU Opening Pool

Opening tasks are split into two concurrent pools:

```
┌─────────────────────────────────────────┐
│  std::thread::scope                     │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ CPU pool      │  │ GPU pool        │  │
│  │ n <= 14       │  │ n > 14          │  │
│  │ rayon threads │  │ rayon threads   │  │
│  │ cpu_pool_size │  │ gpu_pool_size   │  │
│  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────┘
```

- **CPU pool:** `num_cores - gpu_pool_size` threads, runs `cpu_full_open_ext2()`
- **GPU pool:** `num_devices * GPU_OPEN_THREADS_PER_DEVICE` threads, runs `commitment.open_ext2()`
- GPU tasks sorted largest-first for load balancing
- Device assignment: `device_ids[edge]` (sticky to commit device)

**Threshold:** `CPU_OPEN_THRESHOLD` (default `14`, env var). `GPU_OPEN_THREADS_PER_DEVICE` (default `12`, env var).

**File:** `src/dag/mod.rs`, opening proof section of `prove()` and `prove_parallel()`.

#### Per-Thread CUDA Streams

The nvcc flag `--default-stream per-thread` (set in `goldilocks-cuda-rs/build.rs`) gives each CPU thread its own CUDA stream. This means multiple opening proof threads on the same GPU device can overlap kernel execution, achieving 20-30% better GPU utilization than serialized streams.

#### Opening Proof Deduplication (Reducer-Based)

When an edge has multiple claims (K > 1), a CPU-based opening reducer combines them into a single claim using random linear combination:

1. Sample `alpha` from transcript
2. Compute `eq_combined = sum_i alpha^i * eq(point_i, x)`
3. Run CPU Ext2 sumcheck on `[poly, eq_combined]`
4. Produce a single combined claim -> single opening proof

This deduplication saves `K - 1` opening proofs per multi-claim edge. In LLaMA 3.1 8B, 129 proofs are saved.

**File:** `src/dag/mod.rs`, opening reducer loop.

#### GPU Re-Upload Path

When GPU memory is insufficient to hold all commitments simultaneously (e.g., YOLO 640x640), the prover downloads commitment data to host (`HostCommitmentCache`) and frees GPU memory. During the opening phase, each edge's data is re-uploaded on demand:

1. Group tasks by edge_id
2. Per edge: `cache.to_device()` (upload codeword + bh_evals, rebuild Merkle tree)
3. Open all claims for that edge
4. Drop GPU commitment (frees memory for next edge)

**Trigger:** Activated when `gpu_store.host_caches` contains data.

**File:** `src/dag/mod.rs`, `goldilocks-cuda-rs/src/basefold.rs` (`HostCommitmentCache::to_device()`).

#### Sparse Polynomial Evaluation

Sparse polynomials (used in lookup proofs) store only non-zero entries. Evaluation at a point uses O(k * n) time instead of O(2^n):

```rust
for (idx, val) in self.evaluations.iter() {
    // Compute eq(idx, point) in O(n) per entry
    result += val * eq_at_index(idx, point);
}
```

**Speedup:** 270s -> 0.9s on lookup proofs.

**File:** `src/poly/sparse.rs`, `evaluate_at_point_ext2()`.

---

### 14.5 Multi-GPU Partition Proving

#### Architecture

The DAG is divided into independent partitions (typically one per GPU). Each partition contains a contiguous range of layers. The proving flow:

```
Output claims + Boundary claims
        |
   Route to partitions (by producer node)
        |
   Fork transcript per partition (domain separation)
        |
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ GPU 0    │ │ GPU 1    │ │ GPU 2    │ │ GPU 3    │
   │ Part. 0  │ │ Part. 1  │ │ Part. 2  │ │ Part. 3  │
   │ backward │ │ backward │ │ backward │ │ backward │
   └──────────┘ └──────────┘ └──────────┘ └──────────┘
        |              |            |            |
   Merge claims from all partitions
        |
   Lookup proofs (range, two_pow) — single transcript
        |
   Opening proofs — multi-GPU parallel
```

**Boundary edges** between partitions are force-committed (Basefold PCS). The verifier checks boundary claims via opening proofs, ensuring cross-partition consistency.

**Transcript forking:** Each partition gets `transcript.fork(k)` which absorbs the partition id for domain separation. The main transcript is used for lookup and opening proofs.

**File:** `src/dag/partition.rs` (partition description), `src/dag/mod.rs` (`prove_parallel`, `verify_parallel`).

**Configuration:** `NUM_PARTITIONS` env var (default 1). `set_partition_boundaries(num_partitions)` selects evenly-spaced layer boundaries.

---

### 14.6 Memory Management

#### Deferred Witness Deallocation

**Problem.** After the backward pass, intermediate witness data (4,762 edges for LLaMA 3.1 8B) is no longer needed. The naive approach sets `w.data = None` for each, triggering `drop()` on large vectors. For LLaMA 8B, this deallocation takes **13 seconds** — half the total prove time. Worse, if done asynchronously in a background thread, the deallocator contends with opening proofs for memory bandwidth, slowing openings from 8s to 17s.

**Solution.** Skip the deallocation entirely when sufficient system RAM is available. The intermediate data stays in memory but is never accessed again. When system memory is constrained (e.g., the Whisper model needed 45.7GB freed), selective freeing is used instead — only dropping edges not needed for lookup or opening proofs.

**File:** `src/dag/mod.rs`, `prove()` and `prove_parallel()`.

**Impact:** 13s savings on LLaMA 3.1 8B (prove: 26.5s -> 13.0s).

#### CUDA Stale Error Clearing

After GPU-intensive phases (sumcheck proving, `gpu_store.free_gpu()`), CUDA may leave a pending error (e.g., `cudaErrorMemoryAllocation`, code 2). `cudaDeviceSynchronize()` alone does NOT clear it — you must call `cudaGetLastError()`. Without clearing, subsequent GPU kernel launches silently fail.

The prover clears stale errors at phase transitions:

```rust
goldilocks_cuda::synchronize();
goldilocks_cuda::get_last_error(); // Clear pending error
```

**Files:** `goldilocks-cuda-rs/src/lib.rs` (`get_last_error`, `peek_at_last_error`, `mem_get_info`), `src/dag/mod.rs` (phase transitions).

---

### 14.7 Environment Variables Reference

All configurable thresholds with their defaults:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ZK_GPU_SUMCHECK_THRESHOLD` | 14 | Einsum/Reducer: GPU sumcheck when total_rounds > this |
| `ZK_GPU_PARTIAL_EVAL_THRESHOLD` | 16 | CPU sumcheck path: GPU partial_eval when n > this |
| `ZK_GPU_FUSED_THRESHOLD` | 16 | Fused permute+peval kernel when n > this (capped at 28) |
| `CPU_OPEN_THRESHOLD` | 14 | Opening proofs: GPU when n > this, CPU otherwise |
| `GPU_OPEN_THREADS_PER_DEVICE` | 12 | Threads per GPU device in opening proof pool |
| `NUM_PARTITIONS` | 1 | Number of partitions for parallel proving (> 1 enables multi-GPU) |
| `NUM_LAYERS` | varies | Number of transformer layers (model-specific) |
| `ZK_OPEN_TIMING` | unset | Set to any value to enable per-round opening proof timing |

---

### 14.8 Monolith Hash Function

Monolith is a hash function designed specifically for the Goldilocks field. It replaces Poseidon2 as the default hash for Merkle tree construction and commitment schemes.

**Monolith vs Poseidon2:**

| Property | Poseidon2 | Monolith |
|----------|-----------|----------|
| State width | 8 | 12 |
| Rate | 4 | 8 |
| Rounds | 30 (4 full + 22 partial + 4 full) | 6 |
| S-box | x^7 (field exponentiation) | Bars (bitwise rotation + AND) |
| Non-linear | All elements (full) or element[0] (partial) | First 4 elements + Feistel squaring |
| MDS | 4×4 circulant (cheap additions) | 12×12 circulant (u128 accumulation) |
| Digest | 4 field elements | 4 field elements (identical) |

**GPU implementation** (`cuda/monolith.cuh`, `cuda/monolith_kernels.cu`):
- Bars: parallel byte-level bitwise rotations — extremely cheap (5 bitwise ops per element)
- Bricks: Feistel Type-3 squaring (reverse loop, 11 `gl_mul` calls)
- Concrete: 12×12 MDS via u128 accumulation — each product is `u64 × small_const` (≤ 26), accumulated into `(acc_lo, acc_hi)`, then a single `reduce128` per row. This gives **1 reduction per row** vs 12 reductions with `gl_mul`.
- **Critical**: Bars operates on raw u64 bits. State values MUST be canonicalized (< p) before Bars, because `gl_add` may return non-canonical values. The `canonicalize()` call before each `bar_64()` ensures correctness.

**Compile-time feature flag**: `monolith` (default on). Both Poseidon2 and Monolith produce 4-element digests, so the `Poseidon2Hash` struct is reused. Conditional dispatch via `#[cfg(feature = "monolith")]` in `merkle.rs`, `basefold.rs`, and `cpu_basefold.rs`.

**Performance (LLaMA 3.1 8B, 32 layers, 4× A100-80GB):**

| Metric | Poseidon2 | Monolith | Improvement |
|--------|-----------|----------|-------------|
| GPU openings | 8.06s | 7.47s | 7% faster |
| CPU openings | 2.88s | 1.56s | 85% faster |
| Commit | 1.44s | 1.19s | 17% faster |
| Verify | 378ms | 239ms | 37% faster |
| **Total prove** | **12.15s** | **11.94s** | **1.7% faster** |

**Files:**
- `cuda/monolith.cuh` — constants, permutation, compress functions
- `cuda/monolith_kernels.cu` — GPU kernel launches for Merkle trees
- `goldilocks-cuda-rs/src/cpu_monolith.rs` — CPU fallback for verification
- `goldilocks-cuda-rs/cuda/wrapper.cu` — FFI wrappers

---

### 14.9 Optimization Impact Summary

Combined impact of all optimizations, measured on 4x A100-80GB:

| Model | Naive Baseline | Optimized (Monolith) | Speedup |
|-------|---------------|----------------------|---------|
| GPT-2 (12L) | ~72s | ~5s | ~14x |
| LLaMA 3.1 8B (32L) | would not run | 11.9s prove + 1.2s commit | - |
| VGG-16 | ~30s | ~2.3s | ~13x |

The largest individual wins:

1. **Level-parallel forward pass**: 17x (GPT-2)
2. **Deferred deallocation**: 2x (LLaMA 8B)
3. **GPU partial eval in CPU path**: 3x (LLaMA)
4. **Bit decomposition range checks**: 2.8x prove, 130x commit size (GPT-2)
5. **Fused permute+peval**: 1.7x (LLaMA)
6. **GPU opening proofs**: 2.2x (GPT-2)
7. **Monolith hash**: 37% faster verification, 17% faster commits

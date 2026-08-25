# zk-torch-3: GPU-Native ZKML with Goldilocks Field

## 1. Vision

**zk-torch-3** re-implements the zk-torch-2 proof system so that every field operation runs on GPU using our custom Goldilocks CUDA kernels (`cuda/` + `goldilocks-cuda-rs/`). The key difference from zk-torch-2:

| Aspect | zk-torch-2 | zk-torch-3 |
|--------|-----------|-----------|
| Field | BN254/BLS12-381 scalar field (256-bit) | Goldilocks (64-bit, p = 2^64 - 2^32 + 1) |
| Backend | arkworks CPU / icicle GPU | Custom CUDA kernels in `cuda/` |
| Polynomial storage | `Vec<F>` on CPU | `DeviceBuffer<u64>` on GPU |
| Commitment | KZH3 (pairing-based MSM) | Basefold (hash-based, GPU Poseidon2 Merkle) |
| Transcript | Merlin (CPU, Keccak) | DuplexChallenger (GPU, Poseidon2 sponge) |
| Extension field | N/A (large prime) | GoldilocksExt2 (X²-7) for challenges |
| Data movement | CPU↔GPU per kernel | Data stays on GPU; only I/O at boundaries |

## 2. Architecture Overview

```
zk-torch-3/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # crate root, config, constants
│   ├── main.rs                 # top-level prover/verifier demo
│   │
│   ├── field.rs                # GoldilocksField CryptoField impl + GPU ops
│   ├── transcript.rs           # Poseidon2-based Fiat-Shamir transcript
│   │
│   ├── poly/
│   │   ├── mod.rs              # MLPoly trait, CryptoField trait
│   │   ├── dense.rs            # DenseMLPoly (GPU-resident via DeviceBuffer)
│   │   └── sparse.rs           # SparseMLPoly (CPU, with GPU eval helper)
│   │
│   ├── commit/
│   │   ├── mod.rs              # MLPolyCommit trait + Commitment trait
│   │   ├── basefold.rs         # Basefold PCS (wraps goldilocks-cuda-rs)
│   │   └── sparse_basefold.rs  # Basefold for sparse polynomials
│   │
│   ├── sumcheck/
│   │   ├── mod.rs              # SumcheckProver trait, SumcheckProof struct
│   │   ├── linear_prover.rs    # GPU-accelerated linear sumcheck
│   │   ├── general_prover.rs   # GPU-accelerated general linear sumcheck
│   │   └── verifier.rs         # Sumcheck verifier (CPU, lightweight)
│   │
│   ├── basicblock/
│   │   ├── mod.rs              # BasicBlock trait, BasicBlockType enum
│   │   ├── add.rs              # Add, Sub
│   │   ├── einsum.rs           # Einsum (GPU matrix/tensor ops)
│   │   ├── scale.rs            # ScaleDown, ScaleUp
│   │   ├── exp.rs              # ExpHelper, TwoPow
│   │   ├── range.rs            # NonNegative (range check)
│   │   ├── permute.rs          # Permute
│   │   ├── reducer.rs          # Reducer (combine multiple claims)
│   │   ├── shape.rs            # ChangeShape
│   │   └── llama.rs            # LLaMA-specific: RMSReciprocal, DivConst, SoftmaxConst, SigmoidConst
│   │
│   ├── dag/
│   │   ├── mod.rs              # Dag struct, commit/prove/verify orchestration
│   │   ├── builder.rs          # DagBuilder DSL
│   │   ├── dense.rs            # dense_add_relu composition
│   │   ├── llama.rs            # LLaMA model graph
│   │   ├── gpt2.rs             # GPT-2 model graph
│   │   ├── bert.rs             # BERT model graph
│   │   └── gptj.rs             # GPT-J model graph
│   │
│   └── util/
│       ├── mod.rs
│       ├── arith.rs            # Arithmetic helpers (pow_2, log2, etc.)
│       ├── config.rs           # YAML config loader
│       ├── shape.rs            # Shape utilities
│       └── serialization.rs    # Proof serialization
│
├── bin/
│   ├── llama.rs
│   ├── gpt2.rs
│   ├── bert.rs
│   └── gptj.rs
```

## 3. Dependencies

```toml
[dependencies]
goldilocks-cuda = { path = "../goldilocks-cuda-rs" }  # Our GPU primitives
plonky2 = { version = "0.2.2", features = ["timing"] }
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
bincode = "1.3"
once_cell = "1.15"
rand = "0.8"
rayon = "1.10"
ndarray = { version = "0.15", features = ["serde"] }
ndarray_einsum_beta = "0.7"
env_logger = "0.10"
log = "0.4"
tract-onnx = "=0.21.6"  # ONNX model loading (reuse from zk-torch-2)
```

**No arkworks, no icicle, no merlin** — all field and crypto operations use our CUDA kernels.

## 4. Core Abstractions

### 4.1 CryptoField Trait

```rust
// src/poly/mod.rs
pub trait CryptoField:
    Clone + Copy + std::fmt::Debug + PartialEq + Send + Sync + 'static
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_u32(n: u32) -> Self;
    fn from_u64(n: u64) -> Self;
    fn to_u64(&self) -> u64;
    fn invert(&self) -> Self;
}
```

**Implementation for `GoldilocksField`**: direct wrapping of the `goldilocks-cuda-rs::GoldilocksField(u64)` type. CPU arithmetic uses the existing modular ops (gl_add, gl_sub, gl_mul defined in Rust). This keeps the CPU path available for verification.

### 4.2 DenseMLPoly — GPU-Resident Multilinear Polynomial

This is the most critical data structure. In zk-torch-2, `DenseMLPoly<F>` stores `Vec<F>` on CPU. In zk-torch-3, it wraps a `DeviceBuffer<u64>` on GPU.

```rust
// src/poly/dense.rs
use goldilocks_cuda::{DeviceBuffer, GoldilocksField};

pub struct DenseMLPoly {
    pub n: usize,                    // number of variables
    pub d_evals: DeviceBuffer<u64>,  // 2^n evaluations on GPU
}
```

**Key methods:**
- `new(n, data: Vec<GoldilocksField>) -> Self` — upload to GPU
- `from_device(n, d_evals: DeviceBuffer<u64>) -> Self` — wrap existing GPU buffer
- `fix_variables(&self, partial_point: &[GoldilocksField]) -> DenseMLPoly` — GPU partial_eval
- `fix_variables_ext2(&self, partial_point: &[GoldilocksExt2]) -> DenseMLPolyExt2` — GPU partial_eval GL→ext2
- `evaluate_at_point(&self, point: &[GoldilocksField]) -> GoldilocksField` — full GPU partial eval to single element
- `evaluations(&self) -> Vec<GoldilocksField>` — download from GPU (expensive, avoid in hot path)
- `len(&self) -> usize` — `1 << n`
- `clone_on_device(&self) -> DenseMLPoly` — GPU-to-GPU copy

**GPU operations used:**
- `partial_eval_gl_device()` for `fix_variables`
- `partial_eval_ext2_device()` for `fix_variables_ext2`
- `DeviceBuffer::from_slice()` for upload
- `DeviceBuffer::to_vec()` for download

### 4.3 DenseMLPolyExt2 — Extension Field Polynomial on GPU

```rust
pub struct DenseMLPolyExt2 {
    pub n: usize,
    pub d_evals: DeviceBuffer<u64>,  // 2^n ext2 elements (2 * 2^n u64s)
}
```

Mirrors `DenseMLPoly` but for extension field. Used after first challenge application in sumcheck.

### 4.4 SparseMLPoly — CPU with GPU Evaluation Helper

Sparse polynomials remain on CPU (they're inherently sparse, so GPU parallelism helps less). But evaluation at a point can use GPU eq_lagrange:

```rust
pub struct SparseMLPoly {
    pub n: usize,
    pub evaluations: HashMap<usize, GoldilocksField>,
    pub selection: SelectionPolynomial,
}
```

For `evaluate_at_point`: compute `eq(r, *)` table on GPU, download the entries at sparse indices, dot-product on CPU.

### 4.5 MLPoly Trait

```rust
pub trait MLPoly: std::fmt::Debug + Send + Sync {
    fn fix_variables(&self, partial_point: &[GoldilocksField]) -> Box<dyn MLPoly>;
    fn n(&self) -> usize;
    fn len(&self) -> usize;
    fn evaluate_at_point(&self, point: &[GoldilocksField]) -> GoldilocksField;
    fn evaluations(&self) -> Vec<GoldilocksField>;
    fn index(&self, index: usize) -> GoldilocksField;
    fn clone_box(&self) -> Box<dyn MLPoly>;
    fn as_any(&self) -> &dyn std::any::Any;
    fn mul_by_scalar(&self, scalar: GoldilocksField) -> Box<dyn MLPoly>;
    fn add(&self, other: &dyn MLPoly) -> Box<dyn MLPoly>;
}
```

### 4.6 Transcript — GPU Poseidon2

```rust
// src/transcript.rs
use goldilocks_cuda::challenger::DuplexChallenger;

pub struct Transcript {
    challenger: DuplexChallenger,
}

impl Transcript {
    pub fn new(_label: &[u8]) -> Self;
    pub fn append_scalar(&mut self, label: &[u8], scalar: &GoldilocksField);
    pub fn append_scalars(&mut self, label: &[u8], scalars: &[GoldilocksField]);
    pub fn append_u64(&mut self, label: &[u8], value: u64);
    pub fn challenge_scalar(&mut self, label: &[u8]) -> GoldilocksField;
    pub fn challenge_ext2(&mut self, label: &[u8]) -> GoldilocksExt2;
    pub fn challenge_vector(&mut self, label: &[u8], len: usize) -> Vec<GoldilocksField>;
}
```

This wraps the GPU `DuplexChallenger` (Poseidon2 sponge). The `label` parameter is absorbed as bytes for domain separation.

## 5. Polynomial Commitment: Basefold

### 5.1 Why Basefold (not KZH3)

KZH3 requires elliptic curve pairings (BN254/BLS12-381), which don't exist for Goldilocks. Basefold is a hash-based PCS that works with any field:
- Commitment = Merkle root of evaluations over a larger domain
- Opening = folding proof + Merkle authentication paths
- No trusted setup needed
- Our GPU kernels already implement the full pipeline

### 5.2 MLPolyCommit Trait

Same interface as zk-torch-2, instantiated with Basefold:

```rust
pub trait MLPolyCommit {
    type CommitmentKey;
    type VerifierKey;
    type Commitment: Commitment;
    type Proof;
    type BatchProof;

    fn setup(n: usize, sf_log: usize, offset: i128, size: usize) -> Self::CommitmentKey;
    fn commit(poly: &DenseMLPoly, key: &Self::CommitmentKey) -> Self::Commitment;
    fn open(commitment: &Self::Commitment, poly: &DenseMLPoly, key: &Self::CommitmentKey, point: &[GoldilocksField]) -> Self::Proof;
    fn verify(commitment: &Self::Commitment, proof: &Self::Proof, key: &Self::VerifierKey, point: &[GoldilocksField]) -> bool;
    fn batch_open(...) -> Self::BatchProof;
    fn batch_verify(...) -> bool;
}
```

### 5.3 BasefoldCommit Implementation

```rust
pub struct BasefoldCommit;
pub struct BasefoldCommitKey {
    pub table: BasefoldTable,  // precomputed folding table on GPU
    pub log_rate: usize,
    pub num_queries: usize,
}
pub type BasefoldCommitment = goldilocks_cuda::BasefoldCommitment;
pub type BasefoldProof = goldilocks_cuda::BasefoldProof;

impl MLPolyCommit for BasefoldCommit {
    type CommitmentKey = BasefoldCommitKey;
    type Commitment = BasefoldCommitment;
    type Proof = BasefoldProof;
    // ... delegates to goldilocks_cuda::BasefoldCommitment::commit/open/verify
}
```

### 5.4 Sparse Basefold

For `SparseMLPoly`, we densify (convert to `DenseMLPoly`) before committing. This is acceptable because sparse polys in zk-torch-2 are typically selection polynomials with at most `2^TABLE_COMMIT_LOG` entries, and are split into blocks.

## 6. Sumcheck Protocol — GPU-Accelerated

### 6.1 LinearSumcheckProver

The hot inner loop is computing round messages. For ℓ polynomials with num_var variables:

```
s_m(c) = Σ_{y ∈ {0,1}^{n-m-1}} Π_{i=1}^ℓ p_i(r_1,...,r_m, c, y)
```

**GPU acceleration strategy:**

Each round, we have ℓ arrays of size `2^(n-m)`. We need to evaluate the round polynomial at ℓ+1 points.

1. **Store all arrays on GPU** as `DeviceBuffer<u64>` (already via `DenseMLPoly.d_evals`)
2. **For each evaluation point c ∈ {0, 1, ..., ℓ}:**
   - Compute the product Π p_i(r_1,...,r_m, c, y) for all y in parallel on GPU
   - Sum-reduce on GPU to get s_m(c)
3. **After verifier challenge r_{m+1}:** fold each polynomial via `partial_eval` (one layer)

**New CUDA kernel needed:** `sumcheck_round_kernel`
- Input: ℓ DeviceBuffer<u64> arrays, each of size `2^(n-m)`
- Output: ℓ+1 field elements (the round polynomial evaluated at {0, 1, ..., ℓ})
- For each y ∈ {0,1}^{n-m-1}, each thread computes Π p_i(c, y) for one c value

We can reuse the existing `basefold.rs` sumcheck product kernels (`sumcheck_product_gl`, `sumcheck_eval_gl`) as a starting point, or write a new generalized kernel.

### 6.2 GeneralLinearSumcheckProver

Used for lookup proofs (range + two_pow). Same GPU strategy but with the "general" inner product form involving selection polynomials.

### 6.3 SumcheckVerifier

Runs on CPU — verification is lightweight (evaluating a degree-ℓ univariate at one point per round). No GPU needed.

## 7. BasicBlock Implementations

Each BasicBlock implements `run`, `prove`, `verify`. The key difference from zk-torch-2: polynomial data lives on GPU.

### 7.1 Witness Structure

```rust
pub struct Witness {
    pub shape: Vec<usize>,
    pub data: Option<Box<dyn MLPoly>>,
    pub data_int: Option<Vec<i128>>,  // CPU copy for integer checks (range, exp)
    pub poly_type: PolyType,
    pub data_type: DataType,
    pub sf: usize,
    pub role: Role,
}
```

Same as zk-torch-2 but `MLPoly` wraps GPU data.

### 7.2 Block-by-Block Strategy

| Block | run() | prove() | Key GPU ops |
|-------|-------|---------|-------------|
| **Add/Sub** | Element-wise add/sub on GPU | Claim reduction (no sumcheck) | `GoldilocksBatch::add/sub` |
| **Einsum** | Permute + broadcast + element-wise mul + fold on GPU | Sumcheck over free+summation vars | `partial_eval`, sumcheck kernels |
| **ScaleDown/ScaleUp** | Element-wise mul/div on GPU | Pass-through + auxiliary range proof | `GoldilocksBatch::mul` |
| **ExpHelper** | Bit decomposition (CPU), field lifting (GPU) | Selection poly sumcheck | GPU eq + partial_eval |
| **TwoPow** | Power-of-two table (GPU) | Lookup proof | GPU eq + partial_eval |
| **NonNegative** | Range check (CPU integer check) | Selection poly sumcheck | GPU eq + partial_eval |
| **Permute** | Reorder evaluations (CPU index manipulation, GPU data gather) | Sumcheck with permuted indices | GPU partial_eval |
| **Reducer** | Identity | Random linear combination sumcheck | GPU eq + partial_eval |
| **ChangeShape** | Reshape/pad (GPU) | Pass-through | `DeviceBuffer` manipulation |
| **RMSReciprocal** | RMS norm reciprocal (CPU integer, GPU field) | Sumcheck | GPU partial_eval + mul |
| **DivConst** | Division by constant (GPU scalar mul) | Pass-through + range | `GoldilocksBatch::mul` |
| **SoftmaxConst/SigmoidConst** | Piecewise linear approx (CPU), field lifting (GPU) | Sumcheck | GPU partial_eval |

### 7.3 Add/Sub (simplest, implement first)

```rust
impl BasicBlock for Add {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // GPU: element-wise add of input[0].d_evals + input[1].d_evals
        let d_result = GoldilocksBatch::add(&inputs[0].d_evals, &inputs[1].d_evals);
        vec![Witness::from_device(...)]
    }

    fn prove(&self, witnesses: &[&Witness], edge_ids: &[usize], out_claims: &[&Claim], transcript: &mut Transcript)
        -> (Vec<SumcheckProof>, Vec<Claim>)
    {
        // Add reduces claims directly: claim on output at point r
        // => claims on input_0 and input_1 at the same point r with same eval contribution
        // No sumcheck needed, just propagate claims
    }
}
```

### 7.4 Einsum (most complex, core of matmul/conv)

The einsum block handles tensor contractions. In zk-torch-2, `prove()` works by:
1. Permuting evaluation arrays so free variables are leftmost, summation variables rightmost
2. Running sumcheck over all variables
3. For each round: computing round message (product of ℓ polynomials summed over boolean hypercube)

**GPU strategy for Einsum prove:**
1. Permute evaluations on GPU (kernel: `permute_evals_kernel` — new)
2. Store permuted arrays as `DenseMLPoly` on GPU
3. Run GPU-accelerated sumcheck (Section 6.1)
4. After sumcheck: the verifier has challenges; open polynomials at the challenge point
5. Return claims on input edges

## 8. DAG Orchestration

### 8.1 Dag Structure

Identical to zk-torch-2's `Dag`:
```rust
pub struct Dag {
    nodes: Vec<Node>,
    num_edges: usize,
    topo: Vec<NodeId>,
    range: Vec<NodeId>,
    two_pow: Vec<NodeId>,
    consumers: Vec<Vec<NodeId>>,
    producers: Vec<Option<NodeId>>,
    input_ports: Vec<EdgeId>,
    output_ports: Vec<EdgeId>,
    // aliases...
}
```

### 8.2 Flow

```
DagBuilder::new()
  .input(shape, dtype)
  .pipe(dense_add_relu(w, b))
  .compile()
→ (Dag, Vec<Vec<Witness>>)

Dag::run(&mut witnesses, feed)              // forward pass, all on GPU
Dag::commit(key, witnesses, commitments)    // Basefold commit each edge polynomial
Dag::prove(key, witnesses, commitments, transcript)  // backward pass sumcheck + openings
Dag::verify(proofs, commitments, transcript)         // check all sumcheck + opening proofs
```

### 8.3 prove() Flow (backward pass)

Same algorithmic structure as zk-torch-2:
1. Open final outputs at random transcript challenges
2. Traverse nodes in reverse topological order
3. For each node: if multiple claims, reduce via Reducer
4. Call `node.kind.prove(witnesses, claims, transcript)` → new claims on inputs
5. After all nodes: prove lookups (range, two_pow)
6. Provide Basefold opening proofs for all irreducible claims

**GPU optimization**: since witnesses are already on GPU, the sumcheck `prove()` calls directly manipulate `DeviceBuffer`s. The only CPU↔GPU sync points are:
- Transcript challenge generation (squeeze Poseidon2 → 1 field element)
- Round message values (download ℓ+1 field elements per round)
- Final claims (download a few field elements)

## 9. Lookup Proofs

### 9.1 Range Proofs (NonNegative)

For each node in `self.range`, prove that the auxiliary witness's selection polynomial has boolean entries. Uses `GeneralLinearSumcheckProver` with eq * selection_poly product.

### 9.2 TwoPow Proofs

Same structure as range, but the lookup table is `{2^(-k) : k = 0..TABLE_SIZE}`.

### 9.3 GPU Acceleration

The selection polynomial is sparse, but the eq polynomial and table polynomial are dense and benefit from GPU:
- `eq_dp_all` on GPU to compute eq(r, x) for all x
- Element-wise product with table values on GPU
- Sum-reduce on GPU

## 10. New CUDA Kernels Needed

| Kernel | Purpose | File |
|--------|---------|------|
| `permute_evals_kernel` | Variable reordering for einsum | `cuda/permute.cuh` |
| `broadcast_kernel` | Duplicate evaluations for broadcast dimensions | `cuda/broadcast.cuh` |
| `elementwise_product_kernel` | Multiply ℓ arrays element-wise | `cuda/elementwise.cuh` |
| `sum_reduce_kernel` | Parallel sum reduction | `cuda/reduce.cuh` |
| `sumcheck_round_kernel` | Full sumcheck round message computation | `cuda/sumcheck.cuh` |
| `scatter_gather_kernel` | Sparse→dense / dense→sparse index operations | `cuda/scatter.cuh` |

**Existing kernels to reuse directly:**
- `partial_eval_gl`, `partial_eval_ext2_from_gl` — for fix_variables
- `eq_dp_all`, `ext2_eq_dp_all` — for eq lagrange computation
- `goldilocks_*_batch` — for element-wise field ops
- `poseidon2_*` — for Merkle trees and Fiat-Shamir
- `basefold_*` — for polynomial commitment
- `ext2_*_batch` — for extension field ops

## 11. Implementation Phases

### Phase 1: Foundation (1-2 days)
1. Create `zk-torch-3/` project with Cargo.toml
2. Implement `field.rs` — CryptoField for GoldilocksField (CPU arithmetic)
3. Implement `poly/mod.rs` — CryptoField trait, MLPoly trait
4. Implement `poly/dense.rs` — GPU-resident DenseMLPoly
5. Implement `transcript.rs` — Poseidon2-based transcript
6. Implement `util/arith.rs` — pow_2, log2_ceil, get_n, f_to_int, etc.
7. **Test**: Create DenseMLPoly, fix_variables, evaluate_at_point vs CPU reference

### Phase 2: Commitment Layer (1-2 days)
1. Implement `commit/mod.rs` — MLPolyCommit trait, Commitment trait
2. Implement `commit/basefold.rs` — wrap goldilocks-cuda-rs Basefold
3. Implement `commit/sparse_basefold.rs` — densify + commit
4. **Test**: Commit, open, verify a random polynomial

### Phase 3: Sumcheck (2-3 days)
1. Implement `sumcheck/mod.rs` — SumcheckProver trait, SumcheckProof struct
2. Implement `sumcheck/linear_prover.rs` — GPU-accelerated linear sumcheck
3. Implement `sumcheck/verifier.rs` — CPU verifier
4. Write new CUDA kernels if needed (`sumcheck_round_kernel`, `sum_reduce_kernel`)
5. **Test**: Prove and verify sumcheck for random product of 2 polynomials

### Phase 4: Basic Blocks (3-5 days)
1. Implement `basicblock/mod.rs` — BasicBlock trait, BasicBlockType enum, Witness, Claim
2. Implement blocks in order of complexity:
   - `add.rs` (Add, Sub) — simplest, no sumcheck
   - `shape.rs` (ChangeShape) — reshape only
   - `scale.rs` (ScaleDown, ScaleUp) — scalar mul + auxiliary
   - `einsum.rs` (Einsum) — full sumcheck, most complex
   - `permute.rs` (Permute) — variable reordering
   - `range.rs` (NonNegative) — selection poly + bool check
   - `exp.rs` (ExpHelper, TwoPow) — bit decomposition + table lookup
   - `reducer.rs` (Reducer) — random linear combination
3. **Test each block**: run → prove → verify for small instances

### Phase 5: DAG (2-3 days)
1. Implement `dag/mod.rs` — Dag struct, run, commit, prove, verify
2. Implement `dag/builder.rs` — DagBuilder DSL
3. Implement `dag/dense.rs` — dense_add_relu composition
4. **Test**: Build a small DAG (input → dense_add_relu → output), run + prove + verify

### Phase 6: Lookup Proofs (1-2 days)
1. Implement `prove_range`, `verify_range` in `dag/mod.rs`
2. Implement `prove_two_pow`, `verify_two_pow` in `dag/mod.rs`
3. Implement `sumcheck/general_prover.rs` — GeneralLinearSumcheckProver
4. **Test**: DAG with ScaleDown (triggers range proof)

### Phase 7: LLaMA-specific Blocks (1-2 days)
1. Implement `basicblock/llama.rs` — RMSReciprocal, DivConst, SoftmaxConst, SigmoidConst
2. **Test each block individually**

### Phase 8: Model Graphs (1-2 days)
1. Implement `dag/llama.rs` — LLaMA model graph
2. Implement `dag/gpt2.rs` — GPT-2 model graph
3. Implement `dag/bert.rs` — BERT model graph
4. Implement `dag/gptj.rs` — GPT-J model graph
5. **Test**: Small model forward pass + prove + verify

### Phase 9: Optimization + Integration (2-3 days)
1. Profile GPU kernel utilization
2. Minimize CPU-GPU sync points
3. Add CUDA stream overlapping where possible
4. End-to-end benchmark vs zk-torch-2

## 12. Constants and Configuration

```rust
// Goldilocks-specific constants
pub const SIGN_BIT: usize = 63;
pub const FIELD_SIZE: usize = 64;
pub const GOLDILOCKS_PRIME: u64 = 0xFFFFFFFF00000001;

// Basefold parameters (configurable via config.yaml)
pub const BASEFOLD_LOG_RATE: usize = 3;   // rate = 1/8 (adjustable)
pub const BASEFOLD_NUM_QUERIES: usize = 80; // security parameter
```

## 13. Key Design Decisions

### 13.1 No Dynamic Dispatch on GPU
The `MLPoly` trait uses `Box<dyn MLPoly>` for polymorphism (dense vs sparse). GPU operations are invoked via concrete types (`DenseMLPoly.d_evals`), never through trait objects.

### 13.2 Lazy Download
`DenseMLPoly::evaluations()` downloads data from GPU. This is only called when:
- Serializing proofs
- Verification (which is lightweight and can be CPU)
- Debug printing

Hot-path operations (fix_variables, evaluate_at_point, sumcheck rounds) use GPU buffers directly.

### 13.3 Extension Field Strategy
Challenges from the transcript are base field (`GoldilocksField`). When the sumcheck protocol needs extension field evaluations (e.g., for batching), we use `GoldilocksExt2` and the `partial_eval_ext2` kernels.

### 13.4 Sparse Polynomial Handling
Sparse polynomials (selection polynomials in range/exp blocks) stay on CPU. They are small and their sparsity pattern doesn't map well to GPU parallelism. For the lookup proof sumcheck, we compute the dense components (eq, table) on GPU and combine with sparse components on CPU.

### 13.5 f_to_int Helper
The `f_to_int` function (field element → signed integer) is needed for witness generation in range/scale/exp blocks. For Goldilocks, elements > p/2 are treated as negative:
```rust
fn f_to_int(f: GoldilocksField) -> i128 {
    let v = f.0;
    if v > GOLDILOCKS_PRIME / 2 {
        v as i128 - GOLDILOCKS_PRIME as i128
    } else {
        v as i128
    }
}
```

## 14. Migration Notes from zk-torch-2

### What changes:
1. **All `F: CryptoField` generics** → concrete `GoldilocksField` (no more type parameters on blocks/dag)
2. **`Vec<F>` evaluation storage** → `DeviceBuffer<u64>` on GPU
3. **KZH3** → Basefold (no SRS, no pairings)
4. **Merlin transcript** → DuplexChallenger (Poseidon2)
5. **arkworks/icicle field ops** → goldilocks-cuda CUDA kernels
6. **Pairing checks** → removed entirely (Basefold is hash-based)
7. **SRS generation/storage** → replaced by BasefoldTable (deterministic, no trusted setup)

### What stays the same:
1. **DAG architecture** — nodes, edges, topological order, backward claim reduction
2. **BasicBlock trait** — run/prove/verify interface
3. **BasicBlockType enum** — all 15 variants
4. **Sumcheck protocol** — same algorithm, different implementation
5. **DagBuilder DSL** — same API for model construction
6. **Model definitions** — LLaMA, GPT-2, BERT, GPT-J graphs
7. **Witness structure** — shape, data, data_int, poly_type, data_type, sf, role
8. **Claim structure** — edge_id, sparse_id, point, eval
9. **Config system** — YAML config with scale factor, table sizes

## 15. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Goldilocks field too small for some ML values | High | Use larger scale factor; verify range doesn't exceed p/2 |
| Basefold proof size larger than KZH3 | Medium | Tune log_rate and num_queries; proof size is O(n * security_param) |
| GPU memory for large models | High | Stream processing; process DAG nodes one at a time, free buffers after use |
| New CUDA kernels have bugs | Medium | Test against CPU reference for each kernel |
| Sumcheck GPU kernel complex to get right | High | Start with CPU sumcheck, GPU-accelerate incrementally |

## 16. Testing Strategy

1. **Unit tests per module**: Field ops, DenseMLPoly, Basefold, Sumcheck, each BasicBlock
2. **Cross-validation**: Compare outputs with zk-torch-2 (using goldilocks feature flag) for identical inputs
3. **Integration test**: Full DAG (input → dense_add_relu → output) run + prove + verify
4. **Model test**: LLaMA-tiny forward pass + proof generation
5. **Regression test**: Ensure all existing goldilocks-cuda-rs tests still pass

## 17. Completion
After all phases (phase 1 to 9) are completed, please verify the code and ensure it works as expected.
Then, please write a summary of the implementation and the results.
Finally, please write a README.md file for the project and you can output
<promise>The entire implementation of zk-torch-3 from phase 1 to 9 is completed</promise>

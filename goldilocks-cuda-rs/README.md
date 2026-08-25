# goldilocks-cuda

GPU-accelerated Goldilocks field arithmetic, extension fields, Poseidon2 hashing, and Basefold polynomial commitment scheme using CUDA.

## Features

- **Goldilocks Field**: Batch add, sub, mul, div, inverse, neg, square, double, exp
- **Extension Fields**: Quadratic (GF(p^2), X^2 - 7) and quintic (GF(p^5), X^5 - 3)
- **Field Conversion**: Batch conversion between base and extension fields
- **Poseidon2 Hash**: Batch permutation, compression, and Merkle tree construction
- **Eq Lagrange**: GPU-accelerated eq(r, x) polynomial over the Boolean hypercube
- **Fiat-Shamir Challenger**: Poseidon2-based duplex sponge challenger (single + batched)
- **GPU-Resident Merkle Tree**: Zero host-transfer tree construction with on-demand path extraction
- **Basefold PCS**: Full commit / open / verify for multilinear polynomials (base + ext2 fields)
- **Batch Basefold**: batch_open / batch_verify for multiple polynomials at multiple points

## Requirements

- CUDA Toolkit (tested with CUDA 11.0+)
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- Rust 1.70+

## Installation

### 1. Set up CUDA

Make sure CUDA is installed and one of the following environment variables is set:

```bash
export CUDA_PATH=/usr/local/cuda
# or
export CUDA_HOME=/usr/local/cuda
```

Alternatively, ensure `nvcc` is in your PATH.

### 2. Add to Cargo.toml

```toml
[dependencies]
goldilocks-cuda = { path = "path/to/goldilocks-cuda-rs" }
```

### 3. Build

```bash
cargo build --release
```

By default, the crate builds for A100 GPUs (sm_80). To build for other architectures, set the `CUDA_ARCH` environment variable:

```bash
CUDA_ARCH=sm_70 cargo build --release   # V100
CUDA_ARCH=sm_80 cargo build --release   # A100 (default)
CUDA_ARCH=sm_90 cargo build --release   # H100
```

---

## API Reference

### Initialization

```rust
goldilocks_cuda::init()?;                        // Initialize CUDA (call once)
let count = goldilocks_cuda::device_count()?;    // Number of GPUs
let name = goldilocks_cuda::device_name(0)?;     // GPU name string
```

---

### Goldilocks Field (`field.rs`)

**Types:**
- `GoldilocksField(u64)` — element of GF(p), p = 2^64 - 2^32 + 1
- `GOLDILOCKS_PRIME: u64` — the prime modulus

**High-level API** (automatic host-device transfers):

```rust
use goldilocks_cuda::prelude::*;

let a: Vec<GoldilocksField> = (0..1000).map(GoldilocksField::new).collect();
let b: Vec<GoldilocksField> = (0..1000).map(GoldilocksField::new).collect();

let sum  = GoldilocksOps::add(&a, &b)?;      // a + b
let diff = GoldilocksOps::sub(&a, &b)?;      // a - b
let prod = GoldilocksOps::mul(&a, &b)?;      // a * b
let inv  = GoldilocksOps::inverse(&a)?;       // a^(-1)
```

**Low-level batch API** (raw `DeviceBuffer`):

```rust
let d_a = DeviceBuffer::from_slice(&raw_u64_data)?;
let d_b = DeviceBuffer::from_slice(&raw_u64_data)?;
let mut d_out = DeviceBuffer::<u64>::new(n)?;

GoldilocksBatch::add(&d_a, &d_b, &mut d_out)?;
GoldilocksBatch::sub(&d_a, &d_b, &mut d_out)?;
GoldilocksBatch::mul(&d_a, &d_b, &mut d_out)?;
GoldilocksBatch::div(&d_a, &d_b, &mut d_out)?;
GoldilocksBatch::inverse(&d_a, &mut d_out)?;
GoldilocksBatch::neg(&d_a, &mut d_out)?;
GoldilocksBatch::square(&d_a, &mut d_out)?;
GoldilocksBatch::double(&d_a, &mut d_out)?;
GoldilocksBatch::exp(&d_a, exponent, &mut d_out)?;
GoldilocksBatch::mul_scalar(scalar, &d_a, &mut d_out)?;
```

---

### Extension Field GF(p^2) (`extension.rs`)

**Types:**
- `GoldilocksExt2 { c0, c1 }` — element c0 + c1*X where X^2 = 7
- `EXT2_W: u64 = 7` — the irreducible polynomial parameter

```rust
let ext = GoldilocksExt2::new(GoldilocksField(1), GoldilocksField(2));
let from_base = GoldilocksExt2::from_base(GoldilocksField(42));  // (42, 0)

// High-level batch operations
let sum  = Ext2Ops::add(&a, &b)?;
let diff = Ext2Ops::sub(&a, &b)?;
let prod = Ext2Ops::mul(&a, &b)?;
let inv  = Ext2Ops::inverse(&a)?;

// Batch conversion
let ext_vec = Ext2Ops::from_base(&base_vec)?;   // GL -> Ext2
let base    = Ext2Ops::to_base(&ext_vec)?;       // Ext2 -> GL (extracts c0)
```

**Low-level batch API**: `Ext2Batch::{add, sub, mul, inverse, neg, square, frobenius, conjugate, exp, mul_scalar}`

---

### Device API (`device.rs`)

Keep data on GPU to avoid repeated transfers. Supports operator overloads (`+`, `-`, `*`).

**`GoldilocksDevice`:**

```rust
let d_a = a.to_device()?;                // Vec<GoldilocksField> -> GPU
let d_b = b.to_device()?;

let d_sum  = d_a.add(&d_b)?;             // All stay on GPU
let d_prod = d_a.mul(&d_b)?;
let d_inv  = d_a.inverse()?;
let d_neg  = d_a.neg()?;
let d_sq   = d_a.square()?;
let d_dbl  = d_a.double()?;
let d_exp  = d_a.exp(7)?;
let d_div  = d_a.div(&d_b)?;

let d_ext2 = d_a.to_ext2()?;             // Embed into GF(p^2) on GPU
let d_ext5 = d_a.to_ext5()?;             // Embed into GF(p^5) on GPU

let result = d_prod.to_host()?;           // GPU -> Vec<GoldilocksField>

// Operator overloads (with references)
let d_sum  = (&d_a + &d_b)?;
let d_diff = (&d_a - &d_b)?;
let d_prod = (&d_a * &d_b)?;
```

**`Ext2Device`**: Same operations plus `frobenius()`, `conjugate()`, `to_base()`.

**`Ext5Device`**: Same operations plus `frobenius()`, `scale(scalar)`, `to_base()`.

**`Poseidon2Device`**: `merkle_root()`, `merkle_tree()`.

---

### Poseidon2 Hashing (`poseidon2.rs`)

**Types:**
- `Poseidon2Hash { elements: [GoldilocksField; 4] }` — 256-bit digest
- `POSEIDON2_WIDTH: usize = 8`, `POSEIDON2_RATE: usize = 4`, `POSEIDON2_DIGEST_SIZE: usize = 4`

```rust
let hash = Poseidon2Hash::from_raw([1, 2, 3, 4]);

// Batch compression
let compressed = Poseidon2Ops::compress_batch(&left, &right)?;

// Merkle tree
let tree = Poseidon2Ops::build_merkle_tree(&leaves)?;   // All layers
let root = Poseidon2Ops::merkle_root(&leaves)?;          // Just the root

// Low-level batch API
Poseidon2Batch::hash(&d_input, &mut d_output)?;
Poseidon2Batch::compress(&d_left, &d_right, &mut d_output)?;
Poseidon2Batch::merkle_layer(&d_input, &mut d_output)?;
```

---

### GPU-Resident Merkle Tree (`merkle.rs`)

Tree built entirely on GPU with zero host transfers during construction. Only the root (32 bytes) is copied to host on demand.

```rust
let tree = DeviceMerkleTree::build_from_gl_codeword(&d_codeword, cw_len)?;
let tree = DeviceMerkleTree::build_from_ext2_codeword(&d_codeword, cw_len_ext2)?;

let root = tree.root()?;                         // 32 bytes D->H
let path = tree.auth_path(leaf_index)?;           // log(N) * 32 bytes D->H
let digest = tree.leaf_digest(leaf_index)?;       // 32 bytes D->H
let n = tree.num_leaves();
```

---

### Eq Lagrange Polynomial (`eq_lagrange.rs`)

GPU-accelerated computation of eq(r, x) = prod_i(r_i*x_i + (1-r_i)*(1-x_i)) over {0,1}^n.

```rust
use goldilocks_cuda::eq_lagrange;

// Base field
let result = eq_lagrange::eq_dp_all(&r)?;         // Vec<GoldilocksField>, len = 2^n

// Extension field
let result = eq_lagrange::ext2_eq_dp_all(&r)?;    // Vec<GoldilocksExt2>, len = 2^n
```

---

### Fiat-Shamir Challenger (`challenger.rs`)

Poseidon2-based duplex sponge for Fiat-Shamir challenge generation.

**Constants:** `CHALLENGER_WIDTH = 8`, `CHALLENGER_RATE = 4`, `CHALLENGER_CAPACITY = 4`

**Single challenger (`DuplexChallenger`):**

```rust
use goldilocks_cuda::challenger::DuplexChallenger;

let mut ch = DuplexChallenger::new()?;

ch.observe(GoldilocksField(123))?;              // Absorb field element
ch.observe_slice(&field_vec)?;                   // Absorb multiple elements
ch.observe_ext2(ext2_val)?;                      // Absorb GF(p^2) element

let challenge = ch.sample()?;                    // Squeeze GoldilocksField
let challenges = ch.sample_array(4)?;            // Squeeze multiple
let ext2_challenge = ch.sample_ext2()?;          // Squeeze GoldilocksExt2
```

**Batched challengers (`ChallengerBatch`):**

```rust
use goldilocks_cuda::challenger::ChallengerBatch;

let mut batch = ChallengerBatch::new(1024)?;     // 1024 independent transcripts

batch.observe(&values)?;                          // One value per transcript
batch.observe_slice(&values, slice_len)?;         // Multiple values per transcript
batch.observe_ext2(&ext2_values)?;                // One Ext2 per transcript

let samples = batch.sample()?;                    // One challenge per transcript
let arrays = batch.sample_array(4)?;              // Multiple per transcript
let ext2s = batch.sample_ext2()?;                 // One Ext2 per transcript
```

---

### Basefold PCS (`basefold.rs`)

GPU-accelerated polynomial commitment scheme for multilinear polynomials using foldable codes.

#### Types

| Type | Description |
|------|-------------|
| `BasefoldCommitment` | Commitment to a multilinear polynomial (GPU-resident) |
| `BasefoldTable` | Precomputed folding table |
| `BasefoldProof` | Base-field opening proof |
| `BasefoldProofExt2` | Extension-field opening proof |
| `BatchBasefoldProof` | Batch opening proof (multiple polys, multiple points) |
| `Evaluation` | A claimed evaluation: `poly[i]` at `points[j]` = `value` |
| `SumcheckOracle<F>` | One round's degree-2 sum-check polynomial (c0, c1, c2) |
| `QueryProof<F>` | Per-query authentication data across folding rounds |
| `IndividualQueryProof` | Per-query data for one individual commitment |
| `FoldingEntry` | A folding table entry (point + weight) |
| `BasefoldVerifier` | Verification methods (all CPU) |
| `BasefoldTranscript` | Trait for Fiat-Shamir challenge generation |
| `TestTranscript` | Deterministic xorshift transcript for testing |

#### Folding Table

```rust
// Generate random folding table
let mut table = BasefoldTable::generate(num_vars, log_rate, num_rounds, seed);
table.upload()?;   // Upload to GPU (required before open)
```

#### Commit

```rust
// From host evaluations (transfers once)
let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate)?;

// From GPU evaluations (zero transfers except 32-byte root)
let comm = BasefoldCommitment::commit_device(&d_evals, num_vars, log_rate)?;

// Access commitment root
let root: Poseidon2Hash = comm.root.clone();
```

#### Single Open / Verify

```rust
// Open at a base-field point
let mut prover_transcript = TestTranscript::new(seed);
let proof = comm.open(&point, &table, &mut prover_transcript, num_queries)?;

// Verify
let mut verifier_transcript = TestTranscript::new(seed);
let valid = BasefoldVerifier::verify(
    &comm.root, &point, &proof, &table, &mut verifier_transcript,
)?;

// Open at an extension-field point
let proof_ext2 = comm.open_ext2(&ext2_point, &table, &mut transcript, num_queries)?;
```

#### Batch Open / Verify

Open multiple polynomials at (possibly different) evaluation points in one proof.

```rust
use goldilocks_cuda::prelude::*;

// Commit multiple polynomials
let comm_a = BasefoldCommitment::commit(&evals_a, num_vars, log_rate)?;
let comm_b = BasefoldCommitment::commit(&evals_b, num_vars, log_rate)?;

// Set up table
let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
table.upload()?;

// Define evaluation claims
let claims = vec![
    Evaluation::new(0, 0, GoldilocksField(eval_a)),   // poly 0 at point 0
    Evaluation::new(1, 1, GoldilocksField(eval_b)),   // poly 1 at point 1
];

// Batch open
let mut prover_transcript = TestTranscript::new(seed);
let proof = batch_open(
    &[&comm_a, &comm_b],
    &[&point_a[..], &point_b[..]],
    &claims,
    &table,
    &mut prover_transcript,
    num_queries,
)?;

// Batch verify (all CPU)
let mut verifier_transcript = TestTranscript::new(seed);
let valid = BasefoldVerifier::batch_verify(
    &[comm_a.root.clone(), comm_b.root.clone()],
    &[&point_a[..], &point_b[..]],
    &claims,
    &proof,
    &table,
    &mut verifier_transcript,
)?;
```

#### Custom Transcript

Implement `BasefoldTranscript` to plug in your own Fiat-Shamir mechanism:

```rust
pub trait BasefoldTranscript {
    fn observe_field(&mut self, value: GoldilocksField);
    fn observe_ext2(&mut self, value: GoldilocksExt2);
    fn observe_hash(&mut self, hash: &Poseidon2Hash);
    fn sample_challenge(&mut self) -> GoldilocksField;
    fn sample_challenge_ext2(&mut self) -> GoldilocksExt2;
}
```

#### Low-Level Basefold Kernels (`BasefoldBatch`)

For building custom protocols on top of the basefold primitives:

```rust
BasefoldBatch::bit_reverse_gl(data, log_n)?;
BasefoldBatch::bit_reverse_ext2(data, log_n)?;
BasefoldBatch::bhc_interpolate(evals, coeffs, bh_evals, num_vars)?;
BasefoldBatch::encode(coeffs, codeword, num_vars, log_rate)?;

// Codeword folding
BasefoldBatch::fold_gl(codeword, table_ptr, challenge, output, pair_count)?;
BasefoldBatch::fold_mixed(codeword, table_ptr, challenge_ext2, output, pair_count)?;
BasefoldBatch::fold_ext2(codeword, table_ptr, challenge_ext2, output, pair_count)?;

// Sum-check primitives
BasefoldBatch::sumcheck_interp_gl(data, pair_count)?;
BasefoldBatch::sumcheck_interp_ext2(data, pair_count)?;
BasefoldBatch::sumcheck_product_gl(eq, bh, pc0, pc1, pc2, pair_count, num_blocks)?;
BasefoldBatch::sumcheck_product_mixed(eq, bh, pc0, pc1, pc2, pair_count, num_blocks)?;
BasefoldBatch::sumcheck_product_ext2(eq, bh, pc0, pc1, pc2, pair_count, num_blocks)?;
BasefoldBatch::sumcheck_eval_gl(data, challenge, output, pair_count)?;
BasefoldBatch::sumcheck_eval_mixed(data, challenge_ext2, output, pair_count)?;
BasefoldBatch::sumcheck_eval_ext2(data, challenge_ext2, output, pair_count)?;

// Dot products (partial reduction to num_blocks elements)
BasefoldBatch::dot_product_gl(a, b, partial, n, num_blocks)?;
BasefoldBatch::dot_product_mixed(a, b, partial, n, num_blocks)?;
```

---

### Memory Management (`memory.rs`)

```rust
// Allocate
let d_buf = DeviceBuffer::<u64>::new(1000)?;             // Uninitialized
let d_buf = DeviceBuffer::from_slice(&host_data)?;       // Upload from host

// Transfer
let host_vec = d_buf.to_vec()?;                          // Full download
let slice = d_buf.read_slice(offset, len)?;              // Partial download
d_buf.copy_from_slice(&host_data)?;                      // Upload (same size)

// Device-to-device
let d_copy = d_buf.clone_on_device()?;                   // Clone
d_dst.copy_from_device(&d_src)?;                         // Copy (same size)

// Info
let n = d_buf.len();
let empty = d_buf.is_empty();
let ptr = d_buf.as_ptr();
let mut_ptr = d_buf.as_mut_ptr();

// Unsafe sub-region access
let offset_ptr = unsafe { d_buf.offset_ptr(n) };
let offset_mut = unsafe { d_buf.offset_mut_ptr(n) };

// Synchronization
synchronize()?;                                           // Wait for all GPU ops
```

---

## Memory Layout

| Type | Size | Layout |
|------|------|--------|
| `GoldilocksField` | 1 u64 | `[value]` |
| `GoldilocksExt2` | 2 u64 | `[c0, c1]` (c0 + c1*X) |
| `GoldilocksExt5` | 5 u64 | `[c0, c1, c2, c3, c4]` |
| `Poseidon2Hash` | 4 u64 | `[e0, e1, e2, e3]` |
| `FoldingEntry` | 2 u64 | `[point, weight]` |

## Error Handling

All operations return `Result<T, CudaError>`:

```rust
use goldilocks_cuda::{CudaError, Result};

// CudaError variants:
// - InitializationFailed   — no GPU or CUDA init failed
// - AllocationFailed       — GPU memory allocation failed
// - MemcpyFailed           — host <-> device transfer failed
// - KernelFailed           — CUDA kernel execution failed
// - SyncFailed             — device synchronization failed
// - NoDevice               — no CUDA device available
// - InvalidArgument(String) — invalid parameters
```

## Performance Tips

1. **Use Device API**: Prefer `to_device()` and device types to minimize CPU-GPU transfers
2. **Batch operations**: Always prefer batch operations over single-element operations
3. **Minimize transfers**: Keep data on GPU as long as possible; only call `to_host()` when needed
4. **Chain operations**: `d_a.mul(&d_b)?.add(&d_c)?` keeps everything on GPU
5. **Power-of-2 sizes**: Merkle tree and basefold operations require power-of-2 sizes
6. **Reuse buffers**: For tight loops, use the low-level `BasefoldBatch` / `GoldilocksBatch` API

## Testing

```bash
cargo test                    # Run all tests (requires CUDA GPU)
cargo test -- --nocapture     # With output
cargo test test_batch_open    # Run specific tests
```

47 tests cover all modules: field ops, extension fields, device API, Poseidon2, Merkle trees, eq Lagrange, challenger, basefold commit/open/verify, and batch open/verify.

## Project Structure

```
goldilocks-cuda-rs/
├── Cargo.toml              # Crate manifest
├── build.rs                # CUDA compilation script
├── README.md               # This file
├── cuda/
│   └── wrapper.cu          # C FFI wrapper for CUDA kernels
└── src/
    ├── lib.rs              # Library entry point, re-exports
    ├── ffi.rs              # Raw FFI bindings to CUDA library
    ├── error.rs            # CudaError type
    ├── memory.rs           # DeviceBuffer, synchronize()
    ├── field.rs            # GoldilocksField, GoldilocksOps, GoldilocksBatch
    ├── extension.rs        # GoldilocksExt2, Ext2Ops, Ext2Batch
    ├── device.rs           # GoldilocksDevice, Ext2Device, Ext5Device, Poseidon2Device
    ├── poseidon2.rs        # Poseidon2Hash, Poseidon2Ops, Poseidon2Batch
    ├── merkle.rs           # DeviceMerkleTree (GPU-resident)
    ├── eq_lagrange.rs      # eq_dp_all, ext2_eq_dp_all
    ├── challenger.rs       # DuplexChallenger, ChallengerBatch
    └── basefold.rs         # BasefoldCommitment, batch_open, BasefoldVerifier, BasefoldBatch
```

## Mathematical Background

### Goldilocks Prime
```
p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
```
Efficient reduction using: 2^64 = 2^32 - 1 (mod p)

### Quadratic Extension
```
GF(p^2) = GF(p)[X] / (X^2 - 7)
```

### Quintic Extension
```
GF(p^5) = GF(p)[X] / (X^5 - 3)
```

### Poseidon2
- Width: 8, Rate: 4, Capacity: 4
- Rounds: 4 external + 22 internal + 4 external
- S-box: x^7

### Basefold PCS
FRI-like polynomial commitment for multilinear polynomials over foldable codes. The commit phase encodes polynomial coefficients via Reed-Solomon-like encoding, builds a Merkle tree. Opening uses an interleaved sum-check + codeword folding protocol. Batch opening combines multiple polynomials via random linear combination with eq(x, t) weights and a two-layer sum-check (outer batching + inner commit phase).

## License

MIT OR Apache-2.0

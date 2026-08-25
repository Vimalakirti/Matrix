# almost-goldilocks-cuda

Rust bindings to the CUDA kernels in `../cuda_almost_goldilocks/` for the
**almost-Goldilocks** prime field

```
P = 2^64 - 2^32 + 1 - 32 = 2^64 - 2^32 - 31 = 0xFFFFFFFEFFFFFFE1
```

This crate is a sibling of `goldilocks-cuda-rs` but targets a slightly
different prime; types are renamed accordingly (e.g. `AlmostGoldilocksField`)
so the two crates can coexist in one workspace without symbol collision.

## Modules

| Module | What it provides |
|---|---|
| `field` | `AlmostGoldilocksField`, host arithmetic, GPU batch ops |
| `extension` | `AlmostGoldilocksExt2 = F_p[X] / (X^2 - 3)`, Karatsuba mul, batch ops |
| `eq_lagrange` | DP `eq(r, x)` over the Boolean hypercube (base + Ext2) |
| `partial_eval` | Multilinear partial evaluation (base + base→Ext2) + fused permute |
| `sumcheck_prover` | `GpuSumcheckState{,Ext2}` with round-message and fold |
| `ajtai` | Ajtai commitment `c = M·z` over `R = F_q[X]/(X^64 + 1)` for binary witnesses. Dense batched (`commit_batched`, `B ∈ {1, 2, 4, 8, 16}`) + sparse (`commit_sparse`) paths. PRG is ChaCha8 with rejection sampling. See `cuda_almost_goldilocks/ajtai.md` for the design. |
| `memory` | `DeviceBuffer<T>`, `synchronize`, `mem_get_info`, ... |

## Differences from `goldilocks-cuda-rs`

- **Prime**: `2^64 - 2^32 - 31` instead of `2^64 - 2^32 + 1`.
- **Reduction**: 2-pass Solinas with wrap constant `c = 2^32 + 31` (33 bits)
  instead of the elegant 1-step Goldilocks reduction. Roughly 1.5–2× slower
  on multiplication; add/sub/neg are essentially identical.
- **Ext2 non-residue**: `W = 3` instead of `7`, because `7` is a quadratic
  residue mod this prime (`Legendre(7) = +1`).
- **Not included**: Poseidon2, Monolith, Challenger, Merkle, Basefold.
  Their round constants are field-specific and would need regeneration.
  Ext5 is also omitted because `gcd(5, P - 1) = 1` makes `X^5 - W` reducible
  for every `W`.

## Build

The build script compiles `cuda/wrapper.cu` against the headers in
`../cuda_almost_goldilocks/` and links it as a static library. Required:

- nvcc (any modern version; system CUDA 11.5 toolkit on Ubuntu is preferred
  for driver compatibility)
- An sm_80+ GPU. Override the compute capability with `CUDA_COMPUTE=89` (or
  similar) for newer cards.

```bash
cargo build
cargo test          # runs integration tests against a CPU u128 reference
```

## Example

```rust
use almost_goldilocks_cuda::{
    init, AlmostGoldilocksField, AlmostGoldilocksOps,
};

init()?;

let a: Vec<AlmostGoldilocksField> = (1..1001).map(AlmostGoldilocksField::new).collect();
let b: Vec<AlmostGoldilocksField> = (1..1001).map(AlmostGoldilocksField::new).collect();
let product = AlmostGoldilocksOps::mul(&a, &b)?;
```

## Layout

```
almost-goldilocks-cuda-rs/
├── Cargo.toml
├── build.rs            # invokes nvcc on cuda/wrapper.cu (-I ../cuda_almost_goldilocks/)
├── cuda/
│   └── wrapper.cu      # extern "C" FFI shims around the kernel headers
├── src/
│   ├── error.rs
│   ├── memory.rs
│   ├── field.rs
│   ├── extension.rs
│   ├── eq_lagrange.rs
│   ├── partial_eval.rs
│   ├── sumcheck_prover.rs
│   ├── ffi.rs          # raw extern "C" bindings
│   └── lib.rs
└── tests/
    └── integration.rs  # GPU vs u128 CPU reference
```

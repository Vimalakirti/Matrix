# cuda_almost_goldilocks

CUDA kernels and headers for the **almost-Goldilocks** prime field

```
P = 2^64 - 2^32 + 1 - 32 = 2^64 - 2^32 - 31 = 0xFFFFFFFEFFFFFFE1
```

This directory mirrors `cuda/` (the Goldilocks port of Plonky3) but is built
around `P` above. The type names are prefixed with `Almost` and the function
prefixes are `agl_`/`aext2_` so the two fields can coexist if linked together.

## What's the same vs. Goldilocks?

Most algorithmic structure is identical — partial evaluation, sumcheck,
eq-Lagrange (DP and Walsh–Hadamard), fused permute + partial-eval — they are
field-generic. Add and subtract are essentially the same instruction
sequences; only constants change.

## What's different?

| | Goldilocks (`cuda/`) | almost-Goldilocks (this dir) |
|---|---|---|
| Prime | `2^64 - 2^32 + 1` | `2^64 - 2^32 - 31` |
| Wrap constant `c` (= `2^64 mod P`) | `2^32 - 1` (32 bits) | `2^32 + 31` (33 bits) |
| `2^96 ≡ -1 (mod P)` | Yes (used for fast `reduce128`) | No — `2^96 ≡ 2^37 + 31` |
| `reduce128` algorithm | 1 step (`-x_hi_hi + x_hi_lo·c`) | 2-pass Solinas iteration |
| Ext2 non-residue `W` | `7` | `3` (Legendre(7) = +1 here; Legendre(3) = -1) |
| Ext5 | `X^5 - 3` is irreducible | **Not provided** (`gcd(5, P-1) = 1`, so `X^5 - W` is always reducible) |
| 2-adicity of `P - 1` | 32 | 5 — NTTs only up to size 32 |

The NTT-flavored constants (`TWO_ADIC_GENERATORS`, `POWERS_OF_TWO`) were
present in `cuda/goldilocks.cuh` but **never read by any kernel** in this
repo. They are intentionally omitted here; `almost_goldilocks_init()` is a
no-op kept only for API symmetry.

## Files

| | |
|---|---|
| `almost_goldilocks.cuh` / `almost_goldilocks_kernels.cu` | Base field arithmetic and batch kernels |
| `almost_extension.cuh` / `almost_extension_kernels.cu` | `GF(p^2) = F_p[X] / (X^2 - 3)` |
| `almost_eq_lagrange.cuh` / `almost_eq_lagrange_kernels.cu` | DP and WHT `eq(r, x)` over the Boolean hypercube |
| `almost_partial_eval.cuh` | Multilinear partial evaluation (base + base→Ext2) |
| `almost_fused_permute_peval.cuh` | Fused permute + partial eval kernel |
| `almost_sumcheck_prover.cuh` | Sumcheck round-message + fold kernels (base + Ext2) |
| `ajtai.cuh` | Ajtai commitment `c = M·z` over `R = F_q[X]/(X^64 + 1)` for binary witnesses — dense batched + sparse paths |
| `ajtai_chacha8.cuh` | ChaCha8 PRG (host + device), rejection-sampled to `F_q` — derives `M` from a public seed |
| `ajtai_cpu_reference.cuh` | Parametric (`D = 4, 8, 64`) CPU reference for the commitment — used by tests |

## Tests

Each test executable builds standalone and produces a `PASS`/`FAIL` summary.

```bash
make all                   # build everything
make test                  # run every test
make test-field            # base field arithmetic
make test-extension        # Ext2 arithmetic
make test-eq-lagrange      # eq(r,x): DP and WHT vs CPU reference
make test-partial-eval     # base and Ext2 folding vs CPU reference
make test-sumcheck         # round message + fold (base and Ext2)
make test-fused            # fused permute + partial-eval kernel
make test-ajtai            # Ajtai commitment (CPU + ChaCha8 + GPU dense/batched/sparse)
```

### What the tests cover

- **`almost_field_test`** — constants, golden vectors (e.g. `(2^32)^2 ≡ c`,
  `(-1)^2 = 1`, `c·2^32 ≡ 2^37 + 31`), 128-bit reduction stress (200k cases
  including corner combinations) against a `__uint128_t` reference,
  canonicalize idempotence, add/sub/mul/square commutativity, associativity,
  distributivity, identity elements, `a + (-a) = 0`, `2·halve(a) = a`,
  `a·a^(-1) = 1`, `agl_mul_by_3` vs `agl_mul(_, 3)`, Fermat
  `a^(P-1) = 1`.
- **`almost_extension_test`** — verifies `Legendre(3) = -1`, golden vectors
  for the Karatsuba mul/square with `W = 3`, batch ops vs CPU reference,
  `a·a^(-1) = 1`, `Frobenius² = id`, `Frobenius = conj`, `norm(a) = a·conj(a)`,
  embedding/extraction round-trip, distributivity.
- **`almost_eq_lagrange_test`** — exercises both the DP and WHT GPU
  implementations across `log_n ∈ {4, 8, 12, 16}` for both base field and
  Ext2, comparing every output to the CPU product formula.
- **`almost_partial_eval_test`** — base-field and base→Ext2 partial
  evaluation across a grid of `(log_n, m)` configurations, verifying every
  output position against a CPU reference.
- **`almost_sumcheck_test`** — round-message and fold kernels at various
  degrees `d ∈ {1..4}` and sizes `log_n ∈ {6, 10}`, both base field and
  Ext2.
- **`almost_fused_test`** — the fused permute + partial-eval kernel with
  random base-field `evals`, random Ext2 `eq_table`, and (currently)
  identity permutation LUTs (which still exercises the LUT-shared-memory
  path and the warp-level reduction); CPU reference uses the same lookup
  formula.

## Performance note

Field multiplication is roughly 1.5–2× slower than Goldilocks because the
2-pass Solinas reduction replaces the cheaper 1-step reduction enabled by
`2^96 ≡ -1` and the 32-bit wrap constant. Add/sub/neg are essentially
unchanged. Ext2 multiplication is unaffected beyond that (Karatsuba still
uses 3 base muls; `mul_by_3` is two adds, cheaper than `mul_by_7` was).

## Not included

- **Poseidon2 / Monolith / Challenger.** Their round constants are
  field-specific (Plonky3 generates them from a Grain-LFSR procedure tied to
  the prime). Carrying them over verbatim from Goldilocks would silently
  produce a different hash; regenerating them is out of scope for this
  field-arithmetic port.
- **Basefold.** Depends on Poseidon2 for Merkle commitments.
- **Ext5.** `gcd(5, P-1) = 1` makes `X^5 - W` reducible for any `W`; Ext5
  would need a different irreducible quintic, which is a structural change,
  not a constant swap.

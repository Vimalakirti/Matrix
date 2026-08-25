# GPU/CPU Optimization Opportunities for zk-torch-3

Based on analysis of goldilocks.md field arithmetic techniques vs current implementation.

## Priority Ranking (Revised after memory traffic analysis)

**Key finding:** Opening proofs consume **81-82%** of prove time (GPT-2: 2.45s/3.04s, BERT: 4.63s/5.66s).
The sumcheck phase within each opening proof round runs **5 separate kernels** (2×eval, 2×interp, 1×product)
that read/write the same eq and bh data through global memory multiple times.
**Total traffic: 544 bytes per pair across 5 kernels; a fused kernel needs only 192 bytes = 65% reduction.**

**Karatsuba ext2_mul (DONE):** Implemented but had negligible impact — confirmed kernels are memory-bound, not compute-bound.

| # | Optimization | Expected Speedup | Effort | Affected Kernels | Rationale |
|---|---|---|---|---|---|
| **1** | **Fused eval+interp+product kernel** | **30-50% on opening proofs (25-40% total prove)** | **Medium** | **open_ext2 sumcheck loop** | **65% less global memory traffic; 5 launches → 1; data stays in registers** |
| 2 | GPU-side product reduction | 2-5% on opening proofs | Low | sumcheck_product_ext2 | Eliminates 3×D→H stalls per round; tiny second-stage GPU reduction instead |
| 3 | Fused Merkle layers | 10-15% on commit | Medium | poseidon2 merkle | Eliminates kernel launch overhead + global mem round-trips between hash layers |
| 4 | Async merkle overlap | 5-10% on opening proofs | Medium | open_ext2 loop | Overlap merkle tree build (stream 1) with next round's sumcheck (stream 2) |
| 5 | CPU fallback for small rounds | 3-5% on opening proofs | Low | open_ext2 late rounds | Rounds with pair_count < 4096 waste GPU launch overhead |
| 6 | Lazy reduction in fold | 2-5% on fold kernels | Low | basefold fold, partial_eval | Removes conditional branches per element |
| 7 | Warp shuffles for reductions | 1-3% on reductions | Low | product, sumcheck round msg | Eliminates shared-mem round-trips in final 5 reduction stages |

**Removed (confirmed no impact):** Shoup constant mul, SoA Ext2 layout, Karatsuba (already done, memory-bound)

---

## 1. SoA Ext2 Layout (Highest Priority)

**Problem:** Ext2 elements stored as interleaved (c0, c1) pairs (AoS). When 32 threads in a warp load Ext2 elements, they access addresses 0,16,32,... (stride-2 in u64), wasting half the memory transaction bandwidth and causing 2× more cache lines to be fetched.

**Solution:** Store c0 and c1 components in separate contiguous arrays. Thread k reads c0[k] and c1[k] from two coalesced streams.

**Impact:** Sumcheck round-message kernel, sumcheck fold kernel, partial_eval kernel, eq_dp kernel — these dominate proving time.

**Affected files:**
- `cuda/sumcheck_prover.cuh` — round message + fold kernels
- `cuda/partial_eval.cuh` — partial eval layers
- `cuda/eq_lagrange.cuh` — eq polynomial computation
- `cuda/extension.cuh` — Ext2 arithmetic device functions
- `goldilocks-cuda-rs/src/sumcheck_prover.rs` — Rust FFI for sumcheck
- `goldilocks-cuda-rs/src/partial_eval.rs` — Rust FFI for partial eval
- `goldilocks-cuda-rs/src/basefold.rs` — uses eq and partial eval
- `zk-torch-3/src/sumcheck/gpu_prover.rs` — packs polynomials for GPU

---

## 2. Shoup-Style Constant Multiplication

**Problem:** Basefold fold, eq evaluation, and partial eval kernels multiply by precomputed constants (table entries, challenge-derived weights) using full `gl_mul` + `reduce128`. When one operand is known ahead of time, half the reduction work is redundant.

**Solution:** For each constant w, precompute w' = floor(w * 2^64 / p). Then `(a * w) mod p` ≈ `a*w - mulhi(a, w') * p`, needing only one correction step instead of the full reduce128 decomposition.

**Impact:** Every basefold fold round, every eq_dp layer, every partial_eval round where the challenge point is fixed for the layer.

---

## 3. Split-Limb Shared Memory (2×u32)

**Problem:** GPUs have 32-bit shared memory banks. Accessing `uint64_t shared[]` causes systematic 2-way bank conflicts when consecutive threads access consecutive u64 elements (thread k touches banks 2k and 2k+1, thread k+16 touches banks 2k and 2k+1 — conflict).

**Solution:** Store shared memory as two u32 arrays (lo[] and hi[]), or pad the u64 array with a dummy u32 every 32 entries.

**Impact:** Sumcheck round-message reduction, basefold reduction, dot-product kernels.

---

## 4. Stockham Basefold Encoding

**Problem:** In-place butterfly encoding (`foldable_domain_layer`) creates strided access patterns in later stages. Stage k has stride 2^k — for k > 5, this exceeds cache line size and causes uncoalesced global memory access.

**Solution:** Stockham autosort: out-of-place passes alternating between two buffers. Each pass reads and writes contiguously. Costs 2× memory but gives consistently coalesced access.

**Impact:** Basefold commit phase (encoding).

---

## 5. Lazy Reduction in Fold

**Problem:** Sumcheck fold computes `p[j] = p[2j] + r * (p[2j+1] - p[2j])`. Each sub/mul/add does its own overflow correction (conditional branch on carry/borrow + epsilon adjustment).

**Solution:** Use `add_no_canonicalize` / `sub_no_canonicalize` for the sub and final add, deferring full canonicalization. The intermediate value stays in [0, 2^65) which is safe for the next round's operations if handled carefully.

**Impact:** Sumcheck fold kernel, partial_eval kernel.

---

## 6. Fused Merkle Layers

**Problem:** Merkle tree built one layer at a time with separate kernel launches. Each launch has overhead (~5-10μs) and the intermediate hash values are written to global memory then re-read.

**Solution:** Fuse 2-4 bottom layers into one kernel. A single thread block computes leaf hashes then immediately compresses up 2-4 levels using registers/shared memory, writing only the top-level result to global.

**Impact:** Basefold commit phase (Merkle tree construction).

---

## 7. Warp Shuffles for Final Reduction Stages

**Problem:** Tree reduction in shared memory for the last 5 stages (32→1 elements) requires `__syncthreads()` between stages and shared memory round-trips.

**Solution:** Replace with `__shfl_down_sync(0xFFFFFFFF, val, offset)` for offsets 16, 8, 4, 2, 1. Eliminates shared memory access and synchronization for intra-warp reduction.

**Impact:** All reduction kernels (sum, dot product, sumcheck round message).

---

## 8. Shift-Based Twiddles for Early Encoding Stages

**Problem:** First few butterfly stages of basefold encoding use twiddle factors that are powers of 2 (since 8 is a 64th root of unity in Goldilocks). Full multiplication is used regardless.

**Solution:** For stages where the twiddle w = 2^k, replace `gl_mul(a, w)` with `(a << k)` + epsilon correction. This saves the `mulhi` instruction entirely.

**Impact:** First ~6 stages of basefold encoding only.

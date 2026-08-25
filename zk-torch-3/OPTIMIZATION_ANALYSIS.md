# Optimization Analysis: Einsum Proving Bottleneck

## Current Performance (4x A100-80GB, 4 Partitions)

| Model | Layers | Baseline Prove | Multi-GPU Prove | Speedup |
|-------|--------|---------------|-----------------|---------|
| GPT-2 | 12 | 10.0s | 8.7s | 1.15x |
| BERT | 24 | 21.6s | 14.6s | 1.48x |
| GPT-J | 28 | 299.9s | 97.7s | 3.07x |
| LLaMA | 32 | 504.6s | 143.7s | 3.51x |

## Where Time Goes

Einsum accounts for **90-92%** of all prove time. Everything else (Add, Sub, ScaleDown, RMSNorm, lookups, openings) is negligible.

```
=== LLaMA 1-layer prove breakdown ===
  Einsum           18.4s   (92%)
  All other nodes   0.01s  ( 0%)
  Lookups+openings  1.6s   ( 8%)
```

## The Hidden Bottleneck: GPU Sumcheck is Never Used

The GPU sumcheck threshold is `total_rounds > 14`. But for all transformer Einsums with batch=1, seq=1, `einsum_helper` decomposes the variables such that most dimensions are handled by **degree-one partial evaluation**, leaving only 12-14 sumcheck rounds:

| Einsum | Weight Shape | Poly Size | Sumcheck Rounds |
|--------|-------------|-----------|----------------|
| Logits | [4096, 32000] | 2^27 (128M elements) | 12 |
| Feedforward | [4096, 11008] | 2^26 (64M elements) | 12-14 |
| Attention proj | [4096, 4096] | 2^24 (16M elements) | 12 |

**Every Einsum takes the CPU path.** The polynomial has up to 128M elements, but only 12-14 get used as sumcheck rounds. The remaining 12-15 variables are folded via `partial_eval_ext2_cpu` on the CPU.

## Profiled Per-Einsum Breakdown (CPU Path)

| Einsum | Poly Size | CPU Permute | CPU Partial Eval | CPU Sumcheck | Total |
|--------|-----------|-------------|-----------------|-------------|-------|
| Logits `[4096,32000]` | 128M | 1857ms (26%) | 5213ms (72%) | **1.3ms** (0.02%) | 7248ms |
| FFwd up `[4096,11008]` | 64M | 1013ms (25%) | 2872ms (72%) | **4.9ms** (0.1%) | 3986ms |
| Attn proj `[4096,4096]` | 16M | 241ms (25%) | 706ms (73%) | **1.3ms** (0.1%) | 966ms |

**The actual sumcheck takes < 5ms. CPU data preparation takes 99.9% of the time.**

- `partial_eval_ext2_cpu`: Folds 12-15 degree-one challenges by iterating over 2^n elements, converting base field to Ext2, and doing linear interpolation. **72-73% of Einsum time.**
- `permute_evals_by_ranges`: Reorders 2^n elements via LUT-based random access. **25-26% of Einsum time.**

## Proposed Optimization: GPU Partial Evaluation in CPU Sumcheck Path

The GPU kernel `partial_eval_ext2_device_u64` already exists and does exactly what `partial_eval_ext2_cpu` does, but runs on the GPU. It's currently only called in the GPU sumcheck path (total_rounds > 14), which is never reached.

### Change

For input polynomials with n > 16 in the CPU sumcheck path:

```
Current flow (CPU-only):
  CPU permute (1000ms) → CPU partial_eval (2800ms) → CPU sumcheck (5ms)

Proposed flow (hybrid GPU+CPU):
  CPU permute (1000ms) → GPU upload (16ms) → GPU partial_eval (5ms) →
  GPU download (0.1ms) → CPU sumcheck (5ms)
```

### Expected Per-Einsum Improvement

| Einsum | Current | Proposed | Speedup |
|--------|---------|----------|---------|
| Logits | 7248ms | ~530ms | 13.7x |
| Feedforward | 3986ms | ~290ms | 13.7x |
| Attention proj | 966ms | ~260ms | 3.7x |

The CPU permutation (~25% of time) remains; the GPU partial eval replaces the CPU partial eval (~73% of time) with a ~5ms GPU operation plus ~16ms transfer.

### Expected Model-Level Impact

| Model | Current Prove (4P/4GPU) | Estimated After | Improvement |
|-------|------------------------|-----------------|-------------|
| GPT-J 28L | 97.7s | ~20-30s | 3-5x |
| LLaMA 32L | 143.7s | ~25-40s | 3.5-5.7x |

Combined with the existing 3.5x from multi-GPU partitioning, the total speedup from baseline would be **12-20x**.

### Further Opportunity: GPU Permutation

If the CPU permutation (~25% of remaining time) can also be moved to GPU:
- Upload original (unpermuted) data once
- Permute on GPU via gather kernel
- Run partial_eval directly on GPU

This would reduce per-Einsum time to ~50-100ms, yielding another 3-5x on top.

Note: A previous attempt at GPU bit permutation (gather-based `bit_permute_gl_ffi`) was slower than CPU due to uncoalesced memory access. A new approach using shared-memory staging or warp-shuffle could improve this.

## Implementation Complexity

**Low.** The core change is ~30 lines in `einsum.rs`:
1. Add a threshold check: `if n > 16 && m > 0`
2. Upload permuted data: `DeviceBuffer::from_slice`
3. Call existing kernel: `partial_eval_ext2_device_u64`
4. Download result: `d_output.to_vec()`
5. Continue with existing CPU sumcheck

No new CUDA kernels needed. No changes to verification. No changes to the proof format.

# zk-torch-3 Optimization Guide

A comprehensive, in-depth document covering every optimization applied in the zk-torch-3 GPU-native ZKML proof system. Each section includes mathematical background, implementation details with real code from the codebase, data flow descriptions, and measured results.

## Table of Contents

1. [GPU Sumcheck Prover](#1-gpu-sumcheck-prover)
2. [Fused GPU Permute + Partial Eval](#2-fused-gpu-permute--partial-eval)
3. [GPU Opening Proof Pre-allocation](#3-gpu-opening-proof-pre-allocation)
4. [Concurrent CPU/GPU Opening Proofs](#4-concurrent-cpugpu-opening-proofs)
5. [Opening Task Deduplication](#5-opening-task-deduplication)
6. [Per-Device Table Caching](#6-per-device-table-caching)
7. [Per-Thread CUDA Streams](#7-per-thread-cuda-streams)
8. [Multi-GPU Partition-Aware Proving](#8-multi-gpu-partition-aware-proving)
9. [Dense Bit Polynomial Range Checks](#9-dense-bit-polynomial-range-checks)
10. [LUT-Based Permutation](#10-lut-based-permutation)
11. [Parallel Einsum Computation](#11-parallel-einsum-computation)
12. [Parallel Forward Pass](#12-parallel-forward-pass)
13. [Parallel Lagrange Basis Evaluation](#13-parallel-lagrange-basis-evaluation)
14. [GPU Partial Eval in CPU Sumcheck Path](#14-gpu-partial-eval-in-cpu-sumcheck-path)
15. [Sparse Polynomial Evaluation](#15-sparse-polynomial-evaluation)
16. [Conv2D Alpha-Power Factorization](#16-conv2d-alpha-power-factorization)
17. [Batch Query Extraction](#17-batch-query-extraction)
18. [Zero-Rebuild Opening Proofs](#18-zero-rebuild-opening-proofs)
19. [Basefold Table Clone Optimization](#19-basefold-table-clone-optimization)
20. [CPU Poseidon2 for Merkle Verification](#20-cpu-poseidon2-for-merkle-verification)
21. [Threshold Configuration Reference](#21-threshold-configuration-reference)

---

## 1. GPU Sumcheck Prover

**Files**: `cuda/sumcheck_prover.cuh`, `goldilocks-cuda-rs/src/sumcheck_prover.rs`, `zk-torch-3/src/sumcheck/gpu_prover.rs`

### Mathematical Background

The sumcheck protocol reduces a claim about a multilinear polynomial sum to a point evaluation. Given ℓ polynomials $p_1, \ldots, p_\ell$ over $n$ variables, we want to prove:

$$S = \sum_{x \in \{0,1\}^n} \prod_{i=1}^{\ell} p_i(x)$$

In each round $m = 0, \ldots, n-1$, the prover computes the *round polynomial*:

$$g_m(X) = \sum_{y \in \{0,1\}^{n-m-1}} \prod_{i=1}^{\ell} p_i(r_0, \ldots, r_{m-1}, X, y)$$

This is a univariate polynomial of degree $\ell$ in $X$. The prover sends $g_m(0), g_m(1), \ldots, g_m(\ell)$ (ℓ+1 evaluations). The verifier checks $g_m(0) + g_m(1) = S_m$ and samples a random challenge $r_m$.

After the challenge, each polynomial is *folded* (partially evaluated at $r_m$):

$$p_i'(y) = p_i(r_0, \ldots, r_m, y) = p_i(\text{even half}) + r_m \cdot (p_i(\text{odd half}) - p_i(\text{even half}))$$

### GPU Architecture

The GPU prover exploits two levels of parallelism:

1. **Round message computation**: For each evaluation point $c \in \{0, 1, \ldots, \ell\}$, sum $\prod_i p_i(c, y)$ over all $y \in \{0,1\}^{n-m-1}$. There are $2^{n-m-1}$ independent terms — perfect for GPU parallelism.

2. **Folding**: After each challenge $r_m$, fold all ℓ polynomials in parallel. Each element $j$ computes $p[j] = p[2j] + r_m \cdot (p[2j+1] - p[2j])$, which are $\ell \cdot 2^{n-m-1}$ independent operations.

### CUDA Kernels (`cuda/sumcheck_prover.cuh`)

**Round message kernel** — computes $g_m(c)$ for one evaluation point $c$:
```cuda
__global__ void sumcheck_round_message_ext2_kernel(
    const uint64_t* d_polys,  // all ℓ polynomials packed contiguously
    int num_poly, int half_n, int stride,
    uint64_t eval_c0, uint64_t eval_c1,  // evaluation point c as Ext2
    uint64_t* d_out  // partial sums per block
) {
    __shared__ uint64_t s_data[BLOCK_SIZE * 2];  // shared memory for reduction

    uint64_t local_c0 = 0, local_c1 = 0;
    // Grid-stride loop: each thread handles multiple y values
    for (int y = tid; y < half_n; y += gridDim.x * blockDim.x) {
        // For each polynomial i, interpolate p_i(c, y) from p_i(0,y) and p_i(1,y)
        // Then multiply all ℓ interpolated values together
        uint64_t prod_c0 = 1, prod_c1 = 0;  // identity in Ext2
        for (int i = 0; i < num_poly; i++) {
            uint64_t v0_c0 = d_polys[i*stride + 2*y];      // p_i(0, y)
            uint64_t v0_c1 = d_polys[i*stride + 2*y + 1];
            uint64_t v1_c0 = d_polys[i*stride + 2*(y+half_n)];  // p_i(1, y)
            uint64_t v1_c1 = d_polys[i*stride + 2*(y+half_n) + 1];
            // interp = v0 + c * (v1 - v0)
            // prod *= interp
        }
        local_c0 += prod_c0;
        local_c1 += prod_c1;
    }

    // Block-level tree reduction in shared memory
    s_data[threadIdx.x * 2] = local_c0;
    s_data[threadIdx.x * 2 + 1] = local_c1;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x*2] = gl_add(s_data[threadIdx.x*2], s_data[(threadIdx.x+s)*2]);
            // ... same for c1
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x * 2] = s_data[0];
        d_out[blockIdx.x * 2 + 1] = s_data[1];
    }
}
```

**Fold kernel** — uses separate input/output buffers to avoid race conditions:
```cuda
__global__ void sumcheck_fold_ext2_kernel(
    const uint64_t* d_input,   // read from here
    uint64_t* d_output,        // write to here (separate buffer!)
    int num_poly, int half_n, int stride,
    uint64_t ch_c0, uint64_t ch_c1  // challenge as Ext2
) {
    for (int y = tid; y < half_n; y += gridDim.x * blockDim.x) {
        for (int i = 0; i < num_poly; i++) {
            int base = i * stride;
            // v0 = d_input[base + 2*y], v1 = d_input[base + 2*(y + half_n)]
            // d_output[base + 2*y] = v0 + ch * (v1 - v0)   (Ext2 arithmetic)
        }
    }
}
```

### Critical Bug: Cross-Warp Race Condition

The original implementation used in-place folding (same buffer for input and output). Thread handling element $y=k$ writes to position $k$, while thread handling element $y=k/2$ reads from position $k$ (as the "odd half"). When these threads are in different warps, the write may complete before the read, producing incorrect results.

**Why small tests passed**: For arrays with ≤ 32 elements, all threads fit in a single warp and execute in lockstep (SIMT), so reads always precede writes.

**Fix**: Separate input/output buffers with a swap after each round:
```rust
// In GpuSumcheckStateExt2:
let mut buf_a = DeviceBuffer::new(total_size)?;  // current input
let mut buf_b = DeviceBuffer::new(total_size)?;  // fold output
// After each round:
std::mem::swap(&mut buf_a, &mut buf_b);
```

### Polynomial Packing

All ℓ input polynomials are packed contiguously in a single GPU buffer with `stride = original_poly_size`:
```
[p0[0], p0[1], ..., p0[2^n-1], p1[0], p1[1], ..., p1[2^n-1], ...]
```
This is uploaded once and accessed in-place for all rounds. The stride shrinks by half each round as polynomials are folded.

### Decision Logic

```rust
// src/basicblock/einsum.rs
fn gpu_sumcheck_threshold() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_SUMCHECK_THRESHOLD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(14)
    })
}

// In Einsum::prove():
if total_rounds > gpu_sumcheck_threshold() {
    // GPU path: GpuLinearSumcheckProver
} else {
    // CPU path: CpuLinearSumcheckProverExt2
}
```

The threshold of 14 was empirically tuned on A100 GPUs. Below 14 rounds, the polynomial has fewer than 2^14 = 16,384 elements per fold — too little work to amortize GPU kernel launch overhead (~10μs per launch).

---

## 2. Fused GPU Permute + Partial Eval

**Files**: `cuda/fused_permute_peval.cuh`, `goldilocks-cuda-rs/src/partial_eval.rs`, `zk-torch-3/src/basicblock/einsum.rs`

### Mathematical Background

Einsum's prove phase begins by reordering polynomial variables so *free variables* (those appearing in the output) come first, and *summation variables* come last. For a polynomial $p(x_0, x_1, \ldots, x_{n-1})$ with variable reordering $\sigma$, we compute:

$$p'(x_{\sigma(0)}, x_{\sigma(1)}, \ldots, x_{\sigma(n-1)})$$

After reordering, the first $m$ summation variables are partially evaluated at challenge points $r_0, \ldots, r_{m-1}$ using the Lagrange basis:

$$q(j) = \sum_{b \in \{0,1\}^m} p'(j, b) \cdot \text{eq}(r, b)$$

where $j$ ranges over the remaining $n - m$ free variable assignments and $\text{eq}(r, b) = \prod_{i=0}^{m-1} (r_i b_i + (1-r_i)(1-b_i))$.

### Naive Approach: 3 Steps

```
Step 1: CPU permute        — O(2^n) memory reads/writes, random access
Step 2: CPU→GPU upload     — O(2^n) data transfer
Step 3: GPU partial_eval   — O(2^n) per round, m rounds
```

For a weight matrix with n=27 (128M elements), Step 1 takes ~200ms on CPU due to random memory access patterns.

### Fused Kernel: 1 Step

The fused kernel computes permutation + partial evaluation in a single GPU pass:

```cuda
// cuda/fused_permute_peval.cuh
__global__ void fused_permute_partial_eval_kernel(
    const uint64_t* evals,          // input polynomial (unpermuted)
    const uint64_t* eq_table,       // precomputed eq(r, b) for b in {0,1}^m
    uint64_t* output,               // output: 2^(n-m) Ext2 elements
    const uint32_t* lo_lut,         // shared mem LUT for low bits
    const uint32_t* hi_lut,         // shared mem LUT for high bits
    int n, int m, int half, int lo_mask
) {
    // Load LUTs into shared memory
    extern __shared__ uint32_t s_mem[];
    uint32_t* s_lo = s_mem;
    uint32_t* s_hi = s_mem + (1 << half);

    // Cooperative load of LUTs
    for (int i = threadIdx.x; i < (1 << half); i += blockDim.x)
        s_lo[i] = lo_lut[i];
    for (int i = threadIdx.x; i < (1 << (n - half)); i += blockDim.x)
        s_hi[i] = hi_lut[i];
    __syncthreads();

    // Grid-stride loop over output positions j
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < (1 << (n-m));
         j += gridDim.x * blockDim.x) {
        uint64_t acc_c0 = 0, acc_c1 = 0;

        // Sum over all 2^m summation assignments
        for (int b = 0; b < (1 << m); b++) {
            // Compute permuted index using split-LUT
            int combined = b + (j << m);  // original index before perm
            int perm_idx = s_lo[combined & lo_mask] | s_hi[combined >> half];

            // Look up evaluation and eq weight
            uint64_t val = evals[perm_idx];
            uint64_t eq_c0 = eq_table[b * 2];
            uint64_t eq_c1 = eq_table[b * 2 + 1];

            // Accumulate: acc += val * eq (base × Ext2 multiplication)
            acc_c0 = gl_add(acc_c0, gl_mul(val, eq_c0));
            acc_c1 = gl_add(acc_c1, gl_mul(val, eq_c1));
        }

        output[j * 2] = acc_c0;
        output[j * 2 + 1] = acc_c1;
    }
}
```

### Split-LUT Design

The permutation $\sigma$ maps an $n$-bit index to another $n$-bit index. Direct computation requires $O(n)$ bit extractions per element. The split-LUT approach:

1. Split the $n$-bit index into low `half` bits and high `n - half` bits
2. Precompute `lo_lut[2^half]`: for each low-bits pattern, compute the full permuted bits contributed by those low bits
3. Precompute `hi_lut[2^(n-half)]`: same for high bits
4. At runtime: `perm(idx) = lo_lut[idx & lo_mask] | hi_lut[idx >> half]`

This works because bit permutations are separable: each output bit depends on exactly one input bit, so the contribution from low and high halves can be OR'd together.

**Shared memory usage**: For n=27, half=13: `lo_lut` = 8K entries × 4B = 32KB, `hi_lut` = 16K entries × 4B = 64KB. Total ~96KB, within A100's 164KB shared memory limit. Uses `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` for large allocations.

### Data Flow

```
Without fused kernel:
  CPU: evals[2^27] --permute--> perm_evals[2^27] --upload--> GPU --partial_eval--> result[2^(27-m)]
  Time: ~200ms (permute) + ~50ms (upload) + ~30ms (partial_eval) = ~280ms

With fused kernel:
  GPU: evals[2^27] already on device --fused_kernel--> result[2^(27-m)]
  Time: ~60ms (single kernel)
```

### Decision Tree in Einsum

```rust
// src/basicblock/einsum.rs, prepare_input_poly()
fn prepare_input_poly(&self, ...) -> Vec<GoldilocksExt2> {
    let needs_permute = !is_identity_perm(&perm_map);
    let m = self.summation_round;  // number of summation variables

    if needs_permute && m > 0 && n > gpu_fused_threshold() {
        // Path A: Fused GPU kernel (single pass)
        fused_permute_partial_eval(evals, &eq_table, &perm_map, n, m)
    } else if needs_permute {
        // Path B: CPU permute (LUT-based) + optional GPU partial_eval
        let permuted = permute_evals_by_ranges(evals, n, &self.permute_vecs[i]);
        if m > 0 && n > gpu_partial_eval_threshold() {
            gpu_partial_eval_ext2(&permuted, &challenges)
        } else {
            cpu_partial_eval_ext2(&permuted, &challenges)
        }
    } else {
        // Path C: No permutation needed
        if m > 0 && n > gpu_partial_eval_threshold() {
            gpu_partial_eval_ext2(evals, &challenges)
        } else {
            cpu_partial_eval_ext2(evals, &challenges)
        }
    }
}
```

### Results (single GPU, vs separate CPU permute + GPU partial_eval)

| Model | Weight matrix size | Before | After | Speedup |
|-------|-------------------|--------|-------|---------|
| GPT-2 12L | n ≤ 20 | 5.48s | 5.15s | 1.06x |
| BERT 24L | n ≤ 20 | 9.82s | 8.85s | 1.11x |
| LLaMA 4L | n = 24-27 | 5.87s | 3.72s | 1.58x |
| LLaMA 8L | n = 24-27 | 10.03s | 5.91s | 1.70x |

Larger weight matrices (n=27 in LLaMA) benefit most because the CPU permutation of 128M elements is the bottleneck eliminated by the fused kernel.

---

## 3. GPU Opening Proof Pre-allocation

**Files**: `goldilocks-cuda-rs/src/basefold.rs` (`open_ext2`, `sumcheck_product_and_reduce_ext2_reuse`)

### Mathematical Background

A Basefold opening proof for a polynomial $p$ of $n$ variables at point $r$ consists of:

1. **Inner-product sumcheck**: Prove $\langle \text{eq}(r, \cdot), p(\cdot) \rangle = p(r)$ via sumcheck
2. **Codeword folding**: After each sumcheck round, fold the codeword (encoded polynomial) using the random challenge
3. **Merkle authentication**: Commit to each folded codeword via Poseidon2 Merkle tree
4. **Query proofs**: At random query indices, provide Merkle auth paths for fold consistency verification

This requires $n - 1$ sumcheck rounds, each producing one oracle (3 field elements), one folded codeword, and one Merkle tree.

### Problem: cudaMalloc Contention

Before optimization, `open_ext2()` allocated ~8 GPU buffers per round:

| Buffer | Purpose | Size (n=22, log_rate=3) |
|--------|---------|------------------------|
| `d_eq_h` | Sumcheck eval output for eq | 2^21 × 16B = 32MB |
| `d_bh_h` | Sumcheck eval output for bh | 2^21 × 16B = 32MB |
| `pc0` | Block partial sum | 256 × 16B = 4KB |
| `pc1` | Block partial sum | 256 × 16B = 4KB |
| `pc2` | Block partial sum | 256 × 16B = 4KB |
| `d_folded` | Fold output codeword | 2^24 × 16B = 256MB |
| Merkle tree | Poseidon2 hash tree | ~2^24 × 32B = 512MB |
| `clone_on_device` | D2D copy for query extraction | 2^24 × 16B = 256MB |

For n=22 (20 inner rounds): ~160 `cudaMalloc` + ~100 `cudaFree`. With 12 rayon threads per GPU competing for CUDA's global memory allocator, severe lock contention occurs.

### Four Optimizations

**Optimization 1: Pre-allocate reduction buffers (pc0/pc1/pc2)**

```rust
// goldilocks-cuda-rs/src/basefold.rs
pub fn sumcheck_product_and_reduce_ext2_reuse(
    eq: &DeviceBuffer<u64>,
    bh: &DeviceBuffer<u64>,
    pair_count: usize,
    pc0: &mut DeviceBuffer<u64>,  // pre-allocated, reused across rounds
    pc1: &mut DeviceBuffer<u64>,
    pc2: &mut DeviceBuffer<u64>,
) -> Result<SumcheckOracle<GoldilocksExt2>> {
    let num_blocks = ((pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
    // Launch kernel into pre-allocated buffers (no cudaMalloc)
    BasefoldBatch::sumcheck_product_ext2(eq, bh, pc0, pc1, pc2, pair_count, num_blocks)?;
    // Download block partials and reduce on host
    let raw0 = pc0.read_range(0, num_blocks * 2)?;
    let raw1 = pc1.read_range(0, num_blocks * 2)?;
    let raw2 = pc2.read_range(0, num_blocks * 2)?;
    // Host reduction: sum all block partials
    let (c0, c1, c2) = host_reduce(&raw0, &raw1, &raw2, num_blocks);
    Ok(SumcheckOracle { c0, c1, c2 })
}
```

Saves 3 × (num_rounds - 2) alloc+free ≈ 60 calls. Each buffer is only 4KB (256 blocks × 2 u64), allocated once at max size.

**Optimization 2: Double-buffer eq/bh eval**

Instead of allocating a new output buffer each round, pre-allocate two buffers and swap:

```rust
// Pre-allocate at max size (round 0 pair_count)
let mut d_eq_a = DeviceBuffer::<u64>::new(initial_pair_count * 2)?;
let mut d_eq_b = DeviceBuffer::<u64>::new(initial_pair_count * 2)?;
let mut eq_is_a = true;

// In the round loop:
if eq_is_a {
    BasefoldBatch::sumcheck_eval_ext2(&d_eq_a, challenge, &mut d_eq_b, pair_count)?;
} else {
    BasefoldBatch::sumcheck_eval_ext2(&d_eq_b, challenge, &mut d_eq_a, pair_count)?;
}
eq_is_a = !eq_is_a;  // swap active buffer
```

The buffer shrinks each round (pair_count halves) but is pre-allocated at max — we just use a prefix. Same pattern for `d_bh_a`/`d_bh_b`. Saves 2 × (num_rounds - 2) alloc+free ≈ 40 calls.

**Optimization 3: Eliminate clone_on_device for folded codewords**

Before: After folding into `d_folded`, clone it (D2D copy) to save for query extraction, then use the original as next input.

After: Use a `Vec<DeviceBuffer>` with `split_at_mut` for zero-copy split borrows:

```rust
// Pre-allocate all fold output buffers upfront
let mut fold_buffers: Vec<DeviceBuffer<u64>> = Vec::with_capacity(num_rounds);
fold_buffers.push(initial_codeword);  // round 0: the initial codeword
for r in 1..num_rounds {
    let cw_size = 1 << (num_vars + log_rate - r);
    fold_buffers.push(DeviceBuffer::<u64>::new(cw_size * 2)?);
}

// In the round loop:
let (prev_slice, cur_slice) = fold_buffers.split_at_mut(round);
let d_cw_input = &prev_slice[round - 1];   // borrow previous round's output
let d_cw_output = &mut cur_slice[0];        // mutable borrow current round's buffer
BasefoldBatch::fold_ext2(d_cw_input, table_ptr, challenge, d_cw_output, cw_pair_count)?;
```

After the loop, `fold_buffers[0..num_rounds-1]` contains all folded codewords for query extraction — no cloning needed. Saves (num_rounds - 1) allocations + D2D copies ≈ 20 calls + significant bandwidth.

**Optimization 4: Pre-allocate fold output buffers**

Fold output sizes are deterministic and strictly decreasing:
- Round 1: $2^{n + \text{log\_rate} - 1}$
- Round 2: $2^{n + \text{log\_rate} - 2}$
- ...
- Round k: $2^{n + \text{log\_rate} - k}$

All buffers are allocated in one batch before the main loop (see Opt 3 code above), avoiding per-round `cudaMalloc` during the latency-critical inner loop.

### Combined Impact

Before (per opening, n=22):
```
160 cudaMalloc + 100 cudaFree + 20 D2D copies
With 12 threads × 4 GPUs = 48 threads competing for allocator
```

After:
```
~30 cudaMalloc at setup (fold buffers + eq/bh pairs + pc buffers)
0 cudaMalloc in round loop
0 D2D copies for query extraction
```

### Results

| Model | GPU Opens Before | GPU Opens After | Speedup |
|-------|-----------------|-----------------|---------|
| GPT-2 12L | 4.09s | 2.78s | 1.47x |
| BERT 24L | 8.43s | 5.02s | 1.68x |
| GPT-J 28L | 7.12s | 5.15s | 1.38x |
| LLaMA 32L | 9.85s | 7.12s | 1.38x |

---

## 4. Concurrent CPU/GPU Opening Proofs

**Files**: `zk-torch-3/src/dag/mod.rs`, `src/commit/cpu_basefold.rs`

### Problem

Opening proofs are 60-90% of prove time for large models. The task distribution is bimodal:
- **Small polynomials** (n ≤ 14): Too little parallelism for GPU. GPU kernel launch overhead (~10μs) dominates. CPU is faster.
- **Large polynomials** (n ≥ 15): Abundant parallelism. GPU is 10x+ faster than CPU.

Running all tasks on GPU wastes time on small polynomials. Running all on CPU wastes GPU compute on large polynomials.

### Solution: Dual Rayon Thread Pools

```rust
// src/dag/mod.rs — opening section of prove()
let cpu_pool_size = (rayon::current_num_threads() - gpu_pool_size).max(1);
let gpu_pool_size = (num_devices * 12).max(1);

let cpu_pool = rayon::ThreadPoolBuilder::new()
    .num_threads(cpu_pool_size).build().unwrap();
let gpu_pool = rayon::ThreadPoolBuilder::new()
    .num_threads(gpu_pool_size).build().unwrap();

std::thread::scope(|s| {
    // CPU handle: process small polynomial openings
    let cpu_handle = s.spawn(|| {
        cpu_pool.install(|| {
            cpu_tasks.par_iter().map(|&task_idx| {
                let (e, _, point) = &tasks[task_idx];
                let commitment = gpu_commitments[*e].as_ref()
                    .expect("GPU commitment missing for CPU opening");

                // Independent per-task transcript
                let mut t = Transcript::new(b"bf-open");
                t.append_ext2(b"", &master_seed);
                t.append_u64(b"", task_idx as u64);

                let proof = cpu_full_open_ext2(commitment, point, table, &mut t, num_queries);
                (task_idx, proof)
            }).collect::<Vec<_>>()
        })
    });

    // GPU handle: process large polynomial openings
    let gpu_handle = s.spawn(|| {
        gpu_pool.install(|| {
            gpu_tasks.par_iter().enumerate().map(|(gpu_idx, &task_idx)| {
                let (e, _, point) = &tasks[task_idx];
                let dev_id = gpu_idx as i32 % num_devices as i32;
                let _ = goldilocks_cuda::set_device(dev_id);

                let commitment = gpu_commitments[*e].as_ref().unwrap();
                let dev_table = &per_device_tables[dev_id as usize];

                let mut t = Transcript::new(b"bf-open");
                t.append_ext2(b"", &master_seed);
                t.append_u64(b"", task_idx as u64);

                let proof = commitment.open_ext2(&point, dev_table, &mut t, num_queries)
                    .expect("GPU open_ext2 failed");
                (task_idx, BasefoldOpeningProof { eval: proof.eval, gpu_proof: proof })
            }).collect::<Vec<_>>()
        })
    });

    let cpu_results = cpu_handle.join().unwrap();
    let gpu_results = gpu_handle.join().unwrap();
});
```

### Why Separate Pools

Using a single rayon pool for both CPU and GPU tasks causes serialization:
- GPU tasks hold a thread for the entire `open_ext2` duration (~300ms), during which the thread does both GPU kernel launches and CPU Merkle hashing
- If all pool threads are holding GPU tasks, CPU tasks starve
- If CPU tasks flood the pool, GPU utilization drops

Separate pools guarantee that CPU tasks always have dedicated threads, and GPU tasks always have dedicated threads with proper device affinity.

### GPU Pool Size: 12 per Device

Empirically tuned: GPU `open_ext2` is not purely GPU-bound — it includes CPU-side Merkle tree construction (~20% of time) and query extraction (~12%). Multiple threads per GPU enable:
1. One thread launching GPU kernels while another hashes Merkle tree
2. Per-thread CUDA streams (Section 7) enable concurrent kernel execution
3. More than 12 starts hitting cudaMalloc contention (even with pre-allocation, Merkle tree allocation isn't pre-allocated)

### Transcript Determinism

Each opening proof uses an independent transcript seeded with a master seed and task index. The master seed comes from the main proof transcript (ensuring it's unpredictable), and the task index ensures different openings use different challenges. The verifier reconstructs the same per-task transcripts by iterating tasks in the same deterministic order.

### CPU Opening Implementation (`cpu_full_open_ext2`)

```rust
// src/commit/cpu_basefold.rs
pub fn cpu_full_open_ext2(
    commitment: &BasefoldCommitment,
    point: &[GoldilocksExt2],
    table: &BasefoldTable,
    transcript: &mut Transcript,
    num_queries: usize,
) -> BasefoldOpeningProof {
    // 1. Download GPU commitment data to CPU
    let bh_evals = commitment.download_bh_evals();  // base-field codeword
    let bh_evals = normalize_to_canonical(&bh_evals);  // fix non-canonical GPU values

    // 2. Compute eq polynomial on CPU
    let eq = evaluate_lagrange_basis_ext2(point);

    // 3. Round 0: mixed mode (bh in F_p, eq in F_{p^2})
    let oracle0 = compute_oracle_mixed(&bh_evals, &eq, pair_count);
    let cw0 = cpu_fold_mixed(&codeword, table, challenge0);
    let tree0 = CpuMerkleTree::build_from_ext2_pairs(&cw0);

    // 4. Remaining rounds: pure Ext2
    for round in 1..num_rounds {
        let oracle = compute_oracle_ext2(&bh_r, &eq_r, pair_count);
        let cw_r = cpu_fold_ext2(&cw_prev, table, challenge_r);
        let tree_r = CpuMerkleTree::build_from_ext2_pairs(&cw_r);
    }

    // 5. Query proofs: Merkle auth paths at random indices
    let queries = extract_query_proofs(&trees, &codewords, query_indices);

    BasefoldOpeningProof { eval, gpu_proof: proof }
}
```

---

## 5. Opening Task Deduplication

**File**: `zk-torch-3/src/dag/mod.rs`

### Problem

The DAG backward pass can generate multiple claims on the same polynomial at the same evaluation point. This happens when:
- An edge feeds into multiple consumer nodes
- All consumers reduce to the same challenge point (e.g., via shared reducer challenges)

Each duplicate claim requires an identical opening proof — same polynomial, same point, same result.

### Solution

Before distributing tasks to CPU/GPU pools, group by `(edge_id, point_bytes)`:

```rust
let mut dedup_map: HashMap<(usize, Vec<u8>), usize> = HashMap::new();
let mut canonical_idx: Vec<usize> = Vec::new();

for (task_idx, (e, _, point)) in tasks.iter().enumerate() {
    // Serialize point to bytes for HashMap key
    let point_bytes: Vec<u8> = point.iter()
        .flat_map(|p| {
            let mut bytes = p.0[0].to_le_bytes().to_vec();
            bytes.extend_from_slice(&p.0[1].to_le_bytes());
            bytes
        })
        .collect();

    let canon = *dedup_map.entry((*e, point_bytes)).or_insert(task_idx);
    canonical_idx.push(canon);
}

// Only compute proofs for canonical tasks (unique_tasks)
let unique_tasks: Vec<usize> = dedup_map.values().copied().collect();

// After proof computation, distribute:
for (task_idx, _) in tasks.iter().enumerate() {
    let canon = canonical_idx[task_idx];
    if canon != task_idx {
        // Clone proof from canonical task
        edge_proofs[task_idx] = edge_proofs[canon].clone();
    }
}
```

The verifier mirrors the same deduplication logic, ensuring transcript consistency. Typically 3-5% of tasks are duplicates. For GPT-2 12L, this saves ~8 GPU opening proofs (~2.4s of GPU time).

---

## 6. Per-Device Table Caching

**File**: `zk-torch-3/src/commit/basefold.rs`

### Problem

The `BasefoldTable` contains precomputed folding coefficients needed by `open_ext2()`. Each GPU device needs its own copy (CUDA peer access isn't always available). Previously, tables were cloned to each device at the start of every `prove()` call:

```
4 devices × 0.8s per clone = 3.2s overhead per prove call
```

### Solution

Pre-clone once during `GpuCommitmentStore` initialization:

```rust
pub struct GpuCommitmentStore {
    pub table: BasefoldTable,           // device 0 (original)
    pub commitments: Vec<Option<BasefoldCommitment>>,
    pub device_ids: Vec<Option<i32>>,
    pub per_device_tables: Vec<BasefoldTable>,  // one per GPU device
}

impl GpuCommitmentStore {
    pub fn new(max_num_vars: usize, log_rate: usize, seed: u64, num_edges: usize) -> Self {
        let mut table = BasefoldTable::generate(max_num_vars, log_rate, num_rounds, seed);
        table.upload().expect("Upload failed");

        let num_devices = goldilocks_cuda::device_count().unwrap_or(1).max(1) as usize;
        let per_device_tables: Vec<BasefoldTable> = (0..num_devices).map(|d| {
            let _ = goldilocks_cuda::set_device(d as i32);
            let _ = goldilocks_cuda::synchronize();
            goldilocks_cuda::get_last_error();  // clear stale errors
            goldilocks_cuda::init_device().expect("init_device failed");
            table.clone_to_current_device().expect("table clone failed")
        }).collect();
        let _ = goldilocks_cuda::set_device(0);

        Self { table, commitments: ..., device_ids: ..., per_device_tables }
    }
}
```

Opening proofs reference `per_device_tables[dev_id]` directly — zero overhead per prove call.

---

## 7. Per-Thread CUDA Streams

**File**: `goldilocks-cuda-rs/build.rs`

### Problem

CUDA's default stream is shared across all host threads on a device. When 12 rayon threads issue kernels on the same GPU, they serialize on the shared default stream:

```
Thread 0: launch kernel A → wait → launch kernel B → wait ...
Thread 1:                    launch kernel C → wait ...
                             ^^^ must wait for Thread 0's kernel A to complete
```

### Solution

Compile with `--default-stream per-thread`:

```rust
// goldilocks-cuda-rs/build.rs
let status = Command::new(&nvcc)
    .args([
        "--default-stream", "per-thread",
        "-gencode", &format!("arch=compute_{compute},code=sm_{compute}"),
        "-O3",
        ...
    ])
    .status()?;
```

Now each host thread gets its own CUDA stream:

```
Thread 0: [kernel A] [kernel B] ...       (stream 0)
Thread 1: [kernel C] [kernel D] ...       (stream 1)
          ^^^ concurrent with Thread 0
```

### Impact

Without per-thread streams: 12 threads on one GPU ≈ 1.5x throughput vs 1 thread.
With per-thread streams: 12 threads ≈ 6-8x throughput (limited by memory bandwidth and allocator contention, not stream serialization).

---

## 8. Multi-GPU Partition-Aware Proving

**Files**: `zk-torch-3/src/dag/partition.rs`, `zk-torch-3/src/dag/mod.rs`

### Problem

The backward pass of the sumcheck proof traverses nodes in reverse topological order. This is inherently sequential — each node's proof depends on claims from its consumers. A 32-layer transformer with 4500+ nodes cannot utilize multiple GPUs.

### Solution: DAG Partitioning

Split the DAG at layer boundaries into independent partitions. Each partition can be proved on a different GPU with an independent transcript.

**Step 1: Select boundaries**

```rust
// src/dag/mod.rs — set_partition_boundaries()
fn set_partition_boundaries(&mut self, num_partitions: usize) {
    // Select evenly-spaced layer boundaries in topological order
    let total_levels = self.topo_levels.len();
    let step = total_levels / num_partitions;
    for k in 1..num_partitions {
        let boundary_level = k * step;
        // Mark all edges crossing this level as boundary edges
        for &node_id in &self.topo_levels[boundary_level] {
            for &edge_id in &self.nodes[node_id].inputs {
                self.boundary_edges.insert(edge_id);
            }
        }
    }
}
```

**Step 2: Force-commit boundary edges**

Boundary edges must be committed via Basefold PCS so the verifier can independently verify each partition:

```rust
fn should_commit(&self, edge_id: EdgeId) -> bool {
    self.boundary_edges.contains(&edge_id)
        || self.self_claim_edges.contains(&edge_id)
        || /* other commit conditions */
}
```

**Step 3: Partition prove with forked transcripts**

```rust
// src/dag/mod.rs — prove_parallel()
fn prove_parallel(&self, ..., transcript: &mut Transcript) {
    // 1. Generate output claims from main transcript
    // 2. Propagate claims to partition boundaries
    // 3. Fork transcript per partition
    let partition_proofs: Vec<PartitionProof> = (0..num_partitions)
        .into_par_iter()
        .map(|k| {
            let _ = goldilocks_cuda::set_device((k % num_devices) as i32);
            let mut t = transcript.fork(k);  // absorbs partition_id for domain separation
            self.prove_partition(k, &mut t, ...)
        })
        .collect();
    // 4. Merge boundary claims
    // 5. Prove lookups + opening proofs (shared across all partitions)
}
```

**Step 4: Transcript forking**

```rust
impl Transcript {
    pub fn fork(&self, partition_id: usize) -> Transcript {
        let mut forked = self.clone();
        forked.append_u64(b"partition", partition_id as u64);
        forked
    }
}
```

This ensures each partition uses different Fiat-Shamir challenges, maintaining soundness.

### Claim Routing Subtlety

After partitions produce boundary claims, claims must be routed to the correct partition by the *producer node's* partition (not the consumer's). Some output edges are consumed internally within the same partition:

```rust
for (edge_id, claim) in boundary_claims {
    let producer = self.producers[edge_id].unwrap();
    let partition = node_to_partition[producer];
    partition_claims[partition].push(claim);
}
```

---

## 9. Dense Bit Polynomial Range Checks

**Files**: `src/basicblock/scale.rs`, `src/basicblock/range.rs`, `src/dag/mod.rs`

### Mathematical Background

Range proofs verify that polynomial values lie in a valid range (e.g., $[0, 2^{20})$ for NonNegative). The original approach used a *selection polynomial* $S(x, y)$ where:
- $x \in \{0,1\}^n$ — input position
- $y \in \{0,1\}^t$ — table index ($t$ = table bits)
- $S(x, y) = 1$ if value at $x$ maps to table entry $y$

For NonNegative with 20-bit range: $t = 20$, auxiliary polynomial has $n + 20$ variables. A polynomial with $n = 17$ variables gets an auxiliary with 37 variables — 128 billion virtual entries.

### New Approach: Dense Bit Polynomial

Instead of a one-hot selection polynomial, decompose each value into its individual bits:

$$B(x, y) = \text{bit } y \text{ of value at position } x$$

where $y \in \{0,1\}^5$ indexes 32 bit positions. The polynomial has $n + 5$ variables and $2^{n+5} = 32 \cdot 2^n$ evaluations.

**Indexing**: $B[x + y \cdot 2^n]$ where $x \in [0, 2^n)$ and $y \in [0, 32)$.

**Correctness constraint**: $\text{value}(x) = \sum_{y=0}^{31} B(x, y) \cdot 2^y$

### Prove Protocol

```rust
// src/dag/mod.rs — prove_range()
fn prove_range(&self, witnesses: &[Vec<Witness>], claims: &mut Vec<Vec<Claim>>,
               transcript: &mut Transcript) -> LookupProof {
    let table_size = 1usize << BIT_TABLE_VARS;  // 32

    // 1. Challenge generation
    let alpha = transcript.challenge_ext2(b"range_alpha");
    let beta = transcript.challenge_ext2(b"range_beta");

    // 2. For each range-checked node, compute partial evaluation of B
    //    Fix x-variables at the claim point → 32-element vector
    let infos: Vec<RangeNodeInfo> = preps.par_iter().map(|prep| {
        let input_size = 1usize << prep.input_num_vars;
        let bit_evals = witnesses[prep.aux_edge][0].data.as_ref().unwrap().evaluations_ref();
        let eq_ext2 = evaluate_lagrange_basis_ext2(&prep.claim_point[..prep.input_num_vars]);

        // part_aux[y] = Σ_x B[x + y*input_size] * eq(r, x)
        let mut part_aux = vec![GoldilocksExt2::zero(); table_size];
        for y in 0..table_size {
            let base = y * input_size;
            let mut acc = GoldilocksExt2::zero();
            for x in 0..input_size {
                if bit_evals[base + x].0 != 0 {
                    acc = ext2_add(acc, eq_ext2[x]);
                }
            }
            part_aux[y] = acc;
        }

        // middle_claim = Σ_y part_aux[y] * 2^y  (reconstructs the value)
        let mc = (0..table_size).map(|y|
            ext2_mul(part_aux[y], GoldilocksExt2::from_base(GoldilocksField(1u64 << y)))
        ).fold(GoldilocksExt2::zero(), ext2_add);

        RangeNodeInfo { part_aux, middle_claim: mc, sum_aux: sum(part_aux) }
    }).collect();

    // 3. Single table sumcheck over 5 variables
    //    Proves: Σ_y combined_aux(y) * (T(y) + α) = expected_sum
    //    where T(y) = 2^y and combined_aux(y) = Σ_i β_i * part_aux_i[y]
    let mut combined_aux = vec![GoldilocksExt2::zero(); table_size];
    for (i, info) in infos.iter().enumerate() {
        for y in 0..table_size {
            combined_aux[y] = ext2_add(combined_aux[y],
                ext2_mul(betas[i], info.part_aux[y]));
        }
    }

    let table_alpha: Vec<GoldilocksExt2> = (0..table_size)
        .map(|y| ext2_add(GoldilocksExt2::from_base(GoldilocksField(1u64 << y)), alpha))
        .collect();

    let mut prover = CpuLinearSumcheckProverExt2::new(BIT_TABLE_VARS, 2, transcript);
    let proof = prover.prove(&mut [combined_aux, table_alpha], transcript);
}
```

### Size Reduction

| Block | Old vars | Old size | New vars | New size | Reduction |
|-------|----------|----------|----------|----------|-----------|
| ScaleDown (sf=10) | n + 10 | 1024 × 2^n | n + 5 | 32 × 2^n | 32x |
| NonNegative (20-bit) | n + 20 | 1M × 2^n | n + 5 | 32 × 2^n | 32,768x |

### Impact

GPT-2 12L:
- Auxiliary commit elements: 2,439M → 18.2M (134x reduction)
- Total prove time: 2.67s → 0.95s (2.8x speedup)
- VGG-16: 3.15s → 0.30s (10.5x speedup)

---

## 10. LUT-Based Permutation

**File**: `src/basicblock/einsum.rs` (function `permute_evals_by_ranges`)

### Problem

Einsum variable reordering requires permuting a polynomial evaluation array. A permutation $\sigma$ on $n$-bit indices maps each index $i$ to $\sigma(i)$ by rearranging bits. The naive approach:

```rust
// O(n) per element — n bit extractions + reassembly
for idx_new in 0..total {
    let mut idx_old = 0;
    for new_var in 0..n {
        if idx_new & (1 << new_var) != 0 {
            idx_old |= 1 << inv_perm[new_var];
        }
    }
    out[idx_new] = evals[idx_old];
}
```

For n=27 (128M elements): 27 × 128M = 3.4 billion bit operations → ~200ms.

### Solution: Two-Half Split LUT

Split the $n$-bit index into low half and high half. Precompute two LUTs:

```rust
// src/basicblock/einsum.rs — permute_evals_by_ranges()
if n > 16 {
    let half = n / 2;
    let lo_mask = (1usize << half) - 1;

    // Precompute LUTs: O(2^(n/2)) each
    let mut lo_lut = vec![0usize; 1 << half];
    for lo_bits in 0..(1 << half) {
        let mut old_idx = 0usize;
        for bit in 0..half {
            if lo_bits & (1 << bit) != 0 {
                old_idx |= 1 << inv_perm[bit];
            }
        }
        lo_lut[lo_bits] = old_idx;
    }

    let mut hi_lut = vec![0usize; 1 << (n - half)];
    for hi_bits in 0..(1 << (n - half)) {
        let mut old_idx = 0usize;
        for bit in 0..(n - half) {
            if hi_bits & (1 << bit) != 0 {
                old_idx |= 1 << inv_perm[half + bit];
            }
        }
        hi_lut[hi_bits] = old_idx;
    }

    // O(1) per element using LUTs
    const PAR_THRESHOLD: usize = 1 << 18;  // 256K
    if total >= PAR_THRESHOLD {
        (0..total).into_par_iter().map(|idx_new| {
            let lo = idx_new & lo_mask;
            let hi = idx_new >> half;
            evals[lo_lut[lo] | hi_lut[hi]]
        }).collect()
    } else {
        // Sequential version for small arrays
    }
}
```

**Why it works**: Bit permutations are separable. Each output bit of $\sigma(i)$ depends on exactly one input bit. The contribution from the low half of $i$ and the high half of $i$ can be independently computed and OR'd together.

**LUT sizes**: For n=27, half=13: `lo_lut` = 8K entries (8KB), `hi_lut` = 16K entries (16KB). Both fit in L1 cache.

### GPU Bit Permutation Attempted but Rejected

A GPU scatter/gather kernel (`bit_permute_gl_ffi`) was implemented but found slower due to random memory access patterns defeating GPU cache hierarchy. The CPU LUT approach with sequential writes is more cache-friendly.

---

## 11. Parallel Einsum Computation

**File**: `src/basicblock/einsum.rs` (function `einsum_compute`)

### Problem

The forward pass computes einsum tensor contractions: $\text{out}[i_1, \ldots, i_k] = \sum_{j_1, \ldots, j_m} \prod_t \text{input}_t[\ldots]$

For large outputs (millions of elements), sequential computation is a bottleneck.

### Solution

Parallelize over output indices using rayon:

```rust
// Little-endian indexing: first dimension has stride 1
let result: Vec<GoldilocksField> = (0..out_size).into_par_iter().map(|out_idx| {
    // Decompose out_idx into multi-dimensional index (little-endian)
    let mut out_multi = Vec::with_capacity(out_dims.len());
    let mut remainder = out_idx;
    for &d in out_dims.iter() {
        out_multi.push(remainder % d);
        remainder /= d;
    }

    // Map output characters to their index values
    let mut index_map: HashMap<char, usize> = HashMap::new();
    for (i, &c) in output_indices.iter().enumerate() {
        index_map.insert(c, out_multi[i]);
    }

    // Sum over all summation index assignments
    let mut sum = GoldilocksField(0);
    for sum_idx in 0..sum_size {
        let mut s_remainder = sum_idx;
        for &c in sum_indices.iter() {
            let d = *dim_map.get(&c).unwrap_or(&1);
            index_map.insert(c, s_remainder % d);
            s_remainder /= d;
        }

        // Product over all input tensors
        let mut product = GoldilocksField(1);
        for (t, input) in inputs.iter().enumerate() {
            // Compute linear index using little-endian stride
            let mut linear_idx = 0;
            let mut stride = 1;
            for i in 0..term_chars[t].len() {
                let c = term_chars[t][i];
                let idx_val = *index_map.get(&c).unwrap_or(&0) % padded_shapes[t][i];
                linear_idx += idx_val * stride;
                stride *= padded_shapes[t][i];
            }
            product = gl_mul(product, input[linear_idx]);
        }
        sum = gl_add(sum, product);
    }
    sum
}).collect();
```

**Key detail**: `term_chars` and `padded_shapes` are precomputed once before the parallel loop to avoid repeated allocation inside the closure.

---

## 12. Parallel Forward Pass

**File**: `zk-torch-3/src/dag/mod.rs`

### Problem

The DAG forward pass computes all intermediate tensors by traversing nodes in topological order. Sequential traversal doesn't exploit the parallelism inherent in models — e.g., in a transformer, the Q/K/V projections at the same layer are independent.

### Solution

Compute topological levels during compilation, then process each level's nodes in parallel:

```rust
// During compile():
let topo_levels: Vec<Vec<NodeId>> = compute_topo_levels(&topo_order, &nodes);

// During run():
for level in &self.topo_levels {
    level.par_iter().for_each(|&node_id| {
        let node = &self.nodes[node_id];
        let inputs: Vec<&Witness> = node.inputs.iter()
            .map(|&e| &witnesses[e])
            .collect();
        let outputs = node.kind.run(&inputs);
        for (i, &e) in node.outputs.iter().enumerate() {
            witnesses[e] = outputs[i].clone();
        }
    });
}
```

### Impact

GPT-2 12L forward pass: 26.2s → 1.5s (17x speedup). Transformers have high parallelism within each layer (3 attention projections + MLP = 4-5 independent einsum operations per layer).

---

## 13. Parallel Lagrange Basis Evaluation

**File**: `src/poly/mod.rs`

### Mathematical Background

The equality polynomial (Lagrange basis) is:

$$\text{eq}(r, x) = \prod_{i=0}^{n-1} (r_i x_i + (1 - r_i)(1 - x_i))$$

For all $x \in \{0,1\}^n$, this table has $2^n$ entries and is built incrementally:

```
Start: evals = [1]
For each variable i:
  For each existing entry j:
    evals[j]            *= (1 - r_i)   // x_i = 0 contribution
    evals[j + 2^i]       = evals[j] * r_i   // x_i = 1 contribution
```

### Parallelization

When `half >= 8192`, the update loop is parallelized using `split_at_mut` for disjoint mutable borrows:

```rust
const EQ_PAR_THRESHOLD: usize = 8192;

pub fn evaluate_lagrange_basis_ext2(point: &[GoldilocksExt2]) -> Vec<GoldilocksExt2> {
    let n = point.len();
    let size = 1usize << n;
    let mut evals = vec![GoldilocksExt2::from_base(GoldilocksField(1)); size];

    for i in 0..n {
        let half = 1usize << i;
        let factor_one = point[i];
        let factor_zero = ext2_sub(
            GoldilocksExt2::from_base(GoldilocksField(1)),
            point[i],
        );

        if half >= EQ_PAR_THRESHOLD {
            let (lo, hi) = evals[..2 * half].split_at_mut(half);
            // Phase 1: set upper half (x_i = 1)
            hi.par_iter_mut().zip(lo.par_iter()).for_each(|(h, l)| {
                *h = ext2_mul(*l, factor_one);
            });
            // Phase 2: update lower half (x_i = 0)
            lo.par_iter_mut().for_each(|l| {
                *l = ext2_mul(*l, factor_zero);
            });
        } else {
            // Sequential for small arrays
            for j in (0..half).rev() {
                evals[j | half] = ext2_mul(evals[j], factor_one);
                evals[j] = ext2_mul(evals[j], factor_zero);
            }
        }
    }
    evals
}
```

**Why `split_at_mut`**: Rust's borrow checker prevents `par_iter_mut` on overlapping slices. `split_at_mut(half)` gives two disjoint mutable slices that can be safely iterated in parallel.

---

## 14. GPU Partial Eval in CPU Sumcheck Path

**File**: `src/basicblock/einsum.rs`

### Problem

For batch=1 inference, many Einsum operations have `total_rounds = 12-14`, below `GPU_SUMCHECK_THRESHOLD = 14`. The CPU sumcheck path is used. But within this path, `partial_eval_ext2_cpu` (folding a polynomial after each sumcheck round) operates on the full polynomial — for 128M-element weight matrices, each fold takes ~50ms on CPU.

With 12-14 rounds: 12 × 50ms = 600ms per Einsum, and there are dozens of Einsum nodes. This CPU fold was 72% of total Einsum prove time.

### Solution

Use GPU `partial_eval_ext2` for the fold step, even within the CPU sumcheck path:

```rust
// src/basicblock/einsum.rs — inside CPU sumcheck loop
fn fold_polynomial(poly: &[GoldilocksExt2], challenge: GoldilocksExt2, n: usize)
    -> Vec<GoldilocksExt2>
{
    if n > gpu_partial_eval_threshold() {
        // GPU path: upload → fold → download
        goldilocks_cuda::partial_eval::partial_eval_ext2(&poly, &challenge)
    } else {
        // CPU path: sequential fold
        partial_eval_ext2_cpu(&poly, &challenge)
    }
}
```

**Key detail**: Must use the high-level API (`partial_eval_ext2`) which correctly allocates a $2^{n-1}$ output buffer. The low-level `partial_eval_ext2_device_u64` requires manual buffer management.

**Stale CUDA error handling**: After `gpu_store.free_gpu()`, a stale `cudaErrorMemoryAllocation` may persist. Must call `synchronize()` + `get_last_error()` before any GPU kernel to clear it.

### Results

| Model | Before (CPU-only fold) | After (GPU fold) | Speedup |
|-------|----------------------|------------------|---------|
| GPT-J 28L | 15.8s | 5.2s | 3.02x |
| LLaMA 32L | 13.6s | 4.4s | 3.11x |
| BERT 24L | 5.3s | 3.4s | 1.54x |
| GPT-2 12L | 3.5s | 3.2s | 1.09x |

Large models benefit most because they have more large weight matrices where the GPU fold is critical.

---

## 15. Sparse Polynomial Evaluation

**File**: `src/poly/sparse.rs`

### Problem

Sparse polynomials (used in lookup proofs) store only nonzero entries. Evaluating at a point $r$ requires:

$$p(r) = \sum_{(i, v_i) \in \text{sparse}} v_i \cdot \text{eq}(r, i)$$

The naive approach computes the full $2^n$-element eq table then indexes into it. For $n = 30+$ (large sparse polynomials in lookup proofs), this creates a 1-billion-element table — causing 270-second lookup proofs.

### Solution: O(k·n) Per-Entry Eq Computation

```rust
// src/poly/sparse.rs
fn evaluate_at_point_ext2(&self, point: &[GoldilocksExt2]) -> GoldilocksExt2 {
    let one = GoldilocksExt2::from_base(GoldilocksField(1));
    let mut result = GoldilocksExt2::zero();

    for (&idx, &val) in &self.evaluations {
        // Compute eq(idx, point) = Π_i (bit_i * point_i + (1-bit_i) * (1-point_i))
        let mut eq_val = one;
        for i in 0..self.n {
            let bit = (idx >> i) & 1;
            let factor = if bit == 1 { point[i] } else { ext2_sub(one, point[i]) };
            eq_val = ext2_mul(eq_val, factor);
        }
        result = ext2_add(result, ext2_mul(eq_val, GoldilocksExt2::from_base(val)));
    }
    result
}
```

For $k$ nonzero entries and $n$ variables: $O(k \cdot n)$ multiplications instead of $O(2^n)$. Since $k \ll 2^n$ for sparse polynomials, this is dramatically faster.

### Impact

Lookup proof time: 270s → 0.9s (300x speedup).

---

## 16. Conv2D Alpha-Power Factorization

**File**: `src/basicblock/conv.rs`

### Mathematical Background

A 1D convolution computes:

$$Y[d, m] = \sum_{c=0}^{C-1} \sum_{k} X[c, m+k] \cdot W[d, c, k]$$

Proving this constraint directly requires a sumcheck over all $(d, m, c, k)$ variables simultaneously, which is expensive for large convolutions.

### Alpha-Power Trick

The key insight is that $\alpha^{i+j} = \alpha^i \cdot \alpha^j$ factorizes the positional constraint:

Define:
$$F[c] = \sum_{i} X_{\text{rev}}[c, i] \cdot \alpha^i$$
$$G[c] = \sum_{j} W_{\text{flat}}[c, j] \cdot \alpha^j$$

where $X_{\text{rev}}[c, i] = X[c, S_{\text{in}} - 1 - i]$ is the reversed input and $W_{\text{flat}}$ is the flattened kernel.

Then:
$$\sum_c F[c] \cdot G[c] = \sum_c \sum_{i,j} X_{\text{rev}}[c,i] \cdot W_{\text{flat}}[c,j] \cdot \alpha^{i+j}$$

The product $\alpha^{i+j}$ with reversal encodes the convolution position constraint: position $m$ in the output corresponds to $i + j = S_{\text{in}} - 1 - m$.

### 4 Cascaded Sumchecks

Instead of one large sumcheck, Conv2D uses 4 smaller sumchecks:

1. **Eq-sumcheck on output spatial**: Reduces the output claim to channel-level claim using eq(r_spatial, k)
2. **Channel F×G sumcheck**: Proves $\sum_c F[c] \cdot G[c] = s_{\alpha\_\text{conv}}$
3. **F → X reduction**: Reduces the claim on $F$ to a claim on $X$ using the alpha power table MLE
4. **G → W_flat reduction**: Reduces the claim on $G$ to a claim on $W_{\text{flat}}$

### Alpha Table MLE

The verifier can efficiently compute the alpha table MLE evaluation:

```rust
// O(l) verifier-computable
fn alpha_table_mle_eval(r: &[GoldilocksExt2], alpha: GoldilocksExt2) -> GoldilocksExt2 {
    // Π_j (1 + r_j · (α^{2^j} - 1))
    let mut result = GoldilocksExt2::from_base(GoldilocksField(1));
    let mut alpha_pow = alpha;  // α, α^2, α^4, α^8, ...
    for j in 0..r.len() {
        let factor = ext2_add(
            GoldilocksExt2::from_base(GoldilocksField(1)),
            ext2_mul(r[j], ext2_sub(alpha_pow, GoldilocksExt2::from_base(GoldilocksField(1)))),
        );
        result = ext2_mul(result, factor);
        alpha_pow = ext2_mul(alpha_pow, alpha_pow);  // square
    }
    result
}
```

This evaluates $\sum_{i=0}^{2^l-1} \alpha^i \cdot \text{eq}(r, i)$ in $O(l)$ operations.

---

## 17. Batch Query Extraction

**Files**: `goldilocks-cuda-rs/src/merkle.rs`, `goldilocks-cuda-rs/src/basefold.rs`

### Problem

Query extraction in `open_ext2()` accounted for 42% of opening time (~196ms for n=22). Root cause: for each of 34 queries × 20 rounds = 680 Merkle auth path lookups, each requiring multiple tiny `cudaMemcpy` calls to read individual tree nodes.

Total: ~10,000+ tiny GPU→CPU transfers, each with ~5μs latency overhead.

### Solution: Level-Based Bulk Download

```rust
// goldilocks-cuda-rs/src/merkle.rs
impl DeviceMerkleTree {
    pub fn batch_auth_paths(&self, leaf_indices: &[usize]) -> Result<Vec<Vec<Poseidon2Hash>>> {
        const LEVEL_BULK_THRESHOLD: usize = 16384;

        // Download each level: bulk if small enough, selective otherwise
        let mut level_data: Vec<Option<Vec<u64>>> = Vec::new();
        for layer in 0..num_layers {
            let level_size = num_leaves >> layer;
            if level_size <= LEVEL_BULK_THRESHOLD {
                // Single bulk cudaMemcpy for entire level
                level_data.push(Some(self.d_tree.read_slice(offset, level_size * 4)?));
            } else {
                level_data.push(None);  // too large, use per-query reads
            }
        }

        // Extract auth paths from downloaded data
        let mut all_paths = Vec::new();
        for &leaf_idx in leaf_indices {
            let mut path = Vec::new();
            let mut idx = leaf_idx;
            for layer in 0..num_layers {
                let sibling_idx = idx ^ 1;
                if let Some(ref data) = level_data[layer] {
                    // Read from host memory — zero GPU transfer
                    path.push(hash_from_slice(&data[sibling_idx * 4..]));
                } else {
                    // Selective GPU read for large levels
                    path.push(self.read_node(layer, sibling_idx)?);
                }
                idx >>= 1;
            }
            all_paths.push(path);
        }
        all_paths
    }
}
```

Also introduced `HostMerkleTree` — a fully downloaded tree for repeated query extraction:

```rust
pub struct HostMerkleTree {
    pub data: Vec<u64>,           // all tree nodes
    pub layer_offsets: Vec<usize>,
    pub num_leaves: usize,
}

impl HostMerkleTree {
    pub fn auth_path(&self, leaf_index: usize) -> Vec<Poseidon2Hash> {
        // Pure CPU indexing — zero GPU transfers
    }
}
```

### Results

| Polynomial size | Before | After | Speedup |
|----------------|--------|-------|---------|
| n=22 (4M leaves) | 196ms | 67-104ms | 2-3x |
| n=20 (1M leaves) | 49-86ms | ~30ms | 1.5-2.5x |

---

## 18. Zero-Rebuild Opening Proofs

**Files**: `zk-torch-3/src/commit/basefold.rs`, `zk-torch-3/src/dag/mod.rs`

### Problem

Originally, opening proofs required re-committing the polynomial (rebuilding the Merkle tree from scratch) because the GPU commitment was freed after the initial commit phase to save memory for sumcheck.

### Solution

Keep GPU commitments alive throughout the entire prove phase:

```rust
pub struct GpuCommitmentStore {
    pub commitments: Vec<Option<BasefoldCommitment>>,  // stay alive!
    pub device_ids: Vec<Option<i32>>,
}

// In prove():
// Commitments are NOT freed between commit and opening phases
// open_ext2() called directly on the stored commitment
let proof = gpu_store.commitments[edge_id].as_ref().unwrap()
    .open_ext2(&point, &dev_table, &mut transcript, num_queries)?;
```

### Non-Canonical Field Value Bug

GPU CUDA kernels can produce non-canonical Goldilocks values (≥ p, e.g., p + 287 instead of 287). When CPU code downloads these values for the CPU opening path, trait-based Add/Sub assumes canonical inputs and produces wrong fold results. The GPU verifier's `gl_*_host` functions normalize via `% p`.

**Fix**: Normalize after download, but preserve raw values for Merkle auth path verification:

```rust
fn normalize_to_canonical(evals: &mut [u64]) {
    for v in evals.iter_mut() {
        if *v >= GOLDILOCKS_PRIME {
            *v %= GOLDILOCKS_PRIME;
        }
    }
}
// BUT: raw (non-normalized) values needed for Merkle hash verification
// because the GPU Merkle tree was built with non-canonical values
```

---

## 19. Basefold Table Clone Optimization

**File**: `goldilocks-cuda-rs/src/basefold.rs`

### Problem

`BasefoldTable::generate()` computes modular inverses for ~33M entries using extended GCD — takes 22.6s per device. With 4 GPUs, sequential generation = 90.3s per prove call.

### Solution

Added `clone_to_current_device()`: copies the CPU-side precomputed data and uploads to the current GPU device, completely bypassing the expensive inverse computation:

```rust
impl BasefoldTable {
    pub fn clone_to_current_device(&self) -> Result<Self> {
        let mut cloned = self.clone_cpu_data();  // O(n) memcpy
        cloned.upload()?;  // GPU upload ~0.8s
        Ok(cloned)
    }
}
```

**Result**: Table setup 90.3s → 3.2s (28x faster). Combined with per-device caching (Section 6), this cost is paid once at initialization.

---

## 20. CPU Poseidon2 for Merkle Verification

**File**: `goldilocks-cuda-rs/src/cpu_poseidon2.rs`

### Purpose

The verifier checks Basefold opening proofs entirely on CPU — no GPU required. This needs a CPU implementation of Poseidon2 (the hash function used for Merkle trees).

### Implementation

8-wide Poseidon2 permutation matching the GPU kernel exactly:

```rust
pub fn poseidon2_permute(state: &mut [u64; 8]) {
    // 4 rounds of full S-box (all 8 elements)
    for round in 0..4 {
        for i in 0..8 { state[i] = sbox(state[i]); }  // x^7
        mds_light_8(state);  // linear layer
        for i in 0..8 { state[i] = gl_add_host(state[i], RC_EXT[round][i]); }
    }
    // 22 rounds of single S-box + diffusion
    for round in 0..22 {
        state[0] = sbox(state[0]);
        state[0] = gl_add_host(state[0], RC_INT[round]);
        diffusion_8(state);  // diagonal matrix + all-sum
    }
    // 4 rounds of full S-box
    for round in 4..8 {
        for i in 0..8 { state[i] = sbox(state[i]); }
        mds_light_8(state);
        for i in 0..8 { state[i] = gl_add_host(state[i], RC_EXT[round][i]); }
    }
}

fn sbox(x: u64) -> u64 {
    let x2 = gl_mul_host(x, x);
    let x4 = gl_mul_host(x2, x2);
    let x3 = gl_mul_host(x2, x);
    gl_mul_host(x4, x3)  // x^7
}
```

### Verification Functions

```rust
pub fn hash_gl_leaf(a: u64, b: u64) -> Poseidon2Hash {
    let mut state = [a, b, 0, 0, 0, 0, 0, 0];
    poseidon2_permute(&mut state);
    Poseidon2Hash::from_raw([state[0], state[1], state[2], state[3]])
}

pub fn verify_auth_path(leaf_hash: Poseidon2Hash, mut index: usize,
                        path: &[Poseidon2Hash], root: &Poseidon2Hash) -> bool {
    let mut current = leaf_hash;
    for sibling in path {
        current = if index & 1 == 0 {
            poseidon2_compress(&current, sibling)  // current is left child
        } else {
            poseidon2_compress(sibling, &current)  // current is right child
        };
        index >>= 1;
    }
    current == *root
}
```

---

## 21. Threshold Configuration Reference

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ZK_GPU_SUMCHECK_THRESHOLD` | 14 | GPU sumcheck when `total_rounds > threshold` |
| `ZK_GPU_PARTIAL_EVAL_THRESHOLD` | 16 | GPU partial eval in CPU sumcheck when `n > threshold` |
| `ZK_GPU_FUSED_THRESHOLD` | 16 | Fused GPU permute+peval when `n > threshold` |
| `CPU_OPEN_THRESHOLD` | 14 | CPU opening when `n ≤ threshold`, GPU otherwise |
| `NUM_PARTITIONS` | 1 | DAG partitions for multi-GPU proving |
| `NUM_LAYERS` | 1-2 | Number of model layers to prove |
| `CUDA_VISIBLE_DEVICES` | All | GPU device selection |
| `RAYON_NUM_THREADS` | All cores | CPU thread count |

### Compile-Time Constants

| Constant | Value | File | Purpose |
|----------|-------|------|---------|
| `EQ_PAR_THRESHOLD` | 8192 | `poly/mod.rs` | Parallel eq evaluation |
| `BIT_TABLE_VARS` | 5 | `scale.rs` | Bit decomposition (32 bits) |
| `PAR_THRESHOLD` | 2^18 | `einsum.rs` | Parallel permutation |
| `MIN_BASEFOLD_VARS` | 2 | `dag/mod.rs` | Min vars for Basefold commit |
| `MAX_BASEFOLD_VARS` | 22 | `dag/mod.rs` | Max vars for Basefold commit |
| `LEVEL_BULK_THRESHOLD` | 16384 | `merkle.rs` | Bulk Merkle level download |

### Tuning Guide

**Maximum GPU utilization** (A100-80GB or larger):
```bash
ZK_GPU_SUMCHECK_THRESHOLD=10 ZK_GPU_OPEN_THRESHOLD=10 \
ZK_GPU_PARTIAL_EVAL_THRESHOLD=10 ZK_GPU_FUSED_THRESHOLD=10
```

**Memory-constrained GPUs** (< 40GB):
```bash
ZK_GPU_SUMCHECK_THRESHOLD=16 ZK_GPU_OPEN_THRESHOLD=18 \
ZK_GPU_PARTIAL_EVAL_THRESHOLD=18 ZK_GPU_FUSED_THRESHOLD=18
```

**CPU-only proving** (no GPU):
```bash
ZK_GPU_SUMCHECK_THRESHOLD=999 ZK_GPU_OPEN_THRESHOLD=999 \
ZK_GPU_PARTIAL_EVAL_THRESHOLD=999 ZK_GPU_FUSED_THRESHOLD=999
```

---

## Performance Evolution

Cumulative impact on GPT-2 12L, 4x A100-80GB:

| Milestone | Prove Time | Change |
|-----------|-----------|--------|
| Initial CPU-only implementation | ~72.7s | baseline |
| + GPU sumcheck + parallel forward pass | ~3.5s | 20x faster |
| + GPU opening proofs | ~2.67s | 1.3x |
| + Dense bit poly range checks | ~0.95s | 2.8x |
| + Fused permute + partial eval | ~0.90s | 1.06x |
| + Full Basefold Merkle soundness | ~8.93s | (adds real proofs) |
| + Per-device table caching | ~6.5s | 1.37x |
| + GPU open pre-allocation + concurrent | ~3.34s | 1.95x |
| **Final (all optimizations + soundness)** | **3.06s** | — |

Note: Adding full Basefold Merkle soundness increased prove time because it replaced dummy proofs with real Merkle auth paths + query verification. Subsequent optimizations recovered most of this cost while maintaining full cryptographic soundness.

### Final Benchmarks (4x A100-80GB, 4 partitions, full soundness)

| Model | Layers | Nodes | Edges | Run | Total Prove | Verify |
|-------|--------|-------|-------|-----|-------------|--------|
| GPT-2 Small | 12 | 1910 | 2778 | 0.65s | 3.06s | 151ms |
| BERT-Large | 24 | 3737 | 5436 | 1.26s | 5.83s | 272ms |
| GPT-J 6B | 28 | 3513 | 5268 | 13.3s | 8.78s | 293ms |
| LLaMA-2 7B | 32 | 4509 | 6701 | 19.8s | 12.22s | 366ms |
| VGG-16 | 13 conv | 139 | 172 | 1.94s | 1.60s | 34ms |
| ResNet-50 | 53 conv | 368 | 424 | 23.6s | 9.51s | 80ms |

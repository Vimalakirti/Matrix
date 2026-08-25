# zk-torch-4 prover performance — findings & next steps

Reference notes for further optimization work. Captures the
optimization history that closed the gap from `zk-torch-4` at ~22 min
(tiny config) → 13 s (full GPT-2 hidden=768), and where the remaining
~17× gap to `zk-torch-3` lives.

---

## 0. Benchmark setup

All numbers below are from:

- Model: GPT-2 Small, `hidden_dim = 768`, `num_heads = 12`,
  `head_dim = 64`, `seq_len = 1`, `NUM_LAYERS = 1`.
- Random weights — all zero (the bench cares about timing, not the
  actual output). Real model weights would behave the same for cost.
- Hardware: 96 threads, 4 GPUs.
- `bench_config.yaml`:
  ```
  sf:
    scale_factor_log: 10
    table_size_log: 10
    table_commit_log: 8
  ```
- Verifier returns `false` for the all-zero config (some intermediate
  leaves NonNegative's table). Doesn't affect timing of the prove
  pipeline (which completes end-to-end).

---

## 1. Headline numbers

**Online (prover-time) vs offline (amortized once per model):** the
constants/weight commits are not part of online prover work — they're
computed once per model and reused across many proofs. zk-torch-3
splits this explicitly (`Weight commits: 0.220s (offline)`); previous
notes incorrectly lumped both into a single "Commit" number for
zk-torch-4. Corrected:

| Stage | zk-torch-3 (Basefold) | zk-torch-4 (fold tree) | ratio (4/3) |
|---|---|---|---|
| Commit **offline** (amortized, not counted) | 220 ms | 1,480 ms  | (n/a — offline) |
| **Commit online** (prover time)             | **103 ms** | **155 ms**    | 1.5× |
| Backward sumcheck                          | 75 ms       | 261 ms      | 3.5× |
| Lookup proofs                              | 83 ms       | (folded in) | — |
| Opening / Fold tree (wall)                 | **627 ms**  | **7,517 ms** | 12× |
| Leaf prep (bit_decompose + MLE eval)       | (no analog) | 5,008 ms    | — |
| **Prove total (online)**                   | **898 ms**  | **12,941 ms** | **~14×** |
| Verify                                     | 41 ms       | 7 ms        | **0.17× (zk-torch-4 wins)** |

Tensor-core multifold kernel (`multifold_mixed_witness_tc_fused`) is
now used for both all-binary and ternary paths (was only ternary).
Small improvement at this scale (~3% off fold tree); the kernel was
designed for `N_ring ≥ ~4K` where it really shines.

`zk-torch-3` parallelizes across 4 GPUs + 48 CPU threads. `zk-torch-4`
uses rayon CPU parallelism + per-arity bucket dispatch on a single CUDA
context.

---

## 2. zk-torch-4 prover op-by-op

### Per stage (current — ~1.0 s online prove, ~85 ms verify, verified)

| Stage | Time | % of online prove |
|---|---|---|
| Commit (online — input-dependent)              | 85 ms    | 8.5% |
| Backward + lookups (parallel bool sumchecks)   | 165 ms   | 16.5% |
| Leaf build (GPU eq + selective add)            | 105 ms   | 10.5% |
| Fold tree (wall, parallel buckets, GPU SP ≥18) | 645 ms   | **64.5%** |
| **Online prover total**                        | 1,000 ms | 100% |
| Commit (offline, amortized — NOT counted)      | 510 ms   | n/a |
| Verify                                         | **85 ms** | — |

### C: parallel bool sumchecks (290 → 165 ms backward total)

`prove_range` had a serial per-aux_num_var bool-sumcheck loop. The
inside of the dag.prove breakdown was 280 ms total, of which 248 ms
was bool sumchecks alone (six groups, 5–74 ms each).

Switched to parallel via transcript forks: each group runs against
`transcript.fork(aux_num_var)` (BTreeMap key — unique per group), then
results are folded back into the parent transcript sequentially in
iteration order. The verifier does the exact same fork pattern.

bool sumchecks 248 ms → **53 ms (4.7×)**. The rest of `range` (table
sumcheck, lagrange tables, accumulation) is small (~35 ms).

### Llama-2 DAG end-to-end

Added `src/bin/llama2.rs` (Llama-2-7B prover bench wired to the
existing `dag::llama::llama_2_7b` builder — uses RoPE, SwiGLU MLP,
llama RMSNorm with the `RMSReciprocal::run` fix). Dims overridable
via `NUM_HEADS / HEAD_DIM / FFN_DIM / VOCAB` env vars; defaults are
Llama-2-7B (32/128/11008/32000).

Bringing up the new bench surfaced a **multi-output reducer bug**: in
`Dag::prove`, the reducer was being invoked with
`reducer_edge_ids = vec![*edge_ids.last().unwrap()]`. For multi-output
nodes (ScaleDown / ScaleUp / ExpHelper — main output + aux),
`edge_ids.last()` is the AUX edge, not `node.outputs[0]`. The reduced
claim got tagged with the wrong edge, and the verifier's
`produced_claims.last().edge_id == node.outputs[0]` check failed.

GPT-2 happened to never hit this path (its multi-output nodes never
ended up with > 1 accumulated claim on `outputs[0]` during the
backward pass). Llama-2 does — e.g., the FFN's SwiGLU produces a
ScaleDown intermediate consumed by two downstream einsums. Fix: use
`vec![node.outputs[0]]`.

**Llama bench, "medium" config (NUM_HEADS=12 HEAD_DIM=64 FFN_DIM=3072
VOCAB=1024 — matches GPT-2 hidden=768)**:

```
Prove:  1.03 s
Verify: 100 ms
Verified: true
```

### I: GPU eq-scratch reuse + shared-eq fast path (modest)

**Reused (d_a, d_b) eq-scratch across leaves** in the GPU SP setup:
`new_device_eq_packed_f` and `new_device_eq_packed_ternary` now
allocate the eq DP scratch buffers ONCE per group and pass them to
each per-leaf `aext2_eq_dp_all_ffi` call. Saves N · 2 cudaMallocs of
`2^arity` Ext2 buffers (~64 MB each at arity 22) per group — small
but real.

**Shared-eq fast path** added: if all leaves in a group have identical
`claim_pt`s (which is the case at fold-tree level ≥ 1 *when all the
group's chunks came from the same level-0 group*), build the eq table
ONCE and D2D-broadcast it. Only triggers when the group is fully
homogeneous; rarely fires for Llama-2-7B since level-1 groups mix
chunks from many different level-0 groups → distinct `shared_r`s.

**Llama-2-7B impact**: 67–70 s prove (noise band). Within run-to-run
variation of the H baseline.

### Diagnosis: where the 60+ s actually goes

Per-group setup at arity 22 / 63 leaves (instrumented):

| | Time |
|---|---|
| cudaMalloc `d_polys` (16 GB)     | 15-30 ms |
| 63 × per-leaf eq build (serial)  | **1.3-1.5 s** |
| Concat + lift kernel             | < 50 ms |

The dominant cost is the **63 sequential GPU eq builds**, each running
22 stages of doubling DP. Each stage launches a kernel + implicit
sync; total per-build is ~22 ms of which actual compute is ~5 ms (the
rest is launch + sync latency). Across 30 groups at L0 of the arity-22
bucket: 30 × 1.4 s = 42 s.

Single-stream serial execution is the bottleneck. Closing this gap
requires either:
1. **CUDA per-stream parallelism** — issue per-leaf eq builds on
   distinct streams so they pipeline across kernel-launch latency.
   Needs stream plumbing through the FFI / DeviceBuffer.
2. **Batched eq build kernel** — fold the per-leaf loop into the
   kernel itself (grid Y = num_leaves), avoiding 22 × N launches in
   favor of 22 stages × 1 launch.
3. **Pinned host memory** for upload-based path. Today's pageable
   transfers cap at ~1 GB/s; pinned achieves ~25 GB/s on PCIe 4.0
   and would make the host-eq + upload alternative competitive.

### Baseline gap

zk-torch-3 (single GPU, same Llama-2-7B): **4.47 s prove / 42 ms verify**.

| | zk-torch-3 | zk-torch-4 now | ratio |
|---|---|---|---|
| Prove (online)  | 4.47 s | 67 s   | 15× |
| Verify          | 42 ms  | 399 ms | 9.5× |

Closing the prove gap to within 2-3× would need at least one of the
batched-eq / multi-stream paths above, plus working multi-GPU
distribution (currently shipped but blocked by host GPU contention).
A 1.5-2× gap is plausible on a dedicated 4-GPU box.

### H: FFN sharding

`llama_mlp` now takes `Vec<EdgeId>` for each of SwiGLU's three projection
matrices (gate / up / down). For N shards along the ffn axis, the
pipeline becomes:

```text
for each shard s in 0..N:
    gate_s = x @ w_1[:, ffn_s]      # [b,s,hidden] @ [hidden, ffn/N]  →  [b,s,ffn/N]
    swish_s = gate_s · sigmoid(gate_s)
    up_s = x @ w_2[:, ffn_s]
    partial_s = (swish_s · up_s) @ w_3[ffn_s, :]   # [b,s,ffn/N] @ [ffn/N, hidden]
output = Σ_s partial_s
```

At full Llama-2-7B (`hidden=4096, ffn=16384`) with N=16, each matmul's
wide-poly drops from arity 26 to **arity 22** — within the GPU
same-point + multifold kernel budget, no CPU fallback needed.

API cascade: `llama_mlp(Vec<EdgeId>, Vec<EdgeId>, Vec<EdgeId>)` →
`llama_block(..., Vec<Witness>, Vec<Witness>, Vec<Witness>, ...)` →
`llama_2_7b(..., Vec<Vec<Witness>>, ...)`. Bench accepts `FFN_SHARDS`
env var (default 1 = un-sharded). `llama3_block` wraps single weights
in `vec![w]` to preserve its behavior.

**Llama-2-7B sweep (single-layer, single-seq, GPU 2 only):**

| `FFN_SHARDS` | Max bucket arity | Prove | Verify | Verified |
|---:|---:|---:|---:|---|
| 1  | 26 (CPU multifold fallback) | 87.6 s | 958 ms | ✓ |
| 2  | 25 (CPU multifold fallback) | 73.2 s | 585 ms | ✓ |
| 4  | 24 (GPU)                    | 67.0 s | 427 ms | ✓ |
| 8  | 23 (GPU)                    | 69.9 s | 319 ms | ✓ |
| 16 | 22 (GPU)                    | **62.0 s** | **374 ms** | ✓ |

Best configuration: `LOGITS_SHARDS=32 FFN_SHARDS=16` — Llama-2-7B
verified end-to-end at **62 s prove / 374 ms verify** on a single A100.

The remaining bottleneck is the *number of fold-tree groups* — with
16 FFN shards the arity-22 bucket has **1842 leaves** across 4 levels
(30 + 7 + 2 + 1 = 40 groups total). Each group does a GPU same-point
sumcheck + multifold + split with its own buffer alloc/free cycle.
Per-group overhead × 40 groups dominates; the longest single bucket
runs ~58 s wall.

Next obvious targets:
1. **GPU buffer pool** to amortize cudaMalloc/free across groups
2. **Multi-GPU bucket dispatch** (already shipped, blocked by host's
   GPU contention)
3. **Batched per-level multifold/sp** — one launch handles every group
   at a level instead of N launches

### Trajectory: Llama-2-7B at full size

| Iteration | Prove | Verify | Notes |
|---|---|---|---|
| First (G with multifold CPU fallback)        | 416 s | 1.0 s | parallelize host multifold next |
| Parallel host multifold                     |  87.6 s | 958 ms | CPU fallback now par-chunks |
| **H: FFN sharding (FFN_SHARDS=16)**         |  **62 s** | **374 ms** | GPU-native for FFN matmuls |

### G: logits sharding + graceful fallbacks

**Logits sharding** — modified `llama_2_7b` to accept `Vec<Witness>` for
the logits head (one weight matrix per shard along the vocab axis).
Each shard becomes its own einsum producing a separate output edge.
For full Llama-2-7B at `LOGITS_SHARDS=32` (4096 × 1024 per shard),
each shard's einsum wide-poly is arity 21 — fits the GPU same-point /
multifold budget. Bench accepts `LOGITS_SHARDS` env override.

**Graceful CPU fallback for GPU-OOM cases**:
- `prove_same_point_gpu_batched` now snapshots the transcript before
  attempting GPU construction. If `new_device_eq_packed_f` /
  `new_device_eq_packed_ternary` / `new_device_eq` returns
  `AllocationFailed`, the transcript is restored and the CPU
  same-point runs. Verifier sees identical state either way.
- `fold_witnesses_gpu` (multifold) now falls back to the (newly
  parallelized) `fold_witnesses_host` when the GPU kernel returns
  `KernelFailed` OR when arity exceeds `ZK4_MULTIFOLD_GPU_MAX_ARITY`
  (default 24). The host fallback uses `par_chunks_mut` over the
  ring-element axis — disjoint output slices, no atomic overhead.
- Fold-tree bucket dispatch goes serial above `ZK4_FOLD_TREE_SERIAL_ARITY`
  (default 25). Avoids the case where multiple very-large-arity
  buckets allocate GPU buffers concurrently and OOM.

**Bug found and fixed in `Dag::prove`**: the reducer was using
`reducer_edge_ids = vec![*edge_ids.last().unwrap()]` — for multi-output
nodes (ScaleDown / ScaleUp / ExpHelper), this is the AUX edge not
`outputs[0]`. The reduced claim got tagged with the wrong `edge_id`
and the verifier rejected. GPT-2 happened to never hit this code path
(its multi-output nodes never accumulated > 1 claim on `outputs[0]`
during backward). Llama-2 does. One-line fix.

### Bench matrix (NUM_LAYERS=1, SEQ_LEN=1, single GPU)

| Config | hidden | ffn | vocab | Logits shards | Prove | Verify | Verified |
|---|---|---|---|---|---|---|---|
| GPT-2 Small        |  768 |  3072 |  —    | n/a | **1.41 s** | 97 ms | ✓ |
| Llama tiny         |  256 |  1024 |   128 |  4  | **0.79 s** | 54 ms | ✓ |
| Llama medium (GPT-2 sized) |  768 |  3072 |  1024 |  1  | **1.03 s** | 100 ms | ✓ |
| Llama mid          | 1024 |  2048 |  4096 |  4  | **1.95 s** | 80 ms | ✓ |
| Llama large        | 1536 |  4096 |  8192 |  8  | **5.02 s** | 283 ms | ✓ |
| **Llama-2-7B (real)** | **4096** | **11008** | **32000** | **32** | **87.6 s** | **958 ms** | **✓** |

Full Llama-2-7B prove is dominated (~80 s of 87 s) by **CPU multifold
fallback for the FFN matmuls** (arity 26 in the SwiGLU's three
projections — `4096 × 16384`). The GPU multifold kernel has a similar
shared-mem / grid-dim limit to the einsum one; sharding the FFN
matmul (analogous to what G does for logits) would lift max bucket
arity to 22 and make the whole pipeline GPU-native. That's the natural
follow-on to G.

### Full Llama-2-7B (F: fused-einsum shared-mem fix)

The `agl_fused_permute_partial_eval_kernel` failed at the 4096 × 32000
logits matmul because the dynamic-shared-memory request (≈ 96 KB for
the lo/hi LUTs at `n = 27, half = 13`) exceeded the default 48 KB
per-block budget. Fix in `cuda/wrapper.cu`: call
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 163*1024)`
once before the first launch — A100 supports up to 164 KB. Small
launches are unaffected (this only raises the *cap*).

Also bumped `TABLE_SIZE_LOG` to 12 (range `[0, 4096)`) for Llama via
`llama2_config.yaml` — at `hidden = 4096`, the LN/RMSNorm `mean_tolerance
= n/2 = 2048` exceeds the GPT-2 default of 1024, so the NonNegative
range check needs the larger table.

**Full Llama-2-7B (NUM_HEADS=32 HEAD_DIM=128, hidden=4096, vocab=32000):**

```
ZK4_GPU_SP_MIN_ARITY=99 MAX_NUM_VARS=27 \
./target/release/llama2 llama2_config.yaml
Prove:    77.4 s
Verify:    1.55 s
Verified:  true
```

CPU-only fold tree at arity 27. GPU same-point still disabled at this
size (each leaf's eq table is 2 GB; 21 leaves × 2 polys × 2 buffers
won't fit in 80 GB). Future levers: shard the logits matmul into
smaller chunks to keep max arity ≤ 22, then re-enable GPU SP; or
multi-GPU distribution.

### D: ternary lift kernel (closes the GPU same-point fallback)

Added `aext2_batched_lift_ternary_single_kernel` + Rust wrapper
`new_device_eq_packed_ternary` mirroring the binary lift path. For each
leaf, upload `(pos, neg)` packed `u64`s (~KB) instead of host-lifted
Ext2 (~MB). Triggered for level-1+ buckets where post-split witnesses
are single-chunk ternary.

Arity-20 ternary bucket setup: **273 ms → 32 ms (8.5×)**. Validation
via `batched_same_point_device_eq_packed_ternary_matches_host` —
matches host-lift across all rounds.

### Leaf-build GPU offload (640 ms → 107 ms)

For each dense edge: build eq table on device from `extended_point`,
then a single batched kernel computes all `b = 21` bit-plane evals
against that shared eq table. Sparse edges with arity ≥ 12 go through
the same path (single-plane case).

New CUDA kernel: `aext2_selective_add_batched_planes_kernel`. For each
of `n_planes` packed binary witnesses, computes
`eval_p = Σ_{i : plane_p[i]=1} eq[i]` against a shared eq table — one
launch handles all planes for an edge. Reduces per-block partials
on the host (small: `num_blocks_x × n_planes × 16 B`).

Rust wrapper: `eq_lagrange::eval_binary_planes_device(claim_pt, planes)`.
Builds eq via `ext2_eq_dp_all_device`, uploads packed planes (concat),
launches the kernel, downloads + reduces partials. Gated on `arity ≥ 12`
in `prove_with_fold_tree` — smaller arities stay on CPU since launch
overhead dominates eq-table compute there.

**Validation**: bit-exact equivalence to CPU
(`eval_binary_planes_device_matches_cpu` test) over 5 random binary
planes at log_n=10. All 271 lib tests pass.

### Multi-GPU bucket dispatch

Plumbing in place; effectiveness gated by free device memory and
co-tenancy. Each fold-tree bucket is assigned to a CUDA device id
from `gpu_device_pool()` (defaults to all visible devices; override
via `ZK4_GPU_DEVICES=2,1,3`). The biggest bucket gets device 0 of the
pool, next biggest device 1, etc.

Inside the rayon task for each bucket, the worker calls
`almost_goldilocks_cuda::set_device(dev)` before any CUDA work — pinning
all subsequent allocations + kernel launches in that task to the
assigned device. Devices are eagerly warmed up at first call (allocate
a tiny buffer on each in parallel) to amortize the ~800 ms CUDA primary
context creation cost away from the critical path.

**Observed in this run** (system shared with other workloads — GPUs 0/1/3
were 61–90% utilized by external jobs; only GPU 2 was idle):

| Config | Fold-tree wall | Prove total |
|---|---|---|
| 1 GPU (GPU 2)                | 949 ms | **1.34 s** |
| 2 GPUs (GPUs 2, 1)           | 801 ms | 1.59 s |
| 3 GPUs (GPUs 2, 1, 3)        | 772 ms | 1.83 s |

Fold-tree wall shrank as expected, but total prove went up: the other
GPUs were contended (someone else's jobs running at 60–90%), so kernel
launches there ran slower than on the idle GPU 2. On a dedicated
multi-GPU host this would distribute cleanly. The code path is correct
(all 271 tests pass; verifier still returns true), and single-GPU mode
is unchanged (1.34 s, matching the pre-change baseline).

### Session trajectory

| Iteration | Prove | Verify | Notes |
|---|---|---|---|
| Session start | 12.94 s | 7 ms | verifier rejected (invalid input) |
| Plane cache (offline `bit_decompose`) | 8.04 s | 7 ms | constants amortized |
| Round-0 binary fast path + parallel-leaf same-point | 3.68 s | 7 ms | still rejected |
| Verifier fix (RMSReciprocal padded vs real dim) | 3.85 s | 247 ms | **verified ✓** |
| Batched transcript absorbs | 3.39 s | 76 ms | prover also benefits |
| GPU batched same-point + on-device eq + binary lift | 1.93 s | 76 ms | GPU finally net-positive |
| GPU leaf build (eq + selective-add) | 1.35 s | 79 ms | leaf build 640 → 107 ms |
| Multi-GPU bucket dispatch | 1.34 s | 76 ms | shipped; gated on per-device load |
| **C: parallel bool sumchecks** + **D: ternary lift kernel** | **~1.0 s** | **~85 ms** | sub-second |
| **E: batched verifier absorbs + multi-output reducer fix + Llama-2 DAG** | **~0.97 s** | **~88 ms** | GPT-2 unchanged; Llama-2 medium config also verifies |
| **F: fused einsum shared-mem fix** | (no GPT-2 change) | (no GPT-2 change) | Full Llama-2-7B end-to-end: 77.4 s prove / 1.55 s verify / verified |
| **G: logits sharding + multifold/SP CPU fallbacks + parallel host multifold** | 1.41 s | 97 ms | Full Llama-2-7B: 87.6 s prove / 958 ms verify / verified |
| **H: FFN sharding (llama_mlp ⇒ Vec<Witness>)** | 1.61 s | 84 ms | Full Llama-2-7B (FFN_SHARDS=16): **62 s prove / 374 ms verify** |

Distance to zk-torch-3 (898 ms prove, 41 ms verify): **~1.1× prover** (essentially parity), 2× verifier — down from 14× / 6× at session start.

Multi-GPU is a no-op on this contended host (other workloads hogging
GPUs 0/1/3); on a dedicated 4-GPU box the wall-time savings on
fold-tree from spreading the arity-22 and arity-20 buckets across
devices will land directly in `Prove total`.

`bit_decompose` for constants is fully offline; per-leaf eq gen and
small-alphabet round-0 sumcheck cut fold-tree time by 2.5× over the
previous configuration.

### Fold tree breakdown (arity-22 final group — single dominant bucket)
- `InstanceState::new` (per-leaf eq table build, parallel across leaves): ~750 ms
- Same-point sumcheck rounds 0–21 (msg + in-place fold): ~600 ms
- Multifold + bucket overhead: ~750 ms
- Bucket wall: 2,311 ms

The optimizations that landed:
1. **Round-0 binary/ternary fast path** — selective-add accumulators replace 3 Ext2 muls per pair (4 accs for binary, 8 for ternary). Round 0 round-msg: ~315 ms → 0.8 ms at arity 22.
2. **Outer-parallel-across-leaves, serial-inside-leaf** — same-point fold went from 1.5 s → 147 ms on round 0 of the arity-22 bucket.
3. **In-place eq/f fold** — zero per-round Vec allocs once witness is Ext2.
4. **Parallel `InstanceState::new`** — 42 leaves' 64 MB eq tables build concurrently.

### Verifier speedup: batched transcript absorbs

Initial post-fix verify took **247 ms**, dominated (~200 ms) by
`absorb_group_commitments` in the fold-tree verify. The hot pattern was

```rust
for &row in &commitment.rows {
    transcript.append_u64(b"ft_mf_comm", row);
}
```

Each commitment has 960 `u64`s, and `append_u64` re-absorbs the 11-byte
label per call → ~12 sponge absorbs per `u64` × 960 rows × 13 chunks ×
many groups ≈ 1 M absorbs ≈ 130 K Monolith permutations.

Added `Transcript::append_u64_slice(label, &[u64])` — label absorbed
once, then the slice — and switched all per-row loops in
`absorb_group_commitments` + the per-bucket tip absorbs to use it.

**Result: verify 247 ms → 76 ms (3.3×).** The prover also benefits
(uses the same helper) — prove dropped ~470 ms from the same change.

### GPU batched same-point sumcheck — fully on-device

A custom CUDA kernel `aext2_sumcheck_batched_round_message_kernel` (and
`..._batched_fold_kernel`) is implemented in
`cuda_almost_goldilocks/almost_sumcheck_prover.cuh` and wrapped by
`GpuBatchedSamePointState` in `almost-goldilocks-cuda-rs`. Layout:
`[leaf_0_eq | leaf_0_f | leaf_1_eq | leaf_1_f | …]` co-resident on device;
one kernel launch per round handles ALL leaves of a per-arity bucket.
Correctness validated against the unbatched `GpuSumcheckStateExt2` over
all 18 rounds at arity 18 (test:
`batched_same_point_arity18_single_leaf`).

**Initial result (host-built eq+f, all uploaded)** at arity-22 bucket:

| Phase | Time |
|---|---|
| Host build (eq tables × 42)         | 1.07 s |
| Host→device upload (5.4 GB)         | **4.39 s** |
| GPU sumcheck (22 rounds, batched)   | 27 ms |
| Total                                | ~5.5 s — slower than CPU |

**Fix: on-device eq + on-device binary lift.** Two follow-on kernels:

1. `ext2_eq_dp_all_device` (already in the CUDA crate) — for each leaf,
   upload only the small `claim_pt` (arity × 16 B) and build the eq
   table on device. D2D-copy the result into the leaf's eq slot in the
   batched buffer.
2. `aext2_batched_lift_binary_kernel` (new) — lift binary `f` from
   packed `u64` bitmasks to Ext2 directly in the f slot of the batched
   buffer. Upload shrinks per leaf from ~128 MB (Ext2 form) to ~64 KB
   (packed bits).

**Final result at arity-22 / 42-leaf bucket**:

| Phase | Time |
|---|---|
| Host claim_pts + packed concat | ~50 ms |
| GPU setup (eq build + lift + alloc)  | ~270 ms (incl. cudaMalloc of d_polys + d_scratch) |
| GPU sumcheck (22 rounds, batched)   | 26 ms |
| Total                                | ~350 ms |

**Net effect on online prove (GPU SP enabled for arity ≥ 18)**:
- 3.39 s → **1.93 s** (43% off, ~2× zk-torch-3 baseline now).
- Fold-tree wall: 2.78 s → 0.92 s.

Default threshold: `ZK4_GPU_SP_MIN_ARITY=18` (set the env var to
override; lowering it can cause GPU OOM when many buckets allocate
concurrent device buffers).

**Validation**: bit-exact equivalence to the all-host construction
across all rounds (`batched_same_point_device_eq_packed_f_matches_host`
test); all 271 lib tests pass.

### Verifier correctness

A separate bug in `RMSReciprocal::run` was found and fixed during this
session: the advice was averaging over the **padded** last-axis size
(`d_pad`, e.g. 1024) while the protocol's `llama_rms_norm` averages over
the **real** size (`self.dim`, e.g. 768). The mismatch made the
reciprocity check `r²·mean(x²) ≈ 1` off by a factor of `d_pad/dim`,
tripping the NonNegative tolerance gate (±2). Fix: use `self.dim` in
`RMSReciprocal::run`. The bench now verifies cleanly with non-zero input
(`small_varied_input`) — values clustered around 1.0 (= field 1024 at
SF=10) so `mean(x²)` doesn't underflow in fixed-point.

### Inside the fold tree (buckets run in parallel across arities)

| Bucket arity | Leaves | Levels | Time | Note |
|---|---|---|---|---|
| **22** | 42 | 1 | **7,677 ms** | ← single dominant bucket; FFN W₁/W₂ b=21 planes |
| 20 | 99 | 2 | 5,448 ms | NonNegative sparse chunks at 2² × 4M |
| 18 | 30 | 1 | 878 ms | |
| 12 | 118 | 2 | 176 ms | |
| 10 | 336 | 3 | 326 ms | |
| 17 | 1 | 1 | 47 ms | |
| 16 | 1 | 1 | 33 ms | |
| 8  | 49 | 1 | 24 ms | |
| 6  | 63 | 1 | 18 ms | |
| 9  | 1 | 1 | 11 ms | |

Wall time = max of bucket times (buckets dispatched via `rayon::par_iter`)
≈ 7.7 s, dominated by **bucket 22**.

---

## 3. zk-torch-3 prover op-by-op

| Stage | Time | % of prove |
|---|---|---|
| Backward — per-op proves (Einsum dominates: 67 ms / 56 nodes) | 75 ms | 9.4% |
| Lookup proofs (range + two_pow)                                | 83 ms | 10.4% |
| Opening reducers                                                | 1 ms  | 0.1% |
| **Opening proofs (Basefold, wall)**                             | **627 ms** | **78.8%** |
| ↳ 55 CPU tasks (n ≤ 14) @ 48 threads                            | 113 ms (within wall) |  |
| ↳ 30 GPU tasks (n > 14, 34 queries) @ 4 GPUs                    | 617 ms (within wall) |  |
| Reducer (intra-backward)                                        | 7 ms  | 0.9% |
| **Prove total** | **795 ms** | 100% |
| Commit | 104 ms | — |
| Verify | 41 ms  | — |

**Bottleneck**: opening proofs (79% of prove, GPU-bound).

---

## 4. Where the gap lives — root causes

### 4.1 Fold-tree arity-22 bucket (7.7 s in zk-torch-4 vs Basefold openings 0.63 s in zk-torch-3) — **12× gap, 58% of zk-torch-4 prove**

The arity-22 bucket holds the two FFN weight matrices (W₁ at `768×3072
→ padded 1024×4096 = 2²²`, W₂ at `3072×768 → padded 4096×1024 = 2²²`),
each with `b = 21–24` bit planes. So ~42 leaves at arity 22.

Cost = same-point sumcheck on 42 leaves at arity 22 = `42 leaves × 22
rounds × 2²¹ Ext2 multiplies/round` ≈ 1.85 G Ext2 ops, **on CPU**.

zk-torch-3 commits each as one Basefold edge and runs ONE opening per
edge on GPU. Total wall time 627 ms across 4 GPUs.

**The fix that closes most of this gap**: GPU sum-of-products
same-point sumcheck (see §6 option A).

### 4.2 Per-plane MLE eval (4.1 s in zk-torch-4) — **31% of zk-torch-4 prove**

740 plane evaluations in leaf-build. Each is `Σ_x f(x) · eq(R, x)` over
a `2^arity`-element Ext2 vector. Concentrated at arity 22 (48 evals × 4M
Ext2 ops each ≈ 200 M ops × 24).

Currently sequential per plane inside `decompose_witness_for_fold_native`.
**Parallelizing the inner loop with rayon would drop this to <500 ms.**

### 4.3 Online commit cost (155 ms vs 103 ms — 1.5× gap, small)

**Not a major bottleneck after correctly separating online vs offline.**
zk-torch-3 splits commit across 4 GPUs (`Committed 86 edges across 4
GPUs in 0.382s`, of which 220 ms is offline weight commits and 103 ms
is online). zk-torch-4's online commit is 155 ms — 1.5× behind. Could
close to parity with multi-GPU dispatch but the absolute gain is small
(~50 ms) given the 7.5 s fold-tree dominance.

The offline commit gap (1.48 s vs 220 ms, 6.7× slower) is also
multi-GPU sharding but doesn't impact online prover time. Worth
addressing if model loading becomes a UX concern, not for prove speed.

---

## 5. What's already optimized (history)

| Change | Win |
|---|---|
| K_BITS=4 fix in `ExpHelper` (was 16) | Removed arity-28 bucket; ~17% prove time |
| M' optimization (commit at native arity) | Commit at hidden=8: 70 s → ~1.6 s; eliminated `2^max_num_vars` broadcast |
| Per-arity fold-tree buckets | Decoupled small-arity buckets from large; eliminated cross-arity broadcast |
| Parallel bucket dispatch (rayon) | Buckets run concurrently; ~35% on prove |
| Aux split (zk-torch-2 z-t-2 style) | Eliminated arity-27/28 buckets entirely; OOM → completes; ~25 s → 13 s |
| Rayon inner-loop parallelism in CPU same-point sumcheck | ~40% on prove |
| Zero-extend (not random-extend) claim point to arity ≥ 6 | Correctness fix for short-arity witnesses |
| Switch all-binary multifold path to `multifold_mixed_witness_tc_fused` (was non-TC `multifold_witness`) | ~3% on fold tree wall at current scale; designed to scale at large N_ring |
| Correctly separate online vs offline commit time | Commit accounting: 1.76 s "total" → 0.155 s online + 1.48 s offline (amortized) |
| **Hoist eq table out of per-plane MLE eval** | Per-plane MLE eval: 3,873 ms → 237 ms (**16×** — was rebuilding eq(R,x) once per bit plane; now once per edge, reused for all b planes) |
| **Parallel fused bit_decompose (no `Vec<Vec<bool>>` intermediate)** | bit_decompose+pack: 665 ms → 63 ms (**10×** — pack 64 values × b bits per rayon task) |
| **Plane cache (`GpuAjtaiStore.planes_cache`)** | bit_decompose moves entirely offline for constants and online-commit-time-only for activations. Leaf build no longer re-decomposes. Net online prove: 8,536 → 8,035 ms |

### Session 2026-05-26 — Llama-2-7B (1-layer, LOGITS_SHARDS=32 FFN_SHARDS=16 MAX_NUM_VARS=22), single A100 (CUDA_VISIBLE_DEVICES=2). Prove ~20 s → ~9 s; verify 0.35 s; verified=true. GPT-2 Small 0.53 s. zk-torch-3 baseline for this config: 4.47 s.

| Change | Win |
|---|---|
| **`group_size_for_arity`** — cap fold-tree group at 31 (not 63) for arity ≥ 24 | arity-24 bucket 14.4 s → 1.5 s. M=63 @ arity-24 needs ~64 GB GPU same-point state → OOM → CPU fell back at 7–9 s/group. Smaller group only relaxes the SuperNeo norm bound (split still emits 13 chunks); costs ≤1 extra level. Prover+verifier both derive it from `max_num_vars`. |
| **`pool_take` best-fit + OOM hygiene** | (a) Reuse any cached buffer in [size, 4·size] — kills the multi-GB cudaFree+cudaMalloc churn when consecutive groups have different leaf counts (arity-24 M=22 group 878 ms → 104 ms). (b) On OOM, consume the sticky `cudaErrorMemoryAllocation` (else the next kernel's `cudaGetLastError` misreports `KernelFailed`), evict stale cross-bucket buffers, retry. |
| **Eliminate per-group eq-table D2H** | The 64 MB eq(R) table was rebuilt+downloaded per internal group (~36 ms; the D2H copy over pageable memory is the entire cost — alloc+GPU-build <0.5 ms). Removed via two identities: `combined_y = Σ2^i·chunk_eval[i]` (split decomposition — multifold defers `combined_y`, verifier ignores internal-group `combined_claim`); and `chunk_i(R) = eval(pos_i) − eval(neg_i)` via `eval_binary_planes_device` (builds eq on-device, returns only 26 scalars). arity-22 bucket 5.84 s → 4.23 s. |

After this session the dominant cost is the GPU same-point sumcheck itself (sp ≈ 2.68 s in the arity-22 bucket: setup/lift 0.86 s + round messages 1.42 s; fold is async-overlapped). Next lever is a **binary round-0 fused kernel** — fold round 0 directly from packed bits (selective-add message, no Ext2 lift), producing the round-1 Ext2 witness. Estimated ~0.9 s. (Note: the `[gpu_sp] fold=` timer is an async artifact — trust the per-group `[group internal] sp=` wall time.)

---

## 6. Remaining bottlenecks — recommended next steps

### Option A — GPU same-point sumcheck (highest leverage)

**Target**: arity-22 bucket (7.7 s → ~1 s estimated).

The existing `GpuSumcheckStateExt2` handles **product sumchecks**
(`Π_j poly_j`), not **sum-of-products** (`Σ_i α^i · Π_j poly_{i,j}`)
which is the same-point sumcheck. Per-leaf dispatch was tried — too
much launch overhead, **no win**.

Real win requires a custom CUDA kernel that processes **all leaves of a
bucket in one launch** per round:
- Input: per-leaf `(eq_i, f_i)` buffers + `α_i` weights
- Output: degree-2 round message `[T(0), T(1), T(2)]`
- Folds applied per-instance after the host samples `r`

Estimated work: ~150 LOC CUDA + Rust binding. The function
`prove_same_point_gpu` in `src/fold/same_point_sumcheck.rs` already
plumbs the per-leaf dispatch + bit-exact CPU oracle test
(`gpu_same_point_matches_cpu_uniform_arity`); the new kernel would be a
drop-in replacement for the inner loops.

### Option B — Multi-GPU commit dispatch (small win, deprioritized)

**Target**: online commit (155 ms → ~30 ms). Offline commit also helped
(1.48 s → ~0.3 s) but doesn't impact online prove time.

Match zk-torch-3's per-edge GPU sharding. zk-torch-3's source (e.g.,
`crate::commit::basefold::commit_dag_in_parallel`) is a reference. Per
edge → GPU index (round-robin or load-balanced).

**Now ranked below A and C** since online commit is already 1.2% of
prove time after the offline/online split correction. A + C alone would
close most of the remaining gap.

### Option C — Parallelize per-plane MLE eval

**Target**: leaf-build per-plane MLE eval (4.1 s → 0.5 s).

In `dag/fold_integration.rs::decompose_witness_for_fold_native`, the
loop:

```rust
let plane_evals: Vec<_> = planes.iter().map(|p| {
    FoldData::Binary(p.clone()).evaluate_at_ext2(extended_point)
}).collect();
```

Change `.iter().map()` → `.par_iter().map()` and parallelize inside
`FoldData::Binary::evaluate_at_ext2` for arity-22 plane-MLE evaluations.

### A + C combined estimate (revised)

| Stage | Before | After A + C |
|---|---|---|
| Commit (online) | 155 ms | 155 ms |
| Backward | 261 ms | 261 ms |
| Leaf build — bit decompose | 658 ms | 658 ms |
| Leaf build — MLE eval | 3,873 ms | ~500 ms (C) |
| Fold tree | 7,517 ms | ~1,500 ms (A) |
| **Online prove total** | **12,941 ms** | **~3,100 ms** |

vs zk-torch-3's 898 ms online — ratio drops from **~14× → ~3.5×**.
Adding B (multi-GPU commit) trims another ~125 ms (~3,000 ms total).

Closing the remaining ~3× to zk-torch-3 needs protocol-level
restructuring — primarily reducing the bit-plane count `b` (currently
`b=21` for signed `[-2^20, 2^20)` decomposition) or amortizing the
fold-tree leaf count by changing the commit granularity.

---

## 7. Architectural notes — why some gaps are irreducible

zk-torch-4's fold tree visits `b × num_committed_edges` leaves (24 × 85
≈ 2040 in this bench), each Ajtai-committed at native arity.
zk-torch-3 has one Basefold opening per edge (85 in this bench).

This is a **fundamental factor-of-`b` (~24×) gap** in leaf count. The
fold tree shines at verification (just transcript replay; no FRI-style
queries) but pays for that with much more prover-side multifold work.

A protocol-level change to reduce `b` (the bit-plane count) would
help — currently `b = 21` for signed two's-complement on `i20`. If the
witness values are bounded by `2^14` (e.g., scaled differently),
`b = 15` would shave fold-tree cost by ~30%.

---

## 8. Test-harness reminders for future runs

```bash
# zk-torch-4 GPT-2 bench (NUM_LAYERS=1 default; bench_config.yaml in CWD)
cd /scratch/bjchen4_icgpu/goldilocks/zk-torch-4
cargo build --release --bin gpt2 --offline
ZK4_TIMING=1 NUM_LAYERS=1 SEQ_LEN=1 ./target/release/gpt2 bench_config.yaml

# zk-torch-3 reference
cd /scratch/bjchen4_icgpu/goldilocks/zk-torch-3
NUM_LAYERS=1 SEQ_LEN=1 ./target/release/gpt2
```

Timing output gated by `ZK4_TIMING=1` so the test suite stays quiet.
Per-bucket fold-tree times printed from `src/fold/tree.rs`'s
`prove_fold_tree`. Per-stage prove breakdown printed from
`src/dag/fold_integration.rs`'s `prove_with_fold_tree`.

Verify is intentionally permissive in `bin/gpt2.rs` (prints warning
instead of asserting) so the bench completes even with random-weight
range-table escapes.

---

## 9. GPT-2 wins, Llama loses — gap analysis (2026-05-29)

### Why zk-4 beats zk-3 on GPT-2 but not on Llama

zk-4 vs zk-3, 1-layer, seq=1, contested GPU 2 (other users running
~50 GB ML workloads on the same card — both systems pay the contention
tax):

| model | zk-3 prove | zk-4 default | zk-4 best | winner |
|---|---|---|---|---|
| GPT-2 (h=768)  | 3.2 s | 1.4 s | 0.9 s (base=64) | zk-4 (3.6×) |
| Llama (h=4096) | 7.6 s | 12.7 s | **7.8 s** (FFN=8, base=16) | tie |

Default = `FFN_SHARDS=16, ZK4_BASE=2` (the previous Llama bench command).

The scaling factor from GPT-2 → Llama on the same prover:

| | GPT-2 → Llama scaling |
|---|---|
| zk-3 Basefold | 2.4× |
| zk-4 fold tree (default) | 9.1× |
| zk-4 fold tree (best)    | 8.7× |

zk-4 scales 3-4× worse with hidden dim than zk-3. That's the entire
story — both win small models (GPT-2 hidden=768) and lose big ones
(Llama hidden=4096) compared to whatever advantage they had at the
small size.

### Where the Llama time goes (zk-4 default, ZK4_TIMING=1)

```
[prove] backward (sumcheck + reducer): 1.53 s
[prove] leaf build total (3995 leaves): 2.82 s
[prove] fold tree:                      8.31 s
   bucket arity=22 leaves=1842 levels=4 → 7.67 s   ← dominant
   bucket arity=10 leaves=1687 levels=3 → 1.69 s
   bucket arity=24 leaves=84  levels=2 → 2.62 s
   ...
Prove total: 12.66 s
```

vs GPT-2 (zk-4 default):

```
[prove] backward: 0.22 s
[prove] leaf build (740 leaves): 0.38 s
[prove] fold tree: 0.89 s
   bucket arity=22 leaves=42 levels=1 → 0.46 s
   ...
Prove total: 1.49 s
```

**Two structural multipliers** turn the same hidden-dim ratio (5.3×)
into a 9× prove-time gap:

1. **Per-edge leaf amplification (`× b`)**. The Ajtai SIS commitment
   forces each value into `b = 21` signed bit-planes; each plane is a
   leaf in the fold tree. zk-3 Basefold opens each polynomial as ONE
   opening. Llama has ~70 committed polynomials → 70 openings in zk-3
   vs ~1500 leaves in zk-4 *at the same arity*.

2. **Bucket concentration at arity ≈ log₂(hidden · ffn_shard)**.
   - GPT-2: largest weights are 768 × 3072 → arity 21, only 2 FFN
     edges. Bucket-22 has 42 leaves (1 group, ~460 ms).
   - Llama (FFN_SHARDS=16): 16 × 3 = 48 FFN weight edges at
     4096 × 1024 → arity 22, plus FFN intermediates. Bucket-22 has
     1842 leaves (30+ groups across 4 fold-tree levels, ~7.7 s).

   Per-group cost in the bucket is ~200 ms (GPU setup + same-point
   sumcheck + multifold, single-stream serial). 30+ groups × 200 ms ≈
   the 7.7 s we see.

GPT-2 doesn't hit (2) because hidden=768 is small enough that no single
bucket has more than ~40 leaves. Llama hits both (1) and (2)
simultaneously, and (2) dominates.

### Closing the gap — what works, what doesn't

Sweep on the same GPU (contested), best of 3 runs:

| FFN_SHARDS | ZK4_BASE | Prove | Notes |
|---|---|---|---|
| 16 | 2 (default) | 12.7 s | baseline |
| 16 | 4 | 11.6 s | -8% |
| 16 | 16 | 10.3 s | -19% |
| 16 | 64 | PANIC | norm bound at arity-20 mixed group |
| 8  | 2 | 10.3 s | -19% |
| 8  | 4 | 9.0 s | -29% |
| 8  | **16** | **7.7 s** | **-39% — stable across 5 runs** |
| 8  | 64 | 7.4 s when it works, panics ~40% | flaky GPU OOM / norm |
| 4  | 2 | OOM | arity-24 multifold runs out of device memory |
| 4  | 16 | PANIC | shared-eq allocation |
| 4  | 64 | 7.9 s | works but same flakiness as ffn=8 base=64 |

**Best stable config: `LOGITS_SHARDS=32 FFN_SHARDS=8 MAX_NUM_VARS=23
ZK4_BASE=16`**. Drops Llama prove from 12.7 s → 7.8 s (-39%), matching
zk-3 on the same hardware/contention. Fold-tree breakdown:

```
arity=22 leaves=218 levels=2 → 3.00 s   (was 1842 leaves, 4 levels, 7.7s)
arity=23 leaves=208 levels=2 → 3.52 s   (new, replaces some arity-22 work)
arity=10 leaves=679 levels=3 → 1.47 s
Total fold tree: 4.09 s
```

Why base=16 helps: each weight edge produces `ceil(b/log₂β) = 6` digit-
planes instead of `b = 21` bit-planes. Arity-22 leaf count drops
~8.4×. The digit-aware multifold (`fold_witnesses_gpu_digit_path`,
landed in commits `28b352d` + `77351bf`) routes each digit-plane
through K=4 ternary chunks with γ-scaling, so per-leaf cost goes up
modestly but the net is a big win.

Why base=64 is flaky:
- Norm bound: `M · T · (β−1) < 2^13`. At M=63, T=128, that needs
  β−1 ≤ 1, i.e. β=2. base=64 only survives because random γ-coefs
  don't hit worst case — sometimes they do.
- Each base-64 leaf decomposes into 6 virtual binary planes; the
  shared-eq same-point GPU path allocates ~6× more device memory
  than base=16. On a contested GPU it hits `AllocationFailed`.

Why FFN_SHARDS=4 fails: pushes attention/FFN matmuls to arity 24,
where the multifold GPU kernel scratch hits the device-memory limit
(`Allocation failed: upload big_neg`).

### Why the remaining 1.8× gap is structural

Estimated uncontested numbers (scale by ~3× contention factor observed
on GPT-2 zk-3): zk-3 Llama ≈ 2.5 s, zk-4 best ≈ 4.0 s. Still 1.6×
slower.

The fundamental disadvantage is the `× b` leaf amplification — even
at base=16 (which collapses `b=21` to 6 digit-planes), zk-4 has 6× more
"opening tasks" than zk-3. Each opening task does GPU work proportional
to `2^arity`, so total prover work ~ `6 × Σ 2^arity_i`. zk-3 Basefold's
work ~ `1 × Σ 2^arity_i · log(2^arity_i) ≈ 22 × Σ 2^arity_i` per
edge, which sounds worse but actually the per-edge constant factor is
much smaller (no per-bucket setup, no multi-level recursion).

To beat zk-3 on Llama would require ONE of:

1. **Smaller `b`**: drop from 21 to ~5-8 bit-planes. Currently
   impossible because committed values include scale-factor × matmul
   accumulation, which doesn't fit in `2^5 = 32`.
2. **Base > 64**: norm bound forbids it under the current SuperNeo
   parameter set (`B = 2^13, T = 128, M = 63`).
3. **Different commit scheme**: e.g., a Brakedown-style or
   linear-code-based PCS that doesn't bit-decompose. Out of scope here.
4. **Per-stream GPU parallelism** for the 30+ groups in the arity-22
   bucket (per `PROVER_PERFORMANCE.md` §I "Diagnosis"). Single biggest
   non-protocol lever still on the table; not yet attempted.

So: gap narrowed from 1.9× to ~1.0× on contested hardware (matches
zk-3); estimated 1.6× on uncontested. Closing the last 1.6× requires
protocol changes (1-3) or non-trivial CUDA-stream work (4).

### Reproduction

```bash
# zk-3 Llama (baseline)
cd /scratch/bjchen4_icgpu/goldilocks/zk-torch-3
CUDA_VISIBLE_DEVICES=2 NUM_LAYERS=1 SEQ_LEN=1 ./target/release/llama

# zk-4 Llama BEST (this section)
cd /scratch/bjchen4_icgpu/goldilocks/zk-torch-4
CUDA_VISIBLE_DEVICES=2 NUM_LAYERS=1 SEQ_LEN=1 \
  LOGITS_SHARDS=32 FFN_SHARDS=8 MAX_NUM_VARS=23 ZK4_BASE=16 \
  ./target/release/llama2 llama2_config.yaml

# zk-4 GPT-2 BEST
CUDA_VISIBLE_DEVICES=2 NUM_LAYERS=1 SEQ_LEN=1 ZK4_BASE=64 \
  ./target/release/gpt2 bench_config.yaml
```

---

## 10. Full-depth (12L / 32L) — measured + extrapolated (2026-05-30)

Tested the §9 "best 1L" config (`FFN_SHARDS=8 ZK4_BASE=16`) as a new
default for Llama. **It regresses sharply at ≥ 4L** on the contested
GPU 2 because base=16's wider per-leaf digit-plane buffers OOM during
same-point sumcheck, falling back to a slow CPU path. Reverted.

Final defaults left at `FFN_SHARDS=16 ZK4_BASE=2` for robustness. Users
benchmarking single-layer can opt in to the §9 config via env override.

### Measured (contested GPU 2, ~30 GB free during runs)

| Model | L | zk-3 prove | zk-4 default | zk-4 §9 config |
|---|---|---|---|---|
| GPT-2  | 1  | 3.27 s | 1.49 s | 0.93 s (base=64) |
| GPT-2  | 4  | 12.44 s | 5.75 s | n/a |
| GPT-2  | 5  | 15.56 s | 6.80 s | n/a |
| GPT-2  | 6+ | **OOM**   | works  | — |
| GPT-2  | 12 | OOM     | **13–16 s** ✓ | — |
| Llama  | 1  | 9.02 s | 12.7 s | **7.80 s** |
| Llama  | 2  | 16.23 s | 21.4 s | 17.60 s |
| Llama  | 4  | 28.23 s | 37.92 s | 119.7 s ✗ (GPU OOM → CPU fallback) |
| Llama  | ≥8 | OOM     | OOM    | OOM |

`Llama 1→4L` marginal per-layer: zk-3 ≈ 6.5 s, zk-4 default ≈ 8.5 s,
zk-4 §9-config ≈ 9.8 s (1→2L) but blows up at 4L.

### Full-depth — uncontested baselines (PROVER_PERFORMANCE.md history)

Numbers from earlier uncontested-GPU runs, before the GPU-2 VLLM
workloads started:

| Model     | L  | zk-3 prove | zk-4 prove (defaults) | ratio (4/3) |
|-----------|----|------------|----------------------|-------------|
| GPT-2     | 12 | 9.50 s     | **4.18 s**           | zk-4 **2.3×** |
| Llama-2   | 32 | 46.3 s     | **136 s**            | zk-3 **2.9×** |

The contested-GPU 1L numbers (zk-3 9.02 vs uncontested 2.23 = ~4×
contention factor, zk-4 12.7 vs uncontested 6.55 = ~2× contention
factor) suggest zk-4 is less GPU-bottlenecked by other workloads.
Extrapolating the §9 1L improvement (−39 %) to uncontested 32L would
give ≈ 83 s for zk-4 §9-config — still ≈ 1.8× zk-3's 46 s. Cannot
verify directly: at 32L the §9 config OOMs even on uncontested GPUs
(per the earlier 4L regression).

### Bottom line for the user's question

* **GPT-2 (full 12L)**: zk-4 wins decisively. ≈ 4.2 s (zk-4) vs ≈ 9.5 s
  (zk-3), 2.3× faster. Unchanged from prior runs.
* **Llama-2 (full 32L)**: zk-3 still wins. Uncontested baselines 46.3 s
  (zk-3) vs 136 s (zk-4 default); the §9 1L speedup doesn't carry to
  32L because base=16 OOMs at multi-layer. **The structural arity-22
  bucket gap (Section 9) compounds with layers**: more layers → more
  edges at arity 22 → more groups → more sequential GPU work and
  more memory pressure simultaneously.

### Reverted defaults

`bin/llama2.rs` defaults restored to `FFN_SHARDS=16 MAX_NUM_VARS=22
ZK4_BASE=2` so multi-layer benches don't break. The §9 best-1L config
is preserved as an opt-in override:

```bash
# Single-layer benchmarks (fastest):
LOGITS_SHARDS=32 FFN_SHARDS=8 MAX_NUM_VARS=23 ZK4_BASE=16 \
  ./target/release/llama2 llama2_config.yaml

# Full-depth (32L; what scales):
LOGITS_SHARDS=32 NUM_LAYERS=32 ./target/release/llama2 llama2_config.yaml
```

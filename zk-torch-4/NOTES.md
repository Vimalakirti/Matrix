# zk-torch-4 — session notes for cross-machine handoff

What landed across recent sessions, and the pitfalls a fresh Claude
session on another machine should know about before extending or
benchmarking any of the bins.

---

## Lib-level fixes (in `src/dag/`)

These were uncovered while porting non-transformer bins from zk-torch-3.
They preserve existing GPT-2/Llama-2 behavior and unblock CV/audio bins.

### `proving.rs` — reducer mixed-arity filter

The reducer combines multiple sumcheck claims on one witness. Conv's
`s_alpha_claim` is a scalar side-channel (`point.len() = 0`), NOT a
multilinear-evaluation claim. The original reducer assumed
`claim.point.len() == witness.n()` and panicked on the eq-table memcpy
for any DAG that emits side-channels (resnet, vgg, unet3d, yolo, whisper).

**Fix:** at the reducer call sites (both prover and verifier), filter
`claim.point.len() == witness.n()` before counting / passing to the
reducer. Empty-point side-channel claims stay separate in
`local_claims` and are handled by their producing block's verify.

### `proving.rs` — self-loop guard restored

Conv pushes its own `y_self_claim` / `s_alpha_claim` back onto its
output edge. Without the `if producer != node_id` guard before
`nodes_to_prove.insert(producer)`, the reducer fix turns the implicit
panic into an infinite loop. zk-torch-3 had this guard with the comment
"Don't re-add self for self-claims (Conv2D)"; the refactor to zk-torch-4
dropped it.

### `proving.rs` — verifier uses `np.produced_claims` directly

Self-claim blocks (Conv2D/3D) read claims by **position** —
`claims[0] = y_self_claim`, `claims[1] = x_claim`, ..., `claims.last()
= out_claim` (5 entries). The verifier was rebuilding flat from
`inputs + local_claims` (3-4 entries) in the wrong order. Fix: pass
`np.produced_claims` (the prover's stored `[produced..., consumed...]`
snapshot) straight to `kind.verify`, mirroring zk-torch-3.

### `mod.rs::should_commit` — skip terminal outputs

Terminal output values (Role::Output with no consumers) can exceed the
Ajtai signed-b-bit range after multi-layer accumulation. They're
public anyway — `output_claims` records the eval and absorbs it into
the transcript. Old behavior was `return self.consumers[edge_id].is_empty()`
(i.e. commit terminal outputs); new behavior is `return false`.

### `whisper.rs` — env-tunable reciprocity tolerance

`WHISPER_RECIP_TOL` env var (default 2) widens the LN's reciprocity
gate `|z − sf| ≤ tol`. Wraps the hardcoded `2` in a `OnceLock`. The
default preserves soundness; smoke runs that can't make the
post-mean variance integer-clean can widen it.

### `whisper.rs` — x_sum broadcast shape fix

`sub(x_sum [1, seq], x_mean_mul_n [1, seq, 1])` was broadcasting to
`[1, seq, seq]` (outer product) because NumPy left-pad turns `[1, seq]`
into `[1, 1, seq]` before broadcasting against `[1, seq, 1]`. Fix:
`change_shape(x_sum, [1, seq, 1])` before the sub. Same pattern fixed
in the ported `bert.rs` and `gptj.rs`.

---

## Working bin configurations

All 11 bins below `Verified: true` on toy configs as of the last
session. Run from `zk-torch-4/`; pass `bench_config.yaml` as `args[1]`
where indicated. Without that arg the bin falls back to
`Config::default()` (`table_size_log = 20`), which gives the sparse-bool
prover an `eq_dense` of ~2^36 Ext2 entries ≈ 1 TB host RAM.

| Bin | Config | Notes |
|---|---|---|
| `gpt2` | default (or bench_config for SEQ_LEN ≥ 16) | Default works; SEQ_LEN ≥ 16 needs bench_config + the layout-aware input. |
| `llama2` | default | Constant `1024` input — layout-agnostic. |
| `bert` | `bench_config.yaml` | LN weights = 1.0 (not zero); other weights zero; input alternates 0/2048 along last axis. |
| `gptj` | `VOCAB=128 NUM_HEADS=4 HEAD_DIM=64 FFN_DIM=1024` (full vocab too big) | Same input pattern as bert. |
| `resnet` | `NUM_LAYERS=1 INPUT_SIZE=32`, `bench_config.yaml` | All `% 2` rand magnitudes. |
| `unet3d` | `NUM_LAYERS=1 INPUT_D/H/W=4`, `bench_config.yaml` | |
| `yolo` | `NUM_STAGES=1 INPUT_SIZE=32`, `bench_config.yaml` | |
| `vgg` (paper) | `NUM_LAYERS=1..3 VGG_VARIANT=11|16`, `bench_config.yaml` | All-zero conv/fc weights, `% 2` biases. |
| `vgg` (verfcnn) | `VGG_STYLE=verfcnn VGG_VARIANT=11 NUM_LAYERS=1`, `bench_config.yaml` | |
| `whisper` | `NUM_ENC_LAYERS=1 NUM_DEC_LAYERS=1 N_AUDIO_CTX=32 N_TEXT_CTX=16 N_STATE=64 N_HEAD=2`, `bench_config.yaml` | Column-major-aware input pattern. |
| `pointpillar` | `NY=16 NX=16 N_PILLARS=32 MAX_POINTS=4`, `bench_config.yaml` | All-zero conv/transpose weights. |

`llama3` exists but verify on the full 8B config needs sharding work
that hasn't been done yet (the DAG hardcodes the un-sharded logits head;
toy configs with `HIDDEN_DIM=512 VOCAB_SIZE=128` may work).

---

## Pitfalls every smoke-run hits

### 1. Pass `bench_config.yaml` as `args[1]`

Without it, `Config::default()` sets `table_size_log = 20`, which
explodes sparse-bool's `eq_dense` to a 1 TB host allocation.
`bench_config.yaml` sets it to 10 (range table covers [0, 1024)).

### 2. Reduce ALL `gen_*` random magnitudes (not just `rand_field_vec`)

CNN bins have multiple random helpers — `rand_field_vec` (inputs),
`gen_conv_weight`, `gen_fc_weight`, `gen_fc_bias`. Each has its own
`% N` modulus. Only reducing `rand_field_vec` is a trap: conv
accumulators still see the unreduced weights. ResNet's 7×7×3 = 147-term
conv with `% 500` weights produces ~270k outputs, way over 1024;
NonNegative clamps every selection to t=0 → `verify_range` correctly
rejects as a soundness violation.

### 3. For deeper CNNs, zero all conv/fc weights

Even `% 2` weights overflow `[0, 1024)` after a single 64→128 stage
(576-term sum × non-trivial ReLU output > 1024). All-zero weights make
every ReLU input = the bias alone (`% 2`), trivially in range. VGG
works this way at any depth.

### 4. The dag's tensor layout is COLUMN-MAJOR

`broadcast_strides` returns `[0, 1, seq_pad]` for shape `[1, seq, h]` —
first axis fastest in the flat buffer. Any bin pattern of the form
`data[i] = f(i % K)` varies along **whichever axis has stride
dividing K**. For shape `[1, seq, h]` padded:

- `flat i = s + h · seq_pad`
- `i % K` varies along `s` (the wrong axis) whenever `K | seq_pad`
- To vary along the LAST axis (the one LN reduces over), use
  `(i / stride_last) % K`

gpt2 silently failed at SEQ_LEN ≥ 16 (any multiple of 16) under the
old `i % 16` pattern. Whisper failed at any seq_len. Fix: pass
`stride_last` into the bin's input generator. Reflected in
gpt2/whisper/bert/gptj.

### 5. Multi-LN networks need LN weights = 1.0 (NOT zero)

With `w_e = 0`, each LN outputs zero; downstream LNs see zero-variance
input → `r = 0` → reciprocity gate fails. BERT has an outer LN
*before* the transformer blocks; that LN consumes all variance if its
weight is zero. Fix: use `1024` (= 1.0 at SF=10) for LN weights
(`attn_norm_w`, `proj_norm_w`, `layer_norm_w`) while keeping LN biases
(`*_norm_b`) and all matmul weights at zero. GPT-J needs the same.
Llama2/3 use RMSNorm without mean subtraction, so constant `1024`
input + zero weights round-trips exactly without this trick.

### 6. WhisperLN needs exact-round input

Whisper-style LN subtracts the per-row mean before the RMS step.
Constant input → `x − mean = 0` → `r = 0` → gate fails. Varied
input → variance > 0 but the integer rounding error in
`r²·mean(x²)/sf` exceeds the default `tolerance = 2` unless variance
is chosen so the round-trip is exact. Use 0/2048 alternating along
the LAST tensor axis: every entry is ±1024 (= ±1.0 in float) after
mean subtraction, so `mean(x²) = 1`, `r = 1`, `z = sf` exactly. The
ported `bert.rs` and `gptj.rs` use the same trick.

---

## Models still missing relative to zk-torch-3

- `dag/deeplabv3plus.rs` — not ported (blocks the deeplabv3 stage of
  the original pointpainting pipeline).
- `dag/dense.rs` — 13-line stub in zk-torch-3, not ported.
- `dag/oneshot.rs` — not ported (blocks 4 oneshot_* bins).
- Bench bins (`bench_einsum`, `bench_permute_partial`,
  `bench_sumcheck`, `bench_thresholds`) — not ported.
- `bin/resnet_mlperf_acc.rs` — easy to add (resnet variant); not done.

---

## File-by-file change summary (recent commits)

- Commit `adb699a`: 6 Tier-1 bin ports + lib fixes (reducer, self-loop,
  verify-flat, terminal-output commit skip, whisper tolerance, whisper
  shape broadcast).
- This commit: bert / gptj / pointpillar / verfcnn-vgg ports (4 DAG
  modules + 3 bins + vgg verfcnn variant + this NOTES.md).

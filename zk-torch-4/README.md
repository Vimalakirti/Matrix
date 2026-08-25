# zk-torch-4

Prover for transformer-model inference proofs on Almost-Goldilocks +
Ajtai SIS + fold-tree. Includes a streaming-inference accumulator
that amortizes PCS openings across N proofs of the same model.

## Build

```bash
cargo build --release
```

All bins land in `target/release/`.

## Quick start

The full model zoo is supported in three proving modes — **monolithic**
(one forward), **streaming** (N inferences sharing weights, amortized via
the accumulator), and **one-shot** (a full `seq_len > 1` / batch in one
proof; for autoregressive decoders this is one-shot AR — a whole T-token
generation in one causal proof). See the coverage table below.

Every bin takes a config YAML as `args[1]`:

- `bench_config.yaml` — scale_factor_log=10, table_size_log=10, table_commit_log=8 (GPT-2 /
  one-shot decoders / encoders / small vision)
- `llama2_config.yaml` — table_size_log=12, table_commit_log=12 (Llama-2)
- `cv_config.yaml` — scale_factor_log=16, table_size_log=18, table_commit_log=8
  (**full-scale vision** — wider fixed-point range for deep CNN activations)

> **`table_commit_log` sets the fold-tree cliff, not `table_size_log`.** Each
> range-check aux chunk commits at arity `input_n + table_commit_log`, and the
> multifold is dense over `2^(arity-6)`. `table_commit_log=8` keeps every
> shipped config off the cliff; it is value-range-independent and sound at any
> setting (the verifier reconstructs the chunked table value), so real-weight
> dynamic range only grows `table_size_log` = chunk *count* (≈ linear, cheap).

## Full-depth commands

### GPT-2 12L (full small)

```bash
# Monolithic
NUM_LAYERS=12 SEQ_LEN=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/gpt2 bench_config.yaml

# Streaming, N=10 inferences, multi-GPU cache enabled
NUM_LAYERS=12 SEQ_LEN=1 N_PROOFS=10 \
  ZK4_STREAM_X_EXT2_CACHE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_gpt2 bench_config.yaml

# Streaming + 8-way DAG partitioning (best — partitions the
# backward-pass sumcheck across GPUs on top of reducer sharding)
NUM_LAYERS=12 SEQ_LEN=1 N_PROOFS=10 NUM_PARTITIONS=8 \
  ZK4_STREAM_X_EXT2_CACHE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_gpt2 bench_config.yaml
```

### Llama-2 32L (full 7B)

```bash
# Monolithic (~140 s prove, ~50 s compile)
NUM_LAYERS=32 SEQ_LEN=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/llama2 llama2_config.yaml

# Monolithic with 8-way DAG partitioning (BIG win — backward-pass
# sumcheck splits across GPUs, the bottleneck of single-GPU prove)
NUM_LAYERS=32 SEQ_LEN=1 NUM_PARTITIONS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/llama2 llama2_config.yaml

# Streaming, N=10, multi-GPU cache + 8-way partitioning (biggest win)
NUM_LAYERS=32 SEQ_LEN=1 N_PROOFS=10 NUM_PARTITIONS=8 \
  ZK4_STREAM_X_EXT2_CACHE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_llama2 llama2_config.yaml

# Streaming, N=10, multi-GPU cache (no partitioning — reducer
# multi-GPU only, backward pass on single GPU)
NUM_LAYERS=32 SEQ_LEN=1 N_PROOFS=10 \
  ZK4_STREAM_X_EXT2_CACHE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_llama2 llama2_config.yaml

# Fallback if cache OOMs on a specific GPU (still 1.35× monolithic)
NUM_LAYERS=32 SEQ_LEN=1 N_PROOFS=10 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_llama2 llama2_config.yaml
```

### Llama-3 8B (full)

The `llama3` bin **defaults to the real 8B shape** (hidden 4096, 32 heads, 8 KV
heads, head_dim 128, FFN 14336, vocab 128256, `MAX_NUM_VARS=29`) — so a full run
is just the layer count. GQA via `NUM_KV_HEADS`. Memory-heavy (2^29 commit key);
use the full 8-GPU node, or override dims for smaller cards.

> ⚠️ **Real decoders need `llama2_config.yaml`, NOT `bench_config.yaml`.**
> bench_config's `table_size_log=10` range table only covers `[0, 1024)`; at
> hidden ≥ 2048 the matmul accumulations exceed it, the NonNegative range check
> silently mis-selects, and the proof fails to verify (`Verified: false`) with no
> other error. A `[range] WARNING: … value(s) fall outside the table` is now
> printed at witness-gen time when this happens — raise `table_size_log`.
> `llama2_config.yaml` (table_size_log=12) covers hidden ≤ 4096. (GPT-2's hidden
> 768 fits under 1024, which is why GPT-2 uses bench_config.)
>
> The `bench_streaming_oneshot_*` bins **default to the real per-model shape**,
> so the shape knobs below are optional (shown for clarity); override with the
> small shape (`NUM_HEADS=8 HEAD_DIM=64 FFN_DIM=2048`) + `bench_config.yaml` for a
> fast smoke. `MAX_NUM_VARS ≥ 26` fits the `4096×16384 = 2^26` FFN edge. If the
> full-vocab LM-head leaf exceeds GPU memory you'll see `[einsum] … host fallback`
> (correct, slower) — raise `VOCAB_SHARDS` or lower `NUM_PARTITIONS`.

```bash
# Monolithic, full 32L 8B (1 token) — note llama2_config (bigger range table)
NUM_LAYERS=32 SEQ_LEN=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/llama3 llama2_config.yaml

# Streaming one-shot AR, N=10, 256-token gen, FULL 8B shape + full vocab
NUM_LAYERS=32 SEQ_LEN=256 N_PROOFS=10 NUM_PARTITIONS=8 \
  NUM_HEADS=32 NUM_KV_HEADS=8 HEAD_DIM=128 FFN_DIM=14336 \
  VOCAB_SIZE=128256 VOCAB_SHARDS=64 MAX_NUM_VARS=26 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_oneshot_llama3 llama2_config.yaml
```

### GPT-J 6B (full)

The `gptj` bins default to the real 6B shape (16 heads, head_dim 256, hidden
4096, FFN 16384, vocab 50400); full depth is 28 layers. Same ⚠️ rule as Llama-3
above: real hidden 4096 needs `llama2_config.yaml` (table_size_log=12), not
bench_config; `MAX_NUM_VARS=28` fits the full un-sharded vocab head.

```bash
# Monolithic, full 28L 6B (1 token)
NUM_LAYERS=28 SEQ_LEN=1 MAX_NUM_VARS=28 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/gptj llama2_config.yaml

# Streaming one-shot AR, N=10, 256-token gen, FULL 6B shape + full vocab
NUM_LAYERS=28 SEQ_LEN=256 N_PROOFS=10 NUM_PARTITIONS=8 \
  NUM_HEADS=16 HEAD_DIM=256 FFN_DIM=16384 \
  VOCAB_SIZE=50400 VOCAB_SHARDS=64 MAX_NUM_VARS=26 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_oneshot_gptj llama2_config.yaml

# Fast amortization smoke (small hidden 512 → bench_config's table is fine):
NUM_LAYERS=28 SEQ_LEN=64 N_PROOFS=10 NUM_PARTITIONS=8 \
  NUM_HEADS=8 HEAD_DIM=64 FFN_DIM=2048 \
  VOCAB_SIZE=50400 VOCAB_SHARDS=32 MAX_NUM_VARS=25 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_oneshot_gptj bench_config.yaml
```

### BERT-Large (full)

Encoder (no AR) — the `bert` bin is fixed at BERT-Large shape (hidden 1024,
16 heads); set depth + sequence length. Streaming amortizes weights across N
encodings.

```bash
# Monolithic, full 24L, seq 512
NUM_LAYERS=24 SEQ_LEN=512 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bert bench_config.yaml

# Streaming, N=10, 8-way partition
NUM_LAYERS=24 SEQ_LEN=512 N_PROOFS=10 NUM_PARTITIONS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_bert bench_config.yaml
```

### Whisper tiny (full)

Encoder + decoder + cross-attention. The `whisper` bins **default to the real
tiny.en shape** (enc 4 / dec 4, n_audio_ctx 1500, n_text_ctx 448, n_state 384,
n_head 6) — so a full run is just the bin. The 1500-frame audio context is the
heavy part; needs the full node, or shrink `N_AUDIO_CTX` for smaller cards.

```bash
# Monolithic, full tiny.en (defaults)
NUM_ENC_LAYERS=4 NUM_DEC_LAYERS=4 N_AUDIO_CTX=1500 N_TEXT_CTX=448 N_STATE=384 N_HEAD=6 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/whisper bench_config.yaml

# Streaming, N=10, 8-way partition
NUM_ENC_LAYERS=4 NUM_DEC_LAYERS=4 N_AUDIO_CTX=1500 N_TEXT_CTX=448 N_STATE=384 N_HEAD=6 \
  N_PROOFS=10 NUM_PARTITIONS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_whisper bench_config.yaml
```

> Full Llama-3-8B / GPT-J-6B / Whisper-1500 are large; the bins default to true
> model scale and target an 8-GPU node. On fewer/smaller cards, override the
> shape knobs (e.g. `HIDDEN_DIM`/`VOCAB_SIZE`/`N_AUDIO_CTX`) and drop
> `MAX_NUM_VARS` accordingly — see *Env knobs*. The command **forms** here are
> verified at smoke scale (1 layer / tiny dims); the full configs are
> memory-bound, not shape-bound.

## Model coverage

Every model supports streaming; decoders/encoders/Whisper also support
one-shot (seq>1), vision supports batch (`BATCH_SIZE`) composed with
streaming (`N_PROOFS`). Transformer/encoder bins pass `bench_config.yaml`
as `args[1]`; full-scale vision uses `cv_config.yaml`. All print `Verified: true`.

| Model | Class | Monolithic | Streaming | One-shot (standalone) | One-shot streaming |
|---|---|---|---|---|---|
| GPT-2 | decoder | `gpt2` | `bench_streaming_gpt2` | `oneshot_gpt2` | `bench_streaming_oneshot_gpt2` |
| Llama-2 | decoder | `llama2` | `bench_streaming_llama2` | `oneshot_llama` | `bench_streaming_oneshot_llama2` |
| Llama-3 | decoder | `llama3` | — | `oneshot_llama3` | `bench_streaming_oneshot_llama3` |
| GPT-J | decoder | `gptj` | — | `oneshot_gptj` | `bench_streaming_oneshot_gptj` |
| BERT | encoder | `bert` | `bench_streaming_bert` | (seq>1 native) | — |
| Whisper | enc+dec | `whisper` | `bench_streaming_whisper` | `oneshot_whisper` | — |
| ResNet-50 | vision | `resnet` | `bench_streaming_resnet` | batch via `BATCH_SIZE` | `bench_streaming_resnet` (`BATCH_SIZE`+N) |
| VGG-16 | vision | `vgg` | `bench_streaming_vgg` | batch via `BATCH_SIZE` | `bench_streaming_vgg` (`BATCH_SIZE`+N) |
| 3D U-Net | vision | `unet3d` | `bench_streaming_unet3d` | batch via `BATCH_SIZE` | `bench_streaming_unet3d` (`BATCH_SIZE`+N) |
| YOLOv11n | vision | `yolo` | `bench_streaming_yolo` | batch via `BATCH_SIZE` | `bench_streaming_yolo` (`BATCH_SIZE`+N) |
| PointPillars | vision | `pointpillar` | `bench_streaming_pointpillar` | batch via `BATCH_SIZE` | `bench_streaming_pointpillar` (`BATCH_SIZE`+N) |

All five vision bins now take `BATCH_SIZE` (B images share one weight commit
within a proof) **and** `N_PROOFS` (weights deferred + amortized across the
stream) — batch and streaming compose in one bin. PointPillar's `BATCH_SIZE`
batches `(pillars, coords)` pairs.

### One-shot AR (decoders + Whisper)

Standalone (one proof of a full T-token generation; `Role::Constant`
selectors). For modest vocab the `lm_head`/argmax are un-sharded; for **full
vocab** (GPT-2 50257, Llama-2 32000) set `VOCAB_SHARDS>1` — see *Full vocab*
below.

> **Shape defaults to the real model.** The `oneshot_*` and
> `bench_streaming_oneshot_*` decoder bins **default to the real per-model shape**
> below — so you only need `NUM_LAYERS` (+ vocab for the full head). The shape
> knobs shown in the commands are explicit for clarity but are now the defaults.
> Only **vocab** still defaults modest (`VOCAB=256`) so the un-sharded head fits
> out of the box — pass `VOCAB_SIZE` + `VOCAB_SHARDS` for the full vocab head.
> For a **fast smoke**, override with the small shape (`NUM_HEADS=8 HEAD_DIM=64
> FFN_DIM=2048`).
>
> | Model | default shape | full vocab | full depth |
> |---|---|---|---|
> | GPT-2 small | *(fixed: hidden 768)* | `VOCAB_SIZE=50257` | `NUM_LAYERS=12` |
> | Llama-2 7B | `NUM_HEADS=32 HEAD_DIM=128 FFN_DIM=11008` | `VOCAB_SIZE=32000` | `NUM_LAYERS=32` |
> | Llama-3 8B | `NUM_HEADS=32 NUM_KV_HEADS=8 HEAD_DIM=128 FFN_DIM=14336` | `VOCAB_SIZE=128256` | `NUM_LAYERS=32` |
> | GPT-J 6B | `NUM_HEADS=16 HEAD_DIM=256 FFN_DIM=16384` | `VOCAB_SIZE=50400` | `NUM_LAYERS=28` |
>
> Standalone `oneshot_{llama,llama3,gptj}` take `VOCAB` (not `VOCAB_SIZE`).

These are **small-dims demos** (the bins default to the real per-model shape, so
small dims are passed explicitly here for a quick run on `bench_config.yaml`).
For the real models use the per-model **(full)** blocks above with
`llama2_config.yaml` — real hidden 4096 overflows bench_config's range table.

```bash
# GPT-2 (shape fixed) — 12L, 256-token generation
NUM_LAYERS=12 SEQ_LEN=256 VOCAB_SIZE=256 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/oneshot_gpt2 bench_config.yaml

NUM_LAYERS=12 SEQ_LEN=256 VOCAB=256 NUM_HEADS=8 HEAD_DIM=64 FFN_DIM=2048 \
  ./target/release/oneshot_llama bench_config.yaml

NUM_LAYERS=12 SEQ_LEN=256 VOCAB=256 NUM_HEADS=8 NUM_KV_HEADS=2 HEAD_DIM=64 FFN_DIM=2048 \
  ./target/release/oneshot_llama3 bench_config.yaml          # GQA via NUM_KV_HEADS

NUM_LAYERS=12 SEQ_LEN=256 VOCAB=256 NUM_HEADS=8 HEAD_DIM=64 FFN_DIM=2048 \
  ./target/release/oneshot_gptj bench_config.yaml

# Whisper decoder — full N_TEXT_CTX-token text generation in one proof
NUM_ENC_LAYERS=1 NUM_DEC_LAYERS=1 N_AUDIO_CTX=32 N_TEXT_CTX=16 N_STATE=128 N_HEAD=2 VOCAB=256 \
  ./target/release/oneshot_whisper bench_config.yaml
```

### One-shot AR streaming (decoders)

Each streamed proof is a full T-token generation; weights are deferred +
amortized into one finalize opening, per-generation selectors are committed
per-proof.

Small-dims demos (real shape is the default — small dims here for speed; for
real models use the per-model **(full)** blocks above with `llama2_config.yaml`).

```bash
NUM_LAYERS=12 SEQ_LEN=256 N_PROOFS=10 NUM_PARTITIONS=8 VOCAB_SIZE=256 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_oneshot_gpt2 bench_config.yaml

NUM_LAYERS=12 SEQ_LEN=256 N_PROOFS=10 NUM_PARTITIONS=8 VOCAB_SIZE=256 NUM_HEADS=8 HEAD_DIM=64 FFN_DIM=2048 \
  ./target/release/bench_streaming_oneshot_llama2 bench_config.yaml

NUM_LAYERS=12 SEQ_LEN=256 N_PROOFS=10 NUM_PARTITIONS=8 VOCAB_SIZE=256 NUM_HEADS=8 NUM_KV_HEADS=2 HEAD_DIM=64 FFN_DIM=2048 \
  ./target/release/bench_streaming_oneshot_llama3 bench_config.yaml

NUM_LAYERS=12 SEQ_LEN=256 N_PROOFS=10 NUM_PARTITIONS=8 VOCAB_SIZE=256 NUM_HEADS=8 HEAD_DIM=64 FFN_DIM=2048 \
  ./target/release/bench_streaming_oneshot_gptj bench_config.yaml
```

### Full vocab (`VOCAB_SHARDS`)

The one-shot argmax range-checks `diffs[seq, vocab]`, which is **dense over
vocab** — its fold-tree leaf is one ~`2^(log(seq)+log(vocab)+table_commit)`
entry table that won't fit GPU memory at full vocab. `VOCAB_SHARDS=N` splits
the LM head + argmax into `N` vocab blocks (mathematically identical: the
one-hot's single 1 lands in one block; `selected = Σ_k selected_k`; a per-shard
`nonneg` check proves dominance over every block). Each leaf's eq-table is then
`2^arity` field-pairs with `arity = log2(seq_pad) + log2(ceil(vocab/N)_pad) +
table_commit_log`. Size shards so `arity ≈ 24–25` (256–512 MB leaves):

```bash
# GPT-2 full vocab 50257, seq256, 12L, 8 GPUs (1 partition/GPU on H200)
NUM_LAYERS=12 SEQ_LEN=256 N_PROOFS=10 NUM_PARTITIONS=8 VOCAB_SIZE=50257 VOCAB_SHARDS=32 \
  ZK4_STREAM_X_EXT2_CACHE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_oneshot_gpt2 bench_config.yaml

# Llama-2 full vocab 32000 — FULL 7B shape (shape is the default now). Real
# hidden 4096 needs llama2_config (table_size_log=12), not bench_config.
NUM_LAYERS=32 SEQ_LEN=256 N_PROOFS=10 NUM_PARTITIONS=8 \
  VOCAB_SIZE=32000 VOCAB_SHARDS=16 MAX_NUM_VARS=26 \
  ./target/release/bench_streaming_oneshot_llama2 llama2_config.yaml
```

Recommended `VOCAB_SHARDS` (target `arity ≤ 25`): `table_commit_log=8`,

| seq_len | GPT-2 (50257) | Llama-2 (32000) |
|--------:|--------------:|----------------:|
|  64     | 8             | 8               |
| 256     | 32            | 16              |
| 1024    | 128           | 64              |

Higher is always safe (smaller leaves, more shards = more commitment
overhead). `VOCAB_SHARDS=1` (default) keeps the original un-sharded path
byte-for-byte. All four decoders accept `VOCAB_SHARDS`, standalone and
streaming: `oneshot_{gpt2,llama,llama3,gptj}` and
`bench_streaming_oneshot_{gpt2,llama2,llama3,gptj}` (standalone llama bins use
`VOCAB`, the rest use `VOCAB_SIZE`). If a leaf still exceeds VRAM (heavy
contention / tiny GPU), the prover now **falls back to host** for the oversized
einsum and same-point evals rather than crashing (slower, correct,
byte-identical transcript).

### Encoder + Whisper streaming

```bash
# BERT (encoder, seq>1 native — no AR); weights amortized across N inferences
NUM_LAYERS=12 SEQ_LEN=128 N_PROOFS=10 NUM_PARTITIONS=8 \
  ./target/release/bench_streaming_bert bench_config.yaml

# Whisper (encoder + decoder + cross-attention). Tiny config shown (verified);
# scale up NUM_ENC/DEC_LAYERS, N_AUDIO_CTX, N_TEXT_CTX, N_STATE, N_HEAD toward
# real tiny.en (4/4, 1500/448, 384, 6) as memory allows — the full audio_ctx
# is heavy.
NUM_ENC_LAYERS=1 NUM_DEC_LAYERS=1 N_AUDIO_CTX=32 N_TEXT_CTX=16 N_STATE=128 N_HEAD=2 N_PROOFS=10 \
  ./target/release/bench_streaming_whisper bench_config.yaml
```

### Vision — full models at MLPerf scale (`cv_config.yaml`)

Use **`cv_config.yaml`** for deep CNNs at real input resolution: `scale_factor_log=16`
gives the fixed-point headroom for deep activations, and `table_commit_log=8`
keeps the range-check fold-tree off the cliff. `BATCH_SIZE` batches images that
share one weight commit; `N_PROOFS` streams + amortizes weights across proofs;
`NUM_PARTITIONS` splits the backward pass across GPUs. All print `Verified: true`.
(Commands assume an 8-GPU node; the indicative per-model times were measured on
4×80 GB — at these sizes multi-GPU adds ~1.1–1.2×, so 8 GPUs are similar-to-faster.)

```bash
# ResNet-50 — full 224x224, 53 conv, 1000-class. ~29 s/img prove.
NUM_LAYERS=53 INPUT_SIZE=224 N_PROOFS=4 BATCH_SIZE=1 MAX_NUM_VARS=28 NUM_PARTITIONS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_resnet cv_config.yaml

# VGG-16 — 32x32 (CIFAR), 13 conv. ~3.5 s/img (the FC head dominates).
NUM_LAYERS=13 INPUT_SIZE=32 N_PROOFS=4 BATCH_SIZE=1 MAX_NUM_VARS=24 NUM_PARTITIONS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_vgg cv_config.yaml

# YOLOv11n — full 640x640, 8 stages (backbone+neck+heads). ~128 s/img.
NUM_STAGES=8 INPUT_SIZE=640 N_PROOFS=4 BATCH_SIZE=1 MAX_NUM_VARS=30 NUM_PARTITIONS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_yolo cv_config.yaml

# 3D U-Net — 64^3 patch, full 6 levels. ~220 s/vol (forward ~145 s of it).
# (128^3 — the MLPerf KiTS19 patch — verifies but exceeds a practical budget;
#  forward witness-gen + prove are each ~30 min. See "Vision scaling" below.)
NUM_LAYERS=6 INPUT_D=64 INPUT_H=64 INPUT_W=64 N_PROOFS=4 BATCH_SIZE=1 MAX_NUM_VARS=28 NUM_PARTITIONS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_unet3d cv_config.yaml

# PointPillars — BEV 248x216, 6000 pillars x 32 pts. ~314 s/inf.
# (Full KITTI 496x432 / 12000 pillars verifies but exceeds a practical budget.)
# Grid + pillar counts come from env, NOT the YAML.
NY=248 NX=216 N_PILLARS=6000 MAX_POINTS=32 N_PROOFS=4 BATCH_SIZE=1 MAX_NUM_VARS=28 NUM_PARTITIONS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_pointpillar cv_config.yaml

# Batch + streaming together (e.g. 2 images/proof x 3 proofs = 6 amortized):
NUM_LAYERS=53 INPUT_SIZE=224 BATCH_SIZE=2 N_PROOFS=3 MAX_NUM_VARS=28 NUM_PARTITIONS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./target/release/bench_streaming_resnet cv_config.yaml
```

#### Vision scaling notes

- **`MAX_NUM_VARS`** must cover the largest committed poly: 24 @≤56 (e.g. VGG @32),
  26 @112, **28 @224**, 30 @640. Too low → commit/fold-tree failure.
- The two largest configs (U-Net **128³**, PointPillar **full KITTI**) verify but
  are gated by the **forward pass** (witness generation), not the prover:
  per-conv `ScaleDown` (the fixed-point rescale, building a sparse range aux) and
  `Conv3D` dominate, and a single-image chain has no width for multi-GPU to help.
  Set `ZKT_RUN_PROFILE=1` to see the per-op-type forward breakdown.
- **`ZKT_RUN_BACKEND=gpu`** runs the forward on the GPU pool (multi-GPU node
  distribution + device-memory bounded). It helps batch / wide nets where ops
  have GPU kernels; for sequential single-image CNNs the CPU forward is
  comparable (many ops have no GPU kernel yet).

### Small vision smoke runs (fast, `bench_config.yaml`)

```bash
NUM_LAYERS=2 INPUT_SIZE=64 N_PROOFS=10 ./target/release/bench_streaming_resnet bench_config.yaml
NUM_LAYERS=2 N_PROOFS=10 ./target/release/bench_streaming_vgg bench_config.yaml
NUM_LAYERS=1 INPUT_D=8 INPUT_H=16 INPUT_W=16 N_PROOFS=10 ./target/release/bench_streaming_unet3d bench_config.yaml
NUM_STAGES=2 INPUT_SIZE=64 N_PROOFS=10 ./target/release/bench_streaming_yolo bench_config.yaml
N_PROOFS=10 ./target/release/bench_streaming_pointpillar bench_config.yaml
```

## Smaller smoke runs (verify everything works fast)

```bash
# GPT-2 1L, single GPU — completes in seconds
NUM_LAYERS=1 SEQ_LEN=1 \
  ./target/release/gpt2 bench_config.yaml

# Llama-2 1L, N=5 streaming smoke run
NUM_LAYERS=1 SEQ_LEN=1 N_PROOFS=5 \
  ZK4_STREAM_X_EXT2_CACHE=1 \
  ./target/release/bench_streaming_llama2 llama2_config.yaml
```

## Env knobs (most useful)

| Var | Default | Effect |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | all | which GPUs the process sees |
| `ZK4_GPU_DEVICES` | all visible | explicit GPU pool (e.g. `0,2,3` to skip a contended GPU 1) |
| `NUM_LAYERS` | 1 | model depth (12 for full GPT-2, 32 for full Llama-2-7B) |
| `SEQ_LEN` | 1 | token count |
| `N_PROOFS` | 5 | streaming bench: how many inferences to stream before finalize |
| `BATCH_SIZE` | 1 | vision bins: images per proof sharing one weight commit (composes with `N_PROOFS`) |
| `NUM_PARTITIONS` | 1 | partition the DAG into N subcircuits, one per GPU. Each partition runs its commits + backward-pass sumcheck in parallel on its assigned GPU. With 8 GPUs, `NUM_PARTITIONS=8` typically gives the biggest prove-time win on deep models |
| `ZK4_STREAM_X_EXT2_CACHE` | unset | enable Methods 1+2 caches (skip per-call witness lift + d_acc alloc). Faster, costs GPU memory |
| `ZK4_STREAM_PARALLEL_REDUCER` | unset | enable Method 3 rayon-parallel reducer (single-GPU; opt-in only, ~6% slower on average) |
| `ZK4_STREAM_PARALLELISM` | 4 | reducer thread pool size if PARALLEL_REDUCER is on |
| `ZK4_KEEP_SP_POOL` | unset | disable the per-fold-tree SP_POOL clear (don't set this — causes the leak we fixed) |
| `ZK4_FOLD_GROUPS_PER_GPU` | 3 | fold-tree worker threads per GPU. Each concurrent worker retains ~10 GB of pooled GPU buffers at arity 22-24; lower to 1-2 if the fold tree OOMs on small cards |
| `ZK4_DEVICE_RESIDENT_FOLD` | 1 | device-resident fold-tree groups (witness data stays on GPU across same-point → multifold → split → chunk-evals). Set 0 to force the host path |
| `ZK4_DEVRES_MIN_ARITY` | 18 | minimum arity for the device-resident group path (smaller arities stay on the host path where CPU same-point wins) |
| `ZK4_DEFER_CONSTANTS` | unset | bin-level: enable defer mode in the regular `gpt2` / `llama2` bins (the streaming bench bins do this internally) |
| `ZK4_B` | 21 | bit-decomposition width |
| `ZK4_BASE` | 2 | commit radix (2 / 4 / 16 / 64). Higher = fewer leaves but more memory per leaf |
| `MAX_NUM_VARS` | 22 (GPT-2 / Llama-2 1L) | poly arity cap; raise to 23+ if you change shard counts |
| `FFN_SHARDS` (llama) | 16 | how many FFN matmul shards |
| `LOGITS_SHARDS` (llama) | 32 | how many logits-head matmul shards |
| `NUM_HEADS`, `HEAD_DIM`, `FFN_DIM`, `VOCAB`/`VOCAB_SIZE`, `HIDDEN_DIM` | per-model real shape | transformer shape (override for smaller smoke tests) |
| `NUM_KV_HEADS` (llama3) | 8 | grouped-query attention KV-head count |
| `NUM_ENC_LAYERS`, `NUM_DEC_LAYERS` (whisper) | 4, 4 | Whisper encoder / decoder depth |
| `N_AUDIO_CTX`, `N_TEXT_CTX`, `N_STATE`, `N_HEAD`, `N_MELS` (whisper) | tiny.en (1500/448/384/6/80) | Whisper audio/text context, width, heads, mel bins |
| `ZK4_TIMING` | unset | print fold-tree timing breakdown |
| `ZKT_RUN_BACKEND` | `cpu` | forward pass backend; `gpu` = multi-GPU node distribution over the device pool |
| `ZKT_RUN_PROFILE` | unset | print per-op-type forward (`dag.run`) timing breakdown |
| `INPUT_SIZE` / `INPUT_D,H,W` | model default | vision input resolution (2D HxW / 3D volume) |
| `NUM_STAGES` (yolo) | 2 | YOLOv11n stages to build (8 = full backbone+neck+heads) |
| `NY`,`NX`,`N_PILLARS`,`MAX_POINTS` (pointpillar) | small | BEV grid + pillar counts (env, not the YAML) |

## Expected speedups (sound throughout)

8-GPU pool, single-stream inference (your hardware will vary):

| Model | Monolithic | Streaming (8 GPU, cache) | Speedup |
|---|---:|---:|:---:|
| GPT-2 12L (N=10) | ~4 s / proof | ~2 s / proof | ~2× |
| Llama-2 32L (N=10) | ~140 s / proof | ~60-80 s / proof | 1.7-2.3× |

Larger N → more amortization (finalize cost / N shrinks). The
asymptote is `prove(defer) + acc-update` per proof.

## Pitfalls

1. **Llama-2 32L cache** holds ~350 GB across GPUs. 8 GPUs at 80 GB
   each = ample headroom, but if you have fewer or smaller cards,
   drop the cache (unset `ZK4_STREAM_X_EXT2_CACHE`).
2. **Don't include contended GPUs in the pool** — the per-device
   shard runs at the slowest device's speed. Use `ZK4_GPU_DEVICES`
   to explicitly skip a busy GPU.
3. **First streaming iter is cold** (cache populate + first prove);
   measure steady-state from iter 2-3+.
4. **`NUM_LAYERS` defaults to 1** in both bins — easy to forget when
   doing full-model runs.

## Verification

Every command above prints `Verified: true` on success. The
streaming bins also report the deferred-claim count and the number
of reducer steps:

```
Verified: true (142 Constant edges × 5 proofs = 710 deferred claims
                → 568 reducer steps → 1 fold-tree open over 142 edges)
```

A mid-stream `panic!` (`AllocationFailed`, `ext2_eq_dp_all_device
failed`, etc.) almost always means GPU memory pressure — fall back
to no cache or fewer GPUs in the pool.

## Where to look in the source

- `src/dag/streaming_accumulator.rs` — the streaming aggregator
  (deferred constants, reducer accumulation, finalize, multi-GPU
  sharding).
- `src/dag/fold_integration.rs` — the per-proof prover / verifier
  with `ZK4_DEFER_CONSTANTS` support.
- `src/basicblock/reducer.rs` — the K=2 reducer-block GPU/CPU paths
  and the cached-buffers variant.
- `src/fold/tree.rs` — the Ajtai fold-tree opening; contains the
  multi-GPU `gpu_device_pool()` and the SP_POOL clear that fixed
  the memory leak.
- `src/bin/bench_streaming_*.rs` — the streaming benches.

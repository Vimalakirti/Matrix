# zk-torch-3: GPU-Native ZKML with Goldilocks Field

A GPU-native zero-knowledge proof system for machine learning inference, built on the Goldilocks field (p = 2^64 - 2^32 + 1) with custom CUDA kernels.

## Overview

zk-torch-3 proves the correctness of neural network inference for transformer and CNN models using a DAG-based sumcheck protocol with Basefold polynomial commitments and Poseidon2 hashing.

| Aspect | zk-torch-2 | zk-torch-3 |
|--------|-----------|-----------|
| Field | BN254/BLS12-381 (256-bit) | Goldilocks (64-bit) |
| Backend | arkworks CPU / icicle GPU | Custom CUDA kernels |
| Commitment | KZH3 (pairing-based) | Basefold (hash-based, Poseidon2 Merkle) |
| Transcript | Merlin (Keccak) | Poseidon2 DuplexChallenger sponge |
| Extension field | N/A | GoldilocksExt2 (X^2 - 7) |
| Multi-GPU | No | Yes (partition-aware parallel proving) |

## Prerequisites

- NVIDIA GPU(s) with CUDA support (tested on A100-80GB)
- CUDA toolkit installed
- Rust toolchain (stable)

```bash
cargo build --release
```

## Supported Models

| Model | Type | Full Layers | Hidden Dim | Input Shape | Data Type |
|-------|------|-------------|------------|-------------|-----------|
| GPT-2 Small | Transformer | 12 | 768 | [1, 1, 768] | Float |
| BERT-Large | Transformer | 24 | 1024 | [1, 1, 1024] | Float |
| GPT-J 6B | Transformer | 28 | 4096 | [1, 1, 4096] | Float |
| LLaMA-2 7B | Transformer (MHA) | 32 | 4096 | [1, 1, 4096] | Float |
| Llama 3.1 8B | Transformer (GQA) | 32 | 4096 | [1, seq, 4096] | Float |
| VGG-16 | CNN | 13 conv | 3-512 | [3, 32, 32] | Uint |
| ResNet-50 | CNN | 53 conv | 64-2048 | [3, 224, 224] | Uint |
| 3D UNet | CNN (3D) | ~60 conv | 32-320 | [1, D, H, W] | Uint |
| YOLOv11n | Object Detection | 80 conv | 16-256 | [3, 640, 640] | Uint |
| Whisper | Speech (Enc-Dec) | 4+4 (tiny) | 384 | [80, 3000] + [1, 448, 384] | Float |
| PointPainting | Autonomous Driving | DeepLabV3+ + PointPillar | 256-2048 | varies | Uint |

## Running Models

### Quick Start (single GPU, reduced layers)

```bash
# Quick test — 1 layer, single GPU
CUDA_VISIBLE_DEVICES=0 cargo run --release --bin gpt2
```

### Full Models with All Accelerated Features (4 GPUs)

To run every model at full scale with maximum acceleration, use 4 GPUs and set `NUM_PARTITIONS=4` for multi-GPU parallel proving:

```bash
# GPT-2 Small — 12 transformer layers
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_LAYERS=12 NUM_PARTITIONS=4 cargo run --release --bin gpt2

# BERT-Large — 24 transformer layers
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_LAYERS=24 NUM_PARTITIONS=4 cargo run --release --bin bert

# GPT-J 6B — 28 transformer layers
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_LAYERS=28 NUM_PARTITIONS=4 cargo run --release --bin gptj

# LLaMA-2 7B — 32 transformer layers
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_LAYERS=32 NUM_PARTITIONS=4 cargo run --release --bin llama

# VGG-16 — 13 conv layers
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_LAYERS=13 NUM_PARTITIONS=4 cargo run --release --bin vgg

# ResNet-50 — 53 conv layers
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_LAYERS=53 NUM_PARTITIONS=4 cargo run --release --bin resnet

# 3D UNet — 6 encoder+decoder levels
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_LAYERS=6 NUM_PARTITIONS=4 INPUT_D=32 INPUT_H=32 INPUT_W=32 cargo run --release --bin unet3d

# YOLOv11n — 8 stages (full model)
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_STAGES=8 NUM_PARTITIONS=4 INPUT_SIZE=128 cargo run --release --bin yolo

# Whisper-tiny — 4 encoder + 4 decoder layers
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_ENC_LAYERS=4 NUM_DEC_LAYERS=4 NUM_PARTITIONS=4 cargo run --release --bin whisper

# Llama 3.1 8B — GQA (32 Q heads, 8 KV heads), multi-token
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_LAYERS=4 SEQ_LEN=1 NUM_PARTITIONS=4 ZK_GPU_FUSED_THRESHOLD=99 cargo run --release --bin llama3

# PointPainting — DeepLabV3+ semantic segmentation + PointPillar 3D detection
CUDA_VISIBLE_DEVICES=0,1,2,3 STAGE=both NUM_PARTITIONS=4 cargo run --release --bin pointpainting
```

### Single-GPU Full Models

If you have only 1 GPU, omit `NUM_PARTITIONS` (defaults to 1, no partitioning):

```bash
CUDA_VISIBLE_DEVICES=0 NUM_LAYERS=12 cargo run --release --bin gpt2
CUDA_VISIBLE_DEVICES=0 NUM_LAYERS=24 cargo run --release --bin bert
CUDA_VISIBLE_DEVICES=0 NUM_LAYERS=28 cargo run --release --bin gptj
CUDA_VISIBLE_DEVICES=0 NUM_LAYERS=32 cargo run --release --bin llama
CUDA_VISIBLE_DEVICES=0 NUM_LAYERS=13 cargo run --release --bin vgg
CUDA_VISIBLE_DEVICES=0 NUM_LAYERS=53 cargo run --release --bin resnet
CUDA_VISIBLE_DEVICES=0 NUM_LAYERS=4 INPUT_D=16 INPUT_H=16 INPUT_W=16 cargo run --release --bin unet3d
CUDA_VISIBLE_DEVICES=0 NUM_STAGES=8 INPUT_SIZE=128 cargo run --release --bin yolo
CUDA_VISIBLE_DEVICES=0 NUM_ENC_LAYERS=4 NUM_DEC_LAYERS=4 cargo run --release --bin whisper
CUDA_VISIBLE_DEVICES=0 NUM_LAYERS=1 SEQ_LEN=1 ZK_GPU_FUSED_THRESHOLD=99 cargo run --release --bin llama3
CUDA_VISIBLE_DEVICES=0 STAGE=deeplabv3 cargo run --release --bin pointpainting
```

### VGG Variants

The VGG binary supports two architecture variants and two weight styles:

```bash
# VGG-16 (default)
VGG_VARIANT=16 cargo run --release --bin vgg

# VGG-11
VGG_VARIANT=11 cargo run --release --bin vgg

# VerfCNN style (no bias, single FC layer)
VGG_STYLE=verfcnn cargo run --release --bin vgg
```

## Environment Variables

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_LAYERS` | 1 (transformers), 2 (VGG) | Number of layers to include (set to full count for complete model) |
| `NUM_STAGES` | 4 | YOLOv11 stage count (1-8, full model = 8) |
| `INPUT_SIZE` | 640 | YOLOv11 spatial input size (e.g., 128, 256, 640) |
| `INPUT_D/H/W` | 32/32/32 | 3D UNet spatial dimensions |
| `NUM_ENC_LAYERS` | 4 | Whisper encoder layers |
| `NUM_DEC_LAYERS` | 4 | Whisper decoder layers |
| `N_STATE` | 384 | Whisper hidden dimension |
| `N_HEAD` | 6 | Whisper attention heads |
| `N_AUDIO_CTX` | 1500 | Whisper audio context length |
| `N_TEXT_CTX` | 448 | Whisper text context length |
| `N_MELS` | 80 | Whisper mel spectrogram bins |
| `SEQ_LEN` | 1 | Llama 3.1 8B sequence length (multi-token inference) |
| `STAGE` | `both` | PointPainting stage: `deeplabv3`, `pointpillar`, or `both` |
| `NUM_PARTITIONS` | 1 | Number of partitions for multi-GPU parallel proving |
| `CUDA_VISIBLE_DEVICES` | All | GPU device selection (e.g., `0,1,2,3`) |
| `RAYON_NUM_THREADS` | All cores | Number of CPU threads for parallel operations |
| `VGG_VARIANT` | `16` | VGG architecture: `11` or `16` |
| `VGG_STYLE` | `paper` | VGG style: `paper` (with bias) or `verfcnn` (no bias) |

### Full Layer Counts

| Model | `NUM_LAYERS`/`NUM_STAGES` for full model |
|-------|-------------------------------------------|
| GPT-2 | 12 |
| BERT | 24 |
| GPT-J | 28 |
| LLaMA-2 | 32 |
| Llama 3.1 8B | 32 |
| VGG-16 | 13 |
| VGG-11 | 8 |
| ResNet-50 | 53 |
| 3D UNet | 6 (encoder+decoder levels) |
| YOLOv11n | 8 (stages) |
| Whisper-tiny | 4+4 (enc+dec) |

### GPU Acceleration Thresholds

These control when GPU kernels are used vs CPU fallback. The defaults are tuned for A100 GPUs. Lower values push more work to GPUs; higher values keep small problems on CPU to avoid kernel launch overhead.

| Variable | Default | Description |
|----------|---------|-------------|
| `ZK_GPU_SUMCHECK_THRESHOLD` | 14 | Use GPU sumcheck when total rounds > threshold |
| `ZK_GPU_OPEN_THRESHOLD` | 16 | Use GPU opening proofs when num_vars >= threshold |
| `ZK_GPU_PARTIAL_EVAL_THRESHOLD` | 16 | Use GPU partial eval in CPU sumcheck path when n > threshold |
| `ZK_GPU_FUSED_THRESHOLD` | 16 | Use fused GPU permute+partial_eval kernel when n > threshold |

For most workloads, the defaults work well. To push everything to GPU:

```bash
ZK_GPU_SUMCHECK_THRESHOLD=10 ZK_GPU_OPEN_THRESHOLD=10 ZK_GPU_PARTIAL_EVAL_THRESHOLD=10 ZK_GPU_FUSED_THRESHOLD=10 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_LAYERS=12 NUM_PARTITIONS=4 cargo run --release --bin gpt2
```

## Output Format

Each binary prints a pipeline of stages:

```
=== GPT-2 Small on Goldilocks (12 layers) ===
Compile: 2.39s
DAG: 1910 nodes, 2778 edges
Run: 654ms
Commit: 1.84s
Prove: 2.85s (+ commit 203ms = 3.06s)
Verify: 151ms

Verified: true
```

- **Run**: Forward pass (compute all intermediate tensors)
- **Commit**: Basefold commitment to non-weight polynomials (online prover cost)
- **Prove**: Sumcheck proofs + lookup proofs + opening proofs
- **Prove (+ commit = total)**: Total online prover time (commit + prove)
- **Verify**: Verifier checks all proofs
- **Verified: true**: Proof is valid

## Benchmark Results (4x A100-80GB, 4 partitions)

| Model | Layers | Nodes | Edges | Run | Total Prove | Verify |
|-------|--------|-------|-------|-----|-------------|--------|
| GPT-2 Small | 12 | 1910 | 2778 | 0.65s | 3.16s | 155ms |
| BERT-Large | 24 | 3737 | 5436 | 1.26s | 6.28s | 272ms |
| GPT-J 6B | 28 | 3513 | 5268 | 13.3s | 8.78s | 293ms |
| LLaMA-2 7B | 32 | 4509 | 6701 | 19.8s | 12.22s | 366ms |
| VGG-16 | 13 conv | 139 | 172 | 1.94s | 1.63s | 36ms |
| ResNet-50 | 53 conv | 368 | 424 | 23.6s | 9.79s | 85ms |
| 3D UNet | 6 levels | 194 | — | 47.5s | 2.4s | 35ms |
| YOLOv11n | 8 stages (128x128) | 2138 | 3039 | 2.1s | 11.6s | 285ms |
| YOLOv11n | 8 stages (640x640) | 2138 | 3039 | — | 75.4s | 233ms |
| Whisper-tiny | 4+4 (1500 audio) | 1699 | 2460 | 490s | 190.7s | 169ms |
| Llama 3.1 8B | 4L, seq=1 | — | — | — | 12.2s | 72ms |
| Llama 3.1 8B | 4L, seq=1, 4 GPU | — | — | — | 7.06s | 65ms |

Total Prove = non-weight commit time + prove time. All models verified: true. Opening proofs include full Basefold Merkle query verification (Poseidon2 auth paths + fold consistency checks).

YOLOv11n at 640x640 uses partition-aware GPU placement: commitments are downloaded to CPU after commit, then re-uploaded per-edge during opening proofs. This enables proving models with 300+ GB of committed data across 4x 80GB GPUs.

## Testing

```bash
cargo test --release
```

91 tests covering field arithmetic, polynomial operations, sumcheck (CPU and GPU), all BasicBlock types, DAG compilation, partitioning, and end-to-end prove/verify for transformers, CNNs, and object detection models.

## Architecture

```
zk-torch-3/src/
  lib.rs                 # Crate root, constants (SF_LOG, GOLDILOCKS_PRIME)
  transcript.rs          # Poseidon2 DuplexChallenger sponge (CPU)
  poly/
    dense.rs             # DenseMLPoly (evaluation-based multilinear)
    sparse.rs            # SparseMLPoly + SelectionPolynomial
  commit/
    basefold.rs          # Basefold PCS (GPU Merkle via Poseidon2)
    cpu_basefold.rs      # CPU opening proofs (parallel)
    gpu_basefold.rs      # GPU opening proofs (double-buffered)
    cpu_poseidon2.rs     # CPU Poseidon2 for Merkle verification
  sumcheck/
    gpu_prover.rs        # GPU sumcheck prover
    cpu_ext2_prover.rs   # CPU Ext2 sumcheck (small polys)
    verifier.rs          # Sumcheck verifier
  basicblock/
    einsum.rs            # Tensor contraction (fused GPU permute+peval)
    conv.rs              # Conv1D/2D/3D, DepthwiseConv2D, ConvTranspose1D/2D/3D, FlattenKernel (alpha-power trick)
    concat.rs            # Concat, ChannelSlice
    pad.rs               # ZeroPad, ZeroPadAsym, ZeroPad3D
    maxpool.rs           # MaxPool2D (2x2 and general)
    subsample.rs         # SubSample2D (strided downsampling)
    instancenorm.rs      # InstanceNorm3D (advice op)
    scale.rs             # ScaleDown, ScaleUp (dense bit poly range check)
    range.rs             # NonNegative (dense bit poly range check)
    relu.rs              # ReLU (advice op)
    ...                  # Add, Sub, Permute, Reducer, etc.
  dag/
    mod.rs               # DAG: run/commit/prove/verify/prove_parallel/verify_parallel
    builder.rs           # DagBuilder DSL
    partition.rs         # Multi-GPU partition-aware proving
    gpt2.rs              # GPT-2 Small model graph
    bert.rs              # BERT-Large model graph
    gptj.rs              # GPT-J 6B model graph
    llama.rs             # LLaMA-2 7B (MHA) + Llama 3.1 8B (GQA) model graphs
    vgg.rs               # VGG-11/16 model graph
    resnet.rs            # ResNet-50 model graph
    unet3d.rs            # 3D UNet model graph
    yolo.rs              # YOLOv11n model graph
    whisper.rs           # Whisper model graph (encoder-decoder with cross-attention)
    deeplabv3plus.rs     # DeepLabV3+ semantic segmentation
    pointpillar.rs       # PointPillar 3D object detection
  bin/
    gpt2.rs, bert.rs, gptj.rs, llama.rs, llama3.rs, vgg.rs, resnet.rs,
    unet3d.rs, yolo.rs, whisper.rs, pointpainting.rs
```

## Key Optimizations

1. **Poseidon2 Transcript**: Real Poseidon2 permutation (x^7 S-box, MDS matrix, 22 internal rounds) with DuplexChallenger sponge (buffered absorb/squeeze, rate=4)
2. **GPU Sumcheck**: Double-buffered fold with swap to avoid cross-warp race conditions
3. **Fused GPU Permute + Partial Eval**: Single CUDA kernel replaces separate CPU permute + GPU partial eval for Einsum
4. **GPU Opening Proofs**: Inner-product sumcheck on GPU with pre-allocated double buffers, multi-GPU parallel via per-thread CUDA streams, per-device table caching
5. **Dense Bit Polynomial Range Checks**: Replaces sparse SelectionPolynomial (32x-32768x smaller auxiliaries)
6. **Partition-Aware Parallel Proving**: DAG partitioned into sections proved in parallel across GPUs with forked transcripts
7. **LUT-Based Permutation**: O(1) per-element permutation via split lookup tables
8. **Dedup Opening Tasks**: Groups by (edge_id, point) to avoid redundant proofs
9. **CPU Ext2 Sumcheck**: Small polynomials (n <= 14) use CPU to avoid GPU launch overhead
10. **Parallel Forward Pass**: Topological levels processed with nodes in parallel via rayon
11. **Conv2D Alpha-Power Trick**: Factorizes convolution constraint via alpha^{i+j} = alpha^i * alpha^j
12. **Full Basefold Soundness**: Merkle auth paths + fold consistency + query proofs via CPU Poseidon2
13. **Partition-Aware GPU Placement**: Edges mapped to producer's partition GPU for commit; download-to-CPU + re-upload for opening proofs enables 300+ GB models across 4x 80GB GPUs
14. **Witness Memory Freeing**: After backward pass, non-essential intermediate edges freed before lookup/opening proofs. Reduces memory for large models (e.g., Whisper frees 1706/2460 edges, ~45 GB)
15. **Optimized prove_range Evaluation**: B(r_x, r_y) computed as O(32) dot product using pre-computed part_aux, avoiding O(2^n) eq table allocation

## Implementation Summary

zk-torch-3 is a complete reimplementation of zk-torch-2 over the Goldilocks field with GPU-native operations:

- **9 implementation phases** completed: Foundation, Commitment, Sumcheck, BasicBlocks, DAG, Lookups, LLaMA-specific blocks, Model graphs, and Optimization/Integration
- **11 production models**: GPT-2 Small (12L), BERT-Large (24L), GPT-J 6B (28L), LLaMA-2 7B (32L, MHA), Llama 3.1 8B (32L, GQA), VGG-16, ResNet-50, 3D UNet, YOLOv11n, Whisper, PointPainting (DeepLabV3+ + PointPillar)
- **35+ BasicBlock types**: Add, Sub, Einsum, ScaleDown, ScaleUp, NonNegative, ExpHelper, TwoPow, Permute, Reducer, ChangeShape, Conv1D (with stride), Conv2D (with stride/dilation), Conv3D, DepthwiseConv2D, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D, FlattenKernel, FlattenKernel3D, Concat, ChannelSlice, ZeroPad, ZeroPadAsym, ZeroPad3D, MaxPoolHelper, Replicate2x2, SubSample2D, GeneralMaxPoolHelper, InstanceNorm3D, ReLU, RMSReciprocal, DivConst, SoftmaxConst, SigmoidConst, PillarMaxPool, ScatterToBEV
- **Sound proofs**: Full Basefold Merkle query verification with Poseidon2 auth paths, fold consistency, and query proofs
- **Multi-GPU**: Partition-aware parallel proving across up to 4 GPUs with forked transcripts
- **91 unit tests** covering all components

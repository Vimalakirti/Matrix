# MLPerf Inference v6.0 Edge — Accuracy Measurement Plan (zk-torch-4)

**Goal.** Measure the *accuracy* of the MLPerf Inference v6.0 **edge** models when their
inference is run through **zk-torch-4's fixed-point arithmetization**, and quantify how much
(if any) accuracy is lost versus the fp32 reference — scored with MLPerf's official accuracy
scripts against MLPerf's 99% / 99.9% targets.

Reference suite: `/scratch/bjchen4_icgpu/goldilocks/research/inference` (MLCommons `inference`, v6.0).
Prover / arithmetization: `/scratch/bjchen4_icgpu/goldilocks/zk-torch-4`.
Precedent: `zk-torch-3/MLPERF_ACCURACY.md` + `zk-torch-3/src/bin/resnet_mlperf_acc.rs` +
`zk-torch-3/scripts/export_resnet50.py` (ResNet reached real Top-1; other 6 were TODO). This plan
ports and completes that effort on zk-4.

---

## 0. Scope decisions (locked)

1. **Exact MLPerf reference architectures only.** Numbers are reported only on the exact v6.0
   reference model. Where zk-4's current builder is a smaller proxy, the exact-model builder is a
   **prerequisite** (see §4 gaps): YOLOv11**l** (not n), Whisper **large-v3** (not tiny.en),
   **PointPainting** (DeepLabV3+ segmentor + PointPillars, not bare PointPillars).
2. **Forward-only accuracy + spot-check proofs.** Accuracy is computed from `dag.run()` — the
   fixed-point forward pass. The zk proof only *certifies* those same witness values; it does not
   change them, so it cannot change the accuracy number. We therefore score the full dataset with
   `dag.run()` (fast, no proving) and separately run `prove_with_fold_tree` + verify on a handful of
   samples per model to confirm the arithmetized forward is in-range and provable. See §6.

---

## 1. What "accuracy under the arithmetization" means

zk-torch-4 represents every value as a field element = `round(x · 2^SF_LOG)` (signed via
`field_from_i64`), where `SF_LOG = scale_factor_log` (`bench_config`=10, `cv_config`=16). The
forward pass `dag.run()` computes the model in this fixed-point domain. Accuracy loss relative to
the fp32 reference comes entirely from arithmetization effects:

- **Fixed-point quantization** of weights, inputs, and activations (`SF_LOG` fractional bits).
- **Rescale truncation** — every matmul/conv emits its product at scale `2·SF_LOG` and a `ScaleDown`
  gadget divides by `2^SF_LOG` with integer rounding (this is the `g.scale` that keeps activations
  bounded; see the CV rescale history in `MEMORY`/`project_zk4_model_coverage`).
- **Nonlinearity approximations** in fixed point: softmax, LayerNorm/RMSNorm `rsqrt`, GELU/SiLU,
  reciprocal for attention `1/sqrt(d)` (`SF_FLOAT/sqrt(head_dim)` rounded), RoPE cos/sin tables.
- **Range-table clamping** — the NonNegative/range lookups assume activations fit
  `[0, 2^table_size_log)` per chunk. If an activation exceeds the table the check *silently
  mis-selects* (a `[range] WARNING … outside the table` is printed at witness-gen time). Overflow ≠
  accuracy noise — it is a correctness/soundness failure and those samples are invalid, not
  "slightly wrong." The harness must treat any range warning as a hard error for that sample.

**Key consequence:** accuracy is a pure function of `dag.run()`. This is what makes the study
tractable — we can sweep whole datasets without paying proving cost, then prove a sample to show it
is a *valid* zk statement.

---

## 2. The seven edge models

| Model | Task | Dataset (edge) | Metric | MLPerf target (99% / 99.9%) | MLPerf scorer (in repo) | zk-4 builder | Fidelity gap |
|---|---|---|---|---|---|---|---|
| ResNet-50 v1.5 | Image classification | ImageNet 2012 val (50k) | Top-1 | 76.46 → **75.70** / 76.42 | `vision/classification_and_detection/tools/accuracy-imagenet.py` | `resnet` (full 53-conv, 224² verified) | **none** |
| BERT-Large | Extractive QA | SQuAD v1.1 (10.8k) | F1 | 90.874 → **89.97** / 90.79 | `language/bert/accuracy-squad.py` | `bert` (hidden 1024, 24L, seq≤512) | **none** |
| 3D-UNet (nnU-Net) | Medical segmentation | KiTS19 (43 eval cases) | mean DICE | 0.86170 → **0.8531** / 0.8608 | `vision/medical_imaging/3d-unet-kits19/accuracy_kits.py` | `unet3d` (InstanceNorm/Conv3D verified) | topology audit |
| Llama-3.1-8B | Summarization | CNN-DailyMail (~13.4k / sampled) | ROUGE-1/2/L | rouge1 38.78 → **38.39** (+r2/rL) | `language/llama3.1-8b/run_accuracy.sh` + `evaluate-accuracy.py` | `llama3` (real 8B default; one-shot AR) | **none** (AR gen loop) |
| YOLOv11**l** | Object detection | COCO-safe subset (~1.5k) | mAP | 0.5287 → **0.5234** | `vision/classification_and_detection/tools/accuracy-coco.py` (+ `yolo/yolo_ultra_map.py`) | `yolo` = **v11n** | **build v11l** |
| Whisper **large-v3** | ASR | LibriSpeech (dev/clean+other) | WER (→ 1−WER acc) | ~WER target | `speech2text/accuracy_eval.py` | `whisper` = **tiny.en** | **build large-v3** |
| PointPainting | 3D object detection | Waymo Open (kitti_format) | mAP | 0.5425 → **0.5371** | `automotive/3d-object-detection/accuracy_waymo.py` | `pointpillar` only | **add DeepLabV3+ segmentor + painting fusion** |

Targets: confirm each against `research/inference/mlperf.conf` / the v6.0 rules table before
reporting; the 99% figures above are computed from the reference fp32 accuracy in each README.

---

## 3. Harness architecture (Python bridge, ported from zk-3)

```
Python: export + preprocess          Rust: arithmetized inference        Python: score
──────────────────────────          ─────────────────────────────       ─────────────────
1. Load reference weights            4. Read weight tensors  ──────────►  8. Read predictions
   (torch/HF/onnx checkpoint)        5. Read preprocessed input           9. Run MLPerf accuracy
2. Fuse (BN-fold etc.), quantize     6. dag.run() fixed-point forward        script → compare to
   w → round(w·2^SF_LOG) → i64       7. Decode output witness:                target (§2)
3. MLPerf preprocessing of input        f_to_int(elem)/2^SF_LOG → pred
   → quantize → binary               (spot: prove_with_fold_tree + verify)
```

Reuse zk-3's simple tensor format verbatim (`resnet_mlperf_acc.rs::read_tensor`):
`[ndim:u32][shape: ndim×u32][data: n×i64]` + a `metadata.txt` (sf_log, shapes, layer configs). i64
values are already field-encoded (negatives via two's-complement into the field).

### 3.1 Shared infrastructure to build once (`src/bin/mlperf/` + `scripts/`)

- **`zk_torch_4::mlperf` module** (new): `read_tensor` / `write_tensor`, `quantize(f32,sf)->i64`,
  `dequantize`, `decode_argmax`, `decode_logits`, and a **range-overflow guard** that scans terminal
  and intermediate witnesses for any value ≥ `2^table_size_log` and fails the sample loudly (wraps
  the existing `[range] WARNING`). Put decode/quantize here so every model bin shares them.
- **Per-model accuracy bin** `src/bin/<model>_mlperf_acc.rs`: builds the *exact* DAG with real
  weights (`Role::Constant`), loops over the preprocessed dataset setting the input witness, runs
  `dag.run`, decodes, appends to a predictions file in the format the MLPerf scorer expects
  (usually loadgen's `mlperf_log_accuracy.json`, or the model's native prediction json).
- **`scripts/export_<model>.py`**: load reference checkpoint, do MLPerf-standard fusion +
  preprocessing, quantize, emit weight tensors + preprocessed inputs + labels/refs. Model the
  ResNet one on `zk-torch-3/scripts/export_resnet50.py`.
- **`scripts/score_<model>.sh`**: invoke the in-repo MLPerf accuracy script on our predictions and
  print `measured / target / pass?`.
- **Config sweep driver** `scripts/acc_sweep.py`: run the bin across `scale_factor_log ∈
  {10,12,14,16,20}` (and `table_size_log`/`MAX_NUM_VARS` set high enough to avoid overflow) and
  tabulate accuracy vs precision. This is the core research output (§5).

### 3.2 Layout correctness (the sharp edge)

Each builder expects a specific tensor layout — e.g. conv weight `[C_out,C_in,kH,kW]` little-endian
`kw|kh|c_in|c_out` with padded entries zeroed (`resnet.rs::gen_conv_weight`), FC `[in,out]`, BERT's
20-tuple weight order, RMSNorm unit weight = `2^SF_LOG` not 0. **The export script must reproduce
these exactly** — the memory notes reconstructing BERT weight order from memory caused an einsum
rank mismatch. Mitigation: for each model, first replace the random `gen_*` in the existing bin with
a *loader* and confirm `Verified: true` on one real sample (proves layout) **before** scaling to the
dataset.

---

## 4. Per-model workstreams

Ordered by readiness. Models with "none" fidelity gap (§2) are the fast path — they exercise the
whole harness end-to-end and de-risk the shared infra before the harder builds.

### Tier A — builder matches, harness only

**A1. ResNet-50 v1.5** *(reference port — do first)*
- Port `zk-torch-3/src/bin/resnet_mlperf_acc.rs` → zk-4 API (Ajtai commit, `prove_with_fold_tree`).
  Use `resnet50_with_biases` equivalent; weights at `sf=SF_LOG`, BN-folded into conv (export script
  already fuses). Use `cv_config.yaml` (sf=16, table_size_log=18) and `MAX_NUM_VARS=28` for 224²
  (per coverage memory — the range aux is sparse, so 224² verifies).
- v1.5 detail: confirm the stride-in-3×3 bottleneck variant matches the zk-4 `resnet50` config.
- Score with `accuracy-imagenet.py`. First target: ≥75.70% Top-1 on the 50k val set.

**A2. BERT-Large / SQuAD v1.1**
- Load HF `bert-large-uncased-whole-word-masking-finetuned-squad`. Exact 20-tuple weight order
  (copy from `src/dag/bert.rs` gen, don't reconstruct). seq_len=384, output start/end logits.
- Decode span → SQuAD prediction json → `accuracy-squad.py` → F1. Target ≥89.97.
- Watch: softmax + LayerNorm fixed-point fidelity; if F1 < target, first suspect SF_LOG (§5).

**A3. Llama-3.1-8B / CNN-DailyMail**
- Real 8B is the `llama3` default shape. Summarization needs **autoregressive generation** →
  use the one-shot AR path (`oneshot_llama3` / the `_hidden` builder + `lm_head` + argmax), looping
  greedy/token-sampled decode to the MLPerf gen length. Full vocab 128256 → `VOCAB_SHARDS` (see
  README full-vocab section). Real weights need `llama2_config.yaml` (table_size_log=12, hidden 4096).
- Emit generated text → `evaluate-accuracy.py` → ROUGE-1/2/L. Target rouge1 ≥38.39.
- Cost: this is the heaviest forward (8B × gen_len × N_samples). Sample the eval set per MLPerf's
  allowed subset; forward-only keeps it affordable. Multi-GPU node.

**A4. 3D-UNet / KiTS19**
- **Audit first:** confirm the zk-4 `unet3d` topology equals the nnU-Net KiTS19 reference (channel
  widths, pooling, InstanceNorm placement, sliding-window inference). The coverage memory shows
  UNet3D verifies with InstanceNorm/Conv3D + `ChannelSlice` sf-propagation — but at *bench* magnitudes,
  not the real nnU-Net widths. Verify real widths fit the b=21 bound / MAX_NUM_VARS at cv_config.
- KiTS19 uses **sliding-window** inference over large volumes → the bin must tile, run each patch
  through `dag.run`, and reassemble before DICE. Non-trivial glue.
- Score with `accuracy_kits.py`. Target mean DICE ≥0.8531.

### Tier B — exact-model build required (prerequisite, then harness)

**B1. YOLOv11l** *(smallest gap)*
- zk-4 has YOLOv11**n**. Build **v11l**: same block topology (CBS/C3k2/SPPF/C2PSA detect head,
  SiLU), larger depth/width multipliers. Extend the `yolo` model file to take v11l channel/depth
  config; the fixed-point rescale pattern (`rescale_conv` in cbs helpers, per coverage memory)
  already exists — this is mostly a config/width scale-up + verifying b-bound at 640² input.
- Decode detections → COCO json → `accuracy-coco.py` (+ `yolo_ultra_map.py` for the ultralytics
  mAP path). Target mAP ≥0.5234 on the COCO-safe subset.

**B2. Whisper large-v3** *(new large builder)*
- zk-4 has tiny.en (enc4/dec4, n_state 384). large-v3 = **32 enc + 32 dec layers, n_state 1280,
  n_head 20, 128 mel bins, n_audio_ctx 1500, vocab 51866**. Scale up the existing whisper builder
  (encoder + causal decoder + cross-attention already implemented) to these dims; verify memory /
  b-bound (this is large — needs the 8-GPU node; may need `N_AUDIO_CTX`/layer sharding to fit
  forward, though forward-only is far cheaper than proving).
- Decoder is AR (one-shot AR path exists, `oneshot_whisper`). Mel preprocessing from LibriSpeech
  audio in the export script (reuse `speech2text` feature extraction).
- Decode transcript → `accuracy_eval.py` → WER. Report accuracy = 1−WER vs target.

**B3. PointPainting** *(largest gap — two networks + fusion)*
- MLPerf PointPainting = **DeepLabV3+ (ResNet-50 backbone, output-stride 16) semantic segmentation**
  that "paints" each LiDAR point with class scores, **+ PointPillars** (`pp_ep36.pth`) 3D detector
  on the painted cloud. zk-4 has only PointPillars.
- Work: (a) build a DeepLabV3+/ResNet-50-os16 DAG (ASPP dilated convs, decoder, bilinear upsample —
  new ops to check for fixed-point rescale support), (b) implement the painting fusion (project
  points into the seg map, concat class scores as extra point features), (c) feed painted pillars
  into the existing `pointpillar` builder. Coords stay sf=0 (per coverage memory).
- Decode boxes → Waymo/kitti format → `accuracy_waymo.py`. Target mAP ≥0.5371. **Highest effort;
  schedule last** and consider whether the DeepLabV3+ dilated-conv/ASPP path needs the same
  sf-propagation fixes the other CNNs needed.

---

## 5. Quantization-sensitivity study (the research contribution)

For each model, sweep `scale_factor_log` (fractional precision) and report the accuracy curve:

- Sweep `SF_LOG ∈ {10, 12, 14, 16, 20}`, with `table_size_log` / `MAX_NUM_VARS` raised enough that
  **no range overflow occurs** (verify via the §3.1 guard — an overflowed run is invalid, not a data
  point). Note: raising SF_LOG grows accumulations, so table/commit sizing must track it.
- Deliverable per model: table of `SF_LOG → metric`, the **minimum SF_LOG that meets the 99% (and
  99.9%) MLPerf target**, and the associated proving cost (poly sizes / prove time from a spot
  proof). This directly answers "what fixed-point precision does a sound zk proof of this model
  need to stay MLPerf-accurate, and what does that cost the prover?"
- Cross-cut: which arithmetization effect dominates the loss per model (report activation-error
  ablation: quantization-only vs +rescale-truncation vs +nonlinearity-approx), by comparing
  `dag.run` outputs to an fp32 reference forward on the same inputs (MSE / top-1 flip rate per layer).

---

## 6. Proof validation (spot checks)

For each model, after the accuracy sweep, run `prove_with_fold_tree` + `verify_with_fold_tree` on a
small sample set (e.g. 3–10 inputs) at the chosen SF_LOG and record `Verified: true`, prove time, and
proof size. Purpose: confirm the arithmetized forward we scored is a *valid, in-range* zk statement
(not just a numeric run), and produce the prover-cost half of the §5 accuracy/cost trade-off. This is
where forward-only meets end-to-end zkML — full-dataset proving is explicitly out of scope.

---

## 7. Sequencing & milestones

1. **M0 — shared infra** ✅ *scaffolded*: `zk_torch_4::mlperf` module (tensor IO, quantize/decode,
   range guard) + ResNet export/loader. See §9.
2. **M1 — ResNet-50 end-to-end** (A1): first real Top-1 on ImageNet; validates the whole pipeline
   and the zk-3→zk-4 port. Gate: ≥75.70% at some SF_LOG, `Verified: true` spot proof.
3. **M2 — BERT + Llama-3.1-8B** (A2, A3): text-model path (span decode + AR generation + full-vocab
   sharding). Gate: F1 ≥89.97; ROUGE-1 ≥38.39.
4. **M3 — 3D-UNet** (A4): sliding-window glue + topology audit. Gate: DICE ≥0.8531.
5. **M4 — YOLOv11l** (B1): build v11l config, detection decode. Gate: mAP ≥0.5234.
6. **M5 — Whisper large-v3** (B2): scale builder to large-v3, mel pipeline, WER.
7. **M6 — PointPainting** (B3): DeepLabV3+ segmentor + painting fusion + PointPillars. Gate: mAP
   ≥0.5371. (Highest risk/effort.)
8. **M7 — sensitivity report** (§5) across all seven + spot-proof cost table (§6).

Tier-A milestones (M1–M3) can proceed in parallel once M0 lands, since they share the module and
only differ in export script + decode + builder wiring.

---

## 8. Risks & open items

- **Range-table overflow at real magnitudes.** Real weights/activations are far larger than the `%2`
  bench values every current bin uses. Each model needs `table_size_log`/`MAX_NUM_VARS`/`ZK4_B`
  sized to the real dynamic range; deep ResNet was solved via per-op sf-propagation, but real nnU-Net
  widths, large-v3, and DeepLabV3+ ASPP are unverified at scale. The §3.1 guard makes overflow loud
  instead of silent-wrong.
- **b-bit commit bound (SuperNeo norm).** Committed accumulations must fit the `b`-bit two's-complement
  window (see `project_superneo_norm_bound`); raising SF_LOG or model width can push edges past it.
  This couples §5's SF_LOG sweep to the commitment parameters — cannot raise precision freely.
- **Layout fidelity.** Per §3.2, the #1 silent-bug source. Loader-vs-`gen_*` "prove one real sample"
  check before dataset scale-up mitigates it.
- **Llama-3.1-8B / Whisper cost.** Even forward-only, these are the heaviest; use MLPerf's allowed
  eval subsets and the 8-GPU node. Full-dataset proving is out of scope by decision §0.2.
- **Dataset access / licensing.** ImageNet val, SQuAD, KiTS19, COCO-safe, Waymo (kitti_format),
  LibriSpeech, CNN-DM — several require download via `mlcr` / registration. Provision datasets before
  each milestone; COCO-safe and Waymo are the fussiest (legal-safe subset scripts are in-repo).
- **Reference-accuracy provenance.** Report fp32 reference *and* MLPerf's quantized (int8) reference
  where published, so the number attributable specifically to zk-4's arithmetization is isolated
  (arithmetized − fp32), not conflated with generic quantization.
```

---

## 9. M0 status — what is built (2026-07-01)

**Shared module `src/mlperf.rs`** (`zk_torch_4::mlperf`, registered in `lib.rs`):
- `quantize(x,sf)` / `dequantize(f,sf)` — the fixed-point convention (`round(x·2^sf)` into the
  field, negatives via `int_to_f`). Unit-tested round-trip.
- `read_tensor` / `write_tensor` / `load_witness` — the Python-bridge format
  `[ndim:u32][shape:u32×ndim][data:u64 LE]`. Unit-tested round-trip.
- `read_metadata` → `Metadata {sf_log, num_conv, num_classes, conv_configs}`.
- `decode_logits` / `decode_argmax` / `topk` — output decoding.
- `range_health_check(&witnesses, table_size_log) → RangeReport` — host-side overflow scan
  (companion to the authoritative `[range] WARNING` in `basicblock/range.rs`).

**ResNet bin `src/bin/resnet_mlperf_acc.rs`** — loads real weights (Int, sf), builds
`resnet50_with_biases`, runs `dag.run()` per image, decodes Top-1, prints accuracy; optional
`ZK4_ACC_PROVE=k` runs prove+verify on the first k images (§6 spot check).

**Export `scripts/export_resnet50.py`** — torchvision `resnet50` (= v1.5), explicit BN-fold, conv
ordering matched to `resnet50_conv_configs()` (stem, then per block conv1/conv2/conv3/downsample),
Almost-Goldilocks prime `0xFFFFFFFEFFFFFFE1`, MLPerf ImageNet preprocessing.

Build: `cargo build --release --bin resnet_mlperf_acc` (✅ compiles; `cargo test --lib mlperf` ✅).

### Run recipe (M1)
```bash
# 1. deps: torchvision is NOT installed in this env (torch 2.5.1+cu124 IS) — install matching wheel:
pip install torchvision==0.20.1
# 2. export real weights + preprocessed val images (sf-log MUST match the config yaml below):
python scripts/export_resnet50.py --sf-log 16 \
    --imagenet-val /path/to/imagenet/val --val-labels /path/to/val.txt \
    --output-dir /tmp/resnet50_export --num-images 100
# 3. run the fixed-point forward + Top-1 (cv_config.yaml → scale_factor_log=16, table_size_log=18):
MAX_NUM_VARS=28 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    ./target/release/resnet_mlperf_acc cv_config.yaml \
    --weights-dir /tmp/resnet50_export --num-images 100
# spot proof on first 3 images: prepend ZK4_ACC_PROVE=3
```

### M1 open items surfaced by the scaffold (validate before trusting the number)
- **Signed input/activations.** Real ImageNet input is mean-subtracted (negative). The conv range
  path is NonNegative-oriented; whether signed activations pass at full 53-conv depth is the M1
  question. A `[range] WARNING` or a non-clean `RangeReport` on the first image means the
  signed/range work in §8 is needed — not a harness bug.
- **Bias broadcast shape.** Conv bias exported as `[c_out,1,1]`, FC bias `[c_out]` (per zk-3);
  confirm `g.add` broadcasts these against `[c_out,h,w]` / `[num_classes]` in zk-4.
- **SF_LOG match.** Bin warns if `metadata.sf_log != config scale_factor_log`; keep them equal.
- **Official scorer.** M0 self-computes Top-1 against `labels.txt` (identical metric to
  `accuracy-imagenet.py`); emitting loadgen `mlperf_log_accuracy.json` for the official script is a
  small M1 add if a submission-format artifact is wanted.

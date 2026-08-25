# MLPerf Inference v6.0 Edge — Accuracy Reproduction Plan

## Goal

Reproduce MLPerf accuracy experiments using zk-torch-3's fixed-point forward pass (`dag.run()`)
to verify that our quantization (SF_LOG=10, ~10-bit fractional precision) preserves model accuracy
within MLPerf's 99% threshold.

## Edge Models (7 total)

| Model | Task | Dataset | Accuracy Target (99%) | zk-torch-3 DAG | Status |
|-------|------|---------|----------------------|-----------------|--------|
| ResNet-50 v1.5 | Image Classification | ImageNet (50K) | 75.71% Top-1 | `resnet.rs` | **In Progress** |
| BERT-Large | QA (SQuAD v1.1) | SQuAD (10.8K) | 89.97% F1 | `bert.rs` | TODO |
| 3D-UNet KiTS19 | Medical Segmentation | KiTS19 (43) | 0.853 DICE | `unet3d.rs` | TODO |
| YOLOv11l | Object Detection | COCO Safe (1.5K) | 52.87% mAP | `yolo.rs` | TODO |
| PointPainting | 3D Detection | Waymo (40K) | 0.542 mAP | `pointpainting.rs` | TODO |
| Llama 3.1 8B | Summarization | CNN-DM (5K) | 38.67% ROUGE1 | `llama3.rs` | TODO |
| Whisper-Large-v3 | Speech-to-Text | LibriSpeech (1.6K) | 96.34% accuracy | `whisper.rs` | TODO |

## Architecture: Python Bridge + Rust Inference

```
Python (preprocessing)          Rust (inference)              Python (evaluation)
─────────────────────          ──────────────────            ───────────────────
1. Load PyTorch weights   →    4. Load weights from file  →  7. Load predictions
2. Quantize to fixed-point     5. Load input from file       8. Run MLPerf accuracy
3. Preprocess dataset input    6. dag.run() forward pass        script (e.g., Top-1)
   Write to binary files       Write output to file
```

## Gaps to Fill

### Gap 1: Weight Serialization (Python → Rust)
- Python: `weight_float32 * 2^SF_LOG → int → write as binary`
- Rust: Read binary file → `Vec<GoldilocksField>` → `Witness`
- Format: Simple binary (shape header + flat i64 values)

### Gap 2: Input Preprocessing
- Python: Standard MLPerf preprocessing (resize, normalize, etc.)
- Quantize normalized float input → fixed-point field elements
- Write preprocessed samples to binary files

### Gap 3: Output Decoding
- Rust: After `dag.run()`, extract output witness
- Convert: `f_to_int(field_elem) / 2^SF_LOG → float prediction`
- Write predictions to file for Python evaluation

### Gap 4: Quantization Sensitivity
- Default SF_LOG=10 gives ~0.001 precision
- If accuracy drops below 99% threshold, increase SF_LOG (12, 14, 16...)
- Trade-off: higher SF_LOG → larger polynomials → slower proving

## Per-Model Notes

### ResNet-50 (Starting Point)
- Model: torchvision `resnet50(pretrained=True)`
- Input: 224×224×3, normalized (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- Output: 1000-class logits → argmax → Top-1 prediction
- MLPerf script: `tools/accuracy-imagenet.py`
- Binary: `src/bin/resnet_mlperf_acc.rs`

### BERT-Large
- Model: HuggingFace `bert-large-uncased-whole-word-masking-finetuned-squad`
- Input: tokenized (input_ids, attention_mask, segment_ids), max_seq_length=384
- Output: start_logits, end_logits → span extraction → F1 score
- MLPerf script: `language/bert/accuracy-squad.py`

### 3D-UNet
- Model: nnUNet KiTS19
- Input: 3D medical volumes (variable size, requires sliding window)
- Output: segmentation masks → DICE score

### Others
- YOLOv11: COCO detection pipeline
- PointPainting: LiDAR + camera fusion
- Llama 3.1: autoregressive generation (need token-by-token)
- Whisper: encoder-decoder with spectrogram input

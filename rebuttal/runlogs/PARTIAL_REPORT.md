# F1(a) / F5 partial report (2026-11-12, after the 11:57 server shutdown)

## F5: full-depth census, per-channel real weights (done)
| Model | Layers | Rescales certified | Min rescale margin | Min accumulator margin | Added lookups |
|---|---|---|---|---|---|
| LLaMA2-7B | 32 | 964 / 964 | 13 bits | 3 bits | +8.69% |
| LLaMA3.1-8B | 32 | 964 / 964 | 15 bits | 3 bits | +7.71% |
| GPT-J-6B | 28 | 732 / 732 | 13 bits | 2 bits | +8.02% |
Every check in all 11 categories is certified; none is violated.
Raw: /scratch/bjchen4_icgpu/goldilocks/zk-torch-4-rebuttal/e2s/out_real_full/

## Table 3: same-protocol ablations on 2 GPUs (partial)
Fixed-coin hash: all four variants 6e1b775ca18bd88221a414e69a29bbe5.
| Model | Variant | Prove (s) | Link (s) | Auth/att. (s) | Peak (GB) | Paper prove / link (s) |
|---|---|---|---|---|---|---|
| GPT-2 | Default | 55.0 | 24.1 | 0.99 | 137 | 50.5 / 23.4 |
| GPT-2 | Kernels off | 150.5 | 123.8 | 1.46 | 137 | 146.7 / 119.1 |
| YOLOv11n | Default | 72.3 | 22.5 | 1.33 | 144 | 73.2 / 22.4 |
| YOLOv11n | Kernels off | 207.3 | 156.6 | 1.15 | 144 | 210.2 / 156.4 |
| Whisper | Default | 92.4 | 36.6 | 1.67 | 149 | 91.1 / 35.4 |
| Whisper | Kernels off | 314.7 | 256.5 | 1.51 | 151 | 303.8 / 246.1 |
- All 18 requests verified.
- GPT-2 Default needed 8, 5 and 5 authentication attempts.
- Dense link and Source-retained are queued. The interrupted Dense link run is in dense.interrupted.
Raw: /scratch/bjchen4_icgpu/goldilocks/zk-torch-4-perf/zk-torch-4/scripts/bench/morpheus/out_t3s/

## Table 4: non-LLM rows (from the Table 1 rerun)
| Model | Prove (s) | Commit | PIOP | Link | Auth |
|---|---|---|---|---|---|
| GPT-2 | 39.8 | 11% | 43% | 37% | 9% |
| BERT-Large | 71.7 | 11% | 51% | 33% | 5% |
| ResNet-50 | 8.1 | 14% | 31% | 16% | 39% |
| VGG-16 | 4.8 | 9% | 24% | 13% | 54% |
| YOLOv11n | 61.3 | 10% | 60% | 25% | 5% |
| 3D-UNet | 54.9 | 17% | 33% | 43% | 6% |
| PointPillars | 49.7 | 2% | 44% | 34% | 20% |
| Whisper | 70.6 | 15% | 49% | 30% | 5% |
Commit = commit+fill+bank; PIOP = piop+prep; Auth = auth+serialize.
Raw: /scratch/bjchen4_icgpu/goldilocks/zk-torch-4-perf/zk-torch-4/scripts/bench/morpheus/out_t1s/

## Table 5
The 4-GPU column is the Table 1 rerun (out_t1s). The 1-GPU column is not run yet.

## Not started
Table 2, the Table 5 one-GPU rows and the convolution ablation (VGG/ResNet).
Queue: /scratch/bjchen4_icgpu/goldilocks/rebuttal_assets/runlogs/resume_f1.sh
Log: /scratch/bjchen4_icgpu/goldilocks/rebuttal_assets/runlogs/resume_f1.out

## 11-17 16:40 status
- resume_f1b.sh: the hash check on the new driver passed for all 4 variants at 13:36 (6e1b775c...).
- The GPU driver then hung. The first timed run (Table 3 Default, GPT-2) had stalls of about 500 s on ordinary GPU calls and ran for 3 h.
- Its timings in out_t3s/base/ are INVALID. Discard them.
- morpheus_e2e is defunct and nvidia-smi is stuck in D state (os_acquire_rwlock_write). The driver needs an admin reset or a reboot.
- Still to do: Table 3 Default/Dense/Source-retained, Table 2, Table 5 g1, convpp. Relaunch with resume_f1b.sh after moving out_t3s/base aside.

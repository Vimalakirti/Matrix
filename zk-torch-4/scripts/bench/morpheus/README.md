# Morpheus GPU evaluation

Reproduces the evaluation tables of the paper (IEEE S&P 2027 submission):
the eleven-model coverage on four A100s and the one-A100 comparisons with
DeepProve and EZKL.

## Build

```bash
cd zk-torch-4
cargo build --release --bins          # morpheus_e2e and the bench_* binaries
```

## Run

```bash
# Baselines (each alone on the machine)
../deepprove/setup.sh $DP && ../deepprove/prepare_cnn.sh $DP
../deepprove/run_all.sh $DP $PWD/../../.. $OUT_DP && ../deepprove/run_cnn.sh $DP $OUT_DP
../ezkl/setup.sh $EZ && ../ezkl/run_gpu.sh $EZ

# Morpheus
DP_WORK=$DP ./run_paper.sh one      # one-GPU comparisons (GPU 2)
./run_paper.sh four                 # 4-GPU coverage, conv ablation, 1-GPU scaling
python3 gpu_tables.py out_paper     # per-row mean/min/max prove, verify, MiB, peak, attempts, stages
```

`run_models.sh <selector ...>` runs single rows. Selectors: model names
(`gpt2 bert llama2 llama3 gptj resnet vgg yolo unet3d pointpillars
whisper`), `dp1s32 dp1s128 dp1cnn` (DeepProve workloads), `ezkl ezkl8`
(EZKL workloads), `convpp` (full-convolution-advice ablation) and `g1`
(every model on one GPU). Useful environment: `OUT`, `N_MONO` (requests
per row), `MODES=mono`, `CUDA_VISIBLE_DEVICES`, `NO_GPU_WAIT=1` (skip the
wait for idle GPUs).

Every request is serialized, decoded and verified, and a row that panics,
is rejected or misses a verification is reported as FAILED. The script
refuses debug overrides (`NO_AUTH`, `NO_VERIFY`, fixed coins).

## Reduced configurations

Host memory (905 GB) and the exact conformance gadgets bound some shapes:

* LLaMA2-7B, LLaMA3-8B and GPT-J-6B run at full depth (32, 32 and 28
  layers), width and vocabulary, one monolithic request each
  (`LLM_MONO`, `LLM_MODES`). A request's witness takes about 36 minutes
  and the run peaks near 670 GB of host memory.
* YOLOv11n at 320x320. At 640x640 the exact SiLU range checks need about
  1,860 sources.
* PointPillars on a 64x64 grid with 1,024 pillars. Exact private grid
  routing is O(pillars x cells). Its graph masks need `SLOTS_LOG=19`
  (more than the default 16 slots per node).
* Whisper-tiny (4+4 layers) on a 500-frame window (`WHISPER_CTX`). The
  exact softmax checks grow with the square of the context; at 1,500
  frames even 2+2 layers exceed host memory.
* VGG-16 at 32x32 and 3D-UNet at 64^3, as in the earlier evaluation.
* 3D-UNet and the LLMs use `REQ_PLANES_LOG=5`.

On one GPU, the link keeps every source on the device (134 MB each), so
BERT-Large (590 sources), YOLOv11n, 3D-UNet and Whisper exceed 80 GB.
`G1_MODELS="..."` restricts the `g1` rows.

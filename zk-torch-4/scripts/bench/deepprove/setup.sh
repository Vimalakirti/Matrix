#!/usr/bin/env bash
# Clone, patch and build DeepProve (CPU and CUDA) for the head-to-head.
#
#   ./setup.sh <work_dir>
#
# Produces:
#   <work_dir>/deep-prove/target/release/bench-llm        CPU build (BN254 HyperKZG)
#   <work_dir>/deep-prove/target-cuda/release/bench-llm   --features cuda (HyperKZGGpu)
#   <work_dir>/deep-prove/target-cuda/release/bench       the same, ONNX CNNs (run_cnn.sh)
#   <work_dir>/cuda12_libs/                               put on LD_LIBRARY_PATH for the CUDA build
#   <work_dir>/run/model_cache/                           GPT-2 + Llama-2-7B from the local HF cache
#
# deepprove.patch changes these things, none touching the protocol
# (README.md lists all of them):
#   1. bench-llm times verification (upstream verifies but does not record it).
#   2. The GPT-2 SafeTensors loader gives the tied LM head its own commitment
#      key ("wte.weight_final_proj"), exactly as DeepProve's GGUF loader
#      already does. Without it the embedding table and the (differently
#      scaled) LM head share one commitment id and context generation aborts
#      with "Found different MLE for polynomial wte.weight" (the workspace
#      release profile keeps debug assertions on).
#   3. The Llama-2 loader drops the size-1 KV-group axis for multi-head attention.
#   4. The CNN bench reads the output back from the device before its accuracy
#      check (`tensor_variant`); upstream fails there on the CUDA build.
set -euo pipefail
WORK=$(realpath "${1:?usage: setup.sh <work_dir>}")
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=9d1a53e2   # "feat: implifying licensing just put Lagrange (#354)", v2.0.1 + 1

mkdir -p "$WORK"
cd "$WORK"
[ -d deep-prove ] || git clone https://github.com/Lagrange-Labs/deep-prove.git
cd deep-prove
git checkout -q "$COMMIT"
git apply --check "$HERE/deepprove.patch" 2>/dev/null && git apply "$HERE/deepprove.patch"

cargo build --release -p zkml --bin bench-llm

# CUDA build. The driver here supports CUDA <= 12.6 and the system nvcc is
# 11.5, whose bindings lack the CUtensorMap API cubecl-cuda needs. Pin cudarc
# to 12.4 bindings; ec-gpu's MSM kernels still build with the system nvcc.
CUDARC_CUDA_VERSION=12040 CARGO_TARGET_DIR=target-cuda \
    cargo build --release --features cuda -p zkml --bin bench-llm --bin bench

# cubecl JIT-compiles through NVRTC at run time. cudarc dlopens
# "libnvrtc.so" first, which resolves to the system CUDA 13.1 copy and emits
# PTX the 12.6 driver rejects (CUDA_ERROR_UNSUPPORTED_PTX_VERSION). Point it
# at the 12.x NVRTC that ships with the PyTorch cu12 wheels.
NV=$(python3 -c "import nvidia,os;print(list(nvidia.__path__)[0])")
mkdir -p "$WORK/cuda12_libs"
ln -sf "$NV/cuda_nvrtc/lib/libnvrtc.so.12" "$WORK/cuda12_libs/libnvrtc.so"
ln -sf "$NV/cuda_nvrtc/lib/libnvrtc.so.12" "$WORK/cuda12_libs/libnvrtc.so.12"
for f in "$NV"/cuda_nvrtc/lib/libnvrtc-builtins.so.12*; do ln -sf "$f" "$WORK/cuda12_libs/"; done
ln -sf "$NV/cuda_runtime/lib/libcudart.so.12" "$WORK/cuda12_libs/libcudart.so"

mkdir -p "$WORK/run"
python3 "$HERE/prepare_models.py" "$WORK/run"

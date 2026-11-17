#!/usr/bin/env bash
# DeepProve's CUDA build on its CIFAR-10 LeNet and COVID-19 CT CNN, one GPU.
#
#   ./run_cnn.sh <work_dir> <out_dir>     (after setup.sh and prepare_cnn.sh)
#
# Three samples each (--num-samples 3); DeepProve's bench CSV records prove,
# verify and proof size per sample. Env: GPU (default 2), TMO (s, 7200).
set -uo pipefail
WORK=$(realpath "${1:?work_dir}")
OUT=$(realpath -m "${2:?out_dir}")
GPU=${GPU:-2}
F=$WORK/cnn_fixtures
A=$WORK/deep-prove/zkml/assets/scripts
mkdir -p "$OUT"
ulimit -s unlimited
cd "$OUT"
for job in "cifar:$A/CNN/cnn-cifar-01.onnx:$F/cifar_input.json" "covid:$F/covid10.onnx:$F/covid_input.json"; do
  IFS=: read -r name onnx io <<<"$job"
  rm -f "${name}_bench.csv"
  ( time env LD_LIBRARY_PATH="$WORK/cuda12_libs:${LD_LIBRARY_PATH:-}" CUDA_VISIBLE_DEVICES=$GPU RUST_LOG=info \
      timeout "${TMO:-7200}" "$WORK/deep-prove/target-cuda/release/bench" --onnx "$onnx" --io "$io" \
      --num-samples 3 --bench "${name}_bench.csv" ) > "${name}_cuda.log" 2>&1
  echo "$name: exit $?"; cat "${name}_bench.csv" 2>/dev/null
done

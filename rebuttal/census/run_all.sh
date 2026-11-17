#!/usr/bin/env bash
# Run the E2 bounds pass on every Table 1 model (OUT=out by default).
cd "$(dirname "$0")"
export OUT=${OUT:-$PWD/out}
for m in gpt2 bert vgg resnet yolo pointpillars whisper unet3d; do ./run_bounds.sh $m; done
# 6B-8B LLMs: per-layer bounds are identical, so reduced-depth graphs
# (1, 2, 3 layers; full width, sequence and vocabulary) are extrapolated.
for m in llama2 llama3 gptj; do for l in 1 2 3; do ./run_bounds.sh $m NUM_LAYERS=$l; done; done
echo ALLDONE

#!/usr/bin/env bash
# Census on every Table 1 model (synthetic weights) plus the real-weight LLM graphs at 2 layers.
cd "$(dirname "$0")"; export BIN=${BIN:-/scratch/bjchen4_icgpu/goldilocks/zk-torch-4-rebuttal/target-sound2/release}
export OUT=${OUT:-$PWD/out} ZK4_RESCALE_REPORT=1
for m in gpt2 bert vgg resnet yolo pointpillars whisper unet3d; do ./run_bounds.sh $m; done
for m in llama2 llama3 gptj; do for l in 1 2 3; do ./run_bounds.sh $m NUM_LAYERS=$l; done; done
R=/scratch/bjchen4_icgpu/goldilocks/rebuttal_assets/real_weights
RC=(WEIGHTS=real ZK4_EXACT_RMS=1 REQ_PLANES_LOG=5 NUM_LAYERS=2)
./run_bounds.sh gpt2 "${RC[@]}" REAL_WEIGHTS=$R/gpt2_hf REAL_RESID_SCALE=2 ZK4_RMS_ELEM_LOG=36
./run_bounds.sh llama2 "${RC[@]}" REAL_WEIGHTS=$R/llama2 REAL_RESID_SCALE=8 ZK4_RMS_ELEM_LOG=40
./run_bounds.sh llama3 "${RC[@]}" REAL_WEIGHTS=$R/llama3 REAL_RESID_SCALE=32 ZK4_RMS_ELEM_LOG=38
./run_bounds.sh gptj "${RC[@]}" REAL_WEIGHTS=$R/gptj REAL_RESID_SCALE=5 ZK4_RMS_ELEM_LOG=40
echo ALLDONE

#!/usr/bin/env bash
# Final census (rescale bound 2^26): every Table 1 model (synthetic weights),
# 6B-8B LLMs at 1/2/3 layers (extrapolated), and the real-weight LLM graphs at 2 layers (out_real).
cd "$(dirname "$0")"; export BIN=/scratch/bjchen4_icgpu/goldilocks/zk-torch-4-rebuttal/target-sound/release ZK4_RESCALE_REPORT=1
export OUT=$PWD/out
for l in 1 2 3; do ./run_bounds.sh gptj NUM_LAYERS=$l; done
for m in gpt2 bert vgg resnet yolo pointpillars whisper unet3d; do ./run_bounds.sh $m; done
for m in llama2 llama3; do for l in 1 2 3; do ./run_bounds.sh $m NUM_LAYERS=$l; done; done
export OUT=$PWD/out_real
R=/scratch/bjchen4_icgpu/goldilocks/rebuttal_assets/real_weights
RC=(WEIGHTS=real ZK4_EXACT_RMS=1 REQ_PLANES_LOG=5 NUM_LAYERS=2)
./run_bounds.sh gpt2 "${RC[@]}" REAL_WEIGHTS=$R/gpt2_hf REAL_RESID_SCALE=2 ZK4_RMS_ELEM_LOG=36
./run_bounds.sh llama2 "${RC[@]}" REAL_WEIGHTS=$R/llama2 REAL_RESID_SCALE=8 ZK4_RMS_ELEM_LOG=40
./run_bounds.sh llama3 "${RC[@]}" REAL_WEIGHTS=$R/llama3 REAL_RESID_SCALE=32 ZK4_RMS_ELEM_LOG=38
./run_bounds.sh gptj "${RC[@]}" REAL_WEIGHTS=$R/gptj REAL_RESID_SCALE=5 ZK4_RMS_ELEM_LOG=40
echo ALLDONE

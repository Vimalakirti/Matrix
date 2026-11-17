#!/usr/bin/env bash
# Final census after the RMS |h| tightening: transformer graphs only (CV graphs have no RMS/LayerNorm).
cd "$(dirname "$0")"; export BIN=/scratch/bjchen4_icgpu/goldilocks/zk-torch-4-rebuttal/target-sound/release ZK4_RESCALE_REPORT=1
export OUT=$PWD/out
R=/scratch/bjchen4_icgpu/goldilocks/rebuttal_assets/real_weights
RC=(WEIGHTS=real ZK4_EXACT_RMS=1 REQ_PLANES_LOG=5 NUM_LAYERS=2)
OUT=$PWD/out_real ./run_bounds.sh gptj "${RC[@]}" REAL_WEIGHTS=$R/gptj REAL_RESID_SCALE=5 ZK4_RMS_ELEM_LOG=40
for m in gpt2 bert whisper; do ./run_bounds.sh $m; done
for m in gptj llama2 llama3; do for l in 1 2 3; do ./run_bounds.sh $m NUM_LAYERS=$l; done; done
export OUT=$PWD/out_real
./run_bounds.sh gpt2 "${RC[@]}" REAL_WEIGHTS=$R/gpt2_hf REAL_RESID_SCALE=2 ZK4_RMS_ELEM_LOG=36
./run_bounds.sh llama2 "${RC[@]}" REAL_WEIGHTS=$R/llama2 REAL_RESID_SCALE=8 ZK4_RMS_ELEM_LOG=40
./run_bounds.sh llama3 "${RC[@]}" REAL_WEIGHTS=$R/llama3 REAL_RESID_SCALE=32 ZK4_RMS_ELEM_LOG=38
echo ALLDONE

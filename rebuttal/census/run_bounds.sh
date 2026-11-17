#!/usr/bin/env bash
# E2: run the CPU bounds pass on the Table 1 configurations of
# scripts/bench/morpheus/run_models.sh (no GPU: CUDA_VISIBLE_DEVICES is empty
# and the pass exits inside Dag::run before the forward pass).
#   run_bounds.sh <model> [extra VAR=value ...]     e.g. run_bounds.sh llama2 NUM_LAYERS=2
set -u
E2=$(cd "$(dirname "$0")" && pwd)
cd /scratch/bjchen4_icgpu/goldilocks/zk-torch-4-rebuttal/zk-torch-4
OUT=${OUT:-$E2/out}; mkdir -p "$OUT"
BIN=${BIN:-/scratch/bjchen4_icgpu/goldilocks/zk-torch-4-rebuttal/target-sound/release}
COMMON=(CUDA_VISIBLE_DEVICES= PROFILE=ternary B0=11 SLOTS_LOG=10 ZK4_EXP_BITS=8 ARGMAX_LOGIT_BITS=20 MORPHEUS_AT_RUN=1 MORPHEUS_LEAN_AUX=1 RAYON_NUM_THREADS=8)
TW=(WEIGHTS=dense ZK4_EXACT_RMS=1)
m=$1; shift
case $m in
  gpt2)   C=(e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=gpt2 NUM_LAYERS=12 SEQ_LEN=128 VOCAB_SIZE=50257) ;;
  bert)   C=(e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=10 MODEL=bert NUM_LAYERS=24 SEQ_LEN=128) ;;
  llama2) C=(e2e "${TW[@]}" REQ_PLANES_LOG=5 ZK4_TABLE_SIZE_LOG=24 MODEL=llama2 NUM_LAYERS=32 SEQ_LEN=128 NUM_HEADS=32 HEAD_DIM=128 FFN_DIM=11008 FFN_SHARDS=1 VOCAB_SIZE=32000) ;;
  llama3) C=(e2e "${TW[@]}" REQ_PLANES_LOG=5 ZK4_TABLE_SIZE_LOG=24 MODEL=llama3 NUM_LAYERS=32 SEQ_LEN=128 NUM_HEADS=32 NUM_KV_HEADS=8 HEAD_DIM=128 FFN_DIM=14336 VOCAB_SIZE=128256) ;;
  gptj)   C=(e2e "${TW[@]}" REQ_PLANES_LOG=5 ZK4_TABLE_SIZE_LOG=24 MODEL=gptj NUM_LAYERS=28 SEQ_LEN=128 NUM_HEADS=16 HEAD_DIM=256 FFN_DIM=16384 VOCAB_SIZE=50400) ;;
  resnet) C=(hook:resnet:cv_config.yaml NUM_LAYERS=53 INPUT_SIZE=224 ZK4_TABLE_SIZE_LOG=12) ;;
  vgg)    C=(hook:vgg:cv_config.yaml VGG_VARIANT=16 NUM_LAYERS=13 INPUT_SIZE=32 ZK4_TABLE_SIZE_LOG=12) ;;
  yolo)   C=(hook:yolo:cv_config.yaml NUM_STAGES=8 INPUT_SIZE=320 ZK4_TABLE_SIZE_LOG=12) ;;
  unet3d) C=(hook:unet3d:cv_config.yaml NUM_LAYERS=6 INPUT_D=64 INPUT_H=64 INPUT_W=64 ZK4_TABLE_SIZE_LOG=12 REQ_PLANES_LOG=5) ;;
  pointpillars) C=(hook:pointpillar:cv_config.yaml NX=64 NY=64 N_PILLARS=1024 MAX_POINTS=32 NUM_ANCHORS=2 NUM_CLASSES=3 ZK4_TABLE_SIZE_LOG=12 MORPHEUS_GRID_ROUTING_MAX_CELLS=2147483648 SLOTS_LOG=19) ;;
  whisper) C=(hook:whisper:llama2_config.yaml NUM_ENC_LAYERS=4 NUM_DEC_LAYERS=4 N_MELS=80 N_AUDIO_CTX=500 N_TEXT_CTX=448 N_STATE=384 N_HEAD=6 ZK4_TABLE_SIZE_LOG=24) ;;
  *) echo "unknown model $m" >&2; exit 2 ;;
esac
tag=$m; for kv in "$@"; do case $kv in REAL_WEIGHTS=*) kv=real;; esac; tag="${tag}_${kv//=/}"; done
kind=${C[0]}; envs=("${C[@]:1}")
log="$OUT/$tag.log"
echo "[$(date +%T)] $tag" | tee "$log"
case $kind in
  e2e) env "${COMMON[@]}" "${envs[@]}" "$@" MORPHEUS_NAME=$m MORPHEUS_BOUNDS="$OUT/$tag" /usr/bin/time -v nice -n 19 $BIN/morpheus_bounds >> "$log" 2>&1 ;;
  hook:*) IFS=: read -r _ bin cfg <<< "$kind"
     env "${COMMON[@]}" "${envs[@]}" "$@" MORPHEUS_NAME=$m MORPHEUS_BOUNDS="$OUT/$tag" /usr/bin/time -v nice -n 19 $BIN/bounds_streaming_$bin "$cfg" >> "$log" 2>&1 ;;
esac
echo "exit $?" >> "$log"
grep -E "Maximum resident|exit |checks|^model" "$log"

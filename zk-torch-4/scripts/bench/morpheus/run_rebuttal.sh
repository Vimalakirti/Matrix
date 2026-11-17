#!/usr/bin/env bash
# Rebuttal experiments (GPU), one job at a time:
#
#   ./run_rebuttal.sh [e3] [e7] [e11]      (default: e3)
#
# E3   Timing variability with the private input, Table 1 configuration
#      (four GPUs). One process per input, so every request is a first
#      request:
#        GPT-2: the Table 1 weights, N_PROMPTS prompts (MORPHEUS_PROMPT_SEED).
#        ResNet-50: trained weights, N_IMAGES ImageNet images (RESNET_REAL,
#        RESNET_IMAGE; run_models.sh row resnet_real).
# E7   Cost of zero knowledge, Table 3 setup (two GPUs, GPT-2, YOLOv11n,
#      Whisper, N_ABL fresh-coin requests): default against masks off.
# E11  Zero-chunk skipping with the link kernels on, Table 3 setup: default
#      against skipping off.
#
# Summaries: python3 rebuttal_summary.py $OUT
set -u
OUT=$(realpath -m "${OUT:-$(dirname "$0")/out_rebuttal}")
cd "$(dirname "$0")"
mkdir -p "$OUT"
R=./run_models.sh
what=${*:-e3}
want() { for w in $what; do [ "$w" = "$1" ] && return 0; done; return 1; }

if want e3; then
  for k in $(seq 1 ${N_PROMPTS:-15}); do
    MORPHEUS_PROMPT_SEED=$k OUT=$OUT/e3/gpt2_p$k MODES=mono N_MONO=1 $R gpt2
  done
  : "${RESNET_REAL:?set RESNET_REAL to an export of scripts/export_resnet50.py}"
  export RESNET_REAL
  for k in $(seq 0 $(( ${N_IMAGES:-15} - 1 ))); do
    RESNET_IMAGE=$k OUT=$OUT/e3/resnet_i$k N_MONO=1 $R resnet_real
  done
fi

# E7 and E11: two-GPU ablations as in Table 3, sharing one default run.
#   noskip  MORPHEUS_NO_SKIP=1: all-zero link chunks are visited too, every
#           link kernel stays on (proof bytes unchanged).
#   nozk    MORPHEUS_NO_ZK=1: every hiding sample is zero and every attempt
#           is accepted; same proof shape, still verified.
# A fixed-coin two-layer GPT-2 request per variant comes first (dev only):
# base and noskip must give the same proof hash; nozk must verify.
if want e7 || want e11; then
  vs="base"; want e11 && vs="$vs noskip"; want e7 && vs="$vs nozk"
  env_of() { case $1 in base) echo "" ;; noskip) echo "MORPHEUS_NO_SKIP=1" ;; nozk) echo "MORPHEUS_NO_ZK=1" ;; esac; }
  SEED=$(printf '5a%.0s' $(seq 64))
  mkdir -p "$OUT/abl"
  for v in $vs; do
    log="$OUT/abl/hash_$v.log"
    ( cd ../../.. && env CUDA_VISIBLE_DEVICES=${ABL_DEVS:-2,0} MORPHEUS_GPUS=2 MORPHEUS_AUTH_GPUS=2 PROFILE=ternary B0=11 SLOTS_LOG=10 ZK4_EXP_BITS=8 MORPHEUS_TIMING=1 ARGMAX_LOGIT_BITS=20 \
      WEIGHTS=dense ZK4_EXACT_RMS=1 ZK4_TABLE_SIZE_LOG=24 MODEL=gpt2 NUM_LAYERS=2 SEQ_LEN=128 VOCAB_SIZE=50257 \
      DEFER=0 N_REQUESTS=1 MORPHEUS_COIN_SEED=$SEED MORPHEUS_PROOF_HASH=1 $(env_of $v) \
      timeout 1800 ./target/release/morpheus_e2e > "$log" 2>&1 )
    echo "[$(date +%T)] hash $v: $(grep -m1 -o 'proof hash [0-9a-f]*' "$log") $(grep -m1 -o 'verify (true)' "$log")"
  done
  for v in $vs; do
    env $(env_of $v) CUDA_VISIBLE_DEVICES=${ABL_DEVS:-2,0} MORPHEUS_GPUS=2 MORPHEUS_AUTH_GPUS=2 \
      OUT=$OUT/abl/$v MODES=mono N_MONO=${N_ABL:-3} $R gpt2 yolo whisper
  done
fi
echo "rebuttal done: $OUT"

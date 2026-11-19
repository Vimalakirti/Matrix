#!/bin/bash
# CPU-only check of a Morpheus transformer with real (pretrained) weights:
# fixed-point generation (morpheus_real_check: tokens, range violations,
# request-tensor widths) + comparison with the float HF model
# (validate_transformer.py, exact and graph-equivalent references).
# No GPU, no proving. Runs niced with 16 rayon / 8 torch threads.
#
#   scripts/real_weights_check.sh MODEL LAYERS|full PROMPT TAG [ENV=VAL ...]
#   env: SEQ (default 128), DRAFT=1 (float draft model for the Jacobi passes),
#        RW (REAL_WEIGHTS subdir, default MODEL), OUT (log dir, default .)
#
# Per-model real-weight knobs (validated; see the report):
#   gpt2   RW=gpt2_hf REAL_RESID_SCALE=2  ZK4_RMS_ELEM_LOG=36 REQ_PLANES_LOG=5
#   llama2         REAL_RESID_SCALE=8  ZK4_RMS_ELEM_LOG=40 REQ_PLANES_LOG=5
#   llama3         REAL_RESID_SCALE=32 ZK4_RMS_ELEM_LOG=38 REQ_PLANES_LOG=5
#   gptj           REAL_RESID_SCALE=5  ZK4_RMS_ELEM_LOG=40 REQ_PLANES_LOG=5
# plus MORPHEUS_TOKEN_CACHE=<dir> to store / reuse the generated suffix.
set -u
M=$1; L=$2; P=$3; TAG=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
R=${REAL_ROOT:-/path/to/workspace/rebuttal_assets/real_weights}
BIN=${BIN:-$HERE/../../target/release/morpheus_real_check}
cd "${OUT:-.}"
case $M in
  gpt2) C="MODEL=gpt2 SEQ_LEN=${SEQ:-128} VOCAB_SIZE=50257" ;;
  llama2) C="REQ_PLANES_LOG=5 MODEL=llama2 SEQ_LEN=${SEQ:-128} NUM_HEADS=32 HEAD_DIM=128 FFN_DIM=11008 FFN_SHARDS=1 VOCAB_SIZE=32000" ;;
  llama3) C="REQ_PLANES_LOG=5 MODEL=llama3 SEQ_LEN=${SEQ:-128} NUM_HEADS=32 NUM_KV_HEADS=8 HEAD_DIM=128 FFN_DIM=14336 VOCAB_SIZE=128256" ;;
  gptj) C="REQ_PLANES_LOG=5 MODEL=gptj SEQ_LEN=${SEQ:-128} NUM_HEADS=16 HEAD_DIM=256 FFN_DIM=16384 VOCAB_SIZE=50400" ;;
  *) echo "unknown model $M" >&2; exit 2 ;;
esac
SNAP=$(grep '^snapshot=' "$R/${RW:-$M}/meta.txt" | cut -d= -f2)
DL=""; [ "$L" = full ] || DL="--layers $L"
DRAFT_ENV=(DRAFT_OFF=1)
[ -n "${DRAFT:-}" ] && DRAFT_ENV=("MORPHEUS_DRAFT_CMD=OMP_NUM_THREADS=8 nice -n 19 python $HERE/draft_server.py --snapshot $SNAP $DL")
env "${DRAFT_ENV[@]}" CUDA_VISIBLE_DEVICES= RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-16} MORPHEUS_GEN_TRACE=1 WEIGHTS=real REAL_WEIGHTS="$R/${RW:-$M}" \
  ZK4_EXACT_RMS=1 ZK4_TABLE_SIZE_LOG=24 ZK4_EXP_BITS=8 ARGMAX_LOGIT_BITS=20 $C NUM_LAYERS=${L/full/} MORPHEUS_PROMPT_SEED=$P \
  CHECK_DUMP=${M}_${TAG}.bin "$@" /usr/bin/time -v nice -n 19 "$BIN" > ${M}_${TAG}.log 2>&1
SA=""; [ -n "$SNAP" ] && SA="--snapshot $SNAP"
for kv in "$@"; do case $kv in REAL_RESID_SCALE=*) [ $M = gpt2 ] && SA="$SA --logit-scale ${kv#*=}";; esac; done
LA=""; [ "$L" = full ] || LA="--layers $L"
OMP_NUM_THREADS=8 nice -n 19 python "$HERE/validate_transformer.py" --model $M --dump ${M}_${TAG}.bin $LA $SA > ${M}_${TAG}.val 2>&1
OMP_NUM_THREADS=8 nice -n 19 python "$HERE/validate_transformer.py" --model $M --dump ${M}_${TAG}.bin $LA $SA --graph-equivalent > ${M}_${TAG}.val0 2>&1

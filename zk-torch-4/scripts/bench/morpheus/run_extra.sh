#!/usr/bin/env bash
# Morpheus (S&P'27) scaling experiments of the paper's evaluation:
#   * multi-GPU strong scaling (deferred, N=2; 4 GPUs come from run_models.sh)
#   * sequence-length scaling of GPT-2 one-shot generation (deferred, N=2)
# Logs: $OUT/<name>_<tag>_defer.log (default scripts/bench/morpheus/out_extra).
set -u
[ "${DEFER_BATCHES+x}" ] && { echo "Unset DEFER_BATCHES for scaling runs" >&2; exit 2; }
# Fail closed: debug and deterministic/private-coin overrides are not paper runs.
for flag in NO_AUTH NO_VERIFY LAYOUT_ONLY LINK_ROWS MORPHEUS_UNCLAIMED MORPHEUS_COIN_SEED; do
  if [ "${!flag+x}" ]; then echo "Refusing benchmark override: $flag" >&2; exit 2; fi
done
# Validate before creating output directories or starting any watcher.
[[ "${N_DEFER:-2}" =~ ^[1-9][0-9]*$ ]] && [ "${N_DEFER:-2}" -le 10000 ] || exit 2
cd "$(dirname "$0")/../../.."
OUT=${OUT:-scripts/bench/morpheus/out_extra}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,0,1}
export PROFILE=ternary B0=11 SLOTS_LOG=17 ZK4_EXP_BITS=8 MORPHEUS_TIMING=1 DEFER=1 N_REQUESTS=${N_DEFER:-2}
export ARGMAX_LOGIT_BITS=${ARGMAX_LOGIT_BITS:-20}
TW=(WEIGHTS=${WEIGHTS:-dense} ZK4_EXACT_RMS=1 ZK4_TABLE_SIZE_LOG=24)

# The server is shared: wait until no process holds GPU memory, so a foreign
# job cannot push a run out of memory or perturb its timing.
wait_gpus() {
  [ -n "${NO_GPU_WAIT:-}" ] && return
  local said=
  # GPU_LOCK (next to this script) marks a timed foreign run, e.g. a baseline.
  while [ -e "$(dirname "$0")/GPU_LOCK" ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)" -gt 0 ]; do
    [ -z "$said" ] && echo "[$(date +%T)] waiting for free GPUs" && said=1
    sleep 60
  done
}
one() {   # tag kind env...
  local tag=$1 kind=$2; shift 2
  local log="$OUT/${tag}_defer.log" try=0
  while true; do
  wait_gpus
  echo "[$(date +%T)] $tag -> $log"
  rm -f "$log.foreign"
  # GPU processes of other users during the run go to $log.foreign; a GPU
  # memory failure with such a process present is retried (RETRIES, 2).
  ( while true; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do u=$(ps -o user= -p "$p" 2>/dev/null); [ -n "$u" ] && [ "$u" != "$USER" ] && echo "$(date +%T) $u $p" >> "$log.foreign"; done; sleep 1; done ) &
  local watch=$!
  printf "MORPHEUS_EXPECT protocol=3 profile=morpheus-ternary-v1 requests=%s batches=%s\n" "$N_REQUESTS" "$N_REQUESTS" > "$log"
  case $kind in
    e2e) env "$@" timeout ${TMO:-14400} ./target/release/morpheus_e2e >> "$log" 2>&1 ;;
    hook:*) IFS=: read -r _ bin cfg units <<< "$kind"
      env "$@" MORPHEUS_AT_RUN=1 MORPHEUS_NAME="$tag" MORPHEUS_UNITS="$units" timeout ${TMO:-14400} ./target/release/$bin "$cfg" >> "$log" 2>&1 ;;
  esac
  local run_status=$?
  printf "MORPHEUS_EXIT status=%s\n" "$run_status" >> "$log"
  kill $watch 2>/dev/null; wait $watch 2>/dev/null
  grep -E "^(prove|amortized, N->inf|verify per)|panicked" "$log" | sed 's/^/    /'
  if [ -s "$log.foreign" ] && grep -qiE "AllocationFailed|out of memory" "$log" && [ $try -lt ${RETRIES:-2} ]; then
    try=$((try + 1))
    echo "[$(date +%T)] $tag: GPU memory failure while another user's job was on the GPUs ($(head -1 "$log.foreign")); retry $try"
    mv "$log" "$log.contended$try"
    continue
  fi
  break
  done
}

for g in ${GPUS:-1 2}; do
  G=(MORPHEUS_GPUS=$g MORPHEUS_AUTH_GPUS=$g)
  one "GPT-2_g$g" e2e "${G[@]}" "${TW[@]}" MODEL=gpt2 NUM_LAYERS=12 SEQ_LEN=128 VOCAB_SIZE=50257
  one "ResNet-50_g$g" hook:bench_streaming_resnet:cv_config.yaml:1 "${G[@]}" NUM_LAYERS=53 INPUT_SIZE=224 ZK4_TABLE_SIZE_LOG=12
  one "VGG-16_g$g" hook:bench_streaming_vgg:cv_config.yaml:1 "${G[@]}" VGG_VARIANT=16 NUM_LAYERS=13 INPUT_SIZE=32 ZK4_TABLE_SIZE_LOG=12
done

for s in ${SEQS:-32 64 256}; do
  one "GPT-2_seq$s" e2e "${TW[@]}" MORPHEUS_GPUS=4 MORPHEUS_AUTH_GPUS=4 MODEL=gpt2 NUM_LAYERS=12 SEQ_LEN=$s VOCAB_SIZE=50257
done

# LLaMA3-8B (32L): GPU scaling (one GPU cannot hold the key cache and the
# streamed sources) and sequence lengths; long witness generation per run.
if [ -n "${LLAMA3:-}" ]; then
  L3=("${TW[@]}" MODEL=llama3 NUM_LAYERS=32 NUM_HEADS=32 NUM_KV_HEADS=8 HEAD_DIM=128 FFN_DIM=14336 VOCAB_SIZE=128256)
  for g in ${L3GPUS:-2}; do one "LLaMA3-8B_g$g" e2e MORPHEUS_GPUS=$g MORPHEUS_AUTH_GPUS=$g SEQ_LEN=128 "${L3[@]}"; done
  for s in ${L3SEQS:-32 64 256}; do one "LLaMA3-8B_seq$s" e2e MORPHEUS_GPUS=4 MORPHEUS_AUTH_GPUS=4 SEQ_LEN=$s "${L3[@]}"; done
fi


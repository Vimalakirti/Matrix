#!/usr/bin/env bash
# Morpheus (S&P'27) end-to-end runs of the paper's eleven models at the
# configurations of scripts/bench/run_paper_e2e.py (the previous evaluation).
#
#   ./run_models.sh [model ...]      (default: all)
#
# Per model: N_MONO (3) monolithic requests (weights opened per request)
# and deferred batches of 2 and 8 requests, each with its measured
# finalization; every proof is serialized, decoded and verified. Logs go
# to $OUT (default scripts/bench/morpheus/out). Transformers run through
# morpheus_e2e; the rest through their benchmark binaries with
# MORPHEUS_AT_RUN (the benchmark's own forward pass is proven).
set -u
# Fail closed: debug and deterministic/private-coin overrides are not paper runs.
for flag in NO_AUTH NO_VERIFY LAYOUT_ONLY LINK_ROWS MORPHEUS_UNCLAIMED MORPHEUS_COIN_SEED; do
  if [ "${!flag+x}" ]; then echo "Refusing benchmark override: $flag" >&2; exit 2; fi
done
# Resolved before the cd below, so the lock works however the script is invoked.
GPU_LOCK_FILE=$(realpath -m "$(dirname "$0")/GPU_LOCK")
cd "$(dirname "$0")/../../.."
OUT=${OUT:-scripts/bench/morpheus/out}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,0,1}
export MORPHEUS_GPUS=${MORPHEUS_GPUS:-4} MORPHEUS_AUTH_GPUS=${MORPHEUS_AUTH_GPUS:-4}
export PROFILE=ternary B0=11 SLOTS_LOG=${SLOTS_LOG:-10} ZK4_EXP_BITS=8 MORPHEUS_TIMING=1
# Public argmax logit bound: |logit| < 2^20 at the logit scale (GPT-2 reaches ~2^16).
export ARGMAX_LOGIT_BITS=${ARGMAX_LOGIT_BITS:-20}
# Transformers: quantized-model-like dense weights (the previous evaluation
# used zero matrices) and the exact RMS reciprocal gadget.
TW=(WEIGHTS=${WEIGHTS:-dense} ZK4_EXACT_RMS=1)
MODES=${MODES:-"mono defer"}

# The server is shared: wait until no process holds GPU memory, so a foreign
# job cannot push a run out of memory or perturb its timing.
wait_gpus() {
  [ -n "${NO_GPU_WAIT:-}" ] && return
  local said=
  # GPU_LOCK (next to this script) marks a timed foreign run, e.g. a baseline.
  while [ -e "$GPU_LOCK_FILE" ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)" -gt 0 ]; do
    [ -z "$said" ] && echo "[$(date +%T)] waiting for free GPUs" && said=1
    sleep 60
  done
}
run() {   # name kind(e2e|hook:bin:cfg:units) env...
  local name=$1 kind=$2; shift 2
  for mode in $MODES; do
    local log="$OUT/${name}_${mode}.log"
    # Monolithic: N_MONO independent openings (fresh private randomness, so
    # the number of authentication attempts varies); the tables report the
    # mean. Deferred: measured finalizations of batches DEFER_BATCHES.
    local expected=${N_MONO:-3} batch_spec=-
    local m=(DEFER=0 N_REQUESTS=$expected)
    if [ "$mode" = defer ]; then
      batch_spec=${DEFER_BATCHES:-2,8}; expected=0
      [[ "$batch_spec" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || return 2
      local sizes j; IFS=, read -ra sizes <<< "$batch_spec"
      for j in "${sizes[@]}"; do expected=$((expected+j)); done
      m=(DEFER=1 DEFER_BATCHES=$batch_spec)
    elif [ "$mode" = mono ]; then
      # An inherited DEFER_BATCHES otherwise overrides N_REQUESTS in the driver.
      [ "${DEFER_BATCHES+x}" ] && { echo "Unset DEFER_BATCHES for mono runs" >&2; return 2; }
    else
      echo "Unknown benchmark mode: $mode" >&2; return 2
    fi
    [[ "$expected" =~ ^[1-9][0-9]*$ ]] && [ "$expected" -le 10000 ] || return 2
    # A run that fails for GPU memory while another user's job held a GPU
    # is retried (up to RETRIES times) once the GPUs are free again.
    local try=0
    while true; do
    wait_gpus
    echo "[$(date +%T)] $name $mode -> $log"
    rm -f "$log.foreign"
    # Peak summed GPU memory (MiB) over the run, sampled every 0.5 s; GPU
    # processes of other users are recorded in $log.foreign.
    # Only this run's devices (MORPHEUS_GPUS of CUDA_VISIBLE_DEVICES) count.
    local ng=$(for kv in "$@"; do case $kv in MORPHEUS_GPUS=*) echo "${kv#*=}";; esac; done | tail -1); ng=${ng:-$MORPHEUS_GPUS}
    local devs=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1-"$ng")
    ( peak=0; while true; do m=$(nvidia-smi -i "$devs" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}'); [ "$m" -gt "$peak" ] && peak=$m && echo "$peak" > "$log.peak"
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do u=$(ps -o user= -p "$p" 2>/dev/null); [ -n "$u" ] && [ "$u" != "$USER" ] && echo "$(date +%T) $u $p" >> "$log.foreign"; done
      sleep 0.5; done ) &
    local sampler=$!
    printf "MORPHEUS_EXPECT protocol=3 profile=morpheus-ternary-v1 requests=%s batches=%s\n" "$expected" "$batch_spec" > "$log"
    case $kind in
      e2e) env "$@" "${m[@]}" timeout ${TMO:-14400} ./target/release/morpheus_e2e >> "$log" 2>&1 ;;
      hook:*) IFS=: read -r _ bin cfg units <<< "$kind"
        env "$@" "${m[@]}" MORPHEUS_AT_RUN=1 MORPHEUS_NAME="$name" MORPHEUS_UNITS="$units" timeout ${TMO:-14400} ./target/release/$bin "$cfg" >> "$log" 2>&1 ;;
    esac
    local run_status=$?
    printf "MORPHEUS_EXIT status=%s\n" "$run_status" >> "$log"
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
    echo "peak GPU memory (sum over devices) $(cat "$log.peak" 2>/dev/null) MiB" >> "$log"
    grep -E "^(prove|request stage|batch J|verify per|proof per|peak GPU)|panicked" "$log" | sed 's/^/    /'
    if [ -s "$log.foreign" ] && grep -qiE "AllocationFailed|out of memory" "$log" && [ $try -lt ${RETRIES:-2} ]; then
      try=$((try + 1))
      echo "[$(date +%T)] $name $mode: GPU memory failure while another user's job was on the GPUs ($(head -1 "$log.foreign")); retry $try"
      mv "$log" "$log.contended$try"
      continue
    fi
    break
    done
  done
}

want() { [ $# -eq 0 ] && return 0; for w in $SEL; do [ "$w" = "$1" ] && return 0; done; return 1; }
SEL="$*"
sel() { [ -z "${SEL// /}" ] && return 0; for w in $SEL; do [ "$w" = "$1" ] && return 0; done; return 1; }

# Per-model configurations (kind, env...), shared by the four-GPU rows and
# the one-GPU scaling rows (g1).
C_gpt2=(e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=gpt2 NUM_LAYERS=12 SEQ_LEN=128 VOCAB_SIZE=50257)
C_bert=(e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=10 MODEL=bert NUM_LAYERS=24 SEQ_LEN=128)
# The 6B-8B LLMs at full depth, width and vocabulary (32, 32 and 28 layers).
# A request's witness takes ~36 minutes and ~670 GB of host memory, so each
# runs one monolithic request (LLM_MONO, LLM_MODES).
C_llama2=(e2e "${TW[@]}" REQ_PLANES_LOG=5 ZK4_TABLE_SIZE_LOG=24 MODEL=llama2 NUM_LAYERS=32 SEQ_LEN=128 NUM_HEADS=32 HEAD_DIM=128 FFN_DIM=11008 FFN_SHARDS=1 VOCAB_SIZE=32000)
C_llama3=(e2e "${TW[@]}" REQ_PLANES_LOG=5 ZK4_TABLE_SIZE_LOG=24 MODEL=llama3 NUM_LAYERS=32 SEQ_LEN=128 NUM_HEADS=32 NUM_KV_HEADS=8 HEAD_DIM=128 FFN_DIM=14336 VOCAB_SIZE=128256)
C_gptj=(e2e "${TW[@]}" REQ_PLANES_LOG=5 ZK4_TABLE_SIZE_LOG=24 MODEL=gptj NUM_LAYERS=28 SEQ_LEN=128 NUM_HEADS=16 HEAD_DIM=256 FFN_DIM=16384 VOCAB_SIZE=50400)
C_resnet=(hook:bench_streaming_resnet:cv_config.yaml:1 NUM_LAYERS=53 INPUT_SIZE=224 ZK4_TABLE_SIZE_LOG=12)
C_vgg=(hook:bench_streaming_vgg:cv_config.yaml:1 VGG_VARIANT=16 NUM_LAYERS=13 INPUT_SIZE=32 ZK4_TABLE_SIZE_LOG=12)
# YOLOv11n at 320x320 (640x640 needs ~1,860 sources: the exact SiLU range checks).
C_yolo=(hook:bench_streaming_yolo:cv_config.yaml:1 NUM_STAGES=8 INPUT_SIZE=${YOLO_INPUT:-320} ZK4_TABLE_SIZE_LOG=12)
C_unet3d=(hook:bench_streaming_unet3d:cv_config.yaml:1 NUM_LAYERS=6 INPUT_D=64 INPUT_H=64 INPUT_W=64 ZK4_TABLE_SIZE_LOG=12 REQ_PLANES_LOG=5)
# PointPillars on a 64x64 grid with 1,024 pillars: exact private grid routing
# is O(pillars x grid cells) (432x496 with 12,000 pillars needs ~9e11 cells).
# Its graph masks need more than 16 slots per node (2^19 in all).
C_pointpillars=(hook:bench_streaming_pointpillar:cv_config.yaml:1 NX=64 NY=64 N_PILLARS=1024 MAX_POINTS=32 NUM_ANCHORS=2 NUM_CLASSES=3 ZK4_TABLE_SIZE_LOG=12 MORPHEUS_GRID_ROUTING_MAX_CELLS=2147483648 SLOTS_LOG=19)
# Whisper-tiny (4+4 layers, full width) on a 10 s window: WHISPER_CTX (500)
# audio frames instead of 1,500. The exact softmax range checks grow with
# the square of the context; at 1,500 frames even a 2+2-layer request
# exceeds 905 GB of host memory.
C_whisper=(hook:bench_streaming_whisper:llama2_config.yaml:1 NUM_ENC_LAYERS=4 NUM_DEC_LAYERS=4 N_MELS=80 N_AUDIO_CTX=${WHISPER_CTX:-500} N_TEXT_CTX=448 N_STATE=384 N_HEAD=6 ZK4_TABLE_SIZE_LOG=24)
MODELS=(gpt2:GPT-2 bert:BERT-Large llama2:LLaMA2-7B llama3:LLaMA3-8B gptj:GPT-J-6B resnet:ResNet-50 vgg:VGG-16 yolo:YOLOv11n unet3d:3D-UNet pointpillars:PointPillars whisper:Whisper)
cfg() { local v="C_$1[@]"; printf '%s\0' "${!v}"; }
for mn in "${MODELS[@]}"; do
  k=${mn%%:*}; sel "$k" || continue
  mapfile -d '' c < <(cfg "$k")
  case $k in
    llama2|llama3|gptj) N_MONO=${LLM_MONO:-1} MODES=${LLM_MODES:-mono} run "${mn#*:}" "${c[@]}" ;;
    *) run "${mn#*:}" "${c[@]}" ;;
  esac
done

# ---- Comparisons (selected explicitly, never part of the default set) ----
# DeepProve-matched configurations (scripts/bench/deepprove): GPT-2 and
# LLaMA2-7B truncated to L layers, 16 prompt tokens, sequence 32 (16
# generated) or 128 (112 generated).
L2=(MODEL=llama2 NUM_HEADS=32 HEAD_DIM=128 FFN_DIM=11008 FFN_SHARDS=1 VOCAB_SIZE=32000)
want_x() { for w in $SEL; do [ "$w" = "$1" ] && return 0; done; return 1; }
# Monolithic only: the comparison reports one accepted proof per request.
want_x dp && MODES=mono && for s in 32 128; do
  [ $s = 32 ] && run GPT-2_s32 e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=gpt2 NUM_LAYERS=12 SEQ_LEN=32 VOCAB_SIZE=50257
  for l in 2 16 32; do
    [ $l = 32 ] && [ $s = 128 ] && continue   # the main LLaMA2-7B run
    run LLaMA2-7B_L${l}_s$s e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 "${L2[@]}" NUM_LAYERS=$l SEQ_LEN=$s
  done
done
# DeepProve's GPU build uses one GPU, so the comparison runs Morpheus on one
# GPU, for GPT-2 only.
G1=(MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1)
{ want_x dp1 || want_x dp1s32 || want_x dp1s128; } && MODES=mono && for s in 32 128; do
  { want_x dp1 || want_x dp1s$s; } || continue
  run GPT-2_s${s}_g1 e2e "${TW[@]}" "${G1[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=gpt2 NUM_LAYERS=12 SEQ_LEN=$s VOCAB_SIZE=50257
done
# DeepProve's own CNNs (zkml/assets/scripts) with their trained weights and
# first input at scale 2^8 (scripts/bench/deepprove/export_cnn.py). COVID:
# the final sigmoid dropped and the last layer padded to 10 zero-extended
# logits, as DeepProve needs to prove it (covid10.onnx).
# Fixtures from scripts/bench/deepprove/prepare_cnn.sh <work_dir>.
CNNFX=${CNNFX:-${DP_WORK:-deepprove_work}/cnn_fixtures}
{ want_x dp1 || want_x dp1cnn; } && MODES=mono run CIFAR-LeNet_g1 hook:bench_deepprove_cnn:cv_config.yaml:1 "${G1[@]}" CNN_FIXTURE=$CNNFX/cifar.json ZK4_TABLE_SIZE_LOG=16
{ want_x dp1 || want_x dp1cnn; } && MODES=mono run COVID-CNN_g1 hook:bench_deepprove_cnn:cv_config.yaml:1 "${G1[@]}" CNN_FIXTURE=$CNNFX/covid10.json ZK4_TABLE_SIZE_LOG=20 REQ_PLANES_LOG=5
# Source-sharding scaling: the same monolithic requests on one GPU (models
# whose sources fit one device; the LLMs need the memory of four).
# One-GPU scaling rows (g1): the same configurations on one device. The
# 6B-8B LLMs are left out (one GPU cannot hold their requests).
# G1_MODELS (e.g. "yolo unet3d") restricts the set.
want_x g1 && MODES=mono && for mn in "${MODELS[@]}"; do
  k=${mn%%:*}; case $k in llama2|llama3|gptj) continue;; esac
  [ -n "${G1_MODELS:-}" ] && [[ " $G1_MODELS " != *" $k "* ]] && continue
  mapfile -d '' c < <(cfg "$k"); run "${mn#*:}_g1" "${c[@]}" "${G1[@]}"
done
# EZKL's two reference circuits at their exported shapes, one GPU.
want_x ezkl && MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} \
  run LeNet-5 hook:bench_streaming_lenet:llama2_config.yaml:1 MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1
want_x ezkl && run nanoGPT hook:bench_streaming_nanogpt:llama2_config.yaml:64 MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1 SEQ_LEN=64
# The EZKL GPU build proves nanoGPT only at sequence length 8 (seq 64 runs out
# of GPU memory in proving-key generation); the identical inference here.
want_x ezkl8 && MODES=mono run nanoGPT_s8 hook:bench_streaming_nanogpt:llama2_config.yaml:8 MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1 SEQ_LEN=8
# Convolution ablation: the polynomial-product reduction with a committed
# full-convolution advice (MORPHEUS_CONV=pp) against the default.
want_x convpp && MODES=mono run VGG-16_pp hook:bench_streaming_vgg:cv_config.yaml:1 MORPHEUS_CONV=pp VGG_VARIANT=16 NUM_LAYERS=13 INPUT_SIZE=32 ZK4_TABLE_SIZE_LOG=12
want_x convpp && MODES=mono run ResNet-50_pp hook:bench_streaming_resnet:cv_config.yaml:1 MORPHEUS_CONV=pp NUM_LAYERS=53 INPUT_SIZE=224 ZK4_TABLE_SIZE_LOG=12
# ResNet-50 with trained weights (BatchNorm folded) and real images:
# RESNET_REAL is an export of scripts/export_resnet50.py, RESNET_IMAGE picks
# the image. Real activations reach 21 bits, so this row needs 32 request
# planes and a 2^22 range table.
want_x resnet_real && MODES=mono run ResNet-50-real hook:bench_streaming_resnet:cv_config.yaml:1 NUM_LAYERS=53 INPUT_SIZE=224 ZK4_TABLE_SIZE_LOG=${RESNET_TSL:-24} REQ_PLANES_LOG=5 SLOTS_LOG=${RESNET_SLOTS:-15}

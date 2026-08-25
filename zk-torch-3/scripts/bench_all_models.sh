#!/usr/bin/env bash
# bench_all_models.sh — Full model benchmark suite
#
# Usage:
#   bash scripts/bench_all_models.sh
#   bash scripts/bench_all_models.sh --quick          # fewer layers for fast check
#   source tuned_thresholds.env && bash scripts/bench_all_models.sh
#
# Reads threshold env vars if set, otherwise uses defaults.
# Auto-detects GPUs and sets NUM_PARTITIONS accordingly.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---full}"

# ── Auto-detect GPUs ─────────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: No GPUs detected" >&2
  exit 1
fi
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)

# ── Read thresholds (from env or defaults) ───────────────────────────────────
SC_TH="${ZK_GPU_SUMCHECK_THRESHOLD:-14}"
PE_TH="${ZK_GPU_PARTIAL_EVAL_THRESHOLD:-16}"
FU_TH="${ZK_GPU_FUSED_THRESHOLD:-16}"
OP_TH="${CPU_OPEN_THRESHOLD:-14}"

# ── Model configurations ────────────────────────────────────────────────────
# Format: "binary:layers:description"
declare -a MODELS
if [ "$MODE" = "--quick" ]; then
  MODELS=(
    "gpt2:4:GPT-2 Small"
    "bert:4:BERT-Large"
    "gptj:1:GPT-J 6B"
    "llama:1:LLaMA-2 7B"
    "vgg:4:VGG-16"
    "resnet:8:ResNet-50"
  )
else
  MODELS=(
    "gpt2:12:GPT-2 Small"
    "bert:24:BERT-Large"
    "gptj:28:GPT-J 6B"
    "llama:32:LLaMA-2 7B"
    "vgg:13:VGG-16"
    "resnet:53:ResNet-50"
  )
fi

# ── Timeout per model (seconds) ─────────────────────────────────────────────
RUN_TIMEOUT=1200

# ── Helper: parse time from Rust Duration debug format ───────────────────────
parse_time_to_seconds() {
  local raw="$1"
  if [[ "$raw" =~ ^([0-9.]+)s$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$raw" =~ ^([0-9.]+)ms$ ]]; then
    echo "${BASH_REMATCH[1]}" | awk '{printf "%.3f", $1/1000}'
  elif [[ "$raw" =~ ^([0-9.]+)µs$ ]] || [[ "$raw" =~ ^([0-9.]+)us$ ]]; then
    echo "${BASH_REMATCH[1]}" | awk '{printf "%.6f", $1/1000000}'
  else
    echo "N/A"
  fi
}

format_time() {
  local val="$1"
  if [ "$val" = "N/A" ] || [ "$val" = "FAIL" ]; then
    echo "$val"
  elif [ "$(echo "$val" | awk '{print ($1 < 1 && $1 > 0)}')" = "1" ]; then
    echo "$val" | awk '{printf "%.0fms", $1*1000}'
  else
    echo "$val" | awk '{printf "%.2fs", $1}'
  fi
}

echo "============================================="
echo "  Full Model Benchmark Suite"
echo "============================================="
echo ""
echo "GPU: ${NUM_GPUS}x ${GPU_NAME}"
echo "Max partitions: ${NUM_GPUS} (per-model: min(layers, gpus))"
echo "Mode: ${MODE}"
echo ""
echo "Thresholds:"
echo "  ZK_GPU_SUMCHECK_THRESHOLD=$SC_TH"
echo "  ZK_GPU_PARTIAL_EVAL_THRESHOLD=$PE_TH"
echo "  ZK_GPU_FUSED_THRESHOLD=$FU_TH"
echo "  CPU_OPEN_THRESHOLD=$OP_TH"
echo ""

# ── Build all binaries ───────────────────────────────────────────────────────
echo "Building all binaries..."
for entry in "${MODELS[@]}"; do
  IFS=: read -r bin layers desc <<< "$entry"
  cargo build --release --bin "$bin" --manifest-path "$PROJECT_DIR/Cargo.toml" 2>/dev/null
done
echo "Build complete."
echo ""

# ── Run benchmarks ───────────────────────────────────────────────────────────
# Store results for table output
declare -a RESULT_NAMES RESULT_LAYERS RESULT_NODES RESULT_EDGES
declare -a RESULT_RUN RESULT_COMMIT RESULT_PROVE RESULT_TOTAL RESULT_VERIFY RESULT_VERIFIED

IDX=0
ALL_VERIFIED=true

for entry in "${MODELS[@]}"; do
  IFS=: read -r bin layers desc <<< "$entry"
  BINARY="$PROJECT_DIR/target/release/${bin}"

  # Use partitions only when enough layers to split meaningfully
  if [ "$layers" -ge "$NUM_GPUS" ]; then
    MODEL_PARTITIONS=$NUM_GPUS
  else
    MODEL_PARTITIONS=1
  fi

  echo "--- ${desc} (${layers} layers, ${MODEL_PARTITIONS} partitions) ---"

  output=$(timeout "$RUN_TIMEOUT" env \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    NUM_LAYERS="$layers" \
    NUM_PARTITIONS="$MODEL_PARTITIONS" \
    ZK_GPU_SUMCHECK_THRESHOLD="$SC_TH" \
    ZK_GPU_PARTIAL_EVAL_THRESHOLD="$PE_TH" \
    ZK_GPU_FUSED_THRESHOLD="$FU_TH" \
    CPU_OPEN_THRESHOLD="$OP_TH" \
    "$BINARY" 2>&1) || {
      echo "  FAILED (timeout or error)"
      RESULT_NAMES[$IDX]="$desc"
      RESULT_LAYERS[$IDX]="$layers"
      RESULT_NODES[$IDX]="?"
      RESULT_EDGES[$IDX]="?"
      RESULT_RUN[$IDX]="FAIL"
      RESULT_COMMIT[$IDX]="FAIL"
      RESULT_PROVE[$IDX]="FAIL"
      RESULT_TOTAL[$IDX]="FAIL"
      RESULT_VERIFY[$IDX]="FAIL"
      RESULT_VERIFIED[$IDX]="false"
      ALL_VERIFIED=false
      IDX=$((IDX + 1))
      continue
    }

  # Parse all fields
  nodes=$(echo "$output" | grep -oP 'DAG: \K[0-9]+(?= nodes)' | head -1)
  edges=$(echo "$output" | grep -oP ', \K[0-9]+(?= edges)' | head -1)
  run_raw=$(echo "$output" | grep -oP 'Run: \K[0-9.]+[a-zµ]+' | head -1)
  commit_raw=$(echo "$output" | grep -oP 'Commit: \K[0-9.]+[a-zµ]+' | head -1)
  prove_raw=$(echo "$output" | grep -oP 'Prove: \K[0-9.]+[a-zµ]+' | head -1)
  total_raw=$(echo "$output" | grep -oP '= \K[0-9.]+[a-zµ]+' | head -1)
  verify_raw=$(echo "$output" | grep -oP 'Verify: \K[0-9.]+[a-zµ]+' | head -1)
  verified=$(echo "$output" | grep -oP 'Verified: \K\w+' | head -1)

  run_s=$(parse_time_to_seconds "${run_raw:-0s}")
  commit_s=$(parse_time_to_seconds "${commit_raw:-0s}")
  prove_s=$(parse_time_to_seconds "${prove_raw:-0s}")
  total_s=$(parse_time_to_seconds "${total_raw:-${prove_raw:-0s}}")
  verify_s=$(parse_time_to_seconds "${verify_raw:-0s}")

  echo "  Nodes: ${nodes:-?}, Edges: ${edges:-?}"
  echo "  Run: $(format_time "$run_s"), Commit: $(format_time "$commit_s")"
  echo "  Prove: $(format_time "$prove_s"), Total: $(format_time "$total_s")"
  echo "  Verify: $(format_time "$verify_s"), Verified: ${verified:-?}"
  echo ""

  RESULT_NAMES[$IDX]="$desc"
  RESULT_LAYERS[$IDX]="$layers"
  RESULT_NODES[$IDX]="${nodes:-?}"
  RESULT_EDGES[$IDX]="${edges:-?}"
  RESULT_RUN[$IDX]="$run_s"
  RESULT_COMMIT[$IDX]="$commit_s"
  RESULT_PROVE[$IDX]="$prove_s"
  RESULT_TOTAL[$IDX]="$total_s"
  RESULT_VERIFY[$IDX]="$verify_s"
  RESULT_VERIFIED[$IDX]="${verified:-false}"

  if [ "${verified:-false}" != "true" ]; then
    ALL_VERIFIED=false
  fi

  IDX=$((IDX + 1))
done

# ── Output markdown table ───────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Results Summary"
echo "============================================="
echo ""
echo "| Model | Layers | Nodes | Edges | Run | Commit | Prove | Total Prove | Verify | Verified |"
echo "|-------|--------|-------|-------|-----|--------|-------|-------------|--------|----------|"

for ((i = 0; i < IDX; i++)); do
  printf "| %-15s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
    "${RESULT_NAMES[$i]}" \
    "${RESULT_LAYERS[$i]}" \
    "${RESULT_NODES[$i]}" \
    "${RESULT_EDGES[$i]}" \
    "$(format_time "${RESULT_RUN[$i]}")" \
    "$(format_time "${RESULT_COMMIT[$i]}")" \
    "$(format_time "${RESULT_PROVE[$i]}")" \
    "$(format_time "${RESULT_TOTAL[$i]}")" \
    "$(format_time "${RESULT_VERIFY[$i]}")" \
    "${RESULT_VERIFIED[$i]}"
done

echo ""
echo "All verified: $ALL_VERIFIED"
echo ""
echo "Configuration:"
echo "  GPU: ${NUM_GPUS}x ${GPU_NAME}"
echo "  Max partitions: $NUM_GPUS"
echo "  ZK_GPU_SUMCHECK_THRESHOLD=$SC_TH"
echo "  ZK_GPU_PARTIAL_EVAL_THRESHOLD=$PE_TH"
echo "  ZK_GPU_FUSED_THRESHOLD=$FU_TH"
echo "  CPU_OPEN_THRESHOLD=$OP_TH"

# ── Save results to CSV ─────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_CSV="bench_all_models_${TIMESTAMP}.csv"
echo "model,layers,nodes,edges,run_s,commit_s,prove_s,total_prove_s,verify_s,verified" > "$RESULTS_CSV"
for ((i = 0; i < IDX; i++)); do
  echo "${RESULT_NAMES[$i]},${RESULT_LAYERS[$i]},${RESULT_NODES[$i]},${RESULT_EDGES[$i]},${RESULT_RUN[$i]},${RESULT_COMMIT[$i]},${RESULT_PROVE[$i]},${RESULT_TOTAL[$i]},${RESULT_VERIFY[$i]},${RESULT_VERIFIED[$i]}" >> "$RESULTS_CSV"
done
echo ""
echo "CSV saved to: $RESULTS_CSV"

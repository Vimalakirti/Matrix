#!/usr/bin/env bash
# DeepProve vs zk-torch-4 on one machine, at the paper's AR workloads.
#
#   ./run_all.sh <work_dir> <zk-torch-4 root> <out_dir>
#
# <work_dir> is what setup.sh built. <zk-torch-4 root> is the checkout whose
# target/release binaries and scripts/bench harness are measured; build it
# first (cargo build --release --bin bench_streaming_oneshot_gpt2 --bin
# bench_streaming_oneshot_llama2).
#
# Everything runs sequentially so no two provers share the CPU or a GPU.
# Knobs (env): GPU (single-device runs, default 2), ALL_GPUS (multi-GPU
# zk-torch-4 runs, default 0,1,2,3), REPS_GPT2 (default 3), REPS_LLAMA (1),
# STAGES (default "dp_cuda": the paper compares GPU builds only; dp_cpu,
# zkt_1 and zkt_all are the earlier CPU-build and zk-torch-4 harness).
set -uo pipefail
WORK=$(realpath "${1:?work_dir}")
ZKT=$(realpath "${2:?zk-torch-4 root}")
OUT=$(realpath -m "${3:?out_dir}")
HERE=$(cd "$(dirname "$0")" && pwd)
GPU=${GPU:-2}
ALL_GPUS=${ALL_GPUS:-0,1,2,3}
REPS_GPT2=${REPS_GPT2:-3}
REPS_LLAMA=${REPS_LLAMA:-1}
STAGES=${STAGES:-"dp_cuda"}
CONFIGS="16:32,16:128"     # (prompt, prompt+generated): P16 G16 and P16 G112
GENS="16,112"
mkdir -p "$OUT"
# DeepProve's prover recurses over its batched lookup expressions, and on
# Llama-2-7B at 16 layers the main thread overflows the default 8 MiB stack.
ulimit -s unlimited

dp() {  # backend bin gpu model reps
    local extra=()
    [ -n "$3" ] && extra=(--gpu "$3")
    LD_LIBRARY_PATH="$WORK/cuda12_libs:${LD_LIBRARY_PATH:-}" \
    python3 "$HERE/run_deepprove.py" --bin "$2" --backend "$1" "${extra[@]}" \
        --run-dir "$WORK/run" --models "$4" --configs "$CONFIGS" --reps "$5" \
        --out "$OUT/deepprove.csv"
}

zkt() {  # label ngpus devices model proofs timeout_s
    (cd "$ZKT/scripts/bench" && python3 run_paper_e2e.py --ar --ar-prompt 16 \
        --seq "$GENS" --gpus "$2" --only "$4" --proofs "$5" --timeout "$6" \
        --env "ZK4_GPU_DEVICES=$3" --out "$OUT/zkt_${1}_${4}.csv")
}

for s in $STAGES; do
    echo "=== stage $s  $(date -Is)"
    case $s in
    dp_cpu)
        dp cpu "$WORK/deep-prove/target/release/bench-llm" "" gpt2 "$REPS_GPT2"
        dp cpu "$WORK/deep-prove/target/release/bench-llm" "" llama2 "$REPS_LLAMA" ;;
    dp_cuda)
        dp cuda "$WORK/deep-prove/target-cuda/release/bench-llm" "$GPU" gpt2 "$REPS_GPT2"
        # Full 32-layer LLaMA2-7B overflows DeepProve's u16 witness ids
        # (README); DP_LLAMA=llama2-L16 runs a truncation.
        [ -n "${DP_LLAMA:-}" ] && dp cuda "$WORK/deep-prove/target-cuda/release/bench-llm" "$GPU" "$DP_LLAMA" "$REPS_LLAMA" ;;
    # Llama-2-7B is not run on one A100: on 8xH200 its P16 G16 / G112 proofs
    # peak at 112 / 212 GB of GPU memory. On one 80 GB A100 the P16 G16 case
    # had not finished after 49 min (232 GB host RSS, GPU idle, one core busy)
    # and was stopped.
    zkt_1)
        zkt 1gpu 1 "$GPU" gpt-2 8 7200 ;;
    # N=2 for Llama-2 (the paper's FAST_PROOFS): N=1 and N->inf are derived
    # from the same prove/acc/finalize split whatever N is, and N=8 would take
    # ~5x longer per case.
    zkt_all)
        n=$(awk -F, '{print NF}' <<<"$ALL_GPUS")
        zkt "${n}gpu" "$n" "$ALL_GPUS" gpt-2 8 7200
        zkt "${n}gpu" "$n" "$ALL_GPUS" llama-2 2 14400 ;;
    esac
done
echo "=== done $(date -Is)"
ls "$OUT"/zkt_*.csv >/dev/null 2>&1 && python3 "$HERE/make_table.py" "$OUT"
cat "$OUT/deepprove.csv"

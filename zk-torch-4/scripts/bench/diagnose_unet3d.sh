#!/usr/bin/env bash
# Diagnose the 3D-UNet illegal-memory-access at 128^3.
#
#   cd zk-torch-4/scripts/bench && ./diagnose_unet3d.sh
#
# Symptom: at 128^3 with a decoder (NUM_LAYERS >= 2) some kernel writes out of
# bounds. CUDA errors are sticky, so it surfaces later as
#
#   device->host copy failed: MemcpyFailed("an illegal memory access ... (700)")
#
# from a lazy host download in Dag::run, which is NOT where the fault happened.
#
# Established so far, all by experiment:
#   - 128^3 encoder-only (NUM_LAYERS=1): clean for 17 minutes
#   - 64^3 with all 6 levels:            clean, verifies
#   - 128^3 NUM_LAYERS=2 (one decoder level): FAULTS in ~2 min   <- the repro
#   - not OOM, not multi-GPU placement (1 GPU fails identically)
#   - not ConvTranspose3D's kernel (disabling it, the fault remains)
#   - not Conv3D's X read, W read or Y write (bounds-guarded, guards never fire)
#
# Stage A is the one that matters: compute-sanitizer names the kernel and the
# address in a single run. It needs driver >= 580 for the CUDA 13 build; the
# machine this was developed on has 560, which is why it was unavailable there.

set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
BIN=$ROOT/target/release/bench_streaming_unet3d
CFG=$ROOT/cv_config.yaml
OUT=${OUT_DIR:-/tmp/unet3d_diag}
mkdir -p "$OUT"

# The minimal reproducer. One GPU on purpose: it faults the same way and keeps
# the sanitizer's output attributable to a single device.
REPRO_ENV=(
  NUM_LAYERS=2 INPUT_D=128 INPUT_H=128 INPUT_W=128
  MAX_NUM_VARS=28 ZK4_TABLE_SIZE_LOG=12 ZK4_TABLE_COMMIT_LOG=2
  N_PROOFS=1 NUM_PARTITIONS=1 BATCH=1
  ZKT_RUN_BACKEND=gpu ZK4_GPU_DEVICES=0
)

echo "=== environment ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv | head -3
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "driver major: $DRIVER  (compute-sanitizer from CUDA 13 needs >= 580)"
[[ -x "$BIN" ]] || { echo "MISSING $BIN -- run: cargo build --release"; exit 1; }
echo

SAN=""
for c in /usr/local/cuda/bin/compute-sanitizer /usr/local/cuda-*/bin/compute-sanitizer \
         /usr/bin/compute-sanitizer; do
  [[ -x "$c" ]] || continue
  # Reject a build whose injection library is missing -- it exits instantly
  # with "Unable to find injection library" and looks like a clean run.
  d=$(dirname "$(dirname "$c")")
  [[ -f "$d/compute-sanitizer/libsanitizer-collection.so" ]] && { SAN=$c; break; }
done

echo "=== [A] compute-sanitizer (the one that answers it outright) ==="
if [[ -n "$SAN" ]]; then
  echo "using $SAN -- expect 10-100x slowdown, allow up to an hour"
  env "${REPRO_ENV[@]}" timeout 5400 "$SAN" --tool memcheck "$BIN" "$CFG" \
      > "$OUT/sanitizer.log" 2>&1
  echo "--- first errors ---"
  grep -E "Invalid|out-of-bounds|of size|Program hit|ERROR SUMMARY" \
      "$OUT/sanitizer.log" | head -25
  grep -A12 "Invalid __global__" "$OUT/sanitizer.log" | head -30
else
  echo "no usable compute-sanitizer found; skipping to [B]"
fi
echo

echo "=== [B] in-kernel bounds guard (works without the sanitizer) ==="
# Range-checks every conv3d access against the real allocation sizes and
# reports the first offender rather than faulting.
env "${REPRO_ENV[@]}" ZK4_CONV_BOUNDS=1 timeout 1800 "$BIN" "$CFG" \
    > "$OUT/bounds.log" 2>&1
grep -E "OOB|illegal memory access|out of bounds|Verified" "$OUT/bounds.log" | head -5
echo

echo "=== [C] bisection: which GPU kernel, by turning them off ==="
for pair in "baseline:" "ct3d_off:ZK4_CT3D_CPU=1" "conv3d_off:ZK4_CONV3D_CPU=1" \
            "both_off:ZK4_CT3D_CPU=1 ZK4_CONV3D_CPU=1"; do
  name=${pair%%:*}; flags=${pair#*:}
  printf "  %-12s " "$name"
  # shellcheck disable=SC2086
  env "${REPRO_ENV[@]}" $flags timeout "${BISECT_TIMEOUT:-1800}" "$BIN" "$CFG" \
      > "$OUT/bisect_$name.log" 2>&1
  # Three outcomes, not two. Reporting a timeout as "clean" is worse than
  # useless: it reads as evidence that the kernel is innocent.
  if   grep -qE "illegal memory access" "$OUT/bisect_$name.log"; then echo "FAULTS"
  elif grep -qE "^Verified: (true|false)"  "$OUT/bisect_$name.log"; then echo "COMPLETED (no fault)"
  else echo "NO VERDICT (timed out at ${BISECT_TIMEOUT:-1800}s -- tells us nothing)"; fi
done

echo
echo "=== done. logs in $OUT/ ==="
echo "Most useful to send back: $OUT/sanitizer.log if stage [A] ran."

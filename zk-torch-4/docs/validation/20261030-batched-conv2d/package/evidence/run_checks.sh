#!/usr/bin/env bash
# Run from any directory. RUSTUP_HOME/CARGO_HOME/PATH may select the toolchain.
set -u
if [ "$#" -ne 4 ]; then echo 'usage: run_checks.sh SOURCE HARNESS TARGET_DIR LOG_DIR' >&2; exit 2; fi
source_dir=$(realpath "$1"); harness=$2; target=$3; logs=$4
mkdir -p "$logs"
python3 "$(dirname "$0")/prepare_harness.py" "$source_dir" "$harness" > "$logs/prepare.json" || exit $?
export CARGO_INCREMENTAL=0 CARGO_TARGET_DIR="$target"
manifest="$harness/zk-torch-4/Cargo.toml"
set +e
cargo check --all-targets --offline --manifest-path "$manifest" > "$logs/all-targets-check.log" 2>&1
check=$?
cargo build --bin cpu_conv_forward --offline --manifest-path "$manifest" > "$logs/cpu-forward-build.log" 2>&1
build=$?
forward=not_run
if [ "$build" -eq 0 ]; then
  RAYON_NUM_THREADS=2 "$target/debug/cpu_conv_forward" "$source_dir/zk-torch-4/bench_config.yaml" > "$logs/cpu-forward.log" 2>&1
  forward=$?
fi
cargo build --bin cpu_conv_replay --offline --manifest-path "$manifest" > "$logs/piop-replay-build.log" 2>&1
replay_build=$?
replay=not_run
if [ "$replay_build" -eq 0 ]; then
  MORPHEUS_PIOP_GPU=0 RAYON_NUM_THREADS=2 "$target/debug/cpu_conv_replay" "$source_dir/zk-torch-4/bench_config.yaml" > "$logs/piop-replay.log" 2>&1
  replay=$?
fi
printf 'all_targets_check=%s\ncpu_forward_build=%s\ncpu_forward_run=%s\npiop_replay_build=%s\npiop_replay_run=%s\n' "$check" "$build" "$forward" "$replay_build" "$replay" > "$logs/status.txt"
cat "$logs/status.txt"
[ "$check" -eq 0 ] && [ "$build" -eq 0 ] && [ "$forward" = 0 ] || exit 1
[ "$replay_build" -eq 0 ] || exit 3
[ "$replay" = 0 ]

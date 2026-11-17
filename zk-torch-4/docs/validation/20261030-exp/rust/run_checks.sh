#!/usr/bin/env bash
# Requires an already available Rust/Cargo toolchain; installs nothing.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
SOURCE=$(cd -- "${1:?usage: run_checks.sh SOURCE NEW_WORK_DIRECTORY}" && pwd)
WORK=${2:?supply a new work directory outside SOURCE}
python3 "$HERE/verify_sources.py" "$SOURCE"
python3 - "$SOURCE" "$WORK" <<'PY'
import pathlib,sys
source,work=map(lambda p:pathlib.Path(p).resolve(),sys.argv[1:])
assert not work.exists(), 'work directory must not already exist'
assert not work.is_relative_to(source), 'work directory must be outside source'
work.mkdir(parents=True)
PY
WORK=$(cd -- "$WORK" && pwd)
# Avoid inherited profile overrides making the replay incomparable.
unset ZK4_EXP_BITS ZK4_SF_LOG ZK4_TABLE_SIZE_LOG ZK4_TABLE_COMMIT_LOG
rustc -Vv > "$WORK/toolchain.txt"
cargo -Vv >> "$WORK/toolchain.txt"
python3 "$HERE/prepare_harness.py" "$SOURCE" "$WORK/typecheck" typecheck > "$WORK/typecheck-adaptations.stdout"
export CARGO_TARGET_DIR="$WORK/target"
cargo check --locked --all-targets --manifest-path "$WORK/typecheck/zk-torch-4/Cargo.toml" > "$WORK/typecheck.log" 2>&1
python3 "$HERE/prepare_harness.py" "$SOURCE" "$WORK/cpu" cpu-graph > "$WORK/cpu-adaptations.stdout"
cargo build --locked --manifest-path "$WORK/cpu/zk-torch-4/Cargo.toml" --bin cpu_basicblock_exp --bin cpu_graph_exp > "$WORK/cpu-build.log" 2>&1
# No CLI arguments: original non-test CONFIG loader uses its built-in default.
# Empty working directory prevents an unrelated config.yaml being picked up.
mkdir "$WORK/runtime"
cd "$WORK/runtime"
for sf in $(seq 0 30); do
  echo "SCALE=$sf BASICBLOCK"
  ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/cpu_basicblock_exp"
  echo "SCALE=$sf GRAPH"
  ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/cpu_graph_exp"
done > "$WORK/cpu-scales.log" 2>&1
COUNT=$(grep -c '^PASS ' "$WORK/cpu-scales.log")
test "$COUNT" -eq 434
printf 'PASS: modified-build all-target typecheck; %s original CPU test-body invocations. Not a production/CUDA build or PIOP replay.\n' "$COUNT"

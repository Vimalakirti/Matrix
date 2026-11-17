#!/usr/bin/env bash
# EZKL GPU build on its reference LeNet-5 and nanoGPT (one GPU).
#
#   ./run_gpu.sh <work_dir> [lenet5|nanogpt8|nanogpt64 ...]   (default: all three)
#
# lenet5:    examples/onnx/lenet_5 of EZKL v22.2.2
# nanogpt8:  models/nanoGPT_seq8 here: EZKL's nanoGPT (4 layers, 4 heads,
#            width 64, vocabulary 65) at sequence length 8 (gen.py; the
#            exported network.onnx and input.json are the measured files)
# nanogpt64: examples/onnx/nanoGPT, length 64. Key generation exhausts the
#            A100's 80 GB here (reported as OOM).
# REPS timed repetitions per model (default 1, as in the paper) after one
# untimed preparation; results in <work_dir>/runs/<label>_r<i>/metrics.json.
# Env: GPU (default 2), REPS.
set -uo pipefail
WORK=$(realpath "${1:?work_dir}"); shift
HERE=$(cd "$(dirname "$0")" && pwd)
. "$WORK/env.sh"
export EZKL_WORK=$WORK RUST_LOG=info SETUP_TIMEOUT=${SETUP_TIMEOUT:-5400}
EX=$WORK/src/ezkl/examples/onnx
mkdir -p "$WORK/logs"
for m in ${@:-lenet5 nanogpt8 nanogpt64}; do
  case $m in
    lenet5) dir=$EX/lenet_5 ;;
    nanogpt8) dir=$HERE/models/nanoGPT_seq8 ;;
    nanogpt64) dir=$EX/nanoGPT ;;
    *) echo "unknown model $m"; exit 2 ;;
  esac
  PHASE=prep python3 "$HERE/prove.py" "${m}_r1" "$WORK/bin/ezkl" "$dir" gpu none > "$WORK/logs/${m}_prep.log" 2>&1
  for r in $(seq 1 "${REPS:-1}"); do
    [ $r -gt 1 ] && { rm -rf "$WORK/runs/${m}_r$r"; cp -r "$WORK/runs/${m}_r1" "$WORK/runs/${m}_r$r"; }
    PHASE=timed python3 "$HERE/prove.py" "${m}_r$r" "$WORK/bin/ezkl" "$dir" gpu none > "$WORK/logs/${m}_r$r.log" 2>&1
    echo "$m r$r: exit $?"
    python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(' prove',m.get('prove_internal_s'),'verify',m.get('verify_internal_s'),'bytes',m.get('proof_bytes'),'verified',m.get('verified'),m.get('killed',''))" "$WORK/runs/${m}_r$r/metrics.json"
  done
done

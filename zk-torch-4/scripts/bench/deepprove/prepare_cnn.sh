#!/usr/bin/env bash
# DeepProve's two ONNX CNNs (zkml/assets/scripts) as inputs for both systems.
#
#   ./prepare_cnn.sh <work_dir>          (after setup.sh)
#
# Writes <work_dir>/cnn_fixtures/:
#   cifar_input.json, covid_input.json   DeepProve's shipped inputs, decompressed
#   covid_nosig.onnx                     COVID CNN without its final Sigmoid
#                                        (DeepProve cannot parse it)
#   covid10.onnx                         ... with the one logit padded by nine zero
#                                        rows (DeepProve panics on a 1-element
#                                        output and fails on 2 outputs)
#   cifar.json, covid10.json             fixed-point exports at scale 2^8 for
#                                        Morpheus (bench_deepprove_cnn)
# Both systems prove the same network: DeepProve runs cnn-cifar-01.onnx and
# covid10.onnx; Morpheus runs the exports of the same files.
set -euo pipefail
WORK=$(realpath "${1:?usage: prepare_cnn.sh <work_dir>}")
HERE=$(cd "$(dirname "$0")" && pwd)
A=$WORK/deep-prove/zkml/assets/scripts
F=$WORK/cnn_fixtures
mkdir -p "$F"
zstd -dqf "$A/CNN/input.json.zst" -o "$F/cifar_input.json"
zstd -dqf "$A/covid/input.json.zst" -o "$F/covid_input.json"
python3 - "$A/covid/cnn-covid.onnx" "$F" <<'PY'
import sys, numpy as np, onnx
from onnx import numpy_helper
src, out = sys.argv[1:3]
m = onnx.load(src); g = m.graph
sig = [n for n in g.node if n.op_type == "Sigmoid"][0]
x = sig.input[0]; g.node.remove(sig)
for n in g.node:
    for i, o in enumerate(n.output):
        if o == x: n.output[i] = g.output[0].name
onnx.checker.check_model(m); onnx.save(m, f"{out}/covid_nosig.onnx")
last = [n for n in g.node if n.op_type == "Gemm"][-1]
for nm in last.input[1:]:
    t = [i for i in g.initializer if i.name == nm][0]
    a = numpy_helper.to_array(t)
    t.CopyFrom(numpy_helper.from_array(np.concatenate([a, np.zeros((9,) + a.shape[1:], a.dtype)], 0), nm))
g.output[0].type.tensor_type.shape.dim[1].dim_value = 10
onnx.checker.check_model(m); onnx.save(m, f"{out}/covid10.onnx")
PY
python3 "$HERE/export_cnn.py" "$A/CNN/cnn-cifar-01.onnx" "$F/cifar_input.json" "$F/cifar.json"
python3 "$HERE/export_cnn.py" "$F/covid10.onnx" "$F/covid_input.json" "$F/covid10.json"
echo "fixtures in $F"

#!/usr/bin/env python3
"""Export DeepProve's ONNX CNNs (Conv/ReLU/MaxPool/Gemm[/Sigmoid]) and the
first input of their input.json as fixed-point integers for
bench_deepprove_cnn (Morpheus). Weights, biases and inputs are rounded at
scale 2^SF; the network is unchanged.

  export_cnn.py <model.onnx> <input.json> <out.json> [SF=8]
"""
import json, sys
import numpy as np
import onnx
from onnx import numpy_helper

onnx_path, inp_path, out_path = sys.argv[1:4]
sf = int(sys.argv[4]) if len(sys.argv) > 4 else 8
m = onnx.load(onnx_path)
init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
q = lambda a: np.rint(np.asarray(a, dtype=np.float64) * (1 << sf)).astype(np.int64)
layers = []
for n in m.graph.node:
    a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
    if n.op_type == "Conv":
        w, b = init[n.input[1]], init[n.input[2]]
        assert a.get("strides", [1, 1]) == [1, 1] and a.get("pads", [0] * 4) == [0] * 4 and a.get("group", 1) == 1
        layers.append({"op": "conv", "co": int(w.shape[0]), "ci": int(w.shape[1]), "k": int(w.shape[2]),
                       "w": q(w).ravel().tolist(), "b": q(b).ravel().tolist()})
    elif n.op_type == "Relu":
        layers.append({"op": "relu"})
    elif n.op_type == "MaxPool":
        assert a["kernel_shape"] == [2, 2] and a["strides"] == [2, 2]
        layers.append({"op": "maxpool"})
    elif n.op_type == "Flatten":
        layers.append({"op": "flatten"})
    elif n.op_type == "Gemm":
        w, b = init[n.input[1]], init[n.input[2]]
        assert a.get("transB", 0) == 1
        layers.append({"op": "fc", "out": int(w.shape[0]), "in": int(w.shape[1]),
                       "w": q(w).ravel().tolist(), "b": q(b).ravel().tolist()})
    elif n.op_type == "Sigmoid":
        layers.append({"op": "sigmoid"})
    else:
        raise SystemExit(f"unsupported op {n.op_type}")
shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim][1:]
x = np.asarray(json.load(open(inp_path))["input_data"][0], dtype=np.float64).ravel()
assert x.size == int(np.prod(shape)), (x.size, shape)
json.dump({"sf": sf, "input_shape": shape, "input": q(x).tolist(), "layers": layers}, open(out_path, "w"))
print(out_path, shape, [l["op"] for l in layers])

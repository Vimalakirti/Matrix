#!/usr/bin/env python3
"""Analyze 3D-UNet ONNX model architecture."""

import onnx
from onnx import numpy_helper, TensorProto
from collections import Counter, OrderedDict

MODEL_PATH = "/scratch/bjchen4/goldilocks/zk-torch-3/3dunet.onnx"

# Dtype mapping
DTYPE_MAP = {
    TensorProto.FLOAT: "float32",
    TensorProto.DOUBLE: "float64",
    TensorProto.FLOAT16: "float16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.INT8: "int8",
    TensorProto.UINT8: "uint8",
    TensorProto.BOOL: "bool",
    TensorProto.STRING: "string",
}

def get_dtype_str(elem_type):
    return DTYPE_MAP.get(elem_type, f"unknown({elem_type})")

def get_shape(type_proto):
    if type_proto.HasField("tensor_type"):
        shape = type_proto.tensor_type.shape
        if shape:
            dims = []
            for d in shape.dim:
                if d.HasField("dim_value"):
                    dims.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append("?")
            return dims
    return None

def get_attr_value(attr):
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
    elif attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode("utf-8") if isinstance(s, bytes) else s for s in attr.strings]
    elif attr.type == onnx.AttributeProto.TENSOR:
        return f"tensor({list(attr.t.dims)})"
    elif attr.type == onnx.AttributeProto.GRAPH:
        return "<graph>"
    else:
        return f"<type={attr.type}>"

def main():
    print("=" * 80)
    print("3D-UNet ONNX Model Analysis")
    print("=" * 80)

    model = onnx.load(MODEL_PATH)
    graph = model.graph

    print(f"\nModel graph name: {graph.name}")
    print(f"IR version: {model.ir_version}")
    print(f"Opset imports:")
    for opset in model.opset_import:
        domain = opset.domain if opset.domain else "ai.onnx"
        print(f"  domain='{domain}', version={opset.version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    if model.doc_string:
        print(f"Doc: {model.doc_string}")

    # Inputs
    print("\n" + "=" * 80)
    print("GRAPH INPUTS")
    print("=" * 80)
    for inp in graph.input:
        shape = get_shape(inp.type)
        dtype = get_dtype_str(inp.type.tensor_type.elem_type) if inp.type.HasField("tensor_type") else "?"
        print(f"  {inp.name}: shape={shape}, dtype={dtype}")

    # Outputs
    print("\n" + "=" * 80)
    print("GRAPH OUTPUTS")
    print("=" * 80)
    for out in graph.output:
        shape = get_shape(out.type)
        dtype = get_dtype_str(out.type.tensor_type.elem_type) if out.type.HasField("tensor_type") else "?"
        print(f"  {out.name}: shape={shape}, dtype={dtype}")

    # Initializers
    print("\n" + "=" * 80)
    print("INITIALIZERS (WEIGHTS)")
    print("=" * 80)
    total_params = 0
    initializer_names = set()
    for init in graph.initializer:
        shape = list(init.dims)
        numel = 1
        for d in shape:
            numel *= d
        total_params += numel
        dtype = get_dtype_str(init.data_type)
        initializer_names.add(init.name)
        print(f"  {init.name}: shape={shape}, numel={numel}, dtype={dtype}")

    print(f"\n  TOTAL PARAMETERS: {total_params:,}")

    # Runtime inputs
    print("\n" + "=" * 80)
    print("RUNTIME INPUTS (non-initializer graph inputs)")
    print("=" * 80)
    for inp in graph.input:
        if inp.name not in initializer_names:
            shape = get_shape(inp.type)
            dtype = get_dtype_str(inp.type.tensor_type.elem_type) if inp.type.HasField("tensor_type") else "?"
            print(f"  {inp.name}: shape={shape}, dtype={dtype}")

    # Build shape map
    shape_map = {}
    for vi in graph.value_info:
        shape_map[vi.name] = get_shape(vi.type)
    for inp in graph.input:
        shape_map[inp.name] = get_shape(inp.type)
    for out in graph.output:
        shape_map[out.name] = get_shape(out.type)

    # All nodes
    print("\n" + "=" * 80)
    print("ALL GRAPH NODES (in order)")
    print("=" * 80)

    op_counter = Counter()

    for i, node in enumerate(graph.node):
        op = node.op_type
        op_counter[op] += 1
        name = node.name if node.name else f"<unnamed_{i}>"

        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = get_attr_value(attr)

        print(f"\n--- Node {i}: {op} ---")
        print(f"  Name: {name}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")

        for inp_name in node.input:
            if inp_name in shape_map:
                print(f"    input '{inp_name}' shape: {shape_map[inp_name]}")
        for out_name in node.output:
            if out_name in shape_map:
                print(f"    output '{out_name}' shape: {shape_map[out_name]}")

        if attrs:
            print(f"  Attributes:")
            for k, v in attrs.items():
                print(f"    {k}: {v}")

    # Summary
    print("\n" + "=" * 80)
    print("OP TYPE SUMMARY")
    print("=" * 80)
    for op, count in sorted(op_counter.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")
    print(f"\n  Total nodes: {sum(op_counter.values())}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Unique op types: {len(op_counter)}")

    # Conv details
    print("\n" + "=" * 80)
    print("CONV LAYER SUMMARY")
    print("=" * 80)
    conv_idx = 0
    for i, node in enumerate(graph.node):
        if node.op_type in ("Conv", "ConvTranspose"):
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = get_attr_value(attr)

            weight_name = node.input[1] if len(node.input) > 1 else "?"
            weight_shape = None
            for init in graph.initializer:
                if init.name == weight_name:
                    weight_shape = list(init.dims)
                    break

            in_shape = shape_map.get(node.input[0], "?")
            out_shape = shape_map.get(node.output[0], "?")

            print(f"  [{conv_idx}] {node.op_type} (node {i}): {node.name}")
            print(f"    weight: {weight_name} shape={weight_shape}")
            print(f"    input shape: {in_shape}")
            print(f"    output shape: {out_shape}")
            print(f"    kernel_shape={attrs.get('kernel_shape', '?')}, "
                  f"strides={attrs.get('strides', '?')}, "
                  f"pads={attrs.get('pads', '?')}, "
                  f"dilations={attrs.get('dilations', '?')}, "
                  f"group={attrs.get('group', '?')}")
            conv_idx += 1

    # Normalization
    print("\n" + "=" * 80)
    print("NORMALIZATION LAYER SUMMARY")
    print("=" * 80)
    norm_ops = {"BatchNormalization", "InstanceNormalization", "GroupNormalization", "LayerNormalization"}
    norm_idx = 0
    for i, node in enumerate(graph.node):
        if node.op_type in norm_ops:
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = get_attr_value(attr)
            in_shape = shape_map.get(node.input[0], "?")
            out_shape = shape_map.get(node.output[0], "?")
            print(f"  [{norm_idx}] {node.op_type} (node {i}): {node.name}")
            print(f"    inputs: {list(node.input)}")
            print(f"    input shape: {in_shape}, output shape: {out_shape}")
            if attrs:
                for k, v in attrs.items():
                    print(f"    {k}: {v}")
            norm_idx += 1
    if norm_idx == 0:
        print("  (none)")

    # Pooling / Upsample / Resize
    print("\n" + "=" * 80)
    print("POOLING / UPSAMPLE / RESIZE SUMMARY")
    print("=" * 80)
    pool_ops = {"MaxPool", "AveragePool", "GlobalAveragePool", "Upsample", "Resize"}
    pool_idx = 0
    for i, node in enumerate(graph.node):
        if node.op_type in pool_ops:
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = get_attr_value(attr)
            in_shape = shape_map.get(node.input[0], "?")
            out_shape = shape_map.get(node.output[0], "?")
            print(f"  [{pool_idx}] {node.op_type} (node {i}): {node.name}")
            print(f"    inputs: {list(node.input)}")
            print(f"    input shape: {in_shape}, output shape: {out_shape}")
            if attrs:
                for k, v in attrs.items():
                    print(f"    {k}: {v}")
            pool_idx += 1
    if pool_idx == 0:
        print("  (none)")

    # Add nodes
    print("\n" + "=" * 80)
    print("ADD (SKIP CONNECTION) SUMMARY")
    print("=" * 80)
    add_idx = 0
    for i, node in enumerate(graph.node):
        if node.op_type == "Add":
            in_shapes = [shape_map.get(n, "?") for n in node.input]
            out_shape = shape_map.get(node.output[0], "?")
            print(f"  [{add_idx}] Add (node {i}): {node.name}")
            print(f"    inputs: {list(node.input)}")
            print(f"    input shapes: {in_shapes}")
            print(f"    output shape: {out_shape}")
            add_idx += 1
    if add_idx == 0:
        print("  (none)")

    # Concat nodes
    print("\n" + "=" * 80)
    print("CONCAT SUMMARY")
    print("=" * 80)
    cat_idx = 0
    for i, node in enumerate(graph.node):
        if node.op_type == "Concat":
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = get_attr_value(attr)
            in_shapes = [shape_map.get(n, "?") for n in node.input]
            out_shape = shape_map.get(node.output[0], "?")
            print(f"  [{cat_idx}] Concat (node {i}): {node.name}")
            print(f"    inputs: {list(node.input)}")
            print(f"    input shapes: {in_shapes}")
            print(f"    output shape: {out_shape}")
            print(f"    axis: {attrs.get('axis', '?')}")
            cat_idx += 1
    if cat_idx == 0:
        print("  (none)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

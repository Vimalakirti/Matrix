#!/usr/bin/env python3
"""Infer and print all intermediate tensor shapes for 3D-UNet."""

import onnx
from onnx import shape_inference, TensorProto

MODEL_PATH = "/scratch/bjchen4/goldilocks/zk-torch-3/3dunet.onnx"

DTYPE_MAP = {
    TensorProto.FLOAT: "float32",
    TensorProto.DOUBLE: "float64",
    TensorProto.FLOAT16: "float16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
}

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
    else:
        return f"<type={attr.type}>"

def main():
    print("Running shape inference...")
    model = onnx.load(MODEL_PATH)
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    # Build shape map from inferred shapes
    shape_map = {}
    for vi in graph.value_info:
        shape_map[vi.name] = get_shape(vi.type)
    for inp in graph.input:
        shape_map[inp.name] = get_shape(inp.type)
    for out in graph.output:
        shape_map[out.name] = get_shape(out.type)

    print("\n" + "=" * 100)
    print("ALL NODES WITH INFERRED SHAPES")
    print("=" * 100)

    for i, node in enumerate(graph.node):
        op = node.op_type
        name = node.name if node.name else f"<unnamed_{i}>"

        # Collect key attributes
        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = get_attr_value(attr)

        in_shapes = []
        for inp_name in node.input:
            s = shape_map.get(inp_name, "?")
            in_shapes.append(f"{inp_name}:{s}")
        out_shapes = []
        for out_name in node.output:
            s = shape_map.get(out_name, "?")
            out_shapes.append(f"{out_name}:{s}")

        # Compact display
        attr_str = ""
        if op in ("Conv", "ConvTranspose"):
            attr_str = f" k={attrs.get('kernel_shape','?')} s={attrs.get('strides','?')} p={attrs.get('pads','?')} g={attrs.get('group','?')}"
        elif op == "Concat":
            attr_str = f" axis={attrs.get('axis','?')}"
        elif op == "InstanceNormalization":
            attr_str = f" eps={attrs.get('epsilon','?')}"

        print(f"[{i:3d}] {op:25s} {name}")
        print(f"      IN:  {', '.join(in_shapes)}")
        print(f"      OUT: {', '.join(out_shapes)}{attr_str}")

    # Print a clean architecture summary with spatial dims
    print("\n" + "=" * 100)
    print("ARCHITECTURE SUMMARY (with spatial dimensions)")
    print("=" * 100)
    
    # Manually trace through the architecture
    for i, node in enumerate(graph.node):
        op = node.op_type
        out_name = node.output[0]
        out_shape = shape_map.get(out_name, "?")
        
        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = get_attr_value(attr)
        
        if op in ("Conv", "ConvTranspose"):
            weight_name = node.input[1] if len(node.input) > 1 else "?"
            # Find weight shape from initializers
            weight_shape = None
            for init in graph.initializer:
                if init.name == weight_name:
                    weight_shape = list(init.dims)
                    break
            k = attrs.get('kernel_shape', '?')
            s = attrs.get('strides', '?')
            p = attrs.get('pads', '?')
            if weight_shape:
                print(f"  Node {i:3d}: {op:16s} W={weight_shape} k={k} s={s} p={p} -> {out_shape}")
            else:
                print(f"  Node {i:3d}: {op:16s} k={k} s={s} p={p} -> {out_shape}")
        elif op == "Concat":
            in_shapes_list = [f"{n}:{shape_map.get(n, '?')}" for n in node.input]
            print(f"  Node {i:3d}: {op:16s} [{', '.join(in_shapes_list)}] axis={attrs.get('axis','?')} -> {out_shape}")
        elif op == "InstanceNormalization":
            print(f"  Node {i:3d}: {op:16s} -> {out_shape}")
        elif op == "Relu":
            print(f"  Node {i:3d}: {op:16s} -> {out_shape}")

    # Final summary of encoder/decoder structure
    print("\n" + "=" * 100)
    print("ENCODER-DECODER FLOW SUMMARY")
    print("=" * 100)
    
    # Compute activation sizes
    total_activations = 0
    max_activation = 0
    for vi in graph.value_info:
        s = get_shape(vi.type)
        if s and all(isinstance(d, int) for d in s):
            numel = 1
            for d in s:
                numel *= d
            total_activations += numel
            if numel > max_activation:
                max_activation = numel
                max_name = vi.name
    
    print(f"\n  Total intermediate activation elements: {total_activations:,}")
    if max_activation > 0:
        print(f"  Largest activation: {max_name} = {max_activation:,} elements")

if __name__ == "__main__":
    main()

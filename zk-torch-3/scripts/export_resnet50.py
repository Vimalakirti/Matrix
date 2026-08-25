#!/usr/bin/env python3
"""
Export ResNet-50 weights from ONNX and preprocessed ImageNet images
for zk-torch-3 accuracy evaluation.

Usage:
    python scripts/export_resnet50.py \
        --onnx-model /path/to/resnet50.onnx \
        --imagenet-val /path/to/imagenet/val \
        --val-labels /path/to/val.txt \
        --output-dir /tmp/resnet50_export \
        --sf-log 10 \
        --num-images 100
"""

import argparse
import os
import struct
import numpy as np
import onnx
from onnx import numpy_helper
from PIL import Image


def next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


GOLDILOCKS_PRIME = (1 << 64) - (1 << 32) + 1


def float_to_field(x, sf_log):
    """Convert float to Goldilocks field element (signed fixed-point)."""
    sf = 1 << sf_log
    val = int(round(x * sf))
    if val >= 0:
        return val % GOLDILOCKS_PRIME
    else:
        neg = (-val) % GOLDILOCKS_PRIME
        return 0 if neg == 0 else GOLDILOCKS_PRIME - neg


def export_conv_weight(w_np, c_in, c_out, kh, kw, sf_log):
    """Convert conv weight [c_out, c_in, kH, kW] to padded little-endian field layout.
    zk-torch-3: idx = kw_i + kh_i * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad
    """
    kh_pad = next_pow2(kh)
    kw_pad = next_pow2(kw)
    c_in_pad = next_pow2(c_in)
    c_out_pad = next_pow2(c_out)
    total = c_out_pad * c_in_pad * kh_pad * kw_pad

    data = np.zeros(total, dtype=np.uint64)
    for d in range(c_out):
        for c in range(c_in):
            for kh_i in range(kh):
                for kw_i in range(kw):
                    idx = kw_i + kh_i * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad
                    data[idx] = float_to_field(w_np[d, c, kh_i, kw_i], sf_log)
    return data


def export_bias(b_np, c_out, sf_log):
    """Convert bias [c_out] to padded field layout."""
    c_out_pad = next_pow2(c_out)
    data = np.zeros(c_out_pad, dtype=np.uint64)
    for i in range(c_out):
        data[i] = float_to_field(b_np[i], sf_log)
    return data


def export_fc_weight(w_np, in_dim, out_dim, sf_log):
    """Convert FC weight to padded field layout (column-major).
    ONNX fc.weight_transposed has shape [in_dim, out_dim].
    zk-torch-3: idx = i + j * in_pad
    """
    in_pad = next_pow2(in_dim)
    out_pad = next_pow2(out_dim)
    total = in_pad * out_pad

    data = np.zeros(total, dtype=np.uint64)
    for j in range(out_dim):
        for i in range(in_dim):
            idx = i + j * in_pad
            data[idx] = float_to_field(w_np[i, j], sf_log)
    return data


def write_tensor(path, data, shape):
    """Write tensor as binary: [ndim: u32] [shape: ndim × u32] [data: n × u64]."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(shape)))
        for s in shape:
            f.write(struct.pack('<I', s))
        f.write(data.astype(np.uint64).tobytes())


def preprocess_image(img_path):
    """Standard ImageNet preprocessing (matches MLPerf) using PIL only."""
    img = Image.open(img_path).convert('RGB')

    # Resize shortest side to 256, preserving aspect ratio
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_w, new_h = int(256 * w / h), 256
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    # Convert to float [0, 1], then normalize
    arr = np.array(img, dtype=np.float32) / 255.0  # [224, 224, 3] HWC
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # Transpose to CHW: [3, 224, 224]
    arr = arr.transpose(2, 0, 1)
    return arr


def export_image(img_np, sf_log):
    """Convert preprocessed image [3, 224, 224] to padded field layout.
    zk-torch-3 CNN convention: w stride=1, h stride=w_pad, c stride=w_pad*h_pad.
    (This matches how pad/conv BasicBlocks index spatial tensors.)
    """
    c, h, w = 3, 224, 224
    c_pad, h_pad, w_pad = next_pow2(c), next_pow2(h), next_pow2(w)
    total = c_pad * h_pad * w_pad

    data = np.zeros(total, dtype=np.uint64)
    for ci in range(c):
        for hi in range(h):
            for wi in range(w):
                idx = wi + hi * w_pad + ci * w_pad * h_pad
                data[idx] = float_to_field(img_np[ci, hi, wi], sf_log)
    return data


def main():
    parser = argparse.ArgumentParser(description='Export ResNet-50 for zk-torch-3')
    parser.add_argument('--onnx-model', type=str,
                        default='/scratch/bjchen4/goldilocks/research/onnx/resnet50.onnx',
                        help='Path to ResNet-50 ONNX model')
    parser.add_argument('--imagenet-val', type=str, default='',
                        help='Path to ImageNet validation directory')
    parser.add_argument('--val-labels', type=str, default='',
                        help='Path to val.txt (image_name label)')
    parser.add_argument('--output-dir', type=str, default='/tmp/resnet50_export',
                        help='Output directory')
    parser.add_argument('--sf-log', type=int, default=10, help='Scale factor log')
    parser.add_argument('--num-images', type=int, default=100, help='Number of images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sf_log = args.sf_log
    print(f"SF_LOG = {sf_log} (scale factor = {1 << sf_log})")

    # 1. Load ONNX model
    print(f"Loading ONNX model: {args.onnx_model}")
    model = onnx.load(args.onnx_model)

    # Build initializer lookup
    inits = {}
    for init in model.graph.initializer:
        inits[init.name] = numpy_helper.to_array(init)

    # 2. Extract Conv layers in graph order (matching zk-torch-3's resnet50_conv_configs)
    conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']
    print(f"Found {len(conv_nodes)} Conv layers")

    conv_weights = []
    conv_biases = []

    for i, node in enumerate(conv_nodes):
        w_name = node.input[1]
        b_name = node.input[2] if len(node.input) > 2 else None

        w_np = inits[w_name]
        c_out, c_in, kh, kw = w_np.shape

        w_data = export_conv_weight(w_np, c_in, c_out, kh, kw, sf_log)
        write_tensor(os.path.join(args.output_dir, f'conv_{i:03d}_weight.bin'), w_data, [c_out, c_in, kh, kw])

        if b_name and b_name in inits:
            b_np = inits[b_name]
            b_data = export_bias(b_np, c_out, sf_log)
            # Shape [c_out, 1, 1] for broadcasting with conv output [c_out, h, w]
            write_tensor(os.path.join(args.output_dir, f'conv_{i:03d}_bias.bin'), b_data, [c_out, 1, 1])
        else:
            b_data = np.zeros(next_pow2(c_out), dtype=np.uint64)
            write_tensor(os.path.join(args.output_dir, f'conv_{i:03d}_bias.bin'), b_data, [c_out, 1, 1])

        conv_weights.append((c_in, c_out, kh, kw))
        if (i + 1) % 10 == 0:
            print(f"  Exported conv {i+1}/{len(conv_nodes)}")

    # 3. Export FC weight and bias
    # Find MatMul node for FC
    fc_w_name = None
    for node in model.graph.node:
        if node.op_type == 'MatMul':
            fc_w_name = node.input[1]
            break

    if fc_w_name and fc_w_name in inits:
        fc_w_np = inits[fc_w_name]  # [2048, 1000] (already transposed in ONNX)
        in_dim, out_dim = fc_w_np.shape
        print(f"FC weight: [{in_dim}, {out_dim}]")
        fc_w_data = export_fc_weight(fc_w_np, in_dim, out_dim, sf_log)
        write_tensor(os.path.join(args.output_dir, 'fc_weight.bin'), fc_w_data, [in_dim, out_dim])
    else:
        print("WARNING: FC weight not found!")

    if 'fc.bias' in inits:
        fc_b_np = inits['fc.bias']
        print(f"FC bias: [{len(fc_b_np)}]")
        fc_b_data = export_bias(fc_b_np, len(fc_b_np), sf_log)
        write_tensor(os.path.join(args.output_dir, 'fc_bias.bin'), fc_b_data, [len(fc_b_np)])

    # 4. Write metadata
    with open(os.path.join(args.output_dir, 'metadata.txt'), 'w') as f:
        f.write(f"sf_log={sf_log}\n")
        f.write(f"num_conv={len(conv_nodes)}\n")
        f.write(f"num_classes=1000\n")
        for i, (c_in, c_out, kh, kw) in enumerate(conv_weights):
            f.write(f"conv_{i}={c_in},{c_out},{kh},{kw}\n")

    # 5. Export ImageNet images (if provided)
    if args.imagenet_val and args.val_labels:
        print(f"Preprocessing {args.num_images} ImageNet images...")

        labels = {}
        with open(args.val_labels) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[parts[0]] = int(parts[1])

        image_files = sorted([f for f in os.listdir(args.imagenet_val) if f.endswith('.JPEG')])
        image_files = image_files[:args.num_images]

        images_dir = os.path.join(args.output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        label_list = []
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(args.imagenet_val, img_name)
            img_np = preprocess_image(img_path)
            img_data = export_image(img_np, sf_log)
            write_tensor(os.path.join(images_dir, f'{i:05d}.bin'), img_data, [3, 224, 224])
            label_list.append(labels.get(img_name, -1))
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(image_files)} images")

        with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
            for i, label in enumerate(label_list):
                f.write(f"{i} {label}\n")

        print(f"  Exported {len(image_files)} images with labels")
    else:
        print("No ImageNet path provided — weights only export.")
        # Write empty labels
        with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
            pass

    print(f"\nDone! Exported to {args.output_dir}")
    print(f"  {len(conv_nodes)} conv weight+bias files")
    print(f"  1 FC weight + 1 FC bias")


if __name__ == '__main__':
    main()

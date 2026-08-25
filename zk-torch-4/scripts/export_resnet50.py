#!/usr/bin/env python3
"""
Export ResNet-50 v1.5 weights (BatchNorm folded into conv) + preprocessed
ImageNet val images for zk-torch-4 accuracy evaluation (MLPerf edge, M1).

Source of truth is torchvision `resnet50` (this IS the v1.5 stride-in-3x3
variant), so BN folding and conv ordering are explicit and match zk-torch-4's
`resnet50_conv_configs()`: stem conv, then per bottleneck block
[conv1(1x1), conv2(3x3), conv3(1x1), downsample(1x1) if present].

Fixed-point encoding MUST match `zk_torch_4::mlperf::quantize`:
    field = int(round(x * 2^sf_log)) lifted into [0, ALMOST_GOLDILOCKS_PRIME).

Usage:
    python scripts/export_resnet50.py \
        --imagenet-val /path/to/imagenet/val \
        --val-labels   /path/to/val.txt \
        --output-dir   /tmp/resnet50_export \
        --sf-log 16 --num-images 100

`val.txt` lines: `<image_filename> <class_index>`. Omit dataset args to export
weights only (metadata + conv/fc tensors, empty labels).
"""

import argparse
import os
import struct
import numpy as np

# Almost-Goldilocks prime (almost-goldilocks-cuda-rs/src/field.rs:12).
ALMOST_GOLDILOCKS_PRIME = 0xFFFFFFFEFFFFFFE1  # 2^64 - 2^32 - 31


def next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def float_to_field(x, sf_log):
    """round(x * 2^sf_log) as a signed fixed-point field element in [0, q)."""
    val = int(round(float(x) * (1 << sf_log)))
    return val % ALMOST_GOLDILOCKS_PRIME if val >= 0 else (ALMOST_GOLDILOCKS_PRIME - ((-val) % ALMOST_GOLDILOCKS_PRIME)) % ALMOST_GOLDILOCKS_PRIME


def write_tensor(path, data_u64, shape):
    """[ndim: u32][shape: ndim x u32][data: n x u64 LE] — matches mlperf::read_tensor."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for s in shape:
            f.write(struct.pack("<I", int(s)))
        f.write(np.asarray(data_u64, dtype="<u8").tobytes())


# ---------------------------------------------------------------------------
# Layout encoders (must match src/dag/resnet.rs + src/bin/resnet.rs gen_*).
# ---------------------------------------------------------------------------

def encode_conv_weight(w, sf_log):
    """[c_out, c_in, kH, kW] -> padded LE: idx = kw + kh*kw_pad + c*kw_pad*kh_pad + d*kw_pad*kh_pad*c_in_pad."""
    c_out, c_in, kh, kw = w.shape
    kwp, khp, cip, cop = next_pow2(kw), next_pow2(kh), next_pow2(c_in), next_pow2(c_out)
    out = np.zeros(cop * cip * khp * kwp, dtype="<u8")
    for d in range(c_out):
        for c in range(c_in):
            for hi in range(kh):
                for wi in range(kw):
                    idx = wi + hi * kwp + c * kwp * khp + d * kwp * khp * cip
                    out[idx] = float_to_field(w[d, c, hi, wi], sf_log)
    return out, [c_out, c_in, kh, kw]


def encode_bias(b, sf_log):
    """[c_out] -> padded [c_out,1,1] (broadcasts against conv output [c_out,h,w])."""
    c_out = b.shape[0]
    out = np.zeros(next_pow2(c_out), dtype="<u8")
    for i in range(c_out):
        out[i] = float_to_field(b[i], sf_log)
    return out, [c_out, 1, 1]


def encode_fc_weight(w_out_in, sf_log):
    """torchvision Linear weight [out,in] -> zk layout [in,out], idx = i + j*in_pad."""
    out_dim, in_dim = w_out_in.shape
    inp, outp = next_pow2(in_dim), next_pow2(out_dim)
    out = np.zeros(inp * outp, dtype="<u8")
    for j in range(out_dim):
        for i in range(in_dim):
            out[i + j * inp] = float_to_field(w_out_in[j, i], sf_log)
    return out, [in_dim, out_dim]


def encode_fc_bias(b, sf_log):
    out_dim = b.shape[0]
    out = np.zeros(next_pow2(out_dim), dtype="<u8")
    for i in range(out_dim):
        out[i] = float_to_field(b[i], sf_log)
    return out, [out_dim]


def encode_image(img_chw, sf_log):
    """[3,224,224] -> padded LE: idx = w + h*w_pad + c*w_pad*h_pad (c_pad = 4)."""
    c, h, w = img_chw.shape
    cp, hp, wp = next_pow2(c), next_pow2(h), next_pow2(w)
    out = np.zeros(cp * hp * wp, dtype="<u8")
    for ci in range(c):
        for hi in range(h):
            for wi in range(w):
                out[wi + hi * wp + ci * wp * hp] = float_to_field(img_chw[ci, hi, wi], sf_log)
    return out, [c, h, w]


# ---------------------------------------------------------------------------
# BN folding: torchvision Bottleneck conv has bias=False, BN follows each conv.
# ---------------------------------------------------------------------------

def fold_bn(conv, bn):
    """Return (w_folded[c_out,c_in,kh,kw], b_folded[c_out]) for conv->bn."""
    w = conv.weight.detach().cpu().numpy().astype(np.float64)
    gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
    beta = bn.bias.detach().cpu().numpy().astype(np.float64)
    mean = bn.running_mean.detach().cpu().numpy().astype(np.float64)
    var = bn.running_var.detach().cpu().numpy().astype(np.float64)
    eps = bn.eps
    scale = gamma / np.sqrt(var + eps)
    w_fold = w * scale[:, None, None, None]
    conv_b = conv.bias.detach().cpu().numpy().astype(np.float64) if conv.bias is not None else 0.0
    b_fold = beta + (conv_b - mean) * scale
    return w_fold, b_fold


def collect_conv_bn_pairs(model):
    """Ordered [(conv, bn), ...] matching resnet50_conv_configs()."""
    pairs = [(model.conv1, model.bn1)]  # stem
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:  # torchvision Bottleneck
            pairs.append((block.conv1, block.bn1))
            pairs.append((block.conv2, block.bn2))
            pairs.append((block.conv3, block.bn3))
            if block.downsample is not None:
                pairs.append((block.downsample[0], block.downsample[1]))
    return pairs


def preprocess_image(img_path):
    """MLPerf ImageNet preprocessing: resize shortest side 256, center-crop 224, normalize."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    nw, nh = (256, round(256 * h / w)) if w < h else (round(256 * w / h), 256)
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - 224) // 2, (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)  # CHW


def main():
    ap = argparse.ArgumentParser(description="Export ResNet-50 v1.5 for zk-torch-4")
    ap.add_argument("--imagenet-val", default="", help="ImageNet val image dir")
    ap.add_argument("--val-labels", default="", help="val.txt: <filename> <class_idx>")
    ap.add_argument("--output-dir", default="/tmp/resnet50_export")
    ap.add_argument("--sf-log", type=int, default=16, help="scale factor log (match config yaml)")
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--weights", default="IMAGENET1K_V2",
                    help="torchvision weights enum (V2 = higher-accuracy v1.5 recipe)")
    args = ap.parse_args()

    import torch
    import torchvision

    os.makedirs(args.output_dir, exist_ok=True)
    sf = args.sf_log
    print(f"SF_LOG={sf} (scale=2^{sf}); prime=0x{ALMOST_GOLDILOCKS_PRIME:016X}")

    print(f"Loading torchvision resnet50(weights={args.weights}) ...")
    model = torchvision.models.resnet50(weights=args.weights)
    model.eval()

    pairs = collect_conv_bn_pairs(model)
    print(f"{len(pairs)} conv+bn pairs (expected 53)")
    configs = []
    for i, (conv, bn) in enumerate(pairs):
        w_fold, b_fold = fold_bn(conv, bn)
        wd, wshape = encode_conv_weight(w_fold, sf)
        bd, bshape = encode_bias(b_fold, sf)
        write_tensor(os.path.join(args.output_dir, f"conv_{i:03d}_weight.bin"), wd, wshape)
        write_tensor(os.path.join(args.output_dir, f"conv_{i:03d}_bias.bin"), bd, bshape)
        c_out, c_in, kh, kw = wshape
        configs.append((c_in, c_out, kh, kw))
        if (i + 1) % 10 == 0:
            print(f"  conv {i+1}/{len(pairs)}")

    fc_w = model.fc.weight.detach().cpu().numpy().astype(np.float64)  # [1000, 2048]
    fc_b = model.fc.bias.detach().cpu().numpy().astype(np.float64)
    fwd, fwshape = encode_fc_weight(fc_w, sf)
    fbd, fbshape = encode_fc_bias(fc_b, sf)
    write_tensor(os.path.join(args.output_dir, "fc_weight.bin"), fwd, fwshape)
    write_tensor(os.path.join(args.output_dir, "fc_bias.bin"), fbd, fbshape)
    print(f"FC weight {fwshape}, bias {fbshape}")

    with open(os.path.join(args.output_dir, "metadata.txt"), "w") as f:
        f.write(f"sf_log={sf}\n")
        f.write(f"num_conv={len(configs)}\n")
        f.write("num_classes=1000\n")
        for i, (c_in, c_out, kh, kw) in enumerate(configs):
            f.write(f"conv_{i}={c_in},{c_out},{kh},{kw}\n")

    # Images
    label_list = []
    if args.imagenet_val and args.val_labels:
        labels = {}
        with open(args.val_labels) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    labels[p[0]] = int(p[1])
        files = sorted(fn for fn in os.listdir(args.imagenet_val)
                       if fn.lower().endswith((".jpeg", ".jpg", ".png")))[:args.num_images]
        img_dir = os.path.join(args.output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        print(f"Preprocessing {len(files)} images ...")
        for i, fn in enumerate(files):
            img = preprocess_image(os.path.join(args.imagenet_val, fn))
            d, shp = encode_image(img, sf)
            write_tensor(os.path.join(img_dir, f"{i:05d}.bin"), d, shp)
            label_list.append(labels.get(fn, -1))
            if (i + 1) % 25 == 0:
                print(f"  img {i+1}/{len(files)}")
    else:
        print("No dataset args — weights-only export.")

    with open(os.path.join(args.output_dir, "labels.txt"), "w") as f:
        for i, lab in enumerate(label_list):
            f.write(f"{i} {lab}\n")

    print(f"\nDone -> {args.output_dir}  ({len(configs)} conv, {len(label_list)} images)")


if __name__ == "__main__":
    main()

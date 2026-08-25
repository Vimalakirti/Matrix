#!/usr/bin/env python3
"""End-to-end evaluation for the paper: full-size models, all three axes.

    # the headline table: every model, all GPUs, every technique on
    python3 run_paper_e2e.py --reps 3

    # the same without deferral, for the amortization comparison
    python3 run_paper_e2e.py --no-deferred --reps 3

    # multi-GPU scaling for one model
    python3 run_paper_e2e.py --only Llama-2 --gpus 1,2,4,8

    # batch / sequence scaling
    python3 run_paper_e2e.py --only GPT-2 --seq 128,256,512,1024

Three axes, matching what the evaluation section needs to claim:

  deferred weight opening   ON by default (`--no-deferred` disables). Runs the
                            streaming binary, which
                            commits the model once and opens its weight claims
                            once per flush of `--proofs` inferences. The
                            reported per-proof cost is the amortized one; the
                            one-time finalize is reported beside it, never
                            folded in.
  multi-GPU scaling         `--gpus 1,2,4,8` sweeps the device pool and prints
                            speedup against the 1-GPU row. The proof is
                            transcript-identical at every device count, so a
                            row with `verified=False` invalidates the sweep
                            rather than just that row.
  batch / sequence > 1      `--seq` and `--batch` override the shape. Defaults
                            are seq 256 for transformers and the matching
                            native input resolution for vision (ResNet/VGG
                            224^2, YOLO 640^2, 3D-UNet 128^3), where BATCH is
                            the scaling axis instead — a CNN's spatial size is
                            part of the workload definition, not a knob. The
                            transformer default is 256 because the forward is
                            quadratic in sequence: GPT-2 at seq 1024 spent 24
                            minutes generating the witness before proving even
                            started. Witness generation is excluded from the
                            reported prover time but not from the wall clock, so
                            the shape has to leave room for the part being
                            measured. Push individual rows back up with --seq.

## What counts as prover time

Excluded, because they are not proving:

  Weight gen    synthetic weights; an artifact of benchmarking
  Compile       DAG construction, done once per model
  Forward       witness generation
  Commit(offline)  model weight commitment — precomputed once per model and
                   reused by every proof, so it is amortized to nothing

Included: online commitment of activations and advice, the backward pass
(graph claim reduction), leaf build, and the opening. `Prove:` from the binary
is that sum; this driver reports it alongside the excluded lines rather than
silently netting them out, so a reader can reconstruct either convention.

## Model coverage

Ten of the eleven target models have builders. PointPainting does not: MLPerf's
PointPainting is DeepLabV3+ semantic segmentation fused into PointPillars, and
only bare PointPillars exists here (see MLPERF_ACCURACY.md B3). It is listed
below as unavailable rather than silently substituted, because reporting
PointPillars under the name PointPainting would misstate the workload.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from harness import Case, require_release_build, run_cases, summarize, write_csv
from models import BENCH, CV, LLAMA, techniques

# ---------------------------------------------------------------------------
# The paper's models, at full size.
#
# Each entry: monolithic binary, streaming binary (None when there is no
# deferred-opening bin), env, config, and the knob whose value `--seq`/`--batch`
# overrides. Lookup-table geometry per model comes from the measured sweep in
# models.py: table_commit_log dominates prover time, proof size and memory, and
# its optimum is per-model (roughly 24 - max_input_n).
# ---------------------------------------------------------------------------
MODELS = [
    # name, mono bin, streaming bin, env, config, shape knob
    ("GPT-2 (12L)", "gpt2", "bench_streaming_gpt2",
     {"NUM_LAYERS": "12", "SEQ_LEN": "256", "MAX_NUM_VARS": "24",
      "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4"}, BENCH, "SEQ_LEN"),

    ("BERT-Large (24L)", "bert", "bench_streaming_bert",
     {"NUM_LAYERS": "24", "SEQ_LEN": "256", "MAX_NUM_VARS": "24",
      "ZK4_TABLE_SIZE_LOG": "10", "ZK4_TABLE_COMMIT_LOG": "4"}, BENCH, "SEQ_LEN"),

    ("Llama-2-7B (32L)", "llama2", "bench_streaming_llama2",
     {"NUM_LAYERS": "32", "SEQ_LEN": "256", "NUM_HEADS": "32", "HEAD_DIM": "128",
      "FFN_DIM": "11008", "VOCAB": "32000", "LOGITS_SHARDS": "32",
      "FFN_SHARDS": "16", "MAX_NUM_VARS": "27",
      "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4"}, LLAMA, "SEQ_LEN"),

    ("Llama-3-8B (32L)", "llama3", "bench_streaming_llama3",
     {"NUM_LAYERS": "32", "SEQ_LEN": "256", "HIDDEN_DIM": "4096",
      "NUM_HEADS": "32", "NUM_KV_HEADS": "8", "HEAD_DIM": "128",
      "FFN_DIM": "14336", "VOCAB_SIZE": "128256",
      # llama3_8b does NOT shard its logits head or FFN -- it takes no shard
      # parameters at all, unlike llama_2_7b (LOGITS_SHARDS 32, FFN_SHARDS 16).
      # With 4x Llama-2's vocab the head is 4096 x 128256, padded to 2^29: one
      # committed edge of 4.3 GB before any proving overhead. MAX_NUM_VARS must
      # cover it, and is set here rather than inherited from the binary default
      # so the cost is visible in the row's env instead of implied.
      # This is the likeliest model to OOM; feasibility.csv is the check.
      "MAX_NUM_VARS": "29",
      "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4"}, LLAMA, "SEQ_LEN"),

    ("GPT-J-6B (28L)", "gptj", "bench_streaming_gptj",
     {"NUM_LAYERS": "28", "SEQ_LEN": "256", "NUM_HEADS": "16", "HEAD_DIM": "256",
      "FFN_DIM": "16384", "VOCAB": "50400", "MAX_NUM_VARS": "27",
      "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4"}, LLAMA, "SEQ_LEN"),

    ("ResNet-50 (224^2)", "resnet", "bench_streaming_resnet",
     {"NUM_LAYERS": "53", "INPUT_SIZE": "224", "MAX_NUM_VARS": "28",
      "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "4"}, CV, "BATCH"),

    # MLPerf has no VGG. The comparison point for this row is VerfCNN, so it
    # runs VerfCNN's configuration (CIFAR-10, 32^2) rather than ImageNet 224^2.
    # Running it at 224^2 would be 49x the pixels and would not compare to
    # anything, so the name carries the config to keep the two from being read
    # as the same workload.
    ("VGG-16 (VerfCNN 32^2)", "vgg", "bench_streaming_vgg",
     {"VGG_VARIANT": "16", "NUM_LAYERS": "13", "INPUT_SIZE": "32",
      "MAX_NUM_VARS": "28",
      "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "8"}, CV, "BATCH"),

    # yolov11n is what dag/yolo.rs builds; the 'l' variant has no builder
    # (MLPERF_ACCURACY.md B1). Input stays at MLPerf's 640^2 -- only the
    # depth/width multipliers differ from the reference, so the resolution is
    # comparable even though the capacity is not.
    ("YOLOv11n (640^2)", "yolo", "bench_streaming_yolo",
     # All 8 stages: stem, C3k/C3k2 blocks, SPPF, the FPN+PAN neck and the
     # detect heads. Stopping at 4 built the stem and two stages only -- a
     # fragment, not the network. Channel widths (3->16->32->64->128->256) are
     # the real v11n ones.
     {"NUM_STAGES": "8", "INPUT_SIZE": "640", "MAX_NUM_VARS": "28",
      "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "2"}, CV, "BATCH"),

    # NUM_LAYERS maps to num_levels. The binary defaults it to 1, which builds
    # ONLY encoder level 0 -- two convolutions, not a U-Net. The full model is
    # 6 encoder levels plus the decoder with skips, ConvTranspose3D and concat.
    # Cost is ~2.7x level 0 alone, not 6x: each level halves spatial in 3D
    # (volume /8) while doubling channels (x2), so per-level work drops 4x and
    # level 0 at full resolution dominates.
    # 64^3, not MLPerf's 128^3 sliding-window patch. At 128^3 the prover
    # exceeds host memory: measured 745 GiB resident after 50 minutes without
    # finishing one volume, and the 8xH200 run was killed by the host OOM
    # killer (exit -9) having reached 129 GiB of GPU allocation. That is a
    # memory wall, not the int-overflow bug fixed in 7e06c1f/6bffffd -- those
    # were real and are fixed, but fixing them exposed this.
    #
    # 64^3 is 8x less volume and completes. It is a DEVIATION from the MLPerf
    # task definition, since the sliding-window patch size is part of that
    # definition, so the row is named for what it actually runs and the
    # deviation travels with the number instead of living only in prose.
    ("3D-UNet (64^3)", "unet3d", "bench_streaming_unet3d",
     {"NUM_LAYERS": "6",
      "INPUT_D": "64", "INPUT_H": "64", "INPUT_W": "64", "MAX_NUM_VARS": "28",
      "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "2"}, CV, "BATCH"),

    ("PointPillars (KITTI)", "pointpillar", "bench_streaming_pointpillar",
     {"NX": "432", "NY": "496", "N_PILLARS": "12000", "MAX_POINTS": "32",
      "NUM_ANCHORS": "2", "NUM_CLASSES": "3", "MAX_NUM_VARS": "28",
      "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "2"}, CV, "BATCH"),

    # Whisper tiny.en at its REAL dimensions, rather than large-v3 with a
    # truncated audio context. Every parameter is the model's own: 4+4 layers,
    # n_state 384, n_head 6, 80 mels, n_audio_ctx 1500 (30s at 50 frames/s),
    # n_text_ctx 448. The previous entry claimed large-v3 while running
    # n_audio_ctx 256 of 1500 -- an exact small model beats a truncated large
    # one, because the truncation is invisible in the row name.
    ("Whisper tiny.en (4+4L)", "whisper", "bench_streaming_whisper",
     {"NUM_ENC_LAYERS": "4", "NUM_DEC_LAYERS": "4", "N_MELS": "80",
      "N_AUDIO_CTX": "1500", "N_TEXT_CTX": "448", "N_STATE": "384",
      "N_HEAD": "6", "MAX_NUM_VARS": "28",
      "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4"}, LLAMA, "N_TEXT_CTX"),
]

# Named here so the gap is visible in the table rather than only in prose.
UNAVAILABLE = {
    "PointPainting": (
        "no builder. MLPerf PointPainting is DeepLabV3+ (ResNet-50 backbone, "
        "output-stride 16) semantic segmentation fused into PointPillars; only "
        "bare PointPillars exists (MLPERF_ACCURACY.md B3). Reporting "
        "PointPillars under this name would misstate the workload."
    ),
}


# Models whose builder proves a batch as ONE graph (folded [b_pad*c_pad, H, W],
# batch bound rather than summed). Everything else still REPLICATES the subgraph
# per image, which multiplies commitment leaves and costs superlinearly -- a
# batch sweep over those measures the absence of the technique, not the
# technique, so the driver labels them instead of quietly reporting the number.
#
# The rest are blocked on channel concat: it joins along the leading axis, which
# under folding is the same axis the batch occupies (see the known-gap entry in
# batched_ops_match_per_image_forward).
FOLDED_BATCH = {"ResNet-50 (224^2)", "VGG-16 (VerfCNN 32^2)"}

# --------------------------------------------------------------------------
# Autoregressive generation rows.
#
# The MODELS rows above prove ONE masked forward pass over SEQ_LEN tokens: a
# prefill. That is not what a generation costs, and it omits the embedding, the
# LM head and the argmax that binds each sampled token to the logits it came
# from. These rows use the one-shot AR binaries, where each streamed proof is a
# full T-token generation with all of those included, so "per token" means per
# GENERATED token and is comparable to systems that report autoregressive
# decoding.
#
# VOCAB_SHARDS is DERIVED, not tabulated. argmax range-checks diffs[seq, vocab]
# densely, and its fold-tree leaf is one dense 2^arity eq table with
#   arity = log2(seq_pad) + log2(vocab_shard_pad) + table_commit_log.
# Un-sharded full vocab reaches arity ~30, a 16 GB table that OOMs on any GPU.
# Deriving the count keeps arity at the target as seq or vocab change, instead
# of a hardcoded table that silently goes stale.
AR_TARGET_ARITY = 24


def vocab_shards(vocab: int, seq: int, tcl: int = 4, target: int = AR_TARGET_ARITY) -> int:
    """Smallest power-of-two shard count keeping the argmax leaf at <= target."""
    seq_bits = max(1, seq - 1).bit_length()
    budget = target - seq_bits - tcl          # bits left for the vocab shard
    if budget < 1:
        return 1 << 12                        # degenerate; caller will see it
    shards = 1
    while shards < (1 << 16):
        per = -(-vocab // shards)             # ceil
        if max(1, per - 1).bit_length() <= budget:
            return shards
        shards <<= 1
    return shards


# (name, binary, env, config).  All four are decoder-only; BERT is an encoder
# with no generation to amortize over, and Whisper's AR decoder has no
# streaming one-shot binary, so neither appears here.
AR_MODELS = [
    ("GPT-2 AR", "bench_streaming_oneshot_gpt2",
     {"NUM_LAYERS": "12", "VOCAB_SIZE": "50257",
      "MAX_NUM_VARS": "24", "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4"},
     "bench_config.yaml"),
    ("GPT-J-6B AR", "bench_streaming_oneshot_gptj",
     {"NUM_LAYERS": "28", "NUM_HEADS": "16", "HEAD_DIM": "256", "FFN_DIM": "16384",
      "VOCAB_SIZE": "50400",
      "MAX_NUM_VARS": "27", "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4"},
     "llama2_config.yaml"),
    ("Llama-2-7B AR", "bench_streaming_oneshot_llama2",
     {"NUM_LAYERS": "32", "NUM_HEADS": "32", "HEAD_DIM": "128", "FFN_DIM": "11008",
      "VOCAB_SIZE": "32000", "FFN_SHARDS": "16",
      "MAX_NUM_VARS": "27", "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4"},
     "llama2_config.yaml"),
    ("Llama-3-8B AR", "bench_streaming_oneshot_llama3",
     {"NUM_LAYERS": "32", "NUM_HEADS": "32", "NUM_KV_HEADS": "8", "HEAD_DIM": "128",
      "FFN_DIM": "14336", "VOCAB_SIZE": "128256",
      "MAX_NUM_VARS": "29", "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4"},
     "llama2_config.yaml"),
]


# Rows that run a proxy for the exact v6.0 reference. Printed beside the table
# so the deviation travels with the numbers instead of living only in prose.
PROXIES = {
    "YOLOv11n (640^2)": "reference is YOLOv11l. Two differences, not one: "
                        "smaller depth/width multipliers, AND the C2PSA "
                        "attention block (m.10) is skipped -- dag/yolo.rs:374 "
                        "treats m9_out as m10_out. Input is 640^2 and all 8 "
                        "stages build, including SPPF, the FPN/PAN neck and "
                        "the detect heads (MLPERF_ACCURACY.md B1)",
    "VGG-16 (VerfCNN 32^2)": "not an MLPerf model; VerfCNN's CIFAR-10 32^2 "
                             "configuration, for comparison against VerfCNN",
    "3D-UNet (64^3)": "MLPerf's reference uses a 128^3 sliding-window patch; "
                      "at that size the prover exceeds host memory (745 GiB "
                      "resident after 50 min without completing one volume). "
                      "This is 64^3, which completes and verifies",
    "Whisper tiny.en (4+4L)": "MLPerf's reference is large-v3; this is tiny.en, "
                              "the model the builder implements, at its full "
                              "dimensions (MLPERF_ACCURACY.md B2)",
}


def visible_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=15).stdout
        return sum(1 for l in out.splitlines() if l.startswith("GPU "))
    except Exception:
        return 1


def build_ar_cases(a, gpu_counts, shapes):
    """One Case per (AR model, gpu count, generation length).

    `shapes` here are GENERATION lengths: the number of tokens each streamed
    proof produces autoregressively, not a prefill width.
    """
    cases = []
    have = visible_gpus()
    ar_prompt = int(getattr(a, "ar_prompt", 16))
    for name, binary, env, config in AR_MODELS:
        if a.only and a.only.lower() not in name.lower():
            continue
        if any(t.strip() and t.strip().lower() in name.lower()
               for t in a.skip.split(",")):
            print(f"skip: {name}")
            continue
        for gpus in gpu_counts:
            if gpus > have:
                print(f"note: {gpus} GPUs requested, {have} visible; dropping")
                continue
            for gen in shapes:
                # `shapes` are GENERATED token counts. The proof covers the
                # prompt as well, so the graph is built at prompt + gen
                # positions, and per-token cost divides by gen alone: the
                # prompt is context the request supplied, not output the
                # prover produced. Reporting prove/(prompt+gen) would flatter
                # the number by counting the prompt as though it were output.
                gen = int(gen) if gen is not None else 16
                total = ar_prompt + gen
                e = dict(env)
                e.update(techniques(gpus, a.forward))
                e["SEQ_LEN"] = str(total)
                e["N_PROOFS"] = str(a.proofs)
                # Derived from TOTAL positions, since that is what argmax
                # range-checks -- see vocab_shards().
                tcl = int(e.get("ZK4_TABLE_COMMIT_LOG", 4))
                e["VOCAB_SHARDS"] = str(vocab_shards(int(e["VOCAB_SIZE"]), total, tcl))
                e.update(dict(kv.split("=", 1) for kv in a.env if "=" in kv))
                layers = int(e.get("NUM_LAYERS", 1))
                if int(e.get("NUM_PARTITIONS", 1)) > layers:
                    e["NUM_PARTITIONS"] = str(max(1, layers))
                label = name
                if len(gpu_counts) > 1:
                    label += f" / {gpus}gpu"
                label += f" / P{ar_prompt} G{gen}"
                cases.append(Case(name=label, binary=binary, env=e, config=config,
                                  group=name, timeout_s=a.timeout))
    return cases


def build_cases(a, gpu_counts, shapes):
    """One Case per (model, gpu count, shape)."""
    if getattr(a, "ar", False):
        return build_ar_cases(a, gpu_counts, shapes)
    cases = []
    have = visible_gpus()
    for name, mono, stream, env, config, knob in MODELS:
        if a.only and a.only.lower() not in name.lower():
            continue
        if any(t.strip() and t.strip().lower() in name.lower()
               for t in a.skip.split(",")):
            print(f"skip: {name}")
            continue
        binary = stream if a.deferred else mono
        if binary is None:
            print(f"note: {name} has no streaming binary; "
                  f"running monolithic with ZK4_DEFER_CONSTANTS instead")
            binary = mono
        for gpus in gpu_counts:
            if gpus > have:
                print(f"note: {gpus} GPUs requested, {have} visible; dropping")
                continue
            for shape in shapes:
                e = dict(env)
                e.update(techniques(gpus, a.forward))
                # A model that has no folded-batch builder replicates the
                # subgraph per image, so batch>1 measures the absence of the
                # technique. Pin those to 1 rather than emit rows that read as
                # a batch result.
                if knob == "BATCH" and name not in FOLDED_BATCH:
                    if shape not in (None, 1):
                        continue
                    e[knob] = "1"
                elif shape is not None:
                    e[knob] = str(shape)
                # Record the knob even when it is left at its default, so every
                # row states the shape it ran at instead of only swept rows.
                e.setdefault(knob, "1")
                if a.deferred:
                    # The streaming bin defers weight opening across N proofs.
                    # Where there is no streaming bin, ZK4_DEFER_CONSTANTS keeps
                    # weight edges out of the leaf set, which is the same
                    # technique for a single proof (its finalize is then owed).
                    if stream is not None:
                        e["N_PROOFS"] = str(a.proofs)
                    else:
                        e["ZK4_DEFER_CONSTANTS"] = "1"
                e.update(dict(kv.split("=", 1) for kv in a.env if "=" in kv))
                # NUM_PARTITIONS cuts the backward pass at layer boundaries, so
                # it cannot exceed the layer count: `Dag::partition` asserts
                # `num_partitions - 1 <= layer_boundaries`. A 2-layer model on 4
                # GPUs panics rather than degrading, which is easy to hit when
                # shrinking a model to smoke-test the sweep.
                layers = next((int(e[k]) for k in
                               ("NUM_LAYERS", "NUM_ENC_LAYERS", "NUM_STAGES")
                               if k in e and e[k].isdigit()), None)
                if layers is not None and int(e.get("NUM_PARTITIONS", 1)) > layers:
                    e["NUM_PARTITIONS"] = str(max(1, layers))
                label = name
                if len(gpu_counts) > 1:
                    label += f" / {gpus}gpu"
                if len(shapes) > 1 and shape is not None:
                    label += f" / {knob.lower()}={shape}"
                cases.append(Case(name=label, binary=binary, env=e, config=config,
                                  group=name, timeout_s=a.timeout))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    # Deferred weight opening is ON by default. It is one of the paper's claimed
    # techniques, and a run without it silently reports an upper bound: the
    # weight edges stay in the leaf set and their claims are opened once per
    # proof instead of once per flush. Defaulting it off invites exactly that
    # mistake, so turning it off now takes an explicit flag.
    ap.add_argument("--no-deferred", dest="deferred", action="store_false",
                    help="DISABLE deferred fixed-weight opening (baseline for "
                         "the amortization comparison; not the headline config)")
    ap.set_defaults(deferred=True)
    ap.add_argument("--proofs", type=int, default=8,
                    help="inferences per flush when --deferred")
    ap.add_argument("--gpus", default="0",
                    help="comma-separated device counts; 0 = all visible")
    ap.add_argument("--seq", default="",
                    help="comma-separated sequence lengths (transformers)")
    ap.add_argument("--batch", default="",
                    help="comma-separated batch sizes (vision)")
    ap.add_argument("--forward", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--only", default="", help="substring filter on model name")
    ap.add_argument("--ar-prompt", dest="ar_prompt", type=int, default=16,
                    help="prompt tokens preceding the generated ones; the proof "
                         "covers prompt+generated, per-token divides by generated")
    ap.add_argument("--ar", action="store_true",
                    help="prove autoregressive GENERATION (AR_MODELS) instead of "
                         "a single masked forward pass; --seq gives generation lengths")
    ap.add_argument("--skip", default="",
                    help="comma-separated substrings to EXCLUDE. Use when a "
                         "stage does not need every model -- e.g. the "
                         "no-deferral baseline makes its point with VGG and "
                         "ResNet and does not need YOLO at 640^2 or 3D-UNet "
                         "at 128^3, which are the expensive rows.")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="extra env override, repeatable")
    ap.add_argument("--out", default="out/paper_e2e.csv")
    a = ap.parse_args()

    gpu_counts = [int(x) for x in a.gpus.split(",") if x.strip()]
    gpu_counts = [g if g else visible_gpus() for g in gpu_counts]
    shape_src = a.seq or a.batch
    shapes = [int(x) for x in shape_src.split(",") if x.strip()] or [None]

    cases = build_cases(a, gpu_counts, shapes)
    if not cases:
        print("no cases selected")
        return 1

    for name, why in UNAVAILABLE.items():
        print(f"UNAVAILABLE  {name}: {why}")
    for name, why in PROXIES.items():
        print(f"PROXY        {name}: {why}")
    if len(shapes) > 1 and any(k == "BATCH" for _, _, _, _, _, k in MODELS
                               if a.only.lower() in _.lower() or not a.only):
        for name, _, _, _, _, knob in MODELS:
            if knob != "BATCH" or name in FOLDED_BATCH:
                continue
            if a.only and a.only.lower() not in name.lower():
                continue
            print(f"REPLICATED   {name}: batch>1 replicates the subgraph per "
                  f"image (no folded-batch builder). Cost grows superlinearly; "
                  f"this is the baseline, not the technique.")

    require_release_build(sorted({c.binary for c in cases}))
    print(f"\n{len(cases)} case(s), reps={a.reps}, "
          f"deferred={a.deferred}{f' (N={a.proofs})' if a.deferred else ''}\n")
    runs = run_cases(cases, reps=a.reps)
    write_csv(runs, a.out)

    rows = summarize(runs)
    print("\n--- prover time excludes weight gen, compile, forward (witness "
          "generation) and offline weight commitment ---")

    def dname(r):
        """Label minus the shape suffix, which is now its own column.

        The stored name keeps the suffix: summarize() keys on (group, name),
        so two sweep rows that differ only by shape would collapse into one if
        the label were shortened at the source. Only the display drops it.
        The GPU-count suffix ("4gpu") has no "=" and is kept.
        """
        return " / ".join(p for p in r["name"].split(" / ") if "=" not in p)

    def num(r, k, scale=1e-3, w=10):
        v = r.get(k, "")
        return f"{float(v) * scale:>{w}.2f}" if v not in ("", None) else f"{'-':>{w}}"

    if a.deferred and any(r.get("stream_perproof_ms") for r in rows):
        # Streaming binaries report an amortized per-proof cost and a one-time
        # finalize. The finalize is shown separately and never folded into the
        # per-proof number: it is sound only once it runs, so quoting a
        # per-proof cost that hides it would overstate the amortization.
        # Verifier cost is two numbers, not one: a per-unit check plus the
        # one-time finalize check that is what makes the streamed batch sound.
        # Quoting only the per-unit half would understate verification.
        # per-image is the headline for a CNN batch: one proof covers `batch`
        # images, so per-proof alone makes batching look like a regression.
        print(f"{'model':<26} {'shape':>10} {'per-proof':>10} {'per-image':>10} "
              f"{'fin/N':>7} {'finalize':>9} {'verify/u':>9} {'fin-ver':>8} "
              f"{'proof/u':>9} {'proof/fin':>10} {'peak':>8}  ver")
        for r in rows:
            peak = (f"{float(r['peak_mem_mib']) / 1024:>8.1f}G"
                    if r.get("peak_mem_mib") else f"{'-':>9}")
            proof = (f"{float(r['proof_bytes']) / 2**20:>9.1f}M"
                     if r.get("proof_bytes") else f"{'-':>10}")
            finp = (f"{float(r['stream_proof_fin_bytes']) / 2**20:>9.1f}M"
                    if r.get("stream_proof_fin_bytes") else f"{'-':>10}")
            # Divide by the batch when the shape axis IS the batch; a
            # sequence-length row already reports one inference per proof.
            sh = r.get("shape", "")
            imgs = 1
            if sh.startswith("batch=") and sh[6:].isdigit():
                imgs = max(1, int(sh[6:]))
            pp = r.get("stream_perproof_ms")
            per_img = (f"{float(pp) / 1e3 / imgs:>10.2f}" if pp else f"{'-':>10}")
            print(f"{dname(r):<26} {r.get('shape', ''):>10} "
                  f"{num(r, 'stream_perproof_ms')} {per_img} "
                  f"{num(r, 'stream_finalize_n_ms', 1e-3, 7)} "
                  f"{num(r, 'stream_finalize_ms', 1e-3, 9)} "
                  f"{num(r, 'stream_verify_ms', 1e-3, 9)} "
                  f"{num(r, 'stream_finalize_verify_ms', 1e-3, 8)} "
                  f"{proof} {finp} {peak}  {r.get('verified', '')}")
        print(f"\n  seconds, over N={a.proofs} inferences against one committed "
              f"model. verify/u is the per-inference check; fin-ver is the "
              f"one-time\n  finalize check that makes the batch sound. Total "
              f"verifier cost is N*verify/u + fin-ver.")
    else:
        print(f"{'model':<30} {'shape':>10} {'commit_on':>10} {'prove':>10} "
              f"{'verify':>9} {'proof':>10} {'peak':>9}  ver")
        for r in rows:
            proof = (f"{float(r['proof_bytes']) / 2**20:>9.1f}M"
                     if r.get("proof_bytes") else f"{'-':>10}")
            peak = (f"{float(r['peak_mem_mib']) / 1024:>8.1f}G"
                    if r.get("peak_mem_mib") else f"{'-':>9}")
            print(f"{dname(r):<30} {r.get('shape', ''):>10} "
                  f"{num(r, 'commit_online_ms')} "
                  f"{num(r, 'prove_ms')} {num(r, 'verify_ms', 1e-3, 9)} "
                  f"{proof} {peak}  {r.get('verified', '')}")

    # --- cost breakdown: where prover time and proof bytes actually go ---
    # Same three buckets on both sides, so they are directly comparable. The
    # proof split is exact (DagProof separates the pieces); the time split is
    # from ZK4_TIMING and covers the phases, not the whole wall clock, so it
    # is shown as measured rather than normalised to 100%.
    if any(r.get("proof_pcs_mb") or r.get("t_fold_tree_ms") for r in rows):
        print(f"\n{'model':<26} | {'PROOF: sc-nolk':>14} {'sc-lookup':>10} {'PCS':>10}"
              f" | {'TIME: sc-node':>14} {'reducer':>9} {'lookup':>9} {'PCS':>9}"
              f" | {'defer-merge':>12}")
        for r in rows:
            def mb(k):
                v = r.get(k, "")
                return f"{float(v):>9.2f}M" if v not in ("", None) else f"{'-':>10}"
            def ms(k, w=9):
                v = r.get(k, "")
                return f"{float(v) / 1e3:>{w-1}.2f}s" if v not in ("", None) else f"{'-':>{w}}"

            def ms_sum(keys, w=9):
                """Sum several phase timers. The lookup phase is range AND
                two_pow; reporting only `t_lookup_range` understated it by 28%
                on llama2 8L/seq64 (two_pow=3.62s of a 16.56s phase). two_pow
                was already parsed and simply never printed."""
                vals = [float(r[k]) for k in keys if r.get(k) not in ("", None)]
                if not vals:
                    return f"{'-':>{w}}"
                return f"{sum(vals) / 1e3:>{w-1}.2f}s"
            dm = (f"{float(r['t_opening_reducer']) / 1e3:>7.2f}s"
                  f"/{int(float(r.get('n_opening_edges') or 0)):>3}e"
                  if r.get("t_opening_reducer") else f"{'-':>12}")
            print(f"{dname(r):<26} | {mb('proof_sc_nonlookup_mb'):>14} "
                  f"{mb('proof_sc_lookup_mb'):>10} {mb('proof_pcs_mb'):>10} | "
                  f"{ms('t_node_prove_ms', 14):>14} {ms('t_reducer_ms')} "
                  f"{ms_sum(('t_lookup_range', 't_lookup_two_pow'))} "
                  f"{ms('t_fold_tree_ms')} | {dm:>12}")
        print("  sc-nolk = node + reducer sumchecks; PCS = fold tree.")
        print("  lookup = range + two_pow (both phases; the CSV keeps them separate).")
        print("  defer-merge = the sumcheck that folds repeated weight claims on one")
        print("  edge into a single opening, with the edge count it covered.")
        print("  TIME columns need ZK4_TIMING and the partitioned path (NUM_PARTITIONS>1);")
        print("  a '-' means the phase line was absent, not that the phase was free.")

    if len(gpu_counts) > 1:
        print("\nscaling (prove, vs the smallest device count):")
        for name, _, _, _, _, _ in MODELS:
            key = "stream_perproof_ms" if a.deferred else "prove_ms"
            mine = [r for r in rows if r["group"] == name and r.get(key)]
            if len(mine) < 2:
                continue
            base = float(mine[0][key])
            for r in mine:
                print(f"  {dname(r):<30} {r.get('shape', ''):>10} "
                      f"{float(r[key]) / 1e3:>8.1f}s "
                      f"{base / float(r[key]):>6.2f}x"
                      + ("" if r.get("verified") else "   [NOT VERIFIED]"))

    bad = [r for r in rows if not r.get("verified")]
    if bad:
        print(f"\nWARNING: {len(bad)} row(s) did not verify; nothing else in "
              f"those rows means anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

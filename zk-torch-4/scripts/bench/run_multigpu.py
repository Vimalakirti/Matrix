#!/usr/bin/env python3
"""Multi-GPU scaling sweep.  Fills Section 9.4 and the intro's speedup number.

    python3 run_multigpu.py --model gpt2,whisper --gpus 1,2,4

Section 7 currently promises scaling "up to eight GPUs, including the
encoder-decoder case". This node has FOUR A100s, so either run on an 8-GPU
node or soften that sentence to the count actually measured. Counts above the
visible device count are dropped with a note rather than silently wrapped
(NUM_PARTITIONS does wrap modulo the pool, so an 8-partition run on 4 cards is
2 partitions per GPU, not 8 GPUs).

Two knobs move together, matching the two levels of parallelism in Section 7.2:

  ZK4_GPU_DEVICES=0,..,k-1   the device pool the terminal opening shards over
                             (level 1; default is every visible device, so it
                             MUST be pinned or the 1-GPU row is not 1 GPU)
  NUM_PARTITIONS=k           subgraphs the backward pass is split into
                             (level 2; default 1)

The proof is transcript-identical across device counts, so `verified` must be
true in every row and prove time is the only thing that moves.
"""

from __future__ import annotations

import argparse
import subprocess

from harness import Case, require_release_build, run_cases, write_csv
from models import BENCH, CV, LLAMA

# model key -> (binary, env, config). Sizes big enough that partitioning has
# something to divide; a 1-layer model has no useful cut.
MODELS = {
    "gpt2":    ("gpt2",   {"NUM_LAYERS": "12", "SEQ_LEN": "64",
                           "MAX_NUM_VARS": "22"},                      BENCH),
    "llama2":  ("llama2", {"NUM_LAYERS": "8", "SEQ_LEN": "64",
                           "MAX_NUM_VARS": "27"},                      LLAMA),
    "bert":    ("bert",   {"NUM_LAYERS": "12", "SEQ_LEN": "64",
                           "MAX_NUM_VARS": "22"},                      BENCH),
    "whisper": ("whisper", {"NUM_ENC_LAYERS": "2", "NUM_DEC_LAYERS": "2",
                            "N_MELS": "16", "N_AUDIO_CTX": "64",
                            "N_TEXT_CTX": "16"},                       LLAMA),
    "resnet":  ("resnet", {"NUM_LAYERS": "53", "INPUT_SIZE": "32"},    CV),
}


def visible_gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=15).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2",
                    help=f"one or more of {','.join(MODELS)} (comma separated)")
    ap.add_argument("--gpus", default="1,2,4",
                    help="device counts to sweep")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--partitions-equal-gpus", action="store_true", default=True,
                    help="set NUM_PARTITIONS to the device count (default)")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="override a model knob, repeatable (e.g. --env NUM_LAYERS=1); "
                         "useful for smoke-testing this script at a tiny size")
    ap.add_argument("--out", default="out/multigpu.csv")
    a = ap.parse_args()
    overrides = dict(kv.split("=", 1) for kv in a.env if "=" in kv)

    have = visible_gpu_count()
    counts = [int(x) for x in a.gpus.split(",") if x.strip()]
    if have:
        too_big = [k for k in counts if k > have]
        if too_big:
            print(f"note: {have} GPU(s) visible; dropping {too_big}")
            counts = [k for k in counts if k <= have]
    if not counts:
        return print("no runnable GPU counts") or 1

    keys = [m.strip() for m in a.model.split(",") if m.strip()]
    unknown = [m for m in keys if m not in MODELS]
    if unknown:
        return print(f"unknown model(s) {unknown}; known: {list(MODELS)}") or 1

    cases = []
    for key in keys:
        binary, env, config = MODELS[key]
        for k in counts:
            e = dict(env)
            # Only ZK4_GPU_DEVICES: combining it with CUDA_VISIBLE_DEVICES
            # would make its indices refer to the remapped space.
            e["ZK4_GPU_DEVICES"] = ",".join(str(i) for i in range(k))
            e["ZK4_TIMING"] = "1"   # emits the scheduler line -> fold_gpus

            if a.partitions_equal_gpus:
                e["NUM_PARTITIONS"] = str(k)
            e.update(overrides)
            cases.append(Case(name=f"{key} / {k} GPU" + ("s" if k > 1 else ""),
                              binary=binary, env=e, config=config,
                              group=key, timeout_s=a.timeout))

    require_release_build([c.binary for c in cases])
    print(f"models={keys}  gpu counts={counts}  reps={a.reps}\n")
    runs = run_cases(cases, reps=a.reps)
    write_csv(runs, a.out)

    # Speedup relative to each model's 1-GPU row, which is the intro number.
    from harness import summarize
    rows = summarize(runs)
    print("\nspeedup vs 1 GPU (prove):")
    for key in keys:
        mine = [r for r in rows if r["group"] == key and r["prove_ms"] != ""]
        base = next((r for r in mine if r["name"].endswith("1 GPU")), None)
        if not base:
            print(f"  {key}: no 1-GPU baseline")
            continue
        for r in mine:
            s = float(base["prove_ms"]) / float(r["prove_ms"])
            want = r["name"].split("/")[-1].strip().split()[0]
            got = r.get("fold_gpus", "")
            mismatch = ("" if got in ("", None) or str(int(float(got))) == want
                        else f"   [pool used {int(float(got))} GPU(s), not {want}]")
            print(f"  {r['name']:<24} {float(r['prove_ms'])/1e3:8.2f}s   {s:5.2f}x"
                  + ("" if r["verified"] else "   [NOT VERIFIED]") + mismatch)

    unver = [r for r in rows if r["prove_ms"] != "" and not r["verified"]]
    if unver:
        print("\nWARNING: the proof must be identical at every device count. "
              "An unverified row means the multi-GPU path changed the "
              "transcript, which contradicts Section 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Per-model comparison of the packed PCS against the fold tree.

Runs every model with all amortization on and reports both openings measured on
the same leaf set in the same process, so the two numbers are not separated by
run-to-run variance.

What "all optimizations on" means here:

  ZK4_DEFER_CONSTANTS=1  deferred fixed-weight opening. Weight edges leave the
                         leaf set entirely, which is also what makes the
                         reported prover time exclude weight commitment — that
                         is precomputed once per model, not per proof.
  ZK4_PCS=packed         packed PCS shadow run alongside the fold tree.
  seq/batch > 1          the paper-profile shapes, not the 1-token smoke ones.
  4 GPUs                 ZK4_GPU_DEVICES set to the visible pool.

The fold tree still produces the shipped proof; the packed path proves the same
leaf set and self-verifies, so a row is only meaningful when `packed_ok` is true.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "target" / "release"

BENCH = "bench_config.yaml"
LLAMA = "llama2_config.yaml"
CV = "cv_config.yaml"

# name, binary, env, config — the paper-profile shapes.
MODELS = [
    ("GPT-2 12L/seq64",   "gpt2",   {"NUM_LAYERS": "12", "SEQ_LEN": "64",
                                     "MAX_NUM_VARS": "22"}, BENCH),
    ("GPT-2 12L/seq256",  "gpt2",   {"NUM_LAYERS": "12", "SEQ_LEN": "256",
                                     "MAX_NUM_VARS": "24"}, BENCH),
    ("BERT 12L/seq64",    "bert",   {"NUM_LAYERS": "12", "SEQ_LEN": "64",
                                     "MAX_NUM_VARS": "22"}, BENCH),
    ("Llama-2 8L/seq64",  "llama2", {"NUM_LAYERS": "8", "SEQ_LEN": "64",
                                     "MAX_NUM_VARS": "27"}, LLAMA),
    ("Llama-3 8L/seq64",  "llama3", {"NUM_LAYERS": "8", "SEQ_LEN": "64"}, LLAMA),
    ("GPT-J 8L/seq64",    "gptj",   {"NUM_LAYERS": "8", "SEQ_LEN": "64",
                                     "MAX_NUM_VARS": "27"}, LLAMA),
    ("ResNet-50 53conv",  "resnet", {"NUM_LAYERS": "53", "INPUT_SIZE": "32"}, CV),
    ("VGG-16 13conv",     "vgg",    {"VGG_VARIANT": "16", "NUM_LAYERS": "13",
                                     "INPUT_SIZE": "64"}, CV),
    ("YOLO 4stage",       "yolo",   {"NUM_STAGES": "4", "INPUT_SIZE": "64"}, CV),
    ("3D-UNet 16^3",      "unet3d", {"INPUT_D": "16", "INPUT_H": "16",
                                     "INPUT_W": "16"}, CV),
    ("PointPillars",      "pointpillar", {}, CV),
    ("Whisper 1+1L",      "whisper", {"NUM_ENC_LAYERS": "1", "NUM_DEC_LAYERS": "1",
                                      "N_MELS": "16", "N_AUDIO_CTX": "64",
                                      "N_TEXT_CTX": "16"}, BENCH),
    ("Whisper 2+2L",      "whisper", {"NUM_ENC_LAYERS": "2", "NUM_DEC_LAYERS": "2",
                                      "N_MELS": "16", "N_AUDIO_CTX": "64",
                                      "N_TEXT_CTX": "16"}, BENCH),
]

PAT = {
    "fold_tree_s":  re.compile(r"\[prove\] fold tree:\s*([0-9.]+)([a-zµ]*)"),
    "prove_s":      re.compile(r"^\s*Prove:\s*([0-9.]+)([a-zµ]*)", re.M),
    "commit_off_s": re.compile(r"Commit \(offline[^)]*\):\s*([0-9.]+)([a-zµ]*)"),
    "commit_on_s":  re.compile(r"Commit \(online[^)]*\):\s*([0-9.]+)([a-zµ]*)"),
    "leaves":       re.compile(r"leaf build total \((\d+) leaves,\s*(\d+) const edges deferred\)"),
}
PACKED = re.compile(
    r"\[packed_pcs/interleaved\]\s+pack ([0-9.]+)s\s+commit ([0-9.]+)s\s+link ([0-9.]+)s"
)
PACKED_TOTAL = re.compile(r"\[packed_pcs\] prove ([0-9.]+)s\s+verify ([0-9.]+)s.*verified (\w+)")
SKIPPED = re.compile(r"\[packed_pcs\] SKIPPED")

_UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9, "": 1.0}


def secs(m) -> float:
    return float(m.group(1)) * _UNIT.get(m.group(2), 1.0)


def visible_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                             timeout=15).stdout
        return sum(1 for l in out.splitlines() if l.startswith("GPU "))
    except Exception:
        return 1


def run(name, binary, env, config, gpus, timeout, logdir):
    exe = RELEASE / binary
    if not exe.exists():
        return {"model": name, "status": f"missing binary {binary}"}
    e = dict(os.environ)
    e.update(env)
    e["ZK4_DEFER_CONSTANTS"] = "1"      # deferred fixed-weight opening
    e["ZK4_PCS"] = "packed"
    e["ZK4_TIMING"] = "1"
    e["ZK4_GPU_DEVICES"] = ",".join(str(i) for i in range(gpus))
    e.setdefault("ZK4_PCS_BUDGET_GB", "40")

    try:
        p = subprocess.run([str(exe), config], cwd=REPO, env=e,
                           capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return {"model": name, "status": f"timeout >{timeout}s"}

    (logdir / f"{name.replace('/', '_').replace(' ', '_')}.log").write_text(out)

    row = {"model": name, "status": "ok"}
    for k, pat in PAT.items():
        m = pat.search(out)
        if m and k == "leaves":
            row["leaves"] = int(m.group(1))
            row["deferred"] = int(m.group(2))
        elif m:
            row[k] = secs(m)
    if SKIPPED.search(out):
        row["status"] = "packed skipped (unsupported leaf repr)"
        return row
    m = PACKED.search(out)
    if m:
        row["pk_pack"] = float(m.group(1))
        row["pk_commit"] = float(m.group(2))
        row["pk_link"] = float(m.group(3))
    m = PACKED_TOTAL.search(out)
    if m:
        row["pk_total"] = float(m.group(1))
        row["pk_verify"] = float(m.group(2))
        row["pk_ok"] = m.group(3) == "true"
    if "pk_total" not in row:
        row["status"] = "no packed output"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=0, help="0 = all visible")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; a model runs if it matches any")
    ap.add_argument("--out", default="out/packed_matrix.csv")
    a = ap.parse_args()

    gpus = a.gpus or visible_gpus() or 1
    logdir = REPO / "scripts" / "bench" / "logs_packed"
    logdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, binary, env, config in MODELS:
        if a.only:
            pats = [x.strip().lower() for x in a.only.split(",") if x.strip()]
            if not any(x in name.lower() for x in pats):
                continue
        print(f"running {name} ...", flush=True)
        r = run(name, binary, env, config, gpus, a.timeout, logdir)
        rows.append(r)
        if r.get("pk_total") is not None:
            ft = r.get("fold_tree_s", float("nan"))
            print(f"    fold tree {ft:8.2f}s   packed {r['pk_total']:8.2f}s "
                  f"({ft / r['pk_total']:.2f}x)  verified={r.get('pk_ok')}", flush=True)
        else:
            print(f"    {r['status']}", flush=True)

    hdr = ("model", "leaves", "deferred", "fold_tree_s", "pk_total", "speedup",
           "pk_pack", "pk_commit", "pk_link", "pk_verify", "pk_ok", "status")
    print("\n" + " | ".join(f"{h}" for h in hdr))
    outp = REPO / "scripts" / "bench" / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            sp = (r["fold_tree_s"] / r["pk_total"]
                  if r.get("pk_total") and r.get("fold_tree_s") else "")
            vals = [r.get("model", ""), r.get("leaves", ""), r.get("deferred", ""),
                    f"{r.get('fold_tree_s', ''):.3f}" if r.get("fold_tree_s") else "",
                    f"{r.get('pk_total', ''):.3f}" if r.get("pk_total") else "",
                    f"{sp:.2f}" if sp else "",
                    f"{r.get('pk_pack', ''):.3f}" if r.get("pk_pack") else "",
                    f"{r.get('pk_commit', ''):.3f}" if r.get("pk_commit") else "",
                    f"{r.get('pk_link', ''):.3f}" if r.get("pk_link") else "",
                    f"{r.get('pk_verify', ''):.3f}" if r.get("pk_verify") else "",
                    r.get("pk_ok", ""), r.get("status", "")]
            line = ",".join(str(v) for v in vals)
            f.write(line + "\n")
            print(line)
    print(f"\nwrote {outp}  (per-model logs in {logdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

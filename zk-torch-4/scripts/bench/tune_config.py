#!/usr/bin/env python3
"""Find the fold-tree-optimal `table_commit_log` for each model.

    python3 tune_config.py --profile full --tcl 4,6,8 --out out/tune_full.csv

Background
----------
Every sparse lookup auxiliary is split into `ceil(table_num_vars /
table_commit_log)` chunks, each committed at arity `input_n +
table_commit_log` (`dag/mod.rs`, the NO_SPARSE_SPLIT block). The fold tree
runs its same-point sumcheck on GPU only while arity <= 24
(`ZK4_GPU_SP_MAX_ARITY`); above that it falls back to CPU and the GPUs go
idle. So `table_commit_log` sets a cliff:

    optimal tcl ~= 24 - max_input_n

Below that value nothing improves, because the model's dense (non-auxiliary)
leaves already sit near arity 24, while the chunk count keeps rising and with
it the range-lookup cost. Measured on llama2 8L/seq64 (4xA100), tcl 12 -> 6
took prove from 97.6s to 37.4s and the proof from 3412 MB to 172 MB.

`table_size_log` is the other half of the pair and is NOT what this script
tunes: it sets range coverage, changes only the chunk count, and measured
free (12 -> 16 was inside run-to-run noise). Pick it from the model family's
activation range, then tune `table_commit_log` here.

Output
------
One CSV row per (model, tcl) with prove/fold-tree/range timings, the top
fold-tree bucket arity, leaf count, proof size and the verdict, plus a
per-model summary naming the fastest verifying tcl. Non-verifying rows are
kept, not dropped: an OOM or a range overflow at one tcl is a result.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE is <repo>/zk-torch-4/scripts/bench, so two levels up IS zk-torch-4.
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from models import zoo  # noqa: E402

# Techniques held fixed across the sweep so only table_commit_log varies.
FIXED = {
    "ZK4_DEVICE_RESIDENT_FOLD": "1", "ZK4_SPARSE_SP": "1",
    "ZK4_SPARSE_BOOL": "1", "ZK4_SHARED_EQ": "1", "ZKT_RUN_BACKEND": "gpu",
    "ZK4_TIMING": "1",
}

PATS = {
    "prove_s":      (re.compile(r"^Prove:\s+(.+)$", re.M), "dur"),
    "fold_tree_s":  (re.compile(r"\[prove\] fold tree:\s+(.+)$", re.M), "dur"),
    "backward_s":   (re.compile(r"\[prove\] partition backward:\s+(.+)$", re.M), "dur"),
    "range_s":      (re.compile(r"lookup proofs: two_pow=\S+ range=(\S+)", re.M), "dur"),
    "two_pow_s":    (re.compile(r"lookup proofs: two_pow=(\S+)", re.M), "dur"),
    "leaves":       (re.compile(r"leaf build total \((\d+) leaves", re.M), "int"),
    "proof_bytes":  (re.compile(r"^Proof size:\s+(\d+)", re.M), "int"),
}
DUR = re.compile(r"(?:(\d+)m)?([\d.]+)(ms|µs|us|ns|s)(?![a-zA-Z])")


def parse_dur(s: str) -> float | None:
    m = DUR.search(s.strip())
    if not m:
        return None
    mins, val, unit = m.group(1), float(m.group(2)), m.group(3)
    sec = {"s": 1.0, "ms": 1e-3, "µs": 1e-6, "us": 1e-6, "ns": 1e-9}[unit] * val
    return sec + (int(mins) * 60 if mins else 0)


def classify(out: str, rc: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    low = out.lower()
    if "out of memory" in low or "oom" in low or rc == -9 or "killed" in low:
        return "oom"
    if "will fail to verify" in low:
        return "range_table_overflow"
    if "panicked at" in low:
        m = re.search(r"panicked at [^\n]*\n\s*([^\n]{0,110})", out)
        return "panic: " + (m.group(1).strip() if m else "?")
    if re.search(r"^Verified: true", out, re.M):
        return "ok"
    if re.search(r"^Verified: false", out, re.M):
        return "verify_failed"
    return f"no_verdict(rc={rc})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "paper", "smoke"])
    ap.add_argument("--tcl", default="4,6,8", help="table_commit_log values to try")
    ap.add_argument("--tsl", default="",
                    help="override table_size_log for every model (default: keep "
                         "each model's own config file value)")
    ap.add_argument("--only", default="", help="substring filter on model name")
    ap.add_argument("--group", default="",
                    help="comma-separated model groups (Transformer, Convolutional, "
                         "Heterogeneous); table_size_log differs by family so the "
                         "groups are normally swept separately")
    ap.add_argument("--partitions", default="4")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--out", default="out/tune.csv")
    a = ap.parse_args()

    tcls = [int(x) for x in a.tcl.split(",") if x.strip()]
    cases = zoo(a.profile, timeout_s=a.timeout)
    if a.group:
        want = {g.strip().lower() for g in a.group.split(",") if g.strip()}
        cases = [c for c in cases if c.group.lower() in want]
    if a.only:
        cases = [c for c in cases if a.only.lower() in c.name.lower()]
    if not cases:
        return print(f"no model matches {a.only!r}") or 1

    os.makedirs(os.path.dirname(os.path.join(HERE, a.out)) or ".", exist_ok=True)
    csv_path = os.path.join(HERE, a.out)
    logdir = os.path.join(HERE, "logs_tune")
    os.makedirs(logdir, exist_ok=True)
    cfgdir = os.path.join(HERE, "cfg_tune")
    os.makedirs(cfgdir, exist_ok=True)

    rows: list[dict] = []
    print(f"profile={a.profile}  models={len(cases)}  tcl={tcls}  "
          f"partitions={a.partitions}  timeout={a.timeout}s", flush=True)

    for ci, case in enumerate(cases, 1):
        # read the model's own config to inherit scale_factor_log / table_size_log
        base = {}
        cfg_src = os.path.join(ROOT, case.config)
        try:
            for line in open(cfg_src):
                m = re.match(r"\s*(scale_factor_log|table_size_log|table_commit_log):\s*(\d+)", line)
                if m:
                    base[m.group(1)] = int(m.group(2))
        except OSError:
            pass
        sfl = base.get("scale_factor_log", 10)
        tsl = int(a.tsl) if a.tsl else base.get("table_size_log", 16)

        print(f"\n[{ci}/{len(cases)}] {case.name}   (binary={case.binary}, "
              f"config={case.config}, sfl={sfl}, tsl={tsl})", flush=True)

        for tcl in tcls:
            if tcl > tsl:
                print(f"    tcl={tcl}: skipped (tcl > table_size_log={tsl})", flush=True)
                continue
            cfg = os.path.join(cfgdir, f"tune_sfl{sfl}_tsl{tsl}_tcl{tcl}.yaml")
            with open(cfg, "w") as f:
                f.write(f"sf:\n  scale_factor_log: {sfl}\n"
                        f"  table_size_log: {tsl}\n  table_commit_log: {tcl}\n")

            env = dict(os.environ, **FIXED, **case.env)
            env["NUM_PARTITIONS"] = a.partitions
            env.setdefault("ZK4_GPU_DEVICES", "0,1,2,3")

            t0 = time.time()
            timed_out = False
            try:
                p = subprocess.run(
                    [os.path.join(ROOT, "target", "release", case.binary), cfg],
                    cwd=ROOT, env=env,
                    capture_output=True, text=True, timeout=a.timeout)
                out, rc = p.stdout + p.stderr, p.returncode
            except subprocess.TimeoutExpired as e:
                out = (e.stdout or "") + (e.stderr or "")
                if isinstance(out, bytes):
                    out = out.decode("utf8", "replace")
                rc, timed_out = -1, True
            wall = time.time() - t0

            slug = re.sub(r"[^A-Za-z0-9]+", "_", case.name).strip("_")
            log = os.path.join(logdir, f"{slug}__tcl{tcl}.log")
            with open(log, "w") as f:
                f.write(out)

            row = {"model": case.name, "binary": case.binary,
                   "scale_factor_log": sfl, "table_size_log": tsl,
                   "table_commit_log": tcl, "wall_s": round(wall, 1),
                   "status": classify(out, rc, timed_out), "log": log}
            for k, (pat, kind) in PATS.items():
                m = pat.search(out)
                if not m:
                    row[k] = ""
                elif kind == "dur":
                    v = parse_dur(m.group(1))
                    row[k] = round(v, 2) if v is not None else ""
                else:
                    row[k] = int(m.group(1))
            ar = [int(x) for x in re.findall(r"bucket arity=(\d+)", out)]
            row["top_bucket_arity"] = max(ar) if ar else ""
            row["n_buckets"] = len(ar)
            pb = row.get("proof_bytes")
            row["proof_mb"] = round(pb / 1e6, 1) if pb else ""
            rows.append(row)

            print(f"    tcl={tcl}: {row['status']:<22} prove={row['prove_s'] or '-'}s "
                  f"ft={row['fold_tree_s'] or '-'}s range={row['range_s'] or '-'}s "
                  f"top_arity={row['top_bucket_arity'] or '-'} "
                  f"proof={row['proof_mb'] or '-'}MB  ({wall:.0f}s wall)", flush=True)

            # rewrite CSV after every run so partial results survive a kill
            if rows:
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    print(f"\nwrote {csv_path}\n")
    print("=== best verifying table_commit_log per model ===")
    for case in cases:
        cand = [r for r in rows if r["model"] == case.name
                and r["status"] == "ok" and r["prove_s"] != ""]
        if not cand:
            bad = [r for r in rows if r["model"] == case.name]
            why = ", ".join(f"tcl{r['table_commit_log']}={r['status']}" for r in bad)
            print(f"  {case.name:<34} NO VERIFYING RUN  ({why})")
            continue
        best = min(cand, key=lambda r: r["prove_s"])
        worst = max(cand, key=lambda r: r["prove_s"])
        gain = f"{worst['prove_s']/best['prove_s']:.2f}x vs tcl={worst['table_commit_log']}" \
            if worst is not best else "single point"
        print(f"  {case.name:<34} tcl={best['table_commit_log']:<3} "
              f"tsl={best['table_size_log']:<3} prove={best['prove_s']:>8.2f}s "
              f"top_arity={best['top_bucket_arity']:<3} "
              f"proof={best['proof_mb']}MB   ({gain})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

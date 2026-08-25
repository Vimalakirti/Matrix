#!/usr/bin/env python3
"""Deferred weight opening.  Fills Section 9.5 and the intro's amortization
number.

    python3 run_deferred.py --model GPT-2 --proofs 1,2,4,8,16

The streaming binaries prove N inferences of one committed model, keeping the
weight commitments fixed and accumulating their evaluation claims, then pay one
terminal opening at the end. They print a per-proof breakdown, the amortized
per-proof total, and the one-time finalize, which is what Section 5.4 predicts
should fall as 1/N.

Each streaming run is compared against the monolithic binary at the same
configuration: that row is the "open every weight in every proof" baseline, so
the ratio is the amortization the paper claims.
"""

from __future__ import annotations

import argparse

from harness import Case, require_release_build, run_cases, summarize, write_csv
from models import STREAMING, zoo

# monolithic counterpart per streaming model, used as the per-proof baseline
MONOLITHIC = {
    "GPT-2": "gpt2", "Llama-2": "llama2", "BERT": "bert",
    "ResNet-50": "resnet", "VGG-16": "vgg", "Whisper": "whisper",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GPT-2",
                    help=f"one or more of {','.join(STREAMING)} (comma separated)")
    ap.add_argument("--proofs", default="1,2,4,8",
                    help="values of N_PROOFS to sweep")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="override a model knob, repeatable (e.g. --env NUM_LAYERS=1); "
                         "useful for smoke-testing this script at a tiny size")
    ap.add_argument("--out", default="out/deferred.csv")
    a = ap.parse_args()
    overrides = dict(kv.split("=", 1) for kv in a.env if "=" in kv)

    keys = [k.strip() for k in a.model.split(",") if k.strip()]
    unknown = [k for k in keys if k not in STREAMING]
    if unknown:
        return print(f"unknown model(s) {unknown}; known: {list(STREAMING)}") or 1
    counts = [c.strip() for c in a.proofs.split(",") if c.strip()]

    cases = []
    for key in keys:
        binary, env, config = STREAMING[key]
        for n in counts:
            cases.append(Case(name=f"{key}, N={n} (deferred)", binary=binary,
                              env=dict(env, N_PROOFS=n, **overrides), config=config,
                              group=key, timeout_s=a.timeout))
        # baseline: the monolithic prover opens every weight on every proof
        mono = MONOLITHIC.get(key)
        if mono:
            cases.append(Case(name=f"{key}, per proof (no deferral)", binary=mono,
                              env=dict(env, **overrides), config=config, group=key,
                              timeout_s=a.timeout))

    require_release_build([c.binary for c in cases])
    print(f"models={keys}  N_PROOFS in {counts}  reps={a.reps}\n")
    runs = run_cases(cases, reps=a.reps)
    write_csv(runs, a.out)

    rows = {r["name"]: r for r in summarize(runs)}
    for key in keys:
        print(f"\n{key}: amortization")
        base = rows.get(f"{key}, per proof (no deferral)", {})
        b = base.get("prove_ms")
        print(f"  {'N':>4}  {'per-proof amortized':>20}  {'finalize':>10}  {'vs no deferral':>15}")
        for n in counts:
            r = rows.get(f"{key}, N={n} (deferred)", {})
            per = r.get("stream_perproof_ms") or r.get("prove_ms") or ""
            fin = r.get("stream_finalize_ms") or ""
            if per in ("", None):
                print(f"  {n:>4}  {(r.get('failure') or 'no data'):>20}")
                continue
            speed = f"{float(b)/float(per):.2f}x" if b not in ("", None) else "-"
            fin_s = f"{float(fin)/1e3:.2f}s" if fin not in ("", None) else "-"
            print(f"  {n:>4}  {float(per)/1e3:>19.2f}s  {fin_s:>10}  {speed:>15}")
        if b in ("", None):
            print("  (no monolithic baseline: the ratio column needs it)")
        else:
            print(f"  baseline (monolithic prove, per proof): {float(b)/1e3:.2f}s")
    print("\nThe per-proof figure should fall as N grows, approaching "
          "prove-with-deferral plus the accumulator update, with the single "
          "terminal opening divided over the stream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

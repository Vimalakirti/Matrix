#!/usr/bin/env python3
"""Sparse versus dense opening.  Fills the intro's memory-reduction number
and the ablation behind Section 5.

    python3 run_sparsity.py --model gpt2 --sizes 64,128,256

For each size it runs the same configuration twice:

  sparse (shipped)  ZK4_SPARSE_SP=1 ZK4_SPARSE_BOOL=1
  dense  (baseline) ZK4_SPARSE_SP=0 ZK4_SPARSE_BOOL=0

both with the fold tree pinned to the host same-point path, for the reason
given at HOST_PATH below, and reports prove time and peak GPU memory for both, plus the sizes at which
the dense path stops completing. That OOM boundary is the "enabling
configurations for which the dense baseline runs out of memory" claim, so a
dense `oom` row is a RESULT, not a failed run: the harness records it and the
table prints it.

Caveat worth respecting when picking sizes: the sparse paths only engage at
arity >= ZK4_SPARSE_SP_MIN_ARITY / ZK4_SPARSE_BOOL_MIN_ARITY, both 20 by
default, and the density gates ZK4_SPARSE_SP_RATIO (8) / ZK4_SPARSE_BOOL_RATIO
(2) must also hold. A configuration whose auxiliaries stay below that shows no difference
because nothing switched, not because sparsity does not help. Sweep upward
until the two curves separate.
"""

from __future__ import annotations

import argparse

from harness import Case, require_release_build, run_cases, summarize, write_csv
from models import BENCH, CV, LLAMA

# Sparsity is about the lookup/range auxiliaries, whose arity grows with the
# activation count, so sequence length (transformers) or spatial size (CV) is
# the knob to sweep.
MODELS = {
    "gpt2":   ("gpt2",   "SEQ_LEN",    {"NUM_LAYERS": "12", "MAX_NUM_VARS": "24"}, BENCH),
    "llama2": ("llama2", "SEQ_LEN",    {"NUM_LAYERS": "8", "MAX_NUM_VARS": "27"},  LLAMA),
    "bert":   ("bert",   "SEQ_LEN",    {"NUM_LAYERS": "12", "MAX_NUM_VARS": "24"}, BENCH),
    "resnet": ("resnet", "INPUT_SIZE", {"NUM_LAYERS": "53"},                       CV),
}

# ZK4_SPARSE_SP is only consulted on the HOST same-point path
# (InstanceState::new). By default the fold tree takes the device-resident /
# GPU path for arity 18..24 and never builds that state, so ZK4_SPARSE_SP=0
# alone is a no-op at exactly the arities that matter. Both sides are
# therefore pinned to the host path, and the flags are the only difference.
HOST_PATH = {
    "ZK4_DEVICE_RESIDENT_FOLD": "0",
    "ZK4_GPU_SP_MAX_ARITY": "0",   # no bucket satisfies arity <= 0
}
SPARSE_ENV = dict(HOST_PATH, ZK4_SPARSE_SP="1", ZK4_SPARSE_BOOL="1")
DENSE_ENV = dict(HOST_PATH, ZK4_SPARSE_SP="0", ZK4_SPARSE_BOOL="0")

# Deliberately NOT part of the pair: NO_SPARSE_SPLIT is presence-only (even
# "=0" disables) and raises every auxiliary's arity from
# input_n + table_commit_log to input_n + table_size_log, which changes the
# fold-tree bucket structure rather than isolating sparsity. Study that axis
# with the config pair bench_config.yaml vs bench_config_single_chunk.yaml.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help=f"one of {','.join(MODELS)}")
    ap.add_argument("--sizes", default="64,128,256",
                    help="values for the swept knob (seq len or input size)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="override a model knob, repeatable (e.g. --env NUM_LAYERS=1); "
                         "useful for smoke-testing this script at a tiny size")
    ap.add_argument("--out", default="out/sparsity.csv")
    a = ap.parse_args()
    overrides = dict(kv.split("=", 1) for kv in a.env if "=" in kv)

    if a.model not in MODELS:
        return print(f"unknown model; known: {list(MODELS)}") or 1
    binary, knob, base_env, config = MODELS[a.model]
    sizes = [s.strip() for s in a.sizes.split(",") if s.strip()]

    cases = []
    for s in sizes:
        common = dict(base_env, **{knob: s}, **overrides)
        cases.append(Case(name=f"{a.model} {knob.lower()}={s}, sparse",
                          binary=binary, env=dict(common, **SPARSE_ENV), config=config,
                          group=f"{knob.lower()}={s}", timeout_s=a.timeout))
        cases.append(Case(name=f"{a.model} {knob.lower()}={s}, dense",
                          binary=binary, env=dict(common, **DENSE_ENV),
                          config=config, group=f"{knob.lower()}={s}",
                          timeout_s=a.timeout))

    require_release_build([binary])
    print(f"model={a.model}  {knob} in {sizes}  reps={a.reps}")
    print("sparse env: " + " ".join(f"{k}={v}" for k, v in SPARSE_ENV.items()))
    print("dense  env: " + " ".join(f"{k}={v}" for k, v in DENSE_ENV.items()) + "\n")
    runs = run_cases(cases, reps=a.reps)
    write_csv(runs, a.out)

    rows = {r["name"]: r for r in summarize(runs)}
    print("\nsparse vs dense:")
    print(f"  {'configuration':<28}{'sparse':>12}{'dense':>12}{'ratio':>8}"
          f"{'sparse mem':>12}{'dense mem':>12}")
    for s in sizes:
        sp = rows.get(f"{a.model} {knob.lower()}={s}, sparse", {})
        dn = rows.get(f"{a.model} {knob.lower()}={s}, dense", {})

        def t(row):
            v = row.get("prove_ms", "")
            return f"{float(v)/1e3:.2f}s" if v not in ("", None) else (row.get("failure") or "-")

        def m(row):
            v = row.get("peak_mem_mib", 0)
            return f"{float(v)/1024:.1f}GiB" if v else "-"

        ratio = "-"
        if sp.get("prove_ms") not in ("", None) and dn.get("prove_ms") not in ("", None):
            ratio = f"{float(dn['prove_ms'])/float(sp['prove_ms']):.2f}x"
        print(f"  {knob.lower()+'='+s:<28}{t(sp):>12}{t(dn):>12}{ratio:>8}"
              f"{m(sp):>12}{m(dn):>12}")

    oom = [s for s in sizes
           if rows.get(f"{a.model} {knob.lower()}={s}, dense", {}).get("failure")
           and not rows.get(f"{a.model} {knob.lower()}={s}, sparse", {}).get("failure")]
    if oom:
        print(f"\ndense path could not complete at {knob.lower()} in {oom} "
              f"while the sparse path did: this is the enabling result.")
    else:
        print("\nno size yet separates the two paths by feasibility. If the times "
              "are also equal, check that the auxiliaries exceed arity 20 "
              "(otherwise no sparse path engaged) and sweep larger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

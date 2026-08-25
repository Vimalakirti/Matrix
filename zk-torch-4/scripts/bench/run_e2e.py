#!/usr/bin/env python3
"""End-to-end proving cost across the model zoo.  Fills Section 9.2.

    python3 run_e2e.py --profile smoke            # validate the pipeline
    python3 run_e2e.py --profile full --reps 3    # the true model sizes
    python3 run_e2e.py --profile paper --reps 3   # reduced, fits a small node

Every run enables the full prover-side technique set (see models.techniques):
sparse opening paths, device-resident fold tree, device-resident witnesses,
terminal-opening sharding over the device pool, and backward-pass graph
partitioning across the GPUs. `--gpus` picks the pool size and defaults to
every visible device; `--forward cpu` and `--no-techniques` exist to isolate
what a technique contributes.

Deferred weight opening is not part of this table: it defers work to a
finalize that only exists across a stream, and a single proof with deferral is
not a standalone pass. run_deferred.py measures its amortization.

Produces out/e2e.csv (medians) + out/e2e.raw.csv (per repetition), then
`python3 make_tables.py out/e2e.csv` writes out/e2e.tex.
"""

from __future__ import annotations

import argparse

from harness import require_release_build, run_cases, write_csv
from models import binaries, techniques, zoo


def visible_gpu_count() -> int:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=15).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full",
                    choices=["full", "paper", "smoke"])
    ap.add_argument("--gpus", type=int, default=0,
                    help="device pool size; 0 (default) uses every visible GPU")
    ap.add_argument("--forward", default="gpu", choices=["gpu", "cpu"],
                    help="where the forward pass runs; gpu keeps witnesses "
                         "device-resident for the commitment kernels")
    ap.add_argument("--no-techniques", action="store_true",
                    help="run the bins bare, for isolating a technique")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-run seconds before the case is marked timeout")
    ap.add_argument("--only", default="",
                    help="substring filter on the model name")
    ap.add_argument("--group", default="",
                    help="comma-separated groups (Transformer, Convolutional, "
                         "Heterogeneous). Cheap models first gets results landing "
                         "sooner on a profile where one row can take an hour.")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="override a model knob, repeatable (e.g. --env NUM_LAYERS=1); "
                         "useful for smoke-testing this script at a tiny size")
    ap.add_argument("--out", default="out/e2e.csv")
    a = ap.parse_args()
    overrides = dict(kv.split("=", 1) for kv in a.env if "=" in kv)

    gpus = a.gpus or visible_gpu_count()
    tech = {} if a.no_techniques else techniques(gpus, a.forward)

    cases = zoo(a.profile, timeout_s=a.timeout)
    for c in cases:
        # model knobs win over the technique bundle, so a model that pins
        # NUM_PARTITIONS or a backend keeps its own value
        # Dict-literal merge, not dict(**a, **b): models.py now sets
        # ZK4_TABLE_SIZE_LOG / ZK4_TABLE_COMMIT_LOG per model, so a --env
        # override of the same key is a duplicate keyword and raises.
        # Precedence is unchanged: techniques < model knobs < --env.
        c.env = {**tech, **c.env, **overrides}
    if a.group:
        want = {g.strip().lower() for g in a.group.split(",") if g.strip()}
        cases = [c for c in cases if c.group.lower() in want]
    if a.only:
        cases = [c for c in cases if a.only.lower() in c.name.lower()]
        if not cases:
            return print(f"no model matches {a.only!r}") or 1
    require_release_build(binaries(a.profile))

    print(f"profile={a.profile}  models={len(cases)}  reps={a.reps}")
    if tech:
        print(f"techniques: {' '.join(f'{k}={v}' for k, v in sorted(tech.items()))}")
    else:
        print("techniques: DISABLED (--no-techniques)")
    print()
    runs = run_cases(cases, reps=a.reps)
    write_csv(runs, a.out)

    bad = [r for r in runs if not r.ok]
    if bad:
        print(f"\n{len(bad)} failed run(s):")
        for r in bad:
            print(f"  {r.group}/{r.name} rep{r.rep}: {r.failure or 'unverified'}"
                  f"  ({r.log})")
        print("\nA row with a failure is not reportable. Inspect the log, then "
              "tune the knobs named in models.py (MAX_NUM_VARS and the *_SHARDS "
              "values are the usual fix for a run that does not fit) or move to "
              "a larger node.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Where the wall clock actually went, per model, from a feasibility CSV.

Splits each row into the FIXED cost (weight gen + compile + offline weight
commit -- paid once per model per stage, independent of N) and the PER-INFERENCE
cost (forward + online commit + prove + verify -- multiplied by N).

That split is what a schedule estimate needs: raising N only multiplies the
second, while adding a stage pays the first again, and the scaling ladder pays
it once per device count.

    python3 where_time_went.py out/feasibility.csv
"""
import csv, sys

FIXED = ["weightgen_ms", "compile_ms", "commit_offline_ms"]
PER_N = ["forward_ms", "commit_online_ms", "prove_ms", "verify_ms",
         "stream_perproof_ms"]

def s(row, keys):
    t = 0.0
    for k in keys:
        v = row.get(k, "")
        if v not in ("", None):
            try: t += float(v)
            except ValueError: pass
    return t / 1000.0

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1
                                else "out/feasibility.csv")))
print(f"{'model':<28} {'fixed':>9} {'per-inf':>9} {'row total':>10}  ver")
tf = tp = 0.0
for r in rows:
    f, p = s(r, FIXED), s(r, PER_N)
    tf += f; tp += p
    print(f"{r['name']:<28} {f:>8.1f}s {p:>8.1f}s {f+p:>9.1f}s  {r.get('verified','')}")
print(f"{'TOTAL':<28} {tf:>8.1f}s {tp:>8.1f}s {tf+tp:>9.1f}s")
print(f"\n  fixed is {tf/max(tf+tp,1)*100:.0f}% of the measured total.")
print(f"  a stage at N proofs costs about  {tf/60:.0f} min + N x {tp/60:.0f} min")
print(f"  a scaling ladder over 1/2/4/8 GPUs pays the fixed part FOUR times.")

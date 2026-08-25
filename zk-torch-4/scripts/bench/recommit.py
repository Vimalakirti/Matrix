#!/usr/bin/env python3
"""Re-parse existing run logs for the ONLINE commit, using the harness's own
patterns rather than ad-hoc greps.

The streaming binaries report the per-proof commit only inside the
per-iteration table; the CSVs written before harness commit 05ca6d3 therefore
recorded commit_online_ms = 0 for every streaming row. This recovers it from
the logs already on disk, so no run has to be repeated.

    cd zk-torch-4/scripts/bench && python3 recommit.py
"""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness

print("run,commit_online_ms,source")
for f in sorted(glob.glob("logs/*.log")):
    txt = open(f, errors="replace").read()
    name = re.sub(r"__r\d+\.log$", "", os.path.basename(f))
    per = harness._STREAM_ITER_COMMIT.findall(txt)
    if per:
        v = sum(float(x) for x in per) / len(per)
        print(f'"{name}",{v:.1f},per-iteration')
        continue
    pat, kind = harness.METRICS["commit_online_ms"]
    m = pat.search(txt)
    if m:
        v = harness.parse_duration_ms(m.group(1).strip())
        if v is not None:
            print(f'"{name}",{v:.1f},summary-line')

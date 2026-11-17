#!/usr/bin/env python3
"""Summarize run_models.sh logs (one directory) for the GPU evaluation tables.

    python3 gpu_tables.py <out_dir> [--json]

Per log NAME_mono.log: requests, mean / min / max prove (s), mean verify (s),
proof (MiB), peak summed GPU memory (GB), accepted attempts, and the mean
prover stage split. Every request must have verified (true); a log with a
panic, a rejection or a missing verification is reported as FAILED.
"""
import json
import re
import sys
from pathlib import Path


def stages(line):
    return {k: float(v) for k, v in re.findall(r"(\w[\w ]*?) ([\d.]+)(?:,|\])", line.split("[", 1)[1])}


def parse(path):
    t = path.read_text(errors="replace")
    pro = re.findall(r"^request (\d+) prove: ([\d.]+)s\s+(\[.*\])", t, re.M)
    ver = re.findall(r"^request (\d+) verify \((\w+)\): ([\d.]+)s\s+(\[.*\])", t, re.M)
    att = [int(a) + 1 for a in re.findall(r"^\[auth[^\]]*\] [\d.]+s, attempt (\d+)", t, re.M)]
    mib = re.search(r"^proof per request\s+([\d.]+) MiB", t, re.M)
    peak = re.search(r"^peak GPU memory \(sum over devices\) (\d+) MiB", t, re.M)
    sup = re.search(r"public response support: (\d+) of (\d+)", t)
    exit_ = re.search(r"^MORPHEUS_EXIT status=(\d+)", t, re.M)
    # Serialized proof parts of request 0 (bytes): the authentication
    # responses are full-width over the source columns; the rest depends on
    # the workload and the reductions.
    pm = re.search(r"proof parts \(bytes\): (.*)", t)
    parts = {k.strip(): int(v) for k, v in re.findall(r"([a-z ]+?) (\d+)(?:,|$)", pm.group(1))} if pm else {}
    ok = bool(pro) and len(ver) == len(pro) and all(v[1] == "true" for v in ver) \
        and "panicked" not in t and (exit_ is None or exit_.group(1) == "0")
    p = [float(x[1]) for x in pro]
    v = [float(x[2]) for x in ver]
    split = {}
    for x in pro:
        for k, val in stages(x[2]).items():
            split[k] = split.get(k, 0.0) + val / len(pro)
    vsplit = {}
    for x in ver:
        for k, val in stages(x[3]).items():
            vsplit[k] = vsplit.get(k, 0.0) + val / len(ver)
    return {
        "ok": ok, "n": len(p),
        "prove": sum(p) / len(p) if p else None, "prove_min": min(p) if p else None, "prove_max": max(p) if p else None,
        "verify": sum(v) / len(v) if v else None,
        "proof_mib": float(mib.group(1)) if mib else None,
        "peak_gb": int(peak.group(1)) / 1024 if peak else None,
        "attempts": att, "support": f"{sup.group(1)}/{sup.group(2)}" if sup else None,
        "split": split, "vsplit": vsplit,
        "parts": parts,
        "nonresp_mib": (sum(parts.values()) - parts.get("responses", 0)) / 2**20 if parts else None,
    }


def main():
    d = Path(sys.argv[1])
    rows = {}
    for f in sorted(d.glob("*_mono.log")):
        rows[f.name[:-len("_mono.log")]] = parse(f)
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
        return
    f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"
    print(f"{'config':24} {'ok':>3} {'n':>2} {'prove':>8} {'min':>8} {'max':>8} {'verify':>7} {'MiB':>8} {'nonresp':>8} {'peakGB':>7} attempts  split")
    for k, r in rows.items():
        sp = ", ".join(f"{a} {b:.1f}" for a, b in sorted(r["split"].items(), key=lambda z: -z[1]))
        print(f"{k:24} {'ok' if r['ok'] else 'FAILED':>3} {r['n']:2} {f(r['prove'])} {f(r['prove_min'])} {f(r['prove_max'])} {f(r['verify'])} {f(r['proof_mib'], 1)} {f(r['nonresp_mib'], 2)} {f(r['peak_gb'], 0)} {r['attempts']}  {sp}")


if __name__ == "__main__":
    main()

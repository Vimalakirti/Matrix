#!/usr/bin/env python3
"""Summarize run_models.sh logs into the paper's end-to-end and deferral tables.

    python3 summarize.py [out_dir]

Per model (MODEL_mono.log, MODEL_defer.log):
  mono      prove per request (weights opened per request, N = 1)
  N=2       amortized prove per request with one finalization over 2 requests
  N->inf    mean deferred per-request prove (finalization amortized away)
  per unit  N->inf divided by the model's unit count (generated tokens for
            decoder-only models, input tokens for BERT, 1 otherwise)
  verify    per-request verifier time (deferred run), finalize verify apart
  proof     per-request proof size estimate, MiB (z at 36 bits/coefficient)
Prover time excludes witness generation and the one-time model
registration; the z extraction into the host proof is reported apart.
"""
import re
import sys
from pathlib import Path

ORDER = ["GPT-2", "BERT-Large", "LLaMA2-7B", "LLaMA3-8B", "GPT-J-6B", "ResNet-50", "VGG-16",
         "YOLOv11n", "3D-UNet", "PointPillars", "Whisper"]
UNITS = {"GPT-2": 112, "LLaMA2-7B": 112, "LLaMA3-8B": 112, "GPT-J-6B": 112, "BERT-Large": 128}


def grab(text, pat, cast=float):
    m = re.search(pat, text, re.M)
    return cast(m.group(1)) if m else None


def parse(path):
    if not path.exists():
        return None
    t = path.read_text(errors="replace")
    reqs = [float(x) for x in re.findall(r"^request \d+ prove: ([\d.]+)s", t, re.M)]
    vers = [float(x) for x in re.findall(r"^request \d+ verify \(true\): ([\d.]+)s", t, re.M)]
    return {
        "ok": "panicked" not in t and "rejected" not in t.replace("tampered finalization rejected", "") and bool(reqs),
        "reqs": reqs,
        "verify": sum(vers) / len(vers) if vers else None,
        "fin": grab(t, r"^finalize \(\d+ requests.*?\): ([\d.]+)s"),
        "fin_verify": grab(t, r"^finalize verify \(true\): ([\d.]+)s"),
        "reg": grab(t, r"model registration \(one-time\): \d+ sources committed in ([\d.]+)s"),
        "proof": grab(t, r"^proof per request\s+([\d.]+) MiB"),
        "fin_proof": grab(t, r"\+ finalize ([\d.]+) MiB"),
        "streamed": "streamed sources" in t,
        "peak": grab(t, r"^peak GPU memory \(sum over devices\) ([\d.]+) MiB"),
        "parts": re.findall(r"^request 0 prove: [\d.]+s\s+\[(.*)\]", t, re.M),
    }


OLD = {  # previous zk-torch-4 paper (8x H200, non-hiding): N->inf prove, verify, proof MB, peak GB; deferral mono/N=2/N->inf
    "GPT-2": (24.1, 0.79, 121.4, 72, (24.3, 22.5, 19.4)), "BERT-Large": (31.6, 0.90, 80.0, 106, (49.1, 40.3, 31.6)),
    "LLaMA2-7B": (184.1, 5.13, 277.4, 208, (488.9, 404.1, 192.3)), "LLaMA3-8B": (204.6, 1.72, 247.1, 143, (607.8, 364.5, 191.0)),
    "GPT-J-6B": (152.4, 1.52, 235.1, 166, (193.0, 187.2, 168.7)), "ResNet-50": (38.4, 1.32, 132.4, 39, (45.1, 40.1, 38.4)),
    "VGG-16": (5.8, 0.57, 111.0, 19, (7.9, 6.9, 5.8)), "YOLOv11n": (105.3, 1.59, 267.2, 76, None),
    "3D-UNet": (420.1, 4.73, 1683.4, 37, None), "PointPillars": (712.8, 4.44, 851.2, 82, None),
    "Whisper": (500.6, 13.03, 2484.5, 111, (564.8, 500.6, 500.6)),
}


def breakdown(path):
    """Mean per-request stage seconds of a deferred log: commit (+fill, bank,
    carry), PIOP (+prep), link, authentication."""
    if not path.exists():
        return None
    rows = re.findall(r"^request \d+ prove: [\d.]+s\s+\[(.*)\]", path.read_text(errors="replace"), re.M)
    if not rows:
        return None
    acc = {}
    for r in rows:
        for kv in r.split(", "):
            k, v = kv.rsplit(" ", 1)
            acc[k] = acc.get(k, 0.0) + float(v) / len(rows)
    g = lambda *ks: sum(acc.get(k, 0.0) for k in ks)
    return {"commit": g("commit", "fill", "bank", "carry"), "piop": g("piop", "prep"), "link": g("link"), "auth": g("auth")}


def sumcheck_share(path, proof_mib):
    """Fraction of proof bytes in sumcheck messages: PIOP K elements (64 B)
    plus the link's (l+1) round polynomials (at most 5 K elements each)."""
    if not path.exists() or not proof_mib:
        return None
    m = re.search(r"PIOP rows, \d+ carries, (\d+) PIOP messages", path.read_text(errors="replace"))
    if not m:
        return None
    b = int(m.group(1)) * 64 + 28 * 5 * 64
    return b / (proof_mib * (1 << 20))


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "out")
    rows = []
    for m in ORDER:
        mono, dfr = parse(out / f"{m}_mono.log"), parse(out / f"{m}_defer.log")
        if not mono and not dfr:
            continue
        u = UNITS.get(m, 1)
        r = {"model": m, "units": u}
        if mono and mono["ok"]:
            r["mono"] = mono["reqs"][0]
            r["mono_verify"] = mono["verify"]
            r["mono_proof"] = mono["proof"]
        if dfr and dfr["ok"]:
            n = len(dfr["reqs"])
            per = sum(dfr["reqs"]) / n
            r["inf"] = per
            r["n2"] = per + (dfr["fin"] or 0) / n
            r["fin"] = dfr["fin"]
            r["verify"] = dfr["verify"]
            r["fin_verify"] = dfr["fin_verify"]
            r["proof"] = dfr["proof"]
            r["fin_proof"] = dfr["fin_proof"]
            r["reg"] = dfr["reg"]
            r["peak"] = dfr["peak"]
        rows.append(r)

    f = lambda x, d=1: "--" if x is None else f"{x:.{d}f}"
    print("| Model | Mono (N=1) | N=2 | N->inf | per unit | Finalize | Verify | Proof MiB | Peak GB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {f(r.get('mono'))} | {f(r.get('n2'))} | {f(r.get('inf'))} | "
              f"{f(r['inf'] / r['units'], 3) if 'inf' in r else '--'} | {f(r.get('fin'))} | "
              f"{f(r.get('verify'), 2)} | {f(r.get('proof'), 0)} | {f(r['peak'] / 1024 if r.get('peak') else None, 0)} |")
    print()
    print("| Model | Morpheus N->inf (4xA100) | Old N->inf (8xH200) | Speedup | Commit | PIOP | Link | Auth | Sumcheck bytes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        o = OLD.get(r["model"])
        b = breakdown(out / f"{r['model']}_defer.log")
        if "inf" not in r:
            continue
        tot = sum(b.values()) if b else None
        sh = (lambda k: f"{100 * b[k] / tot:.0f}%") if b else (lambda k: "--")
        sc = sumcheck_share(out / f"{r['model']}_defer.log", r.get("proof"))
        print(f"| {r['model']} | {r['inf']:.1f} | {o[0] if o else '--'} | {(o[0] / r['inf']) if o else 0:.2f}x | {sh('commit')} | {sh('piop')} | {sh('link')} | {sh('auth')} | {100 * sc:.2f}% |" if sc else "")
    print()
    print("% LaTeX rows: Model & Prove (N->inf) & Per unit & Verify & Proof")
    for r in rows:
        if "inf" in r:
            print(f"{r['model']:12} & {r['inf']:.1f} & {r['inf'] / r['units']:.3f} & {r['verify']:.2f} & {r['proof']:.0f}\\\\")
    print()
    print("% Deferral: Model & Monolithic & N=2 & N->inf & Speedup")
    for r in rows:
        if "inf" in r and "mono" in r:
            print(f"{r['model']:12} & {r['mono']:.1f} & {r['n2']:.1f} & {r['inf']:.1f} & {r['mono'] / r['inf']:.2f}$\\times$\\\\")


if __name__ == "__main__":
    main()

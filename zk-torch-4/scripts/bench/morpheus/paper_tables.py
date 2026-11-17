#!/usr/bin/env python3
"""LaTeX table bodies of the S&P'27 evaluation from run_models.sh /
run_extra.sh logs (wire format v3: proof sizes are serialized lengths).

    python3 paper_tables.py [out_dir] [extra_dir]

Monolithic logs hold N_MONO independent requests (fresh private coins, so
the number of authentication attempts varies); every time is their mean.
Deferred logs hold measured finalizations of batches J = 2 and J = 8.
Prover time covers source commitments through serialization; it excludes
witness generation and the one-time registration (tab:offline).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize import ORDER, UNITS  # noqa: E402
from evidence_admission import admitted

TEX = {"3D-UNet": "3D U-Net"}
MiB = 1 << 20

# Baselines measured on the same server: prove, verify (s), proof (MiB),
# hardware. DeepProve: scripts/bench/deepprove, best backend per row. EZKL:
# CPU build at e196b111 (nanoGPT: setup OOM at 905 GB host memory); the GPU
# build, when measured, is in EZKL_GPU.
EZKL = {"LeNet-5": (57.86, 0.84, 18810 / MiB, "CPU"), "nanoGPT": None}
# GPU build: v22.2.2 with open-source icicle v2.8.0 (MSM and NTT on the GPU),
# the newest build that runs on driver 560; one A100. nanoGPT (seq 64, k=25)
# runs out of GPU memory in proving-key generation.
EZKL_GPU = {"LeNet-5": (45.0, 0.46, 3072 / MiB, "GPU"), "nanoGPT": (None, None, None, "PK OOM")}
# GPT-2 only, both systems on one A100. DeepProve: its CUDA build (inference
# and HyperKZG MSMs on the GPU, sumchecks and lookups on the CPU), mean of 3.
DEEPPROVE = {("GPT-2", 32): (160.1, 2.30, 8.8, "GPU"), ("GPT-2", 128): (248.0, 2.69, 10.8, "GPU")}


def text(path):
    return path.read_text(errors="replace") if path.exists() else ""


def parse(path):
    t = text(path)
    if not t:
        return None
    # Reject the complete log before converting any measurement. Keep the
    # offline-table fields available even for a rejected row.
    rejected = {"ok": False, "oom": "AllocationFailed" in t or "out of memory" in t.lower(), "witness": None, "reg": None}
    if not admitted(t):
        return rejected
    try:
        return parse_admitted(t)
    except (ValueError, OverflowError):
        # Optional breakdown/registration fields are untrusted too.
        return rejected


def parse_admitted(t):
    reqs = [float(x) for x in re.findall(r"^request \d+ prove: ([\d.]+)s", t, re.M)]
    vers = [float(x) for x in re.findall(r"^request \d+ verify \(true\): ([\d.]+)s", t, re.M)]
    g = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, t, re.M))
    batches = {int(m[0]): dict(fin=float(m[1]), fin_verify=float(m[2]), fin_mib=float(m[3]), per=float(m[4]), verify=float(m[5]), mib=float(m[6]))
               for m in re.findall(r"^batch J=(\d+)\s+finalize\s+([\d.]+) s, verify ([\d.]+) s, ([\d.]+) MiB; per accepted request ([\d.]+) s \(.*?\), verify ([\d.]+) s, ([\d.]+) MiB", t, re.M)}
    parts = re.search(r"proof parts \(bytes\): (.*)", t)
    stages = re.findall(r"^request \d+ prove: [\d.]+s\s+\[(.*)\]", t, re.M)
    attempts = [int(a) + 1 for a in re.findall(r"^\[auth[^\]]*\] [\d.]+s, attempt (\d+)", t, re.M)]
    bad = "panicked" in t or "rejected" in t.replace("tampered finalization rejected", "")
    return {
        "ok": not bad,
        "oom": "AllocationFailed" in t or "out of memory" in t.lower(),
        "reqs": reqs,
        "prove": sum(reqs) / len(reqs) if reqs else None,
        "verify": sum(vers) / len(vers) if vers else None,
        "proof": g(r"^proof per request\s+([\d.]+) MiB"),
        "peak": g(r"^peak GPU memory \(sum over devices\) ([\d.]+) MiB"),
        "batches": batches,
        "parts": dict((k, int(v)) for k, v in (kv.rsplit(" ", 1) for kv in parts.group(1).split(", "))) if parts else None,
        "stages": stages,
        "attempts": attempts,
        "witness": g(r"witness ([0-9.]+)s"),
        "reg": (lambda m: (int(m.group(1)), float(m.group(2))) if m else None)(re.search(r"registration \(one-time\): (\d+) sources committed in ([0-9.]+)s", t)),
    }


def breakdown(d):
    """Mean stage seconds: commit (fill, bank, carry), PIOP (prep), link,
    authentication (with serialization of the proof)."""
    acc = {}
    for r in d["stages"]:
        for kv in r.split(", "):
            k, v = kv.rsplit(" ", 1)
            acc[k] = acc.get(k, 0.0) + float(v) / len(d["stages"])
    g = lambda *ks: sum(acc.get(k, 0.0) for k in ks)
    return {"commit": g("commit", "fill", "bank", "carry"), "piop": g("piop", "prep"), "link": g("link"), "auth": g("auth", "serialize")}


def main():
    here = Path(__file__).parent
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "out_v3"
    ext = Path(sys.argv[2]) if len(sys.argv) > 2 else out.parent / out.name.replace("out_", "out_extra_")
    ok = lambda d: d is not None and d["ok"]
    e2e, dfr, brk, bytes_mono, off = [], [], [], [], []
    for m in ORDER:
        mono, d = parse(out / f"{m}_mono.log"), parse(out / f"{m}_defer.log")
        name = TEX.get(m, m)
        u = UNITS.get(m, 1)
        if not ok(mono):
            e2e.append(f"{name:12} & -- & -- & -- & --\\\\")
            bytes_mono.append(f"{name} & --\\\\")
        else:
            t = mono["prove"]
            per = f"{t / u:.3f}" if u > 1 else f"{t:.1f}"
            peak = f"{mono['peak'] / 1024:.0f}" if mono.get("peak") else "--"
            e2e.append(f"{name:12} & {t:.1f} & {per} & {mono['verify']:.2f} & {peak}\\\\")
            bytes_mono.append(f"{name} & {mono['proof']:.0f}\\\\")
            b = breakdown(mono)
            tot = sum(b.values())
            pc = lambda k: f"{100 * b[k] / tot:.0f}\\%"
            p = mono["parts"]
            if p:
                sc = (p["piop"] + p["link rounds"]) / sum(p.values())
                brk.append(f"{name:12} & {pc('commit')} & {pc('piop')} & {pc('link')} & {pc('auth')} & {100 * sc:.1f}\\% & {100 - 100 * sc:.1f}\\%\\\\")
        if ok(mono) and ok(d) and 2 in d["batches"] and 8 in d["batches"]:
            req = d["prove"]
            b2, b8 = d["batches"][2], d["batches"][8]
            dfr.append(f"{name:12} & {mono['prove']:.1f} & {b2['per']:.1f} & {b8['per']:.1f} & {req:.1f} & {mono['prove'] / req:.2f}$\\times$ & {b8['fin']:.1f}\\\\")
        w = mono and mono["witness"]
        r = d and d["reg"]
        off.append(f"{name:12} & " + (f"{w:.1f}" if w else "--") + " & " + (f"{r[1]:.2f} & {r[0]}" if r else "-- & --") + "\\\\")

    # Comparison with existing systems.
    pri, bytes_prior = [], []
    def mrow(path):
        x = parse(path)
        return (x["prove"], x["verify"], x["proof"]) if ok(x) and x["proof"] is not None else None
    def fmt(v, span="--"):
        return f"\\multicolumn{{2}}{{c}}{{{span}}}" if not v else f"{v[0]:.1f} & {v[1]:.2f}"
    for m in ("LeNet-5", "nanoGPT"):
        e = EZKL.get(m)
        pri.append(f"{m} & EZKL (CPU) & " + (f"{e[0]:.1f} & {e[1]:.2f}" if e else "\\multicolumn{2}{c}{setup OOM}") + " \\\\")
        if e:
            bytes_prior.append(f"{m} & EZKL (CPU) & {e[2]:.3f}\\\\")
        if m in EZKL_GPU:
            g = EZKL_GPU[m]
            pri.append(f"        & EZKL (GPU) & " + (f"{g[0]:.1f} & {g[1]:.2f}" if g and g[0] else "\\multicolumn{2}{c}{" + (g[3] if g else "--") + "}") + " \\\\")
            if g and g[2]:
                bytes_prior.append(f"{m} & EZKL (GPU) & {g[2]:.3f}\\\\")
        v = mrow(out / f"{m}_mono.log")
        pri.append(f"        & \\zkt (1 GPU) & {fmt(v)} \\\\")
        if v:
            bytes_prior.append(f"{m} & \\zkt (1 GPU) & {v[2]:.1f}\\\\")
        pri.append("\\midrule")
    for (m, s_), v in DEEPPROVE.items():
        g1 = out / f"{m}_s{s_}_g1_mono.log"
        lab = f"{m} ({s_ - 16} tok.)"
        pri.append(f"{lab} & DeepProve ({v[3]}) & {v[0]:.1f} & {v[1]:.2f} \\\\")
        bytes_prior.append(f"{lab} & DeepProve ({v[3]}) & {v[2]:.1f}\\\\")
        r1 = mrow(g1)
        pri.append(f"        & \\zkt (1 GPU) & " + fmt(r1, "OOM" if g1.exists() and parse(g1)["oom"] else "--") + " \\\\")
        if r1:
            bytes_prior.append(f"{lab} & \\zkt (1 GPU) & {r1[2]:.1f}\\\\")
        pri.append("\\midrule")
    pri = pri[:-1]

    conv, bytes_conv = [], []
    for m in ("VGG-16", "ResNet-50"):
        pp, rt = mrow(out / f"{m}_pp_mono.log"), mrow(out / f"{m}_mono.log")
        conv += [f"{m} & VerfCNN product & {fmt(pp)} \\\\", f"        & Relation tables & {fmt(rt)} \\\\", "\\midrule"]
        bytes_conv += [f"{m} & VerfCNN product & " + (f"{pp[2]:.1f}" if pp else "--") + "\\\\",
                       f"{m} & Relation tables & " + (f"{rt[2]:.1f}" if rt else "--") + "\\\\"]
    conv = conv[:-1]

    # Scaling: deferred request stage (finalization excluded), per generated
    # token for GPT-2 / LLaMA3-8B and per input for the CNNs.
    def stage(path):
        d = parse(path)
        return (d["prove"] if ok(d) else None), d
    scl = []
    for m, u in [("GPT-2", 112), ("LLaMA3-8B", 112), ("ResNet-50", 1), ("VGG-16", 1)]:
        v, oom = {}, {}
        for g in (1, 2):
            v[g], d = stage(ext / f"{m}_g{g}_defer.log")
            oom[g] = (m == "LLaMA3-8B" and g == 1) or bool(d and d["oom"])
        v[4] = stage(out / f"{m}_defer.log")[0]
        f = lambda x, g: "OOM" if x is None and oom.get(g) else ("--" if x is None else (f"{x / u:.3f}" if u > 1 else f"{x:.2f}"))
        base = v[1] or v[2]
        sp = (f"{v[1] / v[4]:.2f}$\\times$" if v[1] else f"{v[2] / v[4]:.2f}$\\times^*$") if base and v[4] else "--"
        scl.append(f"{m:9} & {f(v[1], 1)} & {f(v[2], 2)} & {f(v[4], 4)} & {sp}\\\\")
    seq = []
    for m in ("GPT-2", "LLaMA3-8B"):
        if seq:
            seq.append("\\midrule")
        cols = {}
        for s_ in (32, 64, 128, 256):
            path = out / f"{m}_defer.log" if s_ == 128 else ext / f"{m}_seq{s_}_defer.log"
            x, d = stage(path)
            cols[s_] = (x, d, s_ - 16)
        row = lambda lab, fn: f"{m} {lab} & " + " & ".join(fn(*cols[k]) for k in (32, 64, 128, 256)) + "\\\\"
        seq.append(row("Prover (s)", lambda x, d, t: f"{x / t:.3f}" if x else "--"))
        seq.append(row("Verifier (s)", lambda x, d, t: f"{d['verify'] / t:.3f}" if x else "--"))
        seq.append(row("Proof (MiB)", lambda x, d, t: f"{d['proof'] / t:.1f}" if x else "--"))

    for title, rows in [("tab:e2e", e2e), ("tab:defer", dfr), ("tab:breakdown", brk), ("tab:prior_systems", pri),
                        ("tab:convbind", conv), ("tab:scaling", scl), ("tab:seqscale", seq), ("tab:offline", off),
                        ("tab:bytes_mono", bytes_mono), ("tab:bytes_prior", bytes_prior), ("tab:bytes_conv", bytes_conv)]:
        print(f"% ---- {title}")
        print("\n".join(rows))


if __name__ == "__main__":
    main()

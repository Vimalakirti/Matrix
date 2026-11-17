#!/usr/bin/env python3
"""DeepProve vs zk-torch-4 table from run_all.sh output.

  make_table.py <out_dir> [base_build]  ->  <out_dir>/comparison.md and .tex

zk-torch-4 CSVs directly in <out_dir> are labelled `base_build`; CSVs in a
subdirectory are labelled with the subdirectory name (e.g. a commit hash).

Everything is per GENERATED token (the paper's unit), plus the per-request
totals beside it. zk-torch-4 rows follow the paper's prover definition:
prove + accumulate + online witness commitment, plus the deferred finalize
spread over N requests. DeepProve has no cross-request amortization, so the
like-for-like single-request number is zk-torch-4 at N=1 (the whole finalize
charged to one request); N=8 is what the harness measured and N->inf is the
paper's amortized limit.
"""
import csv, re, statistics, sys
from collections import defaultdict
from pathlib import Path

NAMES = {"gpt2": "GPT-2", "llama2": "LLaMA2-7B",
         "GPT-2 AR": "GPT-2", "Llama-2-7B AR": "LLaMA2-7B"}


def model_name(key, layers=None):
    """Paper name; truncated runs get their layer count, e.g. LLaMA2-7B (8L)."""
    m = re.fullmatch(r"(\w+?)-L(\d+)", key)
    if m:
        key, layers = m.group(1), int(m.group(2))
    full = {"GPT-2": 12, "LLaMA2-7B": 32}
    name = NAMES.get(key, key)
    return name if layers in (None, full.get(name)) else f"{name} ({layers}L)"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def deepprove_rows(out):
    p = out / "deepprove.csv"
    if not p.exists():
        return []
    groups = defaultdict(list)
    for r in csv.DictReader(open(p)):
        groups[(r["model"], r["backend"], int(r["prompt"]), int(r["total"]))].append(r)
    rows = []
    for (model, backend, prompt, total), rs in groups.items():
        ok = [r for r in rs if r["status"] == "ok"]
        gen = total - prompt
        hw = "96-core CPU" if backend == "cpu" else "1xA100 + CPU"
        if not ok:
            rows.append(dict(model=model_name(model), gen=gen, system=f"DeepProve ({backend})",
                             hw=hw, note=rs[-1]["status"]))
            continue
        prove = med(fnum(r["prove_s"]) for r in ok)
        ver = med(fnum(r["verify_s"]) for r in ok)
        rows.append(dict(
            model=model_name(model), gen=gen, system=f"DeepProve ({backend})", hw=hw,
            req_prove=prove, req_verify=ver,
            tok_prove=prove / gen, tok_verify=ver / gen if ver else None,
            proof_mib=med(fnum(r["proof_MiB"]) for r in ok),
            host_gib=med(fnum(r["host_peak_GiB"]) for r in ok),
            gpu_gib=med(fnum(r["gpu_peak_GiB"]) for r in ok),
            offline=med(fnum(r["context_gen_s"]) for r in ok),
            witgen=med(fnum(r["inference_s"]) for r in ok),
            note=f"n={len(ok)}" + (f", {len(rs)-len(ok)} failed" if len(rs) > len(ok) else "")))
    return rows


def zkt_forward_s(log):
    """Steady-state witness generation per request: mean of the "[ i/N] run"
    lines in the bench log, skipping request 1 (it carries one-time cache
    fills). The AR harness does not record this in its CSV."""
    try:
        runs = [float(x) for x in re.findall(r"\[\s*\d+/\d+\] run\s+([\d.]+)ms",
                                             Path(log).read_text(errors="replace"))]
    except OSError:
        return None
    runs = runs[1:] or runs
    return sum(runs) / len(runs) / 1000.0 if runs else None


def zkt_rows(out, base_build):
    """zk-torch-4 CSVs directly in <out> belong to `base_build`; CSVs in a
    subdirectory belong to the build that subdirectory is named after."""
    rows = []
    paths = sorted(out.glob("zkt_*gpu_*.csv")) + sorted(out.glob("*/zkt_*gpu_*.csv"))
    for p in paths:
        if p.name.endswith(".raw.csv"):
            continue
        build = base_build if p.parent == out else p.parent.name
        ngpu = p.name.split("_")[1]
        for r in csv.DictReader(open(p)):
            gen = int(r["shape"].split("=")[1]) - 16
            env = dict(kv.split("=", 1) for kv in r["env"].split())
            model = model_name(r["group"], int(env.get("NUM_LAYERS", 0)) or None)
            sys_ = f"zk-torch-4@{build} ({ngpu})"
            hw = f"{ngpu.replace('gpu', '')}xA100"
            if r["verified"] != "True":
                rows.append(dict(model=model, gen=gen, system=sys_, hw=hw,
                                 note=r["failure"] or "not verified"))
                continue
            s = lambda k: (fnum(r[k]) or 0.0) / 1000.0
            n = int(env["N_PROOFS"])
            base = s("stream_prove_ms") + s("stream_acc_ms") + s("commit_online_ms")
            fin, fin_ver = s("stream_finalize_ms"), s("stream_finalize_verify_ms")
            ver_u = s("stream_verify_ms")
            pu = (fnum(r["stream_proof_bytes"]) or 0) / 2**20
            pf = (fnum(r["stream_proof_fin_bytes"]) or 0) / 2**20
            for label, k in (("N=1", 1), (f"N={n}", n), ("N->inf", None)):
                req = base + (fin / k if k else 0.0)
                rv = ver_u + (fin_ver / k if k else 0.0)
                rows.append(dict(
                    model=model, gen=gen, system=f"{sys_} {label}", hw=hw,
                    req_prove=req, req_verify=rv, tok_prove=req / gen, tok_verify=rv / gen,
                    proof_mib=pu + (pf / k if k else 0.0),
                    gpu_gib=(fnum(r["peak_mem_mib"]) or 0) / 1024,
                    offline=s("commit_offline_ms"), witgen=zkt_forward_s(r["log"]),
                    note=" ".join([f"finalize {fin:.1f}s/{fin_ver:.2f}s verify"]
                                  + [f"{k}={v}" for k, v in env.items() if k.startswith("AGL_")])))
    return rows


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def main():
    out = Path(sys.argv[1])
    base_build = sys.argv[2] if len(sys.argv) > 2 else "base"
    rows = deepprove_rows(out) + zkt_rows(out, base_build)
    rows.sort(key=lambda r: (r["model"], r["gen"], not r["system"].startswith("DeepProve"),
                             r["system"]))
    hdr = ("| Model | Gen | System | HW | Prove/req (s) | Prove/tok (s) | Verify/req (s) "
           "| Proof (MiB) | Peak GPU (GiB) | Peak host (GiB) | Offline commit (s) "
           "| Witness gen/req (s) | Note |")
    md = [hdr, "|" + "---|" * 13]
    for r in rows:
        md.append("| " + " | ".join([
            r["model"], str(r["gen"]), r["system"], r["hw"],
            fmt(r.get("req_prove"), 1), fmt(r.get("tok_prove"), 3), fmt(r.get("req_verify"), 2),
            fmt(r.get("proof_mib"), 1), fmt(r.get("gpu_gib"), 1), fmt(r.get("host_gib"), 1),
            fmt(r.get("offline"), 1), fmt(r.get("witgen"), 1), r.get("note", "")]) + " |")
    (out / "comparison.md").write_text("\n".join(md) + "\n")

    # LaTeX: per generated token, P16 G16 only (the prior-work table's workload).
    tex = []
    for r in rows:
        if r["gen"] != 16 or r.get("tok_prove") is None:
            continue
        sysname = re.sub(r"N=(\d+)", r"$N{=}\1$", r["system"])
        sysname = (sysname.replace("N->inf", r"$N{\to}\infty$")
                   .replace("zk-torch-4", r"\zkt{}"))
        tex.append(f"{sysname} & {r['model']} & {r['tok_prove']:.3f} & "
                   f"{r['tok_verify']:.3f} & {r['proof_mib']:.1f}\\\\")
    (out / "comparison.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

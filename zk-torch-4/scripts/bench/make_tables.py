#!/usr/bin/env python3
"""Turn benchmark CSVs into LaTeX tables for the evaluation section.

    python3 make_tables.py out/                 # every CSV it recognizes
    python3 make_tables.py out/e2e.csv          # just one

Writes a .tex next to each .csv. The output uses booktabs, which the paper
already loads, and \\zkt for the system name. Tables are emitted with the
column set each paper subsection needs; edit TABLES below rather than the
generator.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------- formatting

def sec(v) -> str:
    """Milliseconds to a paper-ready duration."""
    if v in ("", None):
        return "---"
    v = float(v)
    if v < 1:
        return f"{v*1e3:.0f}\\,\\textmu s"
    if v < 1000:
        return f"{v:.0f}\\,ms"
    return f"{v/1e3:.2f}\\,s"


def mib(v) -> str:
    if v in ("", None) or float(v) == 0:
        return "---"
    v = float(v)
    return f"{v/1024:.1f}\\,GiB" if v >= 1024 else f"{v:.0f}\\,MiB"


def size(v) -> str:
    if v in ("", None):
        return "---"
    v = float(v)
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if v >= div:
            return f"{v/div:.1f}\\,{unit}"
    return f"{v:.0f}\\,B"


def integer(v) -> str:
    return "---" if v in ("", None) else f"{int(float(v))}"


def check(v) -> str:
    if isinstance(v, str):
        v = v.strip().lower() == "true"
    return r"\checkmark" if v else r"$\times$"


def raw(v) -> str:
    return "---" if v in ("", None) or v == "" else str(v).replace("_", r"\_")


# ------------------------------------------------------------- table recipes

Col = tuple  # (header, csv_column, formatter)

TABLES: Dict[str, dict] = {
    "e2e": {
        "caption": (r"End-to-end \zkt proving cost. Offline commitment is the "
                    r"one-time weight commitment, amortized over all proofs of "
                    r"the model. Medians of the reported repetitions."),
        "label": "tab:e2e",
        "cols": [("Model", "name", raw),
                 ("Forward", "forward_ms", sec),
                 ("Commit (offline)", "commit_offline_ms", sec),
                 ("Commit (online)", "commit_online_ms", sec),
                 ("Prove", "prove_ms", sec),
                 ("Verify", "verify_ms", sec),
                 ("Proof", "proof_bytes", size),
                 ("Peak GPU", "peak_mem_mib", mib),
                 (r"Ver.", "verified", check)],
    },
    "multigpu": {
        "caption": (r"Multi-GPU scaling of the \zkt prover. Each row fixes a "
                    r"model and varies the number of GPUs; partitions are set "
                    r"to match the device count."),
        "label": "tab:multigpu",
        "cols": [("Model / GPUs", "name", raw),
                 ("Partitions", "partitions", integer),
                 ("Boundary claims", "boundaries", integer),
                 ("Prove", "prove_ms", sec),
                 ("Verify", "verify_ms", sec),
                 ("Peak GPU", "peak_mem_mib", mib),
                 (r"Ver.", "verified", check)],
    },
    "sparsity": {
        "caption": (r"Sparsity-aware opening versus the dense baseline. "
                    r"\emph{oom} marks configurations the dense path cannot "
                    r"complete on the same hardware."),
        "label": "tab:sparsity",
        "cols": [("Configuration", "name", raw),
                 ("Prove", "prove_ms", sec),
                 ("Peak GPU", "peak_mem_mib", mib),
                 ("Outcome", "failure", lambda v: raw(v) if v else r"\checkmark")],
    },
    "deferred": {
        "caption": (r"Deferred weight opening. The per-proof cost includes the "
                    r"accumulator update; the one-time finalize is the single "
                    r"terminal opening shared by the whole stream."),
        "label": "tab:deferred",
        "cols": [("Configuration", "name", raw),
                 ("Prove (defer)", "stream_prove_ms", sec),
                 ("Acc. update", "stream_acc_ms", sec),
                 ("Finalize / N", "stream_finalize_n_ms", sec),
                 ("Per-proof total", "stream_perproof_ms", sec),
                 ("Finalize (one-time)", "stream_finalize_ms", sec),
                 (r"Ver.", "verified", check)],
    },
}


def emit(rows: List[dict], spec: dict) -> str:
    cols = spec["cols"]
    align = "l" + "r" * (len(cols) - 1)
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           f"\\caption{{{spec['caption']}}}",
           f"\\label{{{spec['label']}}}",
           f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", r"\toprule",
           " & ".join(h for h, _, _ in cols) + r" \\", r"\midrule"]
    last_group = None
    for row in rows:
        group = row.get("group", "")
        if group and group != last_group and len(set(r.get("group") for r in rows)) > 1:
            out.append(rf"\multicolumn{{{len(cols)}}}{{@{{}}l}}{{\emph{{{raw(group)}}}}} \\")
            last_group = group
        out.append(" & ".join(fmt(row.get(key, "")) for _, key, fmt in cols) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def convert(path: Path) -> Optional[Path]:
    kind = path.stem.split(".")[0]
    spec = TABLES.get(kind)
    if spec is None:
        print(f"skip {path.name}: no recipe for '{kind}' "
              f"(known: {', '.join(TABLES)})")
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"skip {path.name}: empty")
        return None
    tex = path.with_suffix(".tex")
    tex.write_text(emit(rows, spec))
    print(f"wrote {tex}  ({len(rows)} rows)")
    return tex


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    targets: List[Path] = []
    for a in argv:
        p = Path(a)
        if p.is_dir():
            targets += sorted(q for q in p.glob("*.csv")
                              if not q.name.endswith(".raw.csv"))
        else:
            targets.append(p)
    if not targets:
        print("no CSVs found")
        return 1
    for t in targets:
        convert(t)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

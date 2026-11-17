#!/usr/bin/env python3
"""Write paper_tables.py rows into the S&P'27 paper repository.

    python3 apply_tables.py <paper_repo> [out_dir]

Replaces the body (first \\midrule after the table's \\label up to its
\\bottomrule) of every generated table in tex/result.tex and
supplement/secondary_results.tex.
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
paper = Path(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else str(here / "out_v3")
text = subprocess.run([sys.executable, str(here / "paper_tables.py"), out], capture_output=True, text=True, check=True).stdout
sec, cur = {}, None
for line in text.splitlines():
    if line.startswith("% ---- "):
        cur = line[7:]
        sec[cur] = []
    elif cur:
        sec[cur].append(line)
for f in ("tex/result.tex", "supplement/secondary_results.tex"):
    p = paper / f
    s = p.read_text()
    for label, rows in sec.items():
        i = s.find("\\label{" + label + "}")
        if i < 0 or not rows:
            continue
        a = s.index("\\midrule\n", i) + len("\\midrule\n")
        b = s.index("\\bottomrule", a)
        s = s[:a] + "\n".join(rows) + "\n" + s[b:]
        print(f"{f}: {label} ({len(rows)} rows)")
    p.write_text(s)

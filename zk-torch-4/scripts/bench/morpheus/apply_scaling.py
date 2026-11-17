#!/usr/bin/env python3
"""Write the measured scaling tables into the S&P'27 evaluation supplement
(supplement/secondary_results.tex), from paper_tables.py output."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
paper = Path(sys.argv[1])
rows = subprocess.run([sys.executable, str(here / "paper_tables.py"), str(here / "out_final2")], capture_output=True, text=True, check=True).stdout
sec, cur = {}, None
for line in rows.splitlines():
    if line.startswith("% ---- "):
        cur = line[7:]
        sec[cur] = []
    elif cur:
        sec[cur].append(line)

p = paper / "supplement/secondary_results.tex"
s = p.read_text()
i = s.find("\\subsection{Planned GPU and Sequence-Length Scaling}")
if i < 0:
    i = s.find("\\subsection{GPU and Sequence-Length Scaling}")
j = s.find("\\end{table}", s.find("\\label{tab:seqscale}")) + len("\\end{table}")
new = r"""\subsection{GPU and Sequence-Length Scaling}
\label{sec:eval_scaling}

\minihead{Multi-GPU scaling}
Table~\ref{tab:scaling} reports strong scaling on one, two and four A100
GPUs with deferred opening and the finalization amortized away. The
workload, parameters and accounting stay fixed across GPU counts. The link
and the source commitments spread their sources over the devices, and
authentication attempts or shares run on every device. The graph PIOP and
the host-side link preparation run on one device and the CPU, so they bound
the speedup. VGG-16 is too small to gain from more devices. LLaMA3-8B does
not fit on one GPU, which must hold the source key cache and every source.

\begin{table}[t]
\centering
\caption{Multi-GPU scaling on A100 GPUs. Transformer values are seconds
per generated token; CNN values are seconds per input. SpUp compares one and
four GPUs ($^*$: two and four GPUs).}
\label{tab:scaling}
\small
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
Model & 1 GPU & 2 GPU & 4 GPU & SpUp\\
\midrule
""" + "\n".join(sec["tab:scaling"]) + r"""
\bottomrule
\end{tabular*}
\end{table}

\minihead{Sequence-length scaling}
\phantomsection
\label{sec:eval_seq}
Table~\ref{tab:seqscale} varies the total sequence length over
$32,64,128,256$ with the prompt fixed at $16$ tokens, so the requests
generate $16,48,112,240$ tokens. Each value is a complete deferred request
divided by its generated tokens. The per-token prover cost falls with the
sequence length because the authentication, the masks and the link setup
are paid once per request. The proof is dominated by the authentication
responses, whose size does not depend on the request, so the proof per token
also falls.

\begin{table}[t]
\centering
\caption{Sequence-length scaling on four A100 GPUs. Columns are total
sequence lengths. Prover and verifier times are seconds per generated
token, and proof sizes MiB per generated token.}
\label{tab:seqscale}
\scriptsize
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
& $32$ & $64$ & $128$ & $256$\\
\midrule
""" + "\n".join(sec["tab:seqscale"][:3]) + "\n\\midrule\n" + "\n".join(sec["tab:seqscale"][3:]) + r"""
\bottomrule
\end{tabular*}
\end{table}"""
s = s[:i] + new + s[j:]
p.write_text(s)
print("\n".join(sec["tab:scaling"] + sec["tab:seqscale"]))

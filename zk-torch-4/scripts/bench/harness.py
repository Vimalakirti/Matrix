"""Benchmark harness for the Morpheus evaluation section.

Runs a prover binary under a given environment, parses the metric lines it
prints, optionally samples GPU memory while it runs, repeats for medians, and
writes tidy CSV.

Nothing here is Morpheus-specific beyond the metric regexes: the bins print
`Duration` values with `{:?}`, so a value can arrive as "4.53s", "553.2ms",
"982.08us" or "12ns", and every parser normalizes to milliseconds.

Usage from a driver script:

    from harness import Case, run_cases, write_csv
    cases = [Case(name="gpt2-12L", binary="gpt2",
                  env={"NUM_LAYERS": "12", "SEQ_LEN": "64"})]
    rows = run_cases(cases, reps=3)
    write_csv(rows, "out/e2e.csv")
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]          # .../zk-torch-4
RELEASE = REPO / "target" / "release"
LOGDIR = Path(os.environ.get("BENCH_LOGDIR", REPO / "scripts" / "bench" / "logs"))

# --------------------------------------------------------------------------
# duration parsing
# --------------------------------------------------------------------------

_UNIT_MS = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1e3,
            "m": 6e4, "h": 3.6e6}
_DUR = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(ns|µs|us|ms|s|m|h)(?![a-zA-Z])")


def parse_duration_ms(text: str) -> Optional[float]:
    """Parse a Rust `{:?}`-formatted Duration into milliseconds.

    Rust prints compound values for large durations ("1m30.5s"), so all
    matched components are summed.
    """
    parts = _DUR.findall(text)
    if not parts:
        return None
    return sum(float(v) * _UNIT_MS[u] for v, u in parts)


# --------------------------------------------------------------------------
# metric extraction
#
# Each entry maps a CSV column to (regex, kind). `kind` is "dur" for a Rust
# Duration, "int" for a bare integer, "bool" for a yes/no marker. Add rows
# here rather than editing drivers.
# --------------------------------------------------------------------------

METRICS = {
    "compile_ms":        (re.compile(r"^\s*Compile:\s*(.+)$", re.M), "dur"),
    "weightgen_ms":      (re.compile(r"^\s*Weight gen:\s*(.+)$", re.M), "dur"),
    "forward_ms":        (re.compile(r"^\s*Forward:\s*(.+)$", re.M), "dur"),
    "commit_offline_ms": (re.compile(r"^\s*(?:Commit \(offline[^)]*\)|Offline commit \([^)]*\)):\s*(.+)$", re.M), "dur"),
    "commit_online_ms":  (re.compile(r"^\s*Commit \(online[^)]*\):\s*(.+)$", re.M), "dur"),
    # The one-shot AR binaries do not print a "Commit (online)" line; they
    # report the per-generation commit inside the per-iteration line
    #   [ 1/2] run 103006.5ms commit 12519.1ms prove 36962.3ms ...
    # Without this the AR rows recorded commit_online_ms = 0, which silently
    # dropped a real cost: 12.5s against 37.0s of prove on Llama-2, about a
    # third again on top of proving. Averaged over iterations below, like the
    # per-iteration verify already is.
    "prove_ms":          (re.compile(r"^\s*Prove:\s*(.+)$", re.M), "dur"),
    "verify_ms":         (re.compile(r"^\s*Verify:\s*(.+)$", re.M), "dur"),
    "proof_bytes":       (re.compile(r"^\s*Proof size:\s*([0-9]+)", re.M), "int"),
    "nodes":             (re.compile(r"\((\d+) nodes,", re.M), "int"),
    "edges":             (re.compile(r",\s*(\d+) edges\)", re.M), "int"),
    "partitions":        (re.compile(r"^\s*Partitions:\s*(\d+)", re.M), "int"),
    "boundaries":        (re.compile(r"boundaries:\s*(\d+)", re.M), "int"),
    # Streaming bins print ms floats with a per-bin noun (per-proof, per-inf,
    # per-img, per-vol, per-gen), and not every bin prints every line, so each
    # component is captured separately and the driver sums what is missing.
    "stream_perproof_ms":  (re.compile(r"^\s*=\s*streaming per-\S+\s*(?:total\s*)?:\s*([\d.]+)\s*ms", re.M), "float"),
    "stream_prove_ms":     (re.compile(r"^\s*prove\s*\(defer\)[^:]*:\s*([\d.]+)\s*ms", re.M), "float"),
    "stream_acc_ms":       (re.compile(r"^\s*\+?\s*acc-update[^:]*:\s*([\d.]+)\s*ms", re.M), "float"),
    "stream_finalize_n_ms":(re.compile(r"^\s*\+?\s*finalize / N[^:]*:\s*([\d.]+)\s*ms", re.M), "float"),
    "stream_finalize_ms":  (re.compile(r"^\s*finalize (?:\(one-time\)|prove)\s*:\s*([\d.]+)\s*ms", re.M), "float"),
    # The one-time finalize verify. Two bins print it on its own line, the
    # other eleven inline as "(+verify N ms)" beside the finalize prove time,
    # so both prefixes feed one capture group. This is the cost that makes a
    # streamed batch sound; without it the per-proof verify is only half the
    # verifier's work.
    "stream_finalize_verify_ms": (re.compile(
        r"(?:\(\+\s*verify\s+|^\s*finalize verify\s*:\s*)([\d.]+)\s*ms", re.M), "float"),
    "stream_proof_bytes":  (re.compile(r"^\s*proof\s+per-unit\s*:\s*(\d+)\s*bytes", re.M), "float"),
    "stream_proof_fin_bytes":(re.compile(r"^\s*proof\s+finalize\s*:\s*(\d+)\s*bytes", re.M), "float"),
    # --- proof-size breakdown, printed by the streaming bins ---
    # DagProof separates these itself, so the split is exact, not estimated.
    "proof_sc_nonlookup_mb": (re.compile(r"sumcheck non-lookup\s*:\s*([\d.]+) MB", re.M), "float"),
    "proof_sc_lookup_mb":    (re.compile(r"sumcheck lookup\s*:\s*([\d.]+) MB", re.M), "float"),
    "proof_pcs_mb":          (re.compile(r"PCS \(fold tree\)\s*:\s*([\d.]+) MB", re.M), "float"),
    "proof_node_mb":         (re.compile(r"\(node ([\d.]+) \+", re.M), "float"),
    "proof_reducer_mb":      (re.compile(r"\+ reducer/edge ([\d.]+)\)", re.M), "float"),
    # --- prover-time breakdown, ZK4_TIMING=1 (stderr) ---
    # Same three buckets as the proof split, so time and size are comparable.
    # The fold tree prints ms or s depending on magnitude, so parse it as a
    # duration rather than assuming a unit.
    "t_fold_tree_ms":    (re.compile(r"\[prove\] fold tree:\s*([\d.]+\s*[a-zµ]*s)", re.M), "dur"),
    "t_lookup_range":    (re.compile(r"\[prove\] lookup proofs:.*?range=([\d.]+\s*[a-zµ]*s)", re.M), "dur"),
    "t_lookup_two_pow":  (re.compile(r"\[prove\] lookup proofs:\s*two_pow=([\d.]+\s*[a-zµ]*s)", re.M), "dur"),
    "t_node_prove_ms":   (re.compile(r"node-prove\s+([\d.]+)ms", re.M), "float"),
    # Anchored to the "(N reducers)" suffix of the partitioned backward's
    # thread-time line. The bare form also matched "opening_reducer 150ms",
    # silently reporting the opening reducer twice under two names.
    "t_reducer_ms":      (re.compile(r"reducer\s+([\d.]+)ms\s*\(\d+\s+reducers\)", re.M), "float"),
    # The claim-merge that deferral pays for: N repeated weight claims on one
    # edge folded into a single opening.
    "t_opening_reducer": (re.compile(r"\[prove\] opening reducer:\s*(\d+) edges in ([\d.]+\s*[a-zµ]*s)", re.M), "dur2"),
    "n_opening_edges":   (re.compile(r"\[prove\] opening reducer:\s*(\d+) edges", re.M), "int"),
    # stderr, with ZK4_TIMING=1: confirms the device pool actually took effect
    "fold_gpus":           (re.compile(r"\[fold_tree\] scheduler:.*?x\s*(\d+)\s*GPUs", re.M), "int"),
}

# gpt2/llama2/oneshot_gpt2 print "Verified (modulo N deferred constant
# claims ...): true" when weights are deferred. That is NOT a standalone
# verify: it is sound only once the streaming finalize runs, so it is tracked
# separately rather than counted as a clean pass.
_STREAM_ITER_VERIFY = re.compile(
    r"^\s*\[\s*\d+/\s*\d+\].*?\bverify\s+([\d.]+)ms", re.M)
_STREAM_ITER_COMMIT = re.compile(
    r"^\s*\[\s*\d+/\s*\d+\].*?\bcommit\s+([\d.]+)ms", re.M)
_VERIFIED = re.compile(r"^\s*Verified\b[^:]*:\s*(true|false)", re.M)
_VERIFIED_MODULO = re.compile(r"^\s*Verified\s*\(modulo", re.M)
# The 11 compact streaming bins hardcode "Verified: true" and signal failure by
# returning early, so the results block must also be present.
_STREAM_RESULTS = re.compile(r"^\s*(?:===\s*Results|Stream summary)", re.M)

# A run is a failure if any of these appear, even with exit code 0.
_FAILURE_PATTERNS = [
    (re.compile(r"out of memory|OutOfMemory|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation", re.I), "oom"),
    (re.compile(r"^thread '.*' panicked", re.M), "panic"),
    (re.compile(r"Killed|signal: 9", re.I), "killed"),
    (re.compile(r"WILL fail to verify", re.I), "range_table_overflow"),
    (re.compile(r"verify failed at|verifier rejected|verify_finalize REJECTED", re.I), "verify_failed"),
]


# --------------------------------------------------------------------------
# GPU memory sampling
# --------------------------------------------------------------------------

def _visible_gpu_indices(env: Dict[str, str]) -> Optional[str]:
    """Which physical GPUs to poll, as an nvidia-smi -i argument."""
    for key in ("ZK4_GPU_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if env.get(key):
            return env[key]
    return None


class _MemSampler(threading.Thread):
    """Polls nvidia-smi and records the peak used-memory sum over the GPUs.

    Reports the peak *delta* over the first sample, so memory already held by
    other processes on a shared node does not inflate the number.
    """

    def __init__(self, gpus: Optional[str], interval: float = 0.05):
        super().__init__(daemon=True)
        self.gpus, self.interval = gpus, interval
        self.peak_mib = 0
        self.baseline_mib: Optional[int] = None
        self._done = threading.Event()

    def _sample(self) -> Optional[int]:
        cmd = ["nvidia-smi", "--query-gpu=memory.used",
               "--format=csv,noheader,nounits"]
        if self.gpus:
            cmd += ["-i", self.gpus]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=10).stdout
            return sum(int(x) for x in out.split() if x.strip().isdigit())
        except Exception:
            return None

    def run(self):
        while not self._done.is_set():
            v = self._sample()
            if v is not None:
                if self.baseline_mib is None:
                    self.baseline_mib = v
                self.peak_mib = max(self.peak_mib, v - self.baseline_mib)
            self._done.wait(self.interval)

    def stop(self) -> int:
        self._done.set()
        self.join(timeout=5)
        return self.peak_mib


# --------------------------------------------------------------------------
# cases and runs
# --------------------------------------------------------------------------

@dataclass
class Case:
    name: str                                  # row label in the paper table
    binary: str                                # bin name under target/release
    env: Dict[str, str] = field(default_factory=dict)
    config: Optional[str] = None               # positional yaml argument
    group: str = ""                            # free-form tag for the driver
    timeout_s: int = 3600
    note: str = ""                             # carried into the CSV


@dataclass
class Run:
    name: str
    group: str
    binary: str
    rep: int
    ok: bool
    verified: Optional[bool]
    failure: str
    peak_mem_mib: int
    wall_ms: float
    metrics: Dict[str, float]
    log: str
    env_desc: str


def _classify(stdout: str, stderr: str, code: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    blob = stdout + "\n" + stderr
    for pat, label in _FAILURE_PATTERNS:
        if pat.search(blob):
            return label
    if code != 0:
        return f"exit{code}"
    return ""


def run_once(case: Case, rep: int, sample_mem: bool = True) -> Run:
    exe = RELEASE / case.binary
    if not exe.exists():
        return Run(case.name, case.group, case.binary, rep, False, None,
                   "missing_binary", 0, 0.0, {}, "", _env_desc(case.env))

    env = dict(os.environ)
    env.update(case.env)
    cmd = [str(exe)] + ([case.config] if case.config else [])

    sampler = None
    if sample_mem and shutil.which("nvidia-smi"):
        sampler = _MemSampler(_visible_gpu_indices(env))
        sampler.start()

    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True,
                              text=True, timeout=case.timeout_s)
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        code, timed_out = -1, True
    wall_ms = (time.time() - t0) * 1e3
    peak = sampler.stop() if sampler else 0

    LOGDIR.mkdir(parents=True, exist_ok=True)
    # Include a short hash of the env in the filename. Without it, two runs of
    # the same model under different configs (a tuned run and its baseline)
    # collide on one path and the second silently destroys the first's phase
    # timings — which is exactly how the YOLO tuned-vs-baseline comparison lost
    # its evidence and could not be diagnosed afterwards.
    env_key = hashlib.sha1(
        "\x00".join(f"{k}={v}" for k, v in sorted(case.env.items())).encode()
    ).hexdigest()[:8]
    log = LOGDIR / f"{case.group or 'run'}__{case.name}__{env_key}__r{rep}.log".replace("/", "_")
    log.write_text(f"$ {' '.join(f'{k}={v}' for k, v in case.env.items())} "
                   f"{' '.join(cmd)}\n\n=== stdout ===\n{stdout}\n=== stderr ===\n{stderr}")

    failure = _classify(stdout, stderr, code, timed_out)
    m = _VERIFIED.search(stdout)
    verified = (m.group(1) == "true") if m else None
    if verified and _VERIFIED_MODULO.search(stdout):
        # verified only modulo deferred weight claims: not reportable alone
        failure = failure or "verified_modulo_deferred"
    if case.binary.startswith("bench_streaming") and not _STREAM_RESULTS.search(stdout):
        failure = failure or "streaming_aborted"

    metrics: Dict[str, float] = {}
    scan = stdout + "\n" + stderr        # ZK4_TIMING / mem traces are on stderr
    for col, (pat, kind) in METRICS.items():
        hit = pat.search(scan)
        if not hit:
            continue
        raw = hit.group(1).strip()
        if kind == "dur":
            v = parse_duration_ms(raw)
        elif kind == "dur2":
            # pattern captures (count, duration); we want the duration
            v = parse_duration_ms(hit.group(2).strip())
        elif kind == "float":
            try:
                v = float(raw)
            except ValueError:
                v = None
        else:
            try:
                v = float(raw)
            except ValueError:
                v = None
        if v is not None:
            metrics[col] = v

    # Most streaming bins print the amortized cost only as components
    # ("prove(defer) per-img", "acc-update per-img") and never as a single
    # total line. Without this sum the per-proof column is empty and the
    # summary silently falls back to the non-streaming table, printing "-"
    # for prove on every model that streams. Reconstruct the total here.
    #
    # finalize/N is deliberately NOT added: it is the amortized share of a
    # one-time cost that is sound only once it actually runs, so it stays a
    # separate column rather than being hidden inside the per-proof number.
    # Streaming bins print per-proof verify only inside the per-iteration
    # table, never as a summary line. Average it so the streaming rows report a
    # verifier cost instead of a blank.
    # The one-shot AR binaries never print a "Commit (online)" line; they report
    # the per-generation commit only inside the per-iteration table. Without
    # this the AR rows recorded commit_online_ms = 0, silently dropping a real
    # cost: 12.5s against 37.0s of prove on Llama-2, about a third again on top
    # of proving.
    if "commit_online_ms" not in metrics:
        per_c = _STREAM_ITER_COMMIT.findall(scan)
        if per_c:
            metrics["commit_online_ms"] = sum(float(x) for x in per_c) / len(per_c)

    per_v = _STREAM_ITER_VERIFY.findall(scan)
    if per_v:
        metrics["stream_verify_ms"] = sum(float(x) for x in per_v) / len(per_v)
    # A streaming row's proof size is the per-unit size; the one-time finalize
    # proof is kept separate for the same reason its time is.
    if "proof_bytes" not in metrics and "stream_proof_bytes" in metrics:
        metrics["proof_bytes"] = metrics["stream_proof_bytes"]

    if "stream_perproof_ms" not in metrics and "stream_prove_ms" in metrics:
        metrics["stream_perproof_ms"] = (metrics["stream_prove_ms"]
                                         + metrics.get("stream_acc_ms", 0.0))

    return Run(case.name, case.group, case.binary, rep,
               ok=(not failure and verified is not False),
               verified=verified, failure=failure, peak_mem_mib=peak,
               wall_ms=wall_ms, metrics=metrics, log=str(log),
               env_desc=_env_desc(case.env))


def _env_desc(env: Dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(env.items()))


def run_cases(cases: Sequence[Case], reps: int = 3,
              sample_mem: bool = True, stop_on_fail: bool = False) -> List[Run]:
    runs: List[Run] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.group}/{case.name}", flush=True)
        for rep in range(1, reps + 1):
            r = run_once(case, rep, sample_mem)
            status = "ok" if r.ok else (r.failure or "unverified")
            prove = r.metrics.get("prove_ms")
            print(f"    rep{rep}: {status}"
                  + (f"  prove={prove/1e3:.2f}s" if prove else "")
                  + (f"  peak={r.peak_mem_mib}MiB" if r.peak_mem_mib else ""),
                  flush=True)
            runs.append(r)
            if not r.ok and stop_on_fail:
                return runs
            if not r.ok and r.failure in ("oom", "missing_binary", "timeout"):
                break      # repeating a hard failure wastes time
    return runs


# --------------------------------------------------------------------------
# aggregation and output
# --------------------------------------------------------------------------

# stream_verify_ms is derived from the per-iteration table rather than scraped
# by a METRICS pattern, so it has to be named here or the summary drops it.
ALL_COLS = list(METRICS.keys()) + ["stream_verify_ms"]


# The shape axis a row was run at: sequence length for transformers, batch for
# CNNs, decoder context for whisper. Recorded as its own column so a sweep row
# says what it swept without anyone parsing the env string to find out.
_SHAPE_KEYS = ("SEQ_LEN", "N_TEXT_CTX", "BATCH")


def _shape_of(env_desc: str) -> str:
    kv = dict(p.split("=", 1) for p in env_desc.split() if "=" in p)
    for k in _SHAPE_KEYS:
        if k in kv:
            return f"{k.lower()}={kv[k]}"
    return ""


def summarize(runs: Sequence[Run]) -> List[dict]:
    """Median over reps, one row per (group, name). Failures are preserved."""
    out = []
    seen = []
    for r in runs:
        key = (r.group, r.name)
        if key in seen:
            continue
        seen.append(key)
        reps = [x for x in runs if (x.group, x.name) == key]
        good = [x for x in reps if x.ok]
        row = {
            "group": r.group, "name": r.name, "binary": r.binary,
            "reps": len(reps), "reps_ok": len(good),
            "verified": all(x.verified for x in good) if good else False,
            "failure": "" if good else (reps[0].failure or "unverified"),
            "peak_mem_mib": max((x.peak_mem_mib for x in reps), default=0),
            "shape": _shape_of(reps[0].env_desc),
            "env": reps[0].env_desc, "log": reps[0].log,
        }
        for col in ALL_COLS:
            vals = [x.metrics[col] for x in good if col in x.metrics]
            row[col] = round(statistics.median(vals), 3) if vals else ""
        out.append(row)
    return out


def write_csv(runs: Sequence[Run], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summarize(runs)
    cols = ["group", "name", "shape", "binary", "reps", "reps_ok", "verified",
            "failure", "peak_mem_mib"] + ALL_COLS + ["env", "log"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    raw = path.with_suffix(".raw.csv")
    with raw.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "name", "rep", "ok", "verified", "failure",
                    "peak_mem_mib", "wall_ms"] + ALL_COLS)
        for r in runs:
            w.writerow([r.group, r.name, r.rep, r.ok, r.verified, r.failure,
                        r.peak_mem_mib, round(r.wall_ms, 1)]
                       + [r.metrics.get(c, "") for c in ALL_COLS])
    print(f"\nwrote {path}\nwrote {raw}  (per-rep)")
    return path


def require_release_build(bins: Sequence[str]) -> None:
    """Fail early and loudly rather than reporting missing_binary per case."""
    missing = [b for b in dict.fromkeys(bins) if not (RELEASE / b).exists()]
    if missing:
        args = " ".join(f"--bin {b}" for b in sorted(missing))
        sys.exit(f"missing release binaries: {', '.join(sorted(missing))}\n"
                 f"build them first:\n  cargo build --release {args}")

#!/usr/bin/env python3
"""Run a prover binary and record a per-GPU utilization trace alongside its
own phase timings.

    python3 gpu_trace.py --env NUM_LAYERS=8 --env SEQ_LEN=64 \
        -- ./target/release/llama2 llama2_config.yaml

Answers "are the GPUs actually busy, and during which phase". Samples
`nvidia-smi --query-gpu=utilization.gpu,memory.used` every 100 ms, and
timestamps the binary's own `[prove] ...` / `[fold_tree] ...` phase lines from
ZK4_TIMING=1 so each phase can be attributed a mean utilization per device.

Reports, per phase: wall seconds, mean and peak utilization of each GPU, and
the count of devices whose mean utilization exceeds 5% (the "engaged" count).
A phase with engaged=1 on a 4-GPU box is leaving three devices idle.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import os
import threading
import time

SAMPLE_S = 0.1
NCPU = os.cpu_count() or 1
# Phase markers the prover prints under ZK4_TIMING=1. Each marks the END of the
# named phase, so a phase spans from the previous marker to this one. Two
# caveats the labels account for below: "scheduler" prints at fold-tree START
# (so it terminates leaf build), and the "[prove] leaf build total" /
# "[prove] fold tree" summaries print AFTER the fold tree has already run.
PHASE_RE = re.compile(
    r"^\[(?:prove|fold_tree)\]\s+(partition backward|lookup proofs|"
    r"leaf build|fold tree|opening reducer|commit|scheduler)\b")


class Sampler(threading.Thread):
    """Polls nvidia-smi and the prover's /proc CPU counters until stopped.

    Keeps (t, [util per gpu], cores_busy). `cores_busy` is the number of CPU
    cores the prover consumed over the interval, from utime+stime deltas: 96.0
    on a 96-core box means fully CPU-saturated. Together the two tell a phase
    apart: high GPU = accelerated, high cores / zero GPU = CPU-bound, both low
    = serialized or waiting.
    """

    def __init__(self, gpus: int):
        super().__init__(daemon=True)
        self.gpus = gpus
        self.samples: list[tuple[float, list[int], float]] = []
        self._done = threading.Event()
        self.pid: int | None = None
        self._prev: tuple[float, float] | None = None

    def _cores(self) -> float:
        if self.pid is None:
            return 0.0
        try:
            with open(f"/proc/{self.pid}/stat") as f:
                parts = f.read().split()
            ticks = (int(parts[13]) + int(parts[14])) / os.sysconf("SC_CLK_TCK")
            now = time.time()
            if self._prev is None:
                self._prev = (now, ticks)
                return 0.0
            dt, dticks = now - self._prev[0], ticks - self._prev[1]
            self._prev = (now, ticks)
            return dticks / dt if dt > 0 else 0.0
        except Exception:
            return 0.0

    def run(self) -> None:
        while not self._done.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                vals = [int(x.strip()) for x in out.splitlines() if x.strip()]
                if vals:
                    self.samples.append((time.time(), vals[:self.gpus], self._cores()))
            except Exception:
                pass
            self._done.wait(SAMPLE_S)

    def stop(self) -> None:
        self._done.set()


def visible_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=15).stdout
        return sum(1 for l in out.splitlines() if l.startswith("GPU "))
    except Exception:
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--label", default="run")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- followed by the binary and its args")
    a = ap.parse_args()
    cmd = [c for c in a.cmd if c != "--"]
    if not cmd:
        return print("need a command after --") or 1

    import os
    env = dict(os.environ, ZK4_TIMING="1")
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v

    ngpu = visible_gpus()
    sampler = Sampler(ngpu)
    marks: list[tuple[float, str]] = []
    lines: list[str] = []

    t0 = time.time()
    sampler.start()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    sampler.pid = p.pid
    for line in p.stdout:
        lines.append(line.rstrip())
        m = PHASE_RE.match(line.strip())
        if m:
            marks.append((time.time(), m.group(1)))
    p.wait()
    sampler.stop()
    sampler.join(timeout=5)
    total = time.time() - t0

    print(f"\n=== {a.label}: {ngpu} GPUs, {len(sampler.samples)} samples, "
          f"{total:.1f}s wall ===")

    def window(lo: float, hi: float):
        return [(v, c) for (t, v, c) in sampler.samples if lo <= t < hi]

    def report(name: str, lo: float, hi: float) -> None:
        w = window(lo, hi)
        if not w:
            print(f"  {name:<20} {hi-lo:6.2f}s  (no samples)")
            return
        per = list(zip(*[v for (v, _) in w]))
        means = [sum(c) / len(c) for c in per]
        peaks = [max(c) for c in per]
        engaged = sum(1 for m in means if m > 5.0)
        cores = [c for (_, c) in w if c > 0]
        cpu = sum(cores) / len(cores) if cores else 0.0
        print(f"  {name:<20} {hi-lo:6.2f}s  gpu_mean="
              f"[{' '.join(f'{m:5.1f}' for m in means)}]  engaged={engaged}/{ngpu}"
              f"  cpu={cpu:5.1f}/{NCPU} cores")

    # Windows are named for what elapsed BEFORE each marker, not for the
    # marker itself. "scheduler" prints when the fold tree starts, so the
    # window ending there is the leaf build; the window after it is the fold
    # tree, which the trailing "leaf build"/"fold tree" summaries close out.
    LABEL = {
        "partition backward": "fwd+commit+backward",
        "lookup proofs":      "lookup proofs (CPU)",
        "opening reducer":    "opening reducer",
        "scheduler":          "leaf build",
        "leaf build":         "FOLD TREE",
        "fold tree":          "(fold tree tail)",
    }
    prev = t0
    for (t, name) in marks:
        report(LABEL.get(name, name), prev, t)
        prev = t
    report("(tail)", prev, t0 + total)
    report("WHOLE RUN", t0, t0 + total)

    print("\n--- phase timings as the binary reported them ---")
    for l in lines:
        if l.startswith("[prove]") or l.startswith("[fold_tree]") \
           or l.startswith("Prove") or l.startswith("Verified") \
           or l.startswith("Proof size") or l.startswith("Partitions"):
            print("  " + l)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

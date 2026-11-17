#!/usr/bin/env python3
"""Prove GPT-2 and Llama-2-7B under DeepProve at the workloads Morpheus reports.

DeepProve proves one causal forward pass over prompt + generated tokens and
binds each generated token to the argmax of the logits before it, which is the
same statement the zk-torch-4 one-shot binaries prove (`run_paper_e2e.py
--ar`). So a (prompt P, total T) pair here is the same workload as
`--ar-prompt P --seq T-P` there:

  P16 G16   = --min-user-len 16 --max-context 32    (prior-work table)
  P16 G112  = --min-user-len 16 --max-context 128   (main AR table)

Reported per GENERATED token, prove / (T - P), to match the paper. DeepProve's
own `token/sec` column divides by T and so counts the prompt as output.

Generation stops early if the model emits EOS. The proof then covers fewer
positions than asked, so every row records the iterations actually run and
`full_len` is false when the run fell short. Fix the prompt with RNG_SEED.

Timing columns come straight from DeepProve's bench CSV:
  prove_full       witness commitment + all sumchecks/lookups + PCS openings
  context_gen      offline: quantized weight commitment + PK/VK
  verify_time      added by verify_timing.patch (upstream does not time it)
Host peak is the child's VmHWM; GPU peak is sampled from nvidia-smi.
"""
import argparse, csv, os, re, subprocess, sys, tempfile, threading, time
from pathlib import Path

MODELS = {
    "gpt2":   ("gpt2",   "openai-community/gpt2"),
    "llama2": ("llama2", "meta-llama/Llama-2-7b-hf"),
}

FIELDS = ["model", "backend", "gpu", "threads", "prompt", "total", "generated",
          "iters_run", "full_len", "rep", "seed",
          "quant_s", "context_gen_s", "inference_s", "prove_s", "verify_s",
          "prove_per_gen_token_s", "verify_per_gen_token_s",
          "proof_bytes", "proof_MiB", "host_peak_GiB", "gpu_peak_GiB",
          "status", "log"]


def gpu_mem_mib(pid):
    """Memory nvidia-smi attributes to `pid` (0 when not listed)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return 0
    tot = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            tot += int(parts[1])
    return tot


def resolve(model):
    """`gpt2`, `llama2`, or a truncation such as `llama2-L8` (first 8 decoder
    blocks at full width; prepare_models.py --truncate creates it)."""
    m = re.fullmatch(r"(\w+?)-L(\d+)", model)
    if not m:
        return MODELS[model]
    tag, repo = MODELS[m.group(1)]
    return tag, f"trunc/{repo.split('/')[-1]}-L{m.group(2)}"


def run_one(a, model, prompt, total, rep, log_dir):
    tag, repo = resolve(model)
    seed = a.seed + rep
    bench_csv = Path(tempfile.mkstemp(prefix=f"dp_{model}_", suffix=".csv",
                                      dir=a.run_dir)[1])
    bench_csv.unlink()
    cmd = [a.bin, "--model", tag, "--hf", repo,
           "--max-context", str(total), "--min-user-len", str(prompt),
           "--bench", str(bench_csv)]
    if a.threads:
        cmd += ["--num-threads", str(a.threads)]
    env = dict(os.environ, RNG_SEED=str(seed),
               RUST_LOG=os.environ.get("RUST_LOG", "info"))
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = a.gpu
    log = log_dir / f"{model}_{a.backend}_P{prompt}_T{total}_r{rep}.log"
    print(f"[run] {' '.join(cmd)}  (seed {seed}) -> {log}", flush=True)

    hwm_kb, gpu_peak, t0 = 0, 0, time.time()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=a.run_dir, env=env, stdout=fh,
                                stderr=subprocess.STDOUT)
        stop = threading.Event()

        def sample_gpu():
            nonlocal gpu_peak
            while not stop.is_set():
                gpu_peak = max(gpu_peak, gpu_mem_mib(proc.pid))
                stop.wait(1.0)
        if a.gpu is not None:
            threading.Thread(target=sample_gpu, daemon=True).start()
        while proc.poll() is None:
            try:
                with open(f"/proc/{proc.pid}/status") as st:
                    m = re.search(r"VmHWM:\s+(\d+)", st.read())
                    if m:
                        hwm_kb = max(hwm_kb, int(m.group(1)))
            except OSError:
                pass
            if time.time() - t0 > a.timeout:
                proc.kill()
                break
            time.sleep(0.5)
        proc.wait()
        stop.set()

    # DeepProve rewrites ./bench-llm.csv (per-function timings) every run; keep
    # each run's copy next to its log.
    per_fn = Path(a.run_dir) / "bench-llm.csv"
    if per_fn.exists():
        per_fn.rename(log.with_suffix(".functions.csv"))
    text = log.read_text(errors="replace")
    iters = [int(x) for x in re.findall(r"Running iteration (\d+)", text)]
    row = dict(model=model, backend=a.backend, gpu=a.gpu or "", threads=a.threads or "all",
               prompt=prompt, total=total, generated=total - prompt,
               iters_run=(max(iters) if iters else ""), rep=rep, seed=seed,
               host_peak_GiB=round(hwm_kb / 2**20, 2),
               gpu_peak_GiB=round(gpu_peak / 1024, 2) if a.gpu is not None else "",
               log=str(log))
    if proc.returncode != 0 or not bench_csv.exists():
        row["status"] = ("timeout" if time.time() - t0 > a.timeout
                         else f"exit {proc.returncode}")
        return row
    with open(bench_csv) as fh:
        r = list(csv.DictReader(fh))[-1]
    bench_csv.unlink()
    ms = lambda k: float(r[k]) / 1000.0 if r.get(k) not in (None, "") else None
    gen = total - prompt
    row.update(quant_s=ms("quantization_time"), context_gen_s=ms("context_generation_time"),
               inference_s=ms("inference_time"), prove_s=ms("prove_full"),
               verify_s=ms("verify_time"), proof_bytes=int(r["proof_size"]),
               proof_MiB=round(int(r["proof_size"]) / 2**20, 3))
    row["prove_per_gen_token_s"] = round(row["prove_s"] / gen, 4)
    if row["verify_s"] is not None:
        row["verify_per_gen_token_s"] = round(row["verify_s"] / gen, 4)
    # The trace is regenerated at the last iteration, so the proven length is
    # the final iteration index; anything below `total` means EOS cut it short.
    row["full_len"] = bool(iters) and max(iters) >= total
    row["status"] = "ok" if row["full_len"] else "short (EOS)"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to DeepProve's bench-llm")
    ap.add_argument("--backend", required=True, help="label: cpu | cuda")
    ap.add_argument("--run-dir", required=True,
                    help="cwd for bench-llm; must hold model_cache/ (prepare_models.py)")
    ap.add_argument("--models", default="gpt2,llama2")
    ap.add_argument("--configs", default="16:32,16:128",
                    help="comma-separated PROMPT:TOTAL pairs")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for the cuda build")
    ap.add_argument("--threads", type=int, default=None, help="rayon threads (default: all)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.run_dir = str(Path(a.run_dir).resolve())
    a.bin = str(Path(a.bin).resolve())

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for model in a.models.split(","):
            for cfg in a.configs.split(","):
                prompt, total = map(int, cfg.split(":"))
                for rep in range(a.reps):
                    row = run_one(a, model, prompt, total, rep, log_dir)
                    w.writerow(row)
                    fh.flush()
                    print(f"[done] {model} {a.backend} P{prompt} T{total} r{rep}: "
                          f"{row['status']} prove={row.get('prove_s')}s "
                          f"verify={row.get('verify_s')}s proof={row.get('proof_MiB')}MiB "
                          f"host={row['host_peak_GiB']}GiB gpu={row['gpu_peak_GiB']}GiB",
                          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

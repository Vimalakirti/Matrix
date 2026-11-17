#!/usr/bin/env python3
"""Full ezkl pipeline for one model/mode with per-stage wall time + peak RSS (VmHWM).
usage: prove.py <label> <ezkl-binary> <model-dir> <gpu|cpu> [backend-dir]
Runs in $EZKL_WORK/runs/<label>/ and writes metrics.json there. The reported
prove and verify times are EZKL's own timers (prove_internal_s,
verify_internal_s); setup (key generation) is timed separately. GPU: device
index $GPU (default 2). PHASE=prep runs only the untimed steps, PHASE=timed
reuses them."""
import json, os, re, subprocess, sys, time, shutil, threading
ROOT = os.environ["EZKL_WORK"]
label, binp, mdir, mode = sys.argv[1:5]
GPU = os.environ.get("GPU", "2")
backend = sys.argv[5] if len(sys.argv) > 5 else None
W = f"{ROOT}/runs/{label}"; os.makedirs(W, exist_ok=True)
for f in ("network.onnx", "input.json"): shutil.copy(f"{mdir}/{f}", W)
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = GPU
env["EZKL_REPO_PATH"] = f"{ROOT}/data"
env["RUST_LOG"] = env.get("RUST_LOG", "info")
if mode == "gpu":
    env["ICICLE_BACKEND_INSTALL_DIR"] = backend; env["ENABLE_ICICLE_GPU"] = "true"
else:
    env.pop("ICICLE_BACKEND_INSTALL_DIR", None); env.pop("ENABLE_ICICLE_GPU", None)
M = {"label": label, "binary": binp, "mode": mode, "backend": backend, "stages": {}}
def gpu_sampler(stop, out):
    while not stop.is_set():
        try:
            s = subprocess.run(["nvidia-smi", "-i", GPU, "--query-gpu=memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            mu, ut = [int(x) for x in s.split(",")]
            out["mem_max"] = max(out.get("mem_max", 0), mu); out["util_max"] = max(out.get("util_max", 0), ut)
            if ut > 0: out["util_nz"] = out.get("util_nz", 0) + 1
            out["n"] = out.get("n", 0) + 1
        except Exception: pass
        stop.wait(1)
def run(name, args, timeout=None):
    log = f"{W}/{name}.log"
    t0 = time.time(); hwm = 0; g = {}; stop = threading.Event()
    th = threading.Thread(target=gpu_sampler, args=(stop, g), daemon=True); th.start()
    with open(log, "w") as fh:
        p = subprocess.Popen([binp] + args, cwd=W, env=env, stdout=fh, stderr=subprocess.STDOUT)
        while p.poll() is None:
            try:
                for line in open(f"/proc/{p.pid}/status"):
                    if line.startswith("VmHWM"): hwm = max(hwm, int(line.split()[1]))
            except Exception: pass
            if timeout and time.time() - t0 > timeout:
                M.setdefault("killed", []).append(f"{name}: timeout {timeout}s"); p.kill(); p.wait(); break
            try:
                cur = int([l for l in open(f"/proc/{p.pid}/status") if l.startswith("VmRSS")][0].split()[1])
            except Exception: cur = 0
            if cur > float(os.environ.get("RSS_LIMIT_GB", "1e9")) * 1048576:
                M.setdefault("killed", []).append(f"{name}: RSS {cur/1048576:.0f} GB > limit"); p.kill(); p.wait(); break
            time.sleep(0.5)
    stop.set(); th.join()
    dt = time.time() - t0
    M["stages"][name] = {"wall_s": round(dt, 3), "peak_rss_gb": round(hwm / 1048576, 2), "rc": p.returncode,
                          "gpu_mem_max_mib": g.get("mem_max"), "gpu_util_max": g.get("util_max"),
                          "gpu_util_nonzero_samples": g.get("util_nz", 0), "gpu_samples": g.get("n", 0)}
    print(name, M["stages"][name], flush=True)
    json.dump(M, open(f"{W}/metrics.json", "w"), indent=1)
    if p.returncode != 0:
        print(open(log).read()[-3000:]); sys.exit(1)
    return open(log).read()
# PHASE=prep: only settings/calibrate/compile/srs (untimed, outside GPU lock); PHASE=timed: skip those
PHASE = os.environ.get("PHASE", "all")
if PHASE != "timed":
    run("gen-settings", ["gen-settings", "-M", "network.onnx"])
    run("calibrate-settings", ["calibrate-settings", "-M", "network.onnx", "-D", "input.json"])
    run("compile-circuit", ["compile-circuit", "-M", "network.onnx", "-S", "settings.json", "--compiled-circuit", "model.compiled"])
    run("get-srs", ["get-srs", "-S", "settings.json"])
else:
    prev = json.load(open(f"{W}/metrics.json"))["stages"]
    M["stages"].update({k: v for k, v in prev.items() if k in ("gen-settings", "calibrate-settings", "compile-circuit", "get-srs")})
M["k"] = json.load(open(f"{W}/settings.json"))["run_args"]["logrows"]; print("k =", M["k"], flush=True)
if PHASE == "prep":
    json.dump(M, open(f"{W}/metrics.json", "w"), indent=1); sys.exit(0)
run("setup", ["setup", "-M", "model.compiled", "--vk-path", "vk.key", "--pk-path", "pk.key"],
    timeout=float(os.environ.get("SETUP_TIMEOUT", "0")) or None)
run("gen-witness", ["gen-witness", "-M", "model.compiled", "-D", "input.json", "-O", "witness.json"])
pl = run("prove", ["prove", "-M", "model.compiled", "-W", "witness.json", "--pk-path", "pk.key", "--proof-path", "proof.json"])
vl = run("verify", ["verify", "-S", "settings.json", "--proof-path", "proof.json", "--vk-path", "vk.key"])
def internal(pat, txt):
    m = re.findall(pat + r" took ([0-9.]+)", txt); return float(m[-1]) if m else None
pr = json.load(open(f"{W}/proof.json"))
M.update(prove_internal_s=internal("proof", pl), verify_internal_s=internal("verify", vl),
         proof_bytes=len(pr["proof"]), verified=bool(re.search(r"verified: true", vl)),
         icicle_lines=[l for l in pl.splitlines() if re.search(r"icicle|GPU|CUDA|device", l, re.I)][:8])
json.dump(M, open(f"{W}/metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in M.items() if k != "stages"}, indent=1))

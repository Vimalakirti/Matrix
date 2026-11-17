# EZKL baseline (GPU build)

[EZKL](https://github.com/zkonduit/ezkl) v22.2.2 built with
`--features icicle` (icicle v2.8 CUDA MSM and NTT, `ENABLE_ICICLE_GPU=true`),
on one A100.

| Workload | Model | Morpheus (`../morpheus/run_models.sh`) |
|---|---|---|
| LeNet-5 | `examples/onnx/lenet_5` | `ezkl` |
| nanoGPT, 8 tokens | `models/nanoGPT_seq8` | `ezkl8` |
| nanoGPT, 64 tokens | `examples/onnx/nanoGPT` | `ezkl` |

`models/nanoGPT_seq8` is EZKL's nanoGPT (4 layers, 4 heads, width 64,
vocabulary 65) exported at sequence length 8 by `gen.py`. The committed
`network.onnx` and `input.json` are the measured files; `gen.py` does not
seed the weights. At length 64, EZKL's key generation exhausts the A100's
80 GB, which the paper reports as OOM.

```bash
./setup.sh <work>                 # private Rust and CUDA 12.6 toolchains, icicle, ezkl
./run_gpu.sh <work>               # lenet5 nanogpt8 nanogpt64 (GPU=2, REPS=1)
```

Reported prove and verify times are EZKL's own timers
(`prove_internal_s`, `verify_internal_s` in `runs/*/metrics.json`). Key
generation (`setup`) and witness generation are timed separately and are
not part of the prove time. Proof bytes are the length of the serialized
proof.

`icicle.patch` points icicle's CUDA runtime bindings at the private CUDA
12.6 toolkit instead of `/usr/local/cuda` (13.1 on this machine, which
icicle v2.8 does not support). It does not change the prover.

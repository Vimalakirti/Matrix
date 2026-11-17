# DeepProve baseline (GPU build)

[DeepProve](https://github.com/Lagrange-Labs/deep-prove) at commit `9d1a53e2`
(v2.0.1 + 1), built with `--features cuda`, on one A100. The paper's
one-GPU comparison uses four DeepProve workloads:

| Workload | DeepProve | Morpheus (`../morpheus/run_models.sh`) |
|---|---|---|
| GPT-2, 32 tokens | `bench-llm --hf openai-community/gpt2 --min-user-len 16 --max-context 32` | `dp1s32` |
| GPT-2, 128 tokens | `... --max-context 128` | `dp1s128` |
| CIFAR-10 LeNet | `bench --onnx cnn-cifar-01.onnx` (its shipped model and inputs) | `dp1cnn` |
| COVID-19 CT CNN | `bench --onnx covid10.onnx` | `dp1cnn` |

The GPT-2 workloads prove one causal pass over the prompt and generated
tokens with a 16-token prompt. The CNNs use DeepProve's own trained ONNX
files (`zkml/assets/scripts`). Morpheus proves the same networks from
fixed-point exports at scale 2^8 (`export_cnn.py`, `bench_deepprove_cnn`).

The COVID model needs two changes before DeepProve can prove it, applied
to both systems by `prepare_cnn.sh`. DeepProve cannot parse its final
Sigmoid, so the model ends at the logit. DeepProve then panics on the
one-element output and fails on two outputs, so the last layer is padded
with nine zero rows to ten outputs.

## Running

```bash
./setup.sh <work>            # clone, patch, build (CPU and CUDA); GPT-2 model cache
./prepare_cnn.sh <work>      # CNN inputs, covid10.onnx, Morpheus fixtures
./run_all.sh <work> <zk-torch-4> <out>   # GPT-2, CUDA build (STAGES=dp_cuda), 3 reps
./run_cnn.sh <work> <out>                # CIFAR and COVID, CUDA build, 3 samples
```

`run_all.sh` writes `deepprove.csv` (prove and verify per run).
`run_cnn.sh` writes DeepProve's bench CSVs (`Proving i`, `Verify i` in ms).
Run baselines alone on the machine. `run_models.sh` waits while any process
holds GPU memory, and a `GPU_LOCK` file next to it holds it off explicitly.

Machine notes: the driver supports CUDA 12.6, the system nvcc is 11.5 and
`/usr/local/cuda` is 13.1. `setup.sh` builds with `CUDARC_CUDA_VERSION=12040`
and puts a private CUDA 12 NVRTC in `<work>/cuda12_libs` for run time.

## Patches (`deepprove.patch`)

None changes the protocol.

1. `bench-llm` records verification time. Upstream verifies but does not
   time it.
2. GPT-2 SafeTensors loader: the tied LM head gets its own commitment key,
   `wte.weight_final_proj`, as DeepProve's GGUF loader already does.
   Without it, context generation aborts with `Found different MLE for
   polynomial wte.weight`.
3. Llama-2 loader: drops the size-1 KV-group axis for multi-head
   attention (Llama-2-7B), a pure reshape.
4. CNN bench: reads the output tensor back from the device before its
   accuracy check (`tensor_variant`). Upstream fails there on the CUDA
   build with "Tensor is unavailable for a wrapped tensor handler".

## LLaMA2-7B

Not compared. DeepProve's full 32-layer model overflows its `u16` lookup
witness ids (`Should not overflow when incrementing witness id`).
`prepare_models.py --truncate` and `run_deepprove.py --models llama2-L16`
remain for truncated runs.

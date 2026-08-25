# ZK-Torch-3 Benchmark Results

Full model benchmarks on Goldilocks field with GPU-accelerated sumcheck and CPU parallel opening proofs.

**Hardware**: 96 CPU threads, NVIDIA A100 80GB (CUDA_VISIBLE_DEVICES=3)

## Current (2026-03-09)

Optimizations: parallel eq table, dedup opening tasks, GPU opening proofs (n>=22 with CPU fallback), free BasefoldTable GPU memory before prove.

| Model | Layers | Nodes | Edges | Run | Commit | Prove | Verify | Verified |
|-------|--------|-------|-------|-----|--------|-------|--------|----------|
| GPT-2 Small | 12 | 1,910 | 2,778 | 0.74s | 8.66s | 10.40s | 43.8ms | true |
| BERT-Large | 24 | 3,737 | 5,436 | 1.46s | 9.14s | 19.62s | 76.6ms | true |
| GPT-J 6B | 28 | 3,513 | 5,268 | 13.93s | 26.73s | 300.0s | 88.7ms | true |
| LLaMA-2 7B | 32 | 4,509 | 6,701 | 21.47s | 47.22s | 479.5s | 132.6ms | true |

GPU opening proofs: All models use GPU for n>=22 tasks (33-56 tasks per model, 0 CPU fallbacks). LLaMA has 1 deduplicated opening task.

## Previous (2026-03-01)

| Model | Layers | Nodes | Edges | Run | Commit | Prove | Verify | Verified |
|-------|--------|-------|-------|-----|--------|-------|--------|----------|
| GPT-2 Small | 12 | 1,910 | 2,778 | 1.47s | 17.30s | 11.26s | 42.6ms | true |
| BERT-Large | 24 | 3,737 | 5,436 | 3.27s | 17.82s | 20.03s | 75.1ms | true |
| GPT-J 6B | 28 | 3,513 | 5,268 | 42.52s | 26.59s | 294.00s | 86.7ms | true |
| LLaMA-2 7B | 32 | 4,509 | 6,701 | 58.77s | 33.53s | 462.75s | 121.8ms | true |

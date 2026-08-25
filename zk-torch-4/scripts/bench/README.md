# Evaluation harness for the Morpheus paper

Scripts that produce the numbers the evaluation section needs, as CSV plus
ready-to-paste LaTeX tables.

## Running the whole evaluation

`./run_paper_all.sh` produces every table in one command. Read
**`RUN_ON_H200.md`** first: it lists what comes out, how to read it (verifier
cost is two numbers; CNN batch rows are per-image, not per-proof), and the
three machine-specific caveats.

## Layout

| file | role |
|---|---|
| `run_paper_all.sh` | the whole evaluation, one command |
| `run_paper_e2e.py` | the paper driver: all 11 models, deferral, multi-GPU, shape sweeps |
| `RUN_ON_H200.md` | how to run it and how to read the output |
| `harness.py` | runs a binary under an env, parses its metric lines, samples GPU memory, repeats for medians, writes CSV |
| `models.py` | the model zoo: paper row name → binary, env, config YAML |
| `run_e2e.py` | end-to-end cost across the zoo |
| `run_multigpu.py` | device-count scaling sweep |
| `run_sparsity.py` | sparse vs dense opening, including the OOM boundary |
| `run_deferred.py` | deferred weight opening vs the monolithic baseline |
| `make_tables.py` | CSV → LaTeX (booktabs, uses `\zkt`) |

## What fills what

| paper item | script | notes |
|---|---|---|
| §9.2 end-to-end table | `run_e2e.py` | forward, commit (offline/online), prove, verify, proof size, peak GPU, verified |
| §9.4 multi-GPU scalability | `run_multigpu.py` | run `--model gpt2,whisper`; see the 4-GPU note below |
| §9.5 deferred-opening ablation | `run_deferred.py` | sweeps `N_PROOFS`; reports per-proof amortized and one-time finalize |
| intro: sparse memory reduction + "dense baseline runs out of memory" | `run_sparsity.py` | a dense `oom` row is the result, not a failure |
| intro: largest model + end-to-end time | `run_e2e.py` | pick the largest row that verifies |
| intro: multi-GPU speedup + GPU count | `run_multigpu.py` | prints speedup vs the 1-GPU row |
| §9.3 comparison with prior systems | — | not scripted; see below |

## Quick start

```bash
cd zk-torch-4
cargo build --release                       # or --bin per model

cd scripts/bench
python3 run_e2e.py --profile smoke --reps 1  # minutes: validates the pipeline
python3 make_tables.py out/                  # CSV -> .tex

python3 run_e2e.py --profile paper --reps 3           # the real table
python3 run_multigpu.py --model gpt2,whisper --gpus 1,2,4
python3 run_sparsity.py --model gpt2 --sizes 64,128,256
python3 run_deferred.py --model GPT-2 --proofs 1,2,4,8
```

Every run writes `out/<name>.csv` (medians), `out/<name>.raw.csv` (per
repetition) and a full stdout/stderr log per run under `logs/`. The CSV carries
the exact env of each row, so any number in the paper can be traced back to the
command that produced it.

## Things that will bite you

**The config YAML is `args[1]` for every binary.** `zk_torch_4::CONFIG_FILE`
reads it; with no argument the built-in default applies (`scale_factor_log 10`,
`table_size_log 20`). `models.py` passes one explicitly per model.

**`bench_config.yaml` is benchmark-only.** It sets `table_size_log: 10` and
says so in its own comment ("production uses 20"). A 2^10 range table does not
cover realistic activation ranges. §9.1 must state the config used for each
model; do not quietly report a `table_size_log: 10` number as the production
configuration.

**`table_commit_log` is the single biggest prover knob, and it is not
obvious.** Each sparse lookup auxiliary is split into
`ceil(table_num_vars / table_commit_log)` chunks (`dag/mod.rs`, the
`NO_SPARSE_SPLIT` block), each committed at arity
`input_n + table_commit_log`. The fold tree runs its same-point sumcheck on
GPU only while arity <= 24 (`ZK4_GPU_SP_MAX_ARITY`, capped by the
`2 * 2 * M * 2^A * 16 B` footprint); above that it falls back to CPU and the
GPUs go idle. So the knob sets a cliff, and the optimum is roughly

    table_commit_log = 24 - max_input_n

which puts the largest auxiliary bucket exactly on the cap. Going lower does
not help: the model's dense leaves already sit near arity 24, so only the
chunk count rises and with it the range-lookup cost. Measured on llama2
8L/seq64, 4xA100, everything else fixed:

| table_commit_log | 12 | 10 | 8 | **6** | 4 |
|---|---|---|---|---|---|
| prove | 97.6s | 52.0s | 45.4s | **37.4s** | 40.2s |
| fold tree | 81.7s | 32.5s | 22.8s | **14.9s** | 12.2s |
| range lookup | 6.2s | 9.5s | 12.6s | 13.6s | 19.7s |
| top bucket arity | 29 | 28 | 26 | **24** | 24 |
| proof size | 3412 MB | 1615 MB | 471 MB | **172 MB** | 97 MB |

All verify. `run tune_config.py` sweeps this per model; it is worth redoing
whenever a model's shapes change, since `max_input_n` moves with them.

**`table_size_log` is nearly free in TIME but NOT in memory.** Prove time
barely moves with it (VGG-16 224^2: 4.50s at both 12 and 18; BERT-Large
24L/384: 452/465/470s at 10/16/24; llama2 8L/seq64: 12->15->16 inside the
+-0.8s noise floor). Peak memory does move, because chunk count is
`ceil(tsl / tcl)`: BERT-Large measured 44.0 GB at tsl 10 versus 122.9 GB at 16
and 98.5 GB at 24. So pick the SMALLEST tsl that covers the model's value
range, not the largest you can afford.

Beware of reading a range-overflow warning as a coverage requirement. A
`NonNegative` failure reporting `max overflow value = 0` is entirely NEGATIVE
values, which no table size can cover, and the prover's "Raise table_size_log
to >= N" hint is misleading in that case. BERT-Large appeared to need `tsl >=
21`; the real cause was the `llama_rms_norm` outer-product bug, and after
fixing it the shipped `tsl = 10` verifies and is the fastest of the three.

**`table_size_log`, historical note on cost.** It
sets range coverage (the values the lookup table can represent) and changes
only the chunk *count*, never the arity. Raising it 12 -> 15 -> 16 on llama2
8L/seq64 measured 38.0 / 36.6 / 38.2s prove with identical 172 MB proofs and
+0.3% leaves, i.e. inside run-to-run noise (which is about +-0.8s on this
box, so re-run before believing any single-run delta of that size). Pick it
from the family's activation range rather than for speed: LLMs want 15-16,
CV is being evaluated at 12 vs 18.

**Proof size is set by `table_commit_log`, not by the model.** The fold tree
buckets leaves by arity and sends one `final_witness` per bucket in the clear,
at that bucket's full 2^arity. Buckets run over a contiguous arity range, so
the total is about twice the largest bucket, and one arity step doubles the
proof. A sparse lookup auxiliary sits at `arity = padded_edge_arity +
table_commit_log`, so `table_commit_log` lands directly in the exponent:
Whisper (1+1L) is 1.75 GB under `llama2_config.yaml` (commit_log 12, one bucket
at arity 28 costing 872 MB) and 115 MB under `bench_config.yaml` (commit_log 8)
— same model, same verdict, 15x the proof. Use the narrowest config a model
actually needs (the binding constraint is the LayerNorm/RMSNorm mean tolerance,
about `hidden/2`), and state the config next to every proof-size number in
§9.1. A proof-size comparison across models on different configs is not a
comparison of the models.

**This node has 4 GPUs, but §7 promises "up to eight".** `nvidia-smi -L`
reports 4×A100-80GB. `NUM_PARTITIONS=8` on 4 cards does not use 8 GPUs, it
round-robins 8 partitions over 4 devices (2 per GPU, double the per-device
memory), and `ZK4_GPU_DEVICES=0,..,7` would name devices that do not exist
without erroring. Either measure on an 8-GPU node or soften §7 and the intro
to the count actually measured. `run_multigpu.py` drops counts above the
visible device count with a note, and cross-checks the pool the prover
actually built (from the `[fold_tree] scheduler:` line under `ZK4_TIMING=1`)
against the count requested.

**"Verified (modulo N deferred constant claims)" is not a pass.** `gpt2`,
`llama2` and `oneshot_gpt2` print that variant when weights are deferred; it is
sound only once the streaming finalize runs. The harness records it as
`verified_modulo_deferred` rather than a clean verify, so it cannot be reported
as a standalone end-to-end number.

**The 11 compact streaming bins hardcode `Verified: true`.** They signal
failure by returning early after a stderr line, so the harness additionally
requires the `=== Results` / `Stream summary` block and flags
`streaming_aborted` otherwise.

**A "WILL fail to verify" warning is a dead run.** It means values left the
range table, so the proof cannot verify. The harness flags it as
`range_table_overflow` rather than letting it look like a slow success.

**`ZK4_SPARSE_SP=0` alone is a no-op at the arities that matter.** It is read
inside the *host* same-point path, but by default the fold tree takes the
device-resident/GPU path for arity 18–24 and never constructs that state. Both
sides of the ablation must therefore be pinned to the host path
(`ZK4_DEVICE_RESIDENT_FOLD=0 ZK4_GPU_SP_MAX_ARITY=0`), which is what
`run_sparsity.py` does. Without that pinning the "dense baseline" silently runs
the same code as the sparse one.

**Sparse paths only engage at arity ≥ 20** (`ZK4_SPARSE_SP_MIN_ARITY`,
`ZK4_SPARSE_BOOL_MIN_ARITY`), and the density gates `ZK4_SPARSE_SP_RATIO` (8)
and `ZK4_SPARSE_BOOL_RATIO` (2) must hold too. Below those, sparse and dense
are the same code path, so equal timings mean "nothing switched", not
"sparsity does not help". Sweep sizes upward until the curves separate.

**`NO_SPARSE_SPLIT` is presence-only** — even `=0` disables splitting. It is
deliberately excluded from the dense baseline: it raises every auxiliary's
arity from `input_n + table_commit_log` to `input_n + table_size_log`, which
changes the fold-tree bucket structure instead of isolating sparsity. Study
that axis with the config pairs `bench_config.yaml` vs
`bench_config_single_chunk.yaml` and `llama2_config.yaml` vs
`llama2_config_more_chunks.yaml`.

**Pin the device pool for the 1-GPU row.** `ZK4_GPU_DEVICES` defaults to every
visible device, so a "1 GPU" row that does not set it is not a 1-GPU row.
`run_multigpu.py` always sets it.

**`verified` must be true in every reported row.** Nothing else in a row means
anything if it is false, and the harness marks such a run failed. In the
multi-GPU table it matters twice over: §7 claims the proof is identical at every
device count, so an unverified row would contradict the paper.

## §9.3, comparison with prior systems

Not scripted, because it needs external artifacts. Two defensible routes:

1. **Published numbers.** Cite each baseline's reported cost, and state the
   hardware and model configuration alongside ours. Cheap, but only comparable
   where the configurations match.
2. **Run the baselines.** `research/` in this repo holds `ezkl`, `zk-torch-2`,
   and a VerfCNN checkout. Anything run locally must be reported with its own
   hardware and config, and it is worth saying explicitly which comparisons are
   like-for-like and which are not.

Whichever route, note that Morpheus is post-quantum and the curve-based
baselines are not, so a raw prover-time comparison flatters them; say so rather
than leaving the reader to notice.

## Adding a metric

Add a row to `METRICS` in `harness.py`: a column name, a regex with one capture
group, and `"dur"` (Rust `{:?}` Duration, normalized to ms) or `"int"`. Then add
the column to the relevant recipe in `TABLES` in `make_tables.py`. No driver
changes needed.

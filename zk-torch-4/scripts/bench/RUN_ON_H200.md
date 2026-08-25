# Running the paper evaluation on 8×H200

One command produces every table:

```bash
cd zk-torch-4/scripts/bench
./run_paper_all.sh
```

The GPU count is autodetected from `nvidia-smi` and drives BOTH the device list
and `NUM_PARTITIONS` for every stage, so check the `=== N GPU(s) in use ===`
line at the top. If a container or `CUDA_VISIBLE_DEVICES` makes it report the
wrong number, force it:

```bash
FORCE_GPUS=8 ./run_paper_all.sh --fast
```

It detects the pool, builds release first, and writes CSVs to `out/` with
per-run logs in `logs/`. Expect it to take many hours; each stage is
independent, so a stage that fails does not lose the ones before it.

To do a shorter pass first:

```bash
./run_paper_all.sh --quick            # feasibility only, one rep, one proof
./run_paper_all.sh --fast             # ~3h: headline + amortization + partial scaling
./run_paper_all.sh --only ResNet      # one model through every stage
```

## How long it takes

One pass over all 11 models at ONE inference each is roughly **40 min** on
8×H200. That comes from measured anchors on 4×A100 — GPT-2 154 s, BERT-Large
310 s, Llama-2 1596 s per inference (forward + commit + prove + verify) —
scaled by ~3× for the newer, larger pool.

The streaming stages multiply that pass by `N_PROOFS`, and each stage by
`--reps`:

| configuration | headline stage alone |
|---|---|
| N=8, reps=3 (**the default**) | **~15 h** |
| N=8, reps=1 | ~5 h |
| N=4, reps=1 | ~2.5 h |
| N=2, reps=1 (`--fast`) | ~1.3 h |

With all seven stages the default plan is **well over a day**. Budget for that,
or use `--fast`.

Three-quarters of the cost is the three large decoders (Llama-2, Llama-3,
GPT-J). Halving their `SEQ_LEN` to 128 cuts attention 4× and is the single
biggest lever if you want the full stage list in less time.

### `--fast`: ~2.9 h

Headline (N=2), no-deferral baseline (N=1), a 1/2/4/8 GPU scaling ladder, the
RQ5 copy-constraint ablation, and a sequence sweep. The headline and
no-deferral stages run with `SEQ_LEN=128` for the five transformer rows.

| stage | estimate |
|---|---|
| headline, N=2, seq 128 | 0.71 h |
| no-deferral, N=1, 8 models | 0.30 h |
| scaling ladder ×4 models | ~1.1 h |
| RQ5 ablation, VGG ×2 | ~0.05 h |
| sequence sweep, Llama-3 + GPT-2 at 32/64/128/256 | ~0.8 h |
| **total** | **~2.9 h** |

**Sequence sweep.** Scoped per model with `--only`, which is load-bearing:
`--seq` feeds each model's OWN shape knob, so an unscoped sweep would set
`BATCH=32…256` on the CNNs — 256 images per proof at 640² on YOLO. Two models
rather than five because the shape of the curve is the claim; the others redraw
it at a different constant. Override with `FAST_SEQ_MODELS` / `FAST_SEQ_LENS`.
Llama-3 is most of the 0.8 h, so `FAST_SEQ_MODELS="GPT-2"` buys nearly all of
it back.

The no-deferral stage skips YOLO, 3D-UNet and PointPillars. It is the
"technique off" half of the amortization pair, and VGG + ResNet already make
the CNN point — the three skipped rows are the expensive ones and add nothing
the other two do not show. Change it with `--skip`.

The ladder, broken out (8-GPU time × ~15 for the 1/2/4/8 sweep):

| model | 8-GPU inference | ladder |
|---|---|---|
| Llama-3-8B | ~127 s | ~0.53 h |
| ResNet-50 | 50 s | 0.21 h |
| GPT-2 | 23 s | 0.10 h |
| VGG-16 | 2 s | 0.01 h |

`SEQ_LEN=128` touches only the transformer rows — no CV or Whisper binary reads
`SEQ_LEN`, so it cannot silently reshape them. Attention is O(seq²) and the rest
O(seq), so halving from 256 buys about 2.2× on the models that dominate the
bill.

**Which models get the scaling ladder.** A ladder costs ~15× the 8-GPU time
(the 1-GPU run alone is ~8×), so all eleven would be ~5.4 h — restricting is
unavoidable. The default is four: Llama-3 and ResNet-50 carry the claim, and
GPT-2 and VGG cost 0.11 h between them while making it stronger. Llama-3 rather
than Llama-2 as the large decoder: now that it is sharded the two are
configured alike, and Llama-3 is the cheaper of the pair. With both ends
of the size range present the sweep shows *where* scaling starts to pay rather
than asserting it at a single size. VGG is expected to be near-flat at ~2 s per
inference; that is the informative bottom of the range, not a failure. Override
with `FAST_SCALING="..."`.

**Spend the spare budget on N.** At `SEQ_LEN=128` the headline fits at larger N:

```bash
FAST_PROOFS=4 ./run_paper_all.sh --fast     # ~3.5 h — over a 3 h budget
```

With four models in the ladder the spare budget is gone: N=4 pushes the total
to ~3.5 h. Either accept that, or drop `GPT-2 VGG` from `FAST_SCALING` to buy
the 0.11 h back — the scaling breadth is worth more than the 0.11 h, the N is
worth more than either, so if you want N=4 inside 3 h, cut a *stage*, not the
ladder.

That matters because the tradeoff at small N runs *against* us — the safe
direction, but still worth removing. The one-time finalize is amortized over N
proofs, so at N=2 the per-proof headline is **pessimistic** by roughly ¾ of
`finalize/N`; at N=4 that halves. Quote N=2 as a lower bound on the technique,
or pay 45 more minutes for N=4.

Note that these are estimates with real error bars: five of the eleven models
have never been timed at their paper configuration on this hardware: YOLO at
8 stages, Whisper at 1500 context and the two new decoder streaming bins.
3D-UNet is now measured at 64³ rather than estimated at 128³.

## What comes out

| file | what it is |
|---|---|
| `manifest.txt` | provenance: git commit + dirty state, host, GPUs, driver, rustc, the config YAMLs, and the full run plan |
| `feasibility.csv` | one rep per model: which configurations complete at all |
| `headline.csv` | main table, every technique on, 3 reps |
| `nodefer.csv` | same with deferral off — the amortization comparison (RQ4) |
| `mono1.csv` | monolithic single proof on 1 GPU — the no-technique reference |
| `scaling.csv` | 1/2/4/8 GPU sweep (RQ3) |
| `seqscale.csv` | sequence sweep, transformers |
| `batchscale_VGG.csv`, `batchscale_ResNet.csv` | batch sweep, folded-batch CNNs |

Every row carries the exact environment that produced it, and `logs/` holds one
file per run with the full command line, stdout **and** stderr — so any number
traces back to the command that made it.

`out/manifest.txt` carries what a per-run log cannot: which build and which
machine. It records the git commit **and whether the tree was dirty** (a dirty
tree means the commit does not identify the build, so it says exactly that),
the GPU inventory and driver version, rustc, and the full contents of the three
config YAMLs — those set `scale_factor_log` and the table geometry, so a CSV
without them is not reproducible. It also captures the resolved `FAST_PROOFS` /
`FAST_SCALING` and the whole run plan.

Keep `manifest.txt` with the CSVs. On its own a CSV says what happened but not
what produced it.

## Reading the numbers

**A row is meaningless unless its `verified` column is true.** The runner prints
a warning for any that are not.

**Prover time excludes** weight generation, compile, the forward pass (witness
generation) and offline weight commitment. Those are reported beside it, not
folded in.

**Verifier cost is two numbers.** `verify/u` is the per-inference check;
`fin-ver` is the one-time finalize check that makes a streamed batch sound.
Total is `N * verify/u + fin-ver`. Quoting only the first understates it.

**For CNN batch rows, read `per-image`, not `per-proof`.** One proof covers
`batch` images, so per-proof alone makes batching look like a regression when
it is a win.

## Memory

`peak` in the tables is the **sum across the GPUs in the pool**, not per-GPU —
`_MemSampler` sums `nvidia-smi memory.used` over the visible devices and reports
the peak delta over its first sample. Divide by the device count for the
per-GPU figure that actually decides whether something fits.

Measured on 4×A100 at seq 256:

| model | peak total | ≈ per GPU | of 80 GB |
|---|---|---|---|
| GPT-2 12L | 143 GB | 36 GB | 45% |
| BERT-Large 24L | 173 GB | 43 GB | 54% |
| Llama-2-7B 32L | 280 GB | 70 GB | **87%** |

Llama-2 was the tight one, not GPT-2. On 8×H200 the per-GPU budget roughly
doubles, 8-way partitioning splits the work further than 4-way did, and
`--fast` runs seq 128 where attention memory drops 4× — so these three have
room.

**Llama-3-8B is the model to watch.** `llama3_8b` takes no sharding parameters
at all, where `llama_2_7b` splits its logits head 32 ways and its FFN 16 ways.
With 4× Llama-2's vocabulary its head is 4096 × 128256, padded to 2^29: a
single committed edge of 4.3 GB before proving overhead, which is why its
`MAX_NUM_VARS` is 29 against 27 for the others. If anything OOMs, expect it
here. `feasibility.csv` answers this in ~20 min, before the long stages run.

## Three things specific to this machine

**`table_commit_log` is tuned for 4×A100 (320 GB) and is not retuned here.** It
sets prover time, proof size and peak memory *together*, and it sits in the
proof-size exponent — one step doubles the proof. On 8×H200 (~1128 GB) the
current values are conservative, so there is real headroom being left unused.
The 804 MB Llama-2 proof measured on A100 is a consequence of that value, not a
floor. If you have the hours, run `tune_config.py` per model first; if not, the
defaults are safe and simply not optimal.

**3D-UNet runs at 64³, not MLPerf's 128³ patch.** At 128³ the forward pass
trips a CUDA illegal memory access (code 700) that is still unexplained: not
OOM, not multi-GPU placement, not ConvTranspose3D, and not any index a
runtime bounds guard can see. The error is sticky, so it surfaces at
whatever runs next. The 64³ row verifies end to end (362 s/volume, 1.68 GB
proof) and its NAME carries the resolution, so it is a labelled deviation
rather than a silent substitution.

**Batch is the CNN scaling axis only for VGG and ResNet.** Those two prove a
batch as one graph. YOLO, 3D-UNet and PointPillars have no folded-batch builder
— they replicate the subgraph per image — so the driver PINS them to batch 1
and drops any larger batch from a sweep. They will not silently produce rows
that read as a batch result.

## Known gaps, printed on every run

- **PointPainting** — `UNAVAILABLE`. MLPerf PointPainting is DeepLabV3+
  (ResNet-50 backbone, output-stride 16) fused into PointPillars; only bare
  PointPillars exists. Reporting PointPillars under that name would misstate
  the workload, so the driver refuses rather than substituting. ResNet-50 and
  PointPillars are reported as their own rows.
- **YOLOv11n, not v11l**, and it differs in **two** ways, not one. The depth
  and width multipliers are smaller (real v11n widths, 3→16→32→64→128→256), and
  the **C2PSA attention block (m.10) is skipped** — `dag/yolo.rs:374` treats
  `m9_out` as `m10_out`. Everything else is there: all 8 stages at 640²,
  including SPPF, the FPN/PAN neck and the detect heads.
- **VGG is the VerfCNN configuration** (CIFAR-10 32²), not ImageNet 224².
  MLPerf has no VGG; the comparison point is VerfCNN, so the workload matches
  it. At 224² the same model measured 62.7 s/image against 3.6 s here — 49× the
  pixels — so the two must never be read as the same row.
- **Whisper is tiny.en at its real dimensions** — 4+4 layers, `n_state` 384,
  `n_head` 6, 80 mels, `n_audio_ctx` 1500 (30 s at 50 frames/s), `n_text_ctx`
  448. MLPerf's reference is large-v3, which has no builder. An exact small
  model is reported rather than a large one with a truncated context, because
  a truncation does not show up in the row name.

All eleven models now have a streaming binary, so every row in the headline
table measures the same thing: one streamed proof is one forward pass over the
shape axis, with weights deferred across the stream and opened once at
finalize. Llama-3 and GPT-J previously had no plain-streaming bin and fell back
to the monolithic path with `ZK4_DEFER_CONSTANTS=1`, which owes its finalize
and so returned `verified_modulo_deferred` rather than a clean pass.

## If something fails

Each row's `failure` column says which of: timeout, OOM, `verified_modulo_deferred`
(sound only once finalize runs — not a standalone pass), or `streaming_aborted`.
The matching file under `logs/` has the full stdout and the exact command line
at the top.

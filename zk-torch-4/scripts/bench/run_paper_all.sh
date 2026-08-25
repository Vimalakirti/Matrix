#!/usr/bin/env bash
# Full paper evaluation, start to finish, on a fresh machine.
#
#   cd zk-torch-4/scripts/bench
#   ./run_paper_all.sh                  # everything, autodetects the GPU pool
#   ./run_paper_all.sh --fast           # ~2.5h budget: see the note below
#   ./run_paper_all.sh --only Llama     # one model through every stage
#
# Produces, under out/:
#   headline.csv      the main table, every technique on
#   nodefer.csv       same, deferral off — the amortization comparison
#   mono1.csv         monolithic single proof on 1 GPU — the no-technique ref
#   scaling.csv       1/2/4/8 GPUs per model
#   seqscale.csv      sequence sweep (transformers)
#   batchscale_*.csv  batch sweep, one file per folded-batch CNN
#
# Every stage writes per-run logs to logs/, and every CSV carries the exact
# environment of each row, so any number traces back to the command that made it.

set -uo pipefail
cd "$(dirname "$0")"

ARGV_ORIG="$*"   # captured before the parse loop consumes them
FAST=0
PCS_ONLY=0
AR_ONLY=0
EZKL_ONLY=0
ONLY=""
EXTRA=()   # note: expanded as ${EXTRA[@]+"${EXTRA[@]}"} so an empty array
           # contributes no argument at all; "${EXTRA[@]:-}" would pass "".
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) echo "--quick removed: its only stage was feasibility, which no" >&2
             echo "  longer exists. Use --fast for the ~2.5h profile." >&2
             exit 2 ;;
    --fast)  FAST=1; shift ;;
    --pcs-only) PCS_ONLY=1; shift ;;
    --ar-only)  AR_ONLY=1; shift ;;
    --ezkl-only) EZKL_ONLY=1; shift ;;
    --only)  ONLY="$2"; shift 2 ;;
    *)       EXTRA+=("$1"); shift ;;
  esac
done
ONLY_ARG=(); [[ -n "$ONLY" ]] && ONLY_ARG=(--only "$ONLY")

mkdir -p out logs

# ---- Provenance manifest -------------------------------------------------
# Everything needed to attribute a number to a machine and a code version, in
# one file. Per-run logs already carry the exact command, stdout and stderr;
# what they do not carry is WHICH BUILD and WHICH HARDWARE produced them, or
# the contents of the config YAML that sets scale_factor_log / table_size_log
# and therefore changes every result. A CSV without this is not reproducible.
MAN=out/manifest.txt
{
  echo "=== run manifest ==="
  echo "date        : $(date -Is)"
  echo "host        : $(hostname) $(uname -srm)"
  echo "cwd         : $(pwd)"
  echo "invocation  : $0 $ARGV_ORIG"
  echo "FAST_PROOFS : ${FAST_PROOFS:-2 (default)}"
  echo "FAST_SCALING: ${FAST_SCALING:-Llama-3 ResNet GPT-2 VGG (default)}"
  echo
  echo "--- code ---"
  echo "git commit  : $(git rev-parse HEAD 2>/dev/null || echo '(not a git tree)')"
  echo "git branch  : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    echo "git state   : DIRTY -- the commit above does NOT identify this build"
    git status --porcelain 2>/dev/null | sed 's/^/              /'
  else
    echo "git state   : clean"
  fi
  echo "rustc       : $(rustc --version 2>/dev/null || echo '?')"
  echo
  echo "--- hardware ---"
  echo "cpus        : $(nproc 2>/dev/null || echo '?')"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null \
    | sed 's/^/              /'
  echo
  echo "--- config YAMLs (these set scale_factor_log / table geometry) ---"
  for c in bench_config.yaml cv_config.yaml llama2_config.yaml; do
    f="../../$c"
    echo "  [$c]"
    if [[ -f "$f" ]]; then sed 's/^/    /' "$f"; else echo "    (missing)"; fi
  done
} > "$MAN" 2>&1
echo "manifest -> $MAN"

# Autodetect, overridable with FORCE_GPUS. Inside a container, or with
# CUDA_VISIBLE_DEVICES set, nvidia-smi can report a different count than the
# machine has -- and every stage sizes its device list AND its partition count
# from this number, so getting it wrong silently changes what is measured.
GPUS=${FORCE_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 1)}
[[ "$GPUS" -lt 1 ]] && GPUS=1
echo "=== $GPUS GPU(s) in use ${FORCE_GPUS:+(forced via FORCE_GPUS)} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /'

# Build first: a stale binary is the easiest way to measure the wrong thing.
echo
echo "=== building (release) ==="
( cd ../.. && cargo build --release ) || { echo "BUILD FAILED"; exit 1; }

# Device-count ladder up to whatever is present.
LADDER=""
for n in 1 2 4 8; do [[ "$n" -le "$GPUS" ]] && LADDER="${LADDER:+$LADDER,}$n"; done
exec > >(tee -a "$MAN") 2>&1   # from here on, everything also lands in the manifest
echo
echo "=== plan ==="
echo "    pool        : $GPUS"
echo "    scaling     : $LADDER"
echo "    deferral    : ON (default). --no-deferred produces the comparison row."
if [[ "$AR_ONLY" -eq 1 ]]; then
  echo "    profile     : --ar-only (no other stage runs)"
  echo "    workload    : autoregressive GENERATION, not a prefill pass"
  echo "                  embedding + transformer + lm_head + argmax per token"
  echo "    prompt      : ${AR_PROMPT:-16} tokens"
  echo "    generated   : ${AR_GEN_LENS:-16,112} tokens (proof covers prompt+generated)"
  echo "    generations : N=${AR_PROOFS:-${FAST_PROOFS:-2}} per model, weights deferred"
elif [[ "$PCS_ONLY" -eq 1 ]]; then
  echo "    profile     : --pcs-only (no model is proven)"
  echo "    experiments : sparsity ablation, polynomial size, device count"
elif [[ "$FAST" -eq 1 ]]; then
  # The plan text has to describe the profile that will actually run. It used
  # to print the full plan's parameters under --fast too, which put wrong
  # provenance in the manifest: N=8 when N=2 ran, a 1-GPU baseline when the
  # baseline used the whole pool, and a seq sweep that was not the one run.
  echo "    profile     : --fast"
  echo "    headline    : amortized per-proof, N=${FAST_PROOFS:-2}, full pool, seq 128"
  echo "    baseline    : monolithic single proof, full pool, seq 128"
  echo "                  (skips YOLO, 3D-UNet, PointPillars)"
  echo "    scaling set : ${FAST_SCALING:-Llama-3 ResNet GPT-2 VGG}"
  echo "    seq sweep   : ${FAST_SEQ_MODELS:-Llama-3 GPT-2} at ${FAST_SEQ_LENS:-32,64,128,256}"
else
  echo "    headline    : amortized per-proof, N=8, full pool"
  echo "    baseline    : monolithic single proof, 1 GPU"
  echo "    seq         : 256 headline; sweep 64,128,256"
fi
echo
echo "    NOTE: table_commit_log values are tuned for 4xA100 (320 GB) and are"
echo "          NOT retuned here. They set prover time, proof size AND peak"
echo "          memory together. On 8xH200 (~1128 GB) they are CONSERVATIVE:"
echo "          there is headroom to raise them, which tune_config.py sweeps."
echo "          3D-UNet runs at 64^3: 128^3 exceeds HOST memory (745 GiB"
echo "          resident without completing one volume), which no amount of"
echo "          GPU headroom fixes."
echo
echo "    Batch is the CNN axis only where the builder proves a batch as ONE"
echo "    graph: VGG and ResNet. The others still replicate per image, which"
echo "    costs superlinearly, so the batch sweep covers the folded models and"
echo "    the driver labels the rest REPLICATED wherever they appear."
echo

run() {  # run <label> <outfile> <timeout> [extra args...]
  local label="$1" out="$2" tmo="$3"; shift 3
  echo
  echo "=== $label -> out/$out ==="
  python3 run_paper_e2e.py --gpus "$GPUS" --timeout "$tmo" --out "out/$out" \
      "${ONLY_ARG[@]}" ${EXTRA[@]+"${EXTRA[@]}"} "$@" 2>&1 \
      | tee "logs/stage_${out%.csv}.txt"
}

# Feasibility used to run first. It is gone: every model in the table now has a
# configuration that completes, so a stage whose only output was "does this
# finish" was paying a full pass over 11 models to answer a question already
# answered. The per-case timeout still catches a row that stops completing.

# ---------------------------------------------------------------------------
# PCS stage as a function so it can run standalone. The experiments take
# minutes; re-running the whole multi-hour profile to collect them would be
# absurd, and out0822 shipped without them because the box was on a commit
# that predated the stage. `--pcs-only` runs exactly this and exits.
# ---------------------------------------------------------------------------
run_pcs_stage() {
# ---- PCS: sparsity, polynomial size, device count ---------------------
# These isolate the commitment scheme from any model. A model fixes its own
# leaf set, so it cannot answer "how does the PCS scale in polynomial size?"
# bench_pcs_scaling synthesizes a leaf set of controlled arity, count and
# DENSITY and runs the same prove_fold_tree the real prover runs.
#
# Sparsity is the axis the opening actually keys on (same_point_sumcheck.rs):
# a sparse enough binary leaf is opened WITHOUT materializing the dense
# 2^arity Ext2 equality table, which is ~1 GB per leaf at arity 26. Round
# messages are byte-identical between the two paths, so proof size must come
# out equal; pcs_row records both so a divergence is visible rather than
# assumed. MEASURED crossover: at arity 20 the two are equal (206 vs 203 ms),
# at 22 sparse wins 1.67x (275 vs 459 ms). The sweep spans that.
#
# Peak GPU memory is sampled here rather than by harness.py, which only wraps
# run_paper_e2e.py. Memory is the primary claim for the sparsity experiment,
# so a stage without it would miss the point.
PCS_CSV=out/pcs.csv
echo "experiment,arity,leaves,dens_log,sparse,gpus,commit_ms,open_ms,proof_bytes,gpu_peak_mib,host_peak_mib,status" > "$PCS_CSV"
pcs_row() {  # pcs_row <experiment> <arity> <leaves> <dens_log> <sparse> <devs>
  local exp="$1" ar="$2" lv="$3" dl="$4" sp="$5" devs="$6"
  local ngpu; ngpu=$(awk -F, '{print NF}' <<<"$devs")
  local of; of=$(mktemp)
  # The binary reports BOTH peaks itself. External sampling was tried three
  # ways and all three failed: --query-gpu=memory.used reads the whole device
  # (it measured an unrelated job), filtering on $! fails because timeout
  # forks, and nvidia-smi reports used_memory=0 for this binary while
  # reporting real values for others. In-process cudaMemGetInfo plus VmHWM
  # cannot miss its own allocations.
  PCS_ARITY="$ar" PCS_LEAVES="$lv" PCS_DENS_LOG="$dl" \
    ZK4_SPARSE_SP="$sp" ZK4_GPU_DEVICES="$devs" REPS="${PCS_REPS:-1}" \
    timeout "${PCS_TIMEOUT:-1800}" ../../target/release/bench_pcs_scaling > "$of" 2>&1
  local rc=$?
  local out; out=$(cat "$of"); rm -f "$of"
  local peak hpeak
  peak=$(grep -oE 'peak_mib=[0-9]+' <<<"$out" | head -1 | cut -d= -f2)
  hpeak=$(grep -oE 'host_peak_mib=[0-9]+' <<<"$out" | head -1 | cut -d= -f2)
  local line; line=$(grep -oE '^\[pcs\].*' <<<"$out" | head -1)
  if [[ -n "$line" ]]; then
    local c o b
    c=$(grep -oE 'commit_ms=[0-9.]+' <<<"$line" | cut -d= -f2)
    o=$(grep -oE 'open_ms=[0-9.]+'   <<<"$line" | cut -d= -f2)
    b=$(grep -oE 'proof_bytes=[0-9]+' <<<"$line" | cut -d= -f2)
    echo "$exp,$ar,$lv,$dl,$sp,$ngpu,$c,$o,$b,${peak:-0},${hpeak:-0},ok" >> "$PCS_CSV"
    printf "  %-9s arity=%-3s lv=%-3s dens=%-3s sparse=%s gpu=%s  commit=%-7sms open=%-9sms proof=%-10s hostpeak=%-7sMiB gpupeak=%sMiB\n" \
           "$exp" "$ar" "$lv" "$dl" "$sp" "$ngpu" "$c" "$o" "$b" "${hpeak:-?}" "${peak:-?}"
  else
    # A failure is a result here: the dense path is expected to run out of
    # memory at high arity, and that is the claim. Record it, do not drop it.
    local why="exit$rc"; grep -qiE 'out of memory|alloc' <<<"$out" && why="oom"
    echo "$exp,$ar,$lv,$dl,$sp,$ngpu,,,,${peak:-0},${hpeak:-0},$why" >> "$PCS_CSV"
    printf "  %-10s arity=%-3s sparse=%s gpus=%s  FAILED (%s)\n" "$exp" "$ar" "$sp" "$ngpu" "$why"
  fi
}
ALLDEV=$(seq -s, 0 $((GPUS-1)))
echo
echo "=== PCS-A: sparsity ablation (dense eq table vs support list) -> $PCS_CSV ==="
for dl in 0 4 8 12; do for sp in 1 0; do
  pcs_row sparsity "${PCS_ABL_ARITY:-26}" 16 "$dl" "$sp" "$ALLDEV"
done; done
echo
echo "=== PCS-B: polynomial size (arity sweep) -> $PCS_CSV ==="
for ar in 18 20 22 24 26; do for sp in 1 0; do
  pcs_row size "$ar" 32 8 "$sp" "$ALLDEV"
done; done
echo
echo "=== PCS-C: device scaling at fixed leaf set -> $PCS_CSV ==="
# LEAF COUNT IS THE POINT HERE, not arity. The fold tree parallelizes across
# groups within a level, and a group holds MAX_PER_GROUP = 63 instances. A
# first version used 64 leaves, which forms 2 groups then 1: there was nothing
# to spread over 8 devices and the result was flat 1712/1748/1777/1751 ms,
# measuring the experiment's design rather than the PCS. 2048 leaves gives
# groups per level [33, 7, 2, 1], so level 0 alone has ~4 groups per GPU on an
# 8-GPU pool. Arity drops to 22 to keep the witness near 1 GB.
#
# For reference, real models are in this regime: out0822's Llama-3 fold tree
# goes 48.3s -> 8.4s across 1->8 GPUs (5.75x) with ~4400 leaves over 8 arity
# buckets.
for n in $(tr ',' ' ' <<<"$LADDER"); do
  pcs_row devices "${PCS_DEV_ARITY:-22}" "${PCS_DEV_LEAVES:-2048}" 8 1 "$(seq -s, 0 $((n-1)))"
done

}

# ---------------------------------------------------------------------------
# Autoregressive generation. The headline table proves ONE masked forward pass
# over SEQ_LEN tokens, which is a prefill: it omits the embedding, the LM head
# and the argmax binding each sampled token to its logits, and its "per token"
# is prefill cost divided by width. These rows prove full T-token GENERATIONS
# with all of that included, so per-token is per GENERATED token and is
# comparable to systems that report autoregressive decoding.
#
# Two lengths on purpose. 16 matches the generation length prior work reports,
# so the comparison is like-for-like. 128 is the headline: a longer generation
# amortizes the fixed per-proof work and is the regime the system is meant for.
# ---------------------------------------------------------------------------
run_ar_stage() {
  # AR_GEN_LENS are GENERATED token counts, each preceded by AR_PROMPT prompt
  # tokens. The proof covers prompt+generated positions; per-token cost divides
  # by the generated count alone, because the prompt is context the request
  # supplied rather than output the prover produced.
  #
  # P16 G16 matches the configuration prior work reports, so that comparison is
  # like-for-like. P16 G112 is the headline: 128 proven positions, of which 112
  # are generated, which is the regime where the fixed per-proof work amortizes.
  local gens="${AR_GEN_LENS:-16,112}"
  local prompt="${AR_PROMPT:-16}"
  echo
  echo "=== AR generation (prompt $prompt, generated $gens, N=${AR_PROOFS:-$FP}) -> out/ar.csv ==="
  python3 run_paper_e2e.py --gpus "$GPUS" --timeout "${AR_TIMEOUT:-14400}" --reps 1 \
      --proofs "${AR_PROOFS:-$FP}" --ar --ar-prompt "$prompt" --seq "$gens" \
      --out out/ar.csv "${ONLY_ARG[@]}" 2>&1 | tee logs/stage_ar.txt
}

# ---------------------------------------------------------------------------
# EZKL head-to-head. Both circuits are the exact shapes EZKL ships in
# examples/onnx (LeNet-5, and nanoGPT at n_layer=4 n_head=4 n_embd=64 vocab=65
# block_size=64), so the comparison is like-for-like rather than a re-spec.
# ONE GPU on purpose: EZKL reports single-device numbers, and these circuits
# are far too small to occupy more.
# ---------------------------------------------------------------------------
run_ezkl_stage() {
  echo
  echo "=== EZKL models (LeNet-5, nanoGPT seq=${EZKL_NANOGPT_SEQ:-1}), 1 GPU, N=${EZKL_PROOFS:-8} -> out/ezkl.csv ==="
  python3 run_ezkl.py --proofs "${EZKL_PROOFS:-8}" --gpu "${EZKL_GPU:-0}" \
      --threads "${EZKL_THREADS:-32}" \
      --timeout "${EZKL_TIMEOUT:-3600}" --out out/ezkl.csv \
      "${ONLY_ARG[@]}" 2>&1 | tee logs/stage_ezkl.txt
}

if [[ "$EZKL_ONLY" -eq 1 ]]; then
  run_ezkl_stage
  echo
  echo "=== EZKL stage done -> $(pwd)/out/ezkl.csv ==="
  exit 0
fi

if [[ "$AR_ONLY" -eq 1 ]]; then
  FP="${FAST_PROOFS:-2}"
  run_ar_stage
  echo
  echo "=== AR stage done -> $(pwd)/out/ar.csv ==="
  exit 0
fi

if [[ "$PCS_ONLY" -eq 1 ]]; then
  run_pcs_stage
  echo
  echo "=== PCS stage done -> $(pwd)/out/pcs.csv ==="
  exit 0
fi

if [[ "$FAST" -eq 1 ]]; then
  # ~3h budget. The cost model: one pass over all 11 models at one inference
  # each is roughly 40 min on 8xH200 (measured anchors: GPT-2 154s, BERT-Large
  # 310s, Llama-2 1596s per inference on 4xA100, scaled by ~3x). The streaming
  # stages multiply that by N_PROOFS and by --reps, so the full default plan
  # (N=8, reps=3, seven stages) is well over a day. This profile keeps the
  # claims that need a pair of numbers and drops the sweeps.
  #
  # Estimated ~1.5h at N=2 / seq128 on 8xH200; N=4 is ~2.2h and N=8 ~3.7h.
  # Override with FAST_PROOFS=4 to spend spare budget on a less pessimistic
  # amortization number.
  #
  # TRADEOFF, and it runs against us rather than for us: at N=2 the one-time
  # finalize is amortized over 2 proofs instead of 8, so the per-proof headline
  # is PESSIMISTIC by roughly 3/4 of finalize/N. Quote it as a lower bound on
  # the technique, or re-run the headline alone at N=8 (~5h) once the
  # configuration is known good.
  # SEQ_LEN=128 applies to the five transformer rows and nothing else -- no CV
  # or Whisper binary reads SEQ_LEN, so this cannot silently reshape them.
  # Attention is O(seq^2) and the rest O(seq), so halving from 256 buys about
  # 2.2x on the models that dominate the bill.
  FP="${FAST_PROOFS:-2}"
  run "headline (fast, N=$FP, seq128)" headline.csv 7200 \
      --reps 1 --proofs "$FP" --env SEQ_LEN=128
  # The no-deferral baseline is the "technique off" half of the amortization
  # pair. It makes its CNN point with VGG and ResNet -- the two that fold-batch
  # -- so YOLO at 640^2, 3D-UNet at 128^3 and PointPillars are skipped: they
  # are the expensive rows and add nothing the other two do not already show.
  run "no-deferral baseline (fast, seq128)" nodefer.csv 7200 \
      --reps 1 --proofs 1 --no-deferred --env SEQ_LEN=128 \
      --skip "YOLO,3D-UNet,PointPillars"
  # Which models get the 1/2/4/8 ladder. A ladder costs ~15x the 8-GPU time
  # (the 1-GPU run alone is ~8x), so all eleven would be ~5.4h -- more than the
  # whole fast budget. Restricting is necessary; WHICH models is a real choice.
  #
  # Llama-3 rather than Llama-2 as the large decoder: now that it is sharded
  # the two are configured alike, and Llama-3 is the cheaper of the pair, so
  # the ladder costs less for the same claim.
  #
  # Default is all four: Llama-3 and ResNet-50 plus GPT-2 and VGG
  # (~0.1h). The large pair carries the claim -- scaling is about spreading
  # work across devices, so it wants models with work to spread. The small pair
  # costs almost nothing and makes the result stronger rather than weaker: with
  # both ends present the sweep shows WHERE scaling starts to pay, instead of
  # asserting it at one size. VGG at ~5s per inference is expected to be
  # near-flat; that is the informative bottom of the range, not a failure.
  SCAL="${FAST_SCALING:-Llama-3 ResNet GPT-2 VGG}"
  echo
  echo "=== scaling (fast: $SCAL) ==="
  for m in $SCAL; do
    python3 run_paper_e2e.py --gpus "$LADDER" --timeout 7200 --reps 1 --proofs 1 \
        --env SEQ_LEN=128 --out "out/scaling_$m.csv" --only "$m" 2>&1 \
        | tee "logs/stage_scaling_$m.txt"
  done
  # RQ5: the conv output-binding copy constraint. VerfCNN uses a grand-product
  # multiset argument; this system uses a masked tensor view (sumcheck C).
  # ZK4_CONV_GRANDPRODUCT=1 selects the grand-product path on the SAME graph,
  # so the pair isolates the copy constraint and nothing else. Measured at
  # VGG 4 layers / 32^2: tensor view 1512ms vs grand product 2664ms per image.
  echo
  echo "=== RQ5: tensor view vs grand product (VGG) -> out/convbind_*.csv ==="
  for mode in tensorview grandproduct; do
    EXTRA_ENV=()
    [[ "$mode" == grandproduct ]] && EXTRA_ENV=(--env ZK4_CONV_GRANDPRODUCT=1)
    python3 run_paper_e2e.py --gpus "$GPUS" --timeout 7200 --reps 1 --proofs "$FP" \
        --only "VGG" --out "out/convbind_$mode.csv" \
        ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} 2>&1 | tee "logs/stage_convbind_$mode.txt"
  done

  # Sequence sweep, transformers only. --seq feeds each model's OWN shape knob,
  # so running it across the table would set BATCH=32..256 on the CNNs; scoping
  # with --only keeps it on SEQ_LEN. One decoder and one encoder-era model
  # rather than all five: the shape of the curve is the claim, and the other
  # three would only redraw it at a different constant.
  SEQ_MODELS="${FAST_SEQ_MODELS:-Llama-3 GPT-2}"
  SEQ_LENS="${FAST_SEQ_LENS:-32,64,128,256}"
  echo
  echo "=== sequence sweep: $SEQ_MODELS at $SEQ_LENS ==="
  for m in $SEQ_MODELS; do
    python3 run_paper_e2e.py --gpus "$GPUS" --timeout 7200 --reps 1 --proofs "$FP" \
        --seq "$SEQ_LENS" --only "$m" --out "out/seqscale_$m.csv" 2>&1 \
        | tee "logs/stage_seqscale_$m.txt"
  done

  run_ar_stage
  run_pcs_stage
  echo
  echo "=== fast profile done. CSVs in out/ ==="
  ls -la out/*.csv 2>/dev/null | sed 's/^/    /'
  exit 0
fi

# 2. Headline table: every technique on.
run "headline (all techniques)" headline.csv 7200 --reps 3

# 3. Same without deferral, same pool. The pair is the amortization claim
#    (RQ4); neither number means much alone.
run "no-deferral baseline" nodefer.csv 7200 --reps 3 --no-deferred

# 3b. Monolithic single proof on one GPU: the "no technique applied" reference
#     the headline is measured against. Separate from stage 3, which isolates
#     deferral alone at the full pool.
echo
echo "=== monolithic baseline (1 GPU, no deferral) -> out/mono1.csv ==="
python3 run_paper_e2e.py --gpus 1 --timeout 7200 --reps 1 --proofs 1 \
    --no-deferred --out out/mono1.csv "${ONLY_ARG[@]}" 2>&1 \
    | tee logs/stage_mono1.txt

# 4. Multi-GPU scaling. The proof is transcript-identical at every device count,
#    so an unverified row invalidates the sweep rather than just that row.
echo
echo "=== scaling 1..$GPUS -> out/scaling.csv ==="
python3 run_paper_e2e.py --gpus "$LADDER" --timeout 7200 --reps 1 \
    --out out/scaling.csv "${ONLY_ARG[@]}" 2>&1 | tee logs/stage_scaling.txt

# 5. Shape sweeps. Sequence is the transformer axis; batch is the CNN axis,
#    because a CNN's input resolution is part of the workload definition.
run "sequence sweep" seqscale.csv 7200 --reps 1 --seq 64,128,256
# 6. Batch sweep, folded-batch models only. Sweeping the replicated models
#    would measure the absence of the technique at 640^2 / 128^3 and cost hours
#    doing it. Per-image is the column that matters here: one proof covers
#    `batch` images.
for m in "VGG" "ResNet"; do
  # Respect --only: a run scoped to one model must not pull these back in.
  if [[ -n "$ONLY" ]] && [[ "${m,,}" != *"${ONLY,,}"* ]] && [[ "${ONLY,,}" != *"${m,,}"* ]]; then
    continue
  fi
  echo
  echo "=== batch sweep: $m -> out/batchscale_$m.csv ==="
  python3 run_paper_e2e.py --gpus "$GPUS" --timeout 7200 --reps 1 \
      --batch 1,2,4,8 --only "$m" --out "out/batchscale_$m.csv" \
      ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "logs/stage_batch_$m.txt"
done

echo
echo "=== done. CSVs in out/, per-run logs in logs/ ==="
ls -la out/*.csv 2>/dev/null | sed 's/^/    /'
echo
echo "Reminder: a row is meaningless unless its 'verified' column is true."
echo "Prover time excludes weight generation, compile, forward (witness"
echo "generation) and offline weight commitment; those are printed beside it."

#!/usr/bin/env bash
# Smoke-test every driver at a tiny size. Proves the harness plumbing works;
# the numbers are meaningless. Each stage prints PASS/FAIL for its own script.
set -u
cd "$(dirname "$0")"
rc=0

stage() { echo; echo "################ $* ################"; }

stage "1/4 run_e2e (9 models, all techniques on)"
python3 run_e2e.py --profile smoke --reps 1 --timeout 900 --out out/smoke_e2e.csv \
  || { echo "STAGE FAIL: run_e2e"; rc=1; }

stage "2/4 run_multigpu (gpt2 1L + whisper tiny, 1 and 2 GPUs)"
python3 run_multigpu.py --model gpt2 --gpus 1,2 --reps 1 --timeout 900 \
  --env NUM_LAYERS=1 --env SEQ_LEN=1 --out out/smoke_multigpu.csv \
  || { echo "STAGE FAIL: run_multigpu(gpt2)"; rc=1; }
python3 run_multigpu.py --model whisper --gpus 1,2 --reps 1 --timeout 900 \
  --env NUM_ENC_LAYERS=1 --env NUM_DEC_LAYERS=1 --env N_AUDIO_CTX=32 \
  --env N_TEXT_CTX=8 --env N_STATE=128 --env N_HEAD=2 --env N_MELS=16 \
  --out out/smoke_multigpu_whisper.csv \
  || { echo "STAGE FAIL: run_multigpu(whisper)"; rc=1; }

stage "3/4 run_sparsity (gpt2 1L, two sizes, host path both sides)"
python3 run_sparsity.py --model gpt2 --sizes 1,2 --reps 1 --timeout 900 \
  --env NUM_LAYERS=1 --env MAX_NUM_VARS=22 --out out/smoke_sparsity.csv \
  || { echo "STAGE FAIL: run_sparsity"; rc=1; }

stage "4/4 run_deferred (gpt2 1L, N=1,2)"
python3 run_deferred.py --model GPT-2 --proofs 1,2 --reps 1 --timeout 900 \
  --env NUM_LAYERS=1 --env SEQ_LEN=1 --out out/smoke_deferred.csv \
  || { echo "STAGE FAIL: run_deferred"; rc=1; }

stage "tables"
python3 make_tables.py out/ || rc=1

echo; echo "################ smoke rc=$rc ################"
exit $rc

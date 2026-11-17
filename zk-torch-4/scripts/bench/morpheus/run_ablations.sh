#!/usr/bin/env bash
# Same-protocol ablations of the paper (Table "Same-Protocol Ablations").
#
#   ./run_ablations.sh [model ...]      (default: gpt2 yolo whisper)
#
# Variants (each changes only how the prover computes, not what it proves):
#   base       default prover
#   nolink     structure-specific link kernels off (zero chunks still skipped)
#   dense      MORPHEUS_DENSE_LINK=1: every chunk, no structure-specific kernels
#   fixedpart  MORPHEUS_AUTH_COLS=0: opening on the link's source placement
#
# 1. Dev-only determinism check: a two-layer GPT-2 request with fixed private
#    coins per variant; all proof hashes must agree. Not a reportable run.
# 2. Timed runs through run_models.sh with fresh coins, N_MONO requests each.
# 3. A summary CSV of prove, link and authentication times.
#
# GPUs: ABL_DEVS (default 2,0) and ABL_NG (default 2). The paper's runs used
# two GPUs; run_models.sh's wait for idle GPUs is skipped (NO_GPU_WAIT=1).
set -u
cd "$(dirname "$0")/../../.."
OUT=${OUT:-scripts/bench/morpheus/out_abl}
DEVS=${ABL_DEVS:-2,0} NG=${ABL_NG:-2}
SEL=${*:-gpt2 yolo whisper}
VARIANTS=${VARIANTS:-"base nolink dense fixedpart"}
# TIMED: the variants to time (default all); others reuse logs already in $OUT.
TIMED=${TIMED:-$VARIANTS}
mkdir -p "$OUT"
env_of() {
  case $1 in
    base) echo "" ;;
    nolink) echo "MORPHEUS_NO_BIN=1 MORPHEUS_NO_FACTOR=1 MORPHEUS_NO_CLAIM=1 MORPHEUS_NO_RB1=1 MORPHEUS_NO_INDEX=1 MORPHEUS_NO_MANYQ=1" ;;
    dense) echo "MORPHEUS_DENSE_LINK=1" ;;
    fixedpart) echo "MORPHEUS_AUTH_COLS=0" ;;
    *) echo "unknown variant $1" >&2; exit 2 ;;
  esac
}

# 1. Determinism check (fixed coins; dev only).
SEED=$(printf '5a%.0s' $(seq 64))
for v in $VARIANTS; do
  log="$OUT/hash_$v.log"
  env CUDA_VISIBLE_DEVICES=$DEVS MORPHEUS_GPUS=$NG MORPHEUS_AUTH_GPUS=$NG PROFILE=ternary B0=11 SLOTS_LOG=10 ZK4_EXP_BITS=8 MORPHEUS_TIMING=1 ARGMAX_LOGIT_BITS=20 \
    WEIGHTS=dense ZK4_EXACT_RMS=1 ZK4_TABLE_SIZE_LOG=24 MODEL=gpt2 NUM_LAYERS=2 SEQ_LEN=128 VOCAB_SIZE=50257 \
    DEFER=0 N_REQUESTS=1 MORPHEUS_COIN_SEED=$SEED MORPHEUS_PROOF_HASH=1 $(env_of $v) \
    timeout 1800 ./target/release/morpheus_e2e > "$log" 2>&1
  echo "[$(date +%T)] $v $(grep -m1 -o 'proof hash [0-9a-f]*' "$log")"
done
n=$(for v in $VARIANTS; do grep -m1 -o 'proof hash [0-9a-f]*' "$OUT/hash_$v.log"; done | sort -u | wc -l)
[ "$n" -eq 1 ] || { echo "variants disagree on the proof hash" >&2; exit 1; }

# 2. Timed runs (fresh coins).
for v in $TIMED; do
  env $(env_of $v) NO_GPU_WAIT=1 CUDA_VISIBLE_DEVICES=$DEVS MORPHEUS_GPUS=$NG MORPHEUS_AUTH_GPUS=$NG \
    OUT=$OUT/$v MODES=mono N_MONO=${N_MONO:-3} scripts/bench/morpheus/run_models.sh $SEL
done

# 3. Summary.
python3 - "$OUT" $VARIANTS <<'EOF'
import csv, glob, os, re, statistics as st, sys
out, variants = sys.argv[1], sys.argv[2:]
w = csv.writer(open(os.path.join(out, 'ablation_summary.csv'), 'w'))
w.writerow(['model', 'variant', 'requests', 'prove_s', 'link_s', 'auth_s', 'auth_attempts', 'auth_per_attempt_s', 'peak_gpu_mib', 'all_verified'])
for v in variants:
    for f in sorted(glob.glob(os.path.join(out, v, '*_mono.log'))):
        t = open(f).read()
        rows = [dict((k, float(x)) for k, x in re.findall(r'(\w+) ([\d.]+)', b)) for b in re.findall(r'^request \d+ prove: [\d.]+s\s+\[([^\]]*)\]', t, re.M)]
        if not rows: continue
        tot = [float(x) for x in re.findall(r'^request \d+ prove: ([\d.]+)s', t, re.M)]
        att = [int(x) for x in re.findall(r'^\[auth, \w+\] [\d.]+s, attempt \d+ AuthTiming \{ attempts: (\d+)', t, re.M)][-len(rows):]
        auth = [r.get('auth', 0) for r in rows]
        peak = re.findall(r'peak GPU memory \(sum over devices\) (\d+)', t)
        w.writerow([os.path.basename(f)[:-9], v, len(rows), round(st.mean(tot), 2), round(st.mean(r.get('link', 0) for r in rows), 2),
                    round(st.mean(auth), 2), ' '.join(map(str, att)), round(sum(auth) / max(1, sum(att)), 3), peak[-1] if peak else '',
                    t.count('verify (true)') >= len(rows) and 'status=0' in t])
EOF
echo "summary: $OUT/ablation_summary.csv"

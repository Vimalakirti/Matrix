#!/usr/bin/env bash
# Resume F1(a) after the 11:57 shutdown: Table 3 Dense link + Source-retained,
# then Table 2, Table 5 one-GPU rows and the convolution ablation.
set -u
B=/scratch/bjchen4_icgpu/goldilocks/zk-torch-4-perf/zk-torch-4/scripts/bench/morpheus; cd $B
# Wait for the driver and for idle GPUs.
until nvidia-smi >/dev/null 2>&1 && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 120; done
echo "[$(date +%T)] GPUs ready"
mv out_t3s/dense out_t3s/dense.interrupted 2>/dev/null
OUT=$B/out_t3s VARIANTS="base nolink dense fixedpart" TIMED="dense fixedpart" ./run_ablations.sh
echo T3_DONE
OUT=$B/out_t2s CNNFX=/scratch/bjchen4_icgpu/goldilocks/deepprove_src/cnn_fixtures ./run_paper.sh one
echo T2_DONE
OUT=$B/out_t5s N_MONO=3 ./run_models.sh g1
OUT=$B/out_t5s N_MONO=3 ./run_models.sh convpp
echo T5_DONE; echo F1_DONE

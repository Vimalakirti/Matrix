#!/usr/bin/env bash
# F1(a) remaining runs after the second outage (new kernel 6.8, driver 580).
set -u
B=/scratch/bjchen4_icgpu/goldilocks/zk-torch-4-perf/zk-torch-4/scripts/bench/morpheus; cd $B
echo "[$(date '+%m-%d %T')] start"
OUT=$B/out_t3s VARIANTS="base nolink dense fixedpart" TIMED="base dense fixedpart" ./run_ablations.sh
echo "[$(date '+%m-%d %T')] T3_DONE"
OUT=$B/out_t2s CNNFX=/scratch/bjchen4_icgpu/goldilocks/deepprove_src/cnn_fixtures ./run_paper.sh one
echo "[$(date '+%m-%d %T')] T2_DONE"
OUT=$B/out_t5s N_MONO=3 ./run_models.sh g1
OUT=$B/out_t5s N_MONO=3 ./run_models.sh convpp
echo "[$(date '+%m-%d %T')] T5_DONE"; echo F1_DONE

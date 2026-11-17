#!/usr/bin/env bash
# The paper's GPU evaluation (Morpheus side), one job at a time.
#
#   DP_WORK=<deepprove work_dir> ./run_paper.sh [one|four|all]   (default all)
#
# one:  one-GPU comparisons on GPU $G1_DEV (default 2): DeepProve's GPT-2 at
#       32 and 128 tokens (five requests at 128) and its CIFAR-10 LeNet and
#       COVID CNN (DP_WORK holds prepare_cnn.sh's fixtures), and EZKL's
#       LeNet-5 and nanoGPT at lengths 8 and 64. Ten requests per row.
# four: the eleven models on four GPUs (three requests each, ten for
#       ResNet-50 and VGG-16; reduced configurations in README.md), the
#       convolution ablation (full-convolution advice) and the one-GPU
#       scaling rows (BERT-Large, YOLOv11n, 3D-UNet and Whisper exceed one
#       GPU's memory and fail there).
# Summaries: python3 gpu_tables.py $OUT. Baselines: ../deepprove (run_all.sh,
# run_cnn.sh) and ../ezkl (run_gpu.sh).
#
# run_models.sh waits until no process holds GPU memory before each run.
# Jobs run sequentially: the LLM witnesses need up to ~200 GB host memory.
set -u
cd "$(dirname "$0")"
export OUT=${OUT:-$(pwd)/out_paper} TMO=${TMO:-14400} LLM_LAYERS=${LLM_LAYERS:-4}
R=./run_models.sh
what=${1:-all}
if [ "$what" = one ] || [ "$what" = all ]; then
  export CNNFX=$(realpath "${CNNFX:-${DP_WORK:?set DP_WORK or CNNFX}/cnn_fixtures}")
  ( export CUDA_VISIBLE_DEVICES=${G1_DEV:-2}
    N_MONO=10 $R dp1cnn
    N_MONO=10 $R dp1s32
    N_MONO=5 $R dp1s128
    N_MONO=10 MODES=mono $R ezkl
    N_MONO=10 $R ezkl8 )
fi
if [ "$what" = four ] || [ "$what" = all ]; then
  export N_MONO=${N_MONO:-3}
  MODES=mono $R yolo unet3d pointpillars whisper gpt2 bert
  # The small CNNs vary most with authentication attempts: ten requests.
  N_MONO=10 MODES=mono $R resnet vgg
  MODES=mono $R llama2 llama3 gptj
  $R convpp
  $R g1
fi
echo "done: $OUT"

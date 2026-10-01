#!/bin/bash

# Runtime control & scheduling
export PARALLEL_MODE="tp"
export SYNC_MODE="async"          # async | sync | local    (gradient sync behavior / GPU utilization)
export MIXED_PRECISION="no"       # bf16 | fp16 | no        (numerical format for compute)
export DYNAMO_BACKEND="no"        # no | inductor | eager   (Torch compile backend)

# Data / shape
export SEQ_LEN=1024               # token sequence length (↑ → more compute, higher power)
export BATCH_SIZE=8              # per-rank microbatch
export ACCUM_STEPS=4              # gradient accumulation (↑ → larger effective batch)

# LORA
export LORA_R=64
export LORA_ALPHA=16
export LORA_DROPOUT=0.05

# Schedule
export LR=2e-5                    # learning rate
export WARMUP_STEPS=100           # warmup steps
export TOTAL_STEPS=999999         # long run; walltime will terminate automatically
export DATASET_SIZE=52000         # number of samples to load
export PIN_MEMORY=1               # 1=use --pin_memory, 0=off (faster host→GPU transfer)

# Sleep
export PRINT_EVERY=1              # log frequency
export SLEEP_EVERY=0              # inject idle every N steps (0 = disabled)
export SLEEP_SEC=0                # idle duration in seconds

# Checkpoint
export CKPT_EVERY=4
export MODEL_SAVE="full"
export INCLUDE_OPTIMIZER=1
export KEEP_LAST=1

# Qwen/Qwen3-235B-A22B \
for GPU_TYPE in "gpu:l40s:4" "gpu:a100:4" "gpu:h100:4" "gpu:h200:4"
do
  for METHOD in "full" "lora" "qlora"
  do
    for MODEL_NAME in \
      "Qwen/Qwen-72B-Chat" \
      "Qwen/Qwen-14B-Chat" \
      "Qwen/Qwen-7B-Chat" \
      "Qwen/Qwen-1_8B"
    do
      export GPU_TYPE
      export METHOD
      export MODEL_NAME
      echo "Submitting: GPU=$GPU_TYPE MODEL=$MODEL_NAME METHOD=$METHOD"
      sbatch --export=ALL --gres=${GPU_TYPE} --time=02:00:00 templates/llm_train_hf_qwen1.sh
    done
  done
done

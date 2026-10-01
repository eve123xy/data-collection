#!/bin/bash

# Runtime control & scheduling
export PARALLEL_MODE="tp"
export SYNC_MODE="async"
export MIXED_PRECISION="no"
export DYNAMO_BACKEND="no"

# Data / shape
export SEQ_LEN=1024
export ACCUM_STEPS=4

# LoRA
export LORA_R=64
export LORA_ALPHA=16
export LORA_DROPOUT=0.05

# Schedule
export LR=2e-5
export WARMUP_STEPS=100
export TOTAL_STEPS=999999
export DATASET_SIZE=52000
export PIN_MEMORY=1

# Sleep
export PRINT_EVERY=1
export SLEEP_EVERY=0
export SLEEP_SEC=0

# Checkpoint
export CKPT_EVERY=4
export MODEL_SAVE="full"
export INCLUDE_OPTIMIZER=1
export KEEP_LAST=1

# Fixed configuration
export GPU_TYPE="gpu:h200:4"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

for METHOD in "qlora" "full"
do
  for BATCH_SIZE in 1 2 4 8 16 32
  do
    export METHOD
    export BATCH_SIZE

    echo "Submitting: GPU=$GPU_TYPE MODEL=$MODEL_NAME METHOD=$METHOD BATCH_SIZE=$BATCH_SIZE"

    sbatch \
      --export=ALL \
      --gres=${GPU_TYPE} \
      --time=00:30:00 \
      templates/llm_train_hf_test.sh
  done
done

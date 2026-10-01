#!/bin/bash

# Model & training mode (strongest power impact)
export MODEL_NAME="meta-llama/Llama-2-70b-chat-hf"
export METHOD="qlora"             # full | lora | qlora

# Runtime control & scheduling
export PARALLEL_MODE="tp"
export SYNC_MODE="async"          # async | sync | local    (gradient sync behavior / GPU utilization)
export MIXED_PRECISION="bf16"       # bf16 | fp16 | no        (numerical format for compute)
export DYNAMO_BACKEND="no"        # no | inductor | eager   (Torch compile backend)

# Data / shape
export SEQ_LEN=2048               # token sequence length (↑ → more compute, higher power)
export BATCH_SIZE=16              # per-rank microbatch
export ACCUM_STEPS=1              # gradient accumulation (↑ → larger effective batch)

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
export CKPT_EVERY=2
export MODEL_SAVE="lora"
export INCLUDE_OPTIMIZER=1
export KEEP_LAST=2

sbatch --export=ALL templates/llm_train_hf.sh

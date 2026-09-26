#!/bin/bash
# Model / parallel
export MODEL_NAME="meta-llama/Llama-2-70b-chat-hf"
export TP_SIZE=4

# Precision
export DTYPE="bfloat16"

# Static runtime memory budget
export MEM_FRAC_STATIC=0.85

# Prompt source
export PROMPT_SOURCE="realistic"

# Fixed-shape workload
export BATCH_SIZE=1
export PROMPT_LEN=2048
export MIN_NEW_TOKENS=64
export MAX_NEW_TOKENS=64

# Prefix-cache policy (choose one explicitly)
export NONCE_TOKENS=0        # average/realistic

# Deterministic decoding
export DO_SAMPLE=0

# Measurement
export WARMUP_STEPS=10
export STEPS=999999
export PRINT_EVERY=10
export SYNC_EACH_ITER=1

# Metrics
export COUNT_TOTAL_TOKENS=0

# No sleep plateaus
export SLEEP_EVERY=10
export SLEEP_SEC=10

sbatch --export=ALL templates/llm_infer_sglang.sh

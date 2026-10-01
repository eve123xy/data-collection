#!/bin/bash
# Change: BATCH_SIZE=4
export MODEL_NAME="meta-llama/Llama-2-70b-chat-hf"
export PARALLEL_MODE="tp"
export DTYPE="fp16"
export LOAD_IN_4BIT=0
export ATTN_IMPL="eager"

export BATCH_SIZE=2
export PROMPT_LEN=2048
export MAX_NEW_TOKENS=64
export MIN_NEW_TOKENS=64

export DO_SAMPLE=0
export NUM_BEAMS=1
export USE_CACHE=1

export WARMUP_STEPS=10
export STEPS=999999
export PRINT_EVERY=10
export SYNC_EACH_ITER=1
export SLEEP_EVERY=0
export SLEEP_SEC=0

sbatch --export=ALL templates/llm_infer_hf.sh

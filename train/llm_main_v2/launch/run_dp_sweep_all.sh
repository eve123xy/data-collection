#!/bin/bash
set -euo pipefail

# ============================================================
# Submit DP sweep jobs:
#   4 GPU types × 3 Llama-2 models × 3 training methods
#   = 36 Slurm jobs
#
# Profiler is enabled by default in llm_train_hf_dp_sweep_profiler.sh.
# Set ENABLE_PROFILER=0 in sbatch --export for clean power-only runs.
# ============================================================

GPU_TYPES=(
  "gpu:l40s:4"
  "gpu:a100:4"
  "gpu:h100:4"
  "gpu:h200:4"
)

MODELS=(
  "meta-llama/Llama-2-13b-chat-hf"
  "google/gemma-3-12b-it"
)

METHODS=(
  "full"
  "lora"
  "qlora"
)

JOB_SCRIPT="templates/llm_train_hf_dp_sweep.sh"
WALLTIME="02:00:00"

for GPU_TYPE in "${GPU_TYPES[@]}"; do
  for MODEL_NAME in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do

      echo "Submitting:"
      echo "  GPU      = ${GPU_TYPE}"
      echo "  MODEL    = ${MODEL_NAME}"
      echo "  METHOD   = ${METHOD}"
      echo "  PROFILER = rank0, microstep 1"
      echo ""

      sbatch \
        --export=ALL,PARALLEL_MODE=dp,ACCUM_STEPS=4,MODEL_NAME="${MODEL_NAME}",METHOD="${METHOD}",ENABLE_PROFILER=1,PROFILE_MICROSTEP=1,PROFILE_ALL_RANKS=0 \
        --gres="${GPU_TYPE}" \
        --time="${WALLTIME}" \
        "${JOB_SCRIPT}"

    done
  done
done

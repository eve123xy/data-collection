#!/bin/bash
#SBATCH --job-name=sglang-infer
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# ============================================================
# Runtime setup: environment and unified logging
# ============================================================
set -euo pipefail
module purge

# Load cluster Python/CUDA environment
ENV_SH="$HOME/env/env.sh"
source "${ENV_SH}"

# Unified output layout
OUT_ROOT="logs_infer"
JOB_TAG="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
OUT_DIR="${OUT_ROOT}/${JOB_TAG}"
mkdir -p "${OUT_DIR}"

POWER_LOG="${OUT_DIR}/power_trace.csv"
RUN_LOG="${OUT_DIR}/infer_runtime.log"
exec 1>>"${OUT_DIR}/stdout.log" 2>>"${OUT_DIR}/stderr.log"

PHASE="inference"
echo "[JOB] ===== START (${PHASE^^}) ====="
printf "%-18s = %s\n" "DATE"     "$(date)"
printf "%-18s = %s\n" "HOSTNAME" "$(hostname)"
printf "%-18s = %s\n" "JOB_TAG"  "${JOB_TAG}"
printf "%-18s = %s\n" "OUT_DIR"  "${OUT_DIR}"
echo

# ----------------------------
# User-configurable params
# Override via: sbatch --export=VAR=VALUE,...
# Keep defaults aligned with llm_infer_sglang.py (synthetic is default).
# ----------------------------

FRAMEWORK="sglang"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
TP_SIZE="${TP_SIZE:-4}"                         # Must match number of visible GPUs for TP.
DTYPE="${DTYPE:-fp16}"                          # fp16|bf16|f32 (depends on model/engine support)
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"     # 1 or 0
MEM_FRAC_STATIC="${MEM_FRAC_STATIC:-0.85}"

# Prompt/workload shape
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_LEN="${PROMPT_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"

# Prompt source (script default is synthetic)
PROMPT_SOURCE="${PROMPT_SOURCE:-synthetic}"     # synthetic|realistic
PROMPT_BANK_ROOT="${PROMPT_BANK_ROOT:-prompt_banks}"  # Used only when PROMPT_SOURCE=realistic
NONCE_TOKENS="${NONCE_TOKENS:-0}"               # Used only when PROMPT_SOURCE=realistic

# Sampling/decoding
DO_SAMPLE="${DO_SAMPLE:-0}"                     # 1 or 0
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"

# Synthetic prompt knobs (used only when PROMPT_SOURCE=synthetic)
SYNTHETIC_BASE="${SYNTHETIC_BASE:-Hello world. }"
SYNTHETIC_REPEAT="${SYNTHETIC_REPEAT:-10000}"

# Measurement controls
SEED="${SEED:-1234}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
STEPS="${STEPS:-1200}"
PRINT_EVERY="${PRINT_EVERY:-10}"
SYNC_EACH_ITER="${SYNC_EACH_ITER:-1}"           # 1 or 0
COUNT_TOTAL_TOKENS="${COUNT_TOTAL_TOKENS:-0}"   # 1 or 0
VERIFY_PROMPT_LEN="${VERIFY_PROMPT_LEN:-0}"     # 1 or 0 (debug; slow)

# Optional plateaus (for power trace segmentation)
SLEEP_EVERY="${SLEEP_EVERY:-0}"
SLEEP_SEC="${SLEEP_SEC:-0}"

# Offline safety (prevents accidental downloads during profiling)
HF_OFFLINE="${HF_OFFLINE:-1}"                   # 1 or 0

# Print run configuration
echo "[CFG] ===== Runtime config ====="
printf "%-22s = %s\n" "JOB_ID"            "${SLURM_JOB_ID}"
printf "%-22s = %s\n" "HOSTNAME"          "$(hostname)"
printf "%-22s = %s\n" "PHASE"             "${PHASE}"
printf "%-22s = %s\n" "FRAMEWORK"         "${FRAMEWORK}"
printf "%-22s = %s\n" "MODEL_NAME"        "${MODEL_NAME}"
printf "%-22s = %s\n" "TP_SIZE"           "${TP_SIZE}"
printf "%-22s = %s\n" "DTYPE"             "${DTYPE}"
printf "%-22s = %s\n" "TRUST_REMOTE_CODE" "${TRUST_REMOTE_CODE}"
printf "%-22s = %s\n" "MEM_FRAC_STATIC"   "${MEM_FRAC_STATIC}"
printf "%-22s = %s\n" "PROMPT_SOURCE"     "${PROMPT_SOURCE}"
printf "%-22s = %s\n" "PROMPT_LEN"        "${PROMPT_LEN}"
printf "%-22s = %s\n" "BATCH_SIZE"        "${BATCH_SIZE}"
printf "%-22s = %s\n" "MAX_NEW_TOKENS"    "${MAX_NEW_TOKENS}"
printf "%-22s = %s\n" "MIN_NEW_TOKENS"    "${MIN_NEW_TOKENS}"
printf "%-22s = %s\n" "NONCE_TOKENS"      "${NONCE_TOKENS}"
printf "%-22s = %s\n" "DO_SAMPLE"         "${DO_SAMPLE}"
printf "%-22s = %s\n" "TEMPERATURE"       "${TEMPERATURE}"
printf "%-22s = %s\n" "TOP_P"             "${TOP_P}"
printf "%-22s = %s\n" "TOP_K"             "${TOP_K}"
printf "%-22s = %s\n" "SEED"              "${SEED}"
printf "%-22s = %s\n" "WARMUP_STEPS"      "${WARMUP_STEPS}"
printf "%-22s = %s\n" "STEPS"             "${STEPS}"
printf "%-22s = %s\n" "PRINT_EVERY"       "${PRINT_EVERY}"
printf "%-22s = %s\n" "SYNC_EACH_ITER"    "${SYNC_EACH_ITER}"
printf "%-22s = %s\n" "COUNT_TOTAL_TOKENS" "${COUNT_TOTAL_TOKENS}"
printf "%-22s = %s\n" "VERIFY_PROMPT_LEN" "${VERIFY_PROMPT_LEN}"
printf "%-22s = %s\n" "SLEEP_EVERY"       "${SLEEP_EVERY}"
printf "%-22s = %s\n" "SLEEP_SEC"         "${SLEEP_SEC}"
printf "%-22s = %s\n" "POWER_LOG"         "${POWER_LOG}"
printf "%-22s = %s\n" "RUN_LOG"           "${RUN_LOG}"
echo ""

# ----------------------------
# Hardware snapshot
# ----------------------------
set +e
num_gpus="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
set -e
echo "[HW]   ===== Hardware ====="
printf "%-22s = %s\n" "NUM_GPUS" "${num_gpus:-0}"
printf "%-22s = %s\n" "GPU_NAME" "${gpu_name:-Unknown}"
echo

# ----------------------------
# Offline mode (recommended for profiling)
# ----------------------------
if [[ "${HF_OFFLINE}" == "1" ]]; then
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  echo "[ENV] HF/Datasets/Transformers offline mode enabled."
else
  echo "[ENV] Offline mode disabled (may download models/datasets)."
fi

# ----------------------------
# Power logger (100ms sampling)
# ----------------------------
echo "[JOB] start nvidia-smi power logging..."
nvidia-smi \
  --query-gpu=timestamp,index,power.draw,clocks.sm,utilization.gpu,utilization.memory \
  --format=csv -lms 100 > "${POWER_LOG}" &
PWR_PID=$!

cleanup() {
  echo "[JOB] stopping power logger (pid=${PWR_PID})..."
  kill "${PWR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ----------------------------
# Build Python flags
# ----------------------------
EXTRA=()

# Boolean flags (convert 1/0 env vars into CLI flags)
if [[ "${DO_SAMPLE}" == "1" ]]; then EXTRA+=( --do_sample ); fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then EXTRA+=( --trust_remote_code ); fi
if [[ "${SYNC_EACH_ITER}" == "1" ]]; then EXTRA+=( --sync_each_iter ); fi
if [[ "${COUNT_TOTAL_TOKENS}" == "1" ]]; then EXTRA+=( --count_total_tokens ); fi
if [[ "${VERIFY_PROMPT_LEN}" == "1" ]]; then EXTRA+=( --verify_prompt_len ); fi

# Prompt mode flags
EXTRA+=( --prompt_source "${PROMPT_SOURCE}" )

# Realistic-only knobs (safe to pass; script will only use them in realistic mode)
EXTRA+=( --prompt_bank_root "${PROMPT_BANK_ROOT}" )
EXTRA+=( --nonce_tokens "${NONCE_TOKENS}" )

# Synthetic-only knobs (safe to pass; script will only use them in synthetic mode)
EXTRA+=( --synthetic_base "${SYNTHETIC_BASE}" )
EXTRA+=( --synthetic_repeat "${SYNTHETIC_REPEAT}" )

# ----------------------------
# Run burn loop
# ----------------------------
echo "[JOB] run SGLang offline inference burn..."
python -u pybench/llm_infer_sglang.py \
  --model "${MODEL_NAME}" \
  --tp_size "${TP_SIZE}" \
  --dtype "${DTYPE}" \
  --mem_fraction_static "${MEM_FRAC_STATIC}" \
  --batch_size "${BATCH_SIZE}" \
  --prompt_len "${PROMPT_LEN}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --min_new_tokens "${MIN_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --seed "${SEED}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --steps "${STEPS}" \
  --print_every "${PRINT_EVERY}" \
  --sleep_every "${SLEEP_EVERY}" \
  --sleep_sec "${SLEEP_SEC}" \
  "${EXTRA[@]}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "[JOB] DONE"
echo "[JOB] power log : ${POWER_LOG}"
echo "[JOB] run log   : ${RUN_LOG}"
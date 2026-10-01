#!/bin/bash
#SBATCH --job-name=hf-infer
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4                       # GPUs per node
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

# ---- User-configurable params (override via --export) ----
FRAMEWORK="hf"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
PARALLEL_MODE="${PARALLEL_MODE:-tp}"      # tp|dp
DTYPE="${DTYPE:-fp16}"                    # fp16|bf16
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"         # 1 or 0
ATTN_IMPL="${ATTN_IMPL:-eager}"           # eager|sdpa|flash_attention_2

BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_LEN="${PROMPT_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"

DO_SAMPLE="${DO_SAMPLE:-0}"               # 1 or 0
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
REP_PEN="${REP_PEN:-1.0}"
LEN_PEN="${LEN_PEN:-1.0}"
EARLY_STOP="${EARLY_STOP:-0}"             # 1 or 0
USE_CACHE="${USE_CACHE:-1}"               # 1 or 0

SEED="${SEED:-1234}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
STEPS="${STEPS:-200}"
PRINT_EVERY="${PRINT_EVERY:-10}"
SYNC_EACH_ITER="${SYNC_EACH_ITER:-1}"     # 1 or 0
SLEEP_EVERY="${SLEEP_EVERY:-0}"
SLEEP_SEC="${SLEEP_SEC:-0}"

# Print run configuration
echo "[CFG] ===== Runtime config ====="
printf "%-18s = %s\n" "JOB_ID"         "${SLURM_JOB_ID}"
printf "%-18s = %s\n" "HOSTNAME"       "$(hostname)"
printf "%-18s = %s\n" "PHASE"          "${PHASE}"
printf "%-18s = %s\n" "FRAMEWORK"      "${FRAMEWORK}"
printf "%-18s = %s\n" "MODEL_NAME"     "${MODEL_NAME}"
printf "%-18s = %s\n" "PARALLEL_MODE"  "${PARALLEL_MODE}"
printf "%-18s = %s\n" "DTYPE"          "${DTYPE}"
printf "%-18s = %s\n" "LOAD_IN_4BIT"   "${LOAD_IN_4BIT}"
printf "%-18s = %s\n" "ATTN_IMPL"      "${ATTN_IMPL}"
printf "%-18s = %s\n" "BATCH_SIZE"     "${BATCH_SIZE}"
printf "%-18s = %s\n" "PROMPT_LEN"     "${PROMPT_LEN}"
printf "%-18s = %s\n" "MAX_NEW_TOKENS" "${MAX_NEW_TOKENS}"
printf "%-18s = %s\n" "MIN_NEW_TOKENS" "${MIN_NEW_TOKENS}"
printf "%-18s = %s\n" "DO_SAMPLE"      "${DO_SAMPLE}"
printf "%-18s = %s\n" "TEMPERATURE"    "${TEMPERATURE}"
printf "%-18s = %s\n" "TOP_P"          "${TOP_P}"
printf "%-18s = %s\n" "TOP_K"          "${TOP_K}"
printf "%-18s = %s\n" "NUM_BEAMS"      "${NUM_BEAMS}"
printf "%-18s = %s\n" "USE_CACHE"      "${USE_CACHE}"
printf "%-18s = %s\n" "SEED"           "${SEED}"
printf "%-18s = %s\n" "WARMUP_STEPS"   "${WARMUP_STEPS}"
printf "%-18s = %s\n" "STEPS"          "${STEPS}"
printf "%-18s = %s\n" "PRINT_EVERY"    "${PRINT_EVERY}"
printf "%-18s = %s\n" "SYNC_EACH_ITER" "${SYNC_EACH_ITER}"
printf "%-18s = %s\n" "SLEEP_EVERY"    "${SLEEP_EVERY}"
printf "%-18s = %s\n" "SLEEP_SEC"      "${SLEEP_SEC}"
printf "%-18s = %s\n" "POWER_LOG"      "${POWER_LOG}"
printf "%-18s = %s\n" "RUN_LOG"        "${RUN_LOG}"
echo ""

# ---- GPU snapshot ----
set +e
num_gpus="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
set -e
echo "[HW]   ===== Hardware ====="
printf "%-20s = %s\n" "NUM_GPUS" "${num_gpus:-0}"
printf "%-20s = %s\n" "GPU_NAME" "${gpu_name:-Unknown}"
echo

# ---- Power logger ----
echo "[JOB] start nvidia-smi power logging..."
nvidia-smi \
  --query-gpu=timestamp,index,power.draw,clocks.sm,utilization.gpu,utilization.memory \
  --format=csv -lms 100 > "${POWER_LOG}" &

# ---- Build flags ----
EXTRA=()
if [[ "${LOAD_IN_4BIT}" == "1" ]]; then EXTRA+=( --load_in_4bit ); fi
if [[ "${DO_SAMPLE}" == "1" ]]; then EXTRA+=( --do_sample ); fi
if [[ "${EARLY_STOP}" == "1" ]]; then EXTRA+=( --early_stopping ); fi
if [[ "${USE_CACHE}" == "1" ]]; then EXTRA+=( --use_cache ); fi
if [[ "${SYNC_EACH_ITER}" == "1" ]]; then EXTRA+=( --sync_each_iter ); fi

# ---- Run inference burn ----
echo "[JOB] run inference burn..."
python -u pybench/llm_infer_hf.py \
  --model "${MODEL_NAME}" \
  --parallel_mode "${PARALLEL_MODE}" \
  --dtype "${DTYPE}" \
  --attn_impl "${ATTN_IMPL}" \
  --batch_size "${BATCH_SIZE}" \
  --prompt_len "${PROMPT_LEN}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --min_new_tokens "${MIN_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --num_beams "${NUM_BEAMS}" \
  --repetition_penalty "${REP_PEN}" \
  --length_penalty "${LEN_PEN}" \
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

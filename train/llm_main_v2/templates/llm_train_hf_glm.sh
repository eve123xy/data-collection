#!/bin/bash
#SBATCH --job-name=hf-train
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/logs_glm/%x-%j.out               # STDOUT log
#SBATCH --error=logs/logs_glm/%x-%j.err                # STDERR log

# =========================
# Setup: env + unified logs
# =========================
set -euo pipefail
module purge

ENV_SH="/scratch/$USER/datacenter/llm_main/env.sh"
source "${ENV_SH}"

LOG_ROOT="logs/logs_glm"
CKPT_ROOT="ckpt_train"
JOB_TAG="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
LOG_DIR="${LOG_ROOT}/${JOB_TAG}"
CKPT_DIR="${CKPT_ROOT}/${JOB_TAG}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CKPT_DIR}"

POWER_LOG="${LOG_DIR}/power_trace.csv"
TRAIN_LOG="${LOG_DIR}/train_runtime.log"

echo "[JOB] ===== START ====="
printf "%-18s = %s\n" "DATE"     "$(date)"
printf "%-18s = %s\n" "ENV_SH"   "${ENV_SH}"
printf "%-18s = %s\n" "JOB_TAG"  "${JOB_TAG}"
printf "%-18s = %s\n" "LOG_DIR"  "${LOG_DIR}"
printf "%-18s = %s\n" "CKPT_DIR" "${CKPT_DIR}"
echo ""

# ==============================================
# Training config (override via sbatch --export)
# ==============================================
PHASE="train"
FRAMEWORK="hf"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
METHOD="${METHOD:-full}"                    # full | lora | qlora
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # bf16 | fp16 | no
DYNAMO_BACKEND="${DYNAMO_BACKEND:-no}"      # no | inductor | eager

SEQ_LEN="${SEQ_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"               # per-rank microbatch
ACCUM_STEPS="${ACCUM_STEPS:-1}"             # gradient accumulation steps

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

LR="${LR:-1e-6}"                            # base LR
WARMUP_STEPS="${WARMUP_STEPS:-10}"          # warmup steps
TOTAL_STEPS="${TOTAL_STEPS:-999999}"
DATASET_SIZE="${DATASET_SIZE:-52000}"
PIN_MEMORY="${PIN_MEMORY:-1}"               # 1 -> --pin_memory

# sleep
PRINT_EVERY="${PRINT_EVERY:-1}"
SLEEP_EVERY="${SLEEP_EVERY:-0}"             # 0 disables
SLEEP_SEC="${SLEEP_SEC:-0}"                 # seconds
# checkpointing
CKPT_EVERY="${CKPT_EVERY:-1}"               # 0 disables
MODEL_SAVE="${MODEL_SAVE:-lora}"            # full | lora
INCLUDE_OPTIMIZER="${INCLUDE_OPTIMIZER:-0}" # 1 saves optimizer+scheduler
KEEP_LAST="${KEEP_LAST:-1}"                 # 0 keeps all

echo "[CFG] ===== Model ====="
printf "%-18s = %s\n" "PHASE"        "${PHASE}"
printf "%-18s = %s\n" "FRAMEWORK"    "${FRAMEWORK}"
printf "%-18s = %s\n" "MODEL"        "${MODEL_NAME}"
printf "%-18s = %s\n" "METHOD"       "${METHOD}"
printf "%-18s = %s\n" "MIXED_PREC"   "${MIXED_PRECISION}"
printf "%-18s = %s\n" "DYNAMO"       "${DYNAMO_BACKEND}"
echo "[CFG] ===== Data ====="
printf "%-18s = %s\n" "SEQ_LEN"      "${SEQ_LEN}"
printf "%-18s = %s\n" "BATCH_SIZE"   "${BATCH_SIZE}"
printf "%-18s = %s\n" "ACCUM_STEPS"  "${ACCUM_STEPS}"
echo "[CFG] ===== LoRA ====="
printf "%-18s = %s\n" "LORA_R"       "${LORA_R}"
printf "%-18s = %s\n" "LORA_ALPHA"   "${LORA_ALPHA}"
printf "%-18s = %s\n" "LORA_DROPOUT" "${LORA_DROPOUT}"
echo "[CFG] ===== Schedule ====="
printf "%-18s = %s\n" "LR"           "${LR}"
printf "%-18s = %s\n" "WARMUP"       "${WARMUP_STEPS}"
printf "%-18s = %s\n" "TOTAL_STEPS"  "${TOTAL_STEPS}"
printf "%-18s = %s\n" "DATASET_SIZE" "${DATASET_SIZE}"
printf "%-18s = %s\n" "PIN_MEMORY"   "${PIN_MEMORY}"
echo "[CFG] ===== Sleep ====="
printf "%-18s = %s\n" "PRINT_EVERY"  "${PRINT_EVERY}"
printf "%-18s = %s\n" "SLEEP_EVERY"  "${SLEEP_EVERY}"
printf "%-18s = %s\n" "SLEEP_SEC"    "${SLEEP_SEC}"
echo "[CFG] ===== Checkpoint ====="
printf "%-18s = %s\n" "CKPT_DIR"     "${CKPT_DIR}"
printf "%-18s = %s\n" "CKPT_EVERY"   "${CKPT_EVERY}"
printf "%-18s = %s\n" "MODEL_SAVE"   "${MODEL_SAVE}"
printf "%-18s = %s\n" "INCLUDE_OPT"  "${INCLUDE_OPTIMIZER}"
printf "%-18s = %s\n" "KEEP_LAST"    "${KEEP_LAST}"
echo

# ====================
# Slurm + distributed
# ====================
NODELIST="${SLURM_NODELIST:-}"
NNODES="${SLURM_NNODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"

MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR=""
if [[ -n "${NODELIST}" ]]; then
  MASTER_ADDR="$(scontrol show hostnames "${NODELIST}" 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${MASTER_ADDR}" ]]; then MASTER_ADDR="$(hostname)"; fi

# Prefer Slurm allocation; fall back to local detection
# - Handle common Slurm formats:
#   SLURM_GPUS_PER_NODE: "4"
#   SLURM_GPUS_ON_NODE : "0,1,2,3" or "gpu:4" or "4"
GPUS_RAW="${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-0}}"
GPUS_ALLOC="0"

if [[ "${GPUS_RAW}" =~ ^[0-9]+$ ]]; then
  GPUS_ALLOC="${GPUS_RAW}"
elif [[ "${GPUS_RAW}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  # comma-separated list of device indices
  GPUS_ALLOC=$(( $(tr -cd ',' <<< "${GPUS_RAW}" | wc -c) + 1 ))
elif [[ "${GPUS_RAW}" =~ ^gpu:([0-9]+)$ ]]; then
  GPUS_ALLOC="${BASH_REMATCH[1]}"
fi

if [[ -z "${GPUS_ALLOC}" || "${GPUS_ALLOC}" == "0" ]]; then
  GPUS_ALLOC="$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')"
fi

PARALLEL_MODE="${PARALLEL_MODE:-dp}"     # dp | tp
SYNC_MODE="${SYNC_MODE:-async}"          # async | sync | local

if [[ "${PARALLEL_MODE}" == "tp" ]]; then
  NUM_PROCS_PER_NODE=1
  # Respect Slurm-provided CUDA_VISIBLE_DEVICES when present.
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "${GPUS_ALLOC}" -gt 0 ]]; then
    CUDA_VISIBLE_DEVICES="$(seq 0 $((GPUS_ALLOC - 1)) | paste -sd, -)"
    export CUDA_VISIBLE_DEVICES
  fi
  export HF_TP_MODE=1                    # python may use device_map="auto"
else
  NUM_PROCS_PER_NODE="${GPUS_ALLOC}"
  unset HF_TP_MODE || true
fi

WORLD_SIZE=$(( NNODES * NUM_PROCS_PER_NODE ))

echo "[SLURM] ===== Env ====="
printf "%-18s = %s\n" "HOSTNAME"     "$(hostname)"
printf "%-18s = %s\n" "JOB_ID"       "${SLURM_JOB_ID}"
printf "%-18s = %s\n" "NODELIST"     "${NODELIST}"
printf "%-18s = %s\n" "NNODES"       "${NNODES}"
printf "%-18s = %s\n" "NODE_RANK"    "${NODE_RANK}"
printf "%-18s = %s\n" "CPUS_TASK"    "${SLURM_CPUS_PER_TASK:-0}"
printf "%-18s = %s\n" "MEM_REQ"      "${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-Unknown}}"
printf "%-18s = %s\n" "GPUS_ALLOC"   "${GPUS_ALLOC}"
echo ""

echo "[DIST] ===== Distributed ====="
printf "%-18s = %s\n" "PARALLEL_MODE" "${PARALLEL_MODE}"
printf "%-18s = %s\n" "SYNC_MODE"     "${SYNC_MODE}"
printf "%-18s = %s\n" "MASTER_ADDR"   "${MASTER_ADDR}"
printf "%-18s = %s\n" "MASTER_PORT"   "${MASTER_PORT}"
printf "%-18s = %s\n" "PROCS_NODE"    "${NUM_PROCS_PER_NODE}"
printf "%-18s = %s\n" "WORLD_SIZE"    "${WORLD_SIZE}"
printf "%-18s = %s\n" "CUDA_DEV"      "${CUDA_VISIBLE_DEVICES:-N/A}"
printf "%-18s = %s\n" "HF_TP_MODE"    "${HF_TP_MODE:-0}"
echo ""

# =========================
# Hardware snapshot
# =========================

# IMPORTANT: probing commands should NOT abort the whole job
set +e

NUM_GPUS_LOCAL="$(timeout 2s nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}' || true)"
GPU_NAME_FIRST="$(timeout 2s nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
GPU_DRIVER="$(timeout 2s nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"

# Fix: strip trailing "| ..." and trim spaces, so output is like "13.0"
CUDA_VER="$(timeout 2s nvidia-smi 2>/dev/null | awk -F'CUDA Version:' '
  /CUDA Version/ {
    v=$2
    sub(/\|.*/,"",v)                 # drop " | ..."
    gsub(/^[ \t]+|[ \t]+$/,"",v)     # trim
    print v; exit
  }' || true)"

GPU_MEM_MIB_FIRST="$(timeout 2s nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
  | head -n1 | awk '{print ($1==""?0:$1)}' || true)"
GPU_MEM_GIB_FIRST=$(( (${GPU_MEM_MIB_FIRST:-0}) / 1024 ))

# Fix: force English output from lscpu to avoid locale mismatch -> "Unknown"
CPU_MODEL="$(LANG=C lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/,"",$2); print $2; exit}' || true)"
CPU_SOCKETS="$(LANG=C lscpu 2>/dev/null | awk -F: '/Socket\(s\)/ {gsub(/^[ \t]+/,"",$2); print $2; exit}' || true)"
CPU_THREADS="$(LANG=C lscpu 2>/dev/null | awk -F: '/^CPU\(s\)/ {gsub(/^[ \t]+/,"",$2); print $2; exit}' || true)"

MEM_TOTAL="$(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || true)"
MEM_FREE="$(free -h 2>/dev/null | awk '/^Mem:/{print $7}' || true)"

# restore strict mode for the rest of the script
set -e

echo "[HW] ===== Snapshot ====="
printf "%-18s = %s\n" "GPUS_SLURM"  "${GPUS_ALLOC:-Unknown}"
printf "%-18s = %s\n" "NUM_GPUS"    "${NUM_GPUS_LOCAL:-Unknown}"
printf "%-18s = %s\n" "GPU_NAME"    "${GPU_NAME_FIRST:-Unknown}"
printf "%-18s = %s\n" "GPU_MEM"     "${GPU_MEM_GIB_FIRST:-0} GiB"
printf "%-18s = %s\n" "DRIVER"      "${GPU_DRIVER:-Unknown}"
printf "%-18s = %s\n" "CUDA_VER"    "${CUDA_VER:-Unknown}"
printf "%-18s = %s\n" "CPU_MODEL"   "${CPU_MODEL:-Unknown}"
printf "%-18s = %s\n" "CPU_SOCKETS" "${CPU_SOCKETS:-Unknown}"
printf "%-18s = %s\n" "CPU_THREADS" "${CPU_THREADS:-Unknown}"
printf "%-18s = %s\n" "MEM_TOTAL"   "${MEM_TOTAL:-Unknown}"
printf "%-18s = %s\n" "MEM_FREE"    "${MEM_FREE:-Unknown}"
echo ""

# ============================================================
# Start power logger in background
# - Sample GPU power every 100 ms
# - Stop it on exit to avoid trailing logging after training ends
# ============================================================
echo "[JOB] start nvidia-smi power logging..."
nvidia-smi \
  --query-gpu=timestamp,index,power.draw,clocks.sm,utilization.gpu,utilization.memory \
  --format=csv -lms 100 > "${POWER_LOG}" &
PWR_PID=$!
trap 'kill ${PWR_PID} 2>/dev/null || true' EXIT
echo ""

# =====================
# Flags + accelerate args
# =====================

# Build extra flags for python based on PIN_MEMORY
PIN_FLAG=""
if [[ "${PIN_MEMORY}" == "1" ]]; then
  PIN_FLAG="--pin_memory"
fi

# Build accelerate args (support WORLD_SIZE=1 or >1)
# ============================================================
ACC_ARGS=()
if [[ "${WORLD_SIZE}" -eq 1 ]]; then
  # Single-process run: let Accelerate handle defaults.
  ACC_ARGS+=( --num_processes=1 )
else
  # Multi-process distributed run.
  # Note: this job script requests a single node. If you later increase --nodes,
  # consider launching via srun and setting --num_processes="${WORLD_SIZE}".
  ACC_ARGS+=(
    --num_machines="${NNODES}"
    --num_processes="${NUM_PROCS_PER_NODE}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
  )
fi

# ============================================================
# Launch distributed training via accelerate
# - tee stdout/stderr into TRAIN_LOG
# ============================================================
echo "[JOB] ===== RUN TRAINING ====="
accelerate launch \
  "${ACC_ARGS[@]}" \
  --mixed_precision="${MIXED_PRECISION}" \
  --dynamo_backend="${DYNAMO_BACKEND}" \
  pybench/llm_train_hf.py \
    --model             "${MODEL_NAME}" \
    --method            "${METHOD}" \
    --sync_mode         "${SYNC_MODE}" \
    --seq_len           "${SEQ_LEN}" \
    --batch_size        "${BATCH_SIZE}" \
    --accum_steps       "${ACCUM_STEPS}" \
    --lora_r            "${LORA_R}" \
    --lora_alpha        "${LORA_ALPHA}" \
    --lora_dropout      "${LORA_DROPOUT}" \
    --lr                "${LR}" \
    --warmup_steps      "${WARMUP_STEPS}" \
    --total_steps       "${TOTAL_STEPS}" \
    --dataset_size      "${DATASET_SIZE}" \
    --print_every       "${PRINT_EVERY}" \
    --sleep_every       "${SLEEP_EVERY}" \
    --sleep_sec         "${SLEEP_SEC}" \
    --ckpt_dir          "${CKPT_DIR}" \
    --ckpt_every        "${CKPT_EVERY}" \
    --model_save        "${MODEL_SAVE}" \
    --include_optimizer "${INCLUDE_OPTIMIZER}" \
    --keep_last         "${KEEP_LAST}"
    ${PIN_FLAG} \
    2>&1 | tee -a "${TRAIN_LOG}"



echo ""
echo "[JOB] DONE at $(date)"

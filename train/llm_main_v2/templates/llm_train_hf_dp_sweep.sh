#!/bin/bash
#SBATCH --job-name=hf-dp-sweep
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/logs/%x-%j.out
#SBATCH --error=logs/logs/%x-%j.err

# One Slurm allocation, six independent Accelerate launches (one per group).
# Within each group, configs increase from small to large.
# Each config runs exactly 4 microsteps = 1 optimizer step.
# CUDA OOM ends only the current group process; the shell then starts the next group.
# PyTorch Profiler profiles one selected microstep per config (rank 0 by default).

set -euo pipefail
module purge

ENV_SH="/scratch/$USER/datacenter/llm_main/env.sh"
source "${ENV_SH}"

export PYTHONUNBUFFERED=1
export PARALLEL_MODE="dp"

# -------------------------
# User-overridable settings
# -------------------------
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
METHOD="${METHOD:-full}"                    # full | lora | qlora
MIXED_PRECISION="${MIXED_PRECISION:-no}"  # no | bf16 | fp16
DYNAMO_BACKEND="${DYNAMO_BACKEND:-no}"
ACCUM_STEPS=4
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
PIN_MEMORY="${PIN_MEMORY:-1}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"

# Profiler: default ON for this diagnostic version.
# Set ENABLE_PROFILER=0 for a clean power-only run.
ENABLE_PROFILER="${ENABLE_PROFILER:-1}"
PROFILE_MICROSTEP="${PROFILE_MICROSTEP:-1}"   # 1-based; first microstep avoids optimizer-step effects
PROFILE_ALL_RANKS="${PROFILE_ALL_RANKS:-0}" # 0 = rank 0 only, 1 = all DP ranks

# Sweep values
BS_VALUES="1,2,4,8,16,32,64,128"
SEQ_VALUES="1,2,4,8,16,32,64,128,256,512,1024,2048,4096"

# Put the Python file here, or override PY_SCRIPT when submitting.
PY_SCRIPT="${PY_SCRIPT:-pybench/llm_train_hf_dp_sweep.py}"

LOG_ROOT="logs/logs"
JOB_TAG="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
LOG_DIR="${LOG_ROOT}/${JOB_TAG}"
GROUP_LOG_DIR="${LOG_DIR}/groups"
PROFILER_DIR="${LOG_DIR}/profiler"
mkdir -p "${GROUP_LOG_DIR}" "${PROFILER_DIR}"

POWER_LOG="${LOG_DIR}/power_trace.csv"
EVENT_LOG="${LOG_DIR}/events.csv"
SUMMARY_LOG="${LOG_DIR}/sweep_summary.csv"
TRAIN_LOG="${LOG_DIR}/train_runtime.log"

printf "group,sweep_type,fixed_value,result,exit_code\n" > "${SUMMARY_LOG}"

# -------------------------
# Detect allocated GPUs
# -------------------------
GPUS_RAW="${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-0}}"
GPUS_ALLOC=0
if [[ "${GPUS_RAW}" =~ ^[0-9]+$ ]]; then
  GPUS_ALLOC="${GPUS_RAW}"
elif [[ "${GPUS_RAW}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  GPUS_ALLOC=$(( $(tr -cd ',' <<< "${GPUS_RAW}" | wc -c) + 1 ))
elif [[ "${GPUS_RAW}" =~ ^gpu:([0-9]+)$ ]]; then
  GPUS_ALLOC="${BASH_REMATCH[1]}"
fi
if [[ -z "${GPUS_ALLOC}" || "${GPUS_ALLOC}" -eq 0 ]]; then
  GPUS_ALLOC="$(nvidia-smi -L | wc -l | awk '{print $1}')"
fi

if [[ "${GPUS_ALLOC}" -ne 4 ]]; then
  echo "[ERROR] This experiment expects exactly 4 GPUs for DP, got ${GPUS_ALLOC}." >&2
  exit 2
fi

# -------------------------
# Hardware / Slurm metadata
# -------------------------
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | xargs)"
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | xargs)"
CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
CPU_MODEL="$(lscpu | awk -F: '/Model name/{gsub(/^[ \t]+/,"",$2); print $2; exit}')"
CPU_SOCKETS="$(lscpu | awk -F: '/Socket\(s\)/{gsub(/^[ \t]+/,"",$2); print $2; exit}')"
CPU_THREADS="$(lscpu | awk -F: '/^CPU\(s\):/{gsub(/^[ \t]+/,"",$2); print $2; exit}')"
MEM_TOTAL="$(free -h | awk '/^Mem:/{print $2}')"
MEM_FREE="$(free -h | awk '/^Mem:/{print $7}')"
NODELIST="${SLURM_JOB_NODELIST:-$(hostname)}"
NNODES="${SLURM_NNODES:-1}"
MASTER_ADDR="$(scontrol show hostnames "${NODELIST}" 2>/dev/null | head -n 1 || hostname)"

# -------------------------
# Print configuration -- keep the metadata fields used by the normal runs
# -------------------------
echo "[JOB] ===== START ====="
printf "%-20s = %s\n" "DATE" "$(date)"
printf "%-20s = %s\n" "ENV_SH" "${ENV_SH}"
printf "%-20s = %s\n" "JOB_TAG" "${JOB_TAG}"
printf "%-20s = %s\n" "LOG_DIR" "${LOG_DIR}"
echo

echo "[CFG] ===== Model ====="
printf "%-20s = %s\n" "PHASE" "train"
printf "%-20s = %s\n" "FRAMEWORK" "hf"
printf "%-20s = %s\n" "MODEL" "${MODEL_NAME}"
printf "%-20s = %s\n" "METHOD" "${METHOD}"
printf "%-20s = %s\n" "MIXED_PREC" "${MIXED_PRECISION}"
printf "%-20s = %s\n" "DYNAMO" "${DYNAMO_BACKEND}"
printf "%-20s = %s\n" "ATTN_IMPL" "eager"

echo "[CFG] ===== Data / Sweep ====="
printf "%-20s = %s\n" "BATCH_SIZE" "sweep:${BS_VALUES}"
printf "%-20s = %s\n" "SEQ_LEN" "sweep:${SEQ_VALUES}"
printf "%-20s = %s\n" "ACCUM_STEPS" "${ACCUM_STEPS}"
printf "%-20s = %s\n" "DATASET" "${TRAIN_DATASET:-<hf-dataset-id>}"
printf "%-20s = %s\n" "PIN_MEMORY" "${PIN_MEMORY}"

echo "[CFG] ===== LoRA ====="
printf "%-20s = %s\n" "LORA_R" "${LORA_R}"
printf "%-20s = %s\n" "LORA_ALPHA" "${LORA_ALPHA}"
printf "%-20s = %s\n" "LORA_DROPOUT" "${LORA_DROPOUT}"

echo "[CFG] ===== Schedule ====="
printf "%-20s = %s\n" "LR" "${LR}"
printf "%-20s = %s\n" "WARMUP" "${WARMUP_STEPS}"
printf "%-20s = %s\n" "MICROSTEPS/CONFIG" "${ACCUM_STEPS}"
printf "%-20s = %s\n" "OPT_STEPS/CONFIG" "1"
printf "%-20s = %s\n" "CHECKPOINT" "disabled"

echo "[CFG] ===== Profiler ====="
printf "%-20s = %s\n" "ENABLE_PROFILER" "${ENABLE_PROFILER}"
printf "%-20s = %s\n" "PROFILE_MICROSTEP" "${PROFILE_MICROSTEP}"
printf "%-20s = %s\n" "PROFILE_ALL_RANKS" "${PROFILE_ALL_RANKS}"
printf "%-20s = %s\n" "PROFILER_DIR" "${PROFILER_DIR}"

echo

echo "[SLURM] ===== Env ====="
printf "%-20s = %s\n" "HOSTNAME" "$(hostname)"
printf "%-20s = %s\n" "JOB_ID" "${SLURM_JOB_ID:-NA}"
printf "%-20s = %s\n" "NODELIST" "${NODELIST}"
printf "%-20s = %s\n" "NNODES" "${NNODES}"
printf "%-20s = %s\n" "CPUS_TASK" "${SLURM_CPUS_PER_TASK:-NA}"
printf "%-20s = %s\n" "MEM_REQ" "${SLURM_MEM_PER_NODE:-NA}"
printf "%-20s = %s\n" "GPUS_ALLOC" "${GPUS_ALLOC}"

echo

echo "[DIST] ===== Distributed ====="
printf "%-20s = %s\n" "PARALLEL_MODE" "dp"
printf "%-20s = %s\n" "MASTER_ADDR" "${MASTER_ADDR}"
printf "%-20s = %s\n" "MASTER_PORT_BASE" "${MASTER_PORT_BASE}"
printf "%-20s = %s\n" "PROCS_NODE" "${GPUS_ALLOC}"
printf "%-20s = %s\n" "WORLD_SIZE" "$((NNODES * GPUS_ALLOC))"
printf "%-20s = %s\n" "CUDA_DEV" "${CUDA_VISIBLE_DEVICES:-all allocated GPUs}"

echo

echo "[HW] ===== Snapshot ====="
printf "%-20s = %s\n" "GPUS_SLURM" "${GPUS_ALLOC}"
printf "%-20s = %s\n" "NUM_GPUS" "$(nvidia-smi -L | wc -l | awk '{print $1}')"
printf "%-20s = %s\n" "GPU_NAME" "${GPU_NAME}"
printf "%-20s = %s MiB\n" "GPU_MEM" "${GPU_MEM_MIB}"
printf "%-20s = %s\n" "DRIVER" "${DRIVER_VER}"
printf "%-20s = %s\n" "CUDA_VER" "${CUDA_VER:-NA}"
printf "%-20s = %s\n" "CPU_MODEL" "${CPU_MODEL}"
printf "%-20s = %s\n" "CPU_SOCKETS" "${CPU_SOCKETS}"
printf "%-20s = %s\n" "CPU_THREADS" "${CPU_THREADS}"
printf "%-20s = %s\n" "MEM_TOTAL" "${MEM_TOTAL}"
printf "%-20s = %s\n" "MEM_FREE" "${MEM_FREE}"

echo
printf "%-20s = %s\n" "POWER_LOG" "${POWER_LOG}"
printf "%-20s = %s\n" "EVENT_LOG" "${EVENT_LOG}"
printf "%-20s = %s\n" "SUMMARY_LOG" "${SUMMARY_LOG}"
printf "%-20s = %s\n" "TRAIN_LOG" "${TRAIN_LOG}"
printf "%-20s = %s\n" "PY_SCRIPT" "${PY_SCRIPT}"
echo

# -------------------------
# One continuous power logger for the entire Slurm job
# -------------------------
echo "[JOB] start nvidia-smi power logging at 100 ms..."
nvidia-smi \
  --query-gpu=timestamp,index,power.draw,power.draw.instant,clocks.sm,utilization.gpu,utilization.memory,temperature.gpu,memory.used \
  --format=csv \
  -lms 100 > "${POWER_LOG}" &
PWR_PID=$!

cleanup() {
  kill "${PWR_PID}" 2>/dev/null || true
  wait "${PWR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

sleep 1
if ! kill -0 "${PWR_PID}" 2>/dev/null; then
  echo "[ERROR] nvidia-smi power logger failed to start." >&2
  exit 3
fi

PIN_FLAG=()
if [[ "${PIN_MEMORY}" == "1" ]]; then
  PIN_FLAG+=(--pin_memory)
fi

PROFILE_FLAG=()
if [[ "${ENABLE_PROFILER}" == "1" ]]; then
  PROFILE_FLAG+=(--profile --profiler_dir "${PROFILER_DIR}" --profile_microstep "${PROFILE_MICROSTEP}")
  if [[ "${PROFILE_ALL_RANKS}" == "1" ]]; then
    PROFILE_FLAG+=(--profile_all_ranks)
  fi
fi

run_group() {
  local idx="$1"
  local group="$2"
  local sweep_type="$3"
  local fixed_value="$4"
  local values="$5"
  local port=$((MASTER_PORT_BASE + idx))
  local group_log="${GROUP_LOG_DIR}/${group}.log"

  echo
  echo "============================================================"
  echo "[GROUP] ${group} | type=${sweep_type} | fixed=${fixed_value} | values=${values} | port=${port}"
  echo "============================================================"

  # Disable -e only around this launch so CUDA OOM does not kill the Slurm job.
  set +e
  accelerate launch \
    --num_machines=1 \
    --num_processes="${GPUS_ALLOC}" \
    --main_process_port="${port}" \
    --mixed_precision="${MIXED_PRECISION}" \
    --dynamo_backend="${DYNAMO_BACKEND}" \
    "${PY_SCRIPT}" \
      --model "${MODEL_NAME}" \
      --method "${METHOD}" \
      --group "${group}" \
      --sweep_type "${sweep_type}" \
      --fixed_value "${fixed_value}" \
      --values "${values}" \
      --accum_steps "${ACCUM_STEPS}" \
      --lr "${LR}" \
      --warmup_steps "${WARMUP_STEPS}" \
      --event_log "${EVENT_LOG}" \
      --lora_r "${LORA_R}" \
      --lora_alpha "${LORA_ALPHA}" \
      --lora_dropout "${LORA_DROPOUT}" \
      "${PIN_FLAG[@]}" \
      "${PROFILE_FLAG[@]}" \
    2>&1 | tee -a "${TRAIN_LOG}" "${group_log}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    echo "[GROUP] ${group}: completed all candidates."
    printf "%s,%s,%s,success,%s\n" "${group}" "${sweep_type}" "${fixed_value}" "${rc}" >> "${SUMMARY_LOG}"
    return 0
  fi

  # Continue only when the current group failed because of CUDA OOM.
  if grep -Eqi 'CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError|CUDA error: out of memory' "${group_log}"; then
    echo "[GROUP] ${group}: CUDA OOM reached -> moving to next group."
    printf "%s,%s,%s,oom,%s\n" "${group}" "${sweep_type}" "${fixed_value}" "${rc}" >> "${SUMMARY_LOG}"
    return 0
  fi

  echo "[ERROR] ${group} failed for a non-OOM reason (exit=${rc}). Stop to avoid hiding a real error." >&2
  printf "%s,%s,%s,error,%s\n" "${group}" "${sweep_type}" "${fixed_value}" "${rc}" >> "${SUMMARY_LOG}"
  exit "${rc}"
}

# ============================================================
# Sweep A: fixed sequence length; increase per-rank microbatch
# ============================================================
run_group 0 "seq512"  "fixed_seq" 512  "${BS_VALUES}"
run_group 1 "seq1024" "fixed_seq" 1024 "${BS_VALUES}"
run_group 2 "seq2048" "fixed_seq" 2048 "${BS_VALUES}"

# ============================================================
# Sweep B: fixed per-rank microbatch; increase sequence length
# ============================================================
run_group 3 "bs1" "fixed_batch" 1 "${SEQ_VALUES}"
run_group 4 "bs4" "fixed_batch" 4 "${SEQ_VALUES}"
run_group 5 "bs8" "fixed_batch" 8 "${SEQ_VALUES}"

echo
echo "[JOB] ===== ALL GROUPS FINISHED ====="
echo "[JOB] power    : ${POWER_LOG}"
echo "[JOB] events   : ${EVENT_LOG}"
echo "[JOB] summary  : ${SUMMARY_LOG}"
echo "[JOB] profiler : ${PROFILER_DIR}"
echo "[JOB] end      : $(date)"

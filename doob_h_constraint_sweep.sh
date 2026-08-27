#!/bin/bash
#SBATCH --job-name=doob_h_constraint_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=1-10%2

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export WANDB_MODE=offline

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch doob_h_constraint_sweep.sh GUIDE_DIR_OR_NAME [SWEEP_NAME]"
    exit 1
fi

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${FT_MODEL:-ft_periodic_seed0}"
GUIDE_DIR_ARG="$1"
SWEEP_NAME="${2:-d48_l2_6000}"
MAX_CONSTRAINTS="${MAX_CONSTRAINTS:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
MAX_CORRECTION="${MAX_CORRECTION:-5.0}"
MAX_LOG_H_RATIO="${MAX_LOG_H_RATIO:-10.0}"
H_CANDIDATE_BATCH_SIZE="${H_CANDIDATE_BATCH_SIZE:-65536}"
COLUMN_ORDER_CSV="${COLUMN_ORDER_CSV:-0,1,2,3,4,5,6,7,8,9}"
LEVEL="${SLURM_ARRAY_TASK_ID}"

if [ "${LEVEL}" -lt 1 ] || [ "${LEVEL}" -gt "${MAX_CONSTRAINTS}" ]; then
    echo "ERROR: array index ${LEVEL} must be in 1..${MAX_CONSTRAINTS}"
    exit 1
fi

IFS=',' read -r -a COLUMN_ORDER <<< "${COLUMN_ORDER_CSV}"
if [ "${#COLUMN_ORDER[@]}" -ne "${MAX_CONSTRAINTS}" ]; then
    echo "ERROR: COLUMN_ORDER_CSV must contain ${MAX_CONSTRAINTS} column indices"
    exit 1
fi

declare -A SEEN_COLUMNS=()
for COLUMN in "${COLUMN_ORDER[@]}"; do
    if ! [[ "${COLUMN}" =~ ^[0-9]+$ ]] || [ "${COLUMN}" -ge "${MAX_CONSTRAINTS}" ]; then
        echo "ERROR: invalid numerical column index ${COLUMN}"
        exit 1
    fi
    if [ -n "${SEEN_COLUMNS[$COLUMN]:-}" ]; then
        echo "ERROR: duplicate numerical column index ${COLUMN}"
        exit 1
    fi
    SEEN_COLUMNS[$COLUMN]=1
done

ACTIVE_COLUMNS=""
for ((INDEX = 0; INDEX < LEVEL; INDEX++)); do
    if [ -n "${ACTIVE_COLUMNS}" ]; then
        ACTIVE_COLUMNS+=","
    fi
    ACTIVE_COLUMNS+="${COLUMN_ORDER[$INDEX]}"
done

CKPT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
CKPT_CANDIDATES=("${CKPT_DIR}"/best_ema_model_*.pt)
if [ ! -e "${CKPT_CANDIDATES[0]}" ] || [ "${#CKPT_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one best EMA checkpoint in ${CKPT_DIR}"
    printf '%s\n' "${CKPT_CANDIDATES[@]}"
    exit 1
fi
BASE_CKPT="${CKPT_CANDIDATES[0]}"

if [[ "${GUIDE_DIR_ARG}" == */* ]]; then
    GUIDE_DIR="${GUIDE_DIR_ARG}"
else
    GUIDE_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/${GUIDE_DIR_ARG}"
fi
GUIDE_CKPT="${GUIDE_DIR}/best_guide.pt"
if [ ! -f "${GUIDE_CKPT}" ]; then
    echo "ERROR: guide checkpoint not found: ${GUIDE_CKPT}"
    exit 1
fi

printf -v LEVEL_PADDED '%02d' "${LEVEL}"
OUTPUT="conditional_samples/${DATANAME}/${MODEL_NAME}_constraint_sweep_${SWEEP_NAME}_k${LEVEL_PADDED}.csv"

echo "========================================"
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Array task        : ${SLURM_ARRAY_TASK_ID}"
echo "Dataset           : ${DATANAME}"
echo "Model             : ${MODEL_NAME}"
echo "Guide checkpoint  : ${GUIDE_CKPT}"
echo "Constraint count  : ${LEVEL}/${MAX_CONSTRAINTS}"
echo "Nested order      : ${COLUMN_ORDER_CSV}"
echo "Active columns    : ${ACTIVE_COLUMNS}"
echo "Samples           : ${NUM_SAMPLES}"
echo "Output            : ${OUTPUT}"
echo "========================================"
nvidia-smi

python -u sample_doob_h.py \
    --guide-ckpt "${GUIDE_CKPT}" \
    --base-ckpt "${BASE_CKPT}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-correction "${MAX_CORRECTION}" \
    --max-log-h-ratio "${MAX_LOG_H_RATIO}" \
    --h-candidate-batch-size "${H_CANDIDATE_BATCH_SIZE}" \
    --active-columns "${ACTIVE_COLUMNS}" \
    --categorical-start-mode section4_posterior \
    --output "${OUTPUT}" \
    --device cuda

echo "Finished constraint-count level ${LEVEL}"

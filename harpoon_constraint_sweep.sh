#!/bin/bash
#SBATCH --job-name=harpoon_constraint_sweep
#SBATCH --output=harpoon_logs/%x_%A_%a.out
#SBATCH --error=harpoon_logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-10%2

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
SWEEP_NAME="${1:-summed_relu_eta02}"
QUERY_FILE="${QUERY_FILE:-constraints/${DATANAME}/fixed_numerical_intervals.json}"
HARPOON_RUNTIME="${HARPOON_RUNTIME:-/scratch/work/agrawaa4/harpoon_runtime}"
CHECKPOINT="${CHECKPOINT:-${HARPOON_RUNTIME}/saved_models/${DATANAME}/diffputer_selfmade.pt}"
MAX_CONSTRAINTS="${MAX_CONSTRAINTS:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.2}"
COLUMN_ORDER_CSV="${COLUMN_ORDER_CSV:-0,1,2,3,4,5,6,7,8,9}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
FIXED_QUERY_PREFIX="${FIXED_QUERY_PREFIX:-qf_fixed_box}"
SEED_BASE="${SEED_BASE:-42}"
ORDERINGS_FILE="${ORDERINGS_FILE:-}"
if [ -n "${ORDERINGS_FILE}" ]; then
    if [ ! -f "${ORDERINGS_FILE}" ]; then
        echo "ERROR: orderings file not found: ${ORDERINGS_FILE}"
        exit 1
    fi
    GLOBAL_TASK="${SLURM_ARRAY_TASK_ID}"
    ORDERING_INDEX=$((GLOBAL_TASK / MAX_CONSTRAINTS))
    LEVEL=$((GLOBAL_TASK % MAX_CONSTRAINTS + 1))
    ORDERING_LINE=$(sed -n "$((ORDERING_INDEX + 1))p" "${ORDERINGS_FILE}")
    if [ -z "${ORDERING_LINE}" ]; then
        echo "ERROR: no ordering ${ORDERING_INDEX} in ${ORDERINGS_FILE}"
        exit 1
    fi
    IFS='|' read -r ORDERING_ID COLUMN_ORDER_CSV <<< "${ORDERING_LINE}"
    FIXED_QUERY_PREFIX="${FIXED_QUERY_PREFIX}_${ORDERING_ID}"
else
    LEVEL="${SLURM_ARRAY_TASK_ID}"
fi

if [ "${DATANAME}" != "shoppers" ]; then
    echo "ERROR: the HARPOON comparison is currently Shoppers-only"
    exit 1
fi
for path in "${QUERY_FILE}" "${CHECKPOINT}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}"
        exit 1
    fi
done
if [ "${LEVEL}" -lt 1 ] || [ "${LEVEL}" -gt "${MAX_CONSTRAINTS}" ]; then
    echo "ERROR: array index ${LEVEL} must be in 1..${MAX_CONSTRAINTS}"
    exit 1
fi

IFS=',' read -r -a COLUMN_ORDER <<< "${COLUMN_ORDER_CSV}"
if [ "${#COLUMN_ORDER[@]}" -ne "${MAX_CONSTRAINTS}" ]; then
    echo "ERROR: COLUMN_ORDER_CSV must contain ${MAX_CONSTRAINTS} model indices"
    exit 1
fi

declare -A SEEN_COLUMNS=()
ACTIVE_COLUMNS=""
for ((INDEX = 0; INDEX < MAX_CONSTRAINTS; INDEX++)); do
    COLUMN="${COLUMN_ORDER[$INDEX]}"
    if ! [[ "${COLUMN}" =~ ^[0-9]+$ ]] || [ "${COLUMN}" -ge "${MAX_CONSTRAINTS}" ]; then
        echo "ERROR: invalid numerical model index ${COLUMN}"
        exit 1
    fi
    if [ -n "${SEEN_COLUMNS[$COLUMN]:-}" ]; then
        echo "ERROR: duplicate numerical model index ${COLUMN}"
        exit 1
    fi
    SEEN_COLUMNS[$COLUMN]=1
    if [ "${INDEX}" -lt "${LEVEL}" ]; then
        if [ -n "${ACTIVE_COLUMNS}" ]; then
            ACTIVE_COLUMNS+=","
        fi
        ACTIVE_COLUMNS+="${COLUMN}"
    fi
done

printf -v LEVEL_PADDED '%02d' "${LEVEL}"
if [ -n "${OUTPUT_ROOT}" ]; then
    mkdir -p "${OUTPUT_ROOT}"
    OUTPUT="${OUTPUT_ROOT}/${FIXED_QUERY_PREFIX}_k${LEVEL_PADDED}.csv"
else
    OUTPUT="conditional_samples/${DATANAME}/harpoon_constraint_sweep_${SWEEP_NAME}_k${LEVEL_PADDED}.csv"
fi

echo "========================================"
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Array task        : ${SLURM_ARRAY_TASK_ID}"
echo "Dataset           : ${DATANAME}"
echo "Checkpoint        : ${CHECKPOINT}"
echo "Runtime data      : ${HARPOON_RUNTIME}"
echo "Constraint count  : ${LEVEL}/${MAX_CONSTRAINTS}"
echo "Nested order      : ${COLUMN_ORDER_CSV}"
echo "Active columns    : ${ACTIVE_COLUMNS}"
echo "AND construction  : sum of squared ReLU interval losses"
echo "Guidance eta      : ${GUIDANCE_SCALE}"
echo "Samples           : ${NUM_SAMPLES}"
echo "Output            : ${OUTPUT}"
echo "========================================"
if [ -f "${OUTPUT}" ] && [ -f "${OUTPUT%.csv}.constraints.json" ]; then
    echo "Existing completed sample found; skipping ${OUTPUT}"
    exit 0
fi
nvidia-smi

python -u sample_harpoon_fixed_box.py \
    --dataname "${DATANAME}" \
    --harpoon-root baselines/harpoon \
    --runtime-root "${HARPOON_RUNTIME}" \
    --checkpoint "${CHECKPOINT}" \
    --query-file "${QUERY_FILE}" \
    --active-columns "${ACTIVE_COLUMNS}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --seed "$((SEED_BASE + LEVEL - 1))" \
    --output "${OUTPUT}" \
    --device cuda

echo "Finished HARPOON constraint-count level ${LEVEL}"

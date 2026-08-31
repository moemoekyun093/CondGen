#!/bin/bash
#SBATCH --job-name=doob_bound_ckpt_samples
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:16g,min-cuda-cc:70
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

BASE_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
BASE_CANDIDATES=("${BASE_DIR}"/best_ema_model_*.pt)
if [ ! -e "${BASE_CANDIDATES[0]}" ] || [ "${#BASE_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one base EMA checkpoint in ${BASE_DIR}"
    exit 1
fi
if [ ! -f "${QUERY_DIR}/${QUERY_ID}.json" ]; then
    echo "ERROR: query not found: ${QUERY_DIR}/${QUERY_ID}.json"
    exit 1
fi

IFS=',' read -r -a TASK_IDS <<< "${MISSING_TASKS_CSV:?missing task list was not exported}"
echo "Bundled bound-embedding checkpoint sampling"
echo "Tasks : ${MISSING_TASKS_CSV}"
echo "Count : ${#TASK_IDS[@]}"
nvidia-smi

for TASK_ID in "${TASK_IDS[@]}"; do
    CHECKPOINT_INDEX=$((TASK_ID % 10 + 1))
    STEP=$((CHECKPOINT_INDEX * 200))
    SERIES_INDEX=$((TASK_ID / 10))
    if [ "${SERIES_INDEX}" -eq 0 ]; then
        SERIES_LABEL="monotone"
        GUIDE_DIR="${MONOTONE_GUIDE_DIR}"
    elif [ "${SERIES_INDEX}" -eq 1 ]; then
        SERIES_LABEL="mlp"
        GUIDE_DIR="${MLP_GUIDE_DIR}"
    else
        echo "ERROR: invalid bundled task ${TASK_ID}"
        exit 1
    fi

    GUIDE_CKPT="${GUIDE_DIR}/guide_${STEP}.pt"
    OUTPUT="${BOUND_SAMPLE_ROOT}/${SERIES_LABEL}/step_$(printf '%04d' "${STEP}").csv"
    if [ ! -f "${GUIDE_CKPT}" ]; then
        echo "ERROR: checkpoint not found: ${GUIDE_CKPT}"
        exit 1
    fi
    mkdir -p "$(dirname "${OUTPUT}")"
    if [ -f "${OUTPUT}" ] && [ -f "${OUTPUT%.csv}.constraints.json" ]; then
        echo "Reusing ${OUTPUT}"
        continue
    fi

    echo "----------------------------------------"
    echo "Embedding  : ${SERIES_LABEL}"
    echo "Train step : ${STEP}"
    echo "Checkpoint : ${GUIDE_CKPT}"
    echo "Output     : ${OUTPUT}"
    python -u sample_doob_query.py \
        --guide-ckpt "${GUIDE_CKPT}" \
        --base-ckpt "${BASE_CANDIDATES[0]}" \
        --query-file "${QUERY_DIR}/${QUERY_ID}.json" \
        --num-samples "${NUM_SAMPLES:-1000}" \
        --batch-size "${NUM_SAMPLES:-1000}" \
        --seed "$((54000 + CHECKPOINT_INDEX))" \
        --output "${OUTPUT}" \
        --device cuda
done

echo "Finished ${#TASK_IDS[@]} bundled sampling tasks"

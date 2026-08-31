#!/bin/bash
#SBATCH --job-name=doob_presence_ckpt_sample
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"
CHECKPOINT_INDEX=$((TASK_ID % 10 + 1))
STEP=$((CHECKPOINT_INDEX * 200))
SERIES_INDEX=$((TASK_ID / 10))
if [ "${SERIES_INDEX}" -eq 0 ]; then
    SERIES_LABEL="active_flags"
    GUIDE_DIR="${ACTIVE_FLAGS_GUIDE_DIR}"
elif [ "${SERIES_INDEX}" -eq 1 ]; then
    SERIES_LABEL="implicit_domain"
    GUIDE_DIR="${IMPLICIT_DOMAIN_GUIDE_DIR}"
else
    echo "ERROR: task ${TASK_ID} has invalid series index ${SERIES_INDEX}"
    exit 1
fi

GUIDE_CKPT="${GUIDE_DIR}/guide_${STEP}.pt"
OUTPUT="${ABLATION_SAMPLE_ROOT}/${SERIES_LABEL}/step_$(printf '%04d' "${STEP}").csv"
BASE_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
BASE_CANDIDATES=("${BASE_DIR}"/best_ema_model_*.pt)
if [ ! -e "${BASE_CANDIDATES[0]}" ] || [ "${#BASE_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one base EMA checkpoint in ${BASE_DIR}"
    exit 1
fi
for REQUIRED in "${GUIDE_CKPT}" "${QUERY_DIR}/${QUERY_ID}.json"; do
    if [ ! -f "${REQUIRED}" ]; then
        echo "ERROR: required file not found: ${REQUIRED}"
        exit 1
    fi
done
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ] && [ -f "${OUTPUT%.csv}.constraints.json" ]; then
    echo "Reusing ${OUTPUT}"
    exit 0
fi

echo "Series     : ${SERIES_LABEL}"
echo "Train step : ${STEP}"
echo "Checkpoint : ${GUIDE_CKPT}"
echo "Output     : ${OUTPUT}"
nvidia-smi

python -u sample_doob_query.py \
    --guide-ckpt "${GUIDE_CKPT}" \
    --base-ckpt "${BASE_CANDIDATES[0]}" \
    --query-file "${QUERY_DIR}/${QUERY_ID}.json" \
    --num-samples "${NUM_SAMPLES:-1000}" \
    --batch-size "${NUM_SAMPLES:-1000}" \
    --seed "$((44000 + CHECKPOINT_INDEX))" \
    --output "${OUTPUT}" \
    --device cuda

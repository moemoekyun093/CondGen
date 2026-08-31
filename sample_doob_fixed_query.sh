#!/bin/bash
#SBATCH --job-name=doob_fixed_query_sample
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

CKPT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
BASE_CANDIDATES=("${CKPT_DIR}"/best_ema_model_*.pt)
if [ ! -e "${BASE_CANDIDATES[0]}" ] || [ "${#BASE_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one base EMA checkpoint in ${CKPT_DIR}"
    exit 1
fi
BASE_CKPT="${BASE_CANDIDATES[0]}"
GUIDE_CKPT="${GUIDE_DIR}/best_guide.pt"
QUERY_FILE="${QUERY_DIR}/${QUERY_ID}.json"
mkdir -p "$(dirname "${SAMPLE_OUTPUT}")"
if [ -f "${SAMPLE_OUTPUT}" ] && [ -f "${SAMPLE_OUTPUT%.csv}.constraints.json" ]; then
    echo "Reusing completed fixed-query sample: ${SAMPLE_OUTPUT}"
    exit 0
fi

python -u sample_doob_query.py \
    --guide-ckpt "${GUIDE_CKPT}" \
    --base-ckpt "${BASE_CKPT}" \
    --query-file "${QUERY_FILE}" \
    --num-samples "${NUM_SAMPLES:-1000}" \
    --batch-size "${NUM_SAMPLES:-1000}" \
    --seed 24001 \
    --output "${SAMPLE_OUTPUT}" \
    --device cuda

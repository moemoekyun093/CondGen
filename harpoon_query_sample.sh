#!/bin/bash
#SBATCH --job-name=harpoon_query
#SBATCH --output=harpoon_logs/%x_%j.out
#SBATCH --error=harpoon_logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
QUERY_FILE="${QUERY_FILE:-data90/${DATANAME}/queries_full/qf_shoppers_b00p5_4.json}"
QUERY_ID="$(basename "${QUERY_FILE}" .json)"
HARPOON_ROOT="${HARPOON_ROOT:-baselines/harpoon}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/scratch/work/agrawaa4/harpoon_runtime}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/saved_models/${DATANAME}/diffputer_selfmade.pt}"
OUTPUT="${OUTPUT:-conditional_samples/${DATANAME}/harpoon_${QUERY_ID}_eta02.csv}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.2}"

mkdir -p harpoon_logs "$(dirname "${OUTPUT}")"

echo "Query       : ${QUERY_FILE}"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Eta         : ${GUIDANCE_SCALE}"
echo "Output      : ${OUTPUT}"

python -u sample_harpoon_full_query.py \
    --dataname "${DATANAME}" \
    --query-file "${QUERY_FILE}" \
    --harpoon-root "${HARPOON_ROOT}" \
    --runtime-root "${RUNTIME_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --num-samples "${NUM_SAMPLES}" \
    --guidance-scale "${GUIDANCE_SCALE}"

echo "Finished HARPOON full-query sampling"

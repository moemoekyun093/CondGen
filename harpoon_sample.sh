#!/bin/bash
#SBATCH --job-name=harpoon_sample
#SBATCH --output=harpoon_logs/%x_%j.out
#SBATCH --error=harpoon_logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-adult}"
QUERY_FILE="${QUERY_FILE:-constraints/${DATANAME}/fixed_numerical_intervals.json}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.2}"
OUTPUT="${OUTPUT:-conditional_samples/${DATANAME}/harpoon_partial.csv}"

python -u sample_harpoon_fixed_box.py \
    --dataname "${DATANAME}" \
    --query-file "${QUERY_FILE}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --output "${OUTPUT}" \
    --device cuda

echo "Finished HARPOON baseline sampling"

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

DATANAME="${DATANAME:-shoppers}"
if [ "${DATANAME}" != "shoppers" ]; then
    echo "ERROR: the paper-aligned HARPOON baseline is currently Shoppers-only"
    exit 1
fi
NUM_SAMPLES="${NUM_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.2}"
OUTPUT="${OUTPUT:-conditional_samples/shoppers/harpoon_paper_range.csv}"
LOWER_BOUNDS="${LOWER_BOUNDS:-Administrative=4}"
UPPER_BOUNDS="${UPPER_BOUNDS:-}"
CONSTRAINT_ARGS=()
SAMPLE_COUNT_ARGS=()
if [ -n "${NUM_SAMPLES}" ]; then
    SAMPLE_COUNT_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi
IFS=',' read -r -a LOWER_BOUND_ITEMS <<< "${LOWER_BOUNDS}"
for ITEM in "${LOWER_BOUND_ITEMS[@]}"; do
    [ -n "${ITEM}" ] && CONSTRAINT_ARGS+=(--lower-bound "${ITEM}")
done
IFS=',' read -r -a UPPER_BOUND_ITEMS <<< "${UPPER_BOUNDS}"
for ITEM in "${UPPER_BOUND_ITEMS[@]}"; do
    [ -n "${ITEM}" ] && CONSTRAINT_ARGS+=(--upper-bound "${ITEM}")
done

echo "HARPOON paper constraint: lower={${LOWER_BOUNDS:-none}} upper={${UPPER_BOUNDS:-none}}"
echo "HARPOON eta: ${GUIDANCE_SCALE}"

python -u sample_harpoon_fixed_box.py \
    --dataname "${DATANAME}" \
    "${SAMPLE_COUNT_ARGS[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    "${CONSTRAINT_ARGS[@]}" \
    --output "${OUTPUT}" \
    --device cuda

echo "Finished HARPOON baseline sampling"

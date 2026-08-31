#!/bin/bash
#SBATCH --job-name=harpoon_fixed_query_sample
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

QUERY_FILE="${QUERY_DIR}/${QUERY_ID}.json"
HARPOON_ROOT="${HARPOON_ROOT:-baselines/harpoon}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/scratch/work/agrawaa4/harpoon_runtime}"
HARPOON_CHECKPOINT="${HARPOON_CHECKPOINT:-${RUNTIME_ROOT}/saved_models/${DATANAME}/diffputer_selfmade.pt}"
mkdir -p harpoon_logs "$(dirname "${HARPOON_SAMPLE}")"

if [ -f "${HARPOON_SAMPLE}" ] && [ -f "${HARPOON_SAMPLE%.csv}.constraints.json" ]; then
    echo "Reusing completed HARPOON fixed-query sample: ${HARPOON_SAMPLE}"
    exit 0
fi
for REQUIRED in "${QUERY_FILE}" "${HARPOON_CHECKPOINT}"; do
    if [ ! -f "${REQUIRED}" ]; then
        echo "ERROR: required file not found: ${REQUIRED}"
        exit 1
    fi
done

echo "HARPOON fixed query: ${QUERY_ID}"
echo "Guidance eta      : ${HARPOON_GUIDANCE_SCALE:-0.2}"
echo "Output            : ${HARPOON_SAMPLE}"
nvidia-smi

python -u sample_harpoon_full_query.py \
    --dataname "${DATANAME}" \
    --query-file "${QUERY_FILE}" \
    --allow-partial-query \
    --harpoon-root "${HARPOON_ROOT}" \
    --runtime-root "${RUNTIME_ROOT}" \
    --checkpoint "${HARPOON_CHECKPOINT}" \
    --output "${HARPOON_SAMPLE}" \
    --num-samples "${HARPOON_NUM_SAMPLES:-1000}" \
    --batch-size "${HARPOON_NUM_SAMPLES:-1000}" \
    --guidance-scale "${HARPOON_GUIDANCE_SCALE:-0.2}" \
    --seed 34001 \
    --device cuda

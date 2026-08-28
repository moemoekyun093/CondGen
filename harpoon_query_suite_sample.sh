#!/bin/bash
#SBATCH --job-name=harpoon_suite
#SBATCH --output=harpoon_logs/%x_%A_%a.out
#SBATCH --error=harpoon_logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
HARPOON_ROOT="${HARPOON_ROOT:-baselines/harpoon}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/scratch/work/agrawaa4/harpoon_runtime}"
CHECKPOINT="${HARPOON_CHECKPOINT:-${RUNTIME_ROOT}/saved_models/${DATANAME}/diffputer_selfmade.pt}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_suite_comparison}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_eta02}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
GUIDANCE_SCALE="${HARPOON_GUIDANCE_SCALE:-0.2}"
SEED_BASE="${SEED_BASE:-10000}"

mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}")
QUERY_INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"
if [ "${QUERY_INDEX}" -lt 0 ] || [ "${QUERY_INDEX}" -ge "${#QUERY_FILES[@]}" ]; then
    echo "ERROR: query array index ${QUERY_INDEX} is out of range"
    exit 1
fi
if [[ ! "${HARPOON_LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: unsafe HARPOON label: ${HARPOON_LABEL}"
    exit 1
fi

QUERY_FILE="${QUERY_FILES[QUERY_INDEX]}"
QUERY_ID="$(basename "${QUERY_FILE}" .json)"
OUTPUT_DIR="${SUITE_SAMPLE_ROOT}/${HARPOON_LABEL}"
OUTPUT="${OUTPUT_DIR}/${QUERY_ID}.csv"
for path in "${QUERY_FILE}" "${CHECKPOINT}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}"
        exit 1
    fi
done
mkdir -p harpoon_logs "${OUTPUT_DIR}"
if [ -f "${OUTPUT}" ] && [ -f "${OUTPUT%.csv}.constraints.json" ]; then
    echo "Existing completed HARPOON sample found; skipping ${QUERY_ID}"
    exit 0
fi

echo "HARPOON query ${QUERY_ID} ($((QUERY_INDEX + 1))/${#QUERY_FILES[@]})"
echo "Checkpoint ${CHECKPOINT}"
echo "Guidance eta ${GUIDANCE_SCALE}"
echo "Output ${OUTPUT}"
nvidia-smi

python -u sample_harpoon_full_query.py \
    --dataname "${DATANAME}" \
    --query-file "${QUERY_FILE}" \
    --allow-partial-query \
    --harpoon-root "${HARPOON_ROOT}" \
    --runtime-root "${RUNTIME_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --seed "$((SEED_BASE + QUERY_INDEX))" \
    --device cuda

#!/bin/bash
#SBATCH --job-name=harpoon_tabdiff
#SBATCH --output=harpoon_logs/%x_%A_%a.out
#SBATCH --error=harpoon_logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/doob_vs_harpoon_style_s50}"
HARPOON_STYLE_LABEL="${HARPOON_STYLE_LABEL:-harpoon_style_tabdiff_eta02_s50}"
HARPOON_STYLE_ETA="${HARPOON_STYLE_ETA:-0.2}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
SEED_BASE="${SEED_BASE:-10000}"

mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}")
QUERY_INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"
if [ "${QUERY_INDEX}" -lt 0 ] || [ "${QUERY_INDEX}" -ge "${#QUERY_FILES[@]}" ]; then
    echo "ERROR: query array index ${QUERY_INDEX} is out of range"
    exit 1
fi
if [[ ! "${HARPOON_STYLE_LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: unsafe method label: ${HARPOON_STYLE_LABEL}"
    exit 1
fi

QUERY_FILE="${QUERY_FILES[QUERY_INDEX]}"
QUERY_ID="$(basename "${QUERY_FILE}" .json)"
OUTPUT_DIR="${SUITE_SAMPLE_ROOT}/${HARPOON_STYLE_LABEL}"
OUTPUT="${OUTPUT_DIR}/${QUERY_ID}.csv"
CKPT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
CKPT_CANDIDATES=("${CKPT_DIR}"/best_ema_model_*.pt)
if [ ! -e "${CKPT_CANDIDATES[0]}" ] || [ "${#CKPT_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one best EMA base checkpoint in ${CKPT_DIR}"
    exit 1
fi
BASE_CKPT="${CKPT_CANDIDATES[0]}"

mkdir -p harpoon_logs "${OUTPUT_DIR}"
if [ -f "${OUTPUT}" ] && [ -f "${OUTPUT%.csv}.constraints.json" ]; then
    echo "Existing completed sample found; skipping ${HARPOON_STYLE_LABEL}/${QUERY_ID}"
    exit 0
fi

echo "Method HARPOON-style guidance on frozen TabDiff"
echo "Query ${QUERY_ID} ($((QUERY_INDEX + 1))/${#QUERY_FILES[@]})"
echo "Base checkpoint ${BASE_CKPT}"
echo "Guidance eta ${HARPOON_STYLE_ETA}"
echo "Reverse steps ${NUM_TIMESTEPS}"
echo "Output ${OUTPUT}"
nvidia-smi

python -u sample_harpoon_style_tabdiff.py \
    --dataname "${DATANAME}" \
    --base-ckpt "${BASE_CKPT}" \
    --query-file "${QUERY_FILE}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --eta "${HARPOON_STYLE_ETA}" \
    --num-timesteps "${NUM_TIMESTEPS}" \
    --seed "$((SEED_BASE + QUERY_INDEX))" \
    --output "${OUTPUT}" \
    --device cuda

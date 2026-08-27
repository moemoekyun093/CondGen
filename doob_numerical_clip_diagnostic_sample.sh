#!/bin/bash
#SBATCH --job-name=doob_clip_diag
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
GUIDE_DIR="${DOOB_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_query_curriculum_d48_l2_12000}"
OUTPUT_ROOT="${CLIP_DIAGNOSTIC_ROOT:-conditional_samples/${DATANAME}/numerical_clip_diagnostic}"
CLIP_CAPS="${CLIP_CAPS:-2,5,10,20}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
SEED_BASE="${SEED_BASE:-20000}"

mapfile -t QUERY_FILES < <(
    python list_accepted_queries.py "${QUERY_DIR}" --one-per-band
)
IFS=',' read -r -a CAPS <<< "${CLIP_CAPS}"
NUM_QUERIES="${#QUERY_FILES[@]}"
NUM_CAPS="${#CAPS[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"
EXPECTED_TASKS=$((NUM_QUERIES * NUM_CAPS))
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${EXPECTED_TASKS}" ]; then
    echo "ERROR: array task ${TASK_ID} is outside the diagnostic grid"
    exit 1
fi
CAP_INDEX=$((TASK_ID / NUM_QUERIES))
QUERY_INDEX=$((TASK_ID % NUM_QUERIES))
CAP="${CAPS[CAP_INDEX]}"
QUERY_FILE="${QUERY_FILES[QUERY_INDEX]}"
QUERY_ID="$(basename "${QUERY_FILE}" .json)"
CAP_LABEL="${CAP//./p}"
OUTPUT_DIR="${OUTPUT_ROOT}/cap_${CAP_LABEL}"
OUTPUT="${OUTPUT_DIR}/${QUERY_ID}.csv"
GUIDE_CKPT="${GUIDE_DIR}/best_guide.pt"

CKPT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
CKPT_CANDIDATES=("${CKPT_DIR}"/best_ema_model_*.pt)
if [ ! -e "${CKPT_CANDIDATES[0]}" ] || [ "${#CKPT_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one best EMA base checkpoint in ${CKPT_DIR}"
    exit 1
fi
BASE_CKPT="${CKPT_CANDIDATES[0]}"
for path in "${QUERY_FILE}" "${GUIDE_CKPT}" "${BASE_CKPT}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}"
        exit 1
    fi
done
mkdir -p "${OUTPUT_DIR}"
if [ -f "${OUTPUT}" ] && [ -f "${OUTPUT%.csv}.guidance.json" ]; then
    echo "Existing clipping diagnostic found; skipping cap=${CAP} query=${QUERY_ID}"
    exit 0
fi

echo "Query ${QUERY_ID} ($((QUERY_INDEX + 1))/${NUM_QUERIES})"
echo "Numerical correction cap ${CAP} ($((CAP_INDEX + 1))/${NUM_CAPS})"
echo "Output ${OUTPUT}"
nvidia-smi

python -u sample_doob_query.py \
    --guide-ckpt "${GUIDE_CKPT}" \
    --base-ckpt "${BASE_CKPT}" \
    --query-file "${QUERY_FILE}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-correction "${CAP}" \
    --seed "$((SEED_BASE + QUERY_INDEX))" \
    --diagnose-guidance \
    --output "${OUTPUT}" \
    --device cuda

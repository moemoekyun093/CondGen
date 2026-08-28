#!/bin/bash
#SBATCH --job-name=doob_suite_sample
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
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
GUIDE_SPECS="${GUIDE_SPECS:-curriculum=tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_query_curriculum_d48_l2_12000}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_suite}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
SEED_BASE="${SEED_BASE:-10000}"
SEED_BY_ARITY="${SEED_BY_ARITY:-0}"

mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}")
IFS=',' read -r -a MODEL_SPECS <<< "${GUIDE_SPECS}"
NUM_QUERIES="${#QUERY_FILES[@]}"
NUM_MODELS="${#MODEL_SPECS[@]}"
EXPECTED_TASKS=$((NUM_QUERIES * NUM_MODELS))
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${EXPECTED_TASKS}" ]; then
    echo "ERROR: array task ${TASK_ID} is outside 0..$((EXPECTED_TASKS - 1))"
    exit 1
fi

MODEL_INDEX=$((TASK_ID / NUM_QUERIES))
QUERY_INDEX=$((TASK_ID % NUM_QUERIES))
SPEC="${MODEL_SPECS[MODEL_INDEX]}"
if [[ "${SPEC}" != *=* ]]; then
    echo "ERROR: GUIDE_SPECS entries must use LABEL=GUIDE_DIRECTORY"
    exit 1
fi
LABEL="${SPEC%%=*}"
GUIDE_DIR="${SPEC#*=}"
if [[ ! "${LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: unsafe model label: ${LABEL}"
    exit 1
fi
QUERY_FILE="${QUERY_FILES[QUERY_INDEX]}"
QUERY_ID="$(basename "${QUERY_FILE}" .json)"
GUIDE_CKPT="${GUIDE_DIR}/best_guide.pt"
OUTPUT_DIR="${SUITE_SAMPLE_ROOT}/${LABEL}"
OUTPUT="${OUTPUT_DIR}/${QUERY_ID}.csv"
SAMPLE_SEED="$((SEED_BASE + QUERY_INDEX))"
if [ "${SEED_BY_ARITY}" = "1" ]; then
    if [[ "${QUERY_ID}" =~ _k([0-9]+)$ ]]; then
        SAMPLE_SEED="$((SEED_BASE + 10#${BASH_REMATCH[1]} - 1))"
    else
        echo "ERROR: SEED_BY_ARITY=1 requires a query id ending in _kNN"
        exit 1
    fi
fi

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
if [ -f "${OUTPUT}" ] && [ -f "${OUTPUT%.csv}.constraints.json" ]; then
    echo "Existing completed sample found; skipping ${LABEL}/${QUERY_ID}"
    exit 0
fi

echo "Model ${LABEL} ($((MODEL_INDEX + 1))/${NUM_MODELS})"
echo "Query ${QUERY_ID} ($((QUERY_INDEX + 1))/${NUM_QUERIES})"
echo "Guide ${GUIDE_CKPT}"
echo "Output ${OUTPUT}"
nvidia-smi

python -u sample_doob_query.py \
    --guide-ckpt "${GUIDE_CKPT}" \
    --base-ckpt "${BASE_CKPT}" \
    --query-file "${QUERY_FILE}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SAMPLE_SEED}" \
    --output "${OUTPUT}" \
    --device cuda

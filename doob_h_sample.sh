#!/bin/bash
#SBATCH --job-name=doob_h_sample
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-1

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export WANDB_MODE=offline

DATANAME="${DATANAME:-shoppers}"
FT_MODEL="${FT_MODEL:-ft_periodic_seed0}"
ORIGINAL_MODEL="${ORIGINAL_MODEL:-original_seed0}"
MODEL_NAMES=(
    "${FT_MODEL}"
    "${ORIGINAL_MODEL}"
)
MODEL_NAME="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"
CKPT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
CKPT_CANDIDATES=("${CKPT_DIR}"/best_ema_model_*.pt)
if [ ! -e "${CKPT_CANDIDATES[0]}" ] || [ "${#CKPT_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one best EMA checkpoint in ${CKPT_DIR}"
    printf '%s\n' "${CKPT_CANDIDATES[@]}"
    exit 1
fi
BASE_CKPT="${CKPT_CANDIDATES[0]}"
GUIDE_DIR_NAME="${GUIDE_DIR_NAME:-doob_h_partial_fixed_box}"
GUIDE_CKPT="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/${GUIDE_DIR_NAME}/best_guide.pt"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
SAMPLE_SUFFIX="${SAMPLE_SUFFIX-_partial}"
OUTPUT="conditional_samples/${DATANAME}/${MODEL_NAME}${SAMPLE_SUFFIX}.csv"
COLUMN_ACTIVE_PROBABILITY="${COLUMN_ACTIVE_PROBABILITY:-0.5}"
ACTIVE_COLUMNS="${ACTIVE_COLUMNS:-}"
QUERY_MASK_ARGS=(--column-active-probability "${COLUMN_ACTIVE_PROBABILITY}")
if [ -n "${ACTIVE_COLUMNS}" ]; then
    QUERY_MASK_ARGS+=(--active-columns "${ACTIVE_COLUMNS}")
fi
CATEGORICAL_START_MODE="${CATEGORICAL_START_MODE:-section4_posterior}"
FIXED_CATEGORICAL="${FIXED_CATEGORICAL:-}"
CATEGORICAL_ARGS=()
if [ -n "${FIXED_CATEGORICAL}" ]; then
    IFS=',' read -r -a FIXED_CATEGORICAL_ITEMS <<< "${FIXED_CATEGORICAL}"
    for ITEM in "${FIXED_CATEGORICAL_ITEMS[@]}"; do
        CATEGORICAL_ARGS+=(--fixed-categorical "${ITEM}")
    done
fi

echo "========================================"
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : ${SLURMD_NODENAME}"
echo "Dataset           : ${DATANAME}"
echo "Base model        : ${MODEL_NAME}"
echo "Base checkpoint   : ${BASE_CKPT}"
echo "Guide checkpoint  : ${GUIDE_CKPT}"
echo "Guidance strength : 1.0 (fixed)"
echo "Categorical Doob  : h(child)/h(current) transition reweighting"
echo "Active num cols   : ${ACTIVE_COLUMNS:-random Bernoulli(${COLUMN_ACTIVE_PROBABILITY}) mask}"
echo "Categorical start : ${CATEGORICAL_START_MODE}"
echo "Fixed categories  : ${FIXED_CATEGORICAL:-none (ordinary t=1 start)}"
echo "Output            : ${OUTPUT}"
echo "========================================"
nvidia-smi

if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: base checkpoint not found: ${BASE_CKPT}"
    exit 1
fi
if [ ! -f "${GUIDE_CKPT}" ]; then
    echo "ERROR: guide checkpoint not found: ${GUIDE_CKPT}"
    exit 1
fi

python sample_doob_h.py \
    --guide-ckpt "${GUIDE_CKPT}" \
    --base-ckpt "${BASE_CKPT}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-correction 5.0 \
    --max-log-h-ratio 10.0 \
    --h-candidate-batch-size 65536 \
    "${QUERY_MASK_ARGS[@]}" \
    --categorical-start-mode "${CATEGORICAL_START_MODE}" \
    "${CATEGORICAL_ARGS[@]}" \
    --output "${OUTPUT}" \
    --device cuda

echo "Finished Doob h-guided sampling"

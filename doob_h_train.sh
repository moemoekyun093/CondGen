#!/bin/bash
#SBATCH --job-name=doob_h_train
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
GUIDE_DIR_NAME="${GUIDE_DIR_NAME:-doob_h_partial_masks_candidate_logh}"
OUTPUT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/${GUIDE_DIR_NAME}"
QUERY_FILE="${QUERY_FILE:-constraints/${DATANAME}/fixed_numerical_intervals.json}"
EPOCHS="${EPOCHS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
COLUMN_ACTIVE_PROBABILITY="${COLUMN_ACTIVE_PROBABILITY:-0.5}"
ALL_ACTIVE_PROBABILITY="${ALL_ACTIVE_PROBABILITY:-0.1}"
ALL_INACTIVE_PROBABILITY="${ALL_INACTIVE_PROBABILITY:-0.1}"
DIAGNOSTIC_BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-1024}"
DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-100}"
H_CANDIDATE_BATCH_SIZE="${H_CANDIDATE_BATCH_SIZE:-16384}"

echo "========================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node            : ${SLURMD_NODENAME}"
echo "Dataset         : ${DATANAME}"
echo "Base model      : ${MODEL_NAME}"
echo "Base checkpoint : ${BASE_CKPT}"
echo "Output          : ${OUTPUT_DIR}"
echo "Fixed query     : ${QUERY_FILE}"
echo "Objective       : separate numerical h-score and categorical log-h guides"
echo "Numerical       : direct sigma(t)^2 * grad_x log h correction"
echo "Categorical     : candidate log h(child) from a shared whole-state scalar network"
echo "Categorical loss: conditional Generator Matching via original TabDiff absorbed loss"
echo "Step mask       : one shared mask; Bernoulli p=${COLUMN_ACTIVE_PROBABILITY}"
echo "Mask anchors    : all-on=${ALL_ACTIVE_PROBABILITY}, all-off=${ALL_INACTIVE_PROBABILITY}"
echo "Endpoint batch  : ${BATCH_SIZE} uniform satisfying rows, with replacement"
echo "Training        : ${EPOCHS} optimizer steps"
echo "EMA diagnostic  : every ${DIAGNOSTIC_EVERY} optimizer steps"
echo "========================================"
nvidia-smi

if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: base checkpoint not found: ${BASE_CKPT}"
    exit 1
fi
if [ ! -f "${QUERY_FILE}" ]; then
    echo "ERROR: fixed query not found: ${QUERY_FILE}"
    echo "Submit doob_h_intervals.sh first"
    exit 1
fi

python -u train_doob_h.py \
    --dataname "${DATANAME}" \
    --base-ckpt "${BASE_CKPT}" \
    --query-file "${QUERY_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --column-active-probability "${COLUMN_ACTIVE_PROBABILITY}" \
    --all-active-probability "${ALL_ACTIVE_PROBABILITY}" \
    --all-inactive-probability "${ALL_INACTIVE_PROBABILITY}" \
    --diagnostic-batch-size "${DIAGNOSTIC_BATCH_SIZE}" \
    --gradient-loss-weight 1.0 \
    --categorical-loss-weight 1.0 \
    --h-candidate-batch-size "${H_CANDIDATE_BATCH_SIZE}" \
    --ema-decay 0.997 \
    --diagnostic-every "${DIAGNOSTIC_EVERY}" \
    --reduce-lr-patience 20 \
    --lr-factor 0.9 \
    --checkpoint-warmup 4000 \
    --checkpoint-every 2000 \
    --d-token 32 \
    --num-layers 2 \
    --n-head 4 \
    --factor 2 \
    --n-frequencies 16 \
    --device cuda

echo "Finished Doob h-guide training"

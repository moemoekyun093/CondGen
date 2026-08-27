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
# Usage: sbatch doob_h_train.sh [GUIDE_DIR_OR_NAME]
# A positional argument takes precedence over the environment variable/default.
GUIDE_DIR_ARG="${1:-${GUIDE_DIR_NAME:-doob_h_partial_masks_concat_d48_l2_h4_f2_6000_candidate_logh}}"
if [[ "${GUIDE_DIR_ARG}" == */* ]]; then
    OUTPUT_DIR="${GUIDE_DIR_ARG}"
else
    OUTPUT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/${GUIDE_DIR_ARG}"
fi
QUERY_FILE="${QUERY_FILE:-constraints/${DATANAME}/fixed_numerical_intervals.json}"
EPOCHS="${EPOCHS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
COLUMN_ACTIVE_PROBABILITY="${COLUMN_ACTIVE_PROBABILITY:-0.5}"
ALL_ACTIVE_PROBABILITY="${ALL_ACTIVE_PROBABILITY:-0.1}"
ALL_INACTIVE_PROBABILITY="${ALL_INACTIVE_PROBABILITY:-0.1}"
DIAGNOSTIC_BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-1024}"
DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-100}"
CHECKPOINT_WARMUP="${CHECKPOINT_WARMUP:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
H_CANDIDATE_BATCH_SIZE="${H_CANDIDATE_BATCH_SIZE:-16384}"
D_TOKEN="${D_TOKEN:-48}"
NUM_LAYERS="${NUM_LAYERS:-2}"
N_HEAD="${N_HEAD:-4}"
FACTOR="${FACTOR:-2}"
N_FREQUENCIES="${N_FREQUENCIES:-16}"
FREQ_SIGMA="${FREQ_SIGMA:-0.05}"
QUERY_MASK_EMBEDDING_DIM="${QUERY_MASK_EMBEDDING_DIM:-8}"

echo "========================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node            : ${SLURMD_NODENAME}"
echo "Dataset         : ${DATANAME}"
echo "Base model      : ${MODEL_NAME}"
echo "Base checkpoint : ${BASE_CKPT}"
echo "Output          : ${OUTPUT_DIR}"
echo "Fixed query     : ${QUERY_FILE}"
echo "Objective       : separate numerical h-score and categorical log-h guides"
echo "Query mask      : per-column binary embedding concatenated and fused into each numerical token"
echo "Guide model     : d=${D_TOKEN}, L=${NUM_LAYERS}, heads=${N_HEAD}, factor=${FACTOR}, frequencies=${N_FREQUENCIES}, mask_dim=${QUERY_MASK_EMBEDDING_DIM}"
echo "Numerical       : direct sigma(t)^2 * grad_x log h correction"
echo "Categorical     : candidate log h(child) from a shared whole-state scalar network"
echo "Categorical loss: conditional Generator Matching via original TabDiff absorbed loss"
echo "Step mask       : one shared mask; Bernoulli p=${COLUMN_ACTIVE_PROBABILITY}"
echo "Mask anchors    : all-on=${ALL_ACTIVE_PROBABILITY}, all-off=${ALL_INACTIVE_PROBABILITY}"
echo "Endpoint batch  : ${BATCH_SIZE} uniform satisfying rows, with replacement"
echo "Training        : ${EPOCHS} optimizer steps"
echo "EMA diagnostic  : every ${DIAGNOSTIC_EVERY} optimizer steps"
echo "Checkpointing   : best after step ${CHECKPOINT_WARMUP}; snapshots every ${CHECKPOINT_EVERY} steps"
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
    --checkpoint-warmup "${CHECKPOINT_WARMUP}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --d-token "${D_TOKEN}" \
    --num-layers "${NUM_LAYERS}" \
    --n-head "${N_HEAD}" \
    --factor "${FACTOR}" \
    --n-frequencies "${N_FREQUENCIES}" \
    --freq-sigma "${FREQ_SIGMA}" \
    --query-mask-embedding-dim "${QUERY_MASK_EMBEDDING_DIM}" \
    --device cuda

echo "Finished Doob h-guide training"

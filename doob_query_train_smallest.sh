#!/bin/bash
#SBATCH --job-name=doob_query_minband
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
TARGET_BAND="${TARGET_BAND:-0.005}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
STEPS="${STEPS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-1e-3}"
D_TOKEN="${D_TOKEN:-48}"
NUM_LAYERS="${NUM_LAYERS:-2}"
N_HEAD="${N_HEAD:-4}"
FACTOR="${FACTOR:-2}"
BOUND_EMBEDDING_DIM="${BOUND_EMBEDDING_DIM:-8}"
ACTIVE_EMBEDDING_DIM="${ACTIVE_EMBEDDING_DIM:-8}"

CKPT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
CKPT_CANDIDATES=("${CKPT_DIR}"/best_ema_model_*.pt)
if [ ! -e "${CKPT_CANDIDATES[0]}" ] || [ "${#CKPT_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one best EMA checkpoint in ${CKPT_DIR}"
    printf '%s\n' "${CKPT_CANDIDATES[@]}"
    exit 1
fi
BASE_CKPT="${CKPT_CANDIDATES[0]}"

GUIDE_DIR_ARG="${1:-${GUIDE_DIR_NAME:-doob_query_minband005_d48_l2_6000}}"
if [[ "${GUIDE_DIR_ARG}" == */* ]]; then
    OUTPUT_DIR="${GUIDE_DIR_ARG}"
else
    OUTPUT_DIR="${CKPT_DIR}/${GUIDE_DIR_ARG}"
fi

echo "========================================"
echo "Dataset          : ${DATANAME}"
echo "Base model       : ${MODEL_NAME}"
echo "Base checkpoint  : ${BASE_CKPT}"
echo "Query suite      : ${QUERY_DIR}"
echo "Target band      : ${TARGET_BAND}"
echo "Query selection  : every accepted query in the target band"
echo "Query sampling   : uniform across selected queries"
echo "Endpoint sampling: uniform satisfying rows, with replacement"
echo "Output           : ${OUTPUT_DIR}"
echo "Training steps   : ${STEPS}"
echo "Batch size       : ${BATCH_SIZE}"
echo "Learning rate    : ${LR}"
echo "Guide            : d=${D_TOKEN}, L=${NUM_LAYERS}, heads=${N_HEAD}, factor=${FACTOR}"
echo "========================================"
nvidia-smi

if [ ! -d "${QUERY_DIR}" ]; then
    echo "ERROR: query directory not found: ${QUERY_DIR}"
    exit 1
fi

python -u train_doob_query_suite.py \
    --dataname "${DATANAME}" \
    --base-ckpt "${BASE_CKPT}" \
    --query-dir "${QUERY_DIR}" \
    --target-band "${TARGET_BAND}" \
    --output-dir "${OUTPUT_DIR}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --d-token "${D_TOKEN}" \
    --num-layers "${NUM_LAYERS}" \
    --n-head "${N_HEAD}" \
    --factor "${FACTOR}" \
    --bound-embedding-dim "${BOUND_EMBEDDING_DIM}" \
    --active-embedding-dim "${ACTIVE_EMBEDDING_DIM}" \
    --checkpoint-warmup 200 \
    --device cuda

echo "Finished minimum-selectivity structured-query training"

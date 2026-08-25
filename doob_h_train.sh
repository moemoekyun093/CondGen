#!/bin/bash
#SBATCH --job-name=doob_h_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export WANDB_MODE=offline

DATANAME="${DATANAME:-news}"
BASE_CKPT="${BASE_CKPT:-tabdiff/ckpt/news/learnable_schedule/best_ema_model_2.215_4619.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-tabdiff/ckpt/${DATANAME}/doob_h_fixed_box}"
QUERY_FILE="${QUERY_FILE:-constraints/${DATANAME}/fixed_numerical_intervals.json}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-512}"

echo "========================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node            : ${SLURMD_NODENAME}"
echo "Dataset         : ${DATANAME}"
echo "Base checkpoint : ${BASE_CKPT}"
echo "Output          : ${OUTPUT_DIR}"
echo "Fixed query     : ${QUERY_FILE}"
echo "Objective       : Section-5 h-score matching in sigma^2-scaled coordinates"
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

python train_doob_h.py \
    --dataname "${DATANAME}" \
    --base-ckpt "${BASE_CKPT}" \
    --query-file "${QUERY_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --d-token 32 \
    --num-layers 2 \
    --n-head 4 \
    --factor 2 \
    --n-frequencies 16 \
    --device cuda

echo "Finished Doob h-guide training"

#!/bin/bash
#SBATCH --job-name=doob_h_sample
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-1

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export WANDB_MODE=offline

DATANAME="${DATANAME:-news}"
MODEL_NAMES=(
    "ft_periodic_L6_d128_seed0"
    "original_L2_d4_seed0"
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
GUIDE_CKPT="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_h_fixed_box/best_guide.pt"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
OUTPUT="conditional_samples/${DATANAME}/${MODEL_NAME}.csv"

echo "========================================"
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : ${SLURMD_NODENAME}"
echo "Dataset           : ${DATANAME}"
echo "Base model        : ${MODEL_NAME}"
echo "Base checkpoint   : ${BASE_CKPT}"
echo "Guide checkpoint  : ${GUIDE_CKPT}"
echo "Guidance strength : 1.0 (fixed)"
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
    --output "${OUTPUT}" \
    --device cuda

echo "Finished Doob h-guided sampling"

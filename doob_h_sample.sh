#!/bin/bash
#SBATCH --job-name=doob_h_sample
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export WANDB_MODE=offline

DATANAME="${DATANAME:-news}"
BASE_CKPT="${BASE_CKPT:-tabdiff/ckpt/news/learnable_schedule/best_ema_model_2.215_4619.pt}"
GUIDE_CKPT="${GUIDE_CKPT:-tabdiff/ckpt/${DATANAME}/doob_h_fixed_box/best_guide.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
OUTPUT="${OUTPUT:-conditional_samples/${DATANAME}/doob_h_fixed_box.csv}"

echo "========================================"
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : ${SLURMD_NODENAME}"
echo "Dataset           : ${DATANAME}"
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

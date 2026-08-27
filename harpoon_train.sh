#!/bin/bash
#SBATCH --job-name=harpoon_train
#SBATCH --output=harpoon_logs/%x_%j.out
#SBATCH --error=harpoon_logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
if [ "${DATANAME}" != "shoppers" ]; then
    echo "ERROR: the paper-aligned HARPOON baseline is currently Shoppers-only"
    exit 1
fi
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
HID_DIM="${HID_DIM:-1024}"
HARPOON_RUNTIME="${HARPOON_RUNTIME:-/scratch/work/agrawaa4/harpoon_runtime}"

echo "Training code     : unchanged baselines/harpoon/train_repaint.py"
echo "Dataset           : ${DATANAME}"
echo "Epochs            : ${EPOCHS}"
echo "Batch size        : ${BATCH_SIZE}"
echo "Hidden dimension  : ${HID_DIM}"
echo "Runtime artifacts : ${HARPOON_RUNTIME}"

mkdir -p "${HARPOON_RUNTIME}"
python -u prepare_harpoon_data.py \
    --dataname "${DATANAME}" \
    --harpoon-root "${HARPOON_RUNTIME}"
(
    cd "${HARPOON_RUNTIME}"
    python -u /scratch/work/agrawaa4/TabDiff/run_harpoon_train_compat.py \
        /scratch/work/agrawaa4/TabDiff/baselines/harpoon/train_repaint.py \
        --dataname "${DATANAME}" \
        --gpu 0 \
        --hid_dim "${HID_DIM}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --timesteps 200 \
        --beta_0 0.0001 \
        --beta_T 0.02
)

echo "Saved HARPOON checkpoint to ${HARPOON_RUNTIME}/saved_models/${DATANAME}/diffputer_selfmade.pt"

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

python -u prepare_harpoon_data.py --dataname "${DATANAME}"
python -u train_harpoon_baseline.py \
    --dataname "${DATANAME}" \
    --device cuda \
    --hid-dim "${HID_DIM}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --timesteps 200

echo "Saved HARPOON checkpoint to baselines/harpoon/saved_models/${DATANAME}/diffputer_selfmade.pt"

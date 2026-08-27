#!/bin/bash
#SBATCH --job-name=harpoon_official_and
#SBATCH --output=harpoon_logs/%x_%j.out
#SBATCH --error=harpoon_logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
NUM_TRIALS="${NUM_TRIALS:-5}"
HARPOON_RUNTIME="${HARPOON_RUNTIME:-/scratch/work/agrawaa4/harpoon_runtime}"
if [ "${DATANAME}" != "shoppers" ]; then
    echo "ERROR: this exact official comparison is Shoppers-only"
    exit 1
fi
if [ ! -f "${HARPOON_RUNTIME}/saved_models/${DATANAME}/diffputer_selfmade.pt" ]; then
    echo "ERROR: train the official HARPOON checkpoint first"
    exit 1
fi

echo "========================================"
echo "Official source    : baselines/harpoon/sampling_harpoon_ohe_tubular_generalconstraints.py"
echo "Dataset            : ${DATANAME}"
echo "Constraint         : both"
echo "AND query          : Administrative >= 4 AND VisitorType == New_Visitor"
echo "Trials             : ${NUM_TRIALS}"
echo "Runtime artifacts  : ${HARPOON_RUNTIME}"
echo "Upstream source is invoked unchanged"
echo "========================================"
nvidia-smi

(
    mkdir -p "${HARPOON_RUNTIME}/experiments"
    cd "${HARPOON_RUNTIME}"
    python -u /scratch/work/agrawaa4/TabDiff/baselines/harpoon/sampling_harpoon_ohe_tubular_generalconstraints.py \
        --dataname "${DATANAME}" \
        --gpu 0 \
        --hid_dim 1024 \
        --batch_size 1024 \
        --timesteps 200 \
        --beta_0 0.0001 \
        --beta_T 0.02 \
        --constraint both \
        --num_trials "${NUM_TRIALS}"
)

echo "Finished unchanged official HARPOON AND experiment"

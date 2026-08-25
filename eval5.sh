#!/bin/bash
#SBATCH --job-name=eval_method
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:80g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-29          # 6 datasets * 5 seeds - 1

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Array ID : $SLURM_ARRAY_TASK_ID"
echo "Node     : $SLURMD_NODENAME"
echo "========================================"
nvidia-smi
export WANDB_MODE=offline
cd /scratch/work/agrawaa4/TabDiff

# --- datasets (news excluded, already done) ---
DATASETS=(adult default shoppers magic beijing diabetes)
N_SEEDS=5

# --- architecture (ft_periodic) ---
ARCH="ft_periodic"

# flatten (dataset, seed) -> array index
ds_idx=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
DATANAME=${DATASETS[$ds_idx]}

EXP_NAME="${ARCH}_seed${SEED}"
CKPT_DIR="tabdiff/ckpt/${DATANAME}/${EXP_NAME}"

CKPT_PATH=$(ls "${CKPT_DIR}"/best_ema_model_*.pt 2>/dev/null | head -n 1)
if [ -z "$CKPT_PATH" ]; then
    echo "No best_ema_model_* found, falling back to latest model_*.pt"
    CKPT_PATH=$(ls "${CKPT_DIR}"/model_*.pt 2>/dev/null | sort -V | tail -n 1)
fi
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: no usable checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "dataset=${DATANAME}  arch=${ARCH}  seed=${SEED}"
echo "Checkpoint : ${CKPT_PATH}"
echo "========================================"

python main.py \
    --dataname ${DATANAME} \
    --mode test \
    --ckpt_path "${CKPT_PATH}" \
    --no_wandb \
    --gpu 0

echo "Finished eval (dataset=${DATANAME}, exp_name=${EXP_NAME})"
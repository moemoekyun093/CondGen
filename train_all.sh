#!/bin/bash
#SBATCH --job-name=train_method
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:80g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2          # one index per config in CONFIGS below; keep this range in sync

echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Array ID    : $SLURM_ARRAY_TASK_ID"
echo "Node        : $SLURMD_NODENAME"
echo "========================================"
nvidia-smi
export WANDB_MODE=offline

# --- FT-Transformer sweep grid. Format: num_layers d_token n_head factor ---
CONFIGS=(
    # "4 64 8 4"
    # "6 64 8 4"
    "6 128 8 4"
)

read NUM_LAYERS D_TOKEN N_HEAD FACTOR <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# --- Unique exp_name per task so checkpoints/wandb dirs never collide ---
EXP_NAME="ft_L${NUM_LAYERS}_d${D_TOKEN}_h${N_HEAD}_f${FACTOR}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Config      : num_layers=${NUM_LAYERS} d_token=${D_TOKEN} n_head=${N_HEAD} factor=${FACTOR}"
echo "Exp name    : ${EXP_NAME}"
echo "========================================"

python main.py \
    --dataname news \
    --mode train \
    --exp_name "${EXP_NAME}" \
    --num_layers ${NUM_LAYERS} \
    --d_token ${D_TOKEN} \
    --n_head ${N_HEAD} \
    --factor ${FACTOR} \
    --gpu 0

echo "Finished (exp_name=${EXP_NAME})"
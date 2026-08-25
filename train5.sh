#!/bin/bash
#SBATCH --job-name=train_method
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:80g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4          # 5 seeds (0..4), one per array task, all run in parallel

echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Array ID    : $SLURM_ARRAY_TASK_ID"
echo "Node        : $SLURMD_NODENAME"
echo "========================================"
nvidia-smi
export WANDB_MODE=offline
cd /scratch/work/agrawaa4/TabDiff

# --- Fixed architecture for this 5-seed batch ---
NUM_LAYERS=6
D_TOKEN=128
N_HEAD=8
FACTOR=4
ARCH="ft_periodic"           # <- "original" for the baseline batch, "ft_periodic" for the FT batch

SEED=$SLURM_ARRAY_TASK_ID

# --- exp_name includes ARCH + seed so the 5 runs never collide ---
EXP_NAME="${ARCH}_L${NUM_LAYERS}_d${D_TOKEN}_seed${SEED}"

echo "Arch=${ARCH}  seed=${SEED}  config: L${NUM_LAYERS} d${D_TOKEN} h${N_HEAD} f${FACTOR}"
echo "Exp name    : ${EXP_NAME}"
echo "========================================"

python main.py \
    --dataname news \
    --mode train \
    --exp_name "${EXP_NAME}" \
    --seed ${SEED} \
    --denoiser_type ${ARCH} \
    --num_layers ${NUM_LAYERS} \
    --d_token ${D_TOKEN} \
    --n_head ${N_HEAD} \
    --factor ${FACTOR} \
    --gpu 0

echo "Finished (exp_name=${EXP_NAME})"
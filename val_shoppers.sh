#!/bin/bash
#SBATCH --job-name=shoppers_small
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2          # 3 configs, 1 seed each

echo "Job ID : $SLURM_JOB_ID   Array ID : $SLURM_ARRAY_TASK_ID"
nvidia-smi
export WANDB_MODE=offline
cd /scratch/work/agrawaa4/TabDiff

DATANAME=shoppers
CONFIGS=(
    "32 2 4"      # d_token num_layers n_head
    "64 2 4"
    "64 4 4"
)
SEED=0

read D_TOKEN NUM_LAYERS N_HEAD <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

EXP_NAME="ft_periodic_small_d${D_TOKEN}_L${NUM_LAYERS}_seed${SEED}"

echo "dataset=${DATANAME} d_token=${D_TOKEN} num_layers=${NUM_LAYERS} n_head=${N_HEAD}"
echo "exp_name=${EXP_NAME}"

python main.py \
    --dataname ${DATANAME} \
    --mode train \
    --exp_name "${EXP_NAME}" \
    --seed ${SEED} \
    --denoiser_type ft_periodic \
    --num_layers ${NUM_LAYERS} \
    --d_token ${D_TOKEN} \
    --n_head ${N_HEAD} \
    --factor 4 \
    --gpu 0

echo "Finished ${DATANAME} / ${EXP_NAME}"
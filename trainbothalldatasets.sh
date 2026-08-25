#!/bin/bash
#SBATCH --job-name=tabdiff_multi
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:80g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=5-9          # diabetes slice: tasks 5..9 -> seeds 0..4

echo "Job ID : $SLURM_JOB_ID   Array ID : $SLURM_ARRAY_TASK_ID"
nvidia-smi
export WANDB_MODE=offline
cd /scratch/work/agrawaa4/TabDiff

N_SEEDS=5
DATANAME=diabetes
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

# --- pick ONE architecture per submission ---
ARCH="original"        # <- "ft_periodic" for the FT batch

# --- architecture-specific hyperparameters ---
if [ "$ARCH" = "ft_periodic" ]; then
    NUM_LAYERS=6; D_TOKEN=128; N_HEAD=8; FACTOR=4
else
    NUM_LAYERS=2; D_TOKEN=4;   N_HEAD=1; FACTOR=32
fi

EXP_NAME="${ARCH}_seed${SEED}"

echo "dataset=${DATANAME}  arch=${ARCH}  seed=${SEED}  exp=${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python main.py \
    --dataname ${DATANAME} \
    --mode train \
    --exp_name "${EXP_NAME}" \
    --seed ${SEED} \
    --denoiser_type ${ARCH} \
    --num_layers ${NUM_LAYERS} \
    --d_token ${D_TOKEN} \
    --n_head ${N_HEAD} \
    --factor ${FACTOR} \
    --gpu 0

echo "Finished ${DATANAME} / ${EXP_NAME}"
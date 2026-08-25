#!/bin/bash
#SBATCH --job-name=tabdiff_rerun
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err
#SBATCH --gres=min-vram:80g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4   # at most 2 concurrent reruns

export WANDB_MODE=offline
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "===================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
nvidia-smi
echo "===================================="

ARCH="ft_periodic"        # <- "original" for the baseline batch

case $SLURM_ARRAY_TASK_ID in
    0) DATANAME=diabetes; SEED=0 ;;
    1) DATANAME=diabetes; SEED=1 ;;
    2) DATANAME=diabetes; SEED=2 ;;
    3) DATANAME=diabetes; SEED=3 ;;
    4) DATANAME=diabetes; SEED=4 ;;
esac

 NUM_LAYERS=6
 D_TOKEN=128
 N_HEAD=8
 FACTOR=4

EXP_NAME="${ARCH}_seed${SEED}"

echo "dataset=${DATANAME} seed=${SEED}"

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

echo "Finished ${DATANAME} seed ${SEED}"
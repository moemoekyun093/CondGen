#!/bin/bash
#SBATCH --job-name=train_method
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

#SBATCH --gres=min-vram:80g,min-cuda-cc:80
#SBATCH --gpus=1


#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00


echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Array ID    : $SLURM_ARRAY_TASK_ID"
echo "Dataset     : $DATASET"
echo "Node        : $SLURMD_NODENAME"
echo "========================================"

nvidia-smi
export WANDB_MODE=offline
python main.py --dataname news --mode train


echo "Finished dataset"
#!/bin/bash
#SBATCH --job-name=attn_sparsity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00

echo "Job ID : $SLURM_JOB_ID"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
cd /scratch/work/agrawaa4/TabDiff

# all datasets except news use the ft_periodic prefix
python attention_sparsity.py \
    --datasets adult default shoppers magic beijing diabetes \
    --seed 0 \
    --exp_prefix ft_periodic \
    --gpu 0

# news uses the longer prefix from its earlier 5-seed batch
python attention_sparsity.py \
    --datasets news \
    --seed 0 \
    --exp_prefix ft_periodic_L6_d128 \
    --gpu 0

echo "Finished attention sparsity diagnostic"
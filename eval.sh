#!/bin/bash
#SBATCH --job-name=eval_method
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:80g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0-2          # one index per exp_name below; keep in sync

echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Array ID    : $SLURM_ARRAY_TASK_ID"
echo "Node        : $SLURMD_NODENAME"
echo "========================================"
nvidia-smi
export WANDB_MODE=offline

# --- Exact trained experiment folder names (from `ls tabdiff/ckpt/news/`) ---
EXP_NAMES=(
    "learnable_schedule"
    "ft_L6_d128_h8_f4_19073334_0"
    # "ft_L6_d128_h8_f4_19071259_2"
    # "ft_L6_d64_h8_f4_19071259_1"
)

EXP_NAME="${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}"
CKPT_DIR="tabdiff/ckpt/news/${EXP_NAME}"

# --- Prefer best_ema_model (wrapped format, includes noise schedules).
# Fall back to model_*.pt (also wrapped) if best_ema hasn't been saved yet
# (e.g. run stopped before epoch 4000). NEVER use bare ema_model_*.pt here --
# it lacks num_schedule/cat_schedule and will silently give wrong samples.
CKPT_PATH=$(ls "${CKPT_DIR}"/best_ema_model_*.pt 2>/dev/null | head -n 1)
if [ -z "$CKPT_PATH" ]; then
    echo "No best_ema_model_* found, falling back to latest model_*.pt"
    CKPT_PATH=$(ls "${CKPT_DIR}"/model_*.pt 2>/dev/null | sort -V | tail -n 1)
fi
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: no usable checkpoint (best_ema_model_* or model_*.pt) found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating exp_name : ${EXP_NAME}"
echo "Checkpoint          : ${CKPT_PATH}"
echo "========================================"

python main.py \
    --dataname news \
    --mode test \
    --ckpt_path "${CKPT_PATH}" \
    --no_wandb \
    --gpu 0

echo "Finished eval (exp_name=${EXP_NAME})"
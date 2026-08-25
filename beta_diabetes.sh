#!/bin/bash
#SBATCH --job-name=sample_alpha_news
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-4          # 5 seeds

echo "Job ID : $SLURM_JOB_ID   Array ID : $SLURM_ARRAY_TASK_ID"
nvidia-smi
cd /scratch/work/agrawaa4/TabDiff

DATANAME=beijing
SEED=$SLURM_ARRAY_TASK_ID
EXP_NAME="ft_periodic_seed${SEED}"   # your news-specific FT-Periodic naming

CKPT_DIR="tabdiff/ckpt/${DATANAME}/${EXP_NAME}"
CKPT=$(ls ${CKPT_DIR}/best_ema_model_*.pt 2>/dev/null | head -n 1)
if [ -z "$CKPT" ]; then
    echo "No best_ema_model found in ${CKPT_DIR}"
    exit 1
fi
EPOCH=$(basename "$CKPT" .pt | awk -F'_' '{print $NF}')

SAMPLES_CSV="tabdiff/result/${DATANAME}/${EXP_NAME}/${EPOCH}/samples.csv"

if [ ! -f "$SAMPLES_CSV" ]; then
    echo "Sampling ${DATANAME}/${EXP_NAME} (epoch ${EPOCH}) ..."
    python main.py --dataname ${DATANAME} --mode test --ckpt_path "${CKPT}" --no_wandb --gpu 0
fi

if [ ! -f "$SAMPLES_CSV" ]; then
    echo "ERROR: sampling did not produce ${SAMPLES_CSV}"
    exit 1
fi

STAGE_DIR="alpha_staging/${EXP_NAME}/${DATANAME}/run_0"
mkdir -p "${STAGE_DIR}"
cp "${SAMPLES_CSV}" "${STAGE_DIR}/${EXP_NAME}_${DATANAME}_run0.csv"

echo "Computing alpha-precision / beta-recall ..."
python alpha_precision_standalone.py \
    --generated_data_folder alpha_staging \
    --target_folder alpha_staging_results \
    --pattern "*${EXP_NAME}*${DATANAME}*.csv"

echo ""
echo "Result:"
cat "alpha_staging_results/${EXP_NAME}/${DATANAME}/run_0/alpha_precision/"*.json
cat "alpha_staging_results/${EXP_NAME}/${DATANAME}/run_0/beta_recall/"*.json
#!/bin/bash
#SBATCH --job-name=center_logw_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

RELgDIFF_PYTHON="/scratch/work/agrawaa4/conda_envs/relgdiff/bin/python"
"${RELgDIFF_PYTHON}" -u evaluate_center_logwidth_multiseed.py \
    --sample-dir "${MULTISEED_SAMPLE_DIR}" \
    --seeds "${SEEDS}" \
    --unconditional-samples "${UNCONDITIONAL_SAMPLES}" \
    --query-file "${QUERY_FILE}" \
    --dataname "${DATANAME}" \
    --data-dir "data/${DATANAME}" \
    --info-file "data/${DATANAME}/info.json" \
    --base-config "tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/config.pkl" \
    --output-dir "${MULTISEED_EVAL_DIR}" \
    --bins "${HISTOGRAM_BINS:-50}"

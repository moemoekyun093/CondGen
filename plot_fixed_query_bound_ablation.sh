#!/bin/bash
#SBATCH --job-name=doob_bound_plot
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

RELgDIFF_PYTHON="/scratch/work/agrawaa4/conda_envs/relgdiff/bin/python"
"${RELgDIFF_PYTHON}" -u plot_fixed_query_checkpoint_histograms.py \
    --series "monotone=${BOUND_SAMPLE_ROOT}/monotone" \
    --series "ordinary_mlp=${BOUND_SAMPLE_ROOT}/mlp" \
    --query-file "${QUERY_DIR}/${QUERY_ID}.json" \
    --dataname "${DATANAME}" \
    --data-dir "data/${DATANAME}" \
    --info-file "data/${DATANAME}/info.json" \
    --base-config "tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/config.pkl" \
    --output-dir "${BOUND_EVAL_DIR}" \
    --bins "${HISTOGRAM_BINS:-50}"

#!/bin/bash
#SBATCH --job-name=doob_bound_token_plot
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
    --series "ordinary_mlp=${TOKEN_SAMPLE_ROOT}/ordinary_mlp" \
    --series "lower_upper_tokens=${TOKEN_SAMPLE_ROOT}/endpoints" \
    --series "center_logwidth_tokens=${TOKEN_SAMPLE_ROOT}/center_logwidth" \
    --query-file "${QUERY_DIR}/${QUERY_ID}.json" \
    --dataname "${DATANAME}" \
    --data-dir "data/${DATANAME}" \
    --info-file "data/${DATANAME}/info.json" \
    --base-config "tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/config.pkl" \
    --output-dir "${TOKEN_EVAL_DIR}" \
    --bins "${HISTOGRAM_BINS:-50}"

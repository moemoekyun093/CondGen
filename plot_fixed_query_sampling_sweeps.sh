#!/bin/bash
#SBATCH --job-name=doob_sampling_sweep_plot
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

RELgDIFF_PYTHON="/scratch/work/agrawaa4/conda_envs/relgdiff/bin/python"
"${RELgDIFF_PYTHON}" -u plot_fixed_query_sampling_sweeps.py \
    --sample-dir "${SWEEP_SAMPLE_ROOT}/ordinary_mlp_center_logwidth" \
    --query-file "${QUERY_DIR}/${QUERY_ID}.json" \
    --dataname "${DATANAME}" \
    --data-dir "data/${DATANAME}" \
    --info-file "data/${DATANAME}/info.json" \
    --base-config "tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/config.pkl" \
    --output-dir "${SWEEP_EVAL_DIR}" \
    --reverse-steps "${REVERSE_STEPS:-50,75,100,150,200}" \
    --guidance-strengths "${GUIDANCE_STRENGTHS:-1,2,5}" \
    --bins "${HISTOGRAM_BINS:-50}"

#!/bin/bash
#SBATCH --job-name=query_generalization_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

RELgDIFF_PYTHON="/scratch/work/agrawaa4/conda_envs/relgdiff/bin/python"
"${RELgDIFF_PYTHON}" -u evaluate_query_generalization.py \
    --query-dir "${QUERY_DIR}" \
    --source-manifest "${SOURCE_QUERY_SPLIT_MANIFEST}" \
    --diagnostic-manifest "${DIAGNOSTIC_QUERY_SPLIT_MANIFEST}" \
    --train-samples "${TRAIN_SAMPLE_ROOT}/${METHOD_LABEL}" \
    --test-samples "${TEST_SAMPLE_DIR}" \
    --query-coordinates "${QUERY_COORDINATES}" \
    --num-plot-bins "${NUM_PLOT_BINS:-10}" \
    --output-dir "${OUTPUT_DIR}"

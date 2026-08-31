#!/bin/bash
#SBATCH --job-name=doob_fixed_query_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

RELgDIFF_PYTHON="/scratch/work/agrawaa4/conda_envs/relgdiff/bin/python"
BASE_CONFIG="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/config.pkl"
"${RELgDIFF_PYTHON}" -u evaluate_doob_query_suite.py \
    --query-dir "${QUERY_DIR}" \
    --query-id "${QUERY_ID}" \
    --method "${METHOD_LABEL}=$(dirname "${SAMPLE_OUTPUT}")" \
    --real-data "synthetic/${DATANAME}/real.csv" \
    --info-file "data/${DATANAME}/info.json" \
    --output-dir "${EVAL_OUTPUT_DIR}"

VIOLATION_ARGS=(
    --samples "${SAMPLE_OUTPUT}"
    --method-label "${METHOD_LABEL}"
    --query-file "${QUERY_DIR}/${QUERY_ID}.json"
    --output-dir "${EVAL_OUTPUT_DIR}"
    --dataname "${DATANAME}"
    --data-dir "data/${DATANAME}"
    --info-file "data/${DATANAME}/info.json"
    --base-config "${BASE_CONFIG}"
    --bins "${VIOLATION_HISTOGRAM_BINS:-40}"
    --print-comparison-rows "${PRINT_HARPOON_ROWS:-20}"
)
if [ -n "${HARPOON_SAMPLE:-}" ]; then
    VIOLATION_ARGS+=(
        --comparison-samples "${HARPOON_SAMPLE}"
        --comparison-label "${HARPOON_LABEL:-harpoon_eta02}"
    )
fi
"${RELgDIFF_PYTHON}" -u evaluate_fixed_query_violation_magnitude.py \
    "${VIOLATION_ARGS[@]}"

#!/bin/bash
#SBATCH --job-name=doob_harpoon_suite_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_suite_comparison}"
DOOB_LABEL="${DOOB_LABEL:-doob_curriculum}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_eta02}"
SUITE_EVAL_DIR="${SUITE_EVAL_DIR:-evaluations/${DATANAME}/doob_vs_harpoon_query_suite}"
REAL_DATA="${REAL_DATA:-synthetic/${DATANAME}/real.csv}"
INFO_FILE="${INFO_FILE:-data/${DATANAME}/info.json}"
EVAL_GROUP_BY="${EVAL_GROUP_BY:-target_band}"

mkdir -p evaluations/slurm "${SUITE_EVAL_DIR}"

python -u evaluate_doob_query_suite.py \
    --query-dir "${QUERY_DIR}" \
    --method "${DOOB_LABEL}=${SUITE_SAMPLE_ROOT}/${DOOB_LABEL}" \
    --method "${HARPOON_LABEL}=${SUITE_SAMPLE_ROOT}/${HARPOON_LABEL}" \
    --real-data "${REAL_DATA}" \
    --info-file "${INFO_FILE}" \
    --group-by "${EVAL_GROUP_BY}" \
    --output-dir "${SUITE_EVAL_DIR}"

echo "Modality columns are included in the grouped evaluation CSV:"
echo "  numeric_joint_miss_rate_mean / categorical_joint_miss_rate_mean"
echo "  numeric_mean_column_miss_rate_mean / categorical_mean_column_miss_rate_mean"

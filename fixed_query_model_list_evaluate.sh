#!/bin/bash
#SBATCH --job-name=fixed_query_list_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_fixed_box_permuted}"
METHOD_SPECS="${METHOD_SPECS:?METHOD_SPECS must be comma-separated LABEL=SAMPLE_DIRECTORY entries}"
BASELINE_LABEL="${BASELINE_LABEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/fixed_box_permuted_model_comparison}"

IFS=',' read -r -a METHODS <<< "${METHOD_SPECS}"
if [ "${#METHODS[@]}" -lt 2 ]; then
    echo "ERROR: at least two methods are required"
    exit 1
fi

ARGS=()
for METHOD in "${METHODS[@]}"; do
    if [[ "${METHOD}" != *=* ]]; then
        echo "ERROR: invalid method specification: ${METHOD}"
        exit 1
    fi
    ARGS+=(--method "${METHOD}")
done
if [ -n "${BASELINE_LABEL}" ]; then
    ARGS+=(--baseline-method "${BASELINE_LABEL}")
fi

mkdir -p evaluations/slurm "${OUTPUT_DIR}"
python -u evaluate_doob_query_suite.py \
    --query-dir "${QUERY_DIR}" \
    "${ARGS[@]}" \
    --real-data "synthetic/${DATANAME}/real.csv" \
    --info-file "data/${DATANAME}/info.json" \
    --group-by arity \
    --output-dir "${OUTPUT_DIR}"

echo "Saved list-driven fixed-query comparison to ${OUTPUT_DIR}"

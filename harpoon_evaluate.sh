#!/bin/bash
#SBATCH --job-name=harpoon_evaluate
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-adult}"
CONDITIONAL_SAMPLES="${CONDITIONAL_SAMPLES:-conditional_samples/${DATANAME}/harpoon_partial.csv}"
UNCONDITIONAL_SAMPLES="${UNCONDITIONAL_SAMPLES:-conditional_samples/${DATANAME}/harpoon_unconditional.csv}"
QUERY_FILE="${QUERY_FILE:-constraints/${DATANAME}/fixed_numerical_intervals.json}"
REAL_DATA="synthetic/${DATANAME}/real.csv"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/harpoon_partial}"

mkdir -p "${OUTPUT_DIR}"
for path in "${CONDITIONAL_SAMPLES}" "${UNCONDITIONAL_SAMPLES}" "${QUERY_FILE}" "${REAL_DATA}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}"
        exit 1
    fi
done

python -u diagnose_doob_samples.py \
    --samples "${CONDITIONAL_SAMPLES}" \
    --query-file "${QUERY_FILE}" \
    --output "${OUTPUT_DIR}/raw_diagnostic.json"

python -u evaluate_doob_density.py \
    --dataname "${DATANAME}" \
    --samples "${CONDITIONAL_SAMPLES}" \
    --unconditional-samples "${UNCONDITIONAL_SAMPLES}" \
    --query-file "${QUERY_FILE}" \
    --real-data "${REAL_DATA}" \
    --output-dir "${OUTPUT_DIR}"

echo "Finished HARPOON baseline evaluation"

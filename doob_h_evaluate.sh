#!/bin/bash
#SBATCH --job-name=doob_h_evaluate
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
FT_MODEL="${FT_MODEL:-ft_periodic_seed0}"
ORIGINAL_MODEL="${ORIGINAL_MODEL:-original_seed0}"
MODEL_NAMES=(
    "${FT_MODEL}"
    "${ORIGINAL_MODEL}"
)
MODEL_NAME="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"
SAMPLES="conditional_samples/${DATANAME}/${MODEL_NAME}.csv"
QUERY_FILE="${QUERY_FILE:-constraints/${DATANAME}/fixed_numerical_intervals.json}"
RAW_REPORT="conditional_samples/${DATANAME}/${MODEL_NAME}.raw_diagnostic.json"
OUTPUT_DIR="conditional_samples/${DATANAME}/${MODEL_NAME}_evaluation"
REAL_DATA="synthetic/${DATANAME}/real.csv"

for path in "${SAMPLES}" "${QUERY_FILE}" "${REAL_DATA}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}"
        exit 1
    fi
done

echo "========================================"
echo "Dataset        : ${DATANAME}"
echo "Model          : ${MODEL_NAME}"
echo "Existing CSV   : ${SAMPLES}"
echo "Query          : ${QUERY_FILE}"
echo "Real reference : ${REAL_DATA}"
echo "Output         : ${OUTPUT_DIR}"
echo "No new samples will be generated"
echo "========================================"

python -u diagnose_doob_samples.py \
    --samples "${SAMPLES}" \
    --query-file "${QUERY_FILE}" \
    --output "${RAW_REPORT}"

python -u evaluate_doob_density.py \
    --dataname "${DATANAME}" \
    --samples "${SAMPLES}" \
    --query-file "${QUERY_FILE}" \
    --real-data "${REAL_DATA}" \
    --output-dir "${OUTPUT_DIR}"

echo "Finished existing-sample diagnostics and density evaluation"

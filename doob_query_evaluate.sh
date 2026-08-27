#!/bin/bash
#SBATCH --job-name=doob_query_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_FILE="${QUERY_FILE:-data90/${DATANAME}/queries_full/qf_shoppers_b00p5_4.json}"
QUERY_ID="$(basename "${QUERY_FILE}" .json)"
SAMPLES="${SAMPLES:-conditional_samples/${DATANAME}/${MODEL_NAME}_${QUERY_ID}_curriculum.csv}"
UNCONDITIONAL_SAMPLES="${UNCONDITIONAL_SAMPLES:-tabdiff/result/${DATANAME}/${MODEL_NAME}/8000/samples.csv}"
REAL_DATA="${REAL_DATA:-synthetic/${DATANAME}/real.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/${MODEL_NAME}_${QUERY_ID}_curriculum}"

mkdir -p "${OUTPUT_DIR}"
for path in "${SAMPLES}" "${UNCONDITIONAL_SAMPLES}" "${QUERY_FILE}" "${REAL_DATA}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}"
        exit 1
    fi
done

python -u diagnose_doob_samples.py \
    --samples "${SAMPLES}" \
    --query-file "${QUERY_FILE}" \
    --output "${OUTPUT_DIR}/raw_diagnostic.json"

python -u evaluate_doob_density.py \
    --dataname "${DATANAME}" \
    --samples "${SAMPLES}" \
    --unconditional-samples "${UNCONDITIONAL_SAMPLES}" \
    --query-file "${QUERY_FILE}" \
    --real-data "${REAL_DATA}" \
    --output-dir "${OUTPUT_DIR}"

echo "Finished structured-query evaluation"

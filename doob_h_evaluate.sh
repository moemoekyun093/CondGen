#!/bin/bash
#SBATCH --job-name=doob_h_evaluate
#SBATCH --output=evaluations/slurm/%x_%A_%a.out
#SBATCH --error=evaluations/slurm/%x_%A_%a.err
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
SAMPLE_SUFFIX="${SAMPLE_SUFFIX-_all_two_guides}"
SAMPLE_NAME="${MODEL_NAME}${SAMPLE_SUFFIX}"
SAMPLES="conditional_samples/${DATANAME}/${SAMPLE_NAME}.csv"
SAVED_ACTIVE_QUERY="conditional_samples/${DATANAME}/${SAMPLE_NAME}.query.json"
if [ -z "${QUERY_FILE:-}" ] && [ -f "${SAVED_ACTIVE_QUERY}" ]; then
    QUERY_FILE="${SAVED_ACTIVE_QUERY}"
else
    QUERY_FILE="${QUERY_FILE:-constraints/${DATANAME}/fixed_numerical_intervals.json}"
fi
EVALUATION_ROOT="${EVALUATION_ROOT:-evaluations}"
OUTPUT_DIR="${EVALUATION_ROOT}/${DATANAME}/${SAMPLE_NAME}"
RAW_REPORT="${OUTPUT_DIR}/raw_diagnostic.json"
REAL_DATA="synthetic/${DATANAME}/real.csv"
UNCONDITIONAL_EPOCH="${UNCONDITIONAL_EPOCH:-8000}"
UNCONDITIONAL_SAMPLES="${UNCONDITIONAL_SAMPLES:-}"
if [ -z "${UNCONDITIONAL_SAMPLES}" ]; then
    UNCONDITIONAL_CANDIDATES=(
        "tabdiff/result/${DATANAME}/${MODEL_NAME}/${UNCONDITIONAL_EPOCH}/samples.csv"
        "tabdiff/result/${DATANAME}/${MODEL_NAME}/all_samples/samples_0.csv"
        "eval/report_runs/${MODEL_NAME}/${DATANAME}/all_samples/samples_0.csv"
    )
    for CANDIDATE in "${UNCONDITIONAL_CANDIDATES[@]}"; do
        if [ -f "${CANDIDATE}" ]; then
            UNCONDITIONAL_SAMPLES="${CANDIDATE}"
            break
        fi
    done
fi

mkdir -p "${OUTPUT_DIR}"

if [ -z "${UNCONDITIONAL_SAMPLES}" ]; then
    echo "ERROR: could not locate unconditional samples for ${MODEL_NAME}"
    echo "Set UNCONDITIONAL_SAMPLES to the matching unconditional samples.csv"
    exit 1
fi

for path in "${SAMPLES}" "${UNCONDITIONAL_SAMPLES}" "${QUERY_FILE}" "${REAL_DATA}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}"
        exit 1
    fi
done

echo "========================================"
echo "Dataset        : ${DATANAME}"
echo "Model          : ${MODEL_NAME}"
echo "Existing CSV   : ${SAMPLES}"
echo "Unconditional  : ${UNCONDITIONAL_SAMPLES}"
echo "Uncond. epoch  : ${UNCONDITIONAL_EPOCH}"
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
    --unconditional-samples "${UNCONDITIONAL_SAMPLES}" \
    --query-file "${QUERY_FILE}" \
    --real-data "${REAL_DATA}" \
    --output-dir "${OUTPUT_DIR}"

echo "Finished existing-sample diagnostics and density evaluation"

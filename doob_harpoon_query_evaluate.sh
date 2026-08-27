#!/bin/bash
#SBATCH --job-name=query_compare
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
DOOB_SAMPLES="${DOOB_SAMPLES:-conditional_samples/${DATANAME}/${MODEL_NAME}_${QUERY_ID}_structured.csv}"
HARPOON_SAMPLES="${HARPOON_SAMPLES:-conditional_samples/${DATANAME}/harpoon_${QUERY_ID}_eta02.csv}"
REAL_DATA="${REAL_DATA:-synthetic/${DATANAME}/real.csv}"
INFO_FILE="${INFO_FILE:-data/${DATANAME}/info.json}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/${QUERY_ID}_doob_vs_harpoon}"

mkdir -p evaluations/slurm "${OUTPUT_DIR}"

python -u evaluate_query_methods.py \
    --dataname "${DATANAME}" \
    --query-file "${QUERY_FILE}" \
    --doob-samples "${DOOB_SAMPLES}" \
    --harpoon-samples "${HARPOON_SAMPLES}" \
    --real-data "${REAL_DATA}" \
    --info-file "${INFO_FILE}" \
    --output-dir "${OUTPUT_DIR}"

echo "Finished paired Doob/HARPOON evaluation"

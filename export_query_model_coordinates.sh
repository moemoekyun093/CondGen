#!/bin/bash
#SBATCH --job-name=query_coordinate_export
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

python -u export_query_model_coordinates.py \
    --dataname "${DATANAME}" \
    --base-exp-name "${MODEL_NAME}" \
    --query-dir "${QUERY_DIR}" \
    --output "${QUERY_COORDINATES}"

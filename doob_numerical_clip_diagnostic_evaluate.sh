#!/bin/bash
#SBATCH --job-name=doob_clip_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
OUTPUT_ROOT="${CLIP_DIAGNOSTIC_ROOT:-conditional_samples/${DATANAME}/numerical_clip_diagnostic}"
EVAL_DIR="${CLIP_DIAGNOSTIC_EVAL_DIR:-evaluations/${DATANAME}/numerical_clip_diagnostic}"
mkdir -p evaluations/slurm "${EVAL_DIR}"

python -u evaluate_numerical_clip_diagnostic.py \
    --sample-root "${OUTPUT_ROOT}" \
    --query-dir "${QUERY_DIR}" \
    --output-dir "${EVAL_DIR}"

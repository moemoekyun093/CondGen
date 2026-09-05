#!/bin/bash
#SBATCH --job-name=tabbyflow_cond
#SBATCH --output=logs/tabbyflow/%x_%A_%a.out
#SBATCH --error=logs/tabbyflow/%x_%A_%a.err
#SBATCH --gres=min-vram:8g,min-cuda-cc:70
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-/scratch/work/agrawaa4/TabDiff}"
cd "${TABDIFF_PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

TABBYFLOW_PYTHON="${TABBYFLOW_PYTHON:?set TABBYFLOW_PYTHON to an environment containing torchdiffeq}"
arguments=(
    --run-dir "${TABBYFLOW_RUN_DIR:?set TABBYFLOW_RUN_DIR}"
    --query-dir "${QUERY_DIR:?set QUERY_DIR}"
    --data-dir "${DATA_DIR:?set DATA_DIR}"
    --info-file "${INFO_FILE:?set INFO_FILE}"
    --output-dir "${SAMPLE_DIR:?set SAMPLE_DIR}"
    --num-samples "${NUM_SAMPLES:-1000}"
    --batch-size "${BATCH_SIZE:-1000}"
    --seed-bases "${SEED_BASES:-10000}"
    --bundle-index "${SLURM_ARRAY_TASK_ID:-0}"
    --bundle-count "${BUNDLE_COUNT:-1}"
    --solver "${TABBYFLOW_SOLVER:-heun}"
    --steps "${TABBYFLOW_STEPS:-50}"
    --device cuda
    --query-split "${QUERY_SPLIT:-test}"
)
[ -n "${TABBYFLOW_CHECKPOINT:-}" ] && arguments+=(--checkpoint "${TABBYFLOW_CHECKPOINT}")
[ -n "${TABBYFLOW_TRANSFORM:-}" ] && arguments+=(--transform-file "${TABBYFLOW_TRANSFORM}")
[ -n "${QUERY_SPLIT_MANIFEST:-}" ] && arguments+=(
    --query-split-manifest "${QUERY_SPLIT_MANIFEST}"
)
exec "${TABBYFLOW_PYTHON}" -u sample_conditional_tabbyflow_suite.py "${arguments[@]}"

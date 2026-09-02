#!/bin/bash
#SBATCH --job-name=query_alpha_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

ALPHA_ENV="${ALPHA_ENV:-/scratch/work/agrawaa4/conda_envs/alpha}"
if [ ! -x "${ALPHA_ENV}/bin/python" ]; then
    echo "ERROR: Alpha environment Python not found: ${ALPHA_ENV}/bin/python"
    exit 1
fi

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:?set QUERY_DIR}"
QUERY_SPLIT_MANIFEST="${QUERY_SPLIT_MANIFEST:?set QUERY_SPLIT_MANIFEST}"
QUERY_SPLIT="${QUERY_SPLIT:-test}"
SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-}"
SEED_BASES="${SEED_BASES:?set SEED_BASES}"
OUTPUT="${ALPHA_BETA_RESULTS:?set ALPHA_BETA_RESULTS}"
EVAL_WORKERS="${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

seed_args=()
IFS=',' read -r -a seeds <<< "${SEED_BASES}"
for seed in "${seeds[@]}"; do
    seed_args+=(--sample-seed-base "${seed}")
done

method_args=()
if [ -n "${METHOD_SPECS:-}" ]; then
    IFS='|' read -r -a method_specs <<< "${METHOD_SPECS}"
    for spec in "${method_specs[@]}"; do
        method_args+=(--method "${spec}")
    done
else
    SAMPLE_ROOT="${SAMPLE_ROOT:?set SUITE_SAMPLE_ROOT for legacy two-method mode}"
    DOOB_LABEL="${DOOB_LABEL:?set DOOB_LABEL or METHOD_SPECS}"
    HARPOON_LABEL="${HARPOON_LABEL:?set HARPOON_LABEL or METHOD_SPECS}"
    method_args+=(
        --method "${DOOB_LABEL}=${SAMPLE_ROOT}/${DOOB_LABEL}"
        --method "${HARPOON_LABEL}=${SAMPLE_ROOT}/${HARPOON_LABEL}"
    )
fi

"${ALPHA_ENV}/bin/python" -u evaluate_synthcity_alpha_suite.py \
    --query-dir "${QUERY_DIR}" \
    --query-split-manifest "${QUERY_SPLIT_MANIFEST}" \
    --query-split "${QUERY_SPLIT}" \
    "${method_args[@]}" \
    "${seed_args[@]}" \
    --workers "${EVAL_WORKERS}" \
    --real-data "${REAL_DATA:-synthetic/${DATANAME}/real.csv}" \
    --info-file "${INFO_FILE:-data/${DATANAME}/info.json}" \
    --output "${OUTPUT}"

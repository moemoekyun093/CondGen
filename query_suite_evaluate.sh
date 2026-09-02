#!/bin/bash
#SBATCH --job-name=query_suite_eval
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

EVAL_PYTHON="${EVAL_PYTHON:-/scratch/work/agrawaa4/conda_envs/relgdiff/bin/python}"
[ -x "${EVAL_PYTHON}" ] || { echo "ERROR: Python not found: ${EVAL_PYTHON}"; exit 1; }
DATANAME="${DATANAME:?set DATANAME}"
QUERY_DIR="${QUERY_DIR:?set QUERY_DIR}"
SUITE_EVAL_DIR="${SUITE_EVAL_DIR:?set SUITE_EVAL_DIR}"
METHOD_SPECS="${METHOD_SPECS:?set pipe-separated LABEL=DIRECTORY methods}"
SEED_BASES="${SEED_BASES:?set SEED_BASES}"

method_args=()
IFS='|' read -r -a methods <<< "${METHOD_SPECS}"
for spec in "${methods[@]}"; do method_args+=(--method "${spec}"); done
seed_args=()
IFS=',' read -r -a seeds <<< "${SEED_BASES}"
for seed in "${seeds[@]}"; do seed_args+=(--sample-seed-base "${seed}"); done
split_args=()
if [ -n "${QUERY_SPLIT_MANIFEST:-}" ]; then
    split_args+=(--query-split-manifest "${QUERY_SPLIT_MANIFEST}" --query-split "${QUERY_SPLIT:-test}")
fi
baseline_args=()
[ -n "${EVAL_BASELINE_METHOD:-}" ] && baseline_args+=(--baseline-method "${EVAL_BASELINE_METHOD}")
alpha_args=()
[ -n "${ALPHA_BETA_RESULTS:-}" ] && alpha_args+=(--alpha-beta-results "${ALPHA_BETA_RESULTS}")
coordinate_args=()
[ -n "${QUERY_COORDINATES:-}" ] && coordinate_args+=(--query-coordinates "${QUERY_COORDINATES}")

mkdir -p evaluations/slurm "${SUITE_EVAL_DIR}"
"${EVAL_PYTHON}" -u evaluate_doob_query_suite.py \
    --query-dir "${QUERY_DIR}" \
    "${method_args[@]}" \
    "${seed_args[@]}" \
    "${split_args[@]}" \
    "${baseline_args[@]}" \
    "${alpha_args[@]}" \
    "${coordinate_args[@]}" \
    --real-data "${REAL_DATA:-synthetic/${DATANAME}/real.csv}" \
    --info-file "${INFO_FILE:-data/${DATANAME}/info.json}" \
    --group-by "${EVAL_GROUP_BY:-target_band}" \
    --interval-width-bins "${INTERVAL_WIDTH_BINS:-10}" \
    --workers "${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}" \
    --filtered-min-rows "${FILTERED_MIN_ROWS:-50}" \
    --output-dir "${SUITE_EVAL_DIR}"

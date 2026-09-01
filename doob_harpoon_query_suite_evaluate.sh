#!/bin/bash
#SBATCH --job-name=doob_harpoon_suite_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

NORMAL_EVAL_ENV="${NORMAL_EVAL_ENV:-/scratch/work/agrawaa4/conda_envs/relgdiff}"
EVAL_PYTHON="${EVAL_PYTHON:-${NORMAL_EVAL_ENV}/bin/python}"
if [ ! -x "${EVAL_PYTHON}" ]; then
    echo "ERROR: relgdiff evaluation Python not found: ${EVAL_PYTHON}"
    exit 1
fi

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_suite_comparison}"
DOOB_LABEL="${DOOB_LABEL:-doob_curriculum}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_eta02}"
SUITE_EVAL_DIR="${SUITE_EVAL_DIR:-evaluations/${DATANAME}/doob_vs_harpoon_query_suite}"
REAL_DATA="${REAL_DATA:-synthetic/${DATANAME}/real.csv}"
INFO_FILE="${INFO_FILE:-data/${DATANAME}/info.json}"
EVAL_GROUP_BY="${EVAL_GROUP_BY:-target_band}"
EVAL_BASELINE_METHOD="${EVAL_BASELINE_METHOD:-}"
QUERY_COORDINATES="${QUERY_COORDINATES:-data90/${DATANAME}/query_splits/query_model_coordinates.json}"
INTERVAL_WIDTH_BINS="${INTERVAL_WIDTH_BINS:-10}"
ALPHA_BETA_RESULTS="${ALPHA_BETA_RESULTS:-}"
SEED_BASES="${SEED_BASES:-}"
EVAL_WORKERS="${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

QUERY_FILTER_ARGS=()
if [ -n "${QUERY_SPLIT_MANIFEST:-}" ]; then
    QUERY_FILTER_ARGS+=(
        --query-split-manifest "${QUERY_SPLIT_MANIFEST}"
        --query-split "${QUERY_SPLIT:?QUERY_SPLIT is required with QUERY_SPLIT_MANIFEST}"
    )
fi
if [ "${QUERY_TEST_SUPPORTED_ONLY:-0}" = "1" ]; then
    QUERY_FILTER_ARGS+=(--test-supported-only)
fi
BASELINE_ARGS=()
if [ -n "${EVAL_BASELINE_METHOD}" ]; then
    BASELINE_ARGS+=(--baseline-method "${EVAL_BASELINE_METHOD}")
fi
ALPHA_ARGS=()
if [ -n "${ALPHA_BETA_RESULTS}" ]; then
    ALPHA_ARGS+=(--alpha-beta-results "${ALPHA_BETA_RESULTS}")
fi
SEED_ARGS=()
if [ -n "${SEED_BASES}" ]; then
    IFS=',' read -r -a seeds <<< "${SEED_BASES}"
    for seed in "${seeds[@]}"; do
        SEED_ARGS+=(--sample-seed-base "${seed}")
    done
fi

mkdir -p evaluations/slurm "${SUITE_EVAL_DIR}"

if [ ! -f "${QUERY_COORDINATES}" ]; then
    COORDINATE_BASE_ARGS=()
    if [ -n "${BASE_CHECKPOINT:-}" ]; then
        COORDINATE_BASE_ARGS+=(--base-ckpt "${BASE_CHECKPOINT}")
    fi
    "${EVAL_PYTHON}" -u export_query_model_coordinates.py \
        --dataname "${DATANAME}" \
        "${COORDINATE_BASE_ARGS[@]}" \
        --base-exp-name "${MODEL_NAME:-ft_periodic_seed0}" \
        --query-dir "${QUERY_DIR}" \
        --output "${QUERY_COORDINATES}"
fi

"${EVAL_PYTHON}" -u evaluate_doob_query_suite.py \
    --query-dir "${QUERY_DIR}" \
    --method "${DOOB_LABEL}=${SUITE_SAMPLE_ROOT}/${DOOB_LABEL}" \
    --method "${HARPOON_LABEL}=${SUITE_SAMPLE_ROOT}/${HARPOON_LABEL}" \
    --real-data "${REAL_DATA}" \
    --info-file "${INFO_FILE}" \
    --group-by "${EVAL_GROUP_BY}" \
    --query-coordinates "${QUERY_COORDINATES}" \
    --interval-width-bins "${INTERVAL_WIDTH_BINS}" \
    --workers "${EVAL_WORKERS}" \
    "${QUERY_FILTER_ARGS[@]}" \
    "${BASELINE_ARGS[@]}" \
    "${ALPHA_ARGS[@]}" \
    "${SEED_ARGS[@]}" \
    --output-dir "${SUITE_EVAL_DIR}"

echo "Modality columns are included in the grouped evaluation CSV:"
echo "  numeric_joint_miss_rate_mean / categorical_joint_miss_rate_mean"
echo "  numeric_mean_column_miss_rate_mean / categorical_mean_column_miss_rate_mean"

#!/bin/bash
#SBATCH --job-name=doob_mask_grid_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

NORMAL_EVAL_ENV="${NORMAL_EVAL_ENV:-/scratch/work/agrawaa4/conda_envs/relgdiff}"
if [ ! -d "${NORMAL_EVAL_ENV}" ] || [ -z "${CONDA_EXE:-}" ]; then
    echo "ERROR: relgdiff or initialized Conda is unavailable"
    exit 1
fi
CONDA_BASE="${CONDA_EXE%/bin/conda}"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${NORMAL_EVAL_ENV}"
set -u
echo "Normal evaluation environment: ${CONDA_PREFIX}"

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_selectivity_mask_grid}"
METHOD_LABEL="${METHOD_LABEL:-doob_masked_25000}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_style_tabdiff_eta02_s50}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/selectivity_mask_grid}"
SUITE_EVAL_DIR="${SUITE_EVAL_DIR:-evaluations/${DATANAME}/selectivity_mask_grid}"
FILTERED_MIN_ROWS="${FILTERED_MIN_ROWS:-50}"
FILTERED_BOOTSTRAP_REPEATS="${FILTERED_BOOTSTRAP_REPEATS:-5}"
FILTERED_BOOTSTRAP_CAP="${FILTERED_BOOTSTRAP_CAP:-1000}"
mkdir -p evaluations/slurm "${SUITE_EVAL_DIR}"

python -u evaluate_doob_query_suite.py \
    --query-dir "${QUERY_DIR}" \
    --method "${METHOD_LABEL}=${SUITE_SAMPLE_ROOT}/${METHOD_LABEL}" \
    --method "${HARPOON_LABEL}=${SUITE_SAMPLE_ROOT}/${HARPOON_LABEL}" \
    --baseline-method "${HARPOON_LABEL}" \
    --filtered-min-rows "${FILTERED_MIN_ROWS}" \
    --filtered-bootstrap-repeats "${FILTERED_BOOTSTRAP_REPEATS}" \
    --filtered-bootstrap-cap "${FILTERED_BOOTSTRAP_CAP}" \
    --real-data "synthetic/${DATANAME}/real.csv" \
    --info-file "data/${DATANAME}/info.json" \
    --group-by target_band \
    --output-dir "${SUITE_EVAL_DIR}"

python -u plot_query_mask_grid_2d.py \
    --per-query "${SUITE_EVAL_DIR}/per_query.csv" \
    --output-dir "${SUITE_EVAL_DIR}" \
    --primary-method "${METHOD_LABEL}" \
    --baseline-method "${HARPOON_LABEL}"

echo "Finished selectivity-by-mask evaluation"

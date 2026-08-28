#!/bin/bash
#SBATCH --job-name=fixed_query_list_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

ALPHA_ENV="${ALPHA_ENV:-/scratch/work/agrawaa4/conda_envs/alpha}"
if [ ! -d "${ALPHA_ENV}" ]; then
    echo "ERROR: SynthCity evaluation environment not found: ${ALPHA_ENV}"
    exit 1
fi
if [ -z "${CONDA_EXE:-}" ]; then
    echo "ERROR: CONDA_EXE is unavailable; submit from a shell with Conda initialized"
    exit 1
fi
CONDA_BASE="${CONDA_EXE%/bin/conda}"
# Some site-provided Conda deactivate hooks read optional backup variables.
# Temporarily disable nounset while Conda switches from the inherited submit
# environment to the dedicated SynthCity environment.
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ALPHA_ENV}"
set -u
echo "Evaluation environment: ${CONDA_PREFIX}"
python -c "from synthcity.metrics import eval_statistical; print('SynthCity AlphaPrecision available')"

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_fixed_box_permuted}"
METHOD_SPECS="${METHOD_SPECS:?METHOD_SPECS must be comma-separated LABEL=SAMPLE_DIRECTORY entries}"
BASELINE_LABEL="${BASELINE_LABEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/fixed_box_permuted_model_comparison}"

IFS=',' read -r -a METHODS <<< "${METHOD_SPECS}"
if [ "${#METHODS[@]}" -lt 2 ]; then
    echo "ERROR: at least two methods are required"
    exit 1
fi

ARGS=()
for METHOD in "${METHODS[@]}"; do
    if [[ "${METHOD}" != *=* ]]; then
        echo "ERROR: invalid method specification: ${METHOD}"
        exit 1
    fi
    ARGS+=(--method "${METHOD}")
done
if [ -n "${BASELINE_LABEL}" ]; then
    ARGS+=(--baseline-method "${BASELINE_LABEL}")
fi

mkdir -p evaluations/slurm "${OUTPUT_DIR}"
python -u evaluate_doob_query_suite.py \
    --query-dir "${QUERY_DIR}" \
    "${ARGS[@]}" \
    --real-data "synthetic/${DATANAME}/real.csv" \
    --info-file "data/${DATANAME}/info.json" \
    --group-by arity \
    --alpha-beta-seed 0 \
    --output-dir "${OUTPUT_DIR}"

echo "Saved list-driven fixed-query comparison to ${OUTPUT_DIR}"

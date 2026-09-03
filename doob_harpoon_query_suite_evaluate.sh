#!/bin/bash
# Deprecated Slurm compatibility wrapper around query_suite_evaluate.sh.
#SBATCH --job-name=doob_harpoon_suite_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-/scratch/work/agrawaa4/TabDiff}"
cd "${TABDIFF_PROJECT_ROOT}"
export TABDIFF_PROJECT_ROOT

DATANAME="${DATANAME:-shoppers}"
DOOB_LABEL="${DOOB_LABEL:-doob_curriculum}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_eta02}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_suite_comparison}"
METHOD_SPECS="${METHOD_SPECS:-${DOOB_LABEL}=${SUITE_SAMPLE_ROOT}/${DOOB_LABEL}|${HARPOON_LABEL}=${SUITE_SAMPLE_ROOT}/${HARPOON_LABEL}}"
QUERY_COORDINATES="${QUERY_COORDINATES:-data90/${DATANAME}/query_splits/query_model_coordinates.json}"
export DATANAME METHOD_SPECS QUERY_COORDINATES

echo "NOTICE: doob_harpoon_query_suite_evaluate.sh is deprecated; using query_suite_evaluate.sh."
exec bash query_suite_evaluate.sh

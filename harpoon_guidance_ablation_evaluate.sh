#!/bin/bash
#SBATCH --job-name=harpoon_ablation_eval
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_suite_comparison}"
GUIDED_LABEL="${GUIDED_LABEL:-harpoon_eta02}"
UNGUIDED_LABEL="${UNGUIDED_LABEL:-harpoon_unconditional}"
OUTPUT_DIR="${HARPOON_ABLATION_EVAL_DIR:-evaluations/${DATANAME}/harpoon_guidance_ablation}"
mkdir -p evaluations/slurm "${OUTPUT_DIR}"

python -u evaluate_doob_query_suite.py \
    --query-dir "${QUERY_DIR}" \
    --method "${GUIDED_LABEL}=${SUITE_SAMPLE_ROOT}/${GUIDED_LABEL}" \
    --method "${UNGUIDED_LABEL}=${SUITE_SAMPLE_ROOT}/${UNGUIDED_LABEL}" \
    --baseline-method "${UNGUIDED_LABEL}" \
    --group-by target_band \
    --output-dir "${OUTPUT_DIR}"

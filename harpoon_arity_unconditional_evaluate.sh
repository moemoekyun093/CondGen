#!/bin/bash
#SBATCH --job-name=harpoon_arity_ablate
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${ARITY_QUERY_DIR:-data90/${DATANAME}/queries_arity_b40}"
SAMPLE_ROOT="${ARITY_SAMPLE_ROOT:-conditional_samples/${DATANAME}/arity_b40_comparison}"
GUIDED_LABEL="${GUIDED_ARITY_LABEL:-harpoon_eta02_arity}"
UNCONDITIONAL_LABEL="${UNCONDITIONAL_ARITY_LABEL:-harpoon_unconditional_shared_b40}"
OUTPUT_DIR="${ARITY_ABLATION_EVAL_DIR:-evaluations/${DATANAME}/harpoon_arity_unconditional_ablation}"
mkdir -p evaluations/slurm "${OUTPUT_DIR}"

python -u evaluate_harpoon_arity_ablation.py \
    --query-dir "${QUERY_DIR}" \
    --guided-samples "${SAMPLE_ROOT}/${GUIDED_LABEL}" \
    --unconditional-source-samples "${SAMPLE_ROOT}/${UNCONDITIONAL_LABEL}" \
    --output-dir "${OUTPUT_DIR}"

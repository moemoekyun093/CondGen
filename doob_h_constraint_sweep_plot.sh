#!/bin/bash
#SBATCH --job-name=doob_h_constraint_plot
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${FT_MODEL:-ft_periodic_seed0}"
SWEEP_NAME="${1:-d48_l2_6000}"
REPORT_GLOB="conditional_samples/${DATANAME}/${MODEL_NAME}_constraint_sweep_${SWEEP_NAME}_k*.constraints.json"
OUTPUT_DIR="evaluations/${DATANAME}/${MODEL_NAME}_constraint_sweep_${SWEEP_NAME}"

mkdir -p "${OUTPUT_DIR}" evaluations/slurm

python -u plot_doob_constraint_sweep.py \
    --reports-glob "${REPORT_GLOB}" \
    --output-dir "${OUTPUT_DIR}"

echo "Saved constraint-count sweep plot to ${OUTPUT_DIR}"

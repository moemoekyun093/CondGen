#!/bin/bash
#SBATCH --job-name=harpoon_doob_compare
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${FT_MODEL:-ft_periodic_seed0}"
DOOB_SWEEP_NAME="${1:-d48_l2_6000}"
HARPOON_SWEEP_NAME="${2:-summed_relu_eta02}"
REAL_DATA="${REAL_DATA:-synthetic/${DATANAME}/real.csv}"
OUTPUT_DIR="evaluations/${DATANAME}/doob_${DOOB_SWEEP_NAME}_vs_harpoon_${HARPOON_SWEEP_NAME}"

mkdir -p "${OUTPUT_DIR}" evaluations/slurm

python -u compare_constraint_sweeps.py \
    --doob-samples-glob "conditional_samples/${DATANAME}/${MODEL_NAME}_constraint_sweep_${DOOB_SWEEP_NAME}_k*.csv" \
    --harpoon-samples-glob "conditional_samples/${DATANAME}/harpoon_constraint_sweep_${HARPOON_SWEEP_NAME}_k*.csv" \
    --real-data "${REAL_DATA}" \
    --info-file "data/${DATANAME}/info.json" \
    --output-dir "${OUTPUT_DIR}"

echo "Finished Doob-versus-HARPOON constraint-count comparison"

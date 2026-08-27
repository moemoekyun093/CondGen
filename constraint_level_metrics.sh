#!/bin/bash
#SBATCH --job-name=constraint_level_metrics
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

if [ "$#" -lt 1 ]; then
    echo "Usage:"
    echo "  sbatch constraint_level_metrics.sh CONFIG_JSON [OUTPUT_DIR]"
    echo "  sbatch constraint_level_metrics.sh --method LABEL GLOB [...] --unconditional LABEL CSV --real-data CSV --info-file JSON --output-dir DIR"
    exit 1
fi

mkdir -p evaluations/slurm
if [[ "$1" == --* ]]; then
    ARGS=("$@")
else
    CONFIG="$1"
    OUTPUT_DIR="${2:-}"
    if [ ! -f "${CONFIG}" ]; then
        echo "ERROR: comparison config not found: ${CONFIG}"
        exit 1
    fi
    ARGS=(--config "${CONFIG}")
    if [ -n "${OUTPUT_DIR}" ]; then
        ARGS+=(--output-dir "${OUTPUT_DIR}")
    fi
fi

python -u plot_constraint_level_metrics.py "${ARGS[@]}"

echo "Finished constraint-level density, C2ST, Alpha Precision, Beta Recall, and violation plots"

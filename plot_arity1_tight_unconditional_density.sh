#!/bin/bash
#SBATCH --job-name=arity1_uncond_density
#SBATCH --output=evaluations/slurm/%x_%j.out
#SBATCH --error=evaluations/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
UNCONDITIONAL_SAMPLES="${UNCONDITIONAL_SAMPLES:-tabdiff/result/${DATANAME}/${MODEL_NAME}/8000/samples.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/arity1_tight_unconditional_density}"
MAX_TARGET_BAND="${MAX_TARGET_BAND:-0.02}"
MAX_REALIZED_SELECTIVITY="${MAX_REALIZED_SELECTIVITY:-0.025}"
WINDOW_WIDTHS="${WINDOW_WIDTHS:-5}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-200}"

for REQUIRED in "${QUERY_DIR}" "${UNCONDITIONAL_SAMPLES}"; do
    if [ ! -e "${REQUIRED}" ]; then
        echo "ERROR: required existing path not found: ${REQUIRED}"
        exit 1
    fi
done
mkdir -p evaluations/slurm "${OUTPUT_DIR}"
RELgDIFF_PYTHON="/scratch/work/agrawaa4/conda_envs/relgdiff/bin/python"
"${RELgDIFF_PYTHON}" -u plot_arity1_tight_unconditional_density.py \
    --query-dir "${QUERY_DIR}" \
    --unconditional-samples "${UNCONDITIONAL_SAMPLES}" \
    --dataname "${DATANAME}" \
    --data-dir "data/${DATANAME}" \
    --info-file "data/${DATANAME}/info.json" \
    --base-config "tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/config.pkl" \
    --output-dir "${OUTPUT_DIR}" \
    --max-target-band "${MAX_TARGET_BAND}" \
    --max-realized-selectivity "${MAX_REALIZED_SELECTIVITY}" \
    --window-widths "${WINDOW_WIDTHS}" \
    --bins "${HISTOGRAM_BINS}"

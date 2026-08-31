#!/bin/bash
#SBATCH --job-name=doob_side_by_side
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
QUERY_ID="${QUERY_ID:-q_shoppers_b00p5_k01_num_1}"
QUERY_FILE="${QUERY_FILE:-data90/${DATANAME}/queries/${QUERY_ID}.json}"
REAL_DATA="${REAL_DATA:-synthetic/${DATANAME}/real.csv}"
UNCONDITIONAL_SAMPLES="${UNCONDITIONAL_SAMPLES:-tabdiff/result/${DATANAME}/${MODEL_NAME}/8000/samples.csv}"
CENTER_WIDTH_SAMPLES="${CENTER_WIDTH_SAMPLES:-conditional_samples/${DATANAME}/fixed_query_center_logwidth_mlp_sampling_sweeps/ordinary_mlp_center_logwidth/steps_050_lambda_1.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/fixed_query_conditional_unconditional_center_width}"
ROWS="${ROWS:-1000}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-50}"

for REQUIRED in "${QUERY_FILE}" "${REAL_DATA}" "${UNCONDITIONAL_SAMPLES}" "${CENTER_WIDTH_SAMPLES}"; do
    if [ ! -f "${REQUIRED}" ]; then
        echo "ERROR: required existing file not found: ${REQUIRED}"
        echo "This evaluation-only script will not generate replacement samples."
        exit 1
    fi
done
mkdir -p evaluations/slurm "${OUTPUT_DIR}"

RELgDIFF_PYTHON="/scratch/work/agrawaa4/conda_envs/relgdiff/bin/python"
"${RELgDIFF_PYTHON}" -u plot_fixed_query_conditional_unconditional_center_width.py \
    --query-file "${QUERY_FILE}" \
    --real-data "${REAL_DATA}" \
    --unconditional-samples "${UNCONDITIONAL_SAMPLES}" \
    --center-width-samples "${CENTER_WIDTH_SAMPLES}" \
    --dataname "${DATANAME}" \
    --data-dir "data/${DATANAME}" \
    --info-file "data/${DATANAME}/info.json" \
    --base-config "tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/config.pkl" \
    --output-dir "${OUTPUT_DIR}" \
    --rows "${ROWS}" \
    --bins "${HISTOGRAM_BINS}"

#!/bin/bash
#SBATCH --job-name=doob_h_intervals
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-news}"
REFERENCE_MODEL="ft_periodic_L6_d128_seed0"
CKPT_DIR="tabdiff/ckpt/${DATANAME}/${REFERENCE_MODEL}"
CKPT_CANDIDATES=("${CKPT_DIR}"/best_ema_model_*.pt)
if [ ! -e "${CKPT_CANDIDATES[0]}" ] || [ "${#CKPT_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one best EMA checkpoint in ${CKPT_DIR}"
    printf '%s\n' "${CKPT_CANDIDATES[@]}"
    exit 1
fi
BASE_CKPT="${CKPT_CANDIDATES[0]}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.30}"
QUERY_FILE="${QUERY_FILE:-constraints/${DATANAME}/fixed_numerical_intervals.json}"

echo "========================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node            : ${SLURMD_NODENAME}"
echo "Dataset         : ${DATANAME}"
echo "Reference model : ${REFERENCE_MODEL}"
echo "Target coverage : ${TARGET_COVERAGE}"
echo "Query file      : ${QUERY_FILE}"
echo "========================================"

if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: base checkpoint not found: ${BASE_CKPT}"
    exit 1
fi

python generate_doob_intervals.py \
    --dataname "${DATANAME}" \
    --base-ckpt "${BASE_CKPT}" \
    --target-coverage "${TARGET_COVERAGE}" \
    --output "${QUERY_FILE}"

echo "Finished deterministic interval generation"

#!/bin/bash
#SBATCH --job-name=doob_query_sample
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
CKPT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
CKPT_CANDIDATES=("${CKPT_DIR}"/best_ema_model_*.pt)
if [ ! -e "${CKPT_CANDIDATES[0]}" ] || [ "${#CKPT_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one best EMA checkpoint in ${CKPT_DIR}"
    exit 1
fi
BASE_CKPT="${CKPT_CANDIDATES[0]}"
GUIDE_DIR_ARG="${1:-${GUIDE_DIR_NAME:-doob_query_structured_d48_l2_6000}}"
if [[ "${GUIDE_DIR_ARG}" == */* ]]; then
    GUIDE_DIR="${GUIDE_DIR_ARG}"
else
    GUIDE_DIR="${CKPT_DIR}/${GUIDE_DIR_ARG}"
fi
GUIDE_CKPT="${GUIDE_DIR}/best_guide.pt"
QUERY_FILE="${QUERY_FILE:-data90/${DATANAME}/queries_full/qf_shoppers_b10p0_0.json}"
QUERY_ID="$(basename "${QUERY_FILE}" .json)"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
OUTPUT="${OUTPUT:-conditional_samples/${DATANAME}/${MODEL_NAME}_${QUERY_ID}_structured.csv}"

echo "========================================"
echo "Dataset         : ${DATANAME}"
echo "Base checkpoint : ${BASE_CKPT}"
echo "Guide checkpoint: ${GUIDE_CKPT}"
echo "Query           : ${QUERY_FILE}"
echo "Output          : ${OUTPUT}"
echo "========================================"
nvidia-smi

for path in "${BASE_CKPT}" "${GUIDE_CKPT}" "${QUERY_FILE}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}"
        exit 1
    fi
done

python -u sample_doob_query.py \
    --guide-ckpt "${GUIDE_CKPT}" \
    --base-ckpt "${BASE_CKPT}" \
    --query-file "${QUERY_FILE}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --output "${OUTPUT}" \
    --device cuda

echo "Finished structured-query Doob sampling"

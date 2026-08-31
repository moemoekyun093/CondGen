#!/bin/bash
#SBATCH --job-name=center_logw_multiseed
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:16g,min-cuda-cc:70
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

BASE_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
BASE_CANDIDATES=("${BASE_DIR}"/best_ema_model_*.pt)
if [ ! -e "${BASE_CANDIDATES[0]}" ] || [ "${#BASE_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one base EMA checkpoint in ${BASE_DIR}"
    exit 1
fi
GUIDE_CKPT="${MLP_CENTER_WIDTH_GUIDE_DIR}/guide_4000.pt"
if [ ! -f "${GUIDE_CKPT}" ]; then
    echo "ERROR: trained checkpoint not found: ${GUIDE_CKPT}"
    exit 1
fi

mkdir -p "${MULTISEED_SAMPLE_DIR}"
IFS=',' read -r -a SEED_VALUES <<< "${SEEDS}"
echo "Ordinary MLP center/log-width paired-seed sampling"
echo "Seeds            : ${SEEDS}"
echo "Reverse steps    : 50"
echo "Guidance lambda  : 1"
nvidia-smi

for SEED in "${SEED_VALUES[@]}"; do
    OUTPUT="${MULTISEED_SAMPLE_DIR}/seed_${SEED}.csv"
    if [ -f "${OUTPUT}" ]; then
        echo "Reusing ${OUTPUT}"
        continue
    fi
    if [ "${SEED}" = "74000" ] && [ -f "${EXISTING_SEED_74000_SAMPLE}" ]; then
        ln -s "$(realpath "${EXISTING_SEED_74000_SAMPLE}")" "${OUTPUT}"
        echo "Linked existing paired seed 74000 sample to ${OUTPUT}"
        continue
    fi
    echo "Sampling seed ${SEED}"
    python -u sample_doob_query.py \
        --guide-ckpt "${GUIDE_CKPT}" \
        --base-ckpt "${BASE_CANDIDATES[0]}" \
        --query-file "${QUERY_FILE}" \
        --num-samples "${NUM_SAMPLES}" \
        --batch-size "${NUM_SAMPLES}" \
        --num-timesteps 50 \
        --guidance-strength 1 \
        --seed "${SEED}" \
        --output "${OUTPUT}" \
        --device cuda
done

echo "Finished center/log-width multiseed sampling"

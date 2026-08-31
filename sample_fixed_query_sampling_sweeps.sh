#!/bin/bash
#SBATCH --job-name=doob_sampling_sweeps
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:16g,min-cuda-cc:70
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1

BASE_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
BASE_CANDIDATES=("${BASE_DIR}"/best_ema_model_*.pt)
if [ ! -e "${BASE_CANDIDATES[0]}" ] || [ "${#BASE_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one base EMA checkpoint in ${BASE_DIR}"
    exit 1
fi

IFS=',' read -r -a REVERSE_STEP_VALUES <<< "${REVERSE_STEPS:-50,75,100,150,200}"
IFS=',' read -r -a LAMBDA_VALUES <<< "${GUIDANCE_STRENGTHS:-1,2,5}"

METHOD="ordinary_mlp_center_logwidth"
GUIDE_CKPT="${MLP_CENTER_WIDTH_GUIDE_DIR}/guide_4000.pt"

echo "Bundled fixed-query sampling sweeps"
echo "Reverse steps at lambda=1 : ${REVERSE_STEPS:-50,75,100,150,200}"
echo "Lambda values at 50 steps : ${GUIDANCE_STRENGTHS:-1,2,5}"
nvidia-smi

if [ -n "${MISSING_TASKS_CSV:-}" ]; then
    echo "Running missing standard step-4000 samples in this allocation"
    bash sample_fixed_query_bound_token_checkpoints.sh
fi

if [ ! -f "${GUIDE_CKPT}" ]; then
    echo "ERROR: missing step-4000 checkpoint: ${GUIDE_CKPT}"
    exit 1
fi
METHOD_DIR="${SWEEP_SAMPLE_ROOT}/${METHOD}"
mkdir -p "${METHOD_DIR}"

for NUM_STEPS in "${REVERSE_STEP_VALUES[@]}"; do
    OUTPUT="${METHOD_DIR}/steps_$(printf '%03d' "${NUM_STEPS}")_lambda_1.csv"
    if [ -f "${OUTPUT}" ]; then
        echo "Reusing ${OUTPUT}"
        continue
    fi
    echo "Sampling ${METHOD}: steps=${NUM_STEPS}, lambda=1"
    python -u sample_doob_query.py \
        --guide-ckpt "${GUIDE_CKPT}" \
        --base-ckpt "${BASE_CANDIDATES[0]}" \
        --query-file "${QUERY_DIR}/${QUERY_ID}.json" \
        --num-samples "${NUM_SAMPLES:-1000}" \
        --batch-size "${NUM_SAMPLES:-1000}" \
        --num-timesteps "${NUM_STEPS}" \
        --guidance-strength 1 \
        --seed 74000 \
        --output "${OUTPUT}" \
        --device cuda
done

for LAMBDA in "${LAMBDA_VALUES[@]}"; do
    LAMBDA_TAG=$(printf '%g' "${LAMBDA}")
    OUTPUT="${METHOD_DIR}/steps_050_lambda_${LAMBDA_TAG}.csv"
    if [ -f "${OUTPUT}" ]; then
        echo "Reusing ${OUTPUT}"
        continue
    fi
    echo "Sampling ${METHOD}: steps=50, lambda=${LAMBDA}"
    python -u sample_doob_query.py \
        --guide-ckpt "${GUIDE_CKPT}" \
        --base-ckpt "${BASE_CANDIDATES[0]}" \
        --query-file "${QUERY_DIR}/${QUERY_ID}.json" \
        --num-samples "${NUM_SAMPLES:-1000}" \
        --batch-size "${NUM_SAMPLES:-1000}" \
        --num-timesteps 50 \
        --guidance-strength "${LAMBDA}" \
        --seed 74000 \
        --output "${OUTPUT}" \
        --device cuda
done

echo "Finished bundled reverse-step and guidance-strength sweeps"

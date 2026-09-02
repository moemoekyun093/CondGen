#!/bin/bash
#SBATCH --job-name=native_baseline_train
#SBATCH --output=logs/baselines/%x_%j.out
#SBATCH --error=logs/baselines/%x_%j.err
#SBATCH --gres=min-vram:16g,min-cuda-cc:70
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00

set -euo pipefail
TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-/scratch/work/agrawaa4/TabDiff}"
cd "${TABDIFF_PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

METHOD="${METHOD:?set METHOD to diffputer or great}"
DATANAME="${DATANAME:?set DATANAME}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
if [ "${METHOD}" = "great" ]; then
    PYTHON_BIN="${GREAT_PYTHON:-/scratch/work/agrawaa4/conda_envs/relgdiff/bin/python}"
elif [ "${METHOD}" = "diffputer" ]; then
    PYTHON_BIN="${DIFFPUTER_PYTHON:-/scratch/work/agrawaa4/conda_envs/tabdiff/bin/python}"
else
    echo "ERROR: METHOD must be diffputer or great"
    exit 1
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python not found: ${PYTHON_BIN}"
    exit 1
fi

args=(
    --method "${METHOD}"
    --dataname "${DATANAME}"
    --train-data "${TRAIN_DATA:-data/${DATANAME}/train.csv}"
    --test-data "${TEST_DATA:-data/${DATANAME}/test.csv}"
    --info-file "${INFO_FILE:-data/${DATANAME}/info.json}"
    --output-dir "${OUTPUT_DIR}"
    --harpoon-root "${HARPOON_ROOT:-baselines/harpoon}"
    --great-root "${GREAT_ROOT:-baselines/great}"
    --seed "${TRAIN_SEED:-42}"
    --device cuda
)
[ -n "${EPOCHS:-}" ] && args+=(--epochs "${EPOCHS}")
[ -n "${TRAIN_BATCH_SIZE:-}" ] && args+=(--batch-size "${TRAIN_BATCH_SIZE}")
[ -n "${LEARNING_RATE:-}" ] && args+=(--learning-rate "${LEARNING_RATE}")
[ -n "${HID_DIM:-}" ] && args+=(--hid-dim "${HID_DIM}")
[ -n "${DIFFUSION_TIMESTEPS:-}" ] && args+=(--timesteps "${DIFFUSION_TIMESTEPS}")
[ -n "${GREAT_LLM:-}" ] && args+=(--llm "${GREAT_LLM}")

mkdir -p logs/baselines "${OUTPUT_DIR}"
echo "Method      : ${METHOD}"
echo "Dataset     : ${DATANAME}"
echo "Output      : ${OUTPUT_DIR}"
echo "Python      : ${PYTHON_BIN}"
nvidia-smi
"${PYTHON_BIN}" -u train_native_query_baseline.py "${args[@]}"

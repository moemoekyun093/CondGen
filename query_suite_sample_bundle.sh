#!/bin/bash
# Sample one method over a query split, processing many query/seed pairs per job.
# This deliberately bundles work to avoid one short Slurm job per query.
#SBATCH --job-name=query_sample_bundle
#SBATCH --output=logs/query_suite/%x_%A_%a.out
#SBATCH --error=logs/query_suite/%x_%A_%a.err
#SBATCH --gres=min-vram:16g,min-cuda-cc:70
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00

set -euo pipefail

TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-/scratch/work/agrawaa4/TabDiff}"
cd "${TABDIFF_PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

METHOD_KIND="${METHOD_KIND:?set METHOD_KIND to doob, harpoon, diffputer, or great}"
METHOD_LABEL="${METHOD_LABEL:?set METHOD_LABEL}"
DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:?set QUERY_DIR}"
QUERY_SPLIT_MANIFEST="${QUERY_SPLIT_MANIFEST:-}"
QUERY_SPLIT="${QUERY_SPLIT:-test}"
SAMPLE_ROOT="${SAMPLE_ROOT:?set SAMPLE_ROOT}"
SEED_BASES="${SEED_BASES:-10000,20000,30000,40000,50000}"
BUNDLE_COUNT="${BUNDLE_COUNT:?set BUNDLE_COUNT}"
BUNDLE_INDEX="${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"

if [[ ! "${METHOD_KIND}" =~ ^(doob|harpoon|diffputer|great)$ ]]; then
    echo "ERROR: METHOD_KIND must be doob, harpoon, diffputer, or great"
    exit 1
fi
if [[ ! "${METHOD_LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: unsafe method label: ${METHOD_LABEL}"
    exit 1
fi

query_filter_args=()
if [ -n "${QUERY_SPLIT_MANIFEST}" ]; then
    query_filter_args+=(
        --query-split-manifest "${QUERY_SPLIT_MANIFEST}"
        --query-split "${QUERY_SPLIT}"
    )
fi
if [ "${QUERY_TEST_SUPPORTED_ONLY:-0}" = "1" ]; then
    query_filter_args+=(--test-supported-only)
fi
mapfile -t QUERY_FILES < <(
    python list_accepted_queries.py "${QUERY_DIR}" "${query_filter_args[@]}"
)
IFS=',' read -r -a BASE_SEEDS <<< "${SEED_BASES}"
NUM_QUERIES="${#QUERY_FILES[@]}"
NUM_SEEDS="${#BASE_SEEDS[@]}"
TOTAL_UNITS=$((NUM_QUERIES * NUM_SEEDS))

if [ "${NUM_QUERIES}" -eq 0 ] || [ "${NUM_SEEDS}" -eq 0 ]; then
    echo "ERROR: no query/seed work was selected"
    exit 1
fi

mkdir -p logs/query_suite "${SAMPLE_ROOT}/${METHOD_LABEL}"
echo "Method       : ${METHOD_KIND}/${METHOD_LABEL}"
echo "Queries      : ${NUM_QUERIES} (${QUERY_SPLIT} split)"
echo "Seed bases   : ${SEED_BASES}"
echo "Bundle       : $((BUNDLE_INDEX + 1))/${BUNDLE_COUNT}"
echo "Sample root  : ${SAMPLE_ROOT}/${METHOD_LABEL}"
nvidia-smi

sampled=0
reused=0
for ((unit = BUNDLE_INDEX; unit < TOTAL_UNITS; unit += BUNDLE_COUNT)); do
    seed_index=$((unit / NUM_QUERIES))
    query_index=$((unit % NUM_QUERIES))
    seed_base="${BASE_SEEDS[seed_index]}"
    if [[ ! "${seed_base}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid seed base: ${seed_base}"
        exit 1
    fi

    query_file="${QUERY_FILES[query_index]}"
    query_id="$(basename "${query_file}" .json)"
    # Preserve the historical direct layout for seed 1 so previous samples are
    # reusable. Additional replicates live in explicit seed directories.
    if [ "${seed_index}" -eq 0 ]; then
        output="${SAMPLE_ROOT}/${METHOD_LABEL}/${query_id}.csv"
    else
        output="${SAMPLE_ROOT}/${METHOD_LABEL}/seed_${seed_base}/${query_id}.csv"
    fi
    if [ -f "${output}" ]; then
        echo "Reuse: ${output}"
        reused=$((reused + 1))
        continue
    fi
    mkdir -p "$(dirname "${output}")"
    sample_seed=$((seed_base + query_index))
    echo "Sample: query=${query_id} seed=${sample_seed} -> ${output}"

    if [ "${METHOD_KIND}" = "doob" ]; then
        guide_checkpoint="${DOOB_GUIDE_DIR:?set DOOB_GUIDE_DIR}/best_guide.pt"
        base_checkpoint="${BASE_CHECKPOINT:?set BASE_CHECKPOINT}"
        python -u sample_doob_query.py \
            --guide-ckpt "${guide_checkpoint}" \
            --base-ckpt "${base_checkpoint}" \
            --query-file "${query_file}" \
            --num-samples "${NUM_SAMPLES}" \
            --batch-size "${BATCH_SIZE}" \
            --num-timesteps "${NUM_TIMESTEPS}" \
            --seed "${sample_seed}" \
            --output "${output}" \
            --device cuda
    elif [ "${METHOD_KIND}" = "harpoon" ]; then
        python -u sample_harpoon_full_query.py \
            --dataname "${DATANAME}" \
            --query-file "${query_file}" \
            --allow-partial-query \
            --harpoon-root "${HARPOON_ROOT:-baselines/harpoon}" \
            --runtime-root "${RUNTIME_ROOT:-/scratch/work/agrawaa4/harpoon_runtime}" \
            --checkpoint "${HARPOON_CHECKPOINT:?set HARPOON_CHECKPOINT}" \
            --output "${output}" \
            --num-samples "${NUM_SAMPLES}" \
            --batch-size "${BATCH_SIZE}" \
            --guidance-scale "${HARPOON_GUIDANCE_SCALE:-0.2}" \
            --seed "${sample_seed}" \
            --device cuda
    else
        if [ "${METHOD_KIND}" = "great" ]; then
            BASELINE_PYTHON="${GREAT_PYTHON:-/scratch/work/agrawaa4/conda_envs/great/bin/python}"
        else
            BASELINE_PYTHON="${DIFFPUTER_PYTHON:-/scratch/work/agrawaa4/conda_envs/tabdiff/bin/python}"
        fi
        if [ ! -x "${BASELINE_PYTHON}" ]; then
            echo "ERROR: baseline Python not found: ${BASELINE_PYTHON}"
            exit 1
        fi
        "${BASELINE_PYTHON}" -u sample_native_query_baseline.py \
            --method "${METHOD_KIND}" \
            --model-path "${BASELINE_MODEL_PATH:?set BASELINE_MODEL_PATH}" \
            --query-file "${query_file}" \
            --dataname "${DATANAME}" \
            --train-data "${TRAIN_DATA:-data/${DATANAME}/train.csv}" \
            --test-data "${TEST_DATA:-data/${DATANAME}/test.csv}" \
            --info-file "${INFO_FILE:-data/${DATANAME}/info.json}" \
            --harpoon-root "${HARPOON_ROOT:-baselines/harpoon}" \
            --great-root "${GREAT_ROOT:-baselines/great}" \
            --num-samples "${NUM_SAMPLES}" \
            --batch-size "${BATCH_SIZE}" \
            --great-max-length "${GREAT_MAX_LENGTH:-512}" \
            --seed "${sample_seed}" \
            --output "${output}" \
            --device cuda
    fi
    sampled=$((sampled + 1))
done

echo "Bundle complete: sampled=${sampled}, reused=${reused}"

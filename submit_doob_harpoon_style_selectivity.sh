#!/bin/bash
# Reuse complete samples, submit only missing work, then compare both methods.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
DOOB_GUIDE_DIR="${DOOB_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_query_tight_curriculum_d48_l2_18000}"
DOOB_LABEL="${DOOB_LABEL:-doob_tight_curriculum_s50}"
HARPOON_STYLE_LABEL="${HARPOON_STYLE_LABEL:-harpoon_style_tabdiff_eta02_s50}"
HARPOON_STYLE_ETA="${HARPOON_STYLE_ETA:-0.2}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
SEED_BASE="${SEED_BASE:-10000}"

# These can point at any existing compatible sample directories. Keep the tight
# curriculum outputs separate so older curriculum samples cannot be reused by
# mistake merely because they have matching query filenames.
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/doob_vs_harpoon_style_s50}"
DOOB_SAMPLE_DIR="${DOOB_SAMPLE_DIR:-${SUITE_SAMPLE_ROOT}/${DOOB_LABEL}}"
HARPOON_STYLE_SAMPLE_DIR="${HARPOON_STYLE_SAMPLE_DIR:-${SUITE_SAMPLE_ROOT}/${HARPOON_STYLE_LABEL}}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/doob_vs_harpoon_style_s50}"

if [ ! -d "${QUERY_DIR}" ]; then
    echo "ERROR: query directory not found: ${QUERY_DIR}"
    exit 1
fi
mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}")
NUM_QUERIES="${#QUERY_FILES[@]}"
if [ "${NUM_QUERIES}" -le 0 ]; then
    echo "ERROR: no accepted queries found"
    exit 1
fi

DOOB_OUTPUT_LABEL="$(basename "${DOOB_SAMPLE_DIR}")"
DOOB_SAMPLE_ROOT="$(dirname "${DOOB_SAMPLE_DIR}")"
HARPOON_OUTPUT_LABEL="$(basename "${HARPOON_STYLE_SAMPLE_DIR}")"
HARPOON_SAMPLE_ROOT="$(dirname "${HARPOON_STYLE_SAMPLE_DIR}")"
for LABEL in "${DOOB_OUTPUT_LABEL}" "${HARPOON_OUTPUT_LABEL}"; do
    if [[ ! "${LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "ERROR: unsafe sample-directory basename: ${LABEL}"
        exit 1
    fi
done

mkdir -p logs harpoon_logs evaluations/slurm \
    "${DOOB_SAMPLE_DIR}" "${HARPOON_STYLE_SAMPLE_DIR}" "${OUTPUT_DIR}"

# A result is complete only when both the generated table and its raw-space
# constraint report exist. Build exact Slurm array lists for missing queries.
DOOB_MISSING=()
HARPOON_MISSING=()
for INDEX in "${!QUERY_FILES[@]}"; do
    QUERY_ID="$(basename "${QUERY_FILES[INDEX]}" .json)"
    DOOB_CSV="${DOOB_SAMPLE_DIR}/${QUERY_ID}.csv"
    HARPOON_CSV="${HARPOON_STYLE_SAMPLE_DIR}/${QUERY_ID}.csv"
    if [ ! -f "${DOOB_CSV}" ] || [ ! -f "${DOOB_CSV%.csv}.constraints.json" ]; then
        DOOB_MISSING+=("${INDEX}")
    fi
    if [ ! -f "${HARPOON_CSV}" ] || [ ! -f "${HARPOON_CSV%.csv}.constraints.json" ]; then
        HARPOON_MISSING+=("${INDEX}")
    fi
done

JOB_IDS=()
if [ "${#DOOB_MISSING[@]}" -eq 0 ]; then
    DOOB_JOB="reused all existing samples"
else
    if [ ! -f "${DOOB_GUIDE_DIR}/best_guide.pt" ]; then
        echo "ERROR: Doob guide is required for missing samples: ${DOOB_GUIDE_DIR}/best_guide.pt"
        exit 1
    fi
    DOOB_ARRAY="$(IFS=,; echo "${DOOB_MISSING[*]}")%${MAX_CONCURRENT}"
    DOOB_SUBMISSION=$(DATANAME="${DATANAME}" MODEL_NAME="${MODEL_NAME}" \
        QUERY_DIR="${QUERY_DIR}" GUIDE_SPECS="${DOOB_OUTPUT_LABEL}=${DOOB_GUIDE_DIR}" \
        SUITE_SAMPLE_ROOT="${DOOB_SAMPLE_ROOT}" NUM_SAMPLES="${NUM_SAMPLES}" \
        NUM_TIMESTEPS="${NUM_TIMESTEPS}" SEED_BASE="${SEED_BASE}" \
        sbatch --parsable --array="${DOOB_ARRAY}" doob_query_suite_sample.sh)
    DOOB_JOB="${DOOB_SUBMISSION%%;*}"
    JOB_IDS+=("${DOOB_JOB}")
fi

if [ "${#HARPOON_MISSING[@]}" -eq 0 ]; then
    HARPOON_JOB="reused all existing samples"
else
    HARPOON_ARRAY="$(IFS=,; echo "${HARPOON_MISSING[*]}")%${MAX_CONCURRENT}"
    HARPOON_SUBMISSION=$(DATANAME="${DATANAME}" MODEL_NAME="${MODEL_NAME}" \
        QUERY_DIR="${QUERY_DIR}" SUITE_SAMPLE_ROOT="${HARPOON_SAMPLE_ROOT}" \
        HARPOON_STYLE_LABEL="${HARPOON_OUTPUT_LABEL}" \
        HARPOON_STYLE_ETA="${HARPOON_STYLE_ETA}" NUM_TIMESTEPS="${NUM_TIMESTEPS}" \
        NUM_SAMPLES="${NUM_SAMPLES}" SEED_BASE="${SEED_BASE}" \
        sbatch --parsable --array="${HARPOON_ARRAY}" \
        harpoon_style_tabdiff_query_suite_sample.sh)
    HARPOON_JOB="${HARPOON_SUBMISSION%%;*}"
    JOB_IDS+=("${HARPOON_JOB}")
fi

METHOD_SPECS="${DOOB_LABEL}=${DOOB_SAMPLE_DIR},${HARPOON_STYLE_LABEL}=${HARPOON_STYLE_SAMPLE_DIR}"
EVAL_SBATCH_ARGS=(--parsable)
if [ "${#JOB_IDS[@]}" -gt 0 ]; then
    DEPENDENCY="afterok"
    for JOB_ID in "${JOB_IDS[@]}"; do
        DEPENDENCY+=":${JOB_ID}"
    done
    EVAL_SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
else
    DEPENDENCY="all samples complete; evaluation starts immediately"
fi
EVAL_SUBMISSION=$(DATANAME="${DATANAME}" QUERY_DIR="${QUERY_DIR}" \
    METHOD_SPECS="${METHOD_SPECS}" BASELINE_LABEL="${DOOB_LABEL}" \
    OUTPUT_DIR="${OUTPUT_DIR}" EVAL_GROUP_BY=target_band \
    sbatch "${EVAL_SBATCH_ARGS[@]}" fixed_query_model_list_evaluate.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "========================================"
echo "Doob vs HARPOON-style TabDiff submitted"
echo "Queries          : ${NUM_QUERIES}"
echo "Doob preexisting : $((NUM_QUERIES - ${#DOOB_MISSING[@]}))/${NUM_QUERIES}"
echo "HARPOON existing : $((NUM_QUERIES - ${#HARPOON_MISSING[@]}))/${NUM_QUERIES}"
echo "Reverse steps    : ${NUM_TIMESTEPS}"
echo "Doob array       : ${DOOB_JOB}"
echo "HARPOON array    : ${HARPOON_JOB}"
echo "Evaluation       : ${EVAL_JOB} (${DEPENDENCY})"
echo "Output           : ${OUTPUT_DIR}"
echo "========================================"

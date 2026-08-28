#!/bin/bash
# Run any list of compatible models on the same nested legacy fixed query.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
FIXED_QUERY="${FIXED_QUERY:-constraints/${DATANAME}/fixed_numerical_intervals.json}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_fixed_box_permuted}"
SPLIT_ROOT="${SPLIT_ROOT:-data90/${DATANAME}}"
SAMPLE_ROOT="${SAMPLE_ROOT:-conditional_samples/${DATANAME}/fixed_box_permuted_model_comparison}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/fixed_box_permuted_model_comparison}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
SEED_BASE="${SEED_BASE:-73010}"
NUM_ORDERINGS="${NUM_ORDERINGS:-12}"
ORDERING_SEED="${ORDERING_SEED:-7301}"

# List format:
# "LABEL|TYPE|CHECKPOINT_OR_GUIDE_DIR|OPTIONAL_GUIDANCE_SCALE|OLD_SAMPLE_GLOBS"
# Separate multiple old globs with a semicolon. Matching uses query contents.
# Supported TYPE values: legacy_doob, structured_doob, harpoon.
# Add, remove, or replace entries here; labels must be unique and shell-safe.
MODELS=(
    "old_h|legacy_doob|tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_h_partial_masks_concat_d48_l2_h4_f2_lr1e3_6000_candidate_logh||conditional_samples/${DATANAME}/${MODEL_NAME}_constraint_sweep_d48_l2_lr1e3_6000_k*.csv;conditional_samples/${DATANAME}/fixed_box_model_comparison/old_h/*.csv"
    "new_h_constraints|structured_doob|tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_query_tight_curriculum_d48_l2_18000||conditional_samples/${DATANAME}/fixed_box_model_comparison/new_h_constraints/*.csv"
    "harpoon_eta02|harpoon|/scratch/work/agrawaa4/harpoon_runtime/saved_models/${DATANAME}/diffputer_selfmade.pt|0.2|conditional_samples/${DATANAME}/harpoon_constraint_sweep_summed_relu_eta02_k*.csv;conditional_samples/${DATANAME}/fixed_box_model_comparison/harpoon_eta02/*.csv"
)

if [ "${#MODELS[@]}" -lt 2 ]; then
    echo "ERROR: MODELS must contain at least two entries"
    exit 1
fi
if [ ! -f "${FIXED_QUERY}" ]; then
    echo "ERROR: fixed query not found: ${FIXED_QUERY}"
    exit 1
fi
mkdir -p logs harpoon_logs evaluations/slurm "${SAMPLE_ROOT}"

# Materialize one canonical structured query per numerical prefix. Every model
# is evaluated against these exact JSON predicates and conditional references.
python generate_fixed_box_query_suite.py \
    --fixed-query "${FIXED_QUERY}" \
    --output-query-dir "${QUERY_DIR}" \
    --split-root "${SPLIT_ROOT}" \
    --real-data "synthetic/${DATANAME}/real.csv" \
    --num-orderings "${NUM_ORDERINGS}" \
    --seed "${ORDERING_SEED}"

mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}")
NUM_QUERIES="${#QUERY_FILES[@]}"
EXPECTED_QUERIES=$((NUM_ORDERINGS * 10))
ORDERINGS_FILE="${QUERY_DIR}/orderings.txt"
if [ "${NUM_QUERIES}" -ne "${EXPECTED_QUERIES}" ]; then
    echo "ERROR: expected ${EXPECTED_QUERIES} permuted fixed-box queries, found ${NUM_QUERIES}"
    exit 1
fi
if [ "$(wc -l < "${ORDERINGS_FILE}")" -ne "${NUM_ORDERINGS}" ]; then
    echo "ERROR: ordering manifest does not contain ${NUM_ORDERINGS} entries"
    exit 1
fi

declare -A SEEN_LABELS=()
JOB_IDS=()
METHODS=()
for SPEC in "${MODELS[@]}"; do
    IFS='|' read -r LABEL TYPE CHECKPOINT EXTRA OLD_SAMPLE_GLOBS <<< "${SPEC}"
    if [[ ! "${LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "ERROR: unsafe or empty model label: ${LABEL}"
        exit 1
    fi
    if [ -n "${SEEN_LABELS[$LABEL]:-}" ]; then
        echo "ERROR: duplicate model label: ${LABEL}"
        exit 1
    fi
    SEEN_LABELS[$LABEL]=1
    SAMPLE_DIR="${SAMPLE_ROOT}/${LABEL}"
    mkdir -p "${SAMPLE_DIR}"

    CACHE_ARGS=(
        --query-dir "${QUERY_DIR}"
        --target-dir "${SAMPLE_DIR}"
    )
    if [ -n "${OLD_SAMPLE_GLOBS:-}" ]; then
        IFS=';' read -r -a SOURCE_GLOBS <<< "${OLD_SAMPLE_GLOBS}"
        for SOURCE_GLOB in "${SOURCE_GLOBS[@]}"; do
            CACHE_ARGS+=(--source-glob "${SOURCE_GLOB}")
        done
    fi
    python reuse_matching_query_samples.py "${CACHE_ARGS[@]}"
    METHODS+=("${LABEL}=${SAMPLE_DIR}")

    COMPLETE_COUNT=0
    for QUERY_FILE in "${QUERY_FILES[@]}"; do
        QUERY_ID="$(basename "${QUERY_FILE}" .json)"
        SAMPLE_FILE="${SAMPLE_DIR}/${QUERY_ID}.csv"
        if [ -f "${SAMPLE_FILE}" ] && [ -f "${SAMPLE_FILE%.csv}.constraints.json" ]; then
            COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
        fi
    done
    if [ "${COMPLETE_COUNT}" -eq "${NUM_QUERIES}" ]; then
        echo "Reusing all ${NUM_QUERIES} completed samples for ${LABEL}; no sampling job submitted"
        continue
    fi
    echo "${LABEL}: ${COMPLETE_COUNT}/${NUM_QUERIES} samples cached; submitting only missing work"

    case "${TYPE}" in
        legacy_doob)
            if [ ! -f "${CHECKPOINT}/best_guide.pt" ]; then
                echo "ERROR: legacy guide not found: ${CHECKPOINT}/best_guide.pt"
                exit 1
            fi
            SUBMISSION=$(DATANAME="${DATANAME}" FT_MODEL="${MODEL_NAME}" \
                OUTPUT_ROOT="${SAMPLE_DIR}" FIXED_QUERY_PREFIX=qf_fixed_box \
                NUM_SAMPLES="${NUM_SAMPLES}" SEED_BASE="${SEED_BASE}" \
                ORDERINGS_FILE="${ORDERINGS_FILE}" \
                sbatch --parsable --array="0-$((NUM_QUERIES - 1))%${MAX_CONCURRENT}" \
                doob_h_constraint_sweep.sh "${CHECKPOINT}" "${LABEL}")
            ;;
        structured_doob)
            if [ ! -f "${CHECKPOINT}/best_guide.pt" ]; then
                echo "ERROR: structured guide not found: ${CHECKPOINT}/best_guide.pt"
                exit 1
            fi
            SUBMISSION=$(DATANAME="${DATANAME}" MODEL_NAME="${MODEL_NAME}" \
                QUERY_DIR="${QUERY_DIR}" GUIDE_SPECS="${LABEL}=${CHECKPOINT}" \
                SUITE_SAMPLE_ROOT="${SAMPLE_ROOT}" NUM_SAMPLES="${NUM_SAMPLES}" \
                SEED_BASE="${SEED_BASE}" SEED_BY_ARITY=1 \
                sbatch --parsable --array="0-$((NUM_QUERIES - 1))%${MAX_CONCURRENT}" \
                doob_query_suite_sample.sh)
            ;;
        harpoon)
            if [ ! -f "${CHECKPOINT}" ]; then
                echo "ERROR: HARPOON checkpoint not found: ${CHECKPOINT}"
                exit 1
            fi
            GUIDANCE_SCALE="${EXTRA:-0.2}"
            SUBMISSION=$(DATANAME="${DATANAME}" QUERY_FILE="${FIXED_QUERY}" \
                CHECKPOINT="${CHECKPOINT}" OUTPUT_ROOT="${SAMPLE_DIR}" \
                FIXED_QUERY_PREFIX=qf_fixed_box NUM_SAMPLES="${NUM_SAMPLES}" \
                SEED_BASE="${SEED_BASE}" GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
                ORDERINGS_FILE="${ORDERINGS_FILE}" \
                sbatch --parsable --array="0-$((NUM_QUERIES - 1))%${MAX_CONCURRENT}" \
                harpoon_constraint_sweep.sh "${LABEL}")
            ;;
        *)
            echo "ERROR: unsupported model type '${TYPE}' in ${SPEC}"
            exit 1
            ;;
    esac
    JOB_ID="${SUBMISSION%%;*}"
    JOB_IDS+=("${JOB_ID}")
    echo "Submitted ${LABEL} (${TYPE}): ${JOB_ID}"
done

METHOD_SPECS=$(IFS=','; echo "${METHODS[*]}")
BASELINE_LABEL="${MODELS[0]%%|*}"
if [ "${#JOB_IDS[@]}" -gt 0 ]; then
    DEPENDENCY="afterok"
    for JOB_ID in "${JOB_IDS[@]}"; do
        DEPENDENCY+=":${JOB_ID}"
    done
    EVAL_SUBMISSION=$(DATANAME="${DATANAME}" QUERY_DIR="${QUERY_DIR}" \
        METHOD_SPECS="${METHOD_SPECS}" BASELINE_LABEL="${BASELINE_LABEL}" \
        OUTPUT_DIR="${OUTPUT_DIR}" \
        sbatch --parsable --dependency="${DEPENDENCY}" \
        fixed_query_model_list_evaluate.sh)
else
    DEPENDENCY="all samples reused; no dependency"
    EVAL_SUBMISSION=$(DATANAME="${DATANAME}" QUERY_DIR="${QUERY_DIR}" \
        METHOD_SPECS="${METHOD_SPECS}" BASELINE_LABEL="${BASELINE_LABEL}" \
        OUTPUT_DIR="${OUTPUT_DIR}" \
        sbatch --parsable fixed_query_model_list_evaluate.sh)
fi
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "========================================"
echo "Fixed-query model-list comparison submitted"
echo "Models     : ${#MODELS[@]}"
echo "Orderings  : ${NUM_ORDERINGS} (identity, reverse, seeded permutations)"
echo "Queries    : ${NUM_QUERIES} nested numerical prefixes"
echo "Evaluation : ${EVAL_JOB} (${DEPENDENCY})"
echo "Output     : ${OUTPUT_DIR}"
echo "========================================"

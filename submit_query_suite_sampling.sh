#!/bin/bash
# Submit any number of model-backed methods over a query split.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_ROOT}"

DATANAME=""
QUERY_DIR=""
QUERY_SPLIT_MANIFEST=""
QUERY_SPLIT="test"
SAMPLE_ROOT=""
NUM_SEEDS=5
SEED_BASES=""
MAX_BUNDLES=4
TARGET_MISSING_PER_BUNDLE=50
NUM_SAMPLES=1000
NUM_TIMESTEPS=50
EVALUATION_OUTPUT=""
RUN_SYNTHCITY=1
TRAIN_DEPENDENCY=""
METHODS=()

usage() {
    cat <<'EOF'
Usage: bash submit_query_suite_sampling.sh --dataname NAME --query-dir DIR \
  --query-split-manifest FILE --method LABEL=KIND:MODEL_PATH [--method ...]

KIND is doob, harpoon, diffputer, or great. MODEL_PATH is a guide directory,
HARPOON checkpoint, DiffPuter adapter directory/checkpoint, or GReaT directory.

Options:
  --query-split train|test   Default: test
  --sample-root DIR          Default: conditional_samples/NAME/modular_suite
  --num-seeds N              Default: 5 (seed bases 10000,20000,...)
  --seed-bases CSV           Explicit seeds; overrides --num-seeds
  --max-bundles N            Long GPU jobs per method, default 4
  --num-samples N            Rows per query/seed, default 1000
  --num-timesteps N          Doob reverse steps, default 50
  --base-checkpoint FILE     Required when any KIND is doob
  --train-data FILE          Baseline training CSV (default data/NAME/train.csv)
  --test-data FILE           Baseline test CSV (default data/NAME/test.csv)
  --info-file FILE           Dataset metadata (default data/NAME/info.json)
  --evaluation-output DIR    Also submit evaluation after all missing samples
  --skip-synthcity           With --evaluation-output, omit Alpha/Beta
  --dependency JOB[:JOB...]  Wait for model training before sampling
EOF
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataname) DATANAME="$2"; shift 2 ;;
        --query-dir) QUERY_DIR="$2"; shift 2 ;;
        --query-split-manifest) QUERY_SPLIT_MANIFEST="$2"; shift 2 ;;
        --query-split) QUERY_SPLIT="$2"; shift 2 ;;
        --sample-root) SAMPLE_ROOT="$2"; shift 2 ;;
        --method) METHODS+=("$2"); shift 2 ;;
        --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --seed-bases) SEED_BASES="$2"; shift 2 ;;
        --max-bundles) MAX_BUNDLES="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --num-timesteps) NUM_TIMESTEPS="$2"; shift 2 ;;
        --base-checkpoint) BASE_CHECKPOINT="$2"; export BASE_CHECKPOINT; shift 2 ;;
        --train-data) TRAIN_DATA="$2"; export TRAIN_DATA; shift 2 ;;
        --test-data) TEST_DATA="$2"; export TEST_DATA; shift 2 ;;
        --info-file) INFO_FILE="$2"; export INFO_FILE; shift 2 ;;
        --evaluation-output) EVALUATION_OUTPUT="$2"; shift 2 ;;
        --skip-synthcity) RUN_SYNTHCITY=0; shift ;;
        --dependency) TRAIN_DEPENDENCY="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1"; usage; exit 1 ;;
    esac
done
if [ -z "${DATANAME}" ] || [ -z "${QUERY_DIR}" ] || \
   [ -z "${QUERY_SPLIT_MANIFEST}" ] || [ "${#METHODS[@]}" -eq 0 ]; then
    usage
    exit 1
fi
if ! [[ "${NUM_SEEDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --num-seeds must be a positive integer"
    exit 1
fi
if [ -z "${SEED_BASES}" ]; then
    SEED_BASES=""
    for ((index=1; index<=NUM_SEEDS; index++)); do
        [ -n "${SEED_BASES}" ] && SEED_BASES+="," 
        SEED_BASES+="$((index * 10000))"
    done
fi
SAMPLE_ROOT="${SAMPLE_ROOT:-conditional_samples/${DATANAME}/modular_suite}"
mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}" \
    --query-split-manifest "${QUERY_SPLIT_MANIFEST}" --query-split "${QUERY_SPLIT}")
IFS=',' read -r -a seeds <<< "${SEED_BASES}"
mkdir -p logs/query_suite "${SAMPLE_ROOT}"
JOB_IDS=()
declare -A SEEN_LABELS=()
for spec in "${METHODS[@]}"; do
    label="${spec%%=*}"
    remainder="${spec#*=}"
    kind="${remainder%%:*}"
    model_path="${remainder#*:}"
    if [ "${label}" = "${spec}" ] || [ "${kind}" = "${remainder}" ] || \
       [[ ! "${label}" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ "${model_path}" == *"|"* ]] || \
       [[ ! "${kind}" =~ ^(doob|harpoon|diffputer|great)$ ]]; then
        echo "ERROR: invalid --method ${spec}; expected LABEL=KIND:MODEL_PATH"
        exit 1
    fi
    if [ -n "${SEEN_LABELS[${label}]:-}" ]; then
        echo "ERROR: duplicate method label ${label}"
        exit 1
    fi
    SEEN_LABELS["${label}"]=1
    missing=0
    for query_file in "${QUERY_FILES[@]}"; do
        query_id="$(basename "${query_file}" .json)"
        for index in "${!seeds[@]}"; do
            if [ "${index}" -eq 0 ]; then
                output="${SAMPLE_ROOT}/${label}/${query_id}.csv"
            else
                output="${SAMPLE_ROOT}/${label}/seed_${seeds[index]}/${query_id}.csv"
            fi
            [ -f "${output}" ] || missing=$((missing + 1))
        done
    done
    if [ "${missing}" -eq 0 ]; then
        echo "Reuse ${label}: all ${#QUERY_FILES[@]} x ${#seeds[@]} samples exist"
        continue
    fi
    bundles=$(((missing + TARGET_MISSING_PER_BUNDLE - 1) / TARGET_MISSING_PER_BUNDLE))
    [ "${bundles}" -gt "${MAX_BUNDLES}" ] && bundles="${MAX_BUNDLES}"
    export METHOD_KIND="${kind}" METHOD_LABEL="${label}" BUNDLE_COUNT="${bundles}"
    export DATANAME QUERY_DIR QUERY_SPLIT_MANIFEST QUERY_SPLIT SAMPLE_ROOT SEED_BASES
    export NUM_SAMPLES NUM_TIMESTEPS
    unset DOOB_GUIDE_DIR HARPOON_CHECKPOINT BASELINE_MODEL_PATH || true
    case "${kind}" in
        doob)
            if [ -z "${BASE_CHECKPOINT:-}" ]; then
                echo "ERROR: --base-checkpoint is required for Doob sampling"
                exit 1
            fi
            export DOOB_GUIDE_DIR="${model_path}"
            ;;
        harpoon) export HARPOON_CHECKPOINT="${model_path}" ;;
        diffputer|great) export BASELINE_MODEL_PATH="${model_path}" ;;
    esac
    dependency_args=()
    [ -n "${TRAIN_DEPENDENCY}" ] && dependency_args+=(--dependency="afterok:${TRAIN_DEPENDENCY}")
    submission=$(sbatch --parsable "${dependency_args[@]}" \
        --array="0-$((bundles - 1))" query_suite_sample_bundle.sh)
    job_id="${submission%%;*}"
    JOB_IDS+=("${job_id}")
    echo "Submitted ${label} (${kind}): job ${job_id}; missing=${missing}; bundles=${bundles}"
done
echo "Sample root: ${SAMPLE_ROOT}"
if [ "${#JOB_IDS[@]}" -gt 0 ]; then
    echo "Sampling jobs: ${JOB_IDS[*]}"
else
    echo "Sampling jobs: none (everything reused)"
fi
if [ -n "${EVALUATION_OUTPUT}" ]; then
    evaluation_args=(
        --dataname "${DATANAME}"
        --query-dir "${QUERY_DIR}"
        --query-split-manifest "${QUERY_SPLIT_MANIFEST}"
        --query-split "${QUERY_SPLIT}"
        --seed-bases "${SEED_BASES}"
        --output-dir "${EVALUATION_OUTPUT}"
    )
    for spec in "${METHODS[@]}"; do
        label="${spec%%=*}"
        evaluation_args+=(--method "${label}=${SAMPLE_ROOT}/${label}")
    done
    if [ "${#JOB_IDS[@]}" -gt 0 ]; then
        dependency="$(IFS=:; echo "${JOB_IDS[*]}")"
        evaluation_args+=(--dependency "${dependency}")
    fi
    [ "${RUN_SYNTHCITY}" = "0" ] && evaluation_args+=(--skip-synthcity)
    bash submit_query_suite_evaluation.sh "${evaluation_args[@]}"
fi

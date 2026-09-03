#!/bin/bash
# Evaluate already-generated samples for arbitrary methods and seed count.
set -euo pipefail
TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${TABDIFF_PROJECT_ROOT}"
export TABDIFF_PROJECT_ROOT

DATANAME=""
QUERY_DIR=""
QUERY_SPLIT_MANIFEST=""
QUERY_SPLIT="test"
OUTPUT_DIR=""
NUM_SEEDS=5
SEED_BASES=""
RUN_SYNTHCITY=1
GROUP_BY="target_band"
METHODS=()
DEPENDENCY=""
QUERY_TEST_SUPPORTED_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash submit_query_suite_evaluation.sh --dataname NAME --query-dir DIR \
  [--query-split-manifest FILE] --method LABEL=SAMPLE_DIRECTORY [--method ...]

Options:
  --query-split train|test    Default: test
  --query-split-manifest F    Optional query-definition split; omit for all queries
  --output-dir DIR            Required destination for CSVs and plots
  --num-seeds N               Default: 5
  --seed-bases CSV            Explicit bases; overrides --num-seeds
  --group-by FIELD            target_band, arity, or mean_interval_width
  --baseline-method LABEL     Optional paired-difference reference
  --dependency JOB[:JOB...]   Wait for sampling jobs
  --skip-synthcity            Do not run Alpha Precision/Beta Recall
  --real-data FILE            Default: synthetic/NAME/real.csv
  --info-file FILE            Default: data/NAME/info.json
  --query-coordinates FILE    Exact transformed interval coordinates
  --filtered-min-rows N       Default: 50
  --test-supported-only       Keep only query files marked test_supported
EOF
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataname) DATANAME="$2"; shift 2 ;;
        --query-dir) QUERY_DIR="$2"; shift 2 ;;
        --query-split-manifest) QUERY_SPLIT_MANIFEST="$2"; shift 2 ;;
        --query-split) QUERY_SPLIT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --method) METHODS+=("$2"); shift 2 ;;
        --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --seed-bases) SEED_BASES="$2"; shift 2 ;;
        --group-by) GROUP_BY="$2"; shift 2 ;;
        --query-coordinates) QUERY_COORDINATES="$2"; export QUERY_COORDINATES; shift 2 ;;
        --filtered-min-rows) FILTERED_MIN_ROWS="$2"; export FILTERED_MIN_ROWS; shift 2 ;;
        --test-supported-only) QUERY_TEST_SUPPORTED_ONLY=1; shift ;;
        --baseline-method) EVAL_BASELINE_METHOD="$2"; export EVAL_BASELINE_METHOD; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --skip-synthcity) RUN_SYNTHCITY=0; shift ;;
        --real-data) REAL_DATA="$2"; export REAL_DATA; shift 2 ;;
        --info-file) INFO_FILE="$2"; export INFO_FILE; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1"; usage; exit 1 ;;
    esac
done
if [ -z "${DATANAME}" ] || [ -z "${QUERY_DIR}" ] || [ -z "${OUTPUT_DIR}" ] || \
   [ "${#METHODS[@]}" -eq 0 ]; then
    usage
    exit 1
fi
if ! [[ "${NUM_SEEDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --num-seeds must be a positive integer"
    exit 1
fi
if [[ ! "${GROUP_BY}" =~ ^(target_band|arity|mean_interval_width)$ ]]; then
    echo "ERROR: invalid --group-by ${GROUP_BY}"
    exit 1
fi
if [ -z "${SEED_BASES}" ]; then
    for ((index=1; index<=NUM_SEEDS; index++)); do
        [ -n "${SEED_BASES}" ] && SEED_BASES+=","
        SEED_BASES+="$((index * 10000))"
    done
fi
METHOD_SPECS=""
declare -A SEEN_LABELS=()
for spec in "${METHODS[@]}"; do
    if [[ "${spec}" != *=* ]]; then echo "ERROR: invalid method ${spec}"; exit 1; fi
    label="${spec%%=*}"
    if [[ ! "${label}" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ "${spec}" == *"|"* ]]; then
        echo "ERROR: unsafe method specification ${spec}"
        exit 1
    fi
    if [ -n "${SEEN_LABELS[${label}]:-}" ]; then
        echo "ERROR: duplicate method label ${label}"
        exit 1
    fi
    SEEN_LABELS["${label}"]=1
    [ -n "${METHOD_SPECS}" ] && METHOD_SPECS+="|"
    METHOD_SPECS+="${spec}"
done
mkdir -p evaluations/slurm "${OUTPUT_DIR}"
export DATANAME QUERY_DIR QUERY_SPLIT_MANIFEST QUERY_SPLIT QUERY_TEST_SUPPORTED_ONLY SEED_BASES METHOD_SPECS
export SUITE_EVAL_DIR="${OUTPUT_DIR}" EVAL_GROUP_BY="${GROUP_BY}"
dep_args=()
[ -n "${DEPENDENCY}" ] && dep_args+=(--dependency="afterok:${DEPENDENCY}")
ALPHA_JOB="skipped"
if [ "${RUN_SYNTHCITY}" = "1" ]; then
    export ALPHA_BETA_RESULTS="${OUTPUT_DIR}/alpha_beta_per_query_seed.csv"
    submission=$(sbatch --parsable "${dep_args[@]}" query_suite_alpha_evaluate.sh)
    ALPHA_JOB="${submission%%;*}"
    dep_args=(--dependency="afterok:${ALPHA_JOB}")
else
    unset ALPHA_BETA_RESULTS || true
fi
submission=$(sbatch --parsable "${dep_args[@]}" query_suite_evaluate.sh)
EVAL_JOB="${submission%%;*}"
echo "Submitted modular evaluation"
echo "Methods       : ${METHOD_SPECS}"
echo "Query split   : ${QUERY_SPLIT}"
echo "Seeds         : ${SEED_BASES}"
echo "Alpha/Beta job: ${ALPHA_JOB}"
echo "Normal job    : ${EVAL_JOB}"
echo "Output        : ${OUTPUT_DIR}"

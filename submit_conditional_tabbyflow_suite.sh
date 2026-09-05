#!/bin/bash
# Submit bundled, training-free TabbyFlow conditioning and optional evaluation.
set -euo pipefail

TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${TABDIFF_PROJECT_ROOT}"

DATANAME="shoppers"
QUERY_DIR="data90/shoppers/queries"
QUERY_SPLIT_MANIFEST="data90/shoppers/query_splits/sampled_arity_stratified_80_20_seed42.json"
QUERY_SPLIT="test"
RUN_DIR=""
CHECKPOINT=""
TRANSFORM=""
SAMPLE_DIR="conditional_samples/shoppers/tabbyflow_conditional_unseen_test"
EVALUATION_DIR=""
NUM_SAMPLES=1000
NUM_SEEDS=5
SEED_BASES=""
BUNDLES=4
SOLVER="heun"
STEPS=50
RUN_SYNTHCITY=0
BASELINE_METHOD=""
EVALUATION_METHODS=()

usage() {
    cat <<'EOF'
Usage: bash submit_conditional_tabbyflow_suite.sh --run-dir DIR [options]

Options:
  --checkpoint FILE           Override DIR/checkpoints/best_ema.pt
  --transform-file FILE       Override DIR/num_transform.joblib
  --query-dir DIR             Default data90/shoppers/queries
  --query-split-manifest FILE Default held-out 80/20 query split
  --query-split train|test    Default test
  --sample-dir DIR            Conditional CSV destination
  --evaluation-dir DIR        Submit evaluation after sampling
  --evaluation-method L=DIR   Add existing samples to the comparison; repeatable
  --baseline-method LABEL     Paired-difference reference in evaluation
  --num-samples N             Rows per query and seed, default 1000
  --num-seeds N               Default 5
  --seed-bases CSV            Explicit seeds, overriding --num-seeds
  --bundles N                 Long array jobs, default 4
  --solver dopri5|euler|heun  Default dependency-free 50-step Heun
  --steps N                   Fixed steps for Euler/Heun, default 50
  --with-synthcity            Also submit Alpha Precision/Beta Recall
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataname) DATANAME="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --transform-file) TRANSFORM="$2"; shift 2 ;;
        --query-dir) QUERY_DIR="$2"; shift 2 ;;
        --query-split-manifest) QUERY_SPLIT_MANIFEST="$2"; shift 2 ;;
        --query-split) QUERY_SPLIT="$2"; shift 2 ;;
        --sample-dir) SAMPLE_DIR="$2"; shift 2 ;;
        --evaluation-dir) EVALUATION_DIR="$2"; shift 2 ;;
        --evaluation-method) EVALUATION_METHODS+=("$2"); shift 2 ;;
        --baseline-method) BASELINE_METHOD="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --seed-bases) SEED_BASES="$2"; shift 2 ;;
        --bundles) BUNDLES="$2"; shift 2 ;;
        --solver) SOLVER="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --with-synthcity) RUN_SYNTHCITY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1"; usage; exit 1 ;;
    esac
done

[ -n "${RUN_DIR}" ] || { usage; exit 1; }
[[ "${QUERY_SPLIT}" =~ ^(train|test)$ ]] || { echo "ERROR: bad query split"; exit 1; }
[[ "${SOLVER}" =~ ^(dopri5|euler|heun)$ ]] || { echo "ERROR: bad solver"; exit 1; }
for value in "${NUM_SAMPLES}" "${NUM_SEEDS}" "${BUNDLES}" "${STEPS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: positive integers required"; exit 1; }
done
if [ -z "${SEED_BASES}" ]; then
    for ((index=1; index<=NUM_SEEDS; index++)); do
        [ -n "${SEED_BASES}" ] && SEED_BASES+=","
        SEED_BASES+="$((index * 10000))"
    done
fi

TABBYFLOW_PYTHON="${TABBYFLOW_PYTHON:-$(command -v python)}"
"${TABBYFLOW_PYTHON}" -c 'import torch, pandas, sklearn, joblib' || {
    echo "ERROR: ${TABBYFLOW_PYTHON} lacks a required TabbyFlow dependency"
    echo "Set TABBYFLOW_PYTHON to the compatible environment's Python executable."
    exit 1
}
if [ "${SOLVER}" = "dopri5" ]; then
    "${TABBYFLOW_PYTHON}" -c 'import torchdiffeq' || {
        echo "ERROR: --solver dopri5 requires torchdiffeq"
        echo "Use the dependency-free default: --solver heun --steps 50"
        exit 1
    }
fi

DATA_DIR="${DATA_DIR:-data/${DATANAME}}"
INFO_FILE="${INFO_FILE:-data/${DATANAME}/info.json}"
mkdir -p logs/tabbyflow "${SAMPLE_DIR}" evaluations/slurm

export TABDIFF_PROJECT_ROOT TABBYFLOW_PYTHON
export TABBYFLOW_RUN_DIR="${RUN_DIR}" TABBYFLOW_CHECKPOINT="${CHECKPOINT}" TABBYFLOW_TRANSFORM="${TRANSFORM}"
export DATANAME DATA_DIR INFO_FILE QUERY_DIR QUERY_SPLIT_MANIFEST QUERY_SPLIT SAMPLE_DIR
export NUM_SAMPLES SEED_BASES BUNDLE_COUNT="${BUNDLES}" TABBYFLOW_SOLVER="${SOLVER}" TABBYFLOW_STEPS="${STEPS}"

submission=$(sbatch --parsable --array="0-$((BUNDLES - 1))" tabbyflow_conditional_sample.sh)
SAMPLE_JOB="${submission%%;*}"
echo "Submitted conditional TabbyFlow sampling: ${SAMPLE_JOB} (${BUNDLES} persistent bundles)"
echo "Samples: ${SAMPLE_DIR}"

if [ -n "${EVALUATION_DIR}" ]; then
    eval_args=(
        --dataname "${DATANAME}"
        --query-dir "${QUERY_DIR}"
        --query-split-manifest "${QUERY_SPLIT_MANIFEST}"
        --query-split "${QUERY_SPLIT}"
        --method "tabbyflow_conditional=${SAMPLE_DIR}"
        --seed-bases "${SEED_BASES}"
        --output-dir "${EVALUATION_DIR}"
        --dependency "${SAMPLE_JOB}"
    )
    for method in "${EVALUATION_METHODS[@]}"; do
        eval_args+=(--method "${method}")
    done
    [ -n "${BASELINE_METHOD}" ] && eval_args+=(--baseline-method "${BASELINE_METHOD}")
    [ "${RUN_SYNTHCITY}" = 0 ] && eval_args+=(--skip-synthcity)
    bash submit_query_suite_evaluation.sh "${eval_args[@]}"
fi

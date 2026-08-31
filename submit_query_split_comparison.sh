#!/bin/bash
# Generic, checkpoint-driven Doob/HARPOON sampling and evaluation workflow.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
QUERY_SPLIT_MANIFEST="${QUERY_SPLIT_MANIFEST:-data90/${DATANAME}/query_splits/sampled_arity_stratified_80_20_seed42.json}"
QUERY_SPLIT="${QUERY_SPLIT:-test}"
DOOB_GUIDE_DIR="${DOOB_GUIDE_DIR:-}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
HARPOON_CHECKPOINT="${HARPOON_CHECKPOINT:-/scratch/work/agrawaa4/harpoon_runtime/saved_models/${DATANAME}/diffputer_selfmade.pt}"
DOOB_LABEL="${DOOB_LABEL:-doob}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_eta02}"
SAMPLE_ROOT="${SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_split_comparison}"
EVAL_DIR="${EVAL_DIR:-evaluations/${DATANAME}/query_split_comparison}"
SEED_BASES="${SEED_BASES:-10000,20000,30000,40000,50000}"
MAX_BUNDLES="${MAX_BUNDLES:-4}"
TARGET_MISSING_PER_BUNDLE="${TARGET_MISSING_PER_BUNDLE:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-}"
RUN_SYNTHCITY="${RUN_SYNTHCITY:-1}"

usage() {
    cat <<'EOF'
Usage: bash submit_query_split_comparison.sh --doob-guide-dir DIR [options]

Checkpoint/path options:
  --doob-guide-dir DIR       Directory containing best_guide.pt
  --base-checkpoint FILE     Frozen TabDiff checkpoint (auto-detected if omitted)
  --harpoon-checkpoint FILE  Official HARPOON checkpoint
  --query-dir DIR            Structured query directory
  --query-split-manifest F   Query-definition train/test manifest
  --query-split NAME         train or test (default: test)
  --sample-root DIR          Shared per-method sample root
  --eval-dir DIR             Evaluation output directory

Naming/runtime options:
  --doob-label NAME
  --harpoon-label NAME
  --seed-bases CSV           Five by default: 10000,...,50000
  --max-bundles N            Maximum long GPU jobs per method (default: 4)
  --num-samples N
  --num-timesteps N          Doob reverse steps (default: 50)
  --train-job-id ID          Depend on a still-running guide training job
  --skip-synthcity           Skip Alpha Precision/Beta Recall
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --doob-guide-dir) DOOB_GUIDE_DIR="$2"; shift 2 ;;
        --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
        --harpoon-checkpoint) HARPOON_CHECKPOINT="$2"; shift 2 ;;
        --query-dir) QUERY_DIR="$2"; shift 2 ;;
        --query-split-manifest) QUERY_SPLIT_MANIFEST="$2"; shift 2 ;;
        --query-split) QUERY_SPLIT="$2"; shift 2 ;;
        --sample-root) SAMPLE_ROOT="$2"; shift 2 ;;
        --eval-dir) EVAL_DIR="$2"; shift 2 ;;
        --doob-label) DOOB_LABEL="$2"; shift 2 ;;
        --harpoon-label) HARPOON_LABEL="$2"; shift 2 ;;
        --seed-bases) SEED_BASES="$2"; shift 2 ;;
        --max-bundles) MAX_BUNDLES="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --num-timesteps) NUM_TIMESTEPS="$2"; shift 2 ;;
        --train-job-id) TRAIN_JOB_ID="$2"; shift 2 ;;
        --skip-synthcity) RUN_SYNTHCITY=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1"; usage; exit 1 ;;
    esac
done

if [ -z "${DOOB_GUIDE_DIR}" ]; then
    echo "ERROR: --doob-guide-dir is required"
    exit 1
fi
if [[ ! "${DOOB_LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]] || \
   [[ ! "${HARPOON_LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: method labels may contain only letters, digits, dot, underscore and dash"
    exit 1
fi
if [ "${QUERY_SPLIT}" != "train" ] && [ "${QUERY_SPLIT}" != "test" ]; then
    echo "ERROR: --query-split must be train or test"
    exit 1
fi
for path in "${QUERY_DIR}" "${QUERY_SPLIT_MANIFEST}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: required path not found: ${path}"
        exit 1
    fi
done
if [ ! -f "${DOOB_GUIDE_DIR}/best_guide.pt" ] && [ -z "${TRAIN_JOB_ID}" ]; then
    echo "ERROR: ${DOOB_GUIDE_DIR}/best_guide.pt does not exist"
    echo "Use --train-job-id ID if training is still running."
    exit 1
fi

if [ -z "${BASE_CHECKPOINT}" ]; then
    shopt -s nullglob
    candidates=("tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"/best_ema_model_*.pt)
    shopt -u nullglob
    if [ "${#candidates[@]}" -ne 1 ]; then
        echo "ERROR: expected one base best_ema checkpoint; pass --base-checkpoint explicitly"
        exit 1
    fi
    BASE_CHECKPOINT="${candidates[0]}"
fi
if [ ! -f "${BASE_CHECKPOINT}" ]; then
    echo "ERROR: base checkpoint not found: ${BASE_CHECKPOINT}"
    exit 1
fi

mapfile -t QUERY_FILES < <(
    python list_accepted_queries.py "${QUERY_DIR}" \
        --query-split-manifest "${QUERY_SPLIT_MANIFEST}" \
        --query-split "${QUERY_SPLIT}"
)
IFS=',' read -r -a BASE_SEEDS <<< "${SEED_BASES}"
if [ "${#BASE_SEEDS[@]}" -ne 5 ]; then
    echo "ERROR: exactly five comma-separated seed bases are required"
    exit 1
fi

missing_count() {
    local label="$1"
    local missing=0
    local query_file query_id seed_index seed_base output
    for query_file in "${QUERY_FILES[@]}"; do
        query_id="$(basename "${query_file}" .json)"
        for seed_index in "${!BASE_SEEDS[@]}"; do
            seed_base="${BASE_SEEDS[seed_index]}"
            if [ "${seed_index}" -eq 0 ]; then
                output="${SAMPLE_ROOT}/${label}/${query_id}.csv"
            else
                output="${SAMPLE_ROOT}/${label}/seed_${seed_base}/${query_id}.csv"
            fi
            if [ ! -f "${output}" ]; then
                missing=$((missing + 1))
            fi
        done
    done
    echo "${missing}"
}

bundle_count() {
    local missing="$1"
    local count=$(((missing + TARGET_MISSING_PER_BUNDLE - 1) / TARGET_MISSING_PER_BUNDLE))
    if [ "${count}" -lt 1 ]; then count=1; fi
    if [ "${count}" -gt "${MAX_BUNDLES}" ]; then count="${MAX_BUNDLES}"; fi
    echo "${count}"
}

mkdir -p logs/query_suite evaluations/slurm "${SAMPLE_ROOT}" "${EVAL_DIR}"
DOOB_MISSING="$(missing_count "${DOOB_LABEL}")"
HARPOON_MISSING="$(missing_count "${HARPOON_LABEL}")"
if [ "${HARPOON_MISSING}" -gt 0 ] && [ ! -f "${HARPOON_CHECKPOINT}" ]; then
    echo "ERROR: ${HARPOON_MISSING} HARPOON replicates are missing, but checkpoint is absent:"
    echo "       ${HARPOON_CHECKPOINT}"
    exit 1
fi

export DATANAME MODEL_NAME QUERY_DIR QUERY_SPLIT_MANIFEST QUERY_SPLIT
export SAMPLE_ROOT SEED_BASES NUM_SAMPLES NUM_TIMESTEPS BASE_CHECKPOINT
export DOOB_GUIDE_DIR HARPOON_CHECKPOINT

DEPENDENCIES=()
DOOB_JOB="none (all five seeds already exist)"
if [ "${DOOB_MISSING}" -gt 0 ]; then
    BUNDLE_COUNT="$(bundle_count "${DOOB_MISSING}")"
    METHOD_KIND=doob METHOD_LABEL="${DOOB_LABEL}"
    export METHOD_KIND METHOD_LABEL BUNDLE_COUNT
    DEP_ARGS=()
    if [ -n "${TRAIN_JOB_ID}" ]; then
        DEP_ARGS+=(--dependency="afterok:${TRAIN_JOB_ID}")
    fi
    submission=$(sbatch --parsable "${DEP_ARGS[@]}" \
        --array="0-$((BUNDLE_COUNT - 1))" query_suite_sample_bundle.sh)
    DOOB_JOB="${submission%%;*}"
    DEPENDENCIES+=("${DOOB_JOB}")
fi

HARPOON_JOB="none (all five seeds already exist)"
if [ "${HARPOON_MISSING}" -gt 0 ]; then
    BUNDLE_COUNT="$(bundle_count "${HARPOON_MISSING}")"
    METHOD_KIND=harpoon METHOD_LABEL="${HARPOON_LABEL}"
    export METHOD_KIND METHOD_LABEL BUNDLE_COUNT
    submission=$(sbatch --parsable \
        --array="0-$((BUNDLE_COUNT - 1))" query_suite_sample_bundle.sh)
    HARPOON_JOB="${submission%%;*}"
    DEPENDENCIES+=("${HARPOON_JOB}")
fi

dependency_args=()
if [ "${#DEPENDENCIES[@]}" -gt 0 ]; then
    dependency="$(IFS=:; echo "${DEPENDENCIES[*]}")"
    dependency_args+=(--dependency="afterok:${dependency}")
fi

export DOOB_LABEL HARPOON_LABEL SUITE_SAMPLE_ROOT="${SAMPLE_ROOT}"
export SUITE_EVAL_DIR="${EVAL_DIR}" REAL_DATA="synthetic/${DATANAME}/real.csv"
export EVAL_GROUP_BY=target_band EVAL_BASELINE_METHOD="${HARPOON_LABEL}"
export QUERY_COORDINATES="${EVAL_DIR}/query_model_coordinates.json"
export INTERVAL_WIDTH_BINS="${INTERVAL_WIDTH_BINS:-10}"

ALPHA_JOB="skipped"
if [ "${RUN_SYNTHCITY}" = "1" ]; then
    ALPHA_BETA_RESULTS="${EVAL_DIR}/alpha_beta_per_query_seed.csv"
    export ALPHA_BETA_RESULTS
    submission=$(sbatch --parsable "${dependency_args[@]}" query_suite_alpha_evaluate.sh)
    ALPHA_JOB="${submission%%;*}"
    dependency_args=(--dependency="afterok:${ALPHA_JOB}")
else
    unset ALPHA_BETA_RESULTS || true
fi

submission=$(sbatch --parsable "${dependency_args[@]}" doob_harpoon_query_suite_evaluate.sh)
EVAL_JOB="${submission%%;*}"

echo "========================================"
echo "Held-out query comparison submitted"
echo "Query split       : ${QUERY_SPLIT} (${#QUERY_FILES[@]} unseen definitions)"
echo "Sampling seeds    : ${SEED_BASES} (5 independent replicates)"
echo "Doob missing      : ${DOOB_MISSING}; job=${DOOB_JOB}"
echo "HARPOON missing   : ${HARPOON_MISSING}; job=${HARPOON_JOB}"
echo "Alpha/Beta job    : ${ALPHA_JOB}"
echo "Normal evaluation : ${EVAL_JOB}"
echo "Doob checkpoint   : ${DOOB_GUIDE_DIR}/best_guide.pt"
echo "Base checkpoint   : ${BASE_CHECKPOINT}"
echo "Sample root       : ${SAMPLE_ROOT}"
echo "Evaluation        : ${EVAL_DIR}"
echo "Plots             : selectivity, arity, transformed mean interval width"
echo "========================================"

#!/bin/bash
# Deprecated compatibility wrapper. Use submit_query_suite_sampling.sh directly.
set -euo pipefail
TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${TABDIFF_PROJECT_ROOT}"
export TABDIFF_PROJECT_ROOT

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
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-}"
RUN_SYNTHCITY=1

usage() {
    cat <<'EOF'
Deprecated compatibility interface:
  bash submit_query_split_comparison.sh --doob-guide-dir DIR [options]

New experiments should use submit_query_suite_sampling.sh with repeated
--method LABEL=KIND:MODEL_PATH arguments.
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
        *) echo "ERROR: unknown option $1"; usage; exit 1 ;;
    esac
done
[ -n "${DOOB_GUIDE_DIR}" ] || { echo "ERROR: --doob-guide-dir is required"; exit 1; }
if [ -z "${BASE_CHECKPOINT}" ]; then
    shopt -s nullglob
    candidates=(tabdiff/ckpt/"${DATANAME}"/"${MODEL_NAME}"/best_ema_model_*.pt)
    shopt -u nullglob
    [ "${#candidates[@]}" -eq 1 ] || {
        echo "ERROR: pass --base-checkpoint explicitly"; exit 1;
    }
    BASE_CHECKPOINT="${candidates[0]}"
fi

args=(
    --dataname "${DATANAME}"
    --query-dir "${QUERY_DIR}"
    --query-split-manifest "${QUERY_SPLIT_MANIFEST}"
    --query-split "${QUERY_SPLIT}"
    --sample-root "${SAMPLE_ROOT}"
    --evaluation-output "${EVAL_DIR}"
    --query-coordinates "${EVAL_DIR}/query_model_coordinates.json"
    --seed-bases "${SEED_BASES}"
    --max-bundles "${MAX_BUNDLES}"
    --num-samples "${NUM_SAMPLES}"
    --num-timesteps "${NUM_TIMESTEPS}"
    --base-checkpoint "${BASE_CHECKPOINT}"
    --baseline-method "${HARPOON_LABEL}"
    --method "${DOOB_LABEL}=doob:${DOOB_GUIDE_DIR}"
    --method "${HARPOON_LABEL}=harpoon:${HARPOON_CHECKPOINT}"
)
[ -n "${TRAIN_JOB_ID}" ] && args+=(--dependency "${TRAIN_JOB_ID}")
[ "${RUN_SYNTHCITY}" = 0 ] && args+=(--skip-synthcity)
echo "NOTICE: submit_query_split_comparison.sh is deprecated; forwarding to the modular pipeline."
exec bash submit_query_suite_sampling.sh "${args[@]}"

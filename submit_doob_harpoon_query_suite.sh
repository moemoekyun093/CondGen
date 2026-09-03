#!/bin/bash
# Deprecated environment-variable wrapper around the modular query pipeline.
set -euo pipefail
TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${TABDIFF_PROJECT_ROOT}"
export TABDIFF_PROJECT_ROOT

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
DOOB_LABEL="${DOOB_LABEL:-doob_curriculum}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_eta02}"
DOOB_GUIDE_DIR="${DOOB_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_query_curriculum_d48_l2_12000}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/scratch/work/agrawaa4/harpoon_runtime}"
HARPOON_CHECKPOINT="${HARPOON_CHECKPOINT:-${RUNTIME_ROOT}/saved_models/${DATANAME}/diffputer_selfmade.pt}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_suite_comparison}"
SUITE_EVAL_DIR="${SUITE_EVAL_DIR:-evaluations/${DATANAME}/doob_vs_harpoon_query_suite}"
SEED_BASES="${SEED_BASES:-10000}"
MAX_BUNDLES="${MAX_BUNDLES:-4}"

if [ -z "${BASE_CHECKPOINT:-}" ]; then
    shopt -s nullglob
    candidates=(tabdiff/ckpt/"${DATANAME}"/"${MODEL_NAME}"/best_ema_model_*.pt)
    shopt -u nullglob
    [ "${#candidates[@]}" -eq 1 ] || {
        echo "ERROR: set BASE_CHECKPOINT explicitly"; exit 1;
    }
    BASE_CHECKPOINT="${candidates[0]}"
fi

args=(
    --dataname "${DATANAME}"
    --query-dir "${QUERY_DIR}"
    --sample-root "${SUITE_SAMPLE_ROOT}"
    --evaluation-output "${SUITE_EVAL_DIR}"
    --group-by "${EVAL_GROUP_BY:-target_band}"
    --query-coordinates "${QUERY_COORDINATES:-data90/${DATANAME}/query_splits/query_model_coordinates.json}"
    --seed-bases "${SEED_BASES}"
    --max-bundles "${MAX_BUNDLES}"
    --base-checkpoint "${BASE_CHECKPOINT}"
    --baseline-method "${HARPOON_LABEL}"
    --method "${DOOB_LABEL}=doob:${DOOB_GUIDE_DIR}"
    --method "${HARPOON_LABEL}=harpoon:${HARPOON_CHECKPOINT}"
)
[ -n "${REAL_DATA:-}" ] && args+=(--real-data "${REAL_DATA}")
[ -n "${INFO_FILE:-}" ] && args+=(--info-file "${INFO_FILE}")
[ -n "${FILTERED_MIN_ROWS:-}" ] && args+=(--filtered-min-rows "${FILTERED_MIN_ROWS}")
if [ -n "${QUERY_SPLIT_MANIFEST:-}" ]; then
    args+=(--query-split-manifest "${QUERY_SPLIT_MANIFEST}" --query-split "${QUERY_SPLIT:-test}")
fi
[ "${QUERY_TEST_SUPPORTED_ONLY:-0}" = 1 ] && args+=(--test-supported-only)
[ -n "${TRAIN_JOB_ID:-}" ] && args+=(--dependency "${TRAIN_JOB_ID}")
[ "${RUN_SYNTHCITY:-1}" = 0 ] && args+=(--skip-synthcity)

export RUNTIME_ROOT
echo "NOTICE: submit_doob_harpoon_query_suite.sh is deprecated; forwarding to the modular pipeline."
exec bash submit_query_suite_sampling.sh "${args[@]}" "$@"

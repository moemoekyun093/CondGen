#!/bin/bash
# Run a bundled HARPOON-style-on-TabDiff eta sweep on unseen Shoppers queries.
set -euo pipefail

TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${TABDIFF_PROJECT_ROOT}"
export TABDIFF_PROJECT_ROOT

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
QUERY_SPLIT_MANIFEST="${QUERY_SPLIT_MANIFEST:-data90/${DATANAME}/query_splits/sampled_arity_stratified_80_20_seed42.json}"
ETA_VALUES="${ETA_VALUES:-0.2,0.5,1,2,5}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"
MAX_BUNDLES="${MAX_BUNDLES:-1}"
SAMPLE_ROOT="${SAMPLE_ROOT:-conditional_samples/${DATANAME}/all_methods_unseen_test_1seed}"
EVAL_DIR="${EVAL_DIR:-evaluations/${DATANAME}/all_methods_unseen_test_1seed/harpoon_style_eta_sweep}"

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
    --query-split-manifest "${QUERY_SPLIT_MANIFEST}"
    --query-split test
    --num-seeds 1
    --num-samples "${NUM_SAMPLES}"
    --num-timesteps "${NUM_TIMESTEPS}"
    --max-bundles "${MAX_BUNDLES}"
    --base-checkpoint "${BASE_CHECKPOINT}"
    --sample-root "${SAMPLE_ROOT}"
    --evaluation-output "${EVAL_DIR}"
    --baseline-method harpoon_eta02
    --evaluation-method "doob_center_logwidth=conditional_samples/${DATANAME}/sampled_arity_unseen_query_comparison/doob_center_logwidth_mlp_curriculum_8000"
    --evaluation-method "harpoon_eta02=conditional_samples/${DATANAME}/sampled_arity_unseen_query_comparison/harpoon_eta02"
    --evaluation-method "diffputer=conditional_samples/${DATANAME}/native_baselines_test/diffputer"
    --evaluation-method "great=conditional_samples/${DATANAME}/native_baselines_test/great"
)

IFS=',' read -r -a etas <<< "${ETA_VALUES}"
for eta in "${etas[@]}"; do
    tag="${eta//./p}"
    args+=(--method "harpoon_style_eta${tag}_s${NUM_TIMESTEPS}=harpoon_style:${eta}")
done
[ -n "${DEPENDENCY:-}" ] && args+=(--dependency "${DEPENDENCY}")
[ -n "${EVALUATION_DEPENDENCY:-}" ] && \
    args+=(--evaluation-dependency "${EVALUATION_DEPENDENCY}")
[ "${RUN_SYNTHCITY:-1}" = 0 ] && args+=(--skip-synthcity)

echo "HARPOON-style eta sweep: ${ETA_VALUES}; ${NUM_TIMESTEPS} reverse steps"
echo "Samples: ${SAMPLE_ROOT}"
echo "Evaluation: ${EVAL_DIR}"
exec bash submit_query_suite_sampling.sh "${args[@]}"

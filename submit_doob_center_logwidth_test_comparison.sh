#!/bin/bash
# Evaluate the 8000-step center/log-width guide on unseen query definitions.
# Extra options are forwarded to the generic checkpoint-driven workflow.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
DOOB_LABEL="${DOOB_LABEL:-doob_center_logwidth_mlp_curriculum_8000}"
DOOB_GUIDE_DIR="${DOOB_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_center_logwidth_mlp_qsplit_multiq8_realized_curriculum_d48_l2_8000}"

# Reuse the historical direct HARPOON CSV as seed 1. Seeds 2--5 are stored in
# seed-specific subdirectories under the same root.
SAMPLE_ROOT="${SAMPLE_ROOT:-conditional_samples/${DATANAME}/sampled_arity_unseen_query_comparison}"
EVAL_DIR="${EVAL_DIR:-evaluations/${DATANAME}/center_logwidth_mlp_vs_harpoon_unseen_test_5seeds}"

bash submit_query_split_comparison.sh \
    --doob-guide-dir "${DOOB_GUIDE_DIR}" \
    --doob-label "${DOOB_LABEL}" \
    --harpoon-label "${HARPOON_LABEL:-harpoon_eta02}" \
    --query-dir "${QUERY_DIR:-data90/${DATANAME}/queries}" \
    --query-split-manifest "${QUERY_SPLIT_MANIFEST:-data90/${DATANAME}/query_splits/sampled_arity_stratified_80_20_seed42.json}" \
    --query-split test \
    --sample-root "${SAMPLE_ROOT}" \
    --eval-dir "${EVAL_DIR}" \
    "$@"

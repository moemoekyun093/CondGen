#!/bin/bash
# Legacy sampled-arity model wrapper around the generic five-seed workflow.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
DOOB_LABEL="${DOOB_LABEL:-doob_sampled_arity_qsplit_multiq8_8000}"
DOOB_GUIDE_DIR="${DOOB_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_sampled_arity_qsplit_multiq8_realized_curriculum_d48_l2_8000}"

bash submit_query_split_comparison.sh \
    --doob-guide-dir "${DOOB_GUIDE_DIR}" \
    --doob-label "${DOOB_LABEL}" \
    --harpoon-label "${HARPOON_LABEL:-harpoon_eta02}" \
    --query-dir "${QUERY_DIR:-data90/${DATANAME}/queries}" \
    --query-split-manifest "${QUERY_SPLIT_MANIFEST:-data90/${DATANAME}/query_splits/sampled_arity_stratified_80_20_seed42.json}" \
    --query-split test \
    --sample-root "${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/sampled_arity_unseen_query_comparison}" \
    --eval-dir "${SUITE_EVAL_DIR:-evaluations/${DATANAME}/sampled_arity_unseen_query_comparison_5seeds}" \
    "$@"

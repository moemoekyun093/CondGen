#!/bin/bash
# Compare the sampled-arity Doob guide with HARPOON on unseen query definitions.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
DOOB_LABEL="${DOOB_LABEL:-doob_sampled_arity_qsplit_multiq8_8000}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_eta02}"
DOOB_GUIDE_DIR="${DOOB_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_sampled_arity_qsplit_multiq8_realized_curriculum_d48_l2_8000}"
QUERY_SPLIT_MANIFEST="${QUERY_SPLIT_MANIFEST:-data90/${DATANAME}/query_splits/sampled_arity_stratified_80_20_seed42.json}"
QUERY_SPLIT=test
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/sampled_arity_unseen_query_comparison}"
SUITE_EVAL_DIR="${SUITE_EVAL_DIR:-evaluations/${DATANAME}/sampled_arity_unseen_query_comparison}"
# Match the original TabDiff/HARPOON density protocol: compare generated rows
# with the real training partition satisfying each held-out query definition.
REAL_DATA="${REAL_DATA:-synthetic/${DATANAME}/real.csv}"
QUERY_TEST_SUPPORTED_ONLY=0
EVAL_GROUP_BY="${EVAL_GROUP_BY:-target_band}"
EVAL_BASELINE_METHOD="${EVAL_BASELINE_METHOD:-${HARPOON_LABEL}}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-}"

export DATANAME MODEL_NAME QUERY_DIR DOOB_LABEL HARPOON_LABEL DOOB_GUIDE_DIR
export SUITE_SAMPLE_ROOT SUITE_EVAL_DIR REAL_DATA QUERY_TEST_SUPPORTED_ONLY
export EVAL_GROUP_BY EVAL_BASELINE_METHOD TRAIN_JOB_ID
export QUERY_SPLIT_MANIFEST QUERY_SPLIT

bash submit_doob_harpoon_query_suite.sh

#!/bin/bash
# Build and evaluate nested partial-arity queries from the 40% query band.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
SOURCE_QUERY_DIR="${SOURCE_QUERY_DIR:-data90/${DATANAME}/queries_full}"
QUERY_DIR="${ARITY_QUERY_DIR:-data90/${DATANAME}/queries_arity_b40}"
ARITIES="${ARITIES:-2,4,8,12,18}"
DOOB_GUIDE_DIR="${DOOB_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_query_tight_curriculum_d48_l2_18000}"
DOOB_LABEL="${DOOB_LABEL:-doob_tight_curriculum_arity}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_eta02_arity}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/arity_b40_comparison}"
SUITE_EVAL_DIR="${SUITE_EVAL_DIR:-evaluations/${DATANAME}/doob_vs_harpoon_arity_b40}"
EVAL_GROUP_BY="arity"
REUSE_HARPOON="${REUSE_HARPOON:-0}"

python generate_query_arity_suite.py \
    --source-query-dir "${SOURCE_QUERY_DIR}" \
    --output-query-dir "${QUERY_DIR}" \
    --source-band 0.4 \
    --arities "${ARITIES}"

export DATANAME MODEL_NAME QUERY_DIR DOOB_GUIDE_DIR DOOB_LABEL HARPOON_LABEL
export SUITE_SAMPLE_ROOT SUITE_EVAL_DIR EVAL_GROUP_BY REUSE_HARPOON
bash submit_doob_harpoon_query_suite.sh

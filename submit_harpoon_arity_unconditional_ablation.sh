#!/bin/bash
# Strict arity sanity check using shared unconditional samples.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
QUERY_TARGET_BAND=0.4
HARPOON_LABEL="${UNCONDITIONAL_ARITY_LABEL:-harpoon_unconditional_shared_b40}"
HARPOON_GUIDANCE_SCALE=0
SUITE_SAMPLE_ROOT="${ARITY_SAMPLE_ROOT:-conditional_samples/${DATANAME}/arity_b40_comparison}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
export DATANAME QUERY_DIR QUERY_TARGET_BAND HARPOON_LABEL HARPOON_GUIDANCE_SCALE
export SUITE_SAMPLE_ROOT

NUM_QUERIES=$(python list_accepted_queries.py "${QUERY_DIR}" --target-band 0.4 | wc -l)
if [ "${NUM_QUERIES}" -le 0 ]; then
    echo "ERROR: no accepted 40% source queries found"
    exit 1
fi
mkdir -p harpoon_logs evaluations/slurm

SAMPLE_SUBMISSION=$(sbatch \
    --parsable \
    --array="0-$((NUM_QUERIES - 1))%${MAX_CONCURRENT}" \
    harpoon_query_suite_sample.sh)
SAMPLE_JOB="${SAMPLE_SUBMISSION%%;*}"
EVAL_SUBMISSION=$(sbatch \
    --parsable \
    --dependency="afterok:${SAMPLE_JOB}" \
    harpoon_arity_unconditional_evaluate.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "Strict HARPOON arity ablation submitted"
echo "  shared unconditional samples: ${SAMPLE_JOB} (${NUM_QUERIES} tasks)"
echo "  monotonicity evaluation      : ${EVAL_JOB}"

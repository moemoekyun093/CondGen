#!/bin/bash
# Submit the controlled numerical correction-cap diagnostic.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
CLIP_CAPS="${CLIP_CAPS:-2,5,10,20}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
export DATANAME QUERY_DIR CLIP_CAPS

NUM_QUERIES=$(python list_accepted_queries.py "${QUERY_DIR}" --one-per-band | wc -l)
IFS=',' read -r -a CAPS <<< "${CLIP_CAPS}"
NUM_TASKS=$((NUM_QUERIES * ${#CAPS[@]}))
mkdir -p logs evaluations/slurm

SAMPLE_SUBMISSION=$(sbatch \
    --parsable \
    --array="0-$((NUM_TASKS - 1))%${MAX_CONCURRENT}" \
    doob_numerical_clip_diagnostic_sample.sh)
SAMPLE_JOB="${SAMPLE_SUBMISSION%%;*}"
EVAL_SUBMISSION=$(sbatch \
    --parsable \
    --dependency="afterok:${SAMPLE_JOB}" \
    doob_numerical_clip_diagnostic_evaluate.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "Numerical clipping diagnostic submitted"
echo "  representative queries: ${NUM_QUERIES}"
echo "  correction caps       : ${CLIP_CAPS}"
echo "  sampling array        : ${SAMPLE_JOB}"
echo "  aggregate evaluation  : ${EVAL_JOB}"

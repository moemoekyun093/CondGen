#!/bin/bash
# Sample HARPOON without guidance and compare to existing eta=0.2 samples.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_suite_comparison}"
GUIDED_LABEL="${GUIDED_LABEL:-harpoon_eta02}"
UNGUIDED_LABEL="${UNGUIDED_LABEL:-harpoon_unconditional}"
HARPOON_LABEL="${UNGUIDED_LABEL}"
HARPOON_GUIDANCE_SCALE=0
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
export DATANAME QUERY_DIR SUITE_SAMPLE_ROOT GUIDED_LABEL UNGUIDED_LABEL
export HARPOON_LABEL HARPOON_GUIDANCE_SCALE

mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}")
NUM_QUERIES="${#QUERY_FILES[@]}"
if [ "${NUM_QUERIES}" -le 0 ]; then
    echo "ERROR: no accepted queries found"
    exit 1
fi
for QUERY_FILE in "${QUERY_FILES[@]}"; do
    QUERY_ID="$(basename "${QUERY_FILE}" .json)"
    GUIDED_SAMPLE="${SUITE_SAMPLE_ROOT}/${GUIDED_LABEL}/${QUERY_ID}.csv"
    if [ ! -f "${GUIDED_SAMPLE}" ]; then
        echo "ERROR: guided HARPOON sample not found: ${GUIDED_SAMPLE}"
        exit 1
    fi
done
mkdir -p harpoon_logs evaluations/slurm

SAMPLE_SUBMISSION=$(sbatch \
    --parsable \
    --array="0-$((NUM_QUERIES - 1))%${MAX_CONCURRENT}" \
    harpoon_query_suite_sample.sh)
SAMPLE_JOB="${SAMPLE_SUBMISSION%%;*}"
EVAL_SUBMISSION=$(sbatch \
    --parsable \
    --dependency="afterok:${SAMPLE_JOB}" \
    harpoon_guidance_ablation_evaluate.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "HARPOON guidance ablation submitted"
echo "  queries             : ${NUM_QUERIES}"
echo "  unguided eta        : 0"
echo "  unconditional array : ${SAMPLE_JOB}"
echo "  paired evaluation   : ${EVAL_JOB}"

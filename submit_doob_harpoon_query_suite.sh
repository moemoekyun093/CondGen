#!/bin/bash
# Submit paired Doob/HARPOON suite sampling and aggregate evaluation.

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

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
DOOB_MAX_CONCURRENT="${DOOB_MAX_CONCURRENT:-8}"
HARPOON_MAX_CONCURRENT="${HARPOON_MAX_CONCURRENT:-8}"
REUSE_HARPOON="${REUSE_HARPOON:-0}"

# The Doob array sampler accepts the generic label=guide-directory format.
GUIDE_SPECS="${DOOB_LABEL}=${DOOB_GUIDE_DIR}"
export DATANAME MODEL_NAME QUERY_DIR DOOB_LABEL HARPOON_LABEL GUIDE_SPECS
export SUITE_SAMPLE_ROOT SUITE_EVAL_DIR
export RUNTIME_ROOT HARPOON_CHECKPOINT
export EVAL_GROUP_BY

if [ ! -d "${QUERY_DIR}" ]; then
    echo "ERROR: query directory not found: ${QUERY_DIR}"
    exit 1
fi
if [ ! -f "${DOOB_GUIDE_DIR}/best_guide.pt" ]; then
    echo "ERROR: trained checkpoint not found: ${DOOB_GUIDE_DIR}/best_guide.pt"
    exit 1
fi
mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}")
NUM_QUERIES="${#QUERY_FILES[@]}"
if [ "${NUM_QUERIES}" -le 0 ]; then
    echo "ERROR: no accepted queries were selected"
    exit 1
fi
mkdir -p logs harpoon_logs evaluations/slurm

if [ "${REUSE_HARPOON}" = "1" ]; then
    for QUERY_FILE in "${QUERY_FILES[@]}"; do
        QUERY_ID="$(basename "${QUERY_FILE}" .json)"
        SAMPLE="${SUITE_SAMPLE_ROOT}/${HARPOON_LABEL}/${QUERY_ID}.csv"
        if [ ! -f "${SAMPLE}" ]; then
            echo "ERROR: reusable HARPOON sample not found: ${SAMPLE}"
            exit 1
        fi
    done
elif [ ! -f "${HARPOON_CHECKPOINT}" ]; then
    echo "ERROR: trained checkpoint not found: ${HARPOON_CHECKPOINT}"
    exit 1
fi

DOOB_SUBMISSION=$(sbatch \
    --parsable \
    --array="0-$((NUM_QUERIES - 1))%${DOOB_MAX_CONCURRENT}" \
    doob_query_suite_sample.sh)
DOOB_JOB="${DOOB_SUBMISSION%%;*}"

DEPENDENCY="afterok:${DOOB_JOB}"
if [ "${REUSE_HARPOON}" = "1" ]; then
    HARPOON_JOB="reused existing samples"
else
    HARPOON_SUBMISSION=$(sbatch \
        --parsable \
        --array="0-$((NUM_QUERIES - 1))%${HARPOON_MAX_CONCURRENT}" \
        harpoon_query_suite_sample.sh)
    HARPOON_JOB="${HARPOON_SUBMISSION%%;*}"
    DEPENDENCY="${DEPENDENCY}:${HARPOON_JOB}"
fi

EVAL_SUBMISSION=$(sbatch \
    --parsable \
    --dependency="${DEPENDENCY}" \
    doob_harpoon_query_suite_evaluate.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "Paired Doob/HARPOON query-suite evaluation submitted"
echo "  accepted queries: ${NUM_QUERIES}"
echo "  Doob array      : ${DOOB_JOB}"
echo "  HARPOON array   : ${HARPOON_JOB}"
echo "  aggregate eval  : ${EVAL_JOB} (${DEPENDENCY})"

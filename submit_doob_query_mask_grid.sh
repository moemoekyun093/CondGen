#!/bin/bash
# Generate, sample, evaluate, and plot the selectivity-by-mask grid.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
SOURCE_QUERY_DIR="${SOURCE_QUERY_DIR:-data90/${DATANAME}/queries_full}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_selectivity_mask_grid}"
BANDS="${BANDS:-0.005,0.01,0.02,0.05,0.1,0.25,0.4}"
ARITIES="${ARITIES:-0,2,4,8,12,18}"
QUERIES_PER_BAND="${QUERIES_PER_BAND:-1}"
MASKS_PER_ARITY="${MASKS_PER_ARITY:-3}"
METHOD_LABEL="${METHOD_LABEL:-doob_masked_25000}"
DOOB_GUIDE_DIR="${DOOB_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_query_tight_curriculum_masked_10_10_80_d48_l2_25000}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/selectivity_mask_grid}"
SUITE_EVAL_DIR="${SUITE_EVAL_DIR:-evaluations/${DATANAME}/selectivity_mask_grid}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"

python generate_query_mask_grid.py \
    --source-query-dir "${SOURCE_QUERY_DIR}" \
    --output-query-dir "${QUERY_DIR}" \
    --real-data "synthetic/${DATANAME}/real.csv" \
    --bands "${BANDS}" \
    --arities "${ARITIES}" \
    --queries-per-band "${QUERIES_PER_BAND}" \
    --masks-per-arity "${MASKS_PER_ARITY}"

if [ ! -f "${DOOB_GUIDE_DIR}/best_guide.pt" ]; then
    echo "ERROR: trained guide not found: ${DOOB_GUIDE_DIR}/best_guide.pt"
    exit 1
fi
mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}")
NUM_QUERIES="${#QUERY_FILES[@]}"
if [ "${NUM_QUERIES}" -le 0 ]; then
    echo "ERROR: no grid queries were generated"
    exit 1
fi
mkdir -p logs evaluations/slurm "${SUITE_SAMPLE_ROOT}" "${SUITE_EVAL_DIR}"

GUIDE_SPECS="${METHOD_LABEL}=${DOOB_GUIDE_DIR}"
export DATANAME MODEL_NAME QUERY_DIR GUIDE_SPECS METHOD_LABEL
export SUITE_SAMPLE_ROOT SUITE_EVAL_DIR NUM_SAMPLES NUM_TIMESTEPS

SAMPLE_SUBMISSION=$(sbatch \
    --parsable \
    --array="0-$((NUM_QUERIES - 1))%${MAX_CONCURRENT}" \
    doob_query_suite_sample.sh)
SAMPLE_JOB="${SAMPLE_SUBMISSION%%;*}"
EVAL_SUBMISSION=$(sbatch \
    --parsable \
    --dependency="afterok:${SAMPLE_JOB}" \
    doob_query_mask_grid_evaluate.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "========================================"
echo "Selectivity-by-mask evaluation submitted"
echo "Queries       : ${NUM_QUERIES}"
echo "Bands         : ${BANDS}"
echo "Arities       : ${ARITIES}"
echo "Masks/arity   : ${MASKS_PER_ARITY} (anchors have one unique mask)"
echo "Sample array  : ${SAMPLE_JOB}"
echo "Evaluation    : ${EVAL_JOB} (afterok:${SAMPLE_JOB})"
echo "Output        : ${SUITE_EVAL_DIR}"
echo "========================================"

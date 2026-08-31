#!/bin/bash
# Train, sample, and evaluate one tight arity-one structured query.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
QUERY_ID="${QUERY_ID:-q_shoppers_b00p5_k01_num_1}"
STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
METHOD_LABEL="${METHOD_LABEL:-doob_fixed_exit_rates_b005_d48_l2_1000}"
GUIDE_DIR="${GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/${METHOD_LABEL}}"
SAMPLE_DIR="${SAMPLE_DIR:-conditional_samples/${DATANAME}/fixed_query_experiment/${METHOD_LABEL}}"
SAMPLE_OUTPUT="${SAMPLE_DIR}/${QUERY_ID}.csv"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-evaluations/${DATANAME}/fixed_query_experiment/${METHOD_LABEL}}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
VIOLATION_HISTOGRAM_BINS="${VIOLATION_HISTOGRAM_BINS:-40}"

QUERY_SPEC="${QUERY_DIR}/${QUERY_ID}.json"
if [ ! -f "${QUERY_SPEC}" ]; then
    echo "ERROR: query does not exist: ${QUERY_SPEC}"
    exit 1
fi
ARITY=$(python -c "import json; q=json.load(open('${QUERY_SPEC}')); print(q.get('arity', len(q['predicates'])))")
TARGET_BAND=$(python -c "import json; q=json.load(open('${QUERY_SPEC}')); print(q['target_band'])")
SUPPORT=$(python -c "import json; q=json.load(open('${QUERY_SPEC}')); print(q['counts']['train'])")
if [ "${ARITY}" -ne 1 ]; then
    echo "ERROR: fixed-query experiment requires arity one, got ${ARITY}"
    exit 1
fi

mkdir -p logs evaluations/slurm "${EVAL_OUTPUT_DIR}"
export DATANAME MODEL_NAME QUERY_DIR QUERY_ID STEPS BATCH_SIZE METHOD_LABEL
export GUIDE_DIR SAMPLE_OUTPUT EVAL_OUTPUT_DIR NUM_SAMPLES VIOLATION_HISTOGRAM_BINS

DEPENDENCIES=()
if [ -f "${GUIDE_DIR}/best_guide.pt" ]; then
    TRAIN_JOB="reused existing checkpoint"
else
    GUIDE_DIR_NAME="${GUIDE_DIR}"
    QUERIES_PER_STEP=1
    LR=1e-3
    D_TOKEN=48
    NUM_LAYERS=2
    N_HEAD=4
    FACTOR=2
    QUERY_SAMPLING_MODE=uniform
    QUERY_SPLIT_MANIFEST=""
    QUERY_SPLIT=""
    PREDICATE_MASK_MODE=full
    export GUIDE_DIR_NAME QUERIES_PER_STEP LR D_TOKEN NUM_LAYERS N_HEAD FACTOR
    export QUERY_SAMPLING_MODE QUERY_SPLIT_MANIFEST QUERY_SPLIT PREDICATE_MASK_MODE
    TRAIN_SUBMISSION=$(sbatch --parsable doob_query_train.sh "${GUIDE_DIR}")
    TRAIN_JOB="${TRAIN_SUBMISSION%%;*}"
    DEPENDENCIES+=("${TRAIN_JOB}")
fi

if [ -f "${SAMPLE_OUTPUT}" ] && [ -f "${SAMPLE_OUTPUT%.csv}.constraints.json" ]; then
    SAMPLE_JOB="reused existing samples"
else
    SAMPLE_DEPENDENCY=()
    if [ "${#DEPENDENCIES[@]}" -gt 0 ]; then
        SAMPLE_DEPENDENCY+=(--dependency="afterok:${DEPENDENCIES[0]}")
    fi
    SAMPLE_SUBMISSION=$(sbatch --parsable "${SAMPLE_DEPENDENCY[@]}" sample_doob_fixed_query.sh)
    SAMPLE_JOB="${SAMPLE_SUBMISSION%%;*}"
    DEPENDENCIES=("${SAMPLE_JOB}")
fi

EVAL_DEPENDENCY=()
if [ "${#DEPENDENCIES[@]}" -gt 0 ]; then
    EVAL_DEPENDENCY+=(--dependency="afterok:${DEPENDENCIES[0]}")
fi
EVAL_SUBMISSION=$(sbatch --parsable "${EVAL_DEPENDENCY[@]}" evaluate_doob_fixed_query.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "========================================"
echo "Fixed-query Doob experiment submitted"
echo "Query             : ${QUERY_ID}"
echo "Arity             : ${ARITY}"
echo "Target selectivity: ${TARGET_BAND}"
echo "Training support  : ${SUPPORT} rows"
echo "Guide             : d48/L2, ${STEPS} optimizer steps"
echo "Training          : ${TRAIN_JOB}"
echo "Sampling          : ${SAMPLE_JOB}"
echo "Evaluation        : ${EVAL_JOB}"
echo "Checkpoint        : ${GUIDE_DIR}"
echo "Samples           : ${SAMPLE_OUTPUT}"
echo "Metrics           : ${EVAL_OUTPUT_DIR}"
echo "Violation bins    : ${VIOLATION_HISTOGRAM_BINS}"
echo "========================================"

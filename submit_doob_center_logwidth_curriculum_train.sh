#!/bin/bash
# Train the ordinary-MLP center/log-width guide on the train-query curriculum.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
mkdir -p logs

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
STEPS="${STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
QUERIES_PER_STEP="${QUERIES_PER_STEP:-8}"
LR="${LR:-1e-3}"
D_TOKEN="${D_TOKEN:-48}"
NUM_LAYERS="${NUM_LAYERS:-2}"
N_HEAD="${N_HEAD:-4}"
FACTOR="${FACTOR:-2}"
GUIDE_DIR_NAME="${GUIDE_DIR_NAME:-doob_center_logwidth_mlp_qsplit_multiq8_realized_curriculum_d${D_TOKEN}_l${NUM_LAYERS}_${STEPS}}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
QUERY_SPLIT_MANIFEST="${QUERY_SPLIT_MANIFEST:-data90/${DATANAME}/query_splits/sampled_arity_stratified_80_20_seed42.json}"
QUERY_SPLIT=train

QUERY_SAMPLING_MODE=curriculum
CURRICULUM_SELECTIVITY_SOURCE=realized_train
CURRICULUM_WARMUP_STEPS="${CURRICULUM_WARMUP_STEPS:-1000}"
CURRICULUM_TRANSITION_STEPS="${CURRICULUM_TRANSITION_STEPS:-2000}"
CURRICULUM_WARMUP_PROBABILITIES="${CURRICULUM_WARMUP_PROBABILITIES:-0.50,0.30,0.20}"
CURRICULUM_FINAL_PROBABILITIES="${CURRICULUM_FINAL_PROBABILITIES:-0.30,0.30,0.40}"

QUERY_ARCHITECTURE=per_token_fusion
BOUND_EMBEDDING_MODE=mlp
BOUND_TOKEN_PARAMETERIZATION=center_logwidth
QUERY_PRESENCE_MODE=active_flags
PREDICATE_MASK_MODE=full
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"

if [ "$((BATCH_SIZE % QUERIES_PER_STEP))" -ne 0 ]; then
    echo "ERROR: BATCH_SIZE must be divisible by QUERIES_PER_STEP"
    exit 1
fi
if [ ! -f "${QUERY_SPLIT_MANIFEST}" ]; then
    echo "ERROR: query split manifest not found: ${QUERY_SPLIT_MANIFEST}"
    exit 1
fi
TRAIN_QUERY_COUNT=$(python list_accepted_queries.py \
    "${QUERY_DIR}" --query-split-manifest "${QUERY_SPLIT_MANIFEST}" \
    --query-split train | wc -l)
TEST_QUERY_COUNT=$(python list_accepted_queries.py \
    "${QUERY_DIR}" --query-split-manifest "${QUERY_SPLIT_MANIFEST}" \
    --query-split test | wc -l)

export DATANAME MODEL_NAME GUIDE_DIR_NAME QUERY_DIR STEPS BATCH_SIZE
export QUERIES_PER_STEP LR D_TOKEN NUM_LAYERS N_HEAD FACTOR CHECKPOINT_EVERY
export QUERY_SPLIT_MANIFEST QUERY_SPLIT QUERY_SAMPLING_MODE
export CURRICULUM_SELECTIVITY_SOURCE CURRICULUM_WARMUP_STEPS
export CURRICULUM_TRANSITION_STEPS CURRICULUM_WARMUP_PROBABILITIES
export CURRICULUM_FINAL_PROBABILITIES PREDICATE_MASK_MODE
export QUERY_ARCHITECTURE BOUND_EMBEDDING_MODE BOUND_TOKEN_PARAMETERIZATION
export QUERY_PRESENCE_MODE

SUBMISSION=$(sbatch --parsable doob_query_train.sh "${GUIDE_DIR_NAME}")
JOB_ID="${SUBMISSION%%;*}"

echo "========================================"
echo "Center/log-width curriculum training submitted"
echo "Job               : ${JOB_ID}"
echo "Dataset           : ${DATANAME}"
echo "Query split       : train (${TRAIN_QUERY_COUNT}); held-out test=${TEST_QUERY_COUNT}"
echo "Optimizer steps   : ${STEPS}"
echo "Batch             : ${QUERIES_PER_STEP} queries x $((BATCH_SIZE / QUERIES_PER_STEP)) rows"
echo "Guide             : ordinary MLP per-column fusion, d${D_TOKEN}/L${NUM_LAYERS}"
echo "Coordinates       : center and log transformed-space width"
echo "Predicate masking : disabled; queries used exactly as written"
echo "Warm curriculum   : ${CURRICULUM_WARMUP_PROBABILITIES} for ${CURRICULUM_WARMUP_STEPS} steps"
echo "Transition        : linear over ${CURRICULUM_TRANSITION_STEPS} steps"
echo "Final curriculum  : ${CURRICULUM_FINAL_PROBABILITIES} from step $((CURRICULUM_WARMUP_STEPS + CURRICULUM_TRANSITION_STEPS + 1))"
echo "Output            : tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/${GUIDE_DIR_NAME}"
echo "========================================"

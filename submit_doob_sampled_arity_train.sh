#!/bin/bash
# Train the structured Doob guide on the accepted sampled-arity query suite.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff
mkdir -p logs

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
GUIDE_DIR_NAME="${GUIDE_DIR_NAME:-doob_sampled_arity_realized_curriculum_d48_l2_8000}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
STEPS="${STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-1e-3}"

# Same three-phase curriculum as the previous tight schedule, with the final
# tight-query mass reduced from 70% to 60% and redistributed to broad/medium.
QUERY_SAMPLING_MODE=curriculum
CURRICULUM_SELECTIVITY_SOURCE=realized_train
CURRICULUM_WARMUP_STEPS="${CURRICULUM_WARMUP_STEPS:-1000}"
CURRICULUM_TRANSITION_STEPS="${CURRICULUM_TRANSITION_STEPS:-2000}"
CURRICULUM_WARMUP_PROBABILITIES="${CURRICULUM_WARMUP_PROBABILITIES:-0.50,0.30,0.20}"
CURRICULUM_FINAL_PROBABILITIES="${CURRICULUM_FINAL_PROBABILITIES:-0.15,0.25,0.60}"

# Each sampled-arity query already specifies its active columns. Do not apply a
# second random mask; keep mixed masking available only through the legacy flag.
PREDICATE_MASK_MODE=full

export DATANAME MODEL_NAME GUIDE_DIR_NAME QUERY_DIR STEPS BATCH_SIZE LR
export QUERY_SAMPLING_MODE CURRICULUM_SELECTIVITY_SOURCE
export CURRICULUM_WARMUP_STEPS CURRICULUM_TRANSITION_STEPS
export CURRICULUM_WARMUP_PROBABILITIES CURRICULUM_FINAL_PROBABILITIES
export PREDICATE_MASK_MODE

SUBMISSION=$(sbatch --parsable doob_query_train.sh "${GUIDE_DIR_NAME}")
JOB_ID="${SUBMISSION%%;*}"

echo "========================================"
echo "Sampled-arity Doob training submitted"
echo "Job             : ${JOB_ID}"
echo "Dataset         : ${DATANAME}"
echo "Query suite     : ${QUERY_DIR}"
echo "Optimizer steps : ${STEPS}"
echo "Batch size      : ${BATCH_SIZE}"
echo "Masking         : disabled (queries used exactly as written)"
echo "Warm curriculum : ${CURRICULUM_WARMUP_PROBABILITIES} (broad,medium,tight)"
echo "Final curriculum: ${CURRICULUM_FINAL_PROBABILITIES} (broad,medium,tight)"
echo "Output          : tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/${GUIDE_DIR_NAME}"
echo "========================================"

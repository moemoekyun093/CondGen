#!/bin/bash
#SBATCH --job-name=doob_query_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=min-vram:32g,min-cuda-cc:80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
CKPT_DIR="tabdiff/ckpt/${DATANAME}/${MODEL_NAME}"
CKPT_CANDIDATES=("${CKPT_DIR}"/best_ema_model_*.pt)
if [ ! -e "${CKPT_CANDIDATES[0]}" ] || [ "${#CKPT_CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one best EMA checkpoint in ${CKPT_DIR}"
    exit 1
fi
BASE_CKPT="${CKPT_CANDIDATES[0]}"
GUIDE_DIR_ARG="${1:-${GUIDE_DIR_NAME:-doob_query_curriculum_d48_l2_12000}}"
if [[ "${GUIDE_DIR_ARG}" == */* ]]; then
    OUTPUT_DIR="${GUIDE_DIR_ARG}"
else
    OUTPUT_DIR="${CKPT_DIR}/${GUIDE_DIR_ARG}"
fi
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_full}"
STEPS="${STEPS:-12000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
QUERIES_PER_STEP="${QUERIES_PER_STEP:-1}"
LR="${LR:-1e-3}"
D_TOKEN="${D_TOKEN:-48}"
NUM_LAYERS="${NUM_LAYERS:-2}"
N_HEAD="${N_HEAD:-4}"
FACTOR="${FACTOR:-2}"
BOUND_EMBEDDING_DIM="${BOUND_EMBEDDING_DIM:-8}"
ACTIVE_EMBEDDING_DIM="${ACTIVE_EMBEDDING_DIM:-8}"
QUERY_SAMPLING_MODE="${QUERY_SAMPLING_MODE:-curriculum}"
QUERY_SPLIT_MANIFEST="${QUERY_SPLIT_MANIFEST:-}"
QUERY_SPLIT="${QUERY_SPLIT:-}"
CURRICULUM_SELECTIVITY_SOURCE="${CURRICULUM_SELECTIVITY_SOURCE:-target_band}"
CURRICULUM_WARMUP_STEPS="${CURRICULUM_WARMUP_STEPS:-2000}"
CURRICULUM_TRANSITION_STEPS="${CURRICULUM_TRANSITION_STEPS:-4000}"
CURRICULUM_WARMUP_PROBABILITIES="${CURRICULUM_WARMUP_PROBABILITIES:-0.70,0.25,0.05}"
CURRICULUM_FINAL_PROBABILITIES="${CURRICULUM_FINAL_PROBABILITIES:-0.25,0.35,0.40}"
CURRICULUM_REFERENCE_METADATA="${CURRICULUM_REFERENCE_METADATA:-}"
PREDICATE_MASK_MODE="${PREDICATE_MASK_MODE:-full}"
RANDOM_PREDICATE_ACTIVE_PROBABILITY="${RANDOM_PREDICATE_ACTIVE_PROBABILITY:-0.5}"
ALL_ACTIVE_QUERY_PROBABILITY="${ALL_ACTIVE_QUERY_PROBABILITY:-0.1}"
ALL_INACTIVE_QUERY_PROBABILITY="${ALL_INACTIVE_QUERY_PROBABILITY:-0.1}"

echo "========================================"
echo "Dataset          : ${DATANAME}"
echo "Base model       : ${MODEL_NAME}"
echo "Base checkpoint  : ${BASE_CKPT}"
echo "Query suite      : ${QUERY_DIR}"
echo "Output           : ${OUTPUT_DIR}"
echo "Training steps   : ${STEPS}"
echo "Batch size       : ${BATCH_SIZE}"
echo "Queries per step : ${QUERIES_PER_STEP}"
echo "Learning rate    : ${LR}"
echo "Query sampling   : ${QUERY_SAMPLING_MODE}"
if [ -n "${QUERY_SPLIT_MANIFEST}" ]; then
    echo "Query split      : ${QUERY_SPLIT} from ${QUERY_SPLIT_MANIFEST}"
fi
echo "Selectivity source: ${CURRICULUM_SELECTIVITY_SOURCE}"
echo "Curriculum       : warmup=${CURRICULUM_WARMUP_STEPS}, transition=${CURRICULUM_TRANSITION_STEPS}"
echo "Warm probabilities: ${CURRICULUM_WARMUP_PROBABILITIES} (broad,medium,tight)"
echo "Final probabilities: ${CURRICULUM_FINAL_PROBABILITIES} (broad,medium,tight)"
if [ -n "${CURRICULUM_REFERENCE_METADATA}" ]; then
    echo "Curriculum source : ${CURRICULUM_REFERENCE_METADATA} (overrides values above)"
fi
echo "Predicate masks  : ${PREDICATE_MASK_MODE}"
if [ "${PREDICATE_MASK_MODE}" = "mixed" ]; then
    echo "Mask mixture     : all-active=${ALL_ACTIVE_QUERY_PROBABILITY}, all-inactive=${ALL_INACTIVE_QUERY_PROBABILITY}, random remainder"
    echo "Random active p  : ${RANDOM_PREDICATE_ACTIVE_PROBABILITY}"
fi
echo "State tokenizer  : frozen base FT-periodic tokenizer"
echo "Numerical query  : monotone lower/upper embeddings after time fusion"
echo "Categorical query: sum of ReLU-frozen base category lookups"
echo "========================================"
nvidia-smi

REFERENCE_ARGS=()
if [ -n "${CURRICULUM_REFERENCE_METADATA}" ]; then
    REFERENCE_ARGS+=(--curriculum-reference-metadata "${CURRICULUM_REFERENCE_METADATA}")
fi
QUERY_SPLIT_ARGS=()
if [ -n "${QUERY_SPLIT_MANIFEST}" ]; then
    QUERY_SPLIT_ARGS+=(
        --query-split-manifest "${QUERY_SPLIT_MANIFEST}"
        --query-split "${QUERY_SPLIT}"
    )
fi

python -u train_doob_query_suite.py \
    --dataname "${DATANAME}" \
    --base-ckpt "${BASE_CKPT}" \
    --query-dir "${QUERY_DIR}" \
    "${QUERY_SPLIT_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --queries-per-step "${QUERIES_PER_STEP}" \
    --lr "${LR}" \
    --d-token "${D_TOKEN}" \
    --num-layers "${NUM_LAYERS}" \
    --n-head "${N_HEAD}" \
    --factor "${FACTOR}" \
    --bound-embedding-dim "${BOUND_EMBEDDING_DIM}" \
    --active-embedding-dim "${ACTIVE_EMBEDDING_DIM}" \
    --query-sampling-mode "${QUERY_SAMPLING_MODE}" \
    --curriculum-selectivity-source "${CURRICULUM_SELECTIVITY_SOURCE}" \
    --curriculum-warmup-steps "${CURRICULUM_WARMUP_STEPS}" \
    --curriculum-transition-steps "${CURRICULUM_TRANSITION_STEPS}" \
    --curriculum-warmup-probabilities "${CURRICULUM_WARMUP_PROBABILITIES}" \
    --curriculum-final-probabilities "${CURRICULUM_FINAL_PROBABILITIES}" \
    "${REFERENCE_ARGS[@]}" \
    --predicate-mask-mode "${PREDICATE_MASK_MODE}" \
    --random-predicate-active-probability "${RANDOM_PREDICATE_ACTIVE_PROBABILITY}" \
    --all-active-query-probability "${ALL_ACTIVE_QUERY_PROBABILITY}" \
    --all-inactive-query-probability "${ALL_INACTIVE_QUERY_PROBABILITY}" \
    --checkpoint-warmup 200 \
    --device cuda

echo "Finished structured-query Doob training"

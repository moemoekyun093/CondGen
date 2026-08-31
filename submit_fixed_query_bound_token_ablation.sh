#!/bin/bash
# Compare ordinary per-column MLP fusion with two constraint-token parameterizations.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
QUERY_ID="${QUERY_ID:-q_shoppers_b00p5_k01_num_1}"
STEPS=2000
CHECKPOINT_EVERY=200
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MLP_GUIDE_DIR="${MLP_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_mlp_bounds_d48_l2_2000}"
ENDPOINT_GUIDE_DIR="${ENDPOINT_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_bound_tokens_lu_d48_l4_2000}"
CENTER_WIDTH_GUIDE_DIR="${CENTER_WIDTH_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_bound_tokens_center_logwidth_d48_l4_2000}"
TOKEN_SAMPLE_ROOT="${TOKEN_SAMPLE_ROOT:-conditional_samples/${DATANAME}/fixed_query_bound_token_ablation}"
MLP_SAMPLE_DIR="${MLP_SAMPLE_DIR:-conditional_samples/${DATANAME}/fixed_query_bound_ablation/mlp}"
ENDPOINT_SAMPLE_DIR="${ENDPOINT_SAMPLE_DIR:-${TOKEN_SAMPLE_ROOT}/endpoints}"
CENTER_WIDTH_SAMPLE_DIR="${CENTER_WIDTH_SAMPLE_DIR:-${TOKEN_SAMPLE_ROOT}/center_logwidth}"
TOKEN_EVAL_DIR="${TOKEN_EVAL_DIR:-evaluations/${DATANAME}/fixed_query_bound_token_ablation}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-50}"

if [ ! -f "${QUERY_DIR}/${QUERY_ID}.json" ]; then
    echo "ERROR: query not found: ${QUERY_DIR}/${QUERY_ID}.json"
    exit 1
fi
mkdir -p logs evaluations/slurm "${TOKEN_SAMPLE_ROOT}" "${TOKEN_EVAL_DIR}"

export DATANAME MODEL_NAME QUERY_DIR QUERY_ID STEPS CHECKPOINT_EVERY BATCH_SIZE
export NUM_SAMPLES MLP_GUIDE_DIR ENDPOINT_GUIDE_DIR CENTER_WIDTH_GUIDE_DIR TOKEN_SAMPLE_ROOT
export MLP_SAMPLE_DIR ENDPOINT_SAMPLE_DIR CENTER_WIDTH_SAMPLE_DIR
export TOKEN_EVAL_DIR HISTOGRAM_BINS
export QUERIES_PER_STEP=1 LR=1e-3 D_TOKEN=48 NUM_LAYERS=4 N_HEAD=4 FACTOR=2
export QUERY_SAMPLING_MODE=uniform PREDICATE_MASK_MODE=full
export QUERY_ARCHITECTURE=alternating_cross_attention
export QUERY_SPLIT_MANIFEST="" QUERY_SPLIT=""

final_checkpoint_complete() {
    local DIRECTORY="$1"
    [ -f "${DIRECTORY}/guide_2000.pt" ]
}

final_sample_complete() {
    local DIRECTORY="$1"
    [ -f "${DIRECTORY}/step_2000.csv" ]
}

TRAIN_DEPENDENCIES=()
if final_sample_complete "${MLP_SAMPLE_DIR}"; then
    MLP_TRAIN="not needed; reused step-2000 sample"
elif final_checkpoint_complete "${MLP_GUIDE_DIR}"; then
    MLP_TRAIN="reused step-2000 checkpoint"
else
    SUBMISSION=$(sbatch --parsable \
        --export=ALL,QUERY_ARCHITECTURE=per_token_fusion,BOUND_EMBEDDING_MODE=mlp,NUM_LAYERS=2 \
        doob_query_train.sh "${MLP_GUIDE_DIR}")
    MLP_TRAIN="${SUBMISSION%%;*}"
    TRAIN_DEPENDENCIES+=("${MLP_TRAIN}")
fi
if final_sample_complete "${ENDPOINT_SAMPLE_DIR}"; then
    ENDPOINT_TRAIN="not needed; reused step-2000 sample"
elif final_checkpoint_complete "${ENDPOINT_GUIDE_DIR}"; then
    ENDPOINT_TRAIN="reused step-2000 checkpoint"
else
    SUBMISSION=$(sbatch --parsable \
        --export=ALL,QUERY_ARCHITECTURE=alternating_cross_attention,BOUND_TOKEN_PARAMETERIZATION=endpoints,NUM_LAYERS=4 \
        doob_query_train.sh "${ENDPOINT_GUIDE_DIR}")
    ENDPOINT_TRAIN="${SUBMISSION%%;*}"
    TRAIN_DEPENDENCIES+=("${ENDPOINT_TRAIN}")
fi
if final_sample_complete "${CENTER_WIDTH_SAMPLE_DIR}"; then
    CENTER_WIDTH_TRAIN="not needed; reused step-2000 sample"
elif final_checkpoint_complete "${CENTER_WIDTH_GUIDE_DIR}"; then
    CENTER_WIDTH_TRAIN="reused step-2000 checkpoint"
else
    SUBMISSION=$(sbatch --parsable \
        --export=ALL,QUERY_ARCHITECTURE=alternating_cross_attention,BOUND_TOKEN_PARAMETERIZATION=center_logwidth,NUM_LAYERS=4 \
        doob_query_train.sh "${CENTER_WIDTH_GUIDE_DIR}")
    CENTER_WIDTH_TRAIN="${SUBMISSION%%;*}"
    TRAIN_DEPENDENCIES+=("${CENTER_WIDTH_TRAIN}")
fi

MISSING_TASKS=()
for TASK in 9 19 29; do
    STEP=$(((TASK % 10 + 1) * 200))
    case "$((TASK / 10))" in
        0) SAMPLE_DIR="${MLP_SAMPLE_DIR}" ;;
        1) SAMPLE_DIR="${ENDPOINT_SAMPLE_DIR}" ;;
        2) SAMPLE_DIR="${CENTER_WIDTH_SAMPLE_DIR}" ;;
    esac
    OUTPUT="${SAMPLE_DIR}/step_$(printf '%04d' "${STEP}").csv"
    if [ ! -f "${OUTPUT}" ]; then
        MISSING_TASKS+=("${TASK}")
    fi
done

SAMPLE_DEPENDENCY=()
if [ "${#TRAIN_DEPENDENCIES[@]}" -gt 0 ]; then
    TEXT=$(IFS=:; echo "${TRAIN_DEPENDENCIES[*]}")
    SAMPLE_DEPENDENCY+=(--dependency="afterok:${TEXT}")
fi
if [ "${#MISSING_TASKS[@]}" -gt 0 ]; then
    MISSING_TASKS_CSV=$(IFS=,; echo "${MISSING_TASKS[*]}")
    export MISSING_TASKS_CSV
    SUBMISSION=$(sbatch --parsable "${SAMPLE_DEPENDENCY[@]}" \
        sample_fixed_query_bound_token_checkpoints.sh)
    SAMPLE_JOB="${SUBMISSION%%;*}"
    PLOT_DEPENDENCY=(--dependency="afterok:${SAMPLE_JOB}")
else
    SAMPLE_JOB="not submitted; reused all step-2000 samples"
    PLOT_DEPENDENCY=()
fi

SUBMISSION=$(sbatch --parsable "${PLOT_DEPENDENCY[@]}" \
    plot_fixed_query_bound_token_ablation.sh)
PLOT_JOB="${SUBMISSION%%;*}"

echo "========================================"
echo "Fixed-query bound-token ablation submitted"
echo "Query             : ${QUERY_ID}"
echo "Baseline          : ordinary MLP per-column fusion, d48/L2"
echo "Token models      : alternating self/cross attention, d48/L4"
echo "Training          : 2000 optimizer steps per guide"
echo "Sampling          : step 2000 only (three methods, one bundled job)"
echo "Ordinary MLP train: ${MLP_TRAIN}"
echo "Lower/upper train : ${ENDPOINT_TRAIN}"
echo "Center/log-w train: ${CENTER_WIDTH_TRAIN}"
echo "Bundled sampling  : ${SAMPLE_JOB}"
echo "Plot              : ${PLOT_JOB}"
echo "Samples           : ${TOKEN_SAMPLE_ROOT}"
echo "MLP samples       : ${MLP_SAMPLE_DIR}"
echo "Evaluation        : ${TOKEN_EVAL_DIR}"
echo "========================================"

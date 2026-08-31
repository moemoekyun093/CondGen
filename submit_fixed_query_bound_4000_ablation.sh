#!/bin/bash
# Continue three step-2000 guides, train the new MLP center/log-width guide,
# then sample and plot all four methods at total training step 4000.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
QUERY_ID="${QUERY_ID:-q_shoppers_b00p5_k01_num_1}"
STEPS=4000
SAMPLE_STEP=4000
CHECKPOINT_EVERY=2000
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"

MLP_SOURCE_DIR="${MLP_SOURCE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_mlp_bounds_d48_l2_2000}"
ENDPOINT_SOURCE_DIR="${ENDPOINT_SOURCE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_bound_tokens_lu_d48_l4_2000}"
CENTER_WIDTH_SOURCE_DIR="${CENTER_WIDTH_SOURCE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_bound_tokens_center_logwidth_d48_l4_2000}"

MLP_GUIDE_DIR="${MLP_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_mlp_bounds_d48_l2_4000}"
MLP_CENTER_WIDTH_GUIDE_DIR="${MLP_CENTER_WIDTH_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_mlp_center_logwidth_d48_l2_4000}"
ENDPOINT_GUIDE_DIR="${ENDPOINT_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_bound_tokens_lu_d48_l4_4000}"
CENTER_WIDTH_GUIDE_DIR="${CENTER_WIDTH_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_bound_tokens_center_logwidth_d48_l4_4000}"

SAMPLE_ROOT="${SAMPLE_ROOT:-conditional_samples/${DATANAME}/fixed_query_bound_4000_ablation}"
MLP_SAMPLE_DIR="${MLP_SAMPLE_DIR:-${SAMPLE_ROOT}/ordinary_mlp_endpoints}"
MLP_CENTER_WIDTH_SAMPLE_DIR="${MLP_CENTER_WIDTH_SAMPLE_DIR:-${SAMPLE_ROOT}/ordinary_mlp_center_logwidth}"
ENDPOINT_SAMPLE_DIR="${ENDPOINT_SAMPLE_DIR:-${SAMPLE_ROOT}/lower_upper_tokens}"
CENTER_WIDTH_SAMPLE_DIR="${CENTER_WIDTH_SAMPLE_DIR:-${SAMPLE_ROOT}/center_logwidth_tokens}"
TOKEN_EVAL_DIR="${TOKEN_EVAL_DIR:-evaluations/${DATANAME}/fixed_query_bound_4000_ablation}"
SWEEP_SAMPLE_ROOT="${SWEEP_SAMPLE_ROOT:-conditional_samples/${DATANAME}/fixed_query_center_logwidth_mlp_sampling_sweeps}"
SWEEP_EVAL_DIR="${SWEEP_EVAL_DIR:-evaluations/${DATANAME}/fixed_query_center_logwidth_mlp_sampling_sweeps}"
REVERSE_STEPS="${REVERSE_STEPS:-50,75,100,150,200}"
GUIDANCE_STRENGTHS="${GUIDANCE_STRENGTHS:-1,2,5}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-50}"

if [ ! -f "${QUERY_DIR}/${QUERY_ID}.json" ]; then
    echo "ERROR: query not found: ${QUERY_DIR}/${QUERY_ID}.json"
    exit 1
fi
mkdir -p logs evaluations/slurm "${SAMPLE_ROOT}" "${TOKEN_EVAL_DIR}"

export DATANAME MODEL_NAME QUERY_DIR QUERY_ID STEPS SAMPLE_STEP CHECKPOINT_EVERY
export BATCH_SIZE NUM_SAMPLES HISTOGRAM_BINS TOKEN_EVAL_DIR
export SWEEP_SAMPLE_ROOT SWEEP_EVAL_DIR REVERSE_STEPS GUIDANCE_STRENGTHS
export MLP_GUIDE_DIR MLP_CENTER_WIDTH_GUIDE_DIR ENDPOINT_GUIDE_DIR CENTER_WIDTH_GUIDE_DIR
export MLP_SAMPLE_DIR MLP_CENTER_WIDTH_SAMPLE_DIR ENDPOINT_SAMPLE_DIR CENTER_WIDTH_SAMPLE_DIR
export QUERIES_PER_STEP=1 LR=1e-3 D_TOKEN=48 N_HEAD=4 FACTOR=2
export QUERY_SAMPLING_MODE=uniform PREDICATE_MASK_MODE=full
export QUERY_PRESENCE_MODE=active_flags QUERY_SPLIT_MANIFEST="" QUERY_SPLIT=""

TRAIN_DEPENDENCIES=()

submit_continuation() {
    local SAMPLE_DIR="$1" OUTPUT_DIR="$2" SOURCE_DIR="$3" ARCH="$4" PARAM="$5" LAYERS="$6"
    local LABEL="$7" SUBMISSION JOB
    if [ -f "${SAMPLE_DIR}/step_4000.csv" ]; then
        printf -v "${LABEL}" '%s' "not needed; reused step-4000 sample"
    elif [ -f "${OUTPUT_DIR}/guide_4000.pt" ]; then
        printf -v "${LABEL}" '%s' "reused step-4000 checkpoint"
    else
        if [ ! -f "${SOURCE_DIR}/guide_2000.pt" ]; then
            echo "ERROR: continuation checkpoint missing: ${SOURCE_DIR}/guide_2000.pt"
            exit 1
        fi
        SUBMISSION=$(sbatch --parsable \
            --export=ALL,QUERY_ARCHITECTURE="${ARCH}",BOUND_EMBEDDING_MODE=mlp,BOUND_TOKEN_PARAMETERIZATION="${PARAM}",NUM_LAYERS="${LAYERS}",RESUME_FROM="${SOURCE_DIR}/guide_2000.pt" \
            doob_query_train.sh "${OUTPUT_DIR}")
        JOB="${SUBMISSION%%;*}"
        printf -v "${LABEL}" '%s' "${JOB} (continued from step 2000)"
        TRAIN_DEPENDENCIES+=("${JOB}")
    fi
}

submit_continuation "${MLP_SAMPLE_DIR}" "${MLP_GUIDE_DIR}" "${MLP_SOURCE_DIR}" \
    per_token_fusion endpoints 2 MLP_TRAIN
submit_continuation "${ENDPOINT_SAMPLE_DIR}" "${ENDPOINT_GUIDE_DIR}" "${ENDPOINT_SOURCE_DIR}" \
    alternating_cross_attention endpoints 4 ENDPOINT_TRAIN
submit_continuation "${CENTER_WIDTH_SAMPLE_DIR}" "${CENTER_WIDTH_GUIDE_DIR}" "${CENTER_WIDTH_SOURCE_DIR}" \
    alternating_cross_attention center_logwidth 4 CENTER_WIDTH_TRAIN

if [ -f "${MLP_CENTER_WIDTH_SAMPLE_DIR}/step_4000.csv" ]; then
    MLP_CENTER_WIDTH_TRAIN="not needed; reused step-4000 sample"
elif [ -f "${MLP_CENTER_WIDTH_GUIDE_DIR}/guide_4000.pt" ]; then
    MLP_CENTER_WIDTH_TRAIN="reused step-4000 checkpoint"
else
    SUBMISSION=$(sbatch --parsable \
        --export=ALL,QUERY_ARCHITECTURE=per_token_fusion,BOUND_EMBEDDING_MODE=mlp,BOUND_TOKEN_PARAMETERIZATION=center_logwidth,NUM_LAYERS=2,RESUME_FROM= \
        doob_query_train.sh "${MLP_CENTER_WIDTH_GUIDE_DIR}")
    MLP_CENTER_WIDTH_TRAIN="${SUBMISSION%%;*} (new model, step 0 to 4000)"
    TRAIN_DEPENDENCIES+=("${SUBMISSION%%;*}")
fi

MISSING_TASKS=()
[ -f "${MLP_SAMPLE_DIR}/step_4000.csv" ] || MISSING_TASKS+=(0)
[ -f "${ENDPOINT_SAMPLE_DIR}/step_4000.csv" ] || MISSING_TASKS+=(1)
[ -f "${MLP_CENTER_WIDTH_SAMPLE_DIR}/step_4000.csv" ] || MISSING_TASKS+=(2)
[ -f "${CENTER_WIDTH_SAMPLE_DIR}/step_4000.csv" ] || MISSING_TASKS+=(3)

SAMPLE_DEPENDENCY=()
if [ "${#TRAIN_DEPENDENCIES[@]}" -gt 0 ]; then
    TEXT=$(IFS=:; echo "${TRAIN_DEPENDENCIES[*]}")
    SAMPLE_DEPENDENCY+=(--dependency="afterok:${TEXT}")
fi
if [ "${#MISSING_TASKS[@]}" -gt 0 ]; then
    MISSING_TASKS_CSV=$(IFS=,; echo "${MISSING_TASKS[*]}")
else
    MISSING_TASKS_CSV=""
fi
SWEEP_METHOD_DIR="${SWEEP_SAMPLE_ROOT}/ordinary_mlp_center_logwidth"
SWEEP_MISSING=0
IFS=',' read -r -a REVERSE_STEP_VALUES <<< "${REVERSE_STEPS}"
IFS=',' read -r -a LAMBDA_VALUES <<< "${GUIDANCE_STRENGTHS}"
for NUM_STEPS in "${REVERSE_STEP_VALUES[@]}"; do
    [ -f "${SWEEP_METHOD_DIR}/steps_$(printf '%03d' "${NUM_STEPS}")_lambda_1.csv" ] \
        || SWEEP_MISSING=1
done
for LAMBDA in "${LAMBDA_VALUES[@]}"; do
    LAMBDA_TAG=$(printf '%g' "${LAMBDA}")
    [ -f "${SWEEP_METHOD_DIR}/steps_050_lambda_${LAMBDA_TAG}.csv" ] \
        || SWEEP_MISSING=1
done
export MISSING_TASKS_CSV
if [ "${#MISSING_TASKS[@]}" -gt 0 ] || [ "${SWEEP_MISSING}" -eq 1 ]; then
    SUBMISSION=$(sbatch --parsable "${SAMPLE_DEPENDENCY[@]}" \
        sample_fixed_query_sampling_sweeps.sh)
    SAMPLE_JOB="${SUBMISSION%%;*}"
    PLOT_DEPENDENCY=(--dependency="afterok:${SAMPLE_JOB}")
else
    SAMPLE_JOB="not submitted; reused standard and sweep samples"
    PLOT_DEPENDENCY=()
fi

SUBMISSION=$(sbatch --parsable "${PLOT_DEPENDENCY[@]}" \
    plot_fixed_query_bound_4000_ablation.sh)
PLOT_JOB="${SUBMISSION%%;*}"
SUBMISSION=$(sbatch --parsable "${PLOT_DEPENDENCY[@]}" \
    plot_fixed_query_sampling_sweeps.sh)
SWEEP_PLOT_JOB="${SUBMISSION%%;*}"

echo "========================================"
echo "Fixed-query 4000-step bound ablation submitted"
echo "Query                    : ${QUERY_ID}"
echo "MLP lower/upper          : ${MLP_TRAIN}"
echo "MLP center/log-width     : ${MLP_CENTER_WIDTH_TRAIN}"
echo "Token lower/upper        : ${ENDPOINT_TRAIN}"
echo "Token center/log-width   : ${CENTER_WIDTH_TRAIN}"
echo "Bundled all sampling     : ${SAMPLE_JOB}"
echo "Plot                     : ${PLOT_JOB}"
echo "Center/log-w sweep plot  : ${SWEEP_PLOT_JOB}"
echo "Samples                  : ${SAMPLE_ROOT}"
echo "Evaluation               : ${TOKEN_EVAL_DIR}"
echo "Sampling sweep           : ${SWEEP_EVAL_DIR}"
echo "========================================"

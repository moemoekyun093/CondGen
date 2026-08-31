#!/bin/bash
# Compare explicit active flags with implicit wide-domain query encoding.

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
INACTIVE_NUMERICAL_BOUND="${INACTIVE_NUMERICAL_BOUND:-10.0}"
ACTIVE_FLAGS_GUIDE_DIR="${ACTIVE_FLAGS_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_active_flags_d48_l2_2000}"
IMPLICIT_DOMAIN_GUIDE_DIR="${IMPLICIT_DOMAIN_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_implicit_domain_d48_l2_2000}"
ABLATION_SAMPLE_ROOT="${ABLATION_SAMPLE_ROOT:-conditional_samples/${DATANAME}/fixed_query_presence_ablation}"
ABLATION_EVAL_DIR="${ABLATION_EVAL_DIR:-evaluations/${DATANAME}/fixed_query_presence_ablation}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-50}"

for REQUIRED in "${QUERY_DIR}/${QUERY_ID}.json"; do
    if [ ! -f "${REQUIRED}" ]; then
        echo "ERROR: required file not found: ${REQUIRED}"
        exit 1
    fi
done
mkdir -p logs evaluations/slurm "${ABLATION_SAMPLE_ROOT}" "${ABLATION_EVAL_DIR}"

export DATANAME MODEL_NAME QUERY_DIR QUERY_ID STEPS CHECKPOINT_EVERY BATCH_SIZE
export NUM_SAMPLES INACTIVE_NUMERICAL_BOUND ACTIVE_FLAGS_GUIDE_DIR
export IMPLICIT_DOMAIN_GUIDE_DIR ABLATION_SAMPLE_ROOT ABLATION_EVAL_DIR HISTOGRAM_BINS
export QUERIES_PER_STEP=1 LR=1e-3 D_TOKEN=48 NUM_LAYERS=2 N_HEAD=4 FACTOR=2
export QUERY_SAMPLING_MODE=uniform PREDICATE_MASK_MODE=full
export QUERY_SPLIT_MANIFEST="" QUERY_SPLIT=""

TRAIN_DEPENDENCIES=()
checkpoint_series_complete() {
    local DIRECTORY="$1"
    local STEP
    for STEP in $(seq 200 200 2000); do
        if [ ! -f "${DIRECTORY}/guide_${STEP}.pt" ]; then
            return 1
        fi
    done
    return 0
}

if checkpoint_series_complete "${ACTIVE_FLAGS_GUIDE_DIR}"; then
    ACTIVE_TRAIN="reused complete checkpoints"
else
    ACTIVE_SUBMISSION=$(sbatch --parsable \
        --export=ALL,QUERY_PRESENCE_MODE=active_flags \
        doob_query_train.sh "${ACTIVE_FLAGS_GUIDE_DIR}")
    ACTIVE_TRAIN="${ACTIVE_SUBMISSION%%;*}"
    TRAIN_DEPENDENCIES+=("${ACTIVE_TRAIN}")
fi
if checkpoint_series_complete "${IMPLICIT_DOMAIN_GUIDE_DIR}"; then
    IMPLICIT_TRAIN="reused complete checkpoints"
else
    IMPLICIT_SUBMISSION=$(sbatch --parsable \
        --export=ALL,QUERY_PRESENCE_MODE=implicit_domain \
        doob_query_train.sh "${IMPLICIT_DOMAIN_GUIDE_DIR}")
    IMPLICIT_TRAIN="${IMPLICIT_SUBMISSION%%;*}"
    TRAIN_DEPENDENCIES+=("${IMPLICIT_TRAIN}")
fi

MISSING_TASKS=()
for TASK in $(seq 0 19); do
    STEP=$(((TASK % 10 + 1) * 200))
    if [ "$((TASK / 10))" -eq 0 ]; then
        LABEL=active_flags
    else
        LABEL=implicit_domain
    fi
    OUTPUT="${ABLATION_SAMPLE_ROOT}/${LABEL}/step_$(printf '%04d' "${STEP}").csv"
    if [ ! -f "${OUTPUT}" ] || [ ! -f "${OUTPUT%.csv}.constraints.json" ]; then
        MISSING_TASKS+=("${TASK}")
    fi
done

SAMPLE_DEPENDENCY=()
if [ "${#TRAIN_DEPENDENCIES[@]}" -gt 0 ]; then
    TRAIN_DEPENDENCY_TEXT=$(IFS=:; echo "${TRAIN_DEPENDENCIES[*]}")
    SAMPLE_DEPENDENCY+=(--dependency="afterok:${TRAIN_DEPENDENCY_TEXT}")
fi
if [ "${#MISSING_TASKS[@]}" -gt 0 ]; then
    MISSING_TASKS_CSV=$(IFS=,; echo "${MISSING_TASKS[*]}")
    export MISSING_TASKS_CSV
    SAMPLE_SUBMISSION=$(sbatch --parsable "${SAMPLE_DEPENDENCY[@]}" \
        sample_fixed_query_presence_checkpoints.sh)
    SAMPLE_JOB="${SAMPLE_SUBMISSION%%;*}"
    PLOT_DEPENDENCY=(--dependency="afterok:${SAMPLE_JOB}")
else
    SAMPLE_JOB="reused all checkpoint samples"
    PLOT_DEPENDENCY=()
fi

PLOT_SUBMISSION=$(sbatch --parsable "${PLOT_DEPENDENCY[@]}" \
    plot_fixed_query_presence_ablation.sh)
PLOT_JOB="${PLOT_SUBMISSION%%;*}"

echo "========================================"
echo "Fixed-query query-presence ablation submitted"
echo "Query             : ${QUERY_ID}"
echo "Training          : 2000 optimizer steps per guide"
echo "Snapshots         : every 200 steps"
echo "Active-flag train : ${ACTIVE_TRAIN}"
echo "Implicit train    : ${IMPLICIT_TRAIN}"
echo "Implicit num box  : [-${INACTIVE_NUMERICAL_BOUND}, +${INACTIVE_NUMERICAL_BOUND}]"
echo "Sampling          : ${SAMPLE_JOB}"
echo "Plot              : ${PLOT_JOB}"
echo "Samples           : ${ABLATION_SAMPLE_ROOT}"
echo "Evaluation        : ${ABLATION_EVAL_DIR}"
echo "========================================"

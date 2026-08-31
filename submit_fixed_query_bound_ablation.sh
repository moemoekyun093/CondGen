#!/bin/bash
# Compare monotone and ordinary MLP numerical endpoint embeddings.

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
MONOTONE_GUIDE_DIR="${MONOTONE_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_monotone_d48_l2_2000}"
MLP_GUIDE_DIR="${MLP_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_mlp_bounds_d48_l2_2000}"
BOUND_SAMPLE_ROOT="${BOUND_SAMPLE_ROOT:-conditional_samples/${DATANAME}/fixed_query_bound_ablation}"
BOUND_EVAL_DIR="${BOUND_EVAL_DIR:-evaluations/${DATANAME}/fixed_query_bound_ablation}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-50}"

if [ ! -f "${QUERY_DIR}/${QUERY_ID}.json" ]; then
    echo "ERROR: query not found: ${QUERY_DIR}/${QUERY_ID}.json"
    exit 1
fi
mkdir -p logs evaluations/slurm "${BOUND_SAMPLE_ROOT}" "${BOUND_EVAL_DIR}"

export DATANAME MODEL_NAME QUERY_DIR QUERY_ID STEPS CHECKPOINT_EVERY BATCH_SIZE
export NUM_SAMPLES MONOTONE_GUIDE_DIR MLP_GUIDE_DIR BOUND_SAMPLE_ROOT
export BOUND_EVAL_DIR HISTOGRAM_BINS
export QUERIES_PER_STEP=1 LR=1e-3 D_TOKEN=48 NUM_LAYERS=2 N_HEAD=4 FACTOR=2
export QUERY_SAMPLING_MODE=uniform PREDICATE_MASK_MODE=full
export QUERY_PRESENCE_MODE=active_flags
export QUERY_SPLIT_MANIFEST="" QUERY_SPLIT=""

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

TRAIN_DEPENDENCIES=()
if checkpoint_series_complete "${MONOTONE_GUIDE_DIR}"; then
    MONOTONE_TRAIN="reused complete checkpoints"
else
    SUBMISSION=$(sbatch --parsable \
        --export=ALL,BOUND_EMBEDDING_MODE=monotone \
        doob_query_train.sh "${MONOTONE_GUIDE_DIR}")
    MONOTONE_TRAIN="${SUBMISSION%%;*}"
    TRAIN_DEPENDENCIES+=("${MONOTONE_TRAIN}")
fi
if checkpoint_series_complete "${MLP_GUIDE_DIR}"; then
    MLP_TRAIN="reused complete checkpoints"
else
    SUBMISSION=$(sbatch --parsable \
        --export=ALL,BOUND_EMBEDDING_MODE=mlp \
        doob_query_train.sh "${MLP_GUIDE_DIR}")
    MLP_TRAIN="${SUBMISSION%%;*}"
    TRAIN_DEPENDENCIES+=("${MLP_TRAIN}")
fi

MISSING_TASKS=()
for TASK in $(seq 0 19); do
    STEP=$(((TASK % 10 + 1) * 200))
    if [ "$((TASK / 10))" -eq 0 ]; then LABEL=monotone; else LABEL=mlp; fi
    OUTPUT="${BOUND_SAMPLE_ROOT}/${LABEL}/step_$(printf '%04d' "${STEP}").csv"
    if [ ! -f "${OUTPUT}" ] || [ ! -f "${OUTPUT%.csv}.constraints.json" ]; then
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
        sample_fixed_query_bound_checkpoints.sh)
    SAMPLE_JOB="${SUBMISSION%%;*}"
    PLOT_DEPENDENCY=(--dependency="afterok:${SAMPLE_JOB}")
else
    SAMPLE_JOB="reused all checkpoint samples"
    PLOT_DEPENDENCY=()
fi

SUBMISSION=$(sbatch --parsable "${PLOT_DEPENDENCY[@]}" \
    plot_fixed_query_bound_ablation.sh)
PLOT_JOB="${SUBMISSION%%;*}"

echo "========================================"
echo "Fixed-query bound-embedding ablation submitted"
echo "Query            : ${QUERY_ID}"
echo "Shared setup     : active flags, d48/L2, 2000 steps"
echo "Snapshots        : every 200 steps"
echo "Monotone train   : ${MONOTONE_TRAIN}"
echo "Ordinary MLP     : ${MLP_TRAIN}"
echo "Bundled sampling : ${SAMPLE_JOB}"
echo "Plot             : ${PLOT_JOB}"
echo "Samples          : ${BOUND_SAMPLE_ROOT}"
echo "Evaluation       : ${BOUND_EVAL_DIR}"
echo "========================================"

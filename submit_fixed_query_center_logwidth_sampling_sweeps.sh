#!/bin/bash
# Sampling-only reverse-step and lambda sweeps. This script never trains models.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
QUERY_ID="${QUERY_ID:-q_shoppers_b00p5_k01_num_1}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MLP_CENTER_WIDTH_GUIDE_DIR="${MLP_CENTER_WIDTH_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_mlp_center_logwidth_d48_l2_4000}"
SWEEP_SAMPLE_ROOT="${SWEEP_SAMPLE_ROOT:-conditional_samples/${DATANAME}/fixed_query_center_logwidth_mlp_sampling_sweeps}"
SWEEP_EVAL_DIR="${SWEEP_EVAL_DIR:-evaluations/${DATANAME}/fixed_query_center_logwidth_mlp_sampling_sweeps}"
REVERSE_STEPS="${REVERSE_STEPS:-50,75,100,150,200}"
GUIDANCE_STRENGTHS="${GUIDANCE_STRENGTHS:-1,2,5}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-50}"

GUIDE_CKPT="${MLP_CENTER_WIDTH_GUIDE_DIR}/guide_4000.pt"
if [ ! -f "${GUIDE_CKPT}" ]; then
    echo "ERROR: trained center/log-width MLP checkpoint is not ready:"
    echo "  ${GUIDE_CKPT}"
    echo "This sampling-only script will not submit training."
    exit 1
fi
if [ ! -f "${QUERY_DIR}/${QUERY_ID}.json" ]; then
    echo "ERROR: query not found: ${QUERY_DIR}/${QUERY_ID}.json"
    exit 1
fi

mkdir -p logs evaluations/slurm "${SWEEP_SAMPLE_ROOT}" "${SWEEP_EVAL_DIR}"
export DATANAME MODEL_NAME QUERY_DIR QUERY_ID NUM_SAMPLES
export MLP_CENTER_WIDTH_GUIDE_DIR SWEEP_SAMPLE_ROOT SWEEP_EVAL_DIR
export REVERSE_STEPS GUIDANCE_STRENGTHS HISTOGRAM_BINS
export MISSING_TASKS_CSV=""

METHOD_DIR="${SWEEP_SAMPLE_ROOT}/ordinary_mlp_center_logwidth"
MISSING=0
IFS=',' read -r -a REVERSE_STEP_VALUES <<< "${REVERSE_STEPS}"
IFS=',' read -r -a LAMBDA_VALUES <<< "${GUIDANCE_STRENGTHS}"
for NUM_STEPS in "${REVERSE_STEP_VALUES[@]}"; do
    [ -f "${METHOD_DIR}/steps_$(printf '%03d' "${NUM_STEPS}")_lambda_1.csv" ] \
        || MISSING=1
done
for LAMBDA in "${LAMBDA_VALUES[@]}"; do
    LAMBDA_TAG=$(printf '%g' "${LAMBDA}")
    [ -f "${METHOD_DIR}/steps_050_lambda_${LAMBDA_TAG}.csv" ] || MISSING=1
done

if [ "${MISSING}" -eq 1 ]; then
    SUBMISSION=$(sbatch --parsable sample_fixed_query_sampling_sweeps.sh)
    SAMPLE_JOB="${SUBMISSION%%;*}"
    PLOT_DEPENDENCY=(--dependency="afterok:${SAMPLE_JOB}")
else
    SAMPLE_JOB="not submitted; reused all sweep samples"
    PLOT_DEPENDENCY=()
fi

SUBMISSION=$(sbatch --parsable "${PLOT_DEPENDENCY[@]}" \
    plot_fixed_query_sampling_sweeps.sh)
PLOT_JOB="${SUBMISSION%%;*}"

echo "========================================"
echo "Center/log-width MLP sampling sweep submitted"
echo "Training submitted : no"
echo "Checkpoint         : ${GUIDE_CKPT}"
echo "Reverse steps      : ${REVERSE_STEPS} at lambda=1"
echo "Lambda sweep       : ${GUIDANCE_STRENGTHS} at 50 steps"
echo "Bundled sampling   : ${SAMPLE_JOB}"
echo "Plot               : ${PLOT_JOB}"
echo "Evaluation         : ${SWEEP_EVAL_DIR}"
echo "========================================"

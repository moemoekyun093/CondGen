#!/bin/bash
# Sampling-only five-seed diagnostic for the ordinary MLP center/log-width guide.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
QUERY_ID="${QUERY_ID:-q_shoppers_b00p5_k01_num_1}"
QUERY_FILE="${QUERY_FILE:-data90/${DATANAME}/queries/${QUERY_ID}.json}"
MLP_CENTER_WIDTH_GUIDE_DIR="${MLP_CENTER_WIDTH_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_fixed_exit_mlp_center_logwidth_d48_l2_4000}"
UNCONDITIONAL_SAMPLES="${UNCONDITIONAL_SAMPLES:-tabdiff/result/${DATANAME}/${MODEL_NAME}/8000/samples.csv}"
EXISTING_SEED_74000_SAMPLE="${EXISTING_SEED_74000_SAMPLE:-conditional_samples/${DATANAME}/fixed_query_center_logwidth_mlp_sampling_sweeps/ordinary_mlp_center_logwidth/steps_050_lambda_1.csv}"
MULTISEED_SAMPLE_DIR="${MULTISEED_SAMPLE_DIR:-conditional_samples/${DATANAME}/fixed_query_center_logwidth_mlp_multiseed}"
MULTISEED_EVAL_DIR="${MULTISEED_EVAL_DIR:-evaluations/${DATANAME}/fixed_query_center_logwidth_mlp_multiseed}"
SEEDS="${SEEDS:-74000,74001,74002,74003,74004}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-50}"

for REQUIRED in "${MLP_CENTER_WIDTH_GUIDE_DIR}/guide_4000.pt" "${QUERY_FILE}" "${UNCONDITIONAL_SAMPLES}"; do
    if [ ! -f "${REQUIRED}" ]; then
        echo "ERROR: required existing file not found: ${REQUIRED}"
        echo "This sampling-only workflow will not train models."
        exit 1
    fi
done
mkdir -p logs evaluations/slurm "${MULTISEED_SAMPLE_DIR}" "${MULTISEED_EVAL_DIR}"
export DATANAME MODEL_NAME QUERY_FILE MLP_CENTER_WIDTH_GUIDE_DIR
export UNCONDITIONAL_SAMPLES EXISTING_SEED_74000_SAMPLE
export MULTISEED_SAMPLE_DIR MULTISEED_EVAL_DIR SEEDS NUM_SAMPLES HISTOGRAM_BINS

MISSING=0
LINK_NEEDED=0
IFS=',' read -r -a SEED_VALUES <<< "${SEEDS}"
for SEED in "${SEED_VALUES[@]}"; do
    if [ ! -f "${MULTISEED_SAMPLE_DIR}/seed_${SEED}.csv" ]; then
        if [ "${SEED}" = "74000" ] && [ -f "${EXISTING_SEED_74000_SAMPLE}" ]; then
            LINK_NEEDED=1
        else
            MISSING=1
        fi
    fi
done
if [ "${MISSING}" -eq 1 ]; then
    SUBMISSION=$(sbatch --parsable sample_center_logwidth_multiseed.sh)
    SAMPLE_JOB="${SUBMISSION%%;*}"
    EVAL_DEPENDENCY=(--dependency="afterok:${SAMPLE_JOB}")
else
    # Submit the lightweight job once if it only needs to create the seed-74000 link.
    if [ "${LINK_NEEDED}" -eq 1 ]; then
        SUBMISSION=$(sbatch --parsable sample_center_logwidth_multiseed.sh)
        SAMPLE_JOB="${SUBMISSION%%;*} (reuse/link only)"
        EVAL_DEPENDENCY=(--dependency="afterok:${SUBMISSION%%;*}")
    else
        SAMPLE_JOB="not submitted; reused all five seed samples"
        EVAL_DEPENDENCY=()
    fi
fi

SUBMISSION=$(sbatch --parsable "${EVAL_DEPENDENCY[@]}" evaluate_center_logwidth_multiseed.sh)
EVAL_JOB="${SUBMISSION%%;*}"
echo "========================================"
echo "Center/log-width MLP multiseed submitted"
echo "Training submitted : no"
echo "Seeds              : ${SEEDS}"
echo "Sampling           : ${SAMPLE_JOB}"
echo "Evaluation         : ${EVAL_JOB}"
echo "Output             : ${MULTISEED_EVAL_DIR}"
echo "========================================"

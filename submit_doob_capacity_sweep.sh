#!/bin/bash
# Submit a controlled Doob-guide capacity sweep. Only guide width/depth change.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

CONFIGS=(
    "64:3"
    "96:3"
)

STEPS="${STEPS:-8000}"
JOB_IDS=()

for CONFIG in "${CONFIGS[@]}"; do
    IFS=: read -r WIDTH DEPTH <<< "${CONFIG}"
    GUIDE_NAME="doob_sampled_arity_qsplit_multiq8_realized_curriculum_d${WIDTH}_l${DEPTH}_${STEPS}"
    GUIDE_PATH="tabdiff/ckpt/shoppers/ft_periodic_seed0/${GUIDE_NAME}"

    if [ -f "${GUIDE_PATH}/best_guide.pt" ]; then
        echo "Skipping completed guide: ${GUIDE_PATH}"
        continue
    fi
    if [ -d "${GUIDE_PATH}" ] && [ -n "$(find "${GUIDE_PATH}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "ERROR: refusing to overwrite incomplete/nonempty guide directory: ${GUIDE_PATH}"
        exit 1
    fi

    SUBMISSION=$(D_TOKEN="${WIDTH}" \
        NUM_LAYERS="${DEPTH}" \
        GUIDE_DIR_NAME="${GUIDE_NAME}" \
        STEPS="${STEPS}" \
        bash submit_doob_sampled_arity_train.sh)

    JOB_ID=$(printf '%s\n' "${SUBMISSION}" | awk '/^Job[[:space:]]*:/ {print $3; exit}')
    if [ -z "${JOB_ID}" ]; then
        echo "ERROR: could not extract Slurm job ID for d${WIDTH}/L${DEPTH}"
        printf '%s\n' "${SUBMISSION}"
        exit 1
    fi
    JOB_IDS+=("${JOB_ID}")
    printf '%s\n' "${SUBMISSION}"
done

echo "========================================"
echo "Capacity sweep submitted"
echo "Configurations : d64/L3, d96/L3"
echo "Steps each     : ${STEPS}"
echo "Jobs           : ${JOB_IDS[*]}"
echo "Existing d48/L2 run is left unchanged"
echo "========================================"

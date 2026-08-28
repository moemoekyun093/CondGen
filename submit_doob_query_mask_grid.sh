#!/bin/bash
# Generate, sample, evaluate, and plot the selectivity-by-mask grid.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
SOURCE_QUERY_DIR="${SOURCE_QUERY_DIR:-data90/${DATANAME}/queries_full}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries_selectivity_mask_grid}"
BANDS="${BANDS:-0.005,0.01,0.02,0.05,0.1,0.25,0.4}"
ARITIES="${ARITIES:-0,2,4,8,12,18}"
QUERIES_PER_BAND="${QUERIES_PER_BAND:-1}"
MASKS_PER_ARITY="${MASKS_PER_ARITY:-3}"
METHOD_LABEL="${METHOD_LABEL:-doob_masked_25000}"
HARPOON_LABEL="${HARPOON_LABEL:-harpoon_style_tabdiff_eta02_s50}"
DOOB_GUIDE_DIR="${DOOB_GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_query_tight_curriculum_masked_10_10_80_d48_l2_25000}"
HARPOON_STYLE_ETA="${HARPOON_STYLE_ETA:-0.2}"
REUSE_HARPOON="${REUSE_HARPOON:-0}"
SUITE_SAMPLE_ROOT="${SUITE_SAMPLE_ROOT:-conditional_samples/${DATANAME}/selectivity_mask_grid}"
SUITE_EVAL_DIR="${SUITE_EVAL_DIR:-evaluations/${DATANAME}/selectivity_mask_grid}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"

python generate_query_mask_grid.py \
    --source-query-dir "${SOURCE_QUERY_DIR}" \
    --output-query-dir "${QUERY_DIR}" \
    --real-data "synthetic/${DATANAME}/real.csv" \
    --bands "${BANDS}" \
    --arities "${ARITIES}" \
    --queries-per-band "${QUERIES_PER_BAND}" \
    --masks-per-arity "${MASKS_PER_ARITY}"

mapfile -t QUERY_FILES < <(python list_accepted_queries.py "${QUERY_DIR}")
NUM_QUERIES="${#QUERY_FILES[@]}"
if [ "${NUM_QUERIES}" -le 0 ]; then
    echo "ERROR: no grid queries were generated"
    exit 1
fi
mkdir -p logs evaluations/slurm "${SUITE_SAMPLE_ROOT}" "${SUITE_EVAL_DIR}"
mkdir -p harpoon_logs

DOOB_MISSING=()
HARPOON_MISSING=()
for INDEX in "${!QUERY_FILES[@]}"; do
    QUERY_ID="$(basename "${QUERY_FILES[INDEX]}" .json)"
    DOOB_SAMPLE="${SUITE_SAMPLE_ROOT}/${METHOD_LABEL}/${QUERY_ID}.csv"
    HARPOON_SAMPLE="${SUITE_SAMPLE_ROOT}/${HARPOON_LABEL}/${QUERY_ID}.csv"
    if [ ! -f "${DOOB_SAMPLE}" ] || [ ! -f "${DOOB_SAMPLE%.csv}.constraints.json" ]; then
        DOOB_MISSING+=("${INDEX}")
    fi
    if [ ! -f "${HARPOON_SAMPLE}" ] || [ ! -f "${HARPOON_SAMPLE%.csv}.constraints.json" ]; then
        HARPOON_MISSING+=("${INDEX}")
    fi
done
if [ "${#DOOB_MISSING[@]}" -gt 0 ] && [ ! -f "${DOOB_GUIDE_DIR}/best_guide.pt" ]; then
    echo "ERROR: trained guide is required for missing samples: ${DOOB_GUIDE_DIR}/best_guide.pt"
    exit 1
fi
if [ "${REUSE_HARPOON}" = "1" ] && [ "${#HARPOON_MISSING[@]}" -gt 0 ]; then
    echo "ERROR: REUSE_HARPOON=1 but ${#HARPOON_MISSING[@]} HARPOON samples are incomplete"
    exit 1
fi

GUIDE_SPECS="${METHOD_LABEL}=${DOOB_GUIDE_DIR}"
HARPOON_STYLE_LABEL="${HARPOON_LABEL}"
export DATANAME MODEL_NAME QUERY_DIR GUIDE_SPECS METHOD_LABEL HARPOON_LABEL
export SUITE_SAMPLE_ROOT SUITE_EVAL_DIR NUM_SAMPLES NUM_TIMESTEPS
export HARPOON_STYLE_ETA HARPOON_STYLE_LABEL

JOB_IDS=()
if [ "${#DOOB_MISSING[@]}" -eq 0 ]; then
    SAMPLE_JOB="reused all existing samples"
else
    DOOB_ARRAY="$(IFS=,; echo "${DOOB_MISSING[*]}")%${MAX_CONCURRENT}"
    SAMPLE_SUBMISSION=$(sbatch \
        --parsable \
        --array="${DOOB_ARRAY}" \
        doob_query_suite_sample.sh)
    SAMPLE_JOB="${SAMPLE_SUBMISSION%%;*}"
    JOB_IDS+=("${SAMPLE_JOB}")
fi
if [ "${REUSE_HARPOON}" = "1" ] || [ "${#HARPOON_MISSING[@]}" -eq 0 ]; then
    HARPOON_JOB="reused all existing samples"
else
    HARPOON_ARRAY="$(IFS=,; echo "${HARPOON_MISSING[*]}")%${MAX_CONCURRENT}"
    HARPOON_SUBMISSION=$(sbatch \
        --parsable \
        --array="${HARPOON_ARRAY}" \
        harpoon_style_tabdiff_query_suite_sample.sh)
    HARPOON_JOB="${HARPOON_SUBMISSION%%;*}"
    JOB_IDS+=("${HARPOON_JOB}")
fi
EVAL_ARGS=(--parsable)
if [ "${#JOB_IDS[@]}" -gt 0 ]; then
    DEPENDENCY="afterok"
    for JOB_ID in "${JOB_IDS[@]}"; do
        DEPENDENCY+=":${JOB_ID}"
    done
    EVAL_ARGS+=(--dependency="${DEPENDENCY}")
else
    DEPENDENCY="all samples complete; evaluation starts immediately"
fi
EVAL_SUBMISSION=$(sbatch "${EVAL_ARGS[@]}" doob_query_mask_grid_evaluate.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "========================================"
echo "Selectivity-by-mask evaluation submitted"
echo "Queries       : ${NUM_QUERIES}"
echo "Bands         : ${BANDS}"
echo "Arities       : ${ARITIES}"
echo "Masks/arity   : ${MASKS_PER_ARITY} (anchors have one unique mask)"
echo "Sample array  : ${SAMPLE_JOB}"
echo "HARPOON array : ${HARPOON_JOB}"
echo "Evaluation    : ${EVAL_JOB} (${DEPENDENCY})"
echo "Output        : ${SUITE_EVAL_DIR}"
echo "========================================"

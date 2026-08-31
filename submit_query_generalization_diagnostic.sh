#!/bin/bash
# Sample a small train-query subset plus all unseen queries, then diagnose coverage.

set -euo pipefail
cd /scratch/work/agrawaa4/TabDiff

DATANAME="${DATANAME:-shoppers}"
MODEL_NAME="${MODEL_NAME:-ft_periodic_seed0}"
METHOD_LABEL="${METHOD_LABEL:-doob_sampled_arity_qsplit_multiq8_8000}"
GUIDE_DIR="${GUIDE_DIR:-tabdiff/ckpt/${DATANAME}/${MODEL_NAME}/doob_sampled_arity_qsplit_multiq8_realized_curriculum_d48_l2_8000}"
QUERY_DIR="${QUERY_DIR:-data90/${DATANAME}/queries}"
SOURCE_QUERY_SPLIT_MANIFEST="${SOURCE_QUERY_SPLIT_MANIFEST:-data90/${DATANAME}/query_splits/sampled_arity_stratified_80_20_seed42.json}"
DIAGNOSTIC_QUERY_SPLIT_MANIFEST="${DIAGNOSTIC_QUERY_SPLIT_MANIFEST:-data90/${DATANAME}/query_splits/generalization_train5_per_band_seed42.json}"
TRAIN_QUERIES_PER_BAND="${TRAIN_QUERIES_PER_BAND:-5}"
TRAIN_SAMPLE_ROOT="${TRAIN_SAMPLE_ROOT:-conditional_samples/${DATANAME}/query_generalization_d48_l2_8000}"
TEST_SAMPLE_DIR="${TEST_SAMPLE_DIR:-conditional_samples/${DATANAME}/sampled_arity_unseen_query_comparison/doob_sampled_arity_qsplit_multiq8_8000}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATANAME}/query_generalization_d48_l2_8000}"
REAL_DATA="${REAL_DATA:-synthetic/${DATANAME}/real.csv}"
QUERY_COORDINATES="${QUERY_COORDINATES:-data90/${DATANAME}/query_splits/query_model_coordinates.json}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-}"

python create_query_generalization_manifest.py \
    --query-dir "${QUERY_DIR}" \
    --source-manifest "${SOURCE_QUERY_SPLIT_MANIFEST}" \
    --train-per-band "${TRAIN_QUERIES_PER_BAND}" \
    --seed 42 \
    --output "${DIAGNOSTIC_QUERY_SPLIT_MANIFEST}"

TRAIN_COUNT=$(python -c "from tabdiff.query_split import load_query_split; print(len(load_query_split('${DIAGNOSTIC_QUERY_SPLIT_MANIFEST}', 'train')))" )
TEST_COUNT=$(python -c "from tabdiff.query_split import load_query_split; print(len(load_query_split('${DIAGNOSTIC_QUERY_SPLIT_MANIFEST}', 'test')))" )
GUIDE_SPECS="${METHOD_LABEL}=${GUIDE_DIR}"
SUITE_SAMPLE_ROOT="${TRAIN_SAMPLE_ROOT}"
QUERY_TEST_SUPPORTED_ONLY=0
mkdir -p logs evaluations/slurm "${OUTPUT_DIR}"

export DATANAME MODEL_NAME METHOD_LABEL GUIDE_DIR GUIDE_SPECS QUERY_DIR
export SOURCE_QUERY_SPLIT_MANIFEST DIAGNOSTIC_QUERY_SPLIT_MANIFEST
export TRAIN_SAMPLE_ROOT TEST_SAMPLE_DIR SUITE_SAMPLE_ROOT OUTPUT_DIR REAL_DATA
export QUERY_COORDINATES QUERY_TEST_SUPPORTED_ONLY

DEPENDENCY_ARGS=()
if [ -n "${TRAIN_JOB_ID}" ]; then
    DEPENDENCY_ARGS+=(--dependency="afterok:${TRAIN_JOB_ID}")
elif [ ! -f "${GUIDE_DIR}/best_guide.pt" ]; then
    echo "ERROR: guide checkpoint is missing; set TRAIN_JOB_ID if training is still running"
    exit 1
fi

mapfile -t TEST_QUERY_FILES < <(
    python list_accepted_queries.py "${QUERY_DIR}" \
        --query-split-manifest "${DIAGNOSTIC_QUERY_SPLIT_MANIFEST}" \
        --query-split test
)
MISSING_TEST=0
for QUERY_FILE in "${TEST_QUERY_FILES[@]}"; do
    QUERY_ID="$(basename "${QUERY_FILE}" .json)"
    if [ ! -f "${TEST_SAMPLE_DIR}/${QUERY_ID}.csv" ]; then
        echo "ERROR: existing d48/L2 test sample is missing: ${TEST_SAMPLE_DIR}/${QUERY_ID}.csv"
        MISSING_TEST=$((MISSING_TEST + 1))
    fi
done
if [ "${MISSING_TEST}" -ne 0 ]; then
    echo "ERROR: ${MISSING_TEST}/${TEST_COUNT} expected test samples are missing; no test resampling was submitted"
    exit 1
fi

QUERY_SPLIT_MANIFEST="${DIAGNOSTIC_QUERY_SPLIT_MANIFEST}"
QUERY_SPLIT=train
export QUERY_SPLIT_MANIFEST QUERY_SPLIT
mapfile -t TRAIN_QUERY_FILES < <(
    python list_accepted_queries.py "${QUERY_DIR}" \
        --query-split-manifest "${DIAGNOSTIC_QUERY_SPLIT_MANIFEST}" \
        --query-split train
)
MISSING_TRAIN_INDICES=()
for INDEX in "${!TRAIN_QUERY_FILES[@]}"; do
    QUERY_ID="$(basename "${TRAIN_QUERY_FILES[INDEX]}" .json)"
    SAMPLE="${TRAIN_SAMPLE_ROOT}/${METHOD_LABEL}/${QUERY_ID}.csv"
    if [ ! -f "${SAMPLE}" ]; then
        MISSING_TRAIN_INDICES+=("${INDEX}")
    fi
done

EVAL_DEPENDENCIES=()
if [ "${#MISSING_TRAIN_INDICES[@]}" -eq 0 ]; then
    TRAIN_ARRAY="reused ${TRAIN_COUNT} existing samples"
else
    ARRAY_INDICES=$(IFS=,; echo "${MISSING_TRAIN_INDICES[*]}")
    TRAIN_SUBMISSION=$(sbatch --parsable "${DEPENDENCY_ARGS[@]}" \
        --array="${ARRAY_INDICES}%${MAX_CONCURRENT}" doob_query_suite_sample.sh)
    TRAIN_ARRAY="${TRAIN_SUBMISSION%%;*}"
    EVAL_DEPENDENCIES+=("${TRAIN_ARRAY}")
fi

if [ -f "${QUERY_COORDINATES}" ]; then
    COORDINATE_JOB="reused existing coordinates"
else
    COORDINATE_SUBMISSION=$(sbatch --parsable export_query_model_coordinates.sh)
    COORDINATE_JOB="${COORDINATE_SUBMISSION%%;*}"
    EVAL_DEPENDENCIES+=("${COORDINATE_JOB}")
fi

EVAL_DEPENDENCY_ARGS=()
if [ "${#EVAL_DEPENDENCIES[@]}" -gt 0 ]; then
    EVAL_DEPENDENCY=$(IFS=:; echo "${EVAL_DEPENDENCIES[*]}")
    EVAL_DEPENDENCY_ARGS+=(--dependency="afterok:${EVAL_DEPENDENCY}")
fi
EVAL_SUBMISSION=$(sbatch --parsable "${EVAL_DEPENDENCY_ARGS[@]}" \
    evaluate_query_generalization.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "========================================"
echo "Query-generalization diagnostic submitted"
echo "Model                    : ${METHOD_LABEL}"
echo "Random train queries     : ${TRAIN_COUNT} (${TRAIN_QUERIES_PER_BAND} per band)"
echo "Unseen test queries      : ${TEST_COUNT}"
echo "Nearest-neighbour pool   : all training queries from source manifest"
echo "Train sampling array     : ${TRAIN_ARRAY}"
echo "Test samples             : reused ${TEST_COUNT} existing CSVs"
echo "Coordinate export        : ${COORDINATE_JOB}"
echo "Evaluation               : ${EVAL_JOB}"
echo "Train samples            : ${TRAIN_SAMPLE_ROOT}/${METHOD_LABEL}"
echo "Test samples             : ${TEST_SAMPLE_DIR}"
echo "Results                  : ${OUTPUT_DIR}"
echo "========================================"

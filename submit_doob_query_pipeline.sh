#!/bin/bash
# Submit curriculum training, sampling, and evaluation as one Slurm chain.

set -euo pipefail

cd /scratch/work/agrawaa4/TabDiff

# These directories must exist before sbatch opens the corresponding log files.
mkdir -p logs evaluations/slurm

TRAIN_SUBMISSION=$(sbatch --parsable doob_query_train.sh)
TRAIN_JOB="${TRAIN_SUBMISSION%%;*}"
SAMPLE_SUBMISSION=$(sbatch \
    --parsable \
    --dependency="afterok:${TRAIN_JOB}" \
    doob_query_sample.sh)
SAMPLE_JOB="${SAMPLE_SUBMISSION%%;*}"
EVAL_SUBMISSION=$(sbatch \
    --parsable \
    --dependency="afterok:${SAMPLE_JOB}" \
    doob_query_evaluate.sh)
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

echo "Doob curriculum pipeline submitted"
echo "  training  : ${TRAIN_JOB}"
echo "  sampling  : ${SAMPLE_JOB} (after training)"
echo "  evaluation: ${EVAL_JOB} (after sampling)"

#!/bin/bash
set -euo pipefail
TABDIFF_PROJECT_ROOT="${TABDIFF_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${TABDIFF_PROJECT_ROOT}"
export TABDIFF_PROJECT_ROOT

METHOD=""
DATANAME=""
OUTPUT_DIR=""
usage() {
    cat <<'EOF'
Usage: bash submit_native_query_baseline_train.sh --method diffputer|great --dataname NAME [options]

Options:
  --output-dir DIR       Default: baselines/checkpoints/NAME/METHOD
  --train-data FILE      Default: data/NAME/train.csv
  --test-data FILE       Default: data/NAME/test.csv
  --info-file FILE       Default: data/NAME/info.json
  --epochs N             Defaults: DiffPuter 1000, GReaT 5
  --batch-size N         Defaults: DiffPuter 1024, GReaT 32
  --learning-rate X      DiffPuter only (default 1e-4)
  --hid-dim N            DiffPuter only (default 1024)
  --timesteps N          DiffPuter only (default 200)
  --llm NAME             GReaT only (default distilgpt2)
EOF
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --dataname) DATANAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --train-data) TRAIN_DATA="$2"; export TRAIN_DATA; shift 2 ;;
        --test-data) TEST_DATA="$2"; export TEST_DATA; shift 2 ;;
        --info-file) INFO_FILE="$2"; export INFO_FILE; shift 2 ;;
        --epochs) EPOCHS="$2"; export EPOCHS; shift 2 ;;
        --batch-size) TRAIN_BATCH_SIZE="$2"; export TRAIN_BATCH_SIZE; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; export LEARNING_RATE; shift 2 ;;
        --hid-dim) HID_DIM="$2"; export HID_DIM; shift 2 ;;
        --timesteps) DIFFUSION_TIMESTEPS="$2"; export DIFFUSION_TIMESTEPS; shift 2 ;;
        --llm) GREAT_LLM="$2"; export GREAT_LLM; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1"; usage; exit 1 ;;
    esac
done
if [[ ! "${METHOD}" =~ ^(diffputer|great)$ ]] || [ -z "${DATANAME}" ]; then
    usage
    exit 1
fi
OUTPUT_DIR="${OUTPUT_DIR:-baselines/checkpoints/${DATANAME}/${METHOD}}"
export METHOD DATANAME OUTPUT_DIR
mkdir -p logs/baselines
submission=$(sbatch --parsable train_native_query_baseline.sh)
job_id="${submission%%;*}"
echo "Submitted ${METHOD} training: ${job_id}"
echo "Dataset: ${DATANAME}"
echo "Output : ${OUTPUT_DIR}"

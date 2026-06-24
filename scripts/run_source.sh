#!/usr/bin/env bash
# Run source training for a single SMAC map.


set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

MAP="${1:?usage: run_source.sh <map> <gpu> [seed] [t_max]}"
GPU="${2:?usage: run_source.sh <map> <gpu> [seed] [t_max]}"
SEED="${3:-1}"
TMAX="${4:-1000000}"

# Activate the conda env named by $CONDA_ENV (default: gct-marl). Skips quietly
# if conda is unavailable or you have already activated your environment.
CONDA_ENV="${CONDA_ENV:-gct-marl}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi
# Avoid the "Descriptors cannot be created directly" protobuf/pysc2 crash.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

mkdir -p logs/source results/source

RUN_NAME="${MAP}_seed${SEED}"
LOG_FILE="logs/source/${RUN_NAME}.out"

echo "[run_source] map=$MAP gpu=cuda:$GPU seed=$SEED t_max=$TMAX run=$RUN_NAME"
echo "[run_source] log -> $LOG_FILE"

exec python -u src/main_source.py \
    --map "$MAP" \
    --seed "$SEED" \
    --device "cuda:$GPU" \
    --t-max "$TMAX" \
    --run-name "$RUN_NAME" \
    > "$LOG_FILE" 2>&1

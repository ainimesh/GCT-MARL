#!/usr/bin/env bash
# Launch a continual-learning run.


set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

RUN_NAME="${1:?usage: launch_continual.sh <run-name> <maps_csv> [gpu] [seed] [tmax_per_phase] [extra...]}"
MAPS="${2:?usage: launch_continual.sh <run-name> <maps_csv> [gpu] [seed] [tmax_per_phase] [extra...]}"
GPU="${3:-0}"
SEED="${4:-0}"
TMAX="${5:-1000000}"
shift 5 2>/dev/null || true
EXTRA="$*"

sess="gct-cont-${RUN_NAME}"
log_file="logs/continual/${RUN_NAME}.out"
res_dir="results/continual/${RUN_NAME}"

if screen -ls 2>/dev/null | grep -q "\.${sess}[[:space:]]"; then
    echo "[launch_continual] SKIP $sess (session already exists)"
    exit 0
fi
if [[ -d "$res_dir" ]]; then
    echo "[launch_continual] SKIP (results dir exists at $res_dir; rm -rf to force fresh)"
    exit 0
fi

mkdir -p logs/continual

CONDA_ENV="${CONDA_ENV:-gct-marl}"
cmd="cd '$HERE' && \
    if command -v conda >/dev/null 2>&1; then source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '$CONDA_ENV' 2>/dev/null || true; fi && \
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && \
    python -u src/main_continual.py \
        --run-name '$RUN_NAME' --maps '$MAPS' \
        --device 'cuda:$GPU' --seed '$SEED' \
        --t-max-per-phase '$TMAX' \
        $EXTRA \
        > '$log_file' 2>&1"

screen -dmS "$sess" bash -c "$cmd; \
    echo '[main_continual] $RUN_NAME exited with code '\$?'; press enter'; \
    exec bash"

echo "[launch_continual] launched $sess"
echo "  maps:    $MAPS"
echo "  gpu:     cuda:$GPU"
echo "  seed:    $SEED"
echo "  tmax/p:  $TMAX"
echo "  extra:   $EXTRA"
echo "  log:     $log_file"
echo "  result:  $res_dir"

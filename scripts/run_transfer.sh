#!/usr/bin/env bash
# Run transfer training (Phase 2): load a frozen source backbone, train the
# target task with the per-view adaptive alignment loss L_xfer (--learn-alphas).


set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

SRC_MAP="${1:?usage: run_transfer.sh <src_map> <tgt_map> <gpu> [seed] [t_max] [extra...]}"
TGT_MAP="${2:?usage: run_transfer.sh <src_map> <tgt_map> <gpu> [seed] [t_max] [extra...]}"
GPU="${3:?usage: run_transfer.sh <src_map> <tgt_map> <gpu> [seed] [t_max] [extra...]}"
SEED="${4:-0}"
TMAX="${5:-1000000}"

# Optional override: SRC_DIR_OVERRIDE points at the run dir to read top_ckpts.json
SRC_DIR_OVERRIDE="${SRC_DIR_OVERRIDE:-}"
SRC_CKPT_OVERRIDE="${SRC_CKPT_OVERRIDE:-}"

# Activate the conda env named by $CONDA_ENV (default: gct-marl). Skips quietly
# if conda is unavailable or you have already activated your environment.
CONDA_ENV="${CONDA_ENV:-gct-marl}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

# Any args after the 5 positionals pass straight through to main_transfer.py

shift 5 2>/dev/null || true
EXTRA_ARGS=("$@")
ALPHA_FLAG=(--learn-alphas)
for a in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
    case "$a" in
        --learn-alphas|--no-xfer-loss|--alpha-o|--alpha-f|--alpha-t)
            ALPHA_FLAG=(); break ;;
    esac
done

# Resolve source ckpt in this order:
#   1. SRC_CKPT_OVERRIDE (env var) -- direct path
#   2. SRC_DIR_OVERRIDE (env var) -- read top_ckpts.json from there
#   3. results/source/<src_map>_seed<S>/top_ckpts.json (best checkpoint)
#   4. results/source/<src_map>_seed<S>/ckpt_final.pt
if [[ -n "$SRC_CKPT_OVERRIDE" ]]; then
    SRC_CKPT="$SRC_CKPT_OVERRIDE"
else
    SRC_DIR="${SRC_DIR_OVERRIDE:-results/source/${SRC_MAP}_seed${SEED}}"
    if [[ -f "$SRC_DIR/top_ckpts.json" ]]; then
        SRC_CKPT=$(SRC_DIR="$SRC_DIR" python -c "
import json, os, sys
sd = os.environ['SRC_DIR']
with open(os.path.join(sd, 'top_ckpts.json')) as f: d = json.load(f)
print(os.path.join(sd, os.path.basename(d[0]['path']))) if d else sys.exit(1)
")
    fi
    if [[ -z "${SRC_CKPT:-}" || ! -f "$SRC_CKPT" ]]; then
        SRC_CKPT="$SRC_DIR/ckpt_final.pt"
    fi
fi

if [[ ! -f "$SRC_CKPT" ]]; then
    echo "[run_transfer] missing source ckpt: $SRC_CKPT" >&2
    exit 1
fi
echo "[run_transfer] using source ckpt: $SRC_CKPT"

mkdir -p logs/transfer results/transfer

RUN_NAME="${SRC_MAP}_to_${TGT_MAP}_seed${SEED}"
LOG_FILE="logs/transfer/${RUN_NAME}.out"

echo "[run_transfer] src=$SRC_MAP -> tgt=$TGT_MAP gpu=cuda:$GPU seed=$SEED t_max=$TMAX run=$RUN_NAME"
echo "[run_transfer] log -> $LOG_FILE"

exec python -u src/main_transfer.py \
    --source-ckpt "$SRC_CKPT" \
    --target-map "$TGT_MAP" \
    --seed "$SEED" \
    --device "cuda:$GPU" \
    --t-max "$TMAX" \
    --run-name "$RUN_NAME" \
    ${ALPHA_FLAG[@]+"${ALPHA_FLAG[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    > "$LOG_FILE" 2>&1

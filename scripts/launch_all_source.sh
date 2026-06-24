#!/usr/bin/env bash
# launch all 9 source-training runs across 2 GPUs available in our server sequentially.
#


set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

SEED="${SEED:-1}"
TMAX="${TMAX:-1000000}"

# Marines (homogeneous, enemy_dim=ally_dim=5, own_dim=1)
GPU0_MAPS=(3m 8m 5m_vs_6m 8m_vs_9m 10m_vs_11m)
# Stalker-vs-Zealot + heterogeneous
GPU1_MAPS=(3s_vs_3z 3s_vs_4z 3s_vs_5z 1c3s5z)

mkdir -p logs/source

run_chain() {
    local gpu="$1"; shift
    local maps=("$@")
    for m in "${maps[@]}"; do
        echo "[launch_all] gpu=$gpu starting $m (seed=$SEED, t_max=$TMAX)"
        bash scripts/run_source.sh "$m" "$gpu" "$SEED" "$TMAX" || \
            echo "[launch_all] gpu=$gpu FAILED on $m (continuing)"
        echo "[launch_all] gpu=$gpu finished $m"
    done
    echo "[launch_all] gpu=$gpu CHAIN DONE"
}


run_chain 0 "${GPU0_MAPS[@]}" > logs/source/_chain_gpu0.out 2>&1 &
PID0=$!
run_chain 1 "${GPU1_MAPS[@]}" > logs/source/_chain_gpu1.out 2>&1 &
PID1=$!

echo "[launch_all] PID gpu0=$PID0  gpu1=$PID1"
echo "[launch_all] tail logs/source/_chain_gpu*.out for orchestrator status"
echo "[launch_all] tail logs/source/<map>_seed${SEED}.out for individual runs"


trap 'echo "[launch_all] caught signal, killing chains"; kill -TERM $PID0 $PID1 2>/dev/null || true; wait $PID0 $PID1 2>/dev/null || true; exit 130' INT TERM

wait $PID0 $PID1
echo "[launch_all] all chains complete"

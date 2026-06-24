"""
Evaluation entry point — load a trained checkpoint and report greedy win rate.

Works with any checkpoint produced by this repo:
  * source     (main_source.py)    -> config has "map"
  * transfer   (main_transfer.py)  -> config has "target_map"
  * continual  (main_continual.py) -> per-phase ckpt; pass --map explicitly

The checkpoint stores the full agent (entity encoder + GRU + GCL backbone +
per-task Q-head) and its EnvSpec. We rebuild the agent, load the weights, and
run `--episodes` greedy (epsilon=0) episodes on the map, reporting the mean
test win rate and episode return.

--ckpt may be a single .pt file OR a run directory. Given a directory, the best
checkpoint is picked automatically (highest test win rate from top_ckpts.json,
falling back to ckpt_final.pt or the most recent ckpt_*.pt).

Usage:
    # point at a run directory -> evaluates its best checkpoint
    python src/evaluate.py --ckpt results/source/3m_seed0
    python src/evaluate.py --ckpt results/transfer/3m_to_8m_seed0
    # or a specific checkpoint file
    python src/evaluate.py --ckpt results/source/3m_seed0/ckpt_t350613_w1.000.pt
    # continual phase checkpoint: the config stores the whole sequence, so
    # pass --map explicitly
    python src/evaluate.py --ckpt results/continual/continual_marines_seed0/phase_3_8m_vs_9m --map 8m_vs_9m

The map is auto-detected from the checkpoint config when possible; pass --map to
override or for continual phase checkpoints.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from smac.env import StarCraft2Env

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import TransferAgent
from obs_parser import EnvSpec
from runner import EpisodeRunner

# Project-wide backbone defaults (MAIL settings). Used only when a checkpoint's
# config does not record the hyper-parameter (e.g. transfer configs do not store
# the backbone dims, which are inherited from the source).
_DEFAULTS = dict(
    embed_dim=64, hidden_dim=64, gcl_dim=64,
    k_nn=5, p_hop=2, l_hop=5,
    lambda1=0.2, lambda2=0.3, temperature=0.5,
    max_move_dim=8, max_enemy_dim=16, max_ally_dim=16, max_own_dim=16,
)


def resolve_ckpt(path: str) -> str:
    """Resolve --ckpt to a concrete .pt file.

    If `path` is a file, return it. If it is a run directory, pick the best
    checkpoint: top_ckpts.json[0] (by basename, so it is portable across
    machines), else ckpt_final.pt, else the most recent ckpt_*.pt.
    """
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise SystemExit(f"--ckpt path does not exist: {path}")

    top = os.path.join(path, "top_ckpts.json")
    if os.path.isfile(top):
        with open(top) as fp:
            entries = json.load(fp)
        if entries:
            cand = os.path.join(path, os.path.basename(entries[0]["path"]))
            if os.path.isfile(cand):
                return cand

    final = os.path.join(path, "ckpt_final.pt")
    if os.path.isfile(final):
        return final

    ckpts = sorted(glob.glob(os.path.join(path, "ckpt_*.pt")), key=os.path.getmtime)
    if ckpts:
        return ckpts[-1]
    raise SystemExit(f"No checkpoint (.pt) found in directory: {path}")


def get_args():
    p = argparse.ArgumentParser(description="Evaluate a GCT-MARL checkpoint.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to a .pt checkpoint OR a run directory (its best "
                        "checkpoint is selected automatically).")
    p.add_argument("--map", type=str, default=None,
                   help="SMAC map to evaluate on. Default: read from the "
                        "checkpoint config (target_map / map). Required for "
                        "continual phase checkpoints.")
    p.add_argument("--episodes", type=int, default=32, help="Greedy test episodes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def resolve_map(cfg: dict, cli_map: str | None) -> str:
    if cli_map:
        return cli_map
    for key in ("target_map", "map"):
        if key in cfg and cfg[key]:
            return cfg[key]
    raise SystemExit(
        "Could not infer the map from the checkpoint config; pass --map "
        "(continual phase checkpoints store the whole sequence, not a single map)."
    )


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_ckpt(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    map_name = resolve_map(cfg, args.map)

    def hp(name):
        return cfg.get(name, _DEFAULTS[name])

    print(f"[EVAL] ckpt={ckpt_path}", flush=True)
    print(f"[EVAL] map={map_name} episodes={args.episodes} device={device}", flush=True)
    if "test_win_rate" in ckpt:
        print(f"[EVAL] checkpoint's recorded test win rate: {ckpt['test_win_rate']:.3f}",
              flush=True)

    env = StarCraft2Env(map_name=map_name, seed=args.seed)
    spec = EnvSpec.from_env(env)
    print(f"[EVAL] spec: n_agents={spec.n_agents} n_enemies={spec.n_enemies} "
          f"obs={spec.obs_dim} actions={spec.n_actions}", flush=True)

    agent = TransferAgent(
        spec=spec, n_actions=spec.n_actions,
        embed_dim=hp("embed_dim"), hidden_dim=hp("hidden_dim"), gcl_dim=hp("gcl_dim"),
        k_nn=hp("k_nn"), p_hop=hp("p_hop"), l_hop=hp("l_hop"),
        lambda1=hp("lambda1"), lambda2=hp("lambda2"), temperature=hp("temperature"),
        max_move_dim=hp("max_move_dim"), max_enemy_dim=hp("max_enemy_dim"),
        max_ally_dim=hp("max_ally_dim"), max_own_dim=hp("max_own_dim"),
    ).to(device)

    try:
        agent.load_state_dict(ckpt["agent"])
    except RuntimeError as e:
        env.close()
        raise SystemExit(
            f"Failed to load agent weights for map '{map_name}'. This usually "
            f"means --map does not match the map the checkpoint was trained on "
            f"(action-space / dim mismatch on the Q-head).\n  {e}"
        )
    agent.eval()

    runner = EpisodeRunner(env=env, agent_net=agent, device=device, spec=spec)
    wins, rets = [], []
    for _ in range(args.episodes):
        _, ret, won, _ = runner.run(epsilon=0.0, evaluate=True)
        wins.append(1.0 if won else 0.0)
        rets.append(ret)
    env.close()

    win_rate = float(np.mean(wins))
    ret_mean = float(np.mean(rets))
    ret_std = float(np.std(rets))
    print("-" * 60, flush=True)
    print(f"[EVAL] map={map_name}  episodes={args.episodes}", flush=True)
    print(f"[EVAL] win_rate    = {win_rate:.3f}", flush=True)
    print(f"[EVAL] mean_return = {ret_mean:.2f} +/- {ret_std:.2f}", flush=True)


if __name__ == "__main__":
    main()

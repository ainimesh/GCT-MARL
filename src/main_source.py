"""
Source-task training entry point.

Trains a TransferAgent + QMixer on a single SMAC map. Saves a checkpoint that
later target tasks can load via main_transfer.py. Records two CSVs per run:
  - metrics.csv : per-episode (noisy, full)
  - eval.csv    : per test interval (clean, used for plots)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from smac.env import StarCraft2Env

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import TransferAgent
from buffer import ReplayBuffer
from learner import SourceLearner
from obs_parser import EnvSpec
from qmix import QMixer
from runner import EpisodeRunner


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--t-max", type=int, default=5_000_000)
    p.add_argument("--buffer-size", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--target-update-interval", type=int, default=200)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-finish", type=float, default=0.05)
    p.add_argument("--epsilon-anneal-steps", type=int, default=50_000)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--gcl-dim", type=int, default=64)
    p.add_argument("--mix-embed-dim", type=int, default=32)
    p.add_argument("--k-nn", type=int, default=5)
    p.add_argument("--p-hop", type=int, default=2)
    p.add_argument("--l-hop", type=int, default=5)
    p.add_argument("--lambda1", type=float, default=0.2)
    p.add_argument("--lambda2", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.5)
    # Encoder padding maxes (must cover every map you intend to transfer to).
    p.add_argument("--max-move-dim", type=int, default=8)
    p.add_argument("--max-enemy-dim", type=int, default=16)
    p.add_argument("--max-ally-dim", type=int, default=16)
    p.add_argument("--max-own-dim", type=int, default=16)
    p.add_argument("--test-interval", type=int, default=10_000)
    p.add_argument("--test-episodes", type=int, default=32)
    p.add_argument("--keep-top-k", type=int, default=5,
                   help="How many best-by-test-win-rate checkpoints to keep on disk.")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="../results/source")
    return p.parse_args()


def epsilon_schedule(t, start, finish, steps):
    if t >= steps:
        return finish
    return start + (finish - start) * t / steps


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[SOURCE] device={device} map={args.map}", flush=True)

    out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out_dir, args.run_name))
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "config.json"), "w") as fp:
        json.dump(vars(args), fp, indent=2)

    env = StarCraft2Env(map_name=args.map, seed=args.seed)
    spec = EnvSpec.from_env(env)
    print(f"[SOURCE] spec: n_agents={spec.n_agents} n_enemies={spec.n_enemies} "
          f"obs={spec.obs_dim} state={spec.state_dim} actions={spec.n_actions} "
          f"move={spec.move_dim} enemy_dim={spec.enemy_dim} ally_dim={spec.ally_dim} "
          f"own_dim={spec.own_dim} ep_limit={spec.episode_limit}", flush=True)

    agent = TransferAgent(
        spec=spec, n_actions=spec.n_actions,
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, gcl_dim=args.gcl_dim,
        k_nn=args.k_nn, p_hop=args.p_hop, l_hop=args.l_hop,
        lambda1=args.lambda1, lambda2=args.lambda2, temperature=args.temperature,
        max_move_dim=args.max_move_dim, max_enemy_dim=args.max_enemy_dim,
        max_ally_dim=args.max_ally_dim, max_own_dim=args.max_own_dim,
    ).to(device)
    mixer = QMixer(
        n_agents=spec.n_agents, state_dim=spec.state_dim, embed_dim=args.mix_embed_dim,
    ).to(device)
    learner = SourceLearner(
        agent=agent, mixer=mixer, spec=spec,
        gamma=args.gamma, lr=args.lr, grad_clip=args.grad_clip,
        beta_gcl=args.beta, target_update_interval=args.target_update_interval,
        device=device,
    )
    buffer = ReplayBuffer(
        buffer_size=args.buffer_size, episode_limit=spec.episode_limit,
        n_agents=spec.n_agents, obs_shape=spec.obs_dim,
        state_shape=spec.state_dim, n_actions=spec.n_actions,
    )
    runner = EpisodeRunner(env=env, agent_net=agent, device=device, spec=spec)

    train_returns = deque(maxlen=100)
    train_wins = deque(maxlen=100)
    last_test_t = -args.test_interval

    # Top-K checkpoint registry.
    best_ckpts: list[dict] = []
    top_ckpts_path = os.path.join(out_root, "top_ckpts.json")
    csv_fp = open(os.path.join(out_root, "metrics.csv"), "w")
    csv_fp.write("env_steps,episode,train_return,train_win,test_return,test_win,td_loss,gcl_loss,grad_norm,epsilon,sec\n")
    eval_fp = open(os.path.join(out_root, "eval.csv"), "w")
    eval_fp.write("step,episodes,wall_time_s,win_rate,mean_reward,std_reward,epsilon,steps_per_sec,td_loss,gcl_loss\n")
    last_eval_step = 0
    last_eval_time = time.time()

    t_total = 0
    episode = 0
    t_start = time.time()
    last_loss = {"td_loss": 0.0, "gcl_loss": 0.0, "grad_norm": 0.0}

    while t_total < args.t_max:
        eps = epsilon_schedule(t_total, args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_steps)
        agent.eval()
        ep_data, ep_ret, ep_won, t_ep = runner.run(epsilon=eps, evaluate=False)
        agent.train()
        buffer.insert(ep_data)
        train_returns.append(ep_ret); train_wins.append(1.0 if ep_won else 0.0)
        t_total += t_ep; episode += 1

        if buffer.can_sample(args.batch_size):
            batch = buffer.sample(args.batch_size).to_torch(device)
            last_loss = learner.update(batch)

        test_ret_mean = float("nan"); test_win_mean = float("nan")
        if t_total - last_test_t >= args.test_interval:
            last_test_t = t_total
            agent.eval()
            test_rets, test_wins = [], []
            for _ in range(args.test_episodes):
                _, r, w, _ = runner.run(epsilon=0.0, evaluate=True)
                test_rets.append(r); test_wins.append(1.0 if w else 0.0)
            agent.train()
            test_ret_mean = float(np.mean(test_rets))
            test_ret_std = float(np.std(test_rets))
            test_win_mean = float(np.mean(test_wins))
            now = time.time()
            wall = now - t_start
            sps = (t_total - last_eval_step) / max(1e-6, now - last_eval_time)
            last_eval_step = t_total; last_eval_time = now
            print(
                f"[t={t_total:>9}] ep={episode:>5} test_ret={test_ret_mean:.2f} "
                f"test_win={test_win_mean:.3f} train_ret={np.mean(train_returns):.2f} "
                f"train_win={np.mean(train_wins):.3f} td={last_loss['td_loss']:.4f} "
                f"gcl={last_loss['gcl_loss']:.4f} eps={eps:.3f} ({wall/60:.1f} min, {sps:.1f} sps)",
                flush=True,
            )
            eval_fp.write(
                f"{t_total},{episode},{wall:.1f},{test_win_mean:.4f},"
                f"{test_ret_mean:.4f},{test_ret_std:.4f},{eps:.4f},{sps:.2f},"
                f"{last_loss['td_loss']:.6f},{last_loss['gcl_loss']:.6f}\n"
            )
            eval_fp.flush()

        csv_fp.write(
            f"{t_total},{episode},{np.mean(train_returns):.4f},{np.mean(train_wins):.4f},"
            f"{test_ret_mean:.4f},{test_win_mean:.4f},"
            f"{last_loss['td_loss']:.6f},{last_loss['gcl_loss']:.6f},{last_loss['grad_norm']:.4f},"
            f"{eps:.4f},{time.time()-t_start:.1f}\n"
        )
        csv_fp.flush()

        # Top-K-by-win-rate checkpointing. Triggered at every test eval.
        if test_win_mean == test_win_mean:  # not NaN
            should_save = (
                len(best_ckpts) < args.keep_top_k or
                # tie-break with mean return so we don't churn at win=0.0
                (test_win_mean, test_ret_mean) >
                (best_ckpts[-1]["win"], best_ckpts[-1]["ret"])
            )
            if should_save:
                ckpt_name = f"ckpt_t{t_total}_w{test_win_mean:.3f}.pt"
                ckpt_path = os.path.join(out_root, ckpt_name)
                torch.save({
                    "agent": agent.state_dict(),
                    "mixer": mixer.state_dict(),
                    "spec": vars(spec), "config": vars(args),
                    "t": t_total, "episode": episode,
                    "test_win_rate": test_win_mean,
                    "test_mean_return": test_ret_mean,
                }, ckpt_path)
                best_ckpts.append({
                    "win": test_win_mean, "ret": test_ret_mean,
                    "t": t_total, "episode": episode, "path": ckpt_path,
                })
                # Sort: highest win first; tie-break by mean return then by t.
                best_ckpts.sort(key=lambda c: (-c["win"], -c["ret"], -c["t"]))
                while len(best_ckpts) > args.keep_top_k:
                    evicted = best_ckpts.pop()
                    try:
                        os.remove(evicted["path"])
                    except OSError:
                        pass
                with open(top_ckpts_path, "w") as fp:
                    json.dump(best_ckpts, fp, indent=2)

    csv_fp.close(); eval_fp.close(); env.close()
    torch.save({
        "agent": agent.state_dict(), "mixer": mixer.state_dict(),
        "spec": vars(spec), "config": vars(args),
        "t": t_total, "episode": episode,
    }, os.path.join(out_root, "ckpt_final.pt"))
    print(f"[SOURCE] done. t={t_total} ep={episode}", flush=True)


if __name__ == "__main__":
    main()

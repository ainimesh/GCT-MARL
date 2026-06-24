"""
Transfer-task training entry point.

Loads a frozen source TransferAgent (entity_encoder + GRU + GCL backbone),
initialises a target TransferAgent on the target SMAC map by copying those
backbone weights, and trains with the contrastive transfer loss:

    L_T = L_TD + beta * L_GCL + gamma_xfer * L_xfer

Records its own eval.csv and metrics.csv under results/transfer/<run-name>/.
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
from obs_parser import EnvSpec
from qmix import QMixer
from runner import EpisodeRunner
from transfer_learner import TransferLearner


BACKBONE_KEYS = ("entity_encoder", "fc_in", "rnn", "gcl")


def load_backbone_(target_agent: TransferAgent, source_state: dict) -> int:
    """Copy backbone weights from a source state_dict into target_agent in-place.

    Returns the number of parameters copied. Raises if any backbone key is
    missing from the source. The Q-head (fc_q) is never copied.
    """
    target_state = target_agent.state_dict()
    copied = 0
    for k, v in source_state.items():
        if any(k.startswith(prefix + ".") or k == prefix for prefix in BACKBONE_KEYS):
            if k in target_state and target_state[k].shape == v.shape:
                target_state[k] = v.clone()
                copied += v.numel()
            else:
                raise RuntimeError(f"Backbone key {k} missing/shape-mismatch in target.")
    target_agent.load_state_dict(target_state)
    return copied


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-ckpt", type=str, required=True)
    p.add_argument("--target-map", type=str, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--t-max", type=int, default=1_000_000)
    p.add_argument("--buffer-size", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--target-update-interval", type=int, default=200)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-finish", type=float, default=0.05)
    p.add_argument("--epsilon-anneal-steps", type=int, default=50_000)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--gamma-xfer", type=float, default=0.5,
                help="Total cross-task L_xfer budget. If --alpha-* are not set, "
                        "this is used as alpha_o with alpha_f=alpha_t=0 (legacy default).")
    p.add_argument("--alpha-o", type=float, default=None,
                help="Per-view L_xfer weight for original view. "
                        "Default None -> use gamma_xfer.")
    p.add_argument("--alpha-f", type=float, default=None,
                help="Per-view L_xfer weight for feature view. Default 0.")
    p.add_argument("--alpha-t", type=float, default=None,
                help="Per-view L_xfer weight for topological view. Default 0.")
    p.add_argument("--learn-alphas", action="store_true",
                help="Learn the per-view alphas via 3 logits + softmax * gamma_xfer "
                        "(option-1 adaptive view weighting). Overrides --alpha-* flags.")
    p.add_argument("--mix-embed-dim", type=int, default=32)
    p.add_argument("--test-interval", type=int, default=10_000)
    p.add_argument("--test-episodes", type=int, default=32)
    p.add_argument("--keep-top-k", type=int, default=5,
                help="How many best-by-test-win-rate checkpoints to keep on disk.")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="../results/transfer")
    p.add_argument("--no-xfer-loss", action="store_true",
                help="Disable L_xfer (ablation: weight init only).")
    return p.parse_args()


def epsilon_schedule(t, start, finish, steps):
    if t >= steps:
        return finish
    return start + (finish - start) * t / steps


def main():
    args = get_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[TRANSFER] device={device} target={args.target_map} src={args.source_ckpt}", flush=True)

    out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out_dir, args.run_name))
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "config.json"), "w") as fp:
        json.dump(vars(args), fp, indent=2)

    # Load source ckpt to read its hyper-parameters and weights.
    src_ckpt = torch.load(args.source_ckpt, map_location="cpu", weights_only=False)
    src_cfg = src_ckpt["config"]

    env = StarCraft2Env(map_name=args.target_map, seed=args.seed)
    spec_T = EnvSpec.from_env(env)
    print(f"[TRANSFER] target spec: n_agents={spec_T.n_agents} n_enemies={spec_T.n_enemies} "
        f"obs={spec_T.obs_dim} state={spec_T.state_dim} actions={spec_T.n_actions} "
        f"ep_limit={spec_T.episode_limit}", flush=True)
    print(f"[TRANSFER] source spec: n_agents={src_ckpt['spec']['n_agents']} "
        f"obs={src_ckpt['spec']['obs_dim']} actions={src_ckpt['spec']['n_actions']}", flush=True)

    # Build TARGET agent + mixer with the same backbone hyper-parameters as source.
    def _make_agent(spec, n_actions):
        return TransferAgent(
            spec=spec, n_actions=n_actions,
            embed_dim=src_cfg["embed_dim"], hidden_dim=src_cfg["hidden_dim"],
            gcl_dim=src_cfg["gcl_dim"], k_nn=src_cfg["k_nn"],
            p_hop=src_cfg["p_hop"], l_hop=src_cfg["l_hop"],
            lambda1=src_cfg["lambda1"], lambda2=src_cfg["lambda2"],
            temperature=src_cfg["temperature"],
            max_move_dim=src_cfg["max_move_dim"], max_enemy_dim=src_cfg["max_enemy_dim"],
            max_ally_dim=src_cfg["max_ally_dim"], max_own_dim=src_cfg["max_own_dim"],
        ).to(device)

    target_agent = _make_agent(spec_T, spec_T.n_actions)
    target_mixer = QMixer(
        n_agents=spec_T.n_agents, state_dim=spec_T.state_dim, embed_dim=args.mix_embed_dim,
    ).to(device)

    # Load source backbone weights into the target agent.
    n_copied = load_backbone_(target_agent, src_ckpt["agent"])
    print(f"[TRANSFER] copied {n_copied} backbone params from source -> target", flush=True)

    # Build the FROZEN source agent for the L_xfer term.
    source_spec_dict = src_ckpt["spec"]
    src_spec = EnvSpec(
        n_agents=source_spec_dict["n_agents"],
        n_enemies=source_spec_dict["n_enemies"],
        n_actions=source_spec_dict["n_actions"],
        move_dim=source_spec_dict["move_dim"],
        enemy_dim=source_spec_dict["enemy_dim"],
        ally_dim=source_spec_dict["ally_dim"],
        own_dim=source_spec_dict["own_dim"],
        obs_dim=source_spec_dict["obs_dim"],
        state_dim=source_spec_dict["state_dim"],
        episode_limit=source_spec_dict["episode_limit"],
        max_enemy_dim=source_spec_dict.get("max_enemy_dim", 0),
        max_ally_dim=source_spec_dict.get("max_ally_dim", 0),
        max_own_dim=source_spec_dict.get("max_own_dim", 0),
        max_n_enemies=source_spec_dict.get("max_n_enemies", 0),
        max_n_allies=source_spec_dict.get("max_n_allies", 0),
    )

    source_agent_for_xfer = _make_agent(spec_T, spec_T.n_actions)  # will receive backbone weights
    load_backbone_(source_agent_for_xfer, src_ckpt["agent"])
    source_agent_for_xfer.eval()
    for p in source_agent_for_xfer.parameters():
        p.requires_grad = False

    if args.learn_alphas:
        ao = af = at = None
        print(f"[TRANSFER] L_xfer weights: LEARNABLE (gamma_xfer={args.gamma_xfer}, "
            f"3 logits init=0 -> softmax start at (1/3, 1/3, 1/3))", flush=True)
    elif args.no_xfer_loss:
        # Hard-disable cross-task alignment.
        ao, af, at = 0.0, 0.0, 0.0
        print(f"[TRANSFER] L_xfer weights: DISABLED (no_xfer_loss=True)", flush=True)
    elif args.alpha_o is None and args.alpha_f is None and args.alpha_t is None:
        # original-view-only with weight = gamma_xfer.
        ao, af, at = args.gamma_xfer, 0.0, 0.0
        print(f"[TRANSFER] L_xfer weights: gamma_xfer={args.gamma_xfer} "
            f"alpha_o={ao} alpha_f={af} alpha_t={at} (sum={ao+af+at})", flush=True)
    else:
        ao = float(args.alpha_o or 0.0)
        af = float(args.alpha_f or 0.0)
        at = float(args.alpha_t or 0.0)
        print(f"[TRANSFER] L_xfer weights: gamma_xfer={args.gamma_xfer} "
            f"alpha_o={ao} alpha_f={af} alpha_t={at} (sum={ao+af+at})", flush=True)
    learner = TransferLearner(
        agent=target_agent, mixer=target_mixer,
        source_agent=source_agent_for_xfer, target_spec=spec_T,
        gamma=args.gamma, lr=args.lr, grad_clip=args.grad_clip,
        beta_gcl=args.beta,
        gamma_xfer=args.gamma_xfer,
        alpha_o=ao, alpha_f=af, alpha_t=at,
        learn_alphas=args.learn_alphas,
        target_update_interval=args.target_update_interval,
        device=device,
    )

    buffer = ReplayBuffer(
        buffer_size=args.buffer_size, episode_limit=spec_T.episode_limit,
        n_agents=spec_T.n_agents, obs_shape=spec_T.obs_dim,
        state_shape=spec_T.state_dim, n_actions=spec_T.n_actions,
    )
    runner = EpisodeRunner(env=env, agent_net=target_agent, device=device, spec=spec_T)

    train_returns = deque(maxlen=100); train_wins = deque(maxlen=100)
    last_test_t = -args.test_interval

    best_ckpts: list[dict] = []
    top_ckpts_path = os.path.join(out_root, "top_ckpts.json")
    csv_fp = open(os.path.join(out_root, "metrics.csv"), "w")
    csv_fp.write("env_steps,episode,train_return,train_win,test_return,test_win,td_loss,gcl_loss,xfer_loss,xfer_o,xfer_f,xfer_t,alpha_o,alpha_f,alpha_t,grad_norm,epsilon,sec\n")
    eval_fp = open(os.path.join(out_root, "eval.csv"), "w")
    eval_fp.write("step,episodes,wall_time_s,win_rate,mean_reward,std_reward,epsilon,steps_per_sec,td_loss,gcl_loss,xfer_loss,xfer_o,xfer_f,xfer_t,alpha_o,alpha_f,alpha_t\n")
    last_eval_step = 0; last_eval_time = time.time()

    t_total = 0; episode = 0
    t_start = time.time()
    last_loss = {"td_loss": 0.0, "gcl_loss": 0.0, "xfer_loss": 0.0,
                "xfer_loss_o": 0.0, "xfer_loss_f": 0.0, "xfer_loss_t": 0.0,
                "alpha_o": 0.0, "alpha_f": 0.0, "alpha_t": 0.0,
                "grad_norm": 0.0}

    while t_total < args.t_max:
        eps = epsilon_schedule(t_total, args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_steps)
        target_agent.eval()
        ep_data, ep_ret, ep_won, t_ep = runner.run(epsilon=eps, evaluate=False)
        target_agent.train()
        buffer.insert(ep_data)
        train_returns.append(ep_ret); train_wins.append(1.0 if ep_won else 0.0)
        t_total += t_ep; episode += 1

        if buffer.can_sample(args.batch_size):
            batch = buffer.sample(args.batch_size).to_torch(device)
            last_loss = learner.update(batch)

        test_ret_mean = float("nan"); test_win_mean = float("nan")
        if t_total - last_test_t >= args.test_interval:
            last_test_t = t_total
            target_agent.eval()
            test_rets, test_wins = [], []
            for _ in range(args.test_episodes):
                _, r, w, _ = runner.run(epsilon=0.0, evaluate=True)
                test_rets.append(r); test_wins.append(1.0 if w else 0.0)
            target_agent.train()
            test_ret_mean = float(np.mean(test_rets))
            test_ret_std = float(np.std(test_rets))
            test_win_mean = float(np.mean(test_wins))
            now = time.time()
            wall = now - t_start
            sps = (t_total - last_eval_step) / max(1e-6, now - last_eval_time)
            last_eval_step = t_total; last_eval_time = now
            print(
                f"[t={t_total:>9}] ep={episode:>5} test_ret={test_ret_mean:.2f} "
                f"test_win={test_win_mean:.3f} train_win={np.mean(train_wins):.3f} "
                f"td={last_loss['td_loss']:.4f} gcl={last_loss['gcl_loss']:.4f} "
                f"xfer={last_loss['xfer_loss']:.4f} eps={eps:.3f} ({wall/60:.1f} min, {sps:.1f} sps)",
                flush=True,
            )
            eval_fp.write(
                f"{t_total},{episode},{wall:.1f},{test_win_mean:.4f},"
                f"{test_ret_mean:.4f},{test_ret_std:.4f},{eps:.4f},{sps:.2f},"
                f"{last_loss['td_loss']:.6f},{last_loss['gcl_loss']:.6f},"
                f"{last_loss['xfer_loss']:.6f},"
                f"{last_loss.get('xfer_loss_o', 0.0):.6f},"
                f"{last_loss.get('xfer_loss_f', 0.0):.6f},"
                f"{last_loss.get('xfer_loss_t', 0.0):.6f},"
                f"{last_loss.get('alpha_o', 0.0):.6f},"
                f"{last_loss.get('alpha_f', 0.0):.6f},"
                f"{last_loss.get('alpha_t', 0.0):.6f}\n"
            )
            eval_fp.flush()

        csv_fp.write(
            f"{t_total},{episode},{np.mean(train_returns):.4f},{np.mean(train_wins):.4f},"
            f"{test_ret_mean:.4f},{test_win_mean:.4f},"
            f"{last_loss['td_loss']:.6f},{last_loss['gcl_loss']:.6f},"
            f"{last_loss['xfer_loss']:.6f},"
            f"{last_loss.get('xfer_loss_o', 0.0):.6f},"
            f"{last_loss.get('xfer_loss_f', 0.0):.6f},"
            f"{last_loss.get('xfer_loss_t', 0.0):.6f},"
            f"{last_loss.get('alpha_o', 0.0):.6f},"
            f"{last_loss.get('alpha_f', 0.0):.6f},"
            f"{last_loss.get('alpha_t', 0.0):.6f},"
            f"{last_loss['grad_norm']:.4f},{eps:.4f},{time.time()-t_start:.1f}\n"
        )
        csv_fp.flush()

        if test_win_mean == test_win_mean:  # not NaN
            should_save = (
                len(best_ckpts) < args.keep_top_k or
                (test_win_mean, test_ret_mean) >
                (best_ckpts[-1]["win"], best_ckpts[-1]["ret"])
            )
            if should_save:
                ckpt_name = f"ckpt_t{t_total}_w{test_win_mean:.3f}.pt"
                ckpt_path = os.path.join(out_root, ckpt_name)
                _ao, _af, _at = learner.current_alphas()
                torch.save({
                    "agent": target_agent.state_dict(),
                    "mixer": target_mixer.state_dict(),
                    "spec": vars(spec_T), "config": vars(args),
                    "t": t_total, "episode": episode,
                    "test_win_rate": test_win_mean,
                    "test_mean_return": test_ret_mean,
                    "alpha_o": _ao, "alpha_f": _af, "alpha_t": _at,
                    "learn_alphas": bool(args.learn_alphas),
                    "alpha_logits": (learner.alpha_logits.detach().cpu().tolist()
                                    if learner.alpha_logits is not None else None),
                }, ckpt_path)
                best_ckpts.append({
                    "win": test_win_mean, "ret": test_ret_mean,
                    "t": t_total, "episode": episode, "path": ckpt_path,
                })
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
    _ao, _af, _at = learner.current_alphas()
    torch.save({
        "agent": target_agent.state_dict(), "mixer": target_mixer.state_dict(),
        "spec": vars(spec_T), "config": vars(args),
        "t": t_total, "episode": episode,
        "alpha_o": _ao, "alpha_f": _af, "alpha_t": _at,
        "learn_alphas": bool(args.learn_alphas),
        "alpha_logits": (learner.alpha_logits.detach().cpu().tolist()
                        if learner.alpha_logits is not None else None),
    }, os.path.join(out_root, "ckpt_final.pt"))
    print(f"[TRANSFER] done. t={t_total} ep={episode}", flush=True)
    print(f"[TRANSFER] final alphas: o={_ao:.4f} f={_af:.4f} t={_at:.4f} (sum={_ao+_af+_at:.4f})", flush=True)


if __name__ == "__main__":
    main()

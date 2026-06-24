"""
Continual training: train a sequence of population-varying SMAC maps with a
SHARED transferable backbone (entity encoder + GRU + 3-view GCL) and per-phase
Q-head + mixer.
"""

from __future__ import annotations

import argparse
import copy
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
from main_transfer import BACKBONE_KEYS, load_backbone_
from obs_parser import EnvSpec
from qmix import QMixer
from runner import EpisodeRunner
from transfer_learner import TransferLearner


# ----------------------------------------------------------------------------- 
# Helpers
# ----------------------------------------------------------------------------- 
def epsilon_schedule(t, start, finish, steps):
    if t >= steps:
        return finish
    return start + (finish - start) * t / steps


def envspec_from_dict(d: dict) -> EnvSpec:
    return EnvSpec(
        n_agents=d["n_agents"], n_enemies=d["n_enemies"], n_actions=d["n_actions"],
        move_dim=d["move_dim"], enemy_dim=d["enemy_dim"], ally_dim=d["ally_dim"],
        own_dim=d["own_dim"], obs_dim=d["obs_dim"], state_dim=d["state_dim"],
        episode_limit=d["episode_limit"],
        max_enemy_dim=d.get("max_enemy_dim", 0), max_ally_dim=d.get("max_ally_dim", 0),
        max_own_dim=d.get("max_own_dim", 0),
        max_n_enemies=d.get("max_n_enemies", 0), max_n_allies=d.get("max_n_allies", 0),
    )


def make_agent(spec: EnvSpec, n_actions: int, args, device) -> TransferAgent:
    return TransferAgent(
        spec=spec, n_actions=n_actions,
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, gcl_dim=args.gcl_dim,
        k_nn=args.k_nn, p_hop=args.p_hop, l_hop=args.l_hop,
        lambda1=args.lambda1, lambda2=args.lambda2, temperature=args.temperature,
        max_move_dim=args.max_move_dim, max_enemy_dim=args.max_enemy_dim,
        max_ally_dim=args.max_ally_dim, max_own_dim=args.max_own_dim,
    ).to(device)


def best_ckpt_path(phase_dir: str) -> str:
    """Return the best top-5 checkpoint path in a phase directory."""
    j = os.path.join(phase_dir, "top_ckpts.json")
    if not os.path.isfile(j):
        # fall back to ckpt_final.pt if top_ckpts.json is missing
        f = os.path.join(phase_dir, "ckpt_final.pt")
        if os.path.isfile(f):
            return f
        raise RuntimeError(f"no top_ckpts.json or ckpt_final.pt in {phase_dir}")
    with open(j) as fp:
        d = json.load(fp)
    if not d:
        raise RuntimeError(f"empty top_ckpts.json in {phase_dir}")
    return d[0]["path"]


# ----------------------------------------------------------------------------- 
# Per-phase training (one map). Uses SourceLearner if frozen_source is None,
# TransferLearner otherwise.
# ----------------------------------------------------------------------------- 
_BACKBONE_MODULES = ("entity_encoder", "fc_in", "rnn", "gcl")


def freeze_backbone_(agent) -> int:
    """Set requires_grad=False on all transferable backbone modules of the
    online agent. Returns the number of frozen parameter tensors."""
    n_frozen = 0
    for module_name in _BACKBONE_MODULES:
        mod = getattr(agent, module_name, None)
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad = False
            n_frozen += 1
    return n_frozen


def run_phase(
    phase_idx: int,
    map_name: str,
    args,
    device,
    init_backbone_state_dict: dict | None,
    use_xfer: bool,
    out_dir: str,
):
    """Train one phase end-to-end. Returns (agent, mixer, spec, best_ckpt_path)."""
    print(f"\n===== Phase {phase_idx}: {map_name} =====", flush=True)
    os.makedirs(out_dir, exist_ok=True)

    env = StarCraft2Env(map_name=map_name, seed=args.seed)
    spec = EnvSpec.from_env(env)
    print(f"[phase {phase_idx}] spec: n_agents={spec.n_agents} n_enemies={spec.n_enemies} "
          f"obs={spec.obs_dim} state={spec.state_dim} actions={spec.n_actions} "
          f"ep_limit={spec.episode_limit}", flush=True)

    agent = make_agent(spec, spec.n_actions, args, device)
    if init_backbone_state_dict is not None:
        # Initialise transferable backbone from previous phase's agent.
        n_copied = load_backbone_(agent, init_backbone_state_dict)
        print(f"[phase {phase_idx}] copied {n_copied} backbone params from previous phase",
              flush=True)
    if getattr(args, "freeze_backbone", False) and init_backbone_state_dict is not None:
        n_frozen = freeze_backbone_(agent)
        print(f"[phase {phase_idx}] FROZEN BACKBONE ablation: "
              f"{n_frozen} backbone tensors set to requires_grad=False; "
              f"only Q-head and mixer will be trained; L_xfer disabled.",
              flush=True)
    mixer = QMixer(
        n_agents=spec.n_agents, state_dim=spec.state_dim, embed_dim=args.mix_embed_dim,
    ).to(device)

    if not use_xfer:
        learner = SourceLearner(
            agent=agent, mixer=mixer, spec=spec,
            gamma=args.gamma, lr=args.lr, grad_clip=args.grad_clip,
            beta_gcl=args.beta, target_update_interval=args.target_update_interval,
            device=device,
        )
        loss_keys = ("td_loss", "gcl_loss", "grad_norm")
    else:
        # Build a frozen source agent for L_xfer using the current phase's
        # spec (so EntityEncoder + parser work cleanly on this phase's obs)
        # and the inherited backbone weights.
        assert init_backbone_state_dict is not None, "use_xfer requires backbone state dict"
        source_agent = make_agent(spec, spec.n_actions, args, device)
        load_backbone_(source_agent, init_backbone_state_dict)
        source_agent.eval()
        for p in source_agent.parameters():
            p.requires_grad = False
        learner = TransferLearner(
            agent=agent, mixer=mixer,
            source_agent=source_agent, target_spec=spec,
            gamma=args.gamma, lr=args.lr, grad_clip=args.grad_clip,
            beta_gcl=args.beta, gamma_xfer=args.gamma_xfer,
            target_update_interval=args.target_update_interval,
            device=device,
        )
        loss_keys = ("td_loss", "gcl_loss", "xfer_loss", "grad_norm")

    buffer = ReplayBuffer(
        buffer_size=args.buffer_size, episode_limit=spec.episode_limit,
        n_agents=spec.n_agents, obs_shape=spec.obs_dim,
        state_shape=spec.state_dim, n_actions=spec.n_actions,
    )
    runner = EpisodeRunner(env=env, agent_net=agent, device=device, spec=spec)

    train_returns = deque(maxlen=100); train_wins = deque(maxlen=100)
    last_test_t = -args.test_interval
    best_ckpts: list[dict] = []
    top_ckpts_path = os.path.join(out_dir, "top_ckpts.json")

    eval_path = os.path.join(out_dir, "eval.csv")
    eval_fp = open(eval_path, "w")
    has_xfer = "xfer_loss" in loss_keys
    eval_fp.write(
        "step,episodes,wall_time_s,win_rate,mean_reward,std_reward,epsilon,"
        "steps_per_sec,td_loss,gcl_loss" + (",xfer_loss" if has_xfer else "") + "\n"
    )
    metrics_path = os.path.join(out_dir, "metrics.csv")
    csv_fp = open(metrics_path, "w")
    csv_fp.write("env_steps,episode,train_return,train_win,test_return,test_win,"
                 "td_loss,gcl_loss" + (",xfer_loss" if has_xfer else "") + ",grad_norm,epsilon,sec\n")

    last_eval_step = 0; last_eval_time = time.time()
    t_total = 0; episode = 0
    t_start = time.time()
    last_loss = {k: 0.0 for k in loss_keys}

    # Early-stop tracking.
    recent_win_history: list[float] = []
    perfect_eval_count: int = 0

    while t_total < args.t_max_per_phase:
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
            for k in loss_keys:
                last_loss.setdefault(k, 0.0)

        test_ret_mean = float("nan"); test_win_mean = float("nan"); test_ret_std = float("nan")
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
            recent_win_history.append(test_win_mean)
            if test_win_mean >= 0.999:
                perfect_eval_count += 1
            now = time.time()
            wall = now - t_start
            sps = (t_total - last_eval_step) / max(1e-6, now - last_eval_time)
            last_eval_step = t_total; last_eval_time = now

            xfer_str = f" xfer={last_loss.get('xfer_loss', 0.0):.4f}" if has_xfer else ""
            print(
                f"[phase {phase_idx} t={t_total:>9}] ep={episode:>5} test_win={test_win_mean:.3f} "
                f"test_ret={test_ret_mean:.2f} train_win={np.mean(train_wins):.3f} "
                f"td={last_loss['td_loss']:.4f} gcl={last_loss.get('gcl_loss',0.0):.4f}{xfer_str} "
                f"eps={eps:.3f} ({wall/60:.1f} min, {sps:.1f} sps)",
                flush=True,
            )

            row = (f"{t_total},{episode},{wall:.1f},{test_win_mean:.4f},"
                   f"{test_ret_mean:.4f},{test_ret_std:.4f},{eps:.4f},{sps:.2f},"
                   f"{last_loss['td_loss']:.6f},{last_loss.get('gcl_loss',0.0):.6f}")
            if has_xfer:
                row += f",{last_loss.get('xfer_loss', 0.0):.6f}"
            eval_fp.write(row + "\n"); eval_fp.flush()

            # Top-K-by-win-rate checkpointing
            should_save = (
                len(best_ckpts) < args.keep_top_k or
                (test_win_mean, test_ret_mean) >
                (best_ckpts[-1]["win"], best_ckpts[-1]["ret"])
            )
            if should_save:
                ckpt_name = f"ckpt_t{t_total}_w{test_win_mean:.3f}.pt"
                ckpt_path = os.path.join(out_dir, ckpt_name)
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
                best_ckpts.sort(key=lambda c: (-c["win"], -c["ret"], -c["t"]))
                while len(best_ckpts) > args.keep_top_k:
                    evicted = best_ckpts.pop()
                    try:
                        os.remove(evicted["path"])
                    except OSError:
                        pass
                with open(top_ckpts_path, "w") as fp:
                    json.dump(best_ckpts, fp, indent=2)

        # per-episode metrics row
        xfer_csv = (f",{last_loss.get('xfer_loss', 0.0):.6f}" if has_xfer else "")
        csv_fp.write(
            f"{t_total},{episode},{np.mean(train_returns):.4f},{np.mean(train_wins):.4f},"
            f"{test_ret_mean:.4f},{test_win_mean:.4f},"
            f"{last_loss['td_loss']:.6f},{last_loss.get('gcl_loss',0.0):.6f}{xfer_csv},"
            f"{last_loss.get('grad_norm',0.0):.4f},{eps:.4f},{time.time()-t_start:.1f}\n"
        )
        csv_fp.flush()

        # Early-stop: trigger if EITHER rule fires.
        consecutive_rule = (
            args.early_stop_n and args.early_stop_thresh
            and len(recent_win_history) >= args.early_stop_n
            and all(w >= args.early_stop_thresh
                    for w in recent_win_history[-args.early_stop_n:])
        )
        perfect_rule = (
            args.early_stop_perfect_count
            and perfect_eval_count >= args.early_stop_perfect_count
        )
        if consecutive_rule or perfect_rule:
            reason = ("consecutive" if consecutive_rule else "perfect")
            if consecutive_rule:
                msg = (f"last {args.early_stop_n} test evals all "
                       f">= {args.early_stop_thresh:.2f}")
            else:
                msg = (f"reached {perfect_eval_count} cumulative perfect "
                       f"(>=99.9%) test evals "
                       f">= --early-stop-perfect-count={args.early_stop_perfect_count}")
            print(f"[phase {phase_idx}] early-stop ({reason}): {msg} "
                  f"(current win={test_win_mean:.3f})", flush=True)
            break

    eval_fp.close(); csv_fp.close(); env.close()
    final = os.path.join(out_dir, "ckpt_final.pt")
    torch.save({
        "agent": agent.state_dict(), "mixer": mixer.state_dict(),
        "spec": vars(spec), "config": vars(args),
        "t": t_total, "episode": episode,
    }, final)

    best = best_ckpt_path(out_dir)
    print(f"[phase {phase_idx}] done. best ckpt: {best}", flush=True)
    return agent, mixer, spec, best


# ----------------------------------------------------------------------------- 
# Backward evaluation: re-evaluate the (current backbone + saved per-task heads)
# on each prior map.
# ----------------------------------------------------------------------------- 
def backward_eval(
    cur_agent: TransferAgent,
    saved_heads: list[dict],
    args,
    device,
    cur_phase: int,
    cur_phase_map: str,
    cur_phase_t: int,
    cur_phase_wall_s: float,
    out_csv_path: str,
):
    cur_backbone_state = cur_agent.state_dict()

    for entry in saved_heads:
        j_phase = entry["phase"]
        j_map   = entry["map"]
        j_spec  = envspec_from_dict(entry["spec_dict"])
        eval_env = StarCraft2Env(map_name=j_map, seed=args.seed)
        eval_spec = EnvSpec.from_env(eval_env)

        eval_agent = make_agent(eval_spec, eval_spec.n_actions, args, device)
        # Plug in current (phase-k) backbone and saved phase-j Q-head.
        load_backbone_(eval_agent, cur_backbone_state)
        eval_agent.fc_q.load_state_dict(entry["qhead"])

        runner = EpisodeRunner(env=eval_env, agent_net=eval_agent, device=device, spec=eval_spec)
        wins, rets = [], []
        for _ in range(args.backward_eval_episodes):
            _, r, w, _ = runner.run(epsilon=0.0, evaluate=True)
            wins.append(1.0 if w else 0.0); rets.append(r)
        win_rate = float(np.mean(wins))
        ret_mean = float(np.mean(rets))
        eval_env.close()

        with open(out_csv_path, "a") as fp:
            fp.write(
                f"{cur_phase},{cur_phase_map},{cur_phase_t},{cur_phase_wall_s:.1f},"
                f"{j_phase},{j_map},backward,{win_rate:.4f},{ret_mean:.4f}\n"
            )
        print(f"  [BT] phase {cur_phase} ({cur_phase_map}) -> eval on phase {j_phase} ({j_map}): "
              f"win={win_rate:.3f} ret={ret_mean:.2f}", flush=True)


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", type=str, required=True,
                   help="Comma-separated sequence, e.g. '3m,5m_vs_6m,8m_vs_9m'")
    p.add_argument("--source-ckpt", type=str, default="",
                   help="Optional ckpt to initialise phase-1 backbone (e.g. a previously "
                        "trained source ckpt of maps[0]).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--t-max-per-phase", type=int, default=1_000_000)
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
    p.add_argument("--gamma-xfer", type=float, default=0.5)
    p.add_argument("--max-move-dim", type=int, default=8)
    p.add_argument("--max-enemy-dim", type=int, default=16)
    p.add_argument("--max-ally-dim", type=int, default=16)
    p.add_argument("--max-own-dim", type=int, default=16)
    p.add_argument("--test-interval", type=int, default=10_000)
    p.add_argument("--test-episodes", type=int, default=32)
    p.add_argument("--backward-eval-episodes", type=int, default=32)
    p.add_argument("--keep-top-k", type=int, default=5)
    p.add_argument("--early-stop-n", type=int, default=4,
                   help="End a phase once this many CONSECUTIVE test evals all ")
    p.add_argument("--early-stop-thresh", type=float, default=0.95,
                   help="Win-rate threshold for the consecutive early-stop rule.")
    p.add_argument("--early-stop-perfect-count", type=int, default=2,
                   help="End a phase once this many CUMULATIVE test evals ")
    p.add_argument("--resume-from-phase", type=int, default=1,
                   help="Skip training of phases 1..(k-1); load their per-phase ")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Frozen-backbone ablation: at every phase k>1, load the ")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="../results/continual")
    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out_dir, args.run_name))
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "phase_heads"), exist_ok=True)
    with open(os.path.join(out_root, "config.json"), "w") as fp:
        json.dump(vars(args), fp, indent=2)

    cont_csv_path = os.path.join(out_root, "continual_eval.csv")
    # Open append mode if resuming so we don't lose forward/backward rows
    # written in earlier phases.
    cont_csv_mode = "a" if args.resume_from_phase > 1 and os.path.isfile(cont_csv_path) else "w"
    with open(cont_csv_path, cont_csv_mode) as fp:
        if cont_csv_mode == "w":
            fp.write("cur_phase,cur_phase_map,cur_phase_t,cur_phase_wall_s,"
                     "eval_phase,eval_map,kind,win_rate,mean_reward\n")

    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    print(f"[CONTINUAL] device={device} sequence: {' -> '.join(maps)}", flush=True)
    print(f"[CONTINUAL] t_max per phase: {args.t_max_per_phase}", flush=True)
    if args.resume_from_phase > 1:
        print(f"[CONTINUAL] RESUMING from phase {args.resume_from_phase}", flush=True)

    saved_heads: list[dict] = []
    init_backbone_state = None

    # Resume mode: load saved heads for phases < resume_from_phase, and load
    # the previous phase's best ckpt as init backbone.
    if args.resume_from_phase > 1:
        for prior_k in range(1, args.resume_from_phase):
            prior_map = maps[prior_k - 1]
            head_path = os.path.join(out_root, "phase_heads",
                                     f"phase_{prior_k}_{prior_map}_qhead_mixer_spec.pt")
            if not os.path.isfile(head_path):
                raise FileNotFoundError(
                    f"resume requires saved head: {head_path} (run "
                    f"scripts/recover_phase_head.py if you killed phase {prior_k} "
                    f"manually)")
            head = torch.load(head_path, map_location="cpu", weights_only=False)
            saved_heads.append({
                "phase": head["phase"], "map": head["map"],
                "spec_dict": head["spec_dict"],
                "qhead": head["qhead"],
                "best_backbone_ckpt": head.get("best_backbone_ckpt", ""),
            })
            print(f"[CONTINUAL] resumed head: phase {prior_k} ({prior_map}) "
                  f"<- {head_path}", flush=True)
        # Init backbone = best ckpt of the LAST loaded prior phase.
        last_head = saved_heads[-1]
        last_ckpt_path = last_head["best_backbone_ckpt"]
        if not last_ckpt_path or not os.path.isfile(last_ckpt_path):
            raise FileNotFoundError(
                f"missing prev-phase best ckpt for resume: {last_ckpt_path}")
        sc = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        init_backbone_state = sc["agent"]
        print(f"[CONTINUAL] resumed init backbone from {last_ckpt_path}", flush=True)
    elif args.source_ckpt:
        sc = torch.load(args.source_ckpt, map_location="cpu", weights_only=False)
        init_backbone_state = sc["agent"]
        print(f"[CONTINUAL] loaded init backbone from {args.source_ckpt}", flush=True)

    # Skip phases before resume_from_phase by slicing the iteration.
    for k, map_name in enumerate(maps, start=1):
        if k < args.resume_from_phase:
            continue
        phase_dir = os.path.join(out_root, f"phase_{k}_{map_name}")
        # Phase 1: train as source (no L_xfer), but optionally seed backbone
        # from --source-ckpt. Phases >1: use L_xfer with previous backbone as
        # the frozen source — UNLESS --freeze-backbone is set, in which case
        # we run a frozen-backbone ablation: copy the backbone, freeze it,
        # and train only the new Q-head + mixer (no L_xfer either).
        use_xfer = (k > 1) and (not args.freeze_backbone)
        agent_k, mixer_k, spec_k, best_ckpt_k = run_phase(
            phase_idx=k, map_name=map_name, args=args, device=device,
            init_backbone_state_dict=init_backbone_state,
            use_xfer=use_xfer,
            out_dir=phase_dir,
        )

        # We log the FINAL test_win from the phase's eval.csv.

        final_eval_path = os.path.join(phase_dir, "eval.csv")
        last = None
        if os.path.isfile(final_eval_path):
            try:
                with open(final_eval_path) as fp:
                    last = fp.readlines()[-1].strip().split(",")
            except Exception:
                pass
        cur_t = int(last[0]) if last else 0
        cur_wall = float(last[2]) if last else 0.0
        cur_win = float(last[3]) if last else float("nan")
        cur_ret = float(last[4]) if last else float("nan")
        with open(cont_csv_path, "a") as fp:
            fp.write(f"{k},{map_name},{cur_t},{cur_wall:.1f},{k},{map_name},forward,"
                     f"{cur_win:.4f},{cur_ret:.4f}\n")

        # Backward evaluation on all prior phases.
        if saved_heads:
            print(f"\n--- Backward eval after phase {k} ({map_name}) ---", flush=True)
            backward_eval(
                cur_agent=agent_k, saved_heads=saved_heads,
                args=args, device=device,
                cur_phase=k, cur_phase_map=map_name,
                cur_phase_t=cur_t, cur_phase_wall_s=cur_wall,
                out_csv_path=cont_csv_path,
            )

        # Save phase-k Q-head + mixer + spec for future backward evals.
        head_path = os.path.join(out_root, "phase_heads",
                                 f"phase_{k}_{map_name}_qhead_mixer_spec.pt")
        torch.save({
            "phase": k, "map": map_name, "spec_dict": vars(spec_k),
            "qhead": agent_k.fc_q.state_dict(),
            "mixer": mixer_k.state_dict(),
            "best_backbone_ckpt": best_ckpt_k,
        }, head_path)
        saved_heads.append({
            "phase": k, "map": map_name, "spec_dict": vars(spec_k),
            "qhead": copy.deepcopy(agent_k.fc_q.state_dict()),
            "best_backbone_ckpt": best_ckpt_k,
        })


        init_backbone_state = copy.deepcopy(agent_k.state_dict())

    print(f"\n[CONTINUAL] all {len(maps)} phases complete.", flush=True)
    print(f"[CONTINUAL] outputs in {out_root}", flush=True)


if __name__ == "__main__":
    main()

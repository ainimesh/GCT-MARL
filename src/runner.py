"""
Episode runner. Collects one episode at a time on a SMAC env and returns a
padded transition dict suitable for insertion into the replay buffer.

The "original-graph" adjacency is built each step from the SMAC visibility
matrix (ally block, symmetrised, no self-loop), as in the MAIL reproduction.
"""

from __future__ import annotations

import numpy as np
import torch

from obs_parser import EnvSpec


def build_visibility_adj(env, n_agents: int) -> np.ndarray:
    if hasattr(env, "get_visibility_matrix"):
        vm = np.asarray(env.get_visibility_matrix(), dtype=np.float32)
        ally_block = vm[:n_agents, :n_agents]
    else:
        ally_block = np.ones((n_agents, n_agents), dtype=np.float32)
    np.fill_diagonal(ally_block, 0.0)
    A = ((ally_block + ally_block.T) > 0).astype(np.float32)
    return A


class EpisodeRunner:
    def __init__(self, env, agent_net, device, spec: EnvSpec):
        self.env = env
        self.net = agent_net
        self.device = device
        self.spec = spec
        self.N = spec.n_agents
        self.T = spec.episode_limit + 1
        self.A = spec.n_actions
        self.O = spec.obs_dim
        self.S = spec.state_dim

    @torch.no_grad()
    def run(self, epsilon: float, evaluate: bool = False):
        env = self.env
        env.reset()
        terminated = False
        ep_ret = 0.0
        won = False
        t = 0

        obs_buf   = np.zeros((self.T, self.N, self.O), dtype=np.float32)
        state_buf = np.zeros((self.T, self.S), dtype=np.float32)
        act_buf   = np.zeros((self.T, self.N), dtype=np.int64)
        avail_buf = np.zeros((self.T, self.N, self.A), dtype=np.float32)
        rew_buf   = np.zeros((self.T, 1), dtype=np.float32)
        term_buf  = np.zeros((self.T, 1), dtype=np.float32)
        fill_buf  = np.zeros((self.T, 1), dtype=np.float32)
        adj_buf   = np.zeros((self.T, self.N, self.N), dtype=np.float32)

        h = self.net.init_hidden(1, self.N, self.device)

        while not terminated:
            obs   = np.asarray(env.get_obs(), dtype=np.float32)
            state = np.asarray(env.get_state(), dtype=np.float32)
            avail = np.asarray(env.get_avail_actions(), dtype=np.float32)
            adj   = build_visibility_adj(env, self.N)

            obs_buf[t]   = obs
            state_buf[t] = state
            avail_buf[t] = avail
            adj_buf[t]   = adj
            fill_buf[t]  = 1.0

            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            adj_t = torch.from_numpy(adj).unsqueeze(0).to(self.device)
            q, h_next, _, _ = self.net(obs_t, h, adj_t, spec=self.spec, compute_gcl_loss=False)
            h = h_next

            q_np = q.squeeze(0).cpu().numpy()
            q_np[avail == 0] = -1e10
            actions = np.zeros(self.N, dtype=np.int64)
            for i in range(self.N):
                if (not evaluate) and np.random.rand() < epsilon:
                    valid = np.where(avail[i] == 1)[0]
                    actions[i] = int(np.random.choice(valid)) if len(valid) > 0 else 0
                else:
                    actions[i] = int(np.argmax(q_np[i]))

            act_buf[t] = actions
            reward, terminated, info = env.step(actions.tolist())
            ep_ret += float(reward)
            rew_buf[t] = reward
            term_buf[t] = float(terminated)
            won = bool(info.get("battle_won", False))
            t += 1
            if t >= self.T - 1 and not terminated:
                terminated = True

        # Bootstrap row.
        obs_buf[t]   = np.asarray(env.get_obs(), dtype=np.float32)
        state_buf[t] = np.asarray(env.get_state(), dtype=np.float32)
        avail_buf[t] = np.asarray(env.get_avail_actions(), dtype=np.float32)
        adj_buf[t]   = build_visibility_adj(env, self.N)
        fill_buf[t]  = 0.0

        ep_data = {
            "obs": obs_buf[: t + 1],
            "state": state_buf[: t + 1],
            "actions": act_buf[: t + 1],
            "avail_actions": avail_buf[: t + 1],
            "reward": rew_buf[: t + 1],
            "terminated": term_buf[: t + 1],
            "filled": fill_buf[: t + 1],
            "adj": adj_buf[: t + 1],
        }
        return ep_data, ep_ret, won, t

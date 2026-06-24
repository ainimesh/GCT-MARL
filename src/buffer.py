"""
Episode replay buffer for MAIL.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch


@dataclass
class EpisodeBatch:
    data: Dict[str, np.ndarray]

    def to_torch(self, device):
        out = {}
        for k, v in self.data.items():
            t = torch.from_numpy(v).to(device)
            if t.dtype == torch.float64:
                t = t.float()
            out[k] = t
        return out


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        episode_limit: int,
        n_agents: int,
        obs_shape: int,
        state_shape: int,
        n_actions: int,
    ):
        self.buffer_size = buffer_size
        self.T = episode_limit + 1   # store final state
        self.N = n_agents
        self.size = 0
        self.idx = 0

        f32 = np.float32
        self.data = {
            "obs":          np.zeros((buffer_size, self.T, n_agents, obs_shape), dtype=f32),
            "state":        np.zeros((buffer_size, self.T, state_shape), dtype=f32),
            "actions":      np.zeros((buffer_size, self.T, n_agents), dtype=np.int64),
            "avail_actions":np.zeros((buffer_size, self.T, n_agents, n_actions), dtype=f32),
            "reward":       np.zeros((buffer_size, self.T, 1), dtype=f32),
            "terminated":   np.zeros((buffer_size, self.T, 1), dtype=f32),
            "filled":       np.zeros((buffer_size, self.T, 1), dtype=f32),
            "adj":          np.zeros((buffer_size, self.T, n_agents, n_agents), dtype=f32),
        }

    def __len__(self):
        return self.size

    def can_sample(self, batch_size: int):
        return self.size >= batch_size

    def insert(self, episode: Dict[str, np.ndarray]):
        idx = self.idx
        for k in self.data:
            arr = episode[k]
            T = arr.shape[0]
            self.data[k][idx, :T] = arr
            self.data[k][idx, T:] = 0
        self.idx = (self.idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> EpisodeBatch:
        idx = np.random.choice(self.size, batch_size, replace=False)
        out = {k: v[idx] for k, v in self.data.items()}
        # Trim to max episode length present in the batch (saves compute).
        # filled has shape (B, T, 1) -> max T-1 such that any sample is filled.
        filled = out["filled"][..., 0]                           # (B, T)
        max_t = int(filled.sum(axis=1).max())
        max_t = max(max_t + 1, 2)                                 # at least 2 for next-state
        max_t = min(max_t, self.T)
        out = {k: v[:, :max_t] for k, v in out.items()}
        return EpisodeBatch(out)

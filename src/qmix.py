"""
QMIX mixing network (Rashid et al., 2020).
Used by MAIL as its credit-assignment backbone.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hyper_hidden: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * embed_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: (B, T, n_agents) chosen Q values per agent, per timestep.
            states:   (B, T, state_dim) global state per timestep.
        Returns:
            q_tot: (B, T, 1) mixed total Q value.
        """
        B, T, N = agent_qs.shape
        states = states.reshape(-1, self.state_dim)            # (B*T, S)
        agent_qs = agent_qs.reshape(-1, 1, N)                  # (B*T, 1, N)

        w1 = torch.abs(self.hyper_w1(states)).reshape(-1, N, self.embed_dim)
        b1 = self.hyper_b1(states).reshape(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)           # (B*T, 1, E)

        w2 = torch.abs(self.hyper_w2(states)).reshape(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states).reshape(-1, 1, 1)
        y = torch.bmm(hidden, w2) + b2                         # (B*T, 1, 1)

        return y.reshape(B, T, 1)

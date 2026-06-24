"""
TransferAgent — population/observation-invariant agent network.

Pipeline (per agent, parameter-shared across agents):

    raw obs (D_obs)
        -> EntityEncoder (TRANSFERABLE — per-entity MLPs, no N-dependence)
            -> e_i (E)
        -> MLP_in -> hidden  -> GRUCell (TRANSFERABLE)
            -> z_i (h)
        -> GCLModule, three views over the team graph (TRANSFERABLE — SGC W_*)
            -> H_o[i] (h)
        -> MLP_q ([z_i || H_o[i]]) -> Q_i over A actions (PER-TASK; not transferred)

The "transferable backbone" is everything except `MLP_q`. When loading a source
checkpoint into a target task, only the Q-head is re-initialised. The mixer is
also per-task (different N and state_dim) and not transferred.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from entity_encoder import EntityEncoder
from gcl_module import GCLModule
from obs_parser import EnvSpec


class TransferAgent(nn.Module):
    def __init__(
        self,
        spec: EnvSpec,
        n_actions: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        gcl_dim: int = 64,
        k_nn: int = 5,
        p_hop: int = 2,
        l_hop: int = 5,
        lambda1: float = 0.2,
        lambda2: float = 0.3,
        temperature: float = 0.5,
        # Encoder-level maxes (allow the same encoder to consume different envs).
        max_move_dim: int = 8,
        max_enemy_dim: int = 16,
        max_ally_dim: int = 16,
        max_own_dim: int = 16,
    ):
        super().__init__()
        self.spec = spec
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.gcl_dim = gcl_dim

        self.entity_encoder = EntityEncoder(
            max_move_dim=max_move_dim,
            max_enemy_dim=max_enemy_dim,
            max_ally_dim=max_ally_dim,
            max_own_dim=max_own_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        self.fc_in = nn.Linear(embed_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.gcl = GCLModule(
            in_dim=hidden_dim,
            out_dim=gcl_dim,
            k_nn=k_nn,
            p_hop=p_hop,
            l_hop=l_hop,
            lambda1=lambda1,
            lambda2=lambda2,
            temperature=temperature,
        )
        # Per-task action head (not transferred).
        self.fc_q = nn.Sequential(
            nn.Linear(hidden_dim + gcl_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def init_hidden(self, batch_size: int, n_agents: int, device: torch.device):
        return torch.zeros(batch_size, n_agents, self.hidden_dim, device=device)

    # ------------------------------------------------------------
    # Backbone separately exposed so the transfer learner can call it on the
    # frozen source weights without ever touching the target Q-head.
    # ------------------------------------------------------------

    def backbone(
        self,
        obs: torch.Tensor,         # (B, N, obs_dim)
        h_in: torch.Tensor,        # (B, N, hidden_dim)
        adj: torch.Tensor,         # (B, N, N)
        spec: EnvSpec,
        compute_gcl_loss: bool = True,
        return_all_views: bool = False,
    ):
        """Return (H_o, z, gcl_loss) by default.
        If return_all_views=True, return (H_o, z, gcl_loss, H_f, H_t)."""
        B, N = obs.shape[0], obs.shape[1]
        e = self.entity_encoder(obs, spec)                          # (B, N, E)
        x_in = F.relu(self.fc_in(e))                                # (B, N, H)
        x_in_flat = x_in.reshape(B * N, -1)
        h_flat = h_in.reshape(B * N, -1)
        z_flat = self.rnn(x_in_flat, h_flat)
        z = z_flat.reshape(B, N, -1)
        if return_all_views:
            H_o, gcl_loss, H_f, H_t = self.gcl(
                z, adj, compute_loss=compute_gcl_loss, return_all_views=True,
            )
            return H_o, z, gcl_loss, H_f, H_t
        H_o, gcl_loss = self.gcl(z, adj, compute_loss=compute_gcl_loss)
        return H_o, z, gcl_loss

    def forward(
        self,
        obs: torch.Tensor,
        h_in: torch.Tensor,
        adj: torch.Tensor,
        spec: EnvSpec | None = None,
        compute_gcl_loss: bool = True,
    ):
        spec = spec or self.spec
        H_o, z, gcl_loss = self.backbone(obs, h_in, adj, spec, compute_gcl_loss)
        q = self.fc_q(torch.cat([z, H_o], dim=-1))                  # (B, N, A)
        # z is the next GRU hidden state.
        return q, z, gcl_loss, H_o

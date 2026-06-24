"""
Entity encoder — converts variable-dim SMAC observations into a fixed-dim
per-agent latent x_i. This is the only module whose input dimension depends
on the env: it absorbs that variability so everything downstream
(GRU + GCL + Q-head) is population- and obs-invariant.

Inputs flow:
    raw obs (D_obs)
        -> parse_obs_batch -> move(D_m), enemy(n_en, D_en), ally(n_al-1, D_al), own(D_own)
        -> pad to (max_n_enemies, max_enemy_dim) etc. (zeros, with masks)
        -> per-entity MLP φ_self / φ_ally / φ_enemy + φ_move
        -> masked mean-pool over alive ally / enemy slots
        -> concat (own, ally_pool, enemy_pool, move_proj)  -> MLP -> x_i (h)

The per-entity MLPs (φ_self, φ_ally, φ_enemy) operate on per-entity feature
slices, so they are population-agnostic by construction (not parameterised by
N_T or N_S). They are the transferable front-end of the agent network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_parser import EnvSpec, parse_obs_batch, pad_last_dim, pad_entity_count


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Module:
    layers = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class EntityEncoder(nn.Module):
    """Per-entity MLP encoder that produces a fixed-dim per-agent feature.
    """

    def __init__(
        self,
        max_move_dim: int,
        max_enemy_dim: int,
        max_ally_dim: int,
        max_own_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.max_move_dim = max_move_dim
        self.max_enemy_dim = max_enemy_dim
        self.max_ally_dim = max_ally_dim
        self.max_own_dim = max_own_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.fc_move = nn.Linear(max_move_dim, embed_dim)
        self.phi_self = _mlp(max_own_dim, hidden_dim, embed_dim)
        self.phi_ally = _mlp(max_ally_dim, hidden_dim, embed_dim)
        self.phi_enemy = _mlp(max_enemy_dim, hidden_dim, embed_dim)

        # Fuse (own, pooled allies, pooled enemies, move) -> per-agent x_i.
        self.fuse = _mlp(4 * embed_dim, hidden_dim, embed_dim)

    def forward(self, obs: torch.Tensor, spec: EnvSpec) -> torch.Tensor:
        """
        Args:
            obs: (B, N, obs_dim) flat observations for this map.
            spec: EnvSpec for this map.
        Returns:
            x: (B, N, embed_dim) per-agent latent feature.
        """
        move, enemy, ally, own = parse_obs_batch(obs, spec)

        # Pad feature widths up to the encoder's max.
        move = pad_last_dim(move, self.max_move_dim)
        enemy = pad_last_dim(enemy, self.max_enemy_dim)
        ally = pad_last_dim(ally, self.max_ally_dim)
        own = pad_last_dim(own, self.max_own_dim)

        enemy_mask = (enemy[..., 0] > 0.5).to(obs.dtype)             # (B, N, n_en)
        ally_mask = (ally[..., 0] > 0.5).to(obs.dtype)               # (B, N, n_al-1)

        # Pad entity-slot counts up to encoder max.
        enemy, enemy_mask = pad_entity_count(enemy, max(self.phi_enemy_in_count(spec), spec.n_enemies), enemy_mask)
        ally, ally_mask = pad_entity_count(ally, max(self.phi_ally_in_count(spec), spec.n_agents - 1), ally_mask)

        # Per-entity embeddings.
        h_self = self.phi_self(own)                                  # (B, N, E)
        h_move = self.fc_move(move)                                  # (B, N, E)
        h_ally = self.phi_ally(ally)                                 # (B, N, n_al, E)
        h_enemy = self.phi_enemy(enemy)                              # (B, N, n_en, E)

        # Masked mean pool. Add tiny epsilon so empty slots don't NaN.
        ally_w = ally_mask.unsqueeze(-1)
        enemy_w = enemy_mask.unsqueeze(-1)
        h_ally_pool = (h_ally * ally_w).sum(dim=-2) / ally_w.sum(dim=-2).clamp(min=1e-6)
        h_enemy_pool = (h_enemy * enemy_w).sum(dim=-2) / enemy_w.sum(dim=-2).clamp(min=1e-6)

        x = self.fuse(torch.cat([h_self, h_ally_pool, h_enemy_pool, h_move], dim=-1))
        return x

    # The encoder doesn't need a fixed n_ally/n_enemy slot width — slot count
    # varies per env, but the mean-pool collapses to a per-agent vector.
    def phi_ally_in_count(self, spec: EnvSpec) -> int:
        return spec.n_agents - 1

    def phi_enemy_in_count(self, spec: EnvSpec) -> int:
        return spec.n_enemies

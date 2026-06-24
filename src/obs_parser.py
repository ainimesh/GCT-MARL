"""
SMAC observation parser.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EnvSpec:
    """Per-environment dimensions needed by the entity encoder."""
    n_agents: int
    n_enemies: int
    n_actions: int
    move_dim: int          # D_m
    enemy_dim: int         # D_en (per enemy)
    ally_dim: int          # D_al (per ally other than self)
    own_dim: int           # D_own
    obs_dim: int           # full flat obs dim
    state_dim: int
    episode_limit: int

    # Padded dims used by the encoder (>= the env's own dims; allows transfer
    # between maps with different per-entity feature widths).
    max_enemy_dim: int = 0
    max_ally_dim: int = 0
    max_own_dim: int = 0
    max_n_enemies: int = 0  # encoder pads to this many enemy slots
    max_n_allies: int = 0   # encoder pads to this many ally slots (other than self)

    @classmethod
    def from_env(
        cls,
        env,
        max_enemy_dim: int | None = None,
        max_ally_dim: int | None = None,
        max_own_dim: int | None = None,
        max_n_enemies: int | None = None,
        max_n_allies: int | None = None,
    ) -> "EnvSpec":
        info = env.get_env_info()
        n_agents = info["n_agents"]
        n_enemies = env.n_enemies
        move_dim = env.get_obs_move_feats_size()
        n_en, en_dim = env.get_obs_enemy_feats_size()
        n_al, al_dim = env.get_obs_ally_feats_size()
        own_dim = env.get_obs_own_feats_size()
        return cls(
            n_agents=n_agents,
            n_enemies=n_enemies,
            n_actions=info["n_actions"],
            move_dim=move_dim,
            enemy_dim=en_dim,
            ally_dim=al_dim,
            own_dim=own_dim,
            obs_dim=info["obs_shape"],
            state_dim=info["state_shape"],
            episode_limit=info["episode_limit"],
            max_enemy_dim=max(en_dim, max_enemy_dim or 0),
            max_ally_dim=max(al_dim, max_ally_dim or 0),
            max_own_dim=max(own_dim, max_own_dim or 0),
            max_n_enemies=max(n_en, max_n_enemies or 0),
            max_n_allies=max(n_al, max_n_allies or 0),
        )


def parse_obs_batch(obs: torch.Tensor, spec: EnvSpec):
    """Slice flat per-agent SMAC observations into structured per-entity tensors.

    Args:
        obs: (..., obs_dim) per-agent flat observation tensor.
        spec: EnvSpec describing the per-entity dimensions of this env.

    Returns:
        move:    (..., D_m)
        enemy:   (..., n_enemies, D_en)
        ally:    (..., n_allies-1, D_al)
        own:     (..., D_own)
    """
    n_en, en_d = spec.n_enemies, spec.enemy_dim
    n_al = spec.n_agents - 1                          # ally slots (excl. self)
    al_d = spec.ally_dim
    own_d = spec.own_dim
    m_d = spec.move_dim

    expected = m_d + n_en * en_d + n_al * al_d + own_d
    assert obs.shape[-1] == expected, (
        f"flat obs size {obs.shape[-1]} != expected {expected} "
        f"(m={m_d}, n_en={n_en}*{en_d}={n_en*en_d}, "
        f"n_al={n_al}*{al_d}={n_al*al_d}, own={own_d})"
    )
    cur = 0
    move = obs[..., cur:cur + m_d];                   cur += m_d
    enemy = obs[..., cur:cur + n_en * en_d].reshape(*obs.shape[:-1], n_en, en_d)
    cur += n_en * en_d
    ally = obs[..., cur:cur + n_al * al_d].reshape(*obs.shape[:-1], n_al, al_d)
    cur += n_al * al_d
    own = obs[..., cur:cur + own_d];                  cur += own_d
    return move, enemy, ally, own


def pad_last_dim(x: torch.Tensor, target: int) -> torch.Tensor:
    """Right-pad the last dim of `x` with zeros up to `target`."""
    cur = x.shape[-1]
    if cur >= target:
        return x[..., :target]
    pad = list(x.shape)
    pad[-1] = target - cur
    return torch.cat([x, x.new_zeros(*pad)], dim=-1)


def pad_entity_count(x: torch.Tensor, target: int, mask: torch.Tensor):
    """Right-pad the second-to-last dim (entity slot dim) up to `target`.

    Args:
        x: (..., n, d)
        target: desired entity-slot count (>= n)
        mask: (..., n) — 1 where entity is valid (alive/visible), 0 where dummy
    Returns:
        x_padded: (..., target, d)
        mask_padded: (..., target)
    """
    cur = x.shape[-2]
    if cur >= target:
        return x[..., :target, :], mask[..., :target]
    pad_x_shape = list(x.shape); pad_x_shape[-2] = target - cur
    pad_m_shape = list(mask.shape); pad_m_shape[-1] = target - cur
    x_padded = torch.cat([x, x.new_zeros(*pad_x_shape)], dim=-2)
    mask_padded = torch.cat([mask, mask.new_zeros(*pad_m_shape)], dim=-1)
    return x_padded, mask_padded

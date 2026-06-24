"""
Source-task learner. Standard QMIX TD with MAIL's auxiliary L_GCL term.

    L = L_TD + beta * L_GCL 

Double-Q action selection at the bootstrap step (online picks argmax, target
evaluates).
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn

from obs_parser import EnvSpec


class SourceLearner:
    def __init__(
        self,
        agent: nn.Module,
        mixer: nn.Module,
        spec: EnvSpec,
        gamma: float = 0.99,
        lr: float = 5e-4,
        grad_clip: float = 10.0,
        beta_gcl: float = 0.2,
        target_update_interval: int = 200,
        device: torch.device = torch.device("cpu"),
    ):
        self.agent = agent
        self.mixer = mixer
        self.target_agent = copy.deepcopy(agent)
        self.target_mixer = copy.deepcopy(mixer)
        for p in self.target_agent.parameters():
            p.requires_grad = False
        for p in self.target_mixer.parameters():
            p.requires_grad = False


        self.params = [p for p in agent.parameters() if p.requires_grad] + \
                      [p for p in mixer.parameters() if p.requires_grad]
        self.optim = torch.optim.RMSprop(self.params, lr=lr, alpha=0.99, eps=1e-5)

        self.spec = spec
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.beta = beta_gcl
        self.target_update_interval = target_update_interval
        self.device = device
        self.train_steps = 0

    def _unroll(self, agent: nn.Module, batch: Dict[str, torch.Tensor], compute_gcl: bool):
        B, T = batch["obs"].shape[0], batch["obs"].shape[1]
        N = self.spec.n_agents
        h = agent.init_hidden(B, N, self.device)
        qs = []
        gcl_total = 0.0
        for t in range(T):
            q, h, gcl_loss, _ = agent(
                obs=batch["obs"][:, t],
                h_in=h,
                adj=batch["adj"][:, t],
                spec=self.spec,
                compute_gcl_loss=compute_gcl,
            )
            qs.append(q)
            if compute_gcl:
                gcl_total = gcl_total + gcl_loss
        Q = torch.stack(qs, dim=1)                                # (B, T, N, A)
        if compute_gcl:
            gcl_total = gcl_total / max(1, T)
        else:
            gcl_total = Q.new_zeros(())
        return Q, gcl_total

    def update(self, batch: Dict[str, torch.Tensor]):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        rewards    = batch["reward"][:, :-1]
        actions    = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1]
        mask       = batch["filled"][:, :-1].clone()
        mask[:, 1:] = mask[:, 1:] * (1.0 - terminated[:, :-1])
        avail      = batch["avail_actions"]

        Q_online, gcl_loss = self._unroll(self.agent, batch, compute_gcl=True)
        with torch.no_grad():
            Q_target, _ = self._unroll(self.target_agent, batch, compute_gcl=False)

        chosen = torch.gather(
            Q_online[:, :-1], dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)                                              # (B, T-1, N)

        # Double-Q: online picks next action, target evaluates.
        Q_online_next = Q_online[:, 1:].clone()
        Q_online_next[avail[:, 1:] == 0] = -1e10
        next_actions = Q_online_next.argmax(dim=-1, keepdim=True)
        target_max = torch.gather(
            Q_target[:, 1:], dim=-1, index=next_actions
        ).squeeze(-1)

        states = batch["state"]
        # Mixer is batched over (B, T-1) — single launch.
        q_tot      = self.mixer(chosen, states[:, :-1])
        q_tot_next = self.target_mixer(target_max, states[:, 1:])

        targets = rewards + self.gamma * (1.0 - terminated) * q_tot_next
        td_err = (q_tot - targets.detach()) * mask
        td_loss = (td_err ** 2).sum() / mask.sum().clamp(min=1.0)

        loss = td_loss + self.beta * gcl_loss

        self.optim.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return {
            "td_loss": float(td_loss.item()),
            "gcl_loss": float(gcl_loss.item()) if isinstance(gcl_loss, torch.Tensor) else 0.0,
            "grad_norm": float(grad_norm),
            "q_tot_mean": float(q_tot.mean().item()),
            "target_mean": float(targets.mean().item()),
        }

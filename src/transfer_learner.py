"""
Transfer-task learner.

Adds a contrastive source-target alignment loss to the standard TD + GCL:

    L_T = L_TD + beta * L_GCL_target + L_xfer

The cross-task alignment L_xfer is now a *weighted sum over the three GCL
views* (per-view ablation):

    L_xfer = alpha_o * InfoNCE(H_o^T, H_o^S)
           + alpha_f * InfoNCE(H_f^T, H_f^S)
           + alpha_t * InfoNCE(H_t^T, H_t^S)

Setting only `alpha_o = gamma_xfer` (and the others to 0) reproduces the
original-view-only behaviour we used in all transfer runs prior to the
ablation. Setting all three equal to gamma_xfer/3 is the "all views, equal
weight" condition. The total cross-task budget alpha_o + alpha_f + alpha_t
equals gamma_xfer in every condition so the loss magnitude stays comparable.

H_o^*, H_f^*, H_t^* are all computed on the SAME target observation batch,
flowing gradient through the online (target) agent and no gradient through
the frozen source agent.
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcl_module import info_nce_pair
from obs_parser import EnvSpec


class TransferLearner:
    def __init__(
        self,
        agent: nn.Module,                    # current target agent
        mixer: nn.Module,                    # current target mixer
        source_agent: nn.Module,             # frozen source backbone 
        target_spec: EnvSpec,
        gamma: float = 0.99,
        lr: float = 5e-4,
        grad_clip: float = 10.0,
        beta_gcl: float = 0.2,

        gamma_xfer: float = 0.5,             # convenience: budget single weight
        alpha_o: float | None = None,
        alpha_f: float | None = None,
        alpha_t: float | None = None,
        learn_alphas: bool = False,
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

        # Frozen source backbone — never updated.
        self.source_agent = source_agent
        self.source_agent.eval()
        for p in self.source_agent.parameters():
            p.requires_grad = False

        self.spec = target_spec
        self.gamma_rl = gamma
        self.grad_clip = grad_clip
        self.beta = beta_gcl
        self.gamma_xfer = float(gamma_xfer)

        #! Three modes:
        #   (a) learn_alphas=True: 3 learnable logits -> softmax * gamma_xfer
        #   (b) any --alpha-* passed: fixed per-view weights
        #   (c) all --alpha-* None: original (alpha_o = gamma_xfer, others 0)
        self.learn_alphas = bool(learn_alphas)
        if self.learn_alphas:
            # Initial logits = 0 -> softmax = (1/3, 1/3, 1/3) -> matches view_all.
            self.alpha_logits = nn.Parameter(torch.zeros(3, device=device))
            self.alpha_o = self.alpha_f = self.alpha_t = float("nan")  # unused
            self._need_aux = True   # all three views always needed when learning
            self._need_f = True
            self._need_t = True
        else:
            if alpha_o is None and alpha_f is None and alpha_t is None:
                self.alpha_o = gamma_xfer
                self.alpha_f = 0.0
                self.alpha_t = 0.0
            else:
                self.alpha_o = float(alpha_o or 0.0)
                self.alpha_f = float(alpha_f or 0.0)
                self.alpha_t = float(alpha_t or 0.0)
            self.alpha_logits = None
            self._need_f = self.alpha_f > 0.0
            self._need_t = self.alpha_t > 0.0
            self._need_aux = self._need_f or self._need_t

        # Optimizer covers agent + mixer + (alpha_logits if learning them).
        self.params = list(agent.parameters()) + list(mixer.parameters())
        if self.learn_alphas:
            self.params.append(self.alpha_logits)
        self.optim = torch.optim.RMSprop(self.params, lr=lr, alpha=0.99, eps=1e-5)

        self.target_update_interval = target_update_interval
        self.device = device
        self.train_steps = 0

    # ------------------------------------------------------------
    def current_alphas(self) -> tuple[float, float, float]:
        """Return (alpha_o, alpha_f, alpha_t) currently in use, summing to
        gamma_xfer (for diagnostics / logging / checkpointing)."""
        if self.learn_alphas:
            with torch.no_grad():
                a = self.gamma_xfer * F.softmax(self.alpha_logits, dim=0)
            return float(a[0].item()), float(a[1].item()), float(a[2].item())
        return self.alpha_o, self.alpha_f, self.alpha_t

    # ------------------------------------------------------------
    def _unroll_online(self, batch: Dict[str, torch.Tensor]):
        """Online unroll over T. Returns Qs, gcl_loss, H_o^T, H_f^T, H_t^T.

        H_f^T and H_t^T are returned as None when no per-view weight needs
        them (saves memory + compute).
        """
        B, T = batch["obs"].shape[0], batch["obs"].shape[1]
        N = self.spec.n_agents
        h = self.agent.init_hidden(B, N, self.device)
        qs, h_os, h_fs, h_ts = [], [], [], []
        gcl_total = 0.0
        for t in range(T):
            # forward returns (q, z_new, gcl_loss, H_o); we additionally need
            # H_f and H_t for the per-view L_xfer (when active).
            # Ask backbone() directly for all views, then run the Q-head here
            # to keep compute equivalent.
            obs_t   = batch["obs"][:, t]
            adj_t   = batch["adj"][:, t]
            if self._need_aux:
                H_o, z_new, gcl_loss, H_f, H_t = self.agent.backbone(
                    obs=obs_t, h_in=h, adj=adj_t, spec=self.spec,
                    compute_gcl_loss=True, return_all_views=True,
                )
                h_fs.append(H_f); h_ts.append(H_t)
            else:
                H_o, z_new, gcl_loss = self.agent.backbone(
                    obs=obs_t, h_in=h, adj=adj_t, spec=self.spec,
                    compute_gcl_loss=True,
                )
            q = self.agent.fc_q(torch.cat([z_new, H_o], dim=-1))
            h = z_new
            qs.append(q); h_os.append(H_o)
            gcl_total = gcl_total + gcl_loss
        Q = torch.stack(qs, dim=1)
        H_o_T = torch.stack(h_os, dim=1)             # (B, T, N, D)
        H_f_T = torch.stack(h_fs, dim=1) if h_fs else None
        H_t_T = torch.stack(h_ts, dim=1) if h_ts else None
        gcl_total = gcl_total / max(1, T)
        return Q, gcl_total, H_o_T, H_f_T, H_t_T

    def _unroll_target(self, batch):
        B, T = batch["obs"].shape[0], batch["obs"].shape[1]
        N = self.spec.n_agents
        h = self.target_agent.init_hidden(B, N, self.device)
        qs = []
        with torch.no_grad():
            for t in range(T):
                q, h, _, _ = self.target_agent(
                    obs=batch["obs"][:, t], h_in=h, adj=batch["adj"][:, t],
                    spec=self.spec, compute_gcl_loss=False,
                )
                qs.append(q)
        return torch.stack(qs, dim=1)

    @torch.no_grad()
    def _unroll_source(self, batch):
        """Run the frozen source backbone on target observations to get
        H_o^S (and H_f^S, H_t^S when needed)."""
        B, T = batch["obs"].shape[0], batch["obs"].shape[1]
        N = self.spec.n_agents
        h = self.source_agent.init_hidden(B, N, self.device)
        h_os, h_fs, h_ts = [], [], []
        for t in range(T):
            if self._need_aux:
                H_o, h, _, H_f, H_t = self.source_agent.backbone(
                    obs=batch["obs"][:, t], h_in=h, adj=batch["adj"][:, t],
                    spec=self.spec, compute_gcl_loss=False, return_all_views=True,
                )
                h_fs.append(H_f); h_ts.append(H_t)
            else:
                H_o, h, _ = self.source_agent.backbone(
                    obs=batch["obs"][:, t], h_in=h, adj=batch["adj"][:, t],
                    spec=self.spec, compute_gcl_loss=False,
                )
            h_os.append(H_o)
        H_o_S = torch.stack(h_os, dim=1)
        H_f_S = torch.stack(h_fs, dim=1) if h_fs else None
        H_t_S = torch.stack(h_ts, dim=1) if h_ts else None
        return H_o_S, H_f_S, H_t_S

    # ------------------------------------------------------------
    def _xfer_view_loss(self, H_target, H_source, fill_mask_t):
        """Per-timestep InfoNCE between two view tensors (B, T, N, D), then
        weighted-average over filled timesteps. Returns a scalar loss."""
        T_full = H_target.shape[1]
        per_t = []
        for t in range(T_full):
            per_t.append(info_nce_pair(H_target[:, t], H_source[:, t], tau=0.5))
        l_t = torch.stack(per_t)                       # (T,)
        per_t_w = fill_mask_t.mean(dim=0)              # (T,)
        w_sum = per_t_w.sum().clamp(min=1.0)
        return (l_t * per_t_w).sum() / w_sum

    # ------------------------------------------------------------
    def update(self, batch: Dict[str, torch.Tensor]):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        rewards    = batch["reward"][:, :-1]
        actions    = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1]
        mask       = batch["filled"][:, :-1].clone()
        mask[:, 1:] = mask[:, 1:] * (1.0 - terminated[:, :-1])
        avail      = batch["avail_actions"]

        Q_online, gcl_loss, H_o_T, H_f_T, H_t_T = self._unroll_online(batch)
        Q_target = self._unroll_target(batch)
        H_o_S, H_f_S, H_t_S = self._unroll_source(batch)

        # TD pieces
        chosen = torch.gather(
            Q_online[:, :-1], dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)
        Q_online_next = Q_online[:, 1:].clone()
        Q_online_next[avail[:, 1:] == 0] = -1e10
        next_actions = Q_online_next.argmax(dim=-1, keepdim=True)
        target_max = torch.gather(
            Q_target[:, 1:], dim=-1, index=next_actions
        ).squeeze(-1)

        states = batch["state"]
        q_tot      = self.mixer(chosen, states[:, :-1])
        q_tot_next = self.target_mixer(target_max, states[:, 1:])

        targets = rewards + self.gamma_rl * (1.0 - terminated) * q_tot_next
        td_err = (q_tot - targets.detach()) * mask
        td_loss = (td_err ** 2).sum() / mask.sum().clamp(min=1.0)

        # Per-view L_xfer
        T_full = H_o_T.shape[1]
        fill_mask_t = batch["filled"][:, :T_full].squeeze(-1)        # (B, T)


        if self.learn_alphas:
            l_o = self._xfer_view_loss(H_o_T, H_o_S, fill_mask_t)
            l_f = self._xfer_view_loss(H_f_T, H_f_S, fill_mask_t)
            l_t = self._xfer_view_loss(H_t_T, H_t_S, fill_mask_t)
            # softmax-normalised weights times the total cross-task budget.
            alphas = self.gamma_xfer * F.softmax(self.alpha_logits, dim=0)
            ao_eff, af_eff, at_eff = alphas[0], alphas[1], alphas[2]
            l_xfer_total = ao_eff * l_o + af_eff * l_f + at_eff * l_t
            ao_log = float(ao_eff.detach().item())
            af_log = float(af_eff.detach().item())
            at_log = float(at_eff.detach().item())
        else:
            l_o = self._xfer_view_loss(H_o_T, H_o_S, fill_mask_t) if self.alpha_o > 0 else td_loss.new_zeros(())
            l_f = (self._xfer_view_loss(H_f_T, H_f_S, fill_mask_t)
                   if (self._need_f and H_f_T is not None) else td_loss.new_zeros(()))
            l_t = (self._xfer_view_loss(H_t_T, H_t_S, fill_mask_t)
                   if (self._need_t and H_t_T is not None) else td_loss.new_zeros(()))
            l_xfer_total = self.alpha_o * l_o + self.alpha_f * l_f + self.alpha_t * l_t
            ao_log, af_log, at_log = self.alpha_o, self.alpha_f, self.alpha_t

        loss = td_loss + self.beta * gcl_loss + l_xfer_total

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
            "gcl_loss": float(gcl_loss.item()),
            "xfer_loss":   float(l_xfer_total.item()),
            "xfer_loss_o": float(l_o.item()),
            "xfer_loss_f": float(l_f.item()),
            "xfer_loss_t": float(l_t.item()),
            "alpha_o":     ao_log,    # current alpha_o being used
            "alpha_f":     af_log,
            "alpha_t":     at_log,
            "grad_norm":   float(grad_norm),
            "q_tot_mean":  float(q_tot.mean().item()),
            "target_mean": float(targets.mean().item()),
        }

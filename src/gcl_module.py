"""
Graph Contrastive Learning module for MAIL (IJCAI-25).

Implements the three-view contrastive learning described in Section 3.2":

  * Original view  v_o : built graph G  (adjacency A from agent communication range)
  * Feature view   v_f : kNN graph G_f built from cosine similarity of agent features
  * Topological v_t   : higher-order view (S_G)^l X for l hops

Encoder is SGC (Wu et al., 2019):  H = S^p X W, with shared W between H_o and H_f.

Losses (per Eq. 6, 8, 9, 10):
    L_f  : feature-preserving InfoNCE between H_o and H_f (= H_o + S_F X W_f)
    L_t  : topology-preserving InfoNCE between H_o (re-encoded with W_r) and H_t
    L_c  : cross-module alignment InfoNCE between H_o and H_r
    L_GCL = L_f + lambda1 * L_t + lambda2 * L_c
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization with self-loops:  S = D^{-1/2} (A + I) D^{-1/2}."""
    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    A_tilde = A + eye
    deg = A_tilde.sum(dim=-1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    # D^{-1/2} A_tilde D^{-1/2}
    S = A_tilde * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)
    return S


def _matrix_power(S: torch.Tensor, p: int) -> torch.Tensor:
    """Iterative matrix power; works for batched matrices."""
    if p <= 1:
        return S
    out = S
    for _ in range(p - 1):
        out = torch.matmul(out, S)
    return out


def build_knn_adj(X: torch.Tensor, k: int) -> torch.Tensor:
    """Build a kNN graph adjacency from cosine similarity of node features.

    Args:
        X: (..., N, F) feature tensor.
        k: number of nearest neighbors.
    Returns:
        A_F: (..., N, N) binary adjacency (symmetrized, no self-loops).
    """
    n = X.shape[-2]
    k = min(k, max(1, n - 1))
    Xn = F.normalize(X, dim=-1, eps=1e-8)
    sim = torch.matmul(Xn, Xn.transpose(-1, -2))  # cosine similarity
    # Mask self-similarity so self never appears in top-k.
    eye = torch.eye(n, device=X.device, dtype=torch.bool).expand_as(sim)
    sim = sim.masked_fill(eye, float("-inf"))
    # Top-k indices along last dim.
    _, idx = torch.topk(sim, k=k, dim=-1)
    A = torch.zeros_like(sim)
    A.scatter_(-1, idx, 1.0)
    A = ((A + A.transpose(-1, -2)) > 0).to(X.dtype)  # symmetrize
    return A


def info_nce_pair(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """InfoNCE loss between two views.

    Implements Eq. 6/8/9 from the paper:
        L(v_i) = -log [ exp(D(h_i^a, h_i^b)/tau) /
                       ( sum_j exp(D(h_i^a, h_j^b)/tau)
                         + sum_{v in {a,b}} sum_{j != i} exp(D(h_i^v, h_j^v)/tau) ) ]
    Discriminator D is cosine similarity.

    Args:
        h_a, h_b: (..., N, D) node embeddings for two views (same shape).
        tau: temperature.
    Returns:
        Scalar loss averaged over the N nodes (and any leading batch dims).
    """
    a = F.normalize(h_a, dim=-1, eps=1e-8)
    b = F.normalize(h_b, dim=-1, eps=1e-8)

    # Cross-view sim (positive on diag), intra-view sims for negatives.
    sim_ab = torch.matmul(a, b.transpose(-1, -2)) / tau     # (..., N, N)
    sim_aa = torch.matmul(a, a.transpose(-1, -2)) / tau
    sim_bb = torch.matmul(b, b.transpose(-1, -2)) / tau

    n = sim_ab.shape[-1]
    eye = torch.eye(n, device=sim_ab.device, dtype=torch.bool).expand_as(sim_ab)

    # Numerator: positive pair (i, i) across the two views.
    pos = torch.diagonal(sim_ab, dim1=-2, dim2=-1)          # (..., N)

    # Denominator: cross-view (all j) + intra-view (j != i).
    # Use logsumexp for numerical stability.
    masked_aa = sim_aa.masked_fill(eye, float("-inf"))
    masked_bb = sim_bb.masked_fill(eye, float("-inf"))
    # Concatenate the rows we sum over: sim_ab[i, :], masked_aa[i, :], masked_bb[i, :]
    denom_logits = torch.cat([sim_ab, masked_aa, masked_bb], dim=-1)  # (..., N, 3N)
    denom = torch.logsumexp(denom_logits, dim=-1)           # (..., N)

    loss = -(pos - denom)                                   # = -log(num/denom)
    return loss.mean()


class GCLModule(nn.Module):
    """Three-view graph contrastive learning module.

    Inputs are agent feature matrices X of shape (B, N, F_in) from the per-agent
    GRU+MLP, plus a binary adjacency A of shape (B, N, N) representing the
    "original" communication graph (agents within communication range).

    Returns the message embedding H_o (used downstream for the individual Q
    function) and the auxiliary L_GCL loss.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k_nn: int = 5,
        p_hop: int = 2,
        l_hop: int = 5,
        lambda1: float = 0.2,
        lambda2: float = 0.3,
        temperature: float = 0.5,
        topo_source: str = "original",  # "original" -> S_G = S; "feature" -> S_G = S_F
    ):
        super().__init__()
        assert l_hop > p_hop, "Higher-order view should use l > p."
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_nn = k_nn
        self.p_hop = p_hop
        self.l_hop = l_hop
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tau = temperature
        self.topo_source = topo_source

        # Eq. 5: shared W_f used by both H_o and H_f.
        self.W_f = nn.Linear(in_dim, out_dim, bias=False)
        # Eq. 7: separate W_r (for H_r) and W_t (for H_t).
        self.W_r = nn.Linear(in_dim, out_dim, bias=False)
        self.W_t = nn.Linear(in_dim, out_dim, bias=False)

        # Standard SGC-style initialization (Glorot).
        for m in (self.W_f, self.W_r, self.W_t):
            nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        compute_loss: bool = True,
        return_all_views: bool = False,
    ):
        """
        Args:
            X: (B, N, F_in) per-agent features.
            A: (B, N, N) binary adjacency for the "original" communication graph.
            compute_loss: if True, also compute the within-task L_GCL loss.
            return_all_views: if True, also return H_f and H_t (used by the
                per-view L_xfer ablation in the transfer learner).
        Returns:
            (H_o, loss)                  if not return_all_views
            (H_o, loss, H_f, H_t)        if return_all_views
        """
        if X.dim() == 2:  # (N, F)
            X = X.unsqueeze(0)
            A = A.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Build normalized propagation matrices.
        S = _normalize_adj(A)
        S_p = _matrix_power(S, self.p_hop)               # S^p
        XW_f = self.W_f(X)
        H_o = torch.matmul(S_p, XW_f)                    # (B, N, D)

        # H_f and H_t are needed when computing the within-task L_GCL OR when
        # the caller asks for them explicitly (per-view L_xfer ablation).
        need_aux = compute_loss or return_all_views
        H_f = None; H_t = None
        if need_aux:
            A_F = build_knn_adj(X, self.k_nn)
            S_F = _normalize_adj(A_F)
            H_f = H_o + torch.matmul(S_F, XW_f)          # Eq. 5

            XW_t = self.W_t(X)
            S_G = S if self.topo_source == "original" else S_F
            S_l = _matrix_power(S_G, self.l_hop)
            H_t = torch.matmul(S_l, XW_t)                # Eq. 7

        if compute_loss:
            XW_r = self.W_r(X)
            H_r = torch.matmul(S_p, XW_r)
            # Three losses (Eqs. 6, 8, 9).
            L_f = info_nce_pair(H_o, H_f, self.tau)
            L_t = info_nce_pair(H_r, H_t, self.tau)
            L_c = info_nce_pair(H_o, H_r, self.tau)
            loss = L_f + self.lambda1 * L_t + self.lambda2 * L_c
        else:
            loss = X.new_zeros(())

        if squeeze:
            H_o = H_o.squeeze(0)
            if H_f is not None: H_f = H_f.squeeze(0)
            if H_t is not None: H_t = H_t.squeeze(0)

        if return_all_views:
            return H_o, loss, H_f, H_t
        return H_o, loss

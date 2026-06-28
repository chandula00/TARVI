"""Graph-based velocity refinement for TARVI.

Adapted from VCA-VAE's GraphVelocityRefiner, simplified for scvi framework.
Uses k-NN graph attention to smooth velocity vectors for local coherence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Single GAT layer operating on sparse k-NN edge index."""

    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.W = nn.Linear(in_dim, n_heads * self.head_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.01)
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_heads * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        N = x.shape[0]
        H, D = self.n_heads, self.head_dim

        h = self.W(x).view(N, H, D)
        src, dst = edge_index[0], edge_index[1]

        e_src = (h[src] * self.a_src.unsqueeze(0)).sum(dim=-1)
        e_dst = (h[dst] * self.a_dst.unsqueeze(0)).sum(dim=-1)
        e = self.leaky_relu(e_src + e_dst)

        # Sparse softmax
        e_max = torch.zeros(N, H, device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, H), e, reduce="amax", include_self=True)
        e_norm = torch.exp(e - e_max[dst])

        e_sum = torch.zeros(N, H, device=x.device)
        e_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), e_norm)
        alpha = e_norm / (e_sum[dst] + 1e-8)
        alpha = self.dropout(alpha)

        msg = alpha.unsqueeze(-1) * h[src]
        out = torch.zeros(N, H, D, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, D), msg)

        return out.view(N, H * D) + self.bias


class VelocityRefiner(nn.Module):
    """Refines ODE-derived velocity using k-NN graph attention.

    Takes raw velocity + latent z, applies GAT layers to produce
    a bounded residual correction, gated so it only adjusts where needed.
    """

    def __init__(self, n_genes, z_dim=10, hidden_dim=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_genes = n_genes

        # Project velocity and cell state into shared space
        self.vel_proj = nn.Linear(n_genes, hidden_dim)
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # GAT layers with residual
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(n_layers):
            self.gat_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, n_heads, dropout))
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        # Velocity correction head
        self.vel_correct = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
            nn.Tanh(),
        )

        # Per-cell gate: how much correction to apply
        self.correction_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, velocity, z, edge_index):
        """
        Args:
            velocity: raw ODE velocity [N, n_genes]
            z: latent cell state [N, z_dim]
            edge_index: [2, E] k-NN graph

        Returns:
            refined_velocity: [N, n_genes]
            gate: [N, 1] correction magnitude
        """
        v_feat = self.vel_proj(velocity)
        z_feat = self.z_proj(z)
        h = self.combine(torch.cat([v_feat, z_feat], dim=-1))

        for gat, norm in zip(self.gat_layers, self.gat_norms):
            h_new = gat(h, edge_index)
            h = norm(h + F.gelu(h_new))

        correction = self.vel_correct(h)
        gate = self.correction_gate(h)

        # Scale correction by velocity magnitude (bounded residual)
        vel_scale = velocity.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        refined = velocity + gate * correction * vel_scale

        return refined, gate

"""
model/baselines/tgat_style.py

Baseline 4 of 7 -- TGAT/TGN-style continuous time-encoded graph attention.

Literature-gap row addressed (Phase 2 table): "TGAT / TGN -- Fixed/rolled-
horizon link prediction on generic temporal graphs, time-encoded
attention/memory. Missing: no physical grounding; no cause structure; no
calibrated any-horizon query."

Design
------
Implements TGAT's core mechanism -- a learned functional time encoding
(Time2Vec-style: phi(dt) = [cos(w_1*dt+b_1), ..., cos(w_d*dt+b_d)])
concatenated onto node features before a single scaled dot-product graph
attention layer -- WITHOUT PG-GTPP's physics-biased logit term and WITHOUT
node-type-specific (heterogeneous) projections, matching the "generic
temporal graph, no physical grounding" description exactly. The time
encoding is applied per window step (the k-step relative offset from the
reference time), then pooled by a GRU exactly like PG-GTPP's own node
summarizer -- reusing that piece keeps the comparison about the graph
layer, not about how the temporal window is pooled.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mark_decoder import MarkDecoder
from .common import BaselineOutput, edge_context_from_nodes


class Time2Vec(nn.Module):
    """phi(dt) = [dt * w_0 + b_0 (linear term), cos(dt * w_i + b_i) for i=1..d-1]."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: [...,] -> [..., dim]"""
        raw = dt.unsqueeze(-1) * self.w + self.b
        return torch.cat([raw[..., :1], torch.cos(raw[..., 1:])], dim=-1)


class _TimeAwareNodeEncoder(nn.Module):
    def __init__(self, f_node: int, hidden_dim: int, time_dim: int = 8) -> None:
        super().__init__()
        self.time2vec = Time2Vec(time_dim)
        self.gru = nn.GRU(input_size=f_node + time_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        b, k, v, f_node = node_features.shape
        # relative time offset of each window step from "now" (the last
        # step), in units of steps -- TGAT's time encoding is a function
        # of elapsed time since the reference event/query time.
        dt = torch.arange(k - 1, -1, -1, device=node_features.device, dtype=torch.float32)
        time_feat = self.time2vec(dt)  # [k, time_dim]
        time_feat = time_feat.unsqueeze(0).unsqueeze(0).expand(b, v, k, -1)
        flat_feat = node_features.permute(0, 2, 1, 3)  # [B, V, k, F]
        combined = torch.cat([flat_feat, time_feat], dim=-1).reshape(b * v, k, -1)
        _, h_n = self.gru(combined)
        return h_n.squeeze(0)  # [B*V, hidden_dim]


class _HomogeneousAttentionLayer(nn.Module):
    """Single-head-per-projection scaled dot-product attention, no
    heterogeneous (per node type) projections, no physics bias term --
    the direct contrast with GraphTransformerLayer in graph_encoder.py."""

    def __init__(self, hidden_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_embed: torch.Tensor, edge_index: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        n, h = node_embed.shape
        q = self.q_proj(node_embed).view(n, self.n_heads, self.head_dim)
        k = self.k_proj(node_embed).view(n, self.n_heads, self.head_dim)
        v = self.v_proj(node_embed).view(n, self.n_heads, self.head_dim)

        if edge_index.shape[1] == 0:
            attn_out = torch.zeros_like(node_embed)
        else:
            src, dst = edge_index[0], edge_index[1]
            src_full = torch.cat([src, dst])
            dst_full = torch.cat([dst, src])
            logits = (q[dst_full] * k[src_full]).sum(-1) / (self.head_dim ** 0.5)

            max_per_dst = torch.full((n, self.n_heads), float("-inf"), device=node_embed.device)
            max_per_dst = max_per_dst.scatter_reduce(
                0, dst_full.unsqueeze(-1).expand(-1, self.n_heads), logits, reduce="amax", include_self=True
            )
            max_per_dst = torch.nan_to_num(max_per_dst, neginf=0.0)
            exp_logits = (logits - max_per_dst[dst_full]).exp()
            sum_per_dst = torch.zeros((n, self.n_heads), device=node_embed.device).scatter_add(
                0, dst_full.unsqueeze(-1).expand(-1, self.n_heads), exp_logits
            ).clamp_min(1e-12)
            weights = exp_logits / sum_per_dst[dst_full]

            weighted = weights.unsqueeze(-1) * v[src_full]
            attn_out = torch.zeros((n, self.n_heads, self.head_dim), device=node_embed.device).scatter_add(
                0, dst_full.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim), weighted
            )
        attn_out = self.out_proj(attn_out.reshape(n, h))
        out = self.norm(node_embed + attn_out)
        return out * node_mask.unsqueeze(-1)


class TGATStyleModel(nn.Module):
    def __init__(self, f_node: int, hidden_dim: int = 64, n_heads: int = 4, mark_trunk_width: int = 32, **_ignored) -> None:
        super().__init__()
        self.node_encoder = _TimeAwareNodeEncoder(f_node, hidden_dim)
        self.attn1 = _HomogeneousAttentionLayer(hidden_dim, n_heads)
        self.attn2 = _HomogeneousAttentionLayer(hidden_dim, n_heads)
        self.intensity_head = nn.Linear(2 * hidden_dim, 1)
        self.mark_decoder = MarkDecoder(hidden_dim=2 * hidden_dim, trunk_width=mark_trunk_width)

    def forward(
        self,
        node_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_physics: torch.Tensor,
    ) -> BaselineOutput:
        b, k, v_max, f_node = node_features.shape
        node_embed = self.node_encoder(node_features)  # [B*V_max, hidden_dim]
        node_mask_flat = node_mask.reshape(b * v_max)

        node_embed = self.attn1(node_embed, edge_index, node_mask_flat)
        node_embed = self.attn2(node_embed, edge_index, node_mask_flat)

        edge_context = edge_context_from_nodes(node_embed, edge_index)
        intensity = F.softplus(self.intensity_head(edge_context).squeeze(-1)) + 1e-8
        marks = self.mark_decoder(edge_context)
        return BaselineOutput(intensity=intensity, marks=marks)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

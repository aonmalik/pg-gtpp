"""
model/baselines/dysat_style.py

Baseline 5 of 7 -- DySAT-style dynamic graph transformer.

Literature-gap row addressed: "DySAT / Dynamic Graph Transformer --
Discrete snapshot self-attention over time. Missing: event-driven graph
indexing; snapshot-uniform sampling mismatched to bursty acoustic events;
no failure-cause semantics."

Design
------
DySAT's defining two-stage structure, reproduced at small scale:
  1. Structural attention -- a graph-attention layer applied
     INDEPENDENTLY to each of the k window snapshots (same node features,
     same candidate edge structure, evaluated at every discrete step) --
     no continuous event-driven indexing, exactly the "snapshot-uniform"
     limitation named in the gap table.
  2. Temporal attention -- a small self-attention block over the k
     per-snapshot structural embeddings for each node, pooling them into
     one final embedding.
No physics bias, no node typing, no cause decomposition -- consistent with
every other baseline in this package.

Note on cost: this baseline is the most compute-heavy of the seven (one
structural-attention pass per window step instead of one pass over a
GRU-pooled summary), which is itself worth reporting as a side observation
-- DySAT-style snapshot processing has a real per-step cost that PG-GTPP's
event-driven design (Phase 3 of the storage-aware architecture doc) is
built to avoid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mark_decoder import MarkDecoder
from .common import BaselineOutput, edge_context_from_nodes


class _StructuralAttention(nn.Module):
    """Single-snapshot graph attention, applied once per window step."""

    def __init__(self, hidden_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

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
        return (node_embed + attn_out) * node_mask.unsqueeze(-1)


class DySATStyleModel(nn.Module):
    def __init__(
        self, f_node: int, hidden_dim: int = 64, n_heads: int = 4, mark_trunk_width: int = 32, **_ignored
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(f_node, hidden_dim)
        self.structural_attn = _StructuralAttention(hidden_dim, n_heads)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
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
        node_mask_flat = node_mask.reshape(b * v_max)

        # Stage 1: structural attention independently at every discrete snapshot.
        snapshot_embeds = []
        for t in range(k):
            h_t = self.input_proj(node_features[:, t, :, :]).reshape(b * v_max, self.hidden_dim)
            h_t = self.structural_attn(h_t, edge_index, node_mask_flat)
            snapshot_embeds.append(h_t)
        snapshots = torch.stack(snapshot_embeds, dim=1)  # [B*V_max, k, hidden_dim]

        # Stage 2: temporal self-attention pooling across the k snapshots.
        temporal_out, _ = self.temporal_attn(snapshots, snapshots, snapshots)
        pooled = self.temporal_norm(temporal_out[:, -1, :])  # embedding "as of now"

        edge_context = edge_context_from_nodes(pooled, edge_index)
        intensity = F.softplus(self.intensity_head(edge_context).squeeze(-1)) + 1e-8
        marks = self.mark_decoder(edge_context)
        return BaselineOutput(intensity=intensity, marks=marks)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

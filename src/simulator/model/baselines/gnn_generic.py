"""
model/baselines/gnn_generic.py

Baseline 3 of 7 -- Generic (homogeneous, untyped) graph neural network,
mean-aggregation message passing, single aggregate hazard.

Literature-gap row addressed: the "no physical grounding; no cause
structure; no calibrated any-horizon query" half of the TGAT/TGN row, kept
architecturally generic (not time-encoded attention -- that is baseline 4,
``tgat_style.py``) so it also stands in as the plain "vanilla GNN link
predictor" contrast that a reviewer will otherwise ask for by name.

Design
------
Two-layer GraphSAGE-style mean-aggregation encoder: unlike
``model/graph_encoder.py::GraphEncoder`` this uses (a) a single shared set
of projections regardless of node type (USV/AUV/IoUT are not
distinguished), (b) plain mean-neighbor aggregation instead of attention,
and (c) no physics-biased logit term of any kind -- ``edge_physics`` is
never read by the encoder. This isolates exactly what "physical grounding"
and "heterogeneous, physics-biased attention" buy PG-GTPP over the
simplest respectable graph baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mark_decoder import MarkDecoder
from .common import BaselineOutput, edge_context_from_nodes


class _NodeTemporalSummary(nn.Module):
    """Same purpose as GraphEncoder's NodeTemporalEncoder, but with no
    type-embedding branch -- this baseline is explicitly homogeneous."""

    def __init__(self, f_node: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=f_node, hidden_size=hidden_dim, batch_first=True)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        b, k, v, f_node = node_features.shape
        flat = node_features.permute(0, 2, 1, 3).reshape(b * v, k, f_node)
        _, h_n = self.gru(flat)
        return h_n.squeeze(0).view(b, v, -1)


class _MeanAggSAGELayer(nn.Module):
    """One GraphSAGE-style layer: h_i' = ReLU(W_self . h_i + W_neigh . mean_{j in N(i)} h_j)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.w_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_neigh = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_embed: torch.Tensor, edge_index: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        n, h = node_embed.shape
        if edge_index.shape[1] == 0:
            agg = torch.zeros_like(node_embed)
        else:
            src, dst = edge_index[0], edge_index[1]
            src_full = torch.cat([src, dst])
            dst_full = torch.cat([dst, src])
            msg = node_embed[src_full]

            deg = torch.zeros(n, device=node_embed.device).scatter_add(
                0, dst_full, torch.ones_like(dst_full, dtype=torch.float32)
            ).clamp_min(1.0)
            agg = torch.zeros(n, h, device=node_embed.device).scatter_add(
                0, dst_full.unsqueeze(-1).expand(-1, h), msg
            ) / deg.unsqueeze(-1)

        out = F.relu(self.w_self(node_embed) + self.w_neigh(agg))
        out = self.norm(out)
        return out * node_mask.unsqueeze(-1)


class GenericGNNModel(nn.Module):
    def __init__(self, f_node: int, hidden_dim: int = 64, mark_trunk_width: int = 32, **_ignored) -> None:
        super().__init__()
        self.node_summary = _NodeTemporalSummary(f_node, hidden_dim)
        self.layer1 = _MeanAggSAGELayer(hidden_dim)
        self.layer2 = _MeanAggSAGELayer(hidden_dim)
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
        node_embed = self.node_summary(node_features).reshape(b * v_max, -1)
        node_mask_flat = node_mask.reshape(b * v_max)

        node_embed = self.layer1(node_embed, edge_index, node_mask_flat)
        node_embed = self.layer2(node_embed, edge_index, node_mask_flat)

        edge_context = edge_context_from_nodes(node_embed, edge_index)
        intensity = F.softplus(self.intensity_head(edge_context).squeeze(-1)) + 1e-8
        marks = self.mark_decoder(edge_context)
        return BaselineOutput(intensity=intensity, marks=marks)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""
model/baselines/rnn_perlink.py

Baseline 2 of 7 -- Shallow per-link RNN, no graph coupling.

Literature-gap row addressed: the neural half of "Underwater acoustic
connectivity/communication prediction (domain literature): ... classical
ML, shallow RNNs, often single-link, not graph-structured; no multi-node
graph coupling (correlated failures, relay effects)."

Design
------
Each endpoint node's own [k, F_NODE] feature window is summarized
independently by a small shared GRU (hidden size ``rnn_hidden``, default
24, deliberately smaller than PG-GTPP's 64-dim encoder so this stays a
"shallow RNN" in the spirit of the literature row, not a relabeled deep
model). An edge's context is the concatenation of its two endpoints' GRU
summaries -- there is no message passing, no attention, and no use of
``edge_physics`` at all: this baseline never sees any other node in the
graph and never sees the physics covariates, which is precisely the
missing capability the literature row identifies ("no multi-node graph
coupling ... no calibrated survival framing ... no cause decomposition").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mark_decoder import MarkDecoder
from .common import BaselineOutput


class ShallowRNNPerLinkModel(nn.Module):
    def __init__(self, f_node: int, rnn_hidden: int = 24, mark_trunk_width: int = 24, **_ignored) -> None:
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.gru = nn.GRU(input_size=f_node, hidden_size=rnn_hidden, batch_first=True)
        self.intensity_head = nn.Linear(2 * rnn_hidden, 1)
        self.mark_decoder = MarkDecoder(hidden_dim=2 * rnn_hidden, trunk_width=mark_trunk_width)

    def _node_embeddings(self, node_features: torch.Tensor) -> torch.Tensor:
        """node_features: [B, k, V_max, F_NODE] -> [B*V_max, rnn_hidden]"""
        b, k, v, f_node = node_features.shape
        flat = node_features.permute(0, 2, 1, 3).reshape(b * v, k, f_node)
        _, h_n = self.gru(flat)
        return h_n.squeeze(0)  # [B*V_max, rnn_hidden]

    def forward(
        self,
        node_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_physics: torch.Tensor,
    ) -> BaselineOutput:
        node_embed_flat = self._node_embeddings(node_features)
        src, dst = edge_index[0], edge_index[1]
        edge_context = torch.cat([node_embed_flat[src], node_embed_flat[dst]], dim=-1)

        intensity = F.softplus(self.intensity_head(edge_context).squeeze(-1)) + 1e-8
        marks = self.mark_decoder(edge_context)
        return BaselineOutput(intensity=intensity, marks=marks)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

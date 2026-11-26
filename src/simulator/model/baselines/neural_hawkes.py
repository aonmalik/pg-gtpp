"""
model/baselines/neural_hawkes.py

Baseline 6 of 7 -- Neural Hawkes / Neural TPP style per-edge intensity.

Literature-gap row addressed: "Neural TPP / Neural Hawkes -- Learned
intensity functions for event sequences, some with mark prediction.
Missing: rarely physically grounded in a domain-specific generative
channel model; competing-risks TPPs with named causes are rare in this
literature."

Design
------
Deliberately the most "black-box point process" baseline: an LSTM
consumes each edge's own two-endpoint raw kinematic history (the k-step
node feature window for both endpoints, concatenated at every step -- no
other node in the graph is visible, no physics covariate is visible) and
outputs the edge's own intensity directly from the final hidden state.
This is the standard neural-Hawkes recipe (intensity as a learned
function of a recurrent hidden state summarizing the point/edge's own
history) applied at the level of one candidate edge instead of one global
event stream, since PG-GTPP's setting is inherently multi-stream (one
point process per edge). No cause structure, no physical grounding, no
multi-node coupling -- the three things this row of the gap table
identifies as absent from the general neural-TPP literature.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mark_decoder import MarkDecoder
from .common import BaselineOutput


class NeuralHawkesPerEdgeModel(nn.Module):
    def __init__(self, f_node: int, hidden_dim: int = 32, mark_trunk_width: int = 32, **_ignored) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Input at each step is the concatenation of both endpoints' raw
        # node features -- an edge-centric sequence, not a node-centric one.
        self.lstm = nn.LSTM(input_size=2 * f_node, hidden_size=hidden_dim, batch_first=True)
        self.intensity_head = nn.Linear(hidden_dim, 1)
        self.mark_decoder = MarkDecoder(hidden_dim=hidden_dim, trunk_width=mark_trunk_width)

    def forward(
        self,
        node_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_physics: torch.Tensor,
    ) -> BaselineOutput:
        b, k, v_max, f_node = node_features.shape
        # permute to a per-node sequence view (matches the convention used
        # by every other baseline / the project's own NodeTemporalEncoder):
        # [B, k, V_max, F] -> [B, V_max, k, F] -> [B*V_max, k, F], so that
        # flattening indexes nodes (matching edge_index's convention), not
        # an accidental (batch, step) merge.
        flat = node_features.permute(0, 2, 1, 3).reshape(b * v_max, k, f_node)

        src, dst = edge_index[0], edge_index[1]
        # [E, k, 2*f_node] -- this edge's own two-endpoint joint history
        edge_seq = torch.cat([flat[src], flat[dst]], dim=-1)

        if edge_seq.shape[0] == 0:
            intensity = torch.zeros(0, device=node_features.device)
            marks = self.mark_decoder(torch.zeros(0, self.hidden_dim, device=node_features.device))
            return BaselineOutput(intensity=intensity, marks=marks)

        _, (h_n, _) = self.lstm(edge_seq)
        edge_state = h_n.squeeze(0)  # [E, hidden_dim]

        intensity = F.softplus(self.intensity_head(edge_state).squeeze(-1)) + 1e-8
        marks = self.mark_decoder(edge_state)
        return BaselineOutput(intensity=intensity, marks=marks)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

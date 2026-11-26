"""
model/baselines/cox_perlink.py

Baseline 1 of 7 -- Classical Cox-style proportional-hazards per-link model.

Literature-gap row addressed (Phase 2, brutal-review doc):
"Underwater acoustic connectivity/communication prediction (domain
literature): per-link SNR/PDR/throughput time-series prediction (classical
ML, shallow RNNs), often single-link, not graph-structured" -- this is the
non-neural half of that row, and also the classical statistical baseline
the identifiability discussion (Fine-Gray / cause-specific Cox literature,
Phase 5 of the roadmap doc) explicitly contrasts PG-GTPP against.

Design
------
log(lambda) = w^T * edge_physics + b, i.e. a linear (log-linear) hazard in
the same four physics covariates PG-GTPP itself uses as its physics base
intensity -- NOT the closed-form physics function (that is the existing
``physics_only`` ablation), but a *learned linear* cause-agnostic hazard
over the same inputs. No graph structure, no node history, no per-cause
decomposition, no neural representation of any kind: this is exactly a
single-event Cox proportional-hazards regression, the textbook competing-
risks-agnostic baseline.

The mark decoder receives the same 4-dim edge_physics vector as its
context (there is nothing richer to give it in a genuinely classical,
non-representation-learning baseline).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mark_decoder import MarkDecoder
from .common import BaselineOutput

PHYSICS_DIM = 4


class CoxPerLinkModel(nn.Module):
    def __init__(self, f_node: int, mark_trunk_width: int = 16, **_ignored) -> None:
        super().__init__()
        # f_node accepted (and ignored) only so this class is a drop-in
        # replacement for PGGTPPModel(f_node=..., hidden_dim=...) in the
        # shared runner script -- a genuinely classical Cox baseline does
        # not use the node feature window at all.
        self.linear_hazard = nn.Linear(PHYSICS_DIM, 1, bias=True)
        nn.init.zeros_(self.linear_hazard.weight)
        nn.init.zeros_(self.linear_hazard.bias)
        self.mark_decoder = MarkDecoder(hidden_dim=PHYSICS_DIM, trunk_width=mark_trunk_width)

    def forward(
        self,
        node_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_physics: torch.Tensor,
    ) -> BaselineOutput:
        log_hazard = self.linear_hazard(edge_physics).squeeze(-1)
        intensity = F.softplus(log_hazard) + 1e-8
        marks = self.mark_decoder(edge_physics)
        return BaselineOutput(intensity=intensity, marks=marks)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

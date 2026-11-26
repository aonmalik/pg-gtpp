"""
model/baselines/physics_soft_reg.py

Baseline 7 of 7 -- Physics-informed GNN with the physics term entered as a
SOFT LOSS REGULARIZER, not as a structurally separate, gated term.

Literature-gap row addressed: "Physics-informed GNNs (general) -- Physics
as a soft regularization term on a GNN loss. Missing: loose theoretical
connection between the physics term and the learned function's
guarantees." This is also the exact architectural contrast the brutal-
review document's Phase 2 table calls out for PG-GTPP: "Physics enters as
a structurally separate, falsifiable base predictor rather than only a
loss penalty" -- this baseline is the "only a loss penalty" arm of that
comparison, so it is the single most direct ablation for defending
PG-GTPP's trust-gated fusion design choice against the closest competing
architectural philosophy in the literature, as opposed to a strawman.

Design
------
Model: architecturally identical to ``gnn_generic.GenericGNNModel`` (same
mean-aggregation SAGE encoder, no physics-biased attention, no node
typing) -- the ONLY difference from that baseline is at training time.
Training: an auxiliary term pulls the neural intensity's log toward the
log of the closed-form physics intensity,
    L = L_TPP + beta * L_mark + gamma * MSE(log(lambda_neural), log(lambda_physics))
via ``physics_regularized_loss`` below, which wraps
``training/loss.py::compute_loss`` unchanged and adds exactly one term.
gamma (``physics_reg_weight``) is a plain float hyperparameter, matching
the project's stated preference for fixed, non-learned combination
weights over additional learned mixing machinery (see fusion.py's own
discussion of why PG-GTPP itself does NOT use a soft-regularization
scheme).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..mark_decoder import MarkDecoder
from .common import BaselineOutput, edge_context_from_nodes
from .gnn_generic import _MeanAggSAGELayer, _NodeTemporalSummary


class PhysicsSoftRegGNNModel(torch.nn.Module):
    def __init__(self, f_node: int, hidden_dim: int = 64, mark_trunk_width: int = 32, **_ignored) -> None:
        super().__init__()
        self.node_summary = _NodeTemporalSummary(f_node, hidden_dim)
        self.layer1 = _MeanAggSAGELayer(hidden_dim)
        self.layer2 = _MeanAggSAGELayer(hidden_dim)
        self.intensity_head = torch.nn.Linear(2 * hidden_dim, 1)
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


@dataclass
class PhysicsRegLossOutput:
    total_loss: torch.Tensor
    tpp_loss: torch.Tensor
    mark_loss: torch.Tensor
    physics_reg_loss: torch.Tensor
    n_events: int
    n_edges: int
    n_at_risk: int


def physics_regularized_loss(
    output: BaselineOutput,
    batch,
    edge_physics: torch.Tensor,
    mark_loss_weight: float = 1.0,
    physics_reg_weight: float = 0.1,
) -> PhysicsRegLossOutput:
    """Wraps ``training/loss.py::compute_loss`` unchanged and adds the one
    soft-regularization term this baseline is defined by. ``edge_physics``
    is the same [E,4] physics-covariate tensor already computed by the
    window builder for every other model in the comparison; the physics
    *hazard* used as the regularization target is its row-sum (the same
    aggregate-intensity convention ``BaselineOutput``/PG-GTPP's
    ``physics_only`` ablation uses), so this baseline is regularized
    toward the identical physics prediction the ``physics_only`` ablation
    reports outright and PG-GTPP fuses structurally.
    """
    from ...training.loss import compute_loss  # local import: avoid a cycle with training/loss.py

    base = compute_loss(output, batch, mark_loss_weight=mark_loss_weight)

    physics_hazard = edge_physics.clamp_min(0).sum(dim=-1).clamp_min(1e-8)
    neural_hazard = output.total_intensity().clamp_min(1e-8)
    risk = batch.edge_at_risk.float()
    if risk.sum() > 0:
        reg = ((neural_hazard.log() - physics_hazard.log()) ** 2 * risk).sum() / risk.sum()
    else:
        reg = neural_hazard.sum() * 0.0

    total = base.total_loss + physics_reg_weight * reg
    return PhysicsRegLossOutput(
        total_loss=total, tpp_loss=base.tpp_loss, mark_loss=base.mark_loss,
        physics_reg_loss=reg, n_events=base.n_events, n_edges=base.n_edges, n_at_risk=base.n_at_risk,
    )

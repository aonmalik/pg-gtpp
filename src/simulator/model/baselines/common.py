"""
model/baselines/common.py

Shared plumbing for every baseline model in this package: a single
``BaselineOutput`` dataclass that satisfies the exact interface
``training/loss.py::compute_loss`` and ``training/eval.py::evaluate``
already require of ``PGGTPPOutput`` (``.total_intensity()`` and ``.marks``
with ``.pdr_mean() / .throughput_mean() / .delay_mean() /
.negative_log_likelihood(...)``).

Design decision
---------------
Every baseline reuses ``model/mark_decoder.py::MarkDecoder`` UNCHANGED for
the communication-quality (PDR/throughput/delay) prediction head. This is
deliberate: the paper's comparison is about *event/hazard modeling*
architecture (graph structure, physical grounding, competing-risk
decomposition), not mark-distribution modeling, so holding the mark head
fixed across all 10 experiments (3 PG-GTPP ablations + 7 baselines) keeps
that axis controlled and the AUC/ECE comparison uncontaminated by
unrelated mark-head differences. Only the encoder + intensity computation
varies per baseline, exactly matching the corresponding row of the
literature-gap table (Phase 2, brutal-review document).

This also means every baseline trains and evaluates through the *exact
same* ``train.py::train()`` / ``eval.py::evaluate()`` / ``loss.py::
compute_loss()`` code already used for PG-GTPP -- no bespoke evaluation
logic per baseline, so the resulting numbers are apples-to-apples by
construction, not just by intention.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mark_decoder import MarkPrediction


@dataclass
class BaselineOutput:
    intensity: torch.Tensor  # [E], already positive (softplus'd)
    marks: MarkPrediction

    def total_intensity(self) -> torch.Tensor:
        return self.intensity

    def cause_intensities(self) -> dict[str, torch.Tensor]:
        """No competing-risk decomposition in any baseline (that
        decomposition is itself part of PG-GTPP's claimed contribution),
        so every baseline reports a single aggregate cause called
        'aggregate' -- present so the same evaluation code path than can
        introspect per-cause outputs from PG-GTPP does not need a special
        case for baselines that do not have per-cause causes."""
        return {"aggregate": self.intensity}


def edge_context_from_nodes(node_embed_flat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Shared helper: derive an [E, 2*hidden_dim] edge context by
    concatenating its two endpoint node embeddings -- identical
    convention to ``PGGTPPModel.forward`` so mark/intensity heads see the
    same kind of input across every model in the comparison."""
    src, dst = edge_index[0], edge_index[1]
    return torch.cat([node_embed_flat[src], node_embed_flat[dst]], dim=-1)

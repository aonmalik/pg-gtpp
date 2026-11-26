"""
model/mark_decoder.py

Predicts communication quality marks (PDR, throughput, delay) using
distribution-appropriate output heads instead of a naive Gaussian.

Design decisions
-----------------
1. Beta for PDR: PDR is bounded in [0, 1]; a Gaussian head places nonzero
   mass outside that range and biases both the point estimate and the
   uncertainty estimate. Flagged explicitly in the earlier brutal-review
   pass. This module uses a Beta head (alpha, beta via softplus).
2. Log-Normal for throughput and delay: both strictly positive and
   right-skewed; a Log-Normal head avoids mass on negative values.
3. Losses live here, next to the distributions they belong to, using
   torch.distributions directly rather than hand-derived formulas.

Data structures
----------------
MarkDecoder -- shared trunk + three distribution-specific heads.
MarkPrediction -- predicted distribution parameters, point-estimate means,
and an NLL method against observed marks.

Storage implications
---------------------
None -- model code only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


@dataclass
class MarkPrediction:
    """Predicted mark distribution parameters for a batch of edges."""

    pdr_alpha: torch.Tensor
    pdr_beta: torch.Tensor
    throughput_log_mean: torch.Tensor
    throughput_log_std: torch.Tensor
    delay_log_mean: torch.Tensor
    delay_log_std: torch.Tensor

    def pdr_mean(self) -> torch.Tensor:
        return self.pdr_alpha / (self.pdr_alpha + self.pdr_beta)

    def throughput_mean(self) -> torch.Tensor:
        return torch.exp(self.throughput_log_mean + 0.5 * self.throughput_log_std ** 2)

    def delay_mean(self) -> torch.Tensor:
        return torch.exp(self.delay_log_mean + 0.5 * self.delay_log_std ** 2)

    def negative_log_likelihood(
        self, pdr: torch.Tensor, throughput_bps: torch.Tensor, delay_s: torch.Tensor
    ) -> torch.Tensor:
        """Total mark NLL (summed across the three marks) for a batch of
        observed values. Inputs are clamped away from distribution-
        undefined boundary values (exact 0/1 for Beta, exact 0 for
        LogNormal)."""
        pdr_clamped = pdr.clamp(_EPS, 1.0 - _EPS)
        pdr_dist = torch.distributions.Beta(self.pdr_alpha, self.pdr_beta)
        pdr_nll = -pdr_dist.log_prob(pdr_clamped)

        throughput_dist = torch.distributions.LogNormal(self.throughput_log_mean, self.throughput_log_std)
        throughput_nll = -throughput_dist.log_prob(throughput_bps.clamp_min(_EPS))

        delay_dist = torch.distributions.LogNormal(self.delay_log_mean, self.delay_log_std)
        delay_nll = -delay_dist.log_prob(delay_s.clamp_min(_EPS))

        return pdr_nll + throughput_nll + delay_nll


class MarkDecoder(nn.Module):
    """Shared trunk + three distribution-specific heads."""

    def __init__(self, hidden_dim: int, trunk_width: int = 32) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden_dim, trunk_width), nn.ReLU())
        self.pdr_head = nn.Linear(trunk_width, 2)
        self.throughput_head = nn.Linear(trunk_width, 2)
        self.delay_head = nn.Linear(trunk_width, 2)

    def forward(self, edge_context: torch.Tensor) -> MarkPrediction:
        h = self.trunk(edge_context)

        pdr_raw = self.pdr_head(h)
        pdr_alpha = F.softplus(pdr_raw[:, 0]) + _EPS
        pdr_beta = F.softplus(pdr_raw[:, 1]) + _EPS

        throughput_raw = self.throughput_head(h)
        throughput_log_mean = throughput_raw[:, 0]
        throughput_log_std = F.softplus(throughput_raw[:, 1]) + _EPS

        delay_raw = self.delay_head(h)
        delay_log_mean = delay_raw[:, 0]
        delay_log_std = F.softplus(delay_raw[:, 1]) + _EPS

        return MarkPrediction(
            pdr_alpha=pdr_alpha, pdr_beta=pdr_beta,
            throughput_log_mean=throughput_log_mean, throughput_log_std=throughput_log_std,
            delay_log_mean=delay_log_mean, delay_log_std=delay_log_std,
        )

"""
training/loss.py

Publication-grade failure-risk objective for PG-GTPP.

The primary supervised event is the first LINK_DOWN after a reference time
for links that are UP at that reference time.  LINK_UP/recovery events are
not counted as failures.  For a model that outputs cause-specific hazards
lambda_c, the likelihood for one at-risk edge over a fixed horizon tau is

    -log L = integral_0^tau lambda(u) du
              - I(event) log lambda_c*(t_event).

The current models provide an instantaneous hazard, so the implementation
uses a piecewise-constant hazard over the forecast horizon.  Under that
explicit assumption the integral is lambda * tau and the objective is an
exact continuous-time survival/competing-risks likelihood for the first
event.  PG-GTPP uses the selected cause-specific intensity; aggregate
baselines use their single total intensity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..dataset.window_builder import WindowBatch

_EPS = 1e-8


@dataclass
class TPPLossOutput:
    total_loss: torch.Tensor
    tpp_loss: torch.Tensor
    mark_loss: torch.Tensor
    n_events: int
    n_edges: int
    n_at_risk: int


def tpp_negative_log_likelihood(
    total_intensity: torch.Tensor,
    event_occurred: torch.Tensor,
    dt: float | torch.Tensor,
    intensity_path: torch.Tensor | None = None,
    quadrature_times: torch.Tensor | None = None,
    event_time_delta: torch.Tensor | None = None,
    event_intensity: torch.Tensor | None = None,
    at_risk_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """First-event continuous-time NLL.

    ``event_intensity`` must be the cause-specific intensity at the observed
    event time when using a competing-risks model.  If omitted, ``total_intensity``
    is used, which is correct for aggregate single-risk baselines.

    With no future path, the current intensity is assumed constant over the
    horizon.  ``event_time_delta`` is required only for API completeness in
    that case; the event term uses the supplied event-time intensity, which
    equals the current intensity under the constant-hazard assumption.
    """
    lam = total_intensity.clamp_min(_EPS)
    y = event_occurred.float()
    if at_risk_mask is None:
        at_risk_mask = torch.ones_like(y)
    risk = at_risk_mask.float()

    if intensity_path is None:
        dt_tensor = torch.as_tensor(dt, dtype=lam.dtype, device=lam.device)
        survival_integral = lam * dt_tensor
    else:
        if quadrature_times is None:
            raise ValueError("quadrature_times required with intensity_path")
        path = intensity_path.clamp_min(_EPS)
        survival_integral = torch.trapz(path, quadrature_times.to(path.device), dim=0)

    event_lam = (event_intensity if event_intensity is not None else lam).clamp_min(_EPS)
    nll = survival_integral - y * torch.log(event_lam)
    # Edges that are already down at t are not part of the failure-risk
    # population.  Return zero for them so the caller can safely average.
    return nll * risk


def competing_risk_nll(
    cause_intensities: dict[str, torch.Tensor],
    event_occurred: torch.Tensor,
    event_cause_id: torch.Tensor,
    horizon_s: float | torch.Tensor,
    at_risk_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact first-event competing-risks NLL under piecewise-constant hazards.

    For censored observations, only the survival integral is used.  For an
    observed failure, log(lambda_c*) is used, where c* is the simulator's
    known failure cause.
    """
    names = list(cause_intensities.keys())
    if not names:
        raise ValueError("cause_intensities cannot be empty")
    stacked = torch.stack([cause_intensities[n].clamp_min(_EPS) for n in names], dim=-1)
    total = stacked.sum(dim=-1).clamp_min(_EPS)

    y = event_occurred.float()
    risk = at_risk_mask.float()
    cause = event_cause_id.long()

    if torch.any((y > 0.5) & ((cause < 0) | (cause >= len(names)))):
        raise ValueError("Every observed failure must have a valid cause id")

    dt = torch.as_tensor(horizon_s, dtype=total.dtype, device=total.device)
    integral = total * dt
    selected = torch.ones_like(total)
    event_mask = y > 0.5
    if event_mask.any():
        selected[event_mask] = stacked[event_mask, :].gather(
            1, cause[event_mask].unsqueeze(-1)
        ).squeeze(-1)
    nll = integral - y * torch.log(selected.clamp_min(_EPS))
    return nll * risk


def _get_event_intensity(
    output: Any,
    batch: WindowBatch,
) -> tuple[torch.Tensor, bool]:
    """Return total intensity and whether the output has competing causes."""
    if hasattr(output, "cause_intensities"):
        causes = output.cause_intensities()
        # BaselineOutput also exposes cause_intensities, but deliberately
        # returns one aggregate cause.  Four named causes means PG-GTPP.
        if set(causes.keys()) == {"aggregate"}:
            return output.total_intensity(), False
        event_mask = batch.edge_link_down_target > 0.5
        selected = torch.zeros_like(output.total_intensity())
        if event_mask.any():
            stacked = torch.stack([causes[n] for n in causes], dim=-1)
            selected[event_mask] = stacked[event_mask].gather(
                1, batch.edge_event_cause_id[event_mask].unsqueeze(-1)
            ).squeeze(-1)
        return selected, True
    return output.total_intensity(), False


def compute_loss(
    output: Any,
    batch: WindowBatch,
    mark_loss_weight: float = 1.0,
) -> TPPLossOutput:
    """Compute failure-risk survival/competing-risk NLL plus mark NLL.

    Only currently-UP edges are in the risk set.  The event is the first
    LINK_DOWN in the lookahead.  Recovery events therefore cannot create
    positive failure labels.
    """
    risk = batch.edge_at_risk
    event = batch.edge_link_down_target
    total_intensity = output.total_intensity().clamp_min(_EPS)

    if total_intensity.numel() == 0 or risk.sum() == 0:
        zero = total_intensity.sum() * 0.0
        return TPPLossOutput(zero, zero, zero, 0, int(total_intensity.numel()), 0)

    if hasattr(output, "cause_intensities") and set(output.cause_intensities().keys()) != {"aggregate"}:
        tpp_vec = competing_risk_nll(
            output.cause_intensities(),
            event_occurred=event,
            event_cause_id=batch.edge_event_cause_id,
            horizon_s=batch.lookahead_s,
            at_risk_mask=risk,
        )
    else:
        tpp_vec = tpp_negative_log_likelihood(
            total_intensity,
            event_occurred=event,
            dt=batch.lookahead_s,
            event_intensity=total_intensity,
            at_risk_mask=risk,
        )

    n_at_risk = int(risk.sum().item())
    tpp_loss = tpp_vec.sum() / max(n_at_risk, 1)

    mark_mask = (risk > 0.5) & (batch.edge_has_event_marks > 0.5)
    n_events = int(mark_mask.sum().item())
    if n_events:
        mark_nll = output.marks.negative_log_likelihood(
            batch.edge_event_pdr,
            batch.edge_event_throughput,
            batch.edge_event_delay,
        )
        mark_loss = (mark_nll * mark_mask.float()).sum() / mark_mask.float().sum().clamp_min(1.0)
    else:
        mark_loss = total_intensity.sum() * 0.0

    total_loss = tpp_loss + mark_loss_weight * mark_loss
    return TPPLossOutput(
        total_loss=total_loss,
        tpp_loss=tpp_loss,
        mark_loss=mark_loss,
        n_events=n_events,
        n_edges=int(total_intensity.numel()),
        n_at_risk=n_at_risk,
    )


def cause_specific_log_likelihood(
    cause_intensities: dict[str, torch.Tensor],
    event_causes: torch.Tensor,
    total_intensity: torch.Tensor,
) -> torch.Tensor:
    """Return log(lambda_c*) for observed competing-risk events.

    Kept as a utility for diagnostics; the survival integral belongs in
    ``competing_risk_nll``.
    """
    names = list(cause_intensities.keys())
    stacked = torch.stack([cause_intensities[n] for n in names], dim=-1).clamp_min(_EPS)
    selected = stacked.gather(-1, event_causes.long().clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return torch.log(selected)

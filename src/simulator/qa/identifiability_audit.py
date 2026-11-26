"""
qa/identifiability_audit.py

The ambiguous-event-fraction metric and the pairwise cause-covariate
correlation matrix — the practical, measurable stand-ins for the
competing-risks identifiability concern raised against Paper 1's theoretical
design (correlated causes threaten the classical non-identifiability result
for competing risks; these two metrics let a specific generated dataset be
checked against that concern rather than leaving it as an abstract worry).

Design decisions
-----------------
1. **Both metrics are computed from the same recomputation pass.** For every
   LINK_DOWN event, this module recomputes the four cause-specific hazard
   values at the event's exact accepted time (using the same
   ``HazardEvaluator`` and position-interpolation rule the generator itself
   used — see ``qa/time_rescaling_test.py``'s design notes for why this
   consistency matters). The ambiguity ratio and the raw four-channel
   hazard vector both come from that single recomputation, rather than two
   separate passes — this is a deliberate efficiency choice, but also a
   correctness one: it guarantees both metrics are evaluated under
   identical conditions.
2. **Ambiguous-event fraction** uses ``CauseHazards.ambiguity_ratio()``
   (already implemented in Phase 3's ``physics_intensity.py``) — the ratio
   of the winning cause's hazard to the runner-up's at the accepted event
   time. An event is flagged "ambiguous" if this ratio falls below a
   threshold (default 1.5x, matching the QA design document's stated
   example threshold). This is deliberately an event-level diagnostic, not
   a training-time gate — the module reports the fraction, it does not
   filter events.
3. **Correlation matrix** is computed only over the *raw* four hazard
   channels for LINK_DOWN events (not over the full continuous covariate
   trajectory), because that is the population any downstream identifiability
   claim actually needs to hold over — the model only ever needs to attribute
   cause for events that occurred, not for every millisecond of the mission.
4. **Pooling across edges.** Like the time-rescaling test, this audit pools
   over every edge in the mission — the identifiability concern (are the
   four causes distinguishable from each other) is a property of the shared
   generative model, not of any one edge, so pooling is the right unit of
   analysis and also gives a much larger, more statistically meaningful
   sample than any single edge's event count would.

Data structures
----------------
``IdentifiabilityAuditResult`` — event count, ambiguous fraction, the raw
4x4 correlation matrix (with labeled axes), and the maximum off-diagonal
correlation (the single number the QA design document's 0.7 threshold
check applies to).

Storage implications
---------------------
None — pure recomputation from already-generated mission artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..event_types import CommunicationEvent, EventType
from ..physics_intensity import HazardEvaluator
from .trajectory_query import interpolate_positions_batch

_CAUSE_LABELS = ("geo", "wave", "doppler", "current")


@dataclass
class IdentifiabilityAuditResult:
    """Result of the pooled ambiguous-event-fraction and correlation audit."""

    n_events: int
    ambiguous_fraction: float
    ambiguity_threshold: float
    correlation_matrix: np.ndarray  # 4x4, ordered per _CAUSE_LABELS
    cause_labels: tuple[str, str, str, str]
    max_off_diagonal_correlation: float
    degenerate_causes: tuple[str, ...]

    def flags_identifiability_risk(self, correlation_threshold: float = 0.7) -> bool:
        """True if the pooled correlation matrix shows any pair of causes
        correlated beyond the QA design document's stated risk threshold —
        the concrete, checkable stand-in for "the disjoint-covariate
        assumption behind the identifiability proposition may not hold in
        this dataset"."""
        return self.max_off_diagonal_correlation > correlation_threshold

    def summary(self) -> str:
        risk = "AT RISK" if self.flags_identifiability_risk() else "OK"
        lines = [
            f"Identifiability audit: n_events={self.n_events}, "
            f"ambiguous_fraction={self.ambiguous_fraction:.3f} "
            f"(threshold ratio < {self.ambiguity_threshold}x), "
            f"max_off_diag_corr={self.max_off_diagonal_correlation:.3f} -> {risk}",
        ]
        if self.degenerate_causes:
            lines.append(
                f"  NOTE: {', '.join(self.degenerate_causes)} showed (near-)zero variance across "
                f"all events — its correlation with other causes is undefined (NaN below), which "
                f"means this cause never meaningfully contributed to any observed failure. This is "
                f"itself a diagnostic finding (see the calibration caveats in the delivery notes), "
                f"not a computation error."
            )
        lines.append("Correlation matrix (" + ", ".join(self.cause_labels) + "):")
        for i, row_label in enumerate(self.cause_labels):
            row_values = " ".join(f"{v:+.3f}" if np.isfinite(v) else "   nan" for v in self.correlation_matrix[i])
            lines.append(f"  {row_label:>8}: {row_values}")
        return "\n".join(lines)


def audit_identifiability(
    hazard_evaluator: HazardEvaluator,
    array: np.ndarray,
    node_ids: list[str],
    timestamps: np.ndarray,
    events: list[CommunicationEvent],
    ambiguity_threshold: float = 1.5,
) -> IdentifiabilityAuditResult:
    """Run the pooled ambiguous-event-fraction and cause-covariate
    correlation audit over every LINK_DOWN event in the mission.
    """
    down_events = [e for e in events if e.event_type == EventType.LINK_DOWN]

    if not down_events:
        empty_matrix = np.full((4, 4), np.nan)
        return IdentifiabilityAuditResult(
            n_events=0, ambiguous_fraction=float("nan"), ambiguity_threshold=ambiguity_threshold,
            correlation_matrix=empty_matrix, cause_labels=_CAUSE_LABELS,
            max_off_diagonal_correlation=float("nan"), degenerate_causes=(),
        )

    v_index = {node_id: i for i, node_id in enumerate(node_ids)}
    t_query = np.array([e.timestamp_s for e in down_events], dtype=np.float64)
    i_idx = np.array([v_index[e.edge_i] for e in down_events])
    j_idx = np.array([v_index[e.edge_j] for e in down_events])

    # Recompute one edge's position at a time is unnecessary here since
    # different events may involve different node pairs; interpolate each
    # event's own (i, j) pair using its own query time, vectorized per node
    # index rather than per event, by grouping — but since `interpolate_positions_batch`
    # already vectorizes over an array of (v_idx fixed, t_query varying) calls,
    # and events here have *varying* v_idx too, evaluate node-by-node via a
    # small Python loop over unique node indices, which is cheap (bounded by
    # fleet size, not event count).
    def batch_positions(idx_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.zeros(len(down_events))
        y = np.zeros(len(down_events))
        z = np.zeros(len(down_events))
        vx = np.zeros(len(down_events))
        vy = np.zeros(len(down_events))
        for v in np.unique(idx_array):
            mask = idx_array == v
            xv, yv, zv, vxv, vyv = interpolate_positions_batch(array, int(v), timestamps, t_query[mask])
            x[mask], y[mask], z[mask], vx[mask], vy[mask] = xv, yv, zv, vxv, vyv
        return x, y, z, vx, vy

    xi, yi, zi, vxi, vyi = batch_positions(i_idx)
    xj, yj, zj, vxj, vyj = batch_positions(j_idx)
    distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)

    geo, wave, doppler, current = hazard_evaluator.cause_hazards_batch(
        distance, xi, yi, zi, xj, yj, zj, vxi, vyi, vxj, vyj, t_query
    )
    hazards_matrix = np.stack([geo, wave, doppler, current], axis=1)  # [n_events, 4]

    sorted_hazards = np.sort(hazards_matrix, axis=1)
    top = sorted_hazards[:, -1]
    second = sorted_hazards[:, -2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(second > 0.0, top / second, np.where(top > 0.0, np.inf, 1.0))
    ambiguous_fraction = float(np.mean(ratio < ambiguity_threshold))

    # A cause with (near-)zero variance across all events cannot have a
    # meaningful correlation with anything — flag it explicitly rather than
    # letting np.corrcoef silently emit NaN via a 0/0 division.
    channel_std = hazards_matrix.std(axis=0)
    degenerate_mask = channel_std < 1e-12
    degenerate_causes = tuple(label for label, is_degenerate in zip(_CAUSE_LABELS, degenerate_mask) if is_degenerate)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation_matrix = np.corrcoef(hazards_matrix, rowvar=False)
    off_diagonal_mask = ~np.eye(4, dtype=bool)
    finite_off_diag = correlation_matrix[off_diagonal_mask]
    finite_off_diag = finite_off_diag[np.isfinite(finite_off_diag)]
    max_off_diag = float(np.max(np.abs(finite_off_diag))) if len(finite_off_diag) else float("nan")

    return IdentifiabilityAuditResult(
        n_events=len(down_events),
        ambiguous_fraction=ambiguous_fraction,
        ambiguity_threshold=ambiguity_threshold,
        correlation_matrix=correlation_matrix,
        cause_labels=_CAUSE_LABELS,
        max_off_diagonal_correlation=max_off_diag,
        degenerate_causes=degenerate_causes,
    )

"""
graph_builder.py

Candidate-edge construction: which node pairs are close enough to be
communication candidates at a given instant, and the theoretical maximum
viable range used to size that candidate set.

Design decisions
-----------------
1. Candidate edges are restricted to two physically meaningful pairings —
   USV-AUV and AUV-IoUT — matching the network topology from the PG-GTPP
   research design (USV-AUV acoustic/USBL links for localization, AUV-IoUT
   links for data collection). AUV-AUV relay links, mentioned as a possible
   future extension in the research design, are deliberately **not**
   included here; scope-creeping them in without a corresponding hazard/
   mark model would silently misrepresent what this phase actually
   generates.
2. ``theoretical_max_range_m`` inverts the *spreading-loss* term of the
   Urick link budget (ignoring the smaller, frequency-dependent absorption
   term) to get a fast, closed-form estimate of the range at which SNR
   margin reaches zero. This is intentionally an approximation — solving
   the full transcendental equation including absorption has no closed
   form — used only to *size* the candidate-edge search radius via
   ``GraphConfig.candidate_range_factor`` (default 1.5x), not as a claim
   about the exact viable range. The exact, instant-by-instant viable range
   (including absorption) is always evaluated separately via
   ``AcousticChannel.snr_margin_db`` wherever hazard/event logic actually
   needs it.
3. Candidate-edge membership is recomputed directly from node positions at
   each query time — never cached or stored — consistent with the
   "recompute, don't store" storage-efficiency principle: the graph
   structure ``G(t)`` is a derived quantity, not a persisted one.

Data structures
----------------
``CandidateEdge`` — a lightweight (node_i, node_j) pair with a fixed,
declared ``edge_kind`` ("usv_auv" or "auv_iout").

Storage implications
---------------------
None — this module produces candidate-edge lists on demand from already-
loaded node state; nothing here is persisted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .mission_state import NodeState, NodeType
from .physics import AcousticChannel


@dataclass(frozen=True)
class CandidateEdge:
    """An undirected candidate communication edge between two nodes.

    ``edge_key`` is computed once at construction (via ``__post_init__``,
    using ``object.__setattr__`` since the dataclass is frozen) rather than
    recomputed on every access. Profiling the event generator's hot path
    showed ``edge_key`` being accessed hundreds of thousands of times per
    mission (edge_states dict lookups happen every thinning round); turning
    a repeatedly-recomputed property into a one-time-computed field removed
    that cost from the hot path entirely.
    """

    node_i: str
    node_j: str
    edge_kind: str  # "usv_auv" or "auv_iout"
    edge_key: tuple[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_key", tuple(sorted((self.node_i, self.node_j))))


def theoretical_max_range_m(channel: AcousticChannel) -> float:
    """Closed-form estimate of the range at which SNR margin reaches zero,
    using spreading loss only (absorption ignored — see module docstring,
    point 2). Returns a finite, positive distance in meters.

    Uses ``channel.total_in_band_noise_db()`` (bandwidth-corrected), not
    the raw per-Hz ``ambient_noise_level_db()`` — this function previously
    used the latter directly, which was dimensionally inconsistent with
    ``source_level_db`` in the same way ``AcousticChannel.snr_db`` was
    before its post-Phase-4 fix (see ``AcousticConfig``'s docstring).
    """
    config = channel.config
    noise = channel.total_in_band_noise_db()
    target_spreading_db = config.source_level_db - noise - config.snr_min_db
    if target_spreading_db <= 0.0:
        # Degenerate configuration: even at zero range the link cannot meet
        # snr_min_db once noise is accounted for. Returning a small positive
        # floor (rather than 0 or a negative/NaN value) keeps downstream
        # candidate-range arithmetic well-defined; callers should treat a
        # very small theoretical range as a configuration warning, not
        # silently proceed as if the link budget were healthy.
        return 1.0
    log10_d = target_spreading_db / (10.0 * config.spreading_loss_exponent)
    return 10.0 ** log10_d


def _distance_m(a: NodeState, b: NodeState) -> float:
    return math.sqrt((a.x_m - b.x_m) ** 2 + (a.y_m - b.y_m) ** 2 + (a.z_m - b.z_m) ** 2)


def build_candidate_edges(
    nodes: list[NodeState],
    channel: AcousticChannel,
    candidate_range_factor: float,
) -> list[CandidateEdge]:
    """Build the candidate-edge list at a single instant, given each node's
    current state. Only USV-AUV and AUV-IoUT pairs within
    ``candidate_range_factor * theoretical_max_range_m(channel)`` are
    included.
    """
    if candidate_range_factor <= 1.0:
        raise ValueError(f"candidate_range_factor must be > 1.0, got {candidate_range_factor}")

    max_range = theoretical_max_range_m(channel) * candidate_range_factor

    usv_nodes = [n for n in nodes if n.node_type == NodeType.USV]
    auv_nodes = [n for n in nodes if n.node_type == NodeType.AUV]
    iout_nodes = [n for n in nodes if n.node_type == NodeType.IOUT]

    edges: list[CandidateEdge] = []
    for usv in usv_nodes:
        for auv in auv_nodes:
            if _distance_m(usv, auv) <= max_range:
                edges.append(CandidateEdge(node_i=usv.node_id, node_j=auv.node_id, edge_kind="usv_auv"))
    for auv in auv_nodes:
        for iout in iout_nodes:
            if _distance_m(auv, iout) <= max_range:
                edges.append(CandidateEdge(node_i=auv.node_id, node_j=iout.node_id, edge_kind="auv_iout"))
    return edges

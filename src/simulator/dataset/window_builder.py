"""
dataset/window_builder.py

Builds a single training sample (a "window") from an already-generated
mission's node trajectory array and event log, at one reference time.

Design decisions
-----------------
1. **The physics base intensity IS the edge's physics covariate vector.**
   ``HazardEvaluator.cause_hazards`` (Phase 3) already computes each cause's
   closed-form hazard as a function of physics covariates only (SNR margin,
   wave height-rate, relative speed, current speed) — there is no separate
   "raw covariate vector" to compute independently. This window builder
   therefore uses ``CauseHazards`` directly as both (a) part of the encoder's
   edge input features and (b) the physics term the model's fusion gate
   combines with a neural correction. This matches the architecture design's
   "physics enters at exactly one principled point" requirement.
2. **One window = one static graph snapshot + a short-horizon lookahead
   target**, not yet a full continuous-time training stream. Phase 5's scope
   is model architecture and the tensor contract it consumes — a complete
   TPP-loss training pipeline (event-stream windows, multi-horizon targets)
   is a later phase. The target here is deliberately simple and useful on
   its own: for each candidate edge, whether it experiences a LINK_DOWN or
   LINK_UP transition within ``(t, t+lookahead_s]``, which is enough to
   validate the model's forward pass and to prototype a first loss function
   later without having committed to the full TPP training loop's design
   yet.
3. **Reference-time node window uses the same sample-grid arithmetic as
   ``qa/trajectory_query.py``** (regular grid, direct index lookup) for
   consistency with the rest of the codebase, though here the window only
   ever needs whole-sample indices (no interpolation), since window
   reference times are chosen to land on trajectory samples.
4. **Candidate edges reuse ``graph_builder.build_candidate_edges``
   unchanged** — the same USV-AUV / AUV-IoUT restricted, range-masked
   candidate set the simulator itself uses, so the model is trained on
   exactly the graph structure the generative process assumes.

Data structures
----------------
``WindowSample`` — one training example: a node feature window tensor, edge
index tensor, edge physics-hazard tensor, and lookahead event-outcome
targets. ``build_window`` constructs one from mission artifacts;
``collate_windows`` batches a list of them (variable node/edge counts across
missions require padding + a batch-pointer scheme, not a naive stack).

Storage implications
---------------------
None — every tensor here is built on demand from already-loaded, in-memory
mission arrays, per the "no precomputed windows stored" storage principle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..event_types import CommunicationEvent, EventType
from ..graph_builder import build_candidate_edges
from ..mission_state import NodeState, NodeType
from ..physics_intensity import HazardEvaluator

_X_COL, _Y_COL, _Z_COL = 0, 1, 2
_NODE_TYPE_TO_ID = {NodeType.USV: 0, NodeType.AUV: 1, NodeType.IOUT: 2}
_EDGE_KIND_TO_ID = {"usv_auv": 0, "auv_iout": 1}


@dataclass
class WindowSample:
    """One training example built from a mission at a single reference time.

    Shapes (``k`` = window length, ``V`` = node count in the window,
    ``E`` = candidate edge count at the reference time):
    """

    node_features: torch.Tensor  # [k, V, F_NODE], float32
    node_type_ids: torch.Tensor  # [V], int64 in {0,1,2} (USV/AUV/IoUT)
    edge_index: torch.Tensor  # [2, E], int64, indices into the V axis
    edge_kind_ids: torch.Tensor  # [E], int64 in {0,1} (usv_auv/auv_iout)
    edge_physics: torch.Tensor  # [E, 4], float32 (geo, wave, doppler, current)
    edge_link_down_target: torch.Tensor  # [E], float32 in {0,1}
    edge_link_up_target: torch.Tensor  # [E], float32 in {0,1}
    edge_any_event_target: torch.Tensor  # [E], float32 in {0,1} -- deprecated aggregate transition target
    edge_at_risk: torch.Tensor  # [E], float32; 1 only when the link is UP at reference time and is eligible for failure-risk forecasting
    edge_event_time_delta: torch.Tensor  # [E], seconds to first LINK_DOWN; lookahead_s when censored
    edge_event_cause_id: torch.Tensor  # [E], -1 when censored, otherwise 0..3 in CAUSE_NAMES order
    edge_event_pdr: torch.Tensor  # [E], float32 -- marks of the FIRST event in (t, t+lookahead], 0 if none
    edge_event_throughput: torch.Tensor  # [E], float32
    edge_event_delay: torch.Tensor  # [E], float32
    edge_has_event_marks: torch.Tensor  # [E], float32 in {0,1} -- mask: only edges with an event have valid marks
    mission_id: str
    reference_time_s: float
    lookahead_s: float


def _events_in_window(
    events: list[CommunicationEvent], t0: float, t1: float
) -> dict[tuple[str, str], list[CommunicationEvent]]:
    by_edge: dict[tuple[str, str], list[CommunicationEvent]] = {}
    for e in events:
        if t0 < e.timestamp_s <= t1:
            by_edge.setdefault(e.edge_key, []).append(e)
    return by_edge



_CAUSE_TO_ID = {"geo": 0, "wave": 1, "doppler": 2, "current": 3}


def _edge_state_at(events: list[CommunicationEvent], edge_key: tuple[str, str], t_ref: float) -> str:
    """Return the simulator link state immediately at ``t_ref``.

    Event generation starts every unseen candidate edge in UP state. Applying
    all prior state changes therefore reconstructs the state without using
    any future information.
    """
    state = "up"
    prior = [e for e in events if e.edge_key == edge_key and e.timestamp_s <= t_ref]
    prior.sort(key=lambda e: e.timestamp_s)
    for event in prior:
        state = "down" if event.event_type == EventType.LINK_DOWN else "up"
    return state


def build_window(
    mission_id: str,
    array: np.ndarray,
    node_ids: list[str],
    node_types: dict[str, NodeType],
    timestamps: np.ndarray,
    events: list[CommunicationEvent],
    hazard_evaluator: HazardEvaluator,
    candidate_range_factor: float,
    reference_t_idx: int,
    window_length: int,
    lookahead_s: float,
) -> WindowSample:
    """Build one ``WindowSample`` from mission artifacts.

    Parameters
    ----------
    reference_t_idx: index into ``timestamps`` marking "now" (t). The node
        feature window covers
        ``[max(0, reference_t_idx - window_length + 1), reference_t_idx]``,
        left-padded with the earliest available sample if the mission has
        fewer than ``window_length`` samples before ``reference_t_idx``.
    lookahead_s: the target horizon — edges are labeled by whether they
        experience a LINK_DOWN / LINK_UP event within
        ``(t, t + lookahead_s]``.
    """
    if not (0 <= reference_t_idx < len(timestamps)):
        raise ValueError(f"reference_t_idx {reference_t_idx} out of range [0, {len(timestamps)})")
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")

    t_ref = float(timestamps[reference_t_idx])

    start_idx = max(0, reference_t_idx - window_length + 1)
    window_array = array[start_idx : reference_t_idx + 1]  # [<=window_length, V, F_NODE]
    if window_array.shape[0] < window_length:
        pad_count = window_length - window_array.shape[0]
        pad = np.repeat(window_array[0:1], pad_count, axis=0)
        window_array = np.concatenate([pad, window_array], axis=0)
    node_features = torch.from_numpy(window_array.astype(np.float32))

    node_type_ids = torch.tensor(
        [_NODE_TYPE_TO_ID[node_types[nid]] for nid in node_ids], dtype=torch.int64
    )

    ref_nodes = [
        NodeState(
            node_id=nid, node_type=node_types[nid], timestamp_s=t_ref,
            x_m=float(array[reference_t_idx, i, _X_COL]),
            y_m=float(array[reference_t_idx, i, _Y_COL]),
            z_m=float(array[reference_t_idx, i, _Z_COL]),
        )
        for i, nid in enumerate(node_ids)
    ]
    candidate_edges = build_candidate_edges(ref_nodes, hazard_evaluator.acoustic_channel, candidate_range_factor)

    v_index = {nid: i for i, nid in enumerate(node_ids)}
    edge_index_list: list[tuple[int, int]] = []
    edge_kind_list: list[int] = []
    edge_physics_list: list[list[float]] = []

    for edge in candidate_edges:
        i, j = v_index[edge.node_i], v_index[edge.node_j]
        xi, yi, zi = (float(array[reference_t_idx, i, c]) for c in (_X_COL, _Y_COL, _Z_COL))
        xj, yj, zj = (float(array[reference_t_idx, j, c]) for c in (_X_COL, _Y_COL, _Z_COL))
        vxi, vyi = float(array[reference_t_idx, i, 3]), float(array[reference_t_idx, i, 4])
        vxj, vyj = float(array[reference_t_idx, j, 3]), float(array[reference_t_idx, j, 4])
        distance = float(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2))

        hazards = hazard_evaluator.cause_hazards(distance, xi, yi, zi, xj, yj, zj, vxi, vyi, vxj, vyj, t_ref)

        edge_index_list.append((i, j))
        edge_kind_list.append(_EDGE_KIND_TO_ID[edge.edge_kind])
        edge_physics_list.append([hazards.geo, hazards.wave, hazards.doppler, hazards.current])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.int64).T  # [2, E]
        edge_kind_ids = torch.tensor(edge_kind_list, dtype=torch.int64)
        edge_physics = torch.tensor(edge_physics_list, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64)
        edge_kind_ids = torch.zeros((0,), dtype=torch.int64)
        edge_physics = torch.zeros((0, 4), dtype=torch.float32)

    events_by_edge = _events_in_window(events, t_ref, t_ref + lookahead_s)
    n_edges = len(candidate_edges)
    down_targets = torch.zeros(n_edges, dtype=torch.float32)
    up_targets = torch.zeros(n_edges, dtype=torch.float32)
    any_event_targets = torch.zeros(n_edges, dtype=torch.float32)
    at_risk = torch.zeros(n_edges, dtype=torch.float32)
    event_time_delta = torch.full((n_edges,), float(lookahead_s), dtype=torch.float32)
    event_cause_id = torch.full((n_edges,), -1, dtype=torch.int64)
    event_pdr = torch.zeros(n_edges, dtype=torch.float32)
    event_throughput = torch.zeros(n_edges, dtype=torch.float32)
    event_delay = torch.zeros(n_edges, dtype=torch.float32)
    has_event_marks = torch.zeros(n_edges, dtype=torch.float32)

    for e_idx, edge in enumerate(candidate_edges):
        current_state = _edge_state_at(events, edge.edge_key, t_ref)
        if current_state != "up":
            # Recovery is not a failure. Keep the edge in the graph so it can
            # still provide messages to neighboring nodes, but exclude it from
            # the failure-risk likelihood and metrics.
            continue

        at_risk[e_idx] = 1.0
        edge_events = events_by_edge.get(edge.edge_key, [])
        down_events = [ev for ev in edge_events if ev.event_type == EventType.LINK_DOWN]
        if down_events:
            first_down = min(down_events, key=lambda ev: ev.timestamp_s)
            down_targets[e_idx] = 1.0
            any_event_targets[e_idx] = 1.0
            event_time_delta[e_idx] = float(first_down.timestamp_s - t_ref)
            event_cause_id[e_idx] = _CAUSE_TO_ID[first_down.cause.value]
            event_pdr[e_idx] = first_down.marks.pdr
            event_throughput[e_idx] = first_down.marks.throughput_bps
            event_delay[e_idx] = first_down.marks.delay_s
            has_event_marks[e_idx] = 1.0
        # ``up_targets`` is retained for backward compatibility/diagnostics,
        # but LINK_UP is never the primary failure target.
        if any(ev.event_type == EventType.LINK_UP for ev in edge_events):
            up_targets[e_idx] = 1.0

    return WindowSample(
        node_features=node_features,
        node_type_ids=node_type_ids,
        edge_index=edge_index,
        edge_kind_ids=edge_kind_ids,
        edge_physics=edge_physics,
        edge_link_down_target=down_targets,
        edge_link_up_target=up_targets,
        edge_any_event_target=any_event_targets,
        edge_at_risk=at_risk,
        edge_event_time_delta=event_time_delta,
        edge_event_cause_id=event_cause_id,
        edge_event_pdr=event_pdr,
        edge_event_throughput=event_throughput,
        edge_event_delay=event_delay,
        edge_has_event_marks=has_event_marks,
        mission_id=mission_id,
        reference_time_s=t_ref,
        lookahead_s=lookahead_s,
    )


@dataclass
class WindowBatch:
    """A batch of ``WindowSample``s with variable node/edge counts, packed
    via padding (nodes) and a batch-pointer scheme (edges)."""

    node_features: torch.Tensor  # [B, k, V_max, F_NODE]
    node_type_ids: torch.Tensor  # [B, V_max]
    node_mask: torch.Tensor  # [B, V_max], 1 for real nodes, 0 for padding
    edge_index: torch.Tensor  # [2, sum(E_b)] — indices offset per-sample into the flattened [B*V_max] node axis
    edge_batch: torch.Tensor  # [sum(E_b)], which batch element each edge belongs to
    edge_kind_ids: torch.Tensor  # [sum(E_b)]
    edge_physics: torch.Tensor  # [sum(E_b), 4]
    edge_link_down_target: torch.Tensor  # [sum(E_b)]
    edge_link_up_target: torch.Tensor  # [sum(E_b)]
    edge_any_event_target: torch.Tensor  # [sum(E_b)]
    edge_at_risk: torch.Tensor  # [sum(E_b)]
    edge_event_time_delta: torch.Tensor  # [sum(E_b)]
    edge_event_cause_id: torch.Tensor  # [sum(E_b)]
    edge_event_pdr: torch.Tensor  # [sum(E_b)]
    edge_event_throughput: torch.Tensor  # [sum(E_b)]
    edge_event_delay: torch.Tensor  # [sum(E_b)]
    edge_has_event_marks: torch.Tensor  # [sum(E_b)]
    lookahead_s: float  # shared across the batch (see collate_windows validation)
    mission_ids: list[str]


def collate_windows(samples: list[WindowSample]) -> WindowBatch:
    """Pack a list of ``WindowSample``s (possibly from different missions,
    with different node/edge counts) into one ``WindowBatch``.

    Node features are zero-padded to the batch's max node count. Edge
    indices are offset per-sample so they index directly into the padded
    ``[B, V_max, ...]`` node tensor when flattened to ``[B*V_max, ...]``
    (the standard "block-diagonal batching" trick for variable-size graphs).
    """
    if not samples:
        raise ValueError("collate_windows requires at least one sample")

    k = samples[0].node_features.shape[0]
    f_node = samples[0].node_features.shape[2]
    v_max = max(s.node_features.shape[1] for s in samples)
    batch_size = len(samples)

    node_features = torch.zeros(batch_size, k, v_max, f_node, dtype=torch.float32)
    node_type_ids = torch.zeros(batch_size, v_max, dtype=torch.int64)
    node_mask = torch.zeros(batch_size, v_max, dtype=torch.float32)

    edge_index_parts = []
    edge_batch_parts = []
    edge_kind_parts = []
    edge_physics_parts = []
    down_target_parts = []
    up_target_parts = []
    any_event_parts = []
    at_risk_parts = []
    event_time_parts = []
    event_cause_parts = []
    event_pdr_parts = []
    event_throughput_parts = []
    event_delay_parts = []
    has_event_marks_parts = []

    lookahead_s = samples[0].lookahead_s
    for sample in samples:
        if sample.lookahead_s != lookahead_s:
            raise ValueError(
                f"collate_windows requires all samples to share the same lookahead_s "
                f"(got {sample.lookahead_s} and {lookahead_s}) -- the TPP loss's piecewise-"
                f"constant-intensity assumption is anchored to a single interval length per batch."
            )

    for b, sample in enumerate(samples):
        v = sample.node_features.shape[1]
        node_features[b, :, :v, :] = sample.node_features
        node_type_ids[b, :v] = sample.node_type_ids
        node_mask[b, :v] = 1.0

        n_edges = sample.edge_index.shape[1]
        if n_edges > 0:
            edge_index_parts.append(sample.edge_index + b * v_max)
            edge_batch_parts.append(torch.full((n_edges,), b, dtype=torch.int64))
            edge_kind_parts.append(sample.edge_kind_ids)
            edge_physics_parts.append(sample.edge_physics)
            down_target_parts.append(sample.edge_link_down_target)
            up_target_parts.append(sample.edge_link_up_target)
            any_event_parts.append(sample.edge_any_event_target)
            at_risk_parts.append(sample.edge_at_risk)
            event_time_parts.append(sample.edge_event_time_delta)
            event_cause_parts.append(sample.edge_event_cause_id)
            event_pdr_parts.append(sample.edge_event_pdr)
            event_throughput_parts.append(sample.edge_event_throughput)
            event_delay_parts.append(sample.edge_event_delay)
            has_event_marks_parts.append(sample.edge_has_event_marks)

    edge_index = torch.cat(edge_index_parts, dim=1) if edge_index_parts else torch.zeros((2, 0), dtype=torch.int64)
    edge_batch = torch.cat(edge_batch_parts) if edge_batch_parts else torch.zeros((0,), dtype=torch.int64)
    edge_kind_ids = torch.cat(edge_kind_parts) if edge_kind_parts else torch.zeros((0,), dtype=torch.int64)
    edge_physics = torch.cat(edge_physics_parts) if edge_physics_parts else torch.zeros((0, 4), dtype=torch.float32)
    down_targets = torch.cat(down_target_parts) if down_target_parts else torch.zeros((0,), dtype=torch.float32)
    up_targets = torch.cat(up_target_parts) if up_target_parts else torch.zeros((0,), dtype=torch.float32)
    any_event_targets = torch.cat(any_event_parts) if any_event_parts else torch.zeros((0,), dtype=torch.float32)
    at_risk_targets = torch.cat(at_risk_parts) if at_risk_parts else torch.zeros((0,), dtype=torch.float32)
    event_time_deltas = torch.cat(event_time_parts) if event_time_parts else torch.zeros((0,), dtype=torch.float32)
    event_cause_ids = torch.cat(event_cause_parts) if event_cause_parts else torch.zeros((0,), dtype=torch.int64)
    event_pdr = torch.cat(event_pdr_parts) if event_pdr_parts else torch.zeros((0,), dtype=torch.float32)
    event_throughput = torch.cat(event_throughput_parts) if event_throughput_parts else torch.zeros((0,), dtype=torch.float32)
    event_delay = torch.cat(event_delay_parts) if event_delay_parts else torch.zeros((0,), dtype=torch.float32)
    has_event_marks = torch.cat(has_event_marks_parts) if has_event_marks_parts else torch.zeros((0,), dtype=torch.float32)

    return WindowBatch(
        node_features=node_features, node_type_ids=node_type_ids, node_mask=node_mask,
        edge_index=edge_index, edge_batch=edge_batch, edge_kind_ids=edge_kind_ids,
        edge_physics=edge_physics, edge_link_down_target=down_targets, edge_link_up_target=up_targets,
        edge_any_event_target=any_event_targets, edge_at_risk=at_risk_targets,
        edge_event_time_delta=event_time_deltas, edge_event_cause_id=event_cause_ids,
        edge_event_pdr=event_pdr,
        edge_event_throughput=event_throughput, edge_event_delay=event_delay,
        edge_has_event_marks=has_event_marks, lookahead_s=lookahead_s,
        mission_ids=[s.mission_id for s in samples],
    )

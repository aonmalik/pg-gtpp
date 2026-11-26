"""
event_generator.py

Ground-truth communication event generation via adaptive Ogata thinning over
a two-state (UP/DOWN) marked point process per candidate edge, driven by the
competing-risks cause-specific hazards from ``physics_intensity.py``.

Design decisions
-----------------
1. **Two-state process per edge.** While UP, an edge fails at the total
   competing-risks rate (``CauseHazards.total``); the accepted failure's
   cause is the dominant cause at the accepted instant
   (``CauseHazards.dominant_cause()``), matching the research design's
   ``argmax_c lambda^c(t)`` cause-assignment rule. While DOWN, an edge
   recovers at ``HazardEvaluator.recovery_hazard`` — a single, symmetric
   hazard (no cause decomposition needed for recovery, consistent with
   ``event_types.py``'s schema where ``LINK_UP`` events carry no cause).
2. **Adaptive, bracket-local Ogata thinning.** Node positions are only known
   exactly at the trajectory's stored sample times (1 Hz by default); within
   each inter-sample bracket ``[t_k, t_{k+1})``, this module reconstructs
   position via linear extrapolation from the bracket-start sample using its
   stored velocity. This reconstruction is **exact**, not approximate, for
   every motion pattern ``trajectory_generator.py`` actually produces
   (constant-velocity waypoint legs, and locally near-linear circular
   motion at the 1 Hz sampling rate) — a deliberate consistency property
   between the two modules, not a coincidence. Within a bracket, a
   dominating rate is estimated by probing the relevant hazard (failure
   total or recovery) at ``GraphConfig.n_thinning_probes_per_bracket``
   points and taking the max, scaled by
   ``GraphConfig.dominating_rate_safety_factor``. Candidate event times are
   then proposed via the standard thinning accept/reject rule; if a
   proposal falls beyond the current bracket, the (memoryless) exponential
   clock simply continues into the next bracket with a freshly estimated
   dominating rate — the standard approach for simulating time-
   inhomogeneous point processes with a locally-bounded intensity, and
   exactly the mechanism the QA verification framework's "Communication
   failure generator (thinning)" section specifies.
3. **Candidate edges are re-evaluated at each bracket's start position**
   (via ``graph_builder.build_candidate_edges``), not fixed for the whole
   mission — an edge that drifts out of candidate range stops being
   processed (and implicitly has no further events), and one that drifts
   into range starts fresh in the UP state. This is a deliberate, documented
   approximation: edge candidacy is held fixed *within* a bracket (matching
   the 1 Hz resolution at which positions are actually known), not
   continuously re-evaluated at every thinning proposal.
4. **Marks are derived purely from the analytic channel model** (a function
   of instantaneous distance only) at the accepted event's time, regardless
   of which cause triggered a failure. This is an explicit, stated
   simplification: Phase 3 does not yet implement cause-specific mark
   degradation (e.g., a wave-triggered failure and a range-triggered
   failure at the same distance currently produce the same marks). Refining
   marks to depend on the triggering cause is a natural extension for a
   later phase, not attempted here.
5. All randomness is drawn from a single named ``RandomManager`` stream per
   mission (``"event_thinning"``), keeping event generation reproducible
   from ``(master_seed, mission_id)`` alone, consistent with every other
   Phase 1-3 module.

Data structures
----------------
``EventGenerator`` — the sole public class; bundles the physics primitives
(via ``HazardEvaluator``), config, and random manager, exposing
``generate(...)`` as the single entry point.

Storage implications
---------------------
Produces a ``list[CommunicationEvent]`` — the exact structure
``TrajectoryWriter.write_event_log`` already accepts unchanged from Phase 1,
so no storage-layer changes were needed to support this phase.
"""

from __future__ import annotations

import math

import numpy as np

from .event_types import CommunicationEvent, CommunicationMarks, make_failure_event, make_recovery_event
from .graph_builder import CandidateEdge, build_candidate_edges
from .mission_config import MissionConfig
from .mission_state import NodeState, NodeType
from .physics import AcousticChannel, CurrentField, WaveField
from .physics_intensity import HazardEvaluator
from .random_manager import RandomManager
from .unmodeled_state import (
    CAUSE_OPERATIONAL_SENSITIVITY,
    operational_stress_components_batch,
    truth_hazard_multiplier,
)

_UP = "up"
_DOWN = "down"

_X_COL, _Y_COL, _Z_COL, _VX_COL, _VY_COL = 0, 1, 2, 3, 4


def _marks_at(channel: AcousticChannel, distance_m: float) -> CommunicationMarks:
    """Closed-form mark generator (PDR, throughput, delay) as a function of
    instantaneous distance only — see module docstring, point 4.
    """
    margin = float(channel.snr_margin_db(distance_m))
    pdr = float(np.clip(0.5 + 0.5 * math.tanh(margin / 5.0), 0.0, 1.0))
    throughput = float(channel.throughput_bps(distance_m)) * pdr
    speed_of_sound_mps = 1500.0
    propagation_delay_s = distance_m / speed_of_sound_mps
    queuing_penalty = 1.0 + (1.0 - pdr) * 2.0  # degraded links incur extra queuing/retransmit delay
    delay_s = propagation_delay_s * queuing_penalty
    marks = CommunicationMarks(pdr=pdr, throughput_bps=max(throughput, 0.0), delay_s=max(delay_s, 0.0))
    marks.validate()
    return marks


class EventGenerator:
    """Generates a mission's ground-truth communication event log via
    adaptive Ogata thinning over the competing-risks hazard model."""

    def __init__(
        self,
        config: MissionConfig,
        random_manager: RandomManager,
        acoustic_channel: AcousticChannel,
        wave_field: WaveField,
        current_field: CurrentField,
    ) -> None:
        config.validate()
        self.config = config
        self.random_manager = random_manager
        self.acoustic_channel = acoustic_channel
        self.hazard_evaluator = HazardEvaluator(acoustic_channel, wave_field, current_field, config.hazard)

    def generate(
        self,
        mission_id: str,
        array: np.ndarray,
        node_ids: list[str],
        node_types: dict[str, NodeType],
        timestamps: np.ndarray,
        unmodeled_state_strength: float = 0.0,
        unmodeled_state_mode: str = "cause_specific",
    ) -> list[CommunicationEvent]:
        """Generate the full mission event log from a propagated trajectory.

        PERFORMANCE NOTE. This is a vectorized, all-candidate-edges-at-once
        implementation, replacing an earlier per-edge Python-loop version
        that profiled at ~85s for a full 1800s/24-node mission (see the
        Phase 3 delivery notes). The bottleneck was that every bracket
        (1 second, by default) probed every candidate edge's hazard
        individually regardless of whether an event ever fired there — an
        O(brackets x edges x probes) cost of Python-level physics calls,
        independent of the actual event rate. This version instead
        evaluates the hazard for *all* candidate edges in a bracket with a
        handful of numpy calls per "round" (a round = one thinning
        proposal step across every still-active edge), turning the
        per-bracket cost into O(rounds) vectorized calls, where rounds is
        typically small (1-3) given calibrated hazard rates. See
        ``_simulate_bracket_vectorized`` for the algorithm.

        REPRODUCIBILITY NOTE. This implementation draws random numbers in a
        different order than the original per-edge-sequential version (all
        active edges draw together, per round, rather than one edge being
        fully resolved before the next begins). This means outputs are
        **not** bit-identical to the original implementation even with the
        same ``master_seed`` — but reproducibility *within* this
        implementation (same seed -> same output, run twice) is fully
        preserved, since all randomness still flows through the single
        ``RandomManager``-derived stream for this mission.

        Parameters
        ----------
        array: ``[T, |V|, F_NODE]`` trajectory array from
            ``TrajectoryGenerator.propagate``.
        node_ids: ordered node-id list matching the array's second axis.
        node_types: mapping from node_id to ``NodeType`` (from the mission's
            initial ``MissionState`` — not re-decoded from the array).
        timestamps: ``[T]`` array of sample times in seconds.
        """
        if unmodeled_state_strength < 0.0:
            raise ValueError("unmodeled_state_strength must be >= 0")
        if unmodeled_state_mode not in {"cause_specific", "total"}:
            raise ValueError("unmodeled_state_mode must be 'cause_specific' or 'total'")
        # Keep the nominal thinning stream completely isolated from the optional
        # latent-state innovation stream.  This is essential for a fair
        # robustness benchmark: changing the mismatch strength must not change
        # the event-thinning random numbers merely because latent q was sampled.
        rng = self.random_manager.get_numpy_rng(mission_id, "event_thinning")
        latent_rng = (
            self.random_manager.get_numpy_rng(mission_id, "operational_state")
            if unmodeled_state_strength > 0.0 else None
        )
        v_index = {node_id: i for i, node_id in enumerate(node_ids)}

        edge_states: dict[tuple[str, str], str] = {}
        # Persistent truth-only operational state.  It is intentionally not
        # stored in MissionArtifacts, so it is unavailable to physics-only
        # inference.  The neural model can only infer it indirectly from the
        # observable history (battery/FIM/motion proxies).
        latent_operational_state: dict[tuple[str, str], float] = {}
        all_events: list[CommunicationEvent] = []

        for t_idx in range(len(timestamps) - 1):
            t0, t1 = float(timestamps[t_idx]), float(timestamps[t_idx + 1])

            bracket_nodes = [
                NodeState(
                    node_id=node_id,
                    node_type=node_types[node_id],
                    timestamp_s=t0,
                    x_m=float(array[t_idx, v_index[node_id], _X_COL]),
                    y_m=float(array[t_idx, v_index[node_id], _Y_COL]),
                    z_m=float(array[t_idx, v_index[node_id], _Z_COL]),
                )
                for node_id in node_ids
            ]
            candidate_edges = build_candidate_edges(
                bracket_nodes, self.acoustic_channel, self.config.graph.candidate_range_factor
            )
            if not candidate_edges:
                continue

            bracket_events, bracket_final_states = self._simulate_bracket_vectorized(
                candidate_edges,
                edge_states,
                array,
                v_index,
                t_idx,
                t0,
                t1,
                rng,
                unmodeled_state_strength,
                latent_operational_state,
                unmodeled_state_mode,
                latent_rng,
            )
            all_events.extend(bracket_events)
            edge_states.update(bracket_final_states)

        all_events.sort(key=lambda e: e.timestamp_s)
        return all_events

    # ------------------------------------------------------------------ #
    # Vectorized, round-based thinning across all candidate edges in one
    # bracket simultaneously
    # ------------------------------------------------------------------ #
    def _simulate_bracket_vectorized(
        self,
        candidate_edges: list[CandidateEdge],
        edge_states: dict[tuple[str, str], str],
        array: np.ndarray,
        v_index: dict[str, int],
        t_idx: int,
        t0: float,
        t1: float,
        rng: np.random.Generator,
        unmodeled_state_strength: float = 0.0,
        latent_operational_state: dict[tuple[str, str], float] | None = None,
        unmodeled_state_mode: str = "cause_specific",
        latent_rng: np.random.Generator | None = None,
    ) -> tuple[list[CommunicationEvent], dict[tuple[str, str], str]]:
        n_edges = len(candidate_edges)
        n_probes = self.config.graph.n_thinning_probes_per_bracket
        safety = self.config.graph.dominating_rate_safety_factor

        i_idx = np.array([v_index[e.node_i] for e in candidate_edges])
        j_idx = np.array([v_index[e.node_j] for e in candidate_edges])

        # base position/velocity for both endpoints, at the bracket's start sample
        xi0, yi0, zi0 = array[t_idx, i_idx, _X_COL], array[t_idx, i_idx, _Y_COL], array[t_idx, i_idx, _Z_COL]
        vxi0, vyi0 = array[t_idx, i_idx, _VX_COL], array[t_idx, i_idx, _VY_COL]
        xj0, yj0, zj0 = array[t_idx, j_idx, _X_COL], array[t_idx, j_idx, _Y_COL], array[t_idx, j_idx, _Z_COL]
        vxj0, vyj0 = array[t_idx, j_idx, _VX_COL], array[t_idx, j_idx, _VY_COL]
        # Existing simulator states that are deliberately outside the nominal
        # four-cause physics evaluator.  They are observable to neural models
        # through node history, but never passed into HazardEvaluator.
        battery_i0 = array[t_idx, i_idx, 6]
        battery_j0 = array[t_idx, j_idx, 6]
        fim_i0 = array[t_idx, i_idx, 7]
        fim_j0 = array[t_idx, j_idx, 7]

        # state: 0 = UP, 1 = DOWN
        state = np.array(
            [0 if edge_states.get(e.edge_key, _UP) == _UP else 1 for e in candidate_edges], dtype=np.int8
        )
        t_current = np.full(n_edges, t0, dtype=np.float64)
        active = np.ones(n_edges, dtype=bool)
        events: list[CommunicationEvent] = []

        # Persistent AR(1)-like operational quality state.  The state is
        # truth-only: it is driven by observable operational stress plus a
        # small innovation, but the realized latent value itself is never
        # appended to the trajectory or passed to HazardEvaluator.
        if latent_operational_state is None:
            latent_operational_state = {}
        if unmodeled_state_mode not in {"cause_specific", "total"}:
            raise ValueError("unmodeled_state_mode must be 'cause_specific' or 'total'")
        if unmodeled_state_strength > 0.0:
            dx0 = xi0 - xj0
            dy0 = yi0 - yj0
            initial_components = operational_stress_components_batch(
                battery_i0, battery_j0, fim_i0, fim_j0,
                vxi0, vyi0, vxj0, vyj0, dx0, dy0,
            )
            initial_stress = (
                0.50 * initial_components[:, 0] +
                0.30 * initial_components[:, 1] +
                0.20 * initial_components[:, 2]
            )
            rho = 0.90
            innovation_std = 0.06
            q0 = np.empty(n_edges, dtype=np.float64)
            for e_idx, edge in enumerate(candidate_edges):
                key = edge.edge_key
                prev = latent_operational_state.get(key)
                if prev is None:
                    q = float(initial_stress[e_idx])
                else:
                    q = rho * float(prev) + (1.0 - rho) * float(initial_stress[e_idx])
                    if latent_rng is None:
                        raise RuntimeError("latent_rng is required when unmodeled_state_strength > 0")
                    q += innovation_std * float(latent_rng.normal())
                    q = float(np.clip(q, 0.0, 1.0))
                latent_operational_state[key] = q
                q0[e_idx] = q
        else:
            q0 = np.zeros(n_edges, dtype=np.float64)

        def positions_at(t_query: np.ndarray, idx: np.ndarray) -> tuple:
            """Positions/distance for the edges selected by ``idx`` at the
            per-edge query times ``t_query`` (``len(t_query) == len(idx)``,
            entries aligned pairwise — this alignment is the invariant every
            caller of ``positions_at``/``hazard_at`` must preserve)."""
            dt = t_query - t0
            xi = xi0[idx] + vxi0[idx] * dt
            yi = yi0[idx] + vyi0[idx] * dt
            xj = xj0[idx] + vxj0[idx] * dt
            yj = yj0[idx] + vyj0[idx] * dt
            distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi0[idx] - zj0[idx]) ** 2)
            return xi, yi, xj, yj, distance

        def hazard_at(t_query: np.ndarray, state_subset: np.ndarray, idx: np.ndarray) -> np.ndarray:
            """Total hazard (failure-total for UP edges, recovery for DOWN
            edges) at the given per-edge query times, for the edges
            selected by ``idx``. ``t_query``, ``state_subset``, and ``idx``
            must all have the same length, pairwise aligned.
            """
            xi, yi, xj, yj, distance = positions_at(t_query, idx)
            out = np.zeros(len(idx), dtype=np.float64)
            up_mask = state_subset == 0
            down_mask = ~up_mask
            if up_mask.any():
                sub = idx[up_mask]
                geo, wave, doppler, current = self.hazard_evaluator.cause_hazards_batch(
                    distance[up_mask], xi[up_mask], yi[up_mask], zi0[sub],
                    xj[up_mask], yj[up_mask], zj0[sub],
                    vxi0[sub], vyi0[sub], vxj0[sub], vyj0[sub],
                    t_query[up_mask],
                )
                base_total = geo + wave + doppler + current
                if unmodeled_state_strength > 0.0:
                    # q0 is the persistent hidden operational state for each
                    # candidate edge during this 1 Hz bracket.  The legacy
                    # total mode multiplies all causes equally.  The new
                    # cause-specific mode deliberately changes relative cause
                    # risks, which creates a ranking challenge for the model.
                    if unmodeled_state_mode == "total":
                        base_total = base_total * truth_hazard_multiplier(
                            q0[idx[up_mask]], unmodeled_state_strength
                        )
                    else:
                        # Reconstruct the observable component stress at the
                        # bracket start and use the same persistent q as the
                        # scalar latent state, with fixed cause sensitivities.
                        # The realized q remains hidden from the model.
                        q = q0[idx[up_mask]]
                        components = operational_stress_components_batch(
                            battery_i0[sub], battery_j0[sub], fim_i0[sub], fim_j0[sub],
                            vxi0[sub], vyi0[sub], vxj0[sub], vyj0[sub],
                            xi[up_mask] - xj[up_mask], yi[up_mask] - yj[up_mask],
                        )
                        # Preserve the persistent latent amplitude while using
                        # fixed cause-specific sensitivities.
                        base_causes = np.stack([geo, wave, doppler, current], axis=-1)
                        sensitivity_score = components @ CAUSE_OPERATIONAL_SENSITIVITY.T
                        # The latent state controls perturbation amplitude while
                        # the fixed sensitivity matrix changes the relative
                        # cause risks.  This is algebraically simple and avoids
                        # the previous q/scalar ratio construction.
                        log_mult = unmodeled_state_strength * q[:, None] * sensitivity_score
                        base_total = np.sum(base_causes * np.exp(log_mult), axis=-1)
                # IMPORTANT: nominal physics must always be assigned, including
                # the exact strength=0 case.  Previously this assignment was
                # accidentally nested inside `if unmodeled_state_strength > 0`,
                # making every nominal failure hazard identically zero.
                out[up_mask] = base_total
            if down_mask.any():
                out[down_mask] = self.hazard_evaluator.recovery_hazard_batch(distance[down_mask])
            return out

        while active.any():
            active_idx = np.nonzero(active)[0]
            n_active = len(active_idx)

            # 1) estimate a dominating rate per active edge by probing its
            #    remaining sub-interval [t_current, t1] — flattened into a
            #    single vectorized call across all (edge, probe) pairs at
            #    once, rather than one call per probe column. This is the
            #    fix for the residual bottleneck a profiling pass revealed:
            #    looping over probe columns in Python called the physics
            #    primitives (wave_field.height_rate, current_field's layer
            #    interpolation) ``n_probes`` times per round with small
            #    arrays, paying per-call Python/numpy overhead ``n_probes``
            #    times over. Flattening pays that overhead once.
            probe_fracs = np.linspace(0.0, 1.0, n_probes)  # [n_probes]
            span = t1 - t_current[active_idx]  # [n_active]
            flat_idx = np.repeat(active_idx, n_probes)  # [n_active * n_probes]
            flat_state = np.repeat(state[active_idx], n_probes)
            flat_t = np.repeat(t_current[active_idx], n_probes) + np.tile(probe_fracs, n_active) * np.repeat(
                span, n_probes
            )
            flat_hazards = hazard_at(flat_t, flat_state, flat_idx)
            probe_hazards = flat_hazards.reshape(n_active, n_probes)

            dominating_rate = probe_hazards.max(axis=1) * safety  # [n_active]

            no_risk = dominating_rate <= 0.0
            if no_risk.any():
                active[active_idx[no_risk]] = False
            still_active_local = ~no_risk
            if not still_active_local.any():
                continue
            active_idx = active_idx[still_active_local]
            dominating_rate = dominating_rate[still_active_local]
            n_active = len(active_idx)

            # 2) propose candidate event times
            u = rng.random(n_active)
            delta = -np.log(np.maximum(1.0 - u, 1e-15)) / dominating_rate
            t_proposed = t_current[active_idx] + delta

            beyond_bracket = t_proposed >= t1
            if beyond_bracket.any():
                active[active_idx[beyond_bracket]] = False
            within_bracket = ~beyond_bracket
            if not within_bracket.any():
                continue
            eval_idx = active_idx[within_bracket]
            eval_t = t_proposed[within_bracket]
            eval_rate = dominating_rate[within_bracket]

            # 3) accept/reject at the actual instantaneous hazard
            actual_hazard = hazard_at(eval_t, state[eval_idx], eval_idx)
            accept_prob = np.minimum(actual_hazard / eval_rate, 1.0)
            u2 = rng.random(len(eval_idx))
            accepted_mask = u2 < accept_prob

            # advance t_current for every edge that proposed in this round
            t_current[eval_idx] = eval_t

            if accepted_mask.any():
                accepted_idx = eval_idx[accepted_mask]
                accepted_t = eval_t[accepted_mask]
                events.extend(
                    self._emit_events(candidate_edges, accepted_idx, accepted_t, state, array, v_index, t_idx, t0)
                )
                state[accepted_idx] = 1 - state[accepted_idx]  # toggle UP<->DOWN

        final_states = {
            candidate_edges[e].edge_key: (_UP if state[e] == 0 else _DOWN) for e in range(n_edges)
        }
        return events, final_states

    def _emit_events(
        self,
        candidate_edges: list[CandidateEdge],
        accepted_idx: np.ndarray,
        accepted_t: np.ndarray,
        state_before_toggle: np.ndarray,
        array: np.ndarray,
        v_index: dict[str, int],
        t_idx: int,
        t0: float,
    ) -> list[CommunicationEvent]:
        """Construct real ``CommunicationEvent`` objects for the (typically
        small) set of accepted events in one thinning round. This step is
        deliberately left as a per-event Python loop — its cost is
        proportional to the actual event count, not to the flat
        brackets x edges bottleneck this rewrite targets, so vectorizing it
        further would add complexity without a meaningful speed benefit.
        """
        events: list[CommunicationEvent] = []
        for local_i, edge_idx in enumerate(accepted_idx):
            edge = candidate_edges[edge_idx]
            t_event = float(accepted_t[local_i])
            was_up = state_before_toggle[edge_idx] == 0

            xi, yi, zi, vxi, vyi = self._interpolate_at(array, v_index[edge.node_i], t_idx, t0, t_event)
            xj, yj, zj, vxj, vyj = self._interpolate_at(array, v_index[edge.node_j], t_idx, t0, t_event)
            distance = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            marks = _marks_at(self.acoustic_channel, distance)

            if was_up:
                cause_hazards = self.hazard_evaluator.cause_hazards(
                    distance, xi, yi, zi, xj, yj, zj, vxi, vyi, vxj, vyj, t_event
                )
                events.append(
                    make_failure_event(edge.node_i, edge.node_j, t_event, cause_hazards.dominant_cause(), marks)
                )
            else:
                events.append(make_recovery_event(edge.node_i, edge.node_j, t_event, marks))
        return events

    @staticmethod
    def _interpolate_at(
        array: np.ndarray, v_idx: int, t_idx: int, t0: float, query_t: float
    ) -> tuple[float, float, float, float, float]:
        """Scalar position/velocity reconstruction at an arbitrary time
        within the bracket ``[t0, t0+dt)`` starting at sample ``t_idx`` —
        exact linear extrapolation from the bracket-start sample using its
        stored velocity, used only in the (small, per-accepted-event)
        ``_emit_events`` path."""
        x0 = float(array[t_idx, v_idx, _X_COL])
        y0 = float(array[t_idx, v_idx, _Y_COL])
        z0 = float(array[t_idx, v_idx, _Z_COL])
        vx0 = float(array[t_idx, v_idx, _VX_COL])
        vy0 = float(array[t_idx, v_idx, _VY_COL])
        dt = query_t - t0
        return x0 + vx0 * dt, y0 + vy0 * dt, z0, vx0, vy0

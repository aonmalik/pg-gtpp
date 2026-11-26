"""
trajectory_generator.py

Propagates node kinematics forward across the full mission duration, at the
storage sample rate, producing the ``[T, |V|, F_NODE]`` array
``trajectory_storage.py`` persists. Phase 1's ``MissionGenerator`` only
produces the t=0 initial condition; this module is what makes that
trajectory more than one sample long.

Design decisions
-----------------
1. Motion is generated **directly at the storage sample rate**
   (``config.envelope.node_sample_rate_hz``, default 1 Hz), not at a finer
   internal resolution. Node kinematics here (patrol circles, waypoint
   following) are smooth, low-frequency motions relative to 1 Hz — unlike
   the wave field, which genuinely needs sub-second internal resolution
   inside the event generator's thinning algorithm (a later concern, not
   this module's). Introducing a separate internal simulation rate here
   would be unnecessary complexity with no accuracy benefit, so it is
   deliberately not done, per the "avoid unnecessary complexity" standard.
2. **USV**: a slow circular patrol around the mission area's centroid, at a
   fixed radius (a fraction of the area's smaller dimension) and constant
   angular speed, with a random initial phase drawn from a seeded stream.
   This is a simple, physically reasonable stand-in for the baseline's
   FIM-optimizing TD3 controller (not available in this session to port
   directly) — it produces smooth, non-degenerate USV motion, which is what
   this phase actually needs (the graph/hazard/event machinery it feeds
   does not depend on the USV policy being FIM-optimal). This substitution
   is stated explicitly here, not silently assumed.
3. **AUVs**: each AUV tours a reproducibly shuffled subset of the mission's
   IoUT node positions as waypoints, moving at ``config.fleet.max_auv_speed_mps``
   in a straight line to the current waypoint, advancing to the next
   waypoint (cyclically) on arrival. This gives the path diversity the QA
   framework's "unrealistic mobility patterns" check (Phase 1's risk table)
   flags as a failure mode to avoid, using only already-configured
   quantities (no new config fields needed).
4. **IoUT**: static, matching the physical scenario (seabed sensor nodes).
5. **Battery**: linear drain from 1.0 to a configured floor over the mission
   duration (a simple, explicit, documented model — not a placeholder).
6. **FIM uncertainty**: modeled as an exponential decay from each AUV's
   initial (Phase 1-sampled) value toward a small floor, with a time
   constant proportional to mission duration. This is an explicit,
   documented **proxy** for localization convergence, not a port of the
   baseline's actual Fisher Information Matrix computation (which requires
   the USBL ranging/geometry machinery not available in this session). Any
   downstream use of this field for anything beyond a rough relative-
   priority signal (e.g. Paper 2's FIM-weighted repositioning heuristic)
   should replace this proxy with the real FIM computation first.
7. All randomness (USV patrol phase, per-AUV waypoint order) is drawn from
   named ``RandomManager`` streams distinct from Phase 1's placement
   streams, so trajectory generation is reproducible independently of
   (and without perturbing) initial placement.

Data structures
----------------
Consumes a Phase 1 ``MissionState`` (initial condition) plus ``MissionConfig``
and ``RandomManager``. Produces a ``[T, |V|, F_NODE]`` ``numpy`` array in the
exact layout ``trajectory_storage.py`` expects (see
``trajectory_storage._NODE_FEATURE_ORDER`` / ``_NODE_TYPE_ORDER``), plus the
ordered ``node_ids`` list.

Storage implications
---------------------
None directly — produces an in-memory array. ``trajectory_storage.py``
persists it via the existing ``write_node_trajectory`` method (already
built to accept an arbitrary-``T`` array).
"""

from __future__ import annotations

import math

import numpy as np

from .mission_config import MissionConfig
from .mission_state import MissionState, NodeState, NodeType
from .random_manager import RandomManager
from .trajectory_storage import _NODE_FEATURE_ORDER, _NODE_TYPE_ORDER, F_NODE

BATTERY_FLOOR = 0.05
FIM_UNCERTAINTY_FLOOR = 0.02
USV_PATROL_RADIUS_FRACTION = 0.3  # fraction of min(area_width, area_height)
USV_PATROL_PERIOD_FRACTION = 0.5  # fraction of mission duration for one full circle


def _node_state_to_feature_row(node: NodeState) -> np.ndarray:
    continuous = np.array([getattr(node, f) for f in _NODE_FEATURE_ORDER], dtype=np.float32)
    one_hot = np.array(
        [1.0 if node.node_type == t else 0.0 for t in _NODE_TYPE_ORDER], dtype=np.float32
    )
    return np.concatenate([continuous, one_hot])


class TrajectoryGenerator:
    """Propagates a mission's node kinematics across the full mission
    duration at the storage sample rate."""

    def __init__(self, config: MissionConfig, random_manager: RandomManager) -> None:
        config.validate()
        self.config = config
        self.random_manager = random_manager

    def propagate(self, initial_state: MissionState) -> tuple[np.ndarray, list[str]]:
        """Propagate ``initial_state`` (a Phase 1 t=0 ``MissionState``)
        forward across the mission duration.

        Returns
        -------
        array: ``[T, |V|, F_NODE]`` float32 array, node order matching
            ``node_ids``.
        node_ids: the ordered node-id list corresponding to the array's
            second axis.
        """
        mission_id = initial_state.metadata.mission_id
        duration_s = self.config.envelope.duration_s
        sample_rate_hz = self.config.envelope.node_sample_rate_hz
        n_steps = int(round(duration_s * sample_rate_hz)) + 1  # inclusive of t=0 and t=duration
        timestamps = np.arange(n_steps, dtype=np.float64) / sample_rate_hz

        node_ids = initial_state.node_ids
        usv_state = initial_state.get_node(
            next(n.node_id for n in initial_state.nodes if n.node_type == NodeType.USV)
        )
        auv_states = [n for n in initial_state.nodes if n.node_type == NodeType.AUV]
        iout_states = [n for n in initial_state.nodes if n.node_type == NodeType.IOUT]

        usv_traj = self._propagate_usv(mission_id, usv_state, timestamps)
        auv_trajs = {
            auv.node_id: self._propagate_auv(mission_id, auv, iout_states, timestamps)
            for auv in auv_states
        }
        iout_trajs = {iout.node_id: self._propagate_static(iout, timestamps) for iout in iout_states}

        per_node_trajs: dict[str, list[NodeState]] = {
            usv_state.node_id: usv_traj,
            **auv_trajs,
            **iout_trajs,
        }

        array = np.zeros((n_steps, len(node_ids), F_NODE), dtype=np.float32)
        for v_idx, node_id in enumerate(node_ids):
            node_traj = per_node_trajs[node_id]
            for t_idx, node in enumerate(node_traj):
                array[t_idx, v_idx, :] = _node_state_to_feature_row(node)

        return array, node_ids

    # ------------------------------------------------------------------ #
    # USV: circular patrol
    # ------------------------------------------------------------------ #
    def _propagate_usv(
        self, mission_id: str, initial: NodeState, timestamps: np.ndarray
    ) -> list[NodeState]:
        rng = self.random_manager.get_numpy_rng(mission_id, "trajectory_usv")
        width, height = self.config.envelope.area_size_m
        cx, cy = width / 2.0, height / 2.0
        radius = USV_PATROL_RADIUS_FRACTION * min(width, height)
        period_s = max(USV_PATROL_PERIOD_FRACTION * self.config.envelope.duration_s, 1.0)
        angular_speed = 2.0 * math.pi / period_s
        phase0 = float(rng.uniform(0.0, 2.0 * math.pi))

        trajectory: list[NodeState] = []
        for t in timestamps:
            angle = phase0 + angular_speed * t
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            vx = -radius * angular_speed * math.sin(angle)
            vy = radius * angular_speed * math.cos(angle)
            heading = math.atan2(vy, vx)
            node = NodeState(
                node_id=initial.node_id,
                node_type=NodeType.USV,
                timestamp_s=float(t),
                x_m=x,
                y_m=y,
                z_m=0.0,
                vx_mps=vx,
                vy_mps=vy,
                heading_rad=heading,
                battery_frac=self._battery_at(1.0, float(t)),
                fim_uncertainty=0.0,  # USV is not a localization target
            )
            node.validate()
            trajectory.append(node)
        return trajectory

    # ------------------------------------------------------------------ #
    # AUV: waypoint tour over IoUT positions
    # ------------------------------------------------------------------ #
    def _propagate_auv(
        self,
        mission_id: str,
        initial: NodeState,
        iout_states: list[NodeState],
        timestamps: np.ndarray,
    ) -> list[NodeState]:
        rng = self.random_manager.get_numpy_rng(mission_id, f"trajectory_auv_{initial.node_id}")
        max_speed = self.config.fleet.max_auv_speed_mps

        if iout_states:
            order = list(range(len(iout_states)))
            rng.shuffle(order)
            waypoints = [(iout_states[i].x_m, iout_states[i].y_m) for i in order]
        else:
            # no IoUT nodes to visit (degenerate config): hold position
            waypoints = [(initial.x_m, initial.y_m)]

        x, y = initial.x_m, initial.y_m
        z = initial.z_m  # AUV holds its sampled depth for the whole mission (no vertical maneuvering in Phase 3)
        waypoint_idx = 0
        dt = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 1.0

        trajectory: list[NodeState] = []
        for t_idx, t in enumerate(timestamps):
            target_x, target_y = waypoints[waypoint_idx]
            dx, dy = target_x - x, target_y - y
            distance_to_target = math.hypot(dx, dy)

            if distance_to_target < 1e-6:
                vx, vy = 0.0, 0.0
            else:
                step = min(max_speed * dt, distance_to_target)
                direction_x, direction_y = dx / distance_to_target, dy / distance_to_target
                vx, vy = direction_x * max_speed, direction_y * max_speed
                if t_idx > 0:  # do not move at t=0; record only, matching Phase 1's initial condition
                    x += direction_x * step
                    y += direction_y * step
                if distance_to_target <= max_speed * dt:
                    waypoint_idx = (waypoint_idx + 1) % len(waypoints)

            heading = math.atan2(vy, vx) if (vx != 0.0 or vy != 0.0) else initial.heading_rad
            node = NodeState(
                node_id=initial.node_id,
                node_type=NodeType.AUV,
                timestamp_s=float(t),
                x_m=x,
                y_m=y,
                z_m=z,
                vx_mps=vx,
                vy_mps=vy,
                heading_rad=heading,
                battery_frac=self._battery_at(initial.battery_frac, float(t)),
                fim_uncertainty=self._fim_uncertainty_at(initial.fim_uncertainty, float(t)),
            )
            node.validate()
            trajectory.append(node)
        return trajectory

    # ------------------------------------------------------------------ #
    # IoUT: static
    # ------------------------------------------------------------------ #
    def _propagate_static(self, initial: NodeState, timestamps: np.ndarray) -> list[NodeState]:
        trajectory: list[NodeState] = []
        for t in timestamps:
            node = NodeState(
                node_id=initial.node_id,
                node_type=NodeType.IOUT,
                timestamp_s=float(t),
                x_m=initial.x_m,
                y_m=initial.y_m,
                z_m=initial.z_m,
                vx_mps=0.0,
                vy_mps=0.0,
                heading_rad=0.0,
                battery_frac=1.0,
                fim_uncertainty=0.0,
            )
            node.validate()
            trajectory.append(node)
        return trajectory

    # ------------------------------------------------------------------ #
    # Simple, explicit proxy models (see module docstring, points 5-6)
    # ------------------------------------------------------------------ #
    def _battery_at(self, initial_frac: float, t_s: float) -> float:
        duration = self.config.envelope.duration_s
        drained_frac = (initial_frac - BATTERY_FLOOR) * (t_s / duration) if duration > 0 else 0.0
        return float(max(BATTERY_FLOOR, initial_frac - drained_frac))

    def _fim_uncertainty_at(self, initial_value: float, t_s: float) -> float:
        tau = max(self.config.envelope.duration_s / 3.0, 1.0)
        decayed = FIM_UNCERTAINTY_FLOOR + (initial_value - FIM_UNCERTAINTY_FLOOR) * math.exp(-t_s / tau)
        return float(max(FIM_UNCERTAINTY_FLOOR, decayed))

"""
mission_generator.py

Mission creation and initialization: fleet-size sampling, node placement, and
metadata construction. Explicitly does NOT propagate trajectories forward in
time — that is a later phase's responsibility (a trajectory generator that
consumes the ``MissionState`` this module produces as its t=0 initial
condition).

Design decisions
-----------------
1. All randomness is drawn through named ``RandomManager`` streams
   (``"fleet_size"``, ``"placement_usv"``, ``"placement_auv"``,
   ``"placement_iout"``), never through global numpy state. This means two
   ``MissionGenerator`` instances constructed with the same config and the
   same ``RandomManager`` master seed produce byte-identical
   ``MissionState`` objects, regardless of call order or process — a direct
   requirement of the reproducibility framework.
2. Placement logic is intentionally simple and physically motivated rather
   than elaborate: the USV starts near the mission area's centroid (its
   baseline TD3 controller, from later phases, repositions it from there);
   AUVs are placed uniformly at random within the area at a randomized depth
   within the envelope's depth range; IoUT nodes are placed uniformly at
   random on the seabed (at the maximum configured depth). This matches the
   mission topology described in the simulator design document and gives
   the diversity later QA checks (path-diversity, mobility realism) expect,
   without introducing unnecessary generative complexity into Phase 1.
3. ``generate()`` re-validates the config defensively (via
   ``validation_utils.validate_mission_config``) even though the config was
   presumably already validated at load time — generation is exactly the
   kind of operation the QA framework insists must fail loudly and early
   rather than silently produce a malformed mission from a config that
   drifted after construction (configs are frozen, but a caller could still
   pass an unvalidated hand-built one).
4. The returned ``MissionState`` is itself validated (structural + physical
   sanity, via ``MissionState.validate()``) before being handed back to the
   caller, so a generation bug (e.g., a placement routine that produces a
   duplicate node_id) is caught at the source rather than downstream in
   storage or training code.

Data structures
----------------
Consumes ``MissionConfig`` (mission_config.py) and ``RandomManager``
(random_manager.py). Produces ``MissionState`` (mission_state.py).

Storage implications
---------------------
None directly — this module produces an in-memory object.
``trajectory_storage.py`` is responsible for persisting it (and, in later
phases, the trajectory that follows from it) in the storage-efficient format.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from .mission_config import MissionConfig
from .mission_state import MissionMetadata, MissionState, NodeState, NodeType
from .random_manager import RandomManager
from .validation_utils import validate_mission_config

logger = logging.getLogger(__name__)

from . import SIMULATOR_VERSION  # noqa: E402  (avoids circular-import ordering issues)


class MissionGenerator:
    """Creates and initializes missions from a config + random manager.

    Phase 1 scope: node placement and metadata only. No physics propagation,
    no hazard generation, no event generation, no graph construction.
    """

    def __init__(self, config: MissionConfig, random_manager: RandomManager) -> None:
        validate_mission_config(config)
        self.config = config
        self.random_manager = random_manager

    def generate(self, mission_id: str, split: str | None = None) -> MissionState:
        """Generate the initial (t=0) state for a single mission.

        Parameters
        ----------
        mission_id:
            Unique, human-readable mission identifier (e.g. ``"mission_00042"``).
            This value, combined with the random manager's master seed, is the
            sole source of reproducibility for this mission.
        split:
            Dataset split label. If ``None``, falls back to ``self.config.split``.
        """
        validate_mission_config(self.config)  # defensive re-check; see module docstring
        resolved_split = split if split is not None else self.config.split
        logger.info("Generating mission %s (split=%s)", mission_id, resolved_split)

        rng_fleet = self.random_manager.get_numpy_rng(mission_id, "fleet_size")
        n_auv = int(
            rng_fleet.integers(self.config.fleet.n_auv_min, self.config.fleet.n_auv_max + 1)
        )
        m_iout = int(
            rng_fleet.integers(self.config.fleet.m_iout_min, self.config.fleet.m_iout_max + 1)
        )

        nodes: list[NodeState] = []
        nodes.append(self._place_usv_for_mission(mission_id))
        nodes.extend(self._place_auvs(mission_id, n_auv))
        nodes.extend(self._place_iout(mission_id, m_iout))

        metadata = MissionMetadata(
            mission_id=mission_id,
            master_seed=self.random_manager.master_seed,
            mission_seed=self.random_manager.derive_mission_seed(mission_id),
            config_hash=self.config.config_hash,
            simulator_version=SIMULATOR_VERSION,
            n_auv=n_auv,
            m_iout=m_iout,
            sea_state=self.config.envelope.sea_state,
            area_width_m=self.config.envelope.area_size_m[0],
            area_height_m=self.config.envelope.area_size_m[1],
            duration_s=self.config.envelope.duration_s,
            node_sample_rate_hz=self.config.envelope.node_sample_rate_hz,
            split=resolved_split,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        state = MissionState(metadata=metadata, nodes=nodes)
        state.validate()
        logger.info("Generated %s", state.summary())
        return state

    # ------------------------------------------------------------------ #
    # Placement routines
    #
    # All three routines draw from a mission-specific, named RNG stream
    # (see class docstring, point 1), so placement is fully reproducible
    # from (master_seed, mission_id) alone.
    # ------------------------------------------------------------------ #
    def _place_usv_for_mission(self, mission_id: str) -> NodeState:
        """USV starts near the area centroid with small random jitter."""
        rng = self.random_manager.get_numpy_rng(mission_id, "placement_usv")
        width, height = self.config.envelope.area_size_m
        cx, cy = width / 2.0, height / 2.0
        jitter_frac = 0.05  # USV starts within 5% of area size from centroid
        x = cx + rng.uniform(-jitter_frac, jitter_frac) * width
        y = cy + rng.uniform(-jitter_frac, jitter_frac) * height
        heading = float(rng.uniform(0.0, 2.0 * np.pi))
        node = NodeState(
            node_id="USV_0",
            node_type=NodeType.USV,
            timestamp_s=0.0,
            x_m=float(x),
            y_m=float(y),
            z_m=0.0,  # surface vehicle
            vx_mps=0.0,
            vy_mps=0.0,
            heading_rad=heading,
            battery_frac=1.0,
            fim_uncertainty=0.0,
        )
        node.validate()
        return node

    def _place_auvs(self, mission_id: str, n_auv: int) -> list[NodeState]:
        rng = self.random_manager.get_numpy_rng(mission_id, "placement_auv")
        width, height = self.config.envelope.area_size_m
        depth_lo, depth_hi = self.config.envelope.depth_range_m
        nodes: list[NodeState] = []
        for i in range(n_auv):
            x = float(rng.uniform(0.0, width))
            y = float(rng.uniform(0.0, height))
            z = float(rng.uniform(depth_lo, depth_hi))
            heading = float(rng.uniform(0.0, 2.0 * np.pi))
            # small random initial FIM uncertainty prior, reflecting that
            # localization has not yet converged at mission start
            fim_uncertainty = float(rng.uniform(0.1, 1.0))
            node = NodeState(
                node_id=f"AUV_{i}",
                node_type=NodeType.AUV,
                timestamp_s=0.0,
                x_m=x,
                y_m=y,
                z_m=z,
                vx_mps=0.0,
                vy_mps=0.0,
                heading_rad=heading,
                battery_frac=1.0,
                fim_uncertainty=fim_uncertainty,
            )
            node.validate()
            nodes.append(node)
        return nodes

    def _place_iout(self, mission_id: str, m_iout: int) -> list[NodeState]:
        rng = self.random_manager.get_numpy_rng(mission_id, "placement_iout")
        width, height = self.config.envelope.area_size_m
        _, depth_hi = self.config.envelope.depth_range_m  # seabed = max depth
        nodes: list[NodeState] = []
        for i in range(m_iout):
            x = float(rng.uniform(0.0, width))
            y = float(rng.uniform(0.0, height))
            node = NodeState(
                node_id=f"IOUT_{i}",
                node_type=NodeType.IOUT,
                timestamp_s=0.0,
                x_m=x,
                y_m=y,
                z_m=float(depth_hi),  # fixed seabed nodes
                vx_mps=0.0,
                vy_mps=0.0,
                heading_rad=0.0,
                battery_frac=1.0,
                fim_uncertainty=0.0,  # IoUT nodes are not localization targets
            )
            node.validate()
            nodes.append(node)
        return nodes

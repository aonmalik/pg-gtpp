"""
mission_state.py

State schema for a single simulated mission: nodes, edges, environment, and
mission-level metadata.

Design decisions
-----------------
1. Unlike ``mission_config.py``'s frozen configs, state dataclasses here are
   mutable (default dataclass behavior) because state evolves over the
   mission timeline in later phases (trajectory propagation, event
   generation). Phase 1 only constructs the *initial* (t=0) state — nothing
   in this file performs time propagation.
2. Fields are flat floats/strings/enums rather than numpy arrays. This keeps
   the schema trivially serializable to a dict/parquet row (a numpy array
   field would need custom (de)serialization logic at every boundary); numpy
   arrays are introduced only inside ``trajectory_storage.py``, where bulk,
   homogeneous, time-indexed data actually benefits from them.
3. ``NodeType`` is a ``str``-valued ``Enum`` so it serializes directly to a
   plain string in JSON/YAML/parquet without a custom encoder, while still
   giving type-checked, IDE-discoverable values in code.
4. ``MissionMetadata`` intentionally duplicates a few fields that also live
   inside ``MissionConfig`` (e.g. ``sea_state``, ``area_size_m``) rather than
   embedding the config object directly. This is deliberate: metadata is what
   gets written to ``mission_index.parquet`` (Phase 1's ``trajectory_storage``
   module) as flat, queryable columns; embedding a nested config object there
   would defeat the point of a flat, scannable index.

Data structures
----------------
``NodeState`` — one physical node (USV/AUV/IoUT) at a single instant.
``EdgeState`` — one candidate communication edge's physics-derived quantities
at a single instant (Phase 1 does not populate these from real physics yet;
the fields exist so later phases have a stable target schema).
``EnvironmentState`` — the wave/current field sample relevant to a node at a
single instant.
``MissionMetadata`` — flat, index-friendly summary of a mission.
``MissionState`` — the aggregate: metadata + a list of initial ``NodeState``.

Storage implications
---------------------
None of these objects are themselves the storage-efficient on-disk format —
that is ``trajectory_storage.py``'s job. This module defines the schema that
``trajectory_storage.py`` writes and ``validation_utils.py`` checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Physical node roles in the USV-AUV-IoUT network."""

    USV = "usv"
    AUV = "auv"
    IOUT = "iout"


@dataclass
class NodeState:
    """State of a single node at a single instant.

    Feature layout matches ``F_node = 10`` from the implementation
    architecture spec: position(3), velocity(2), heading(1), battery(1),
    node-type(encoded separately, not one-hot here — one-hot expansion is a
    modeling-layer concern, not a state-schema concern), FIM-uncertainty(1).
    """

    node_id: str
    node_type: NodeType
    timestamp_s: float
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    heading_rad: float = 0.0
    battery_frac: float = 1.0
    fim_uncertainty: float = 0.0

    def validate(self) -> None:
        if not (0.0 <= self.battery_frac <= 1.0):
            raise ValueError(
                f"NodeState({self.node_id}).battery_frac must be in [0, 1], "
                f"got {self.battery_frac}"
            )
        if self.fim_uncertainty < 0.0:
            raise ValueError(
                f"NodeState({self.node_id}).fim_uncertainty must be >= 0, "
                f"got {self.fim_uncertainty}"
            )
        if self.z_m < 0.0:
            raise ValueError(
                f"NodeState({self.node_id}).z_m (depth) must be >= 0, got {self.z_m}"
            )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "timestamp_s": self.timestamp_s,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "vx_mps": self.vx_mps,
            "vy_mps": self.vy_mps,
            "heading_rad": self.heading_rad,
            "battery_frac": self.battery_frac,
            "fim_uncertainty": self.fim_uncertainty,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeState":
        data = dict(data)
        data["node_type"] = NodeType(data["node_type"])
        return cls(**data)

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x_m, self.y_m, self.z_m)

    @property
    def horizontal_speed_mps(self) -> float:
        return (self.vx_mps ** 2 + self.vy_mps ** 2) ** 0.5


@dataclass
class EdgeState:
    """Physics-derived state of a single candidate communication edge.

    Populated by later phases (acoustic_channel.py, wave_field.py,
    current_field.py); the schema is fixed here (``F_edge = 9``, matching the
    implementation architecture spec) so downstream code has a stable target.
    All fields default to 0.0 in Phase 1, since no physics propagation has
    run yet.
    """

    node_i: str
    node_j: str
    timestamp_s: float
    distance_m: float = 0.0
    snr_db: float = 0.0
    transmission_loss_db: float = 0.0
    noise_level_db: float = 0.0
    wave_term: float = 0.0
    current_term: float = 0.0
    delta_v_mps: float = 0.0
    delta_z_m: float = 0.0
    pdr_ema: float = 0.0

    def validate(self) -> None:
        if self.node_i == self.node_j:
            raise ValueError("EdgeState cannot connect a node to itself")
        if self.distance_m < 0.0:
            raise ValueError(f"EdgeState({self.node_i},{self.node_j}).distance_m must be >= 0")
        if not (0.0 <= self.pdr_ema <= 1.0):
            raise ValueError(
                f"EdgeState({self.node_i},{self.node_j}).pdr_ema must be in [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_i": self.node_i,
            "node_j": self.node_j,
            "timestamp_s": self.timestamp_s,
            "distance_m": self.distance_m,
            "snr_db": self.snr_db,
            "transmission_loss_db": self.transmission_loss_db,
            "noise_level_db": self.noise_level_db,
            "wave_term": self.wave_term,
            "current_term": self.current_term,
            "delta_v_mps": self.delta_v_mps,
            "delta_z_m": self.delta_z_m,
            "pdr_ema": self.pdr_ema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeState":
        return cls(**data)

    @property
    def edge_key(self) -> tuple[str, str]:
        """Canonical, order-independent key for this undirected edge."""
        return tuple(sorted((self.node_i, self.node_j)))  # type: ignore[return-value]


@dataclass
class EnvironmentState:
    """Wave/current field sample relevant to a specific node at an instant."""

    node_id: str
    timestamp_s: float
    wave_height_m: float = 0.0
    current_u_mps: float = 0.0  # eastward component
    current_v_mps: float = 0.0  # northward component
    sea_state: str = "moderate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "timestamp_s": self.timestamp_s,
            "wave_height_m": self.wave_height_m,
            "current_u_mps": self.current_u_mps,
            "current_v_mps": self.current_v_mps,
            "sea_state": self.sea_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentState":
        return cls(**data)


@dataclass
class MissionMetadata:
    """Flat, index-friendly summary of a single mission.

    This is exactly the row schema written to ``mission_index.parquet`` by
    ``trajectory_storage.py`` — every field here is a scalar, by design, so
    the index remains cheap to scan and filter without deserializing nested
    objects.
    """

    mission_id: str
    master_seed: int
    mission_seed: int
    config_hash: str
    simulator_version: str
    n_auv: int
    m_iout: int
    sea_state: str
    area_width_m: float
    area_height_m: float
    duration_s: float
    node_sample_rate_hz: float
    split: str
    created_at: str  # ISO-8601 UTC timestamp
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.n_auv < 1:
            raise ValueError("MissionMetadata.n_auv must be >= 1")
        if self.m_iout < 1:
            raise ValueError("MissionMetadata.m_iout must be >= 1")
        if self.duration_s <= 0.0:
            raise ValueError("MissionMetadata.duration_s must be > 0")
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"MissionMetadata.split invalid: {self.split}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "master_seed": self.master_seed,
            "mission_seed": self.mission_seed,
            "config_hash": self.config_hash,
            "simulator_version": self.simulator_version,
            "n_auv": self.n_auv,
            "m_iout": self.m_iout,
            "sea_state": self.sea_state,
            "area_width_m": self.area_width_m,
            "area_height_m": self.area_height_m,
            "duration_s": self.duration_s,
            "node_sample_rate_hz": self.node_sample_rate_hz,
            "split": self.split,
            "created_at": self.created_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionMetadata":
        return cls(**data)


@dataclass
class MissionState:
    """Aggregate mission state: metadata plus the initial node configuration.

    Phase 1 scope: ``nodes`` holds exactly one snapshot (t=0). Later phases
    that propagate trajectories operate on sequences of ``NodeState`` outside
    this class (see ``trajectory_storage.py``'s array-based format) — this
    class is the *initialization*-time representation only, matching the
    "no trajectory simulation yet, only initialization" scope of Phase 1's
    ``mission_generator.py``.
    """

    metadata: MissionMetadata
    nodes: list[NodeState] = field(default_factory=list)

    def validate(self) -> None:
        self.metadata.validate()
        node_ids = [n.node_id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("MissionState contains duplicate node_id values")
        for node in self.nodes:
            node.validate()
        usv_count = sum(1 for n in self.nodes if n.node_type == NodeType.USV)
        if usv_count != 1:
            raise ValueError(f"MissionState must contain exactly 1 USV, found {usv_count}")
        auv_count = sum(1 for n in self.nodes if n.node_type == NodeType.AUV)
        if auv_count != self.metadata.n_auv:
            raise ValueError(
                f"MissionState AUV count ({auv_count}) does not match "
                f"metadata.n_auv ({self.metadata.n_auv})"
            )
        iout_count = sum(1 for n in self.nodes if n.node_type == NodeType.IOUT)
        if iout_count != self.metadata.m_iout:
            raise ValueError(
                f"MissionState IoUT count ({iout_count}) does not match "
                f"metadata.m_iout ({self.metadata.m_iout})"
            )

    def get_node(self, node_id: str) -> NodeState:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"No node with node_id={node_id!r} in mission {self.metadata.mission_id}")

    @property
    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def summary(self) -> str:
        return (
            f"MissionState(id={self.metadata.mission_id}, "
            f"nodes={self.node_count} "
            f"[1 USV, {self.metadata.n_auv} AUV, {self.metadata.m_iout} IoUT], "
            f"sea_state={self.metadata.sea_state}, "
            f"split={self.metadata.split}, "
            f"config_hash={self.metadata.config_hash})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionState":
        metadata = MissionMetadata.from_dict(data["metadata"])
        nodes = [NodeState.from_dict(n) for n in data["nodes"]]
        return cls(metadata=metadata, nodes=nodes)

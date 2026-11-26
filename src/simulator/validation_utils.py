"""
validation_utils.py

Cross-cutting validation utilities used by every other Phase 1 module:
configuration validation, mission-state validation, on-disk schema
validation, and storage-budget validation.

Design decisions
-----------------
1. This module deliberately contains *functions*, not a class hierarchy —
   validation here is stateless and each function validates exactly one
   artifact type (config / state / trajectory array / event log schema /
   storage footprint). This mirrors how the QA framework document treats
   validation as a set of independent, individually-reportable checks, each
   of which should be individually callable from a certification-report
   generator in a later phase.
2. Validation functions *raise* on hard violations (via typed exceptions
   defined here) rather than returning booleans — callers that need a
   report of many checks (e.g., a future certification-report generator)
   are expected to catch exceptions per-check and aggregate results, not to
   have this module silently swallow errors into a boolean.
3. ``validate_storage_budget`` is the one exception to "raise on violation":
   it returns a ``StorageValidationResult`` (Pass/Warning/Fail) rather than
   raising, because storage-budget status is a policy decision for the
   caller (e.g., a dataset-build script may choose to warn-and-continue
   rather than hard-fail), matching the QA framework's explicit
   Pass/Warning/Fail decision-rule design for the dataset certification
   report.
4. Each "what to check" list here is deliberately the mechanical subset of
   the QA framework's Phase 3/Phase 4 checks that can be verified from
   Phase-1-available artifacts (config, initial state, node-trajectory array
   shape/dtype, event-log schema, on-disk footprint). Checks that require
   generated physics/hazard data (e.g., the ambiguous-event-fraction metric,
   the time-rescaling theorem test) are out of scope until the phases that
   produce that data exist, and are explicitly not stubbed out here — this
   module only validates what Phase 1 can actually produce.

Data structures
----------------
``ConfigValidationError``, ``MissionStateValidationError``,
``SchemaValidationError`` — typed exceptions, one per artifact category, so
callers can catch selectively.
``StorageValidationResult`` — a small dataclass carrying the Pass/Warning/Fail
verdict plus the measured footprint, for the storage-budget check.

Storage implications
---------------------
``validate_storage_budget`` is the mechanical enforcement of the 20-30 GB
preferred / 50 GB hard-limit constraint from the implementation architecture
document — it is meant to be run after every dataset-generation batch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .mission_config import ConfigValidationError

if TYPE_CHECKING:
    from .mission_config import MissionConfig
    from .mission_state import MissionState

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1024 ** 3

_REQUIRED_EVENT_LOG_COLUMNS = {
    "event_id",
    "edge_i",
    "edge_j",
    "timestamp_s",
    "event_type",
    "cause",
    "pdr",
    "throughput_bps",
    "delay_s",
}
_VALID_EVENT_TYPES = {"link_up", "link_down"}
_VALID_CAUSES = {"geo", "wave", "doppler", "current", ""}


class MissionStateValidationError(ValueError):
    """Raised when a mission state fails structural or physical validation."""


class SchemaValidationError(ValueError):
    """Raised when an on-disk artifact does not match its expected schema."""


class StorageStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class StorageValidationResult:
    """Result of a storage-budget check against a base directory."""

    status: StorageStatus
    total_bytes: int
    total_gb: float
    preferred_limit_gb: float
    hard_limit_gb: float
    file_count: int
    breakdown_bytes: dict[str, int]

    def summary(self) -> str:
        return (
            f"StorageValidationResult(status={self.status.value}, "
            f"total={self.total_gb:.3f} GB, "
            f"preferred_limit={self.preferred_limit_gb} GB, "
            f"hard_limit={self.hard_limit_gb} GB, "
            f"files={self.file_count})"
        )


# ---------------------------------------------------------------------- #
# Configuration validation
# ---------------------------------------------------------------------- #
def validate_mission_config(config: "MissionConfig") -> None:
    """Re-validate a ``MissionConfig``, raising ``ConfigValidationError`` on
    any violation. This wraps ``MissionConfig.validate()`` (the single source
    of truth for config validation logic) so that all Phase 1 modules depend
    on this module for validation entry points rather than reaching into
    ``mission_config.py`` directly — keeping validation call sites
    discoverable in one place. ``ConfigValidationError`` itself is imported
    from ``mission_config.py``, not redefined here, so a ``try/except
    ConfigValidationError`` in either module catches the same exception type.
    """
    config.validate()


# ---------------------------------------------------------------------- #
# Mission-state validation
# ---------------------------------------------------------------------- #
def validate_mission_state(state: "MissionState", config: "MissionConfig") -> None:
    """Validate a generated ``MissionState`` against the config that should
    have produced it — catches mismatches between what ``MissionGenerator``
    produced and what the config specifies (e.g., a placement bug that drops
    a node), beyond the self-consistency checks ``MissionState.validate()``
    already performs.
    """
    try:
        state.validate()
    except ValueError as exc:
        raise MissionStateValidationError(str(exc)) from exc

    if state.metadata.config_hash != config.config_hash:
        raise MissionStateValidationError(
            f"MissionState.metadata.config_hash ({state.metadata.config_hash}) does not "
            f"match the provided config's hash ({config.config_hash}) — possible config "
            f"drift between generation and validation."
        )

    if not (config.fleet.n_auv_min <= state.metadata.n_auv <= config.fleet.n_auv_max):
        raise MissionStateValidationError(
            f"MissionState.metadata.n_auv ({state.metadata.n_auv}) outside configured "
            f"range [{config.fleet.n_auv_min}, {config.fleet.n_auv_max}]"
        )
    if not (config.fleet.m_iout_min <= state.metadata.m_iout <= config.fleet.m_iout_max):
        raise MissionStateValidationError(
            f"MissionState.metadata.m_iout ({state.metadata.m_iout}) outside configured "
            f"range [{config.fleet.m_iout_min}, {config.fleet.m_iout_max}]"
        )

    width, height = config.envelope.area_size_m
    for node in state.nodes:
        if not (0.0 <= node.x_m <= width) or not (0.0 <= node.y_m <= height):
            raise MissionStateValidationError(
                f"Node {node.node_id} position ({node.x_m}, {node.y_m}) lies outside "
                f"the configured area ({width} x {height})"
            )
        depth_lo, depth_hi = config.envelope.depth_range_m
        # USV is a surface vehicle and is exempt from the depth-range check
        from .mission_state import NodeType as _NodeType  # local import avoids cycle

        if node.node_type != _NodeType.USV and not (depth_lo - 1e-6 <= node.z_m <= depth_hi + 1e-6):
            raise MissionStateValidationError(
                f"Node {node.node_id} depth ({node.z_m}) outside configured range "
                f"[{depth_lo}, {depth_hi}]"
            )


# ---------------------------------------------------------------------- #
# Trajectory array schema validation
# ---------------------------------------------------------------------- #
def validate_node_trajectory_array(
    array: np.ndarray,
    expected_feature_dim: int,
    expected_node_count: int | None = None,
) -> None:
    """Validate the shape, dtype-safety, and finiteness of a stored node-
    trajectory array (``[T, |V|, F_node]``, as written by
    ``trajectory_storage.py``).
    """
    if array.ndim != 3:
        raise SchemaValidationError(
            f"Node trajectory array must be 3-D [T, |V|, F_node], got shape {array.shape}"
        )
    t, v, f = array.shape
    if f != expected_feature_dim:
        raise SchemaValidationError(
            f"Node trajectory array feature dim {f} does not match expected {expected_feature_dim}"
        )
    if expected_node_count is not None and v != expected_node_count:
        raise SchemaValidationError(
            f"Node trajectory array node count {v} does not match expected {expected_node_count}"
        )
    if t < 1:
        raise SchemaValidationError("Node trajectory array must have at least one timestep")

    finite = np.isfinite(array.astype(np.float32, copy=False))
    if not bool(finite.all()):
        n_bad = int((~finite).sum())
        raise SchemaValidationError(
            f"Node trajectory array contains {n_bad} non-finite (NaN/Inf) values"
        )


# ---------------------------------------------------------------------- #
# Event-log schema validation
# ---------------------------------------------------------------------- #
def validate_event_log_schema(records: list[dict]) -> None:
    """Validate a list of flat event records (as produced by
    ``CommunicationEvent.to_record()``) against the required event-log
    schema, including basic per-edge time ordering.
    """
    if not records:
        return  # an empty event log is valid (e.g., a calm-sea-state mission)

    missing_columns = _REQUIRED_EVENT_LOG_COLUMNS - set(records[0].keys())
    if missing_columns:
        raise SchemaValidationError(f"Event log records missing required columns: {missing_columns}")

    last_timestamp_by_edge: dict[tuple[str, str], float] = {}
    for i, record in enumerate(records):
        missing = _REQUIRED_EVENT_LOG_COLUMNS - set(record.keys())
        if missing:
            raise SchemaValidationError(f"Event log record {i} missing columns: {missing}")
        if record["event_type"] not in _VALID_EVENT_TYPES:
            raise SchemaValidationError(
                f"Event log record {i} has invalid event_type: {record['event_type']!r}"
            )
        if record["cause"] not in _VALID_CAUSES:
            raise SchemaValidationError(
                f"Event log record {i} has invalid cause: {record['cause']!r}"
            )
        if record["event_type"] == "link_down" and not record["cause"]:
            raise SchemaValidationError(
                f"Event log record {i} is link_down but has no cause (leakage/label bug)"
            )
        if not (0.0 <= record["pdr"] <= 1.0):
            raise SchemaValidationError(f"Event log record {i} has out-of-range pdr: {record['pdr']}")

        edge_key = tuple(sorted((record["edge_i"], record["edge_j"])))
        ts = float(record["timestamp_s"])
        if not math.isfinite(ts) or ts < 0.0:
            raise SchemaValidationError(f"Event log record {i} has invalid timestamp_s: {ts}")
        prev_ts = last_timestamp_by_edge.get(edge_key)
        if prev_ts is not None and ts < prev_ts:
            raise SchemaValidationError(
                f"Event log record {i} for edge {edge_key} is out of time order "
                f"({ts} < previous {prev_ts}) — possible generation bug"
            )
        last_timestamp_by_edge[edge_key] = ts


# ---------------------------------------------------------------------- #
# Storage-budget validation
# ---------------------------------------------------------------------- #
def validate_storage_budget(
    base_dir: str | Path,
    preferred_limit_gb: float = 30.0,
    hard_limit_gb: float = 50.0,
) -> StorageValidationResult:
    """Walk ``base_dir`` and compute total on-disk footprint, broken down by
    top-level subdirectory, returning a Pass/Warning/Fail verdict against the
    storage budget from the implementation architecture document.

    This function never raises on a budget violation — see module docstring,
    point 3 — callers decide policy based on the returned status.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return StorageValidationResult(
            status=StorageStatus.PASS,
            total_bytes=0,
            total_gb=0.0,
            preferred_limit_gb=preferred_limit_gb,
            hard_limit_gb=hard_limit_gb,
            file_count=0,
            breakdown_bytes={},
        )

    breakdown: dict[str, int] = {}
    total_bytes = 0
    file_count = 0
    for path in base_dir.rglob("*"):
        if path.is_file():
            size = path.stat().st_size
            total_bytes += size
            file_count += 1
            top_level = path.relative_to(base_dir).parts[0] if path.parent != base_dir else path.name
            breakdown[top_level] = breakdown.get(top_level, 0) + size

    total_gb = total_bytes / _BYTES_PER_GB
    if total_gb > hard_limit_gb:
        status = StorageStatus.FAIL
    elif total_gb > preferred_limit_gb:
        status = StorageStatus.WARNING
    else:
        status = StorageStatus.PASS

    result = StorageValidationResult(
        status=status,
        total_bytes=total_bytes,
        total_gb=total_gb,
        preferred_limit_gb=preferred_limit_gb,
        hard_limit_gb=hard_limit_gb,
        file_count=file_count,
        breakdown_bytes=breakdown,
    )
    logger.info("Storage budget check: %s", result.summary())
    return result

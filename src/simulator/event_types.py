"""
event_types.py

Event schema for the PG-GTPP simulator: communication state-change events,
hazard-activation diagnostic events, and the failure-event specialization
used for training labels.

Design decisions
-----------------
1. ``EventType`` and ``CauseType`` are ``str``-valued ``Enum``s so they
   serialize directly to plain strings in parquet/JSON without a custom
   encoder, matching the pattern used for ``NodeType`` in ``mission_state.py``.
2. ``CommunicationEvent`` is the single, general-purpose record written to the
   event log (matching the ``event_log`` schema in the storage-efficient
   dataset design: edge id pair, timestamp, event type, cause, marks). A
   "failure event" is not a separate class — it is a ``CommunicationEvent``
   with ``event_type == LINK_DOWN`` and a non-``None`` cause. Modeling it as a
   distinct subclass would duplicate the schema for no semantic gain; instead
   ``make_failure_event`` is a validated *factory function* that enforces
   exactly those two constraints at construction time. This keeps one schema,
   one serialization path, and one storage table for all link state changes,
   which is what ``trajectory_storage.py`` and the QA framework's leakage
   tests (shifted-window regeneration, time-rescaling theorem) both assume.
3. ``HazardEvent`` is intentionally a separate, lighter record. It represents
   a *diagnostic* observation (a cause-specific hazard crossing a
   significance threshold) used for the hazard-activation-timeline
   visualization and the ambiguous-event-fraction QA metric — it is not a
   training label and does not need the mark fields a ``CommunicationEvent``
   carries. Keeping it separate avoids overloading one record type with two
   different purposes (ground-truth label vs. diagnostic signal), which would
   make the leakage-detection tests harder to reason about.
4. Every event dataclass exposes ``to_record()`` (a flat dict with enum
   values already unwrapped to strings) in addition to ``to_dict()``/
   ``from_dict()``, because ``trajectory_storage.py`` writes rows directly to
   a pandas/pyarrow table and needs flat, primitive-typed dicts, not nested
   dataclass/enum objects.

Data structures
----------------
``EventType`` — {LINK_UP, LINK_DOWN}.
``CauseType`` — {GEOMETRIC, WAVE, DOPPLER, CURRENT}, matching the four named
physical degradation mechanisms from the PG-GTPP research design.
``CommunicationMarks`` — the (PDR, throughput, delay) mark vector attached to
an event.
``CommunicationEvent`` — the primary stored event record.
``HazardEvent`` — a lightweight, non-label diagnostic record.

Storage implications
---------------------
Event records are the "event-based storage" the implementation architecture
mandates: only state *changes* are represented, never continuous per-timestep
covariate trajectories. A mission with ~500 events (per the storage budget
analysis) at ~20 bytes/event in packed form is a negligible fraction of the
~0.95 MB/mission total.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    """Communication link state-change type."""

    LINK_UP = "link_up"
    LINK_DOWN = "link_down"


class CauseType(str, Enum):
    """Named physical degradation mechanism, matching the competing-risks
    cause decomposition (geo / wave / doppler / current) from the PG-GTPP
    research design."""

    GEOMETRIC = "geo"
    WAVE = "wave"
    DOPPLER = "doppler"
    CURRENT = "current"


def _new_event_id() -> str:
    return uuid.uuid4().hex


@dataclass
class CommunicationMarks:
    """Quality mark vector attached to a communication event."""

    pdr: float = 0.0
    throughput_bps: float = 0.0
    delay_s: float = 0.0

    def validate(self) -> None:
        if not (0.0 <= self.pdr <= 1.0):
            raise ValueError(f"CommunicationMarks.pdr must be in [0, 1], got {self.pdr}")
        if self.throughput_bps < 0.0:
            raise ValueError("CommunicationMarks.throughput_bps must be >= 0")
        if self.delay_s < 0.0:
            raise ValueError("CommunicationMarks.delay_s must be >= 0")

    def to_dict(self) -> dict[str, float]:
        return {"pdr": self.pdr, "throughput_bps": self.throughput_bps, "delay_s": self.delay_s}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "CommunicationMarks":
        return cls(**data)


@dataclass
class CommunicationEvent:
    """A single communication link state-change record.

    This is the record type persisted to ``event_log`` by
    ``trajectory_storage.py``. ``cause`` is ``None`` for recovery
    (``LINK_UP``) events by default (recovery-cause attribution is out of
    scope for Phase 1) and mandatory for events constructed via
    ``make_failure_event``.
    """

    edge_i: str
    edge_j: str
    timestamp_s: float
    event_type: EventType
    cause: Optional[CauseType] = None
    marks: CommunicationMarks = field(default_factory=CommunicationMarks)
    event_id: str = field(default_factory=_new_event_id)

    def validate(self) -> None:
        if self.edge_i == self.edge_j:
            raise ValueError("CommunicationEvent cannot connect a node to itself")
        if self.timestamp_s < 0.0:
            raise ValueError("CommunicationEvent.timestamp_s must be >= 0")
        if self.event_type == EventType.LINK_DOWN and self.cause is None:
            raise ValueError("CommunicationEvent with event_type=LINK_DOWN must specify a cause")
        self.marks.validate()

    @property
    def edge_key(self) -> tuple[str, str]:
        """Canonical, order-independent key for this undirected edge."""
        return tuple(sorted((self.edge_i, self.edge_j)))  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "edge_i": self.edge_i,
            "edge_j": self.edge_j,
            "timestamp_s": self.timestamp_s,
            "event_type": self.event_type.value,
            "cause": self.cause.value if self.cause is not None else None,
            "marks": self.marks.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommunicationEvent":
        data = dict(data)
        data["event_type"] = EventType(data["event_type"])
        data["cause"] = CauseType(data["cause"]) if data.get("cause") is not None else None
        data["marks"] = CommunicationMarks.from_dict(data["marks"])
        return cls(**data)

    def to_record(self) -> dict[str, Any]:
        """Flat, primitive-typed representation for a tabular (parquet) row."""
        return {
            "event_id": self.event_id,
            "edge_i": self.edge_i,
            "edge_j": self.edge_j,
            "timestamp_s": self.timestamp_s,
            "event_type": self.event_type.value,
            "cause": self.cause.value if self.cause is not None else "",
            "pdr": self.marks.pdr,
            "throughput_bps": self.marks.throughput_bps,
            "delay_s": self.marks.delay_s,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "CommunicationEvent":
        """Inverse of ``to_record`` — reconstructs from a flat parquet row."""
        cause_raw = record.get("cause", "")
        return cls(
            event_id=record["event_id"],
            edge_i=record["edge_i"],
            edge_j=record["edge_j"],
            timestamp_s=float(record["timestamp_s"]),
            event_type=EventType(record["event_type"]),
            cause=CauseType(cause_raw) if cause_raw else None,
            marks=CommunicationMarks(
                pdr=float(record["pdr"]),
                throughput_bps=float(record["throughput_bps"]),
                delay_s=float(record["delay_s"]),
            ),
        )


def make_failure_event(
    edge_i: str,
    edge_j: str,
    timestamp_s: float,
    cause: CauseType,
    marks: Optional[CommunicationMarks] = None,
) -> CommunicationEvent:
    """Factory for a LINK_DOWN event with a mandatory, explicit cause.

    This is the single construction path failure events should go through —
    enforcing "a failure event always has a cause" at the type-construction
    boundary rather than relying on callers to remember to set it.
    """
    event = CommunicationEvent(
        edge_i=edge_i,
        edge_j=edge_j,
        timestamp_s=timestamp_s,
        event_type=EventType.LINK_DOWN,
        cause=cause,
        marks=marks if marks is not None else CommunicationMarks(),
    )
    event.validate()
    return event


def make_recovery_event(
    edge_i: str,
    edge_j: str,
    timestamp_s: float,
    marks: Optional[CommunicationMarks] = None,
) -> CommunicationEvent:
    """Factory for a LINK_UP (recovery) event."""
    event = CommunicationEvent(
        edge_i=edge_i,
        edge_j=edge_j,
        timestamp_s=timestamp_s,
        event_type=EventType.LINK_UP,
        cause=None,
        marks=marks if marks is not None else CommunicationMarks(pdr=1.0),
    )
    event.validate()
    return event


@dataclass
class HazardEvent:
    """A diagnostic record of a cause-specific hazard crossing a significance
    threshold — used for QA visual diagnostics (hazard-activation timelines)
    and the ambiguous-event-fraction metric, not as a training label."""

    edge_i: str
    edge_j: str
    timestamp_s: float
    cause: CauseType
    intensity_value: float
    event_id: str = field(default_factory=_new_event_id)

    def validate(self) -> None:
        if self.edge_i == self.edge_j:
            raise ValueError("HazardEvent cannot connect a node to itself")
        if self.intensity_value < 0.0:
            raise ValueError("HazardEvent.intensity_value must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "edge_i": self.edge_i,
            "edge_j": self.edge_j,
            "timestamp_s": self.timestamp_s,
            "cause": self.cause.value,
            "intensity_value": self.intensity_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HazardEvent":
        data = dict(data)
        data["cause"] = CauseType(data["cause"])
        return cls(**data)

    def to_record(self) -> dict[str, Any]:
        return self.to_dict()

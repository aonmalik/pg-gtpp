"""
trajectory_storage.py

Storage-efficient persistence for missions: node trajectories, event logs,
and the mission index, plus lazy, on-demand loading.

Design decisions
-----------------
1. **Node trajectories** are stored as compressed ``.npz`` files, one per
   mission, with the array cast to ``float16``. Two things follow from this
   choice, stated honestly rather than glossed over: (a) ``float16`` storage
   is a pure storage optimization — arrays are upcast to ``float32``
   immediately on load, before any numeric computation touches them; (b)
   because the array is *compressed*, true OS-level memory-mapping is not
   possible (mmap requires an uncompressed, fixed-offset file layout). This
   is an explicit, documented tradeoff: "lazy loading" in this module means
   *load-on-first-access, not at corpus-construction time*, not
   *zero-copy mmap*. This is an acceptable tradeoff specifically because
   each mission's array is small (~0.9 MB per the storage budget analysis)
   — full decompression per access is negligible regardless of corpus size,
   since missions are never all decompressed simultaneously.
2. **Event logs** are stored as a single, partitioned Parquet dataset
   (partitioned by ``mission_id``) rather than one small file per mission.
   Parquet's columnar layout gives free, effective compression on the
   already-small event data, and partitioning avoids the filesystem
   overhead of thousands of tiny separate files, which is called out
   explicitly as a failure mode in the QA framework's schema design.
3. **The mission index** (``mission_index.parquet``) is a single, flat,
   append-only Parquet file. It is the *only* place per-mission metadata is
   duplicated outside the per-mission artifacts themselves, and it is
   designed to be small enough (one row per mission, all scalar columns) to
   load entirely into memory even for a 20,000-mission corpus — this is
   what lets ``MissionDataStore`` answer "which missions are in the test
   split" without touching any bulk trajectory data.
4. Writing is append-safe: ``TrajectoryWriter.write_mission`` will refuse to
   silently overwrite an existing mission's node-trajectory file (raises)
   unless ``overwrite=True`` is explicitly passed, matching the
   reproducibility framework's "additive, immutable dataset versions"
   principle — accidental in-place mutation of an already-generated mission
   is a hard error, not a silent occurrence.
5. ``MissionDataStore`` never loads more than one mission's bulk data into
   memory at a time unless the caller explicitly iterates and retains
   references — there is no corpus-wide cache inside this class.

Data structures
----------------
``TrajectoryWriter`` — writes node trajectories, event logs, and mission
index rows.
``MissionDataStore`` — reads the mission index and lazily loads per-mission
artifacts on demand.

Storage implications
---------------------
This module is the direct implementation of the storage budget analyzed in
the implementation architecture document: ~0.95 MB/mission, dominated by the
compressed float16 node-trajectory array, with event logs and index rows
contributing a negligible fraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .event_types import CommunicationEvent
from .mission_state import MissionMetadata, MissionState, NodeState, NodeType
from .validation_utils import (
    SchemaValidationError,
    validate_event_log_schema,
    validate_node_trajectory_array,
)

logger = logging.getLogger(__name__)

_NODE_TRAJECTORY_DIRNAME = "raw_trajectories"
_EVENT_LOG_DIRNAME = "event_log"
_MISSION_INDEX_FILENAME = "mission_index.parquet"
_NODE_FEATURE_ORDER = (
    "x_m",
    "y_m",
    "z_m",
    "vx_mps",
    "vy_mps",
    "heading_rad",
    "battery_frac",
    "fim_uncertainty",
)
# Node-type one-hot columns are appended after the continuous features above,
# giving F_node = len(_NODE_FEATURE_ORDER) + 3 (USV/AUV/IOUT) = 11.
# (The implementation architecture document's F_node=10 collapses type into a
# single categorical column; here it is stored as an explicit one-hot triple
# instead, which is the more storage-schema-robust choice — a categorical
# code is an implicit contract on ordering, whereas a one-hot triple is
# self-describing and trivially validated column-by-column.)
_NODE_TYPE_ORDER = (NodeType.USV, NodeType.AUV, NodeType.IOUT)
F_NODE = len(_NODE_FEATURE_ORDER) + len(_NODE_TYPE_ORDER)


class MissionAlreadyExistsError(FileExistsError):
    """Raised when attempting to write a mission that already exists on disk
    without explicitly passing ``overwrite=True``."""


def _node_state_to_feature_row(node: NodeState) -> np.ndarray:
    continuous = np.array([getattr(node, f) for f in _NODE_FEATURE_ORDER], dtype=np.float32)
    one_hot = np.array(
        [1.0 if node.node_type == t else 0.0 for t in _NODE_TYPE_ORDER], dtype=np.float32
    )
    return np.concatenate([continuous, one_hot])


class TrajectoryWriter:
    """Writes missions to the storage-efficient on-disk format."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.node_trajectory_dir = self.base_dir / _NODE_TRAJECTORY_DIRNAME
        self.event_log_dir = self.base_dir / _EVENT_LOG_DIRNAME
        self.mission_index_path = self.base_dir / _MISSION_INDEX_FILENAME
        self.node_trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.event_log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Node trajectory
    # ------------------------------------------------------------------ #
    def _node_trajectory_path(self, mission_id: str) -> Path:
        return self.node_trajectory_dir / f"{mission_id}.npz"

    def write_node_trajectory(
        self, mission_id: str, node_ids: list[str], array: np.ndarray, overwrite: bool = False
    ) -> Path:
        """Write a ``[T, |V|, F_NODE]`` float array to a compressed, float16
        ``.npz`` file, alongside the ordered ``node_ids`` needed to interpret
        the array's second axis.
        """
        validate_node_trajectory_array(array, expected_feature_dim=F_NODE, expected_node_count=len(node_ids))
        path = self._node_trajectory_path(mission_id)
        if path.exists() and not overwrite:
            raise MissionAlreadyExistsError(
                f"Node trajectory for mission {mission_id!r} already exists at {path}; "
                f"pass overwrite=True to replace it."
            )
        np.savez_compressed(
            path,
            array=array.astype(np.float16),
            node_ids=np.array(node_ids, dtype=object),
            feature_names=np.array(list(_NODE_FEATURE_ORDER) + [t.value for t in _NODE_TYPE_ORDER], dtype=object),
        )
        logger.debug("Wrote node trajectory for %s to %s (%d bytes)", mission_id, path, path.stat().st_size)
        return path

    def write_initial_state_as_trajectory(
        self, state: MissionState, overwrite: bool = False
    ) -> Path:
        """Convenience wrapper: write a Phase-1 ``MissionState`` (a single
        t=0 snapshot, no propagation yet) as a length-1 node trajectory
        array, so the on-disk format is already forward-compatible with
        later phases that write full-length trajectories through the same
        interface.
        """
        node_ids = state.node_ids
        rows = np.stack([_node_state_to_feature_row(n) for n in state.nodes], axis=0)  # [|V|, F_NODE]
        array = rows[np.newaxis, :, :]  # [T=1, |V|, F_NODE]
        return self.write_node_trajectory(state.metadata.mission_id, node_ids, array, overwrite=overwrite)

    # ------------------------------------------------------------------ #
    # Event log
    # ------------------------------------------------------------------ #
    def write_event_log(self, mission_id: str, events: list[CommunicationEvent]) -> Path:
        """Write a mission's event log as one partition of the Parquet
        event-log dataset. An empty ``events`` list is valid and still
        writes a (zero-row, schema-only) partition, so the dataset's
        partition set always matches the set of generated missions exactly.
        """
        records = [e.to_record() for e in events]
        validate_event_log_schema(records)

        mission_partition_dir = self.event_log_dir / f"mission_id={mission_id}"
        mission_partition_dir.mkdir(parents=True, exist_ok=True)
        path = mission_partition_dir / "part-0.parquet"

        if records:
            table = pa.Table.from_pylist(records)
        else:
            # zero-row table with the correct schema, so downstream readers
            # never have to special-case "mission generated zero events"
            empty_schema = pa.schema(
                [
                    ("event_id", pa.string()),
                    ("edge_i", pa.string()),
                    ("edge_j", pa.string()),
                    ("timestamp_s", pa.float64()),
                    ("event_type", pa.string()),
                    ("cause", pa.string()),
                    ("pdr", pa.float64()),
                    ("throughput_bps", pa.float64()),
                    ("delay_s", pa.float64()),
                ]
            )
            table = pa.Table.from_pylist([], schema=empty_schema)

        pq.write_table(table, path, compression="zstd")
        logger.debug("Wrote %d event(s) for %s to %s", len(records), mission_id, path)
        return path

    # ------------------------------------------------------------------ #
    # Mission index
    # ------------------------------------------------------------------ #
    def append_mission_index(self, metadata: MissionMetadata) -> None:
        """Append one row to ``mission_index.parquet``. Refuses to append a
        duplicate ``mission_id`` (append-only, no in-place mutation).
        """
        row = metadata.to_dict()
        row["extra"] = str(row["extra"])  # flatten dict to a string for a stable columnar type

        if self.mission_index_path.exists():
            existing = pq.read_table(self.mission_index_path).to_pandas()
            if metadata.mission_id in set(existing["mission_id"]):
                raise MissionAlreadyExistsError(
                    f"mission_id {metadata.mission_id!r} already present in "
                    f"{self.mission_index_path}; the mission index is append-only."
                )
            updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        else:
            updated = pd.DataFrame([row])

        pq.write_table(pa.Table.from_pandas(updated, preserve_index=False), self.mission_index_path)
        logger.debug("Appended mission %s to mission index", metadata.mission_id)

    # ------------------------------------------------------------------ #
    # Convenience: write everything for a Phase-1 mission at once
    # ------------------------------------------------------------------ #
    def write_mission(
        self,
        state: MissionState,
        events: list[CommunicationEvent] | None = None,
        overwrite: bool = False,
        array: np.ndarray | None = None,
        node_ids: list[str] | None = None,
    ) -> None:
        """Write a mission's node trajectory, event log, and index row in
        one call.

        ``array``/``node_ids`` should be the full, multi-timestep trajectory
        from ``TrajectoryGenerator.propagate`` whenever one exists (i.e.,
        for every mission generated since Phase 3). If omitted, this falls
        back to ``write_initial_state_as_trajectory`` (a length-1 array from
        ``state`` alone) — the only correct behavior for Phase 1 callers
        that have no propagated trajectory yet, but a silent, storage-
        relevant bug for any Phase-3-or-later caller that forgets to pass
        the full array: this method used to *always* take the length-1
        path regardless of caller phase, discarding the real trajectory
        every time. That bug is why an early batch-generation run measured
        ~24 KB/mission instead of the real, expected sub-megabyte figure —
        the omitted array/node_ids parameters are the fix.
        """
        if array is not None and node_ids is not None:
            self.write_node_trajectory(state.metadata.mission_id, node_ids, array, overwrite=overwrite)
        else:
            self.write_initial_state_as_trajectory(state, overwrite=overwrite)
        self.write_event_log(state.metadata.mission_id, events or [])
        self.append_mission_index(state.metadata)


class MissionDataStore:
    """Reads the mission index and lazily loads per-mission artifacts."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.node_trajectory_dir = self.base_dir / _NODE_TRAJECTORY_DIRNAME
        self.event_log_dir = self.base_dir / _EVENT_LOG_DIRNAME
        self.mission_index_path = self.base_dir / _MISSION_INDEX_FILENAME
        self._index_cache: pd.DataFrame | None = None

    def _load_index(self) -> pd.DataFrame:
        if self._index_cache is None:
            if not self.mission_index_path.exists():
                raise FileNotFoundError(f"No mission index found at {self.mission_index_path}")
            self._index_cache = pq.read_table(self.mission_index_path).to_pandas()
        return self._index_cache

    def refresh(self) -> None:
        """Force the next index access to re-read from disk (e.g. after a
        concurrent writer has appended new missions)."""
        self._index_cache = None

    def mission_ids(self, split: str | None = None) -> list[str]:
        index = self._load_index()
        if split is not None:
            index = index[index["split"] == split]
        return list(index["mission_id"])

    def load_metadata(self, mission_id: str) -> MissionMetadata:
        index = self._load_index()
        rows = index[index["mission_id"] == mission_id]
        if rows.empty:
            raise KeyError(f"mission_id {mission_id!r} not found in mission index")
        row = rows.iloc[0].to_dict()
        row["extra"] = {}  # 'extra' was flattened to str at write time; Phase 1 does not round-trip it
        return MissionMetadata.from_dict(row)

    def load_node_trajectory(self, mission_id: str) -> tuple[np.ndarray, list[str]]:
        """Lazily load a mission's node-trajectory array (upcast to
        float32) and its ordered node-id list. See module docstring, point
        1, for why this is "load on access", not a true zero-copy mmap.
        """
        path = self.node_trajectory_dir / f"{mission_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"No node trajectory found for mission {mission_id!r} at {path}")
        with np.load(path, allow_pickle=True) as data:
            array = data["array"].astype(np.float32)
            node_ids = list(data["node_ids"])
        validate_node_trajectory_array(array, expected_feature_dim=F_NODE, expected_node_count=len(node_ids))
        return array, node_ids

    def load_event_log(self, mission_id: str) -> pd.DataFrame:
        partition_path = self.event_log_dir / f"mission_id={mission_id}" / "part-0.parquet"
        if not partition_path.exists():
            raise FileNotFoundError(f"No event log found for mission {mission_id!r} at {partition_path}")
        table = pq.read_table(partition_path)
        df = table.to_pandas()
        if len(df) > 0:
            validate_event_log_schema(df.to_dict(orient="records"))
        return df

    def load_mission(self, mission_id: str) -> tuple[MissionMetadata, np.ndarray, list[str], pd.DataFrame]:
        """Convenience: load everything for a single mission in one call."""
        metadata = self.load_metadata(mission_id)
        array, node_ids = self.load_node_trajectory(mission_id)
        events = self.load_event_log(mission_id)
        return metadata, array, node_ids, events

    def __len__(self) -> int:
        return len(self._load_index())

    def __repr__(self) -> str:  # pragma: no cover - trivial
        try:
            n = len(self)
        except FileNotFoundError:
            n = 0
        return f"MissionDataStore(base_dir={self.base_dir}, missions={n})"

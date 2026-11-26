"""
training/streaming_dataset.py

A disk-backed, lazily-loaded mission source for training -- the production
counterpart to train.py's MissionCache, which holds every mission's full
array in memory simultaneously and is explicitly documented there as not a
corpus-scale solution. StreamingMissionSource instead reads missions on
demand from an already-generated, on-disk corpus (via Phase 1's
MissionDataStore), with a small bounded cache so nearby training steps
that reuse the same mission don't re-read it from disk every time.

Design decisions
-----------------
1. Physics and events are reconstructed from disk, not re-simulated.
   MissionDataStore persists the node trajectory array and the event log,
   but not the WaveField/CurrentField/HazardEvaluator objects -- those are
   cheap, closed-form, and exactly reproducible from (master_seed,
   mission_id) via RandomManager. This re-derives them rather than storing
   them, consistent with "recompute, don't store" -- it does NOT re-run
   the trajectory/event generators (those outputs ARE stored and read
   directly, since they are the expensive, stochastic-thinning-driven
   artifacts the storage design exists to avoid regenerating).

   PRECISION CAVEAT (inherited from Phase 1's storage format, not
   introduced here): node trajectories are stored as float16 (see
   trajectory_storage.py's design rationale -- a deliberate storage-size
   tradeoff). A mission loaded from disk therefore has node positions
   accurate to float16 precision (~3 decimal digits, verified to a
   relative error under 0.05% on real generated data), not the exact
   float32/float64 values an in-memory MissionCache would hold for a
   freshly-generated mission. This produces correspondingly tiny
   (sub-percent) differences in derived physics quantities like hazard
   values. This is a pre-existing property of the storage format that
   applies to every consumer of stored trajectories, not something
   specific to streaming -- documented here because this module is the
   first one to compare streaming-loaded output directly against an
   in-memory reference.
2. Node types are decoded from the stored array's one-hot columns
   (indices 8:11, the USV/AUV/IoUT block), not re-derived from node-ID
   naming conventions. Verified to round-trip exactly against ground
   truth before being relied on here.
3. StreamingMissionSource deliberately mirrors MissionCache's public
   interface (sample_batch, split, all_windows) so train()/evaluate() work
   unchanged against either data source.
4. The bounded LRU cache size is a plain constructor parameter, not
   auto-tuned.
5. config and master_seed must be supplied by the caller, not inferred
   from the on-disk corpus alone -- validated against each mission's
   stored config_hash, raising loudly on a mismatch rather than silently
   reconstructing wrong physics from a guessed config.

Data structures
----------------
StreamingMissionSource -- the disk-backed mission source.

Storage implications
---------------------
None beyond what MissionDataStore already persists -- reads existing
on-disk artifacts, does not write new ones.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..dataset.window_builder import WindowBatch, build_window, collate_windows
from ..event_types import CommunicationEvent
from ..mission_config import MissionConfig
from ..mission_state import NodeType
from ..physics import AcousticChannel, CurrentField, WaveField
from ..physics_intensity import HazardEvaluator
from ..random_manager import RandomManager
from ..trajectory_storage import MissionDataStore, _NODE_TYPE_ORDER
from .train import MissionArtifacts


class MissionConfigMismatchError(ValueError):
    """Raised when a mission's stored config_hash does not match the
    config supplied to StreamingMissionSource."""


def _decode_node_types(array: np.ndarray, node_ids: list[str]) -> dict[str, NodeType]:
    """Decode each node's type from the one-hot block at array columns 8:11."""
    type_block = array[0, :, 8:11]
    return {
        node_ids[i]: _NODE_TYPE_ORDER[int(np.argmax(type_block[i]))]
        for i in range(len(node_ids))
    }


def load_mission_artifacts(
    store: MissionDataStore, mission_id: str, config: MissionConfig, master_seed: int
) -> MissionArtifacts:
    """Load one mission's artifacts from disk, reconstructing the physics
    primitives and event list from stored trajectory/event-log data."""
    metadata = store.load_metadata(mission_id)
    if metadata.config_hash != config.config_hash:
        raise MissionConfigMismatchError(
            f"Mission {mission_id!r} was generated with config_hash={metadata.config_hash!r}, "
            f"but the supplied config has config_hash={config.config_hash!r}. Loading it with a "
            f"mismatched config would reconstruct incorrect physics silently -- refusing instead."
        )

    array, node_ids = store.load_node_trajectory(mission_id)
    node_types = _decode_node_types(array, node_ids)
    timestamps = np.arange(array.shape[0], dtype=np.float64) / config.envelope.node_sample_rate_hz

    events_df = store.load_event_log(mission_id)
    events: list[CommunicationEvent] = [
        CommunicationEvent.from_record(row) for row in events_df.to_dict(orient="records")
    ]

    rm = RandomManager(master_seed=master_seed)
    wave_field = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
    current_field = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
    acoustic_channel = AcousticChannel(config.acoustic)
    evaluator = HazardEvaluator(acoustic_channel, wave_field, current_field, config.hazard)

    return MissionArtifacts(
        mission_id=mission_id, array=array, node_ids=node_ids, node_types=node_types,
        timestamps=timestamps, events=events, evaluator=evaluator,
        candidate_range_factor=config.graph.candidate_range_factor,
    )


class StreamingMissionSource:
    """Disk-backed mission source with a small bounded LRU cache, mirroring
    MissionCache's public interface."""

    def __init__(
        self,
        store: MissionDataStore,
        config: MissionConfig,
        master_seed: int,
        mission_ids: list[str] | None = None,
        cache_size: int = 16,
    ) -> None:
        self.store = store
        self.config = config
        self.master_seed = master_seed
        self.mission_ids = mission_ids if mission_ids is not None else store.mission_ids()
        if not self.mission_ids:
            raise ValueError("StreamingMissionSource has no missions to draw from")
        self.cache_size = cache_size
        self._cache: OrderedDict[str, MissionArtifacts] = OrderedDict()

    def _get(self, mission_id: str) -> MissionArtifacts:
        if mission_id in self._cache:
            self._cache.move_to_end(mission_id)
            return self._cache[mission_id]

        artifacts = load_mission_artifacts(self.store, mission_id, self.config, self.master_seed)
        self._cache[mission_id] = artifacts
        self._cache.move_to_end(mission_id)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return artifacts

    def sample_batch(
        self, rng: np.random.Generator, batch_size: int, window_length: int, lookahead_s: float
    ) -> WindowBatch:
        """Sample batch_size random (mission, reference_t_idx) pairs,
        lazily loading each referenced mission from disk (or the cache)."""
        samples = []
        for _ in range(batch_size):
            mission_id = self.mission_ids[rng.integers(0, len(self.mission_ids))]
            mission = self._get(mission_id)
            valid = [i for i, t in enumerate(mission.timestamps)
                     if float(t) + lookahead_s <= float(mission.timestamps[-1]) + 1e-9]
            if not valid:
                raise ValueError(f"mission {mission_id} is shorter than lookahead_s={lookahead_s}")
            t_idx = int(valid[rng.integers(0, len(valid))])
            samples.append(
                build_window(
                    mission.mission_id, mission.array, mission.node_ids, mission.node_types,
                    mission.timestamps, mission.events, mission.evaluator, mission.candidate_range_factor,
                    reference_t_idx=t_idx, window_length=window_length, lookahead_s=lookahead_s,
                )
            )
        return collate_windows(samples)

    def split(self, val_fraction: float, seed: int) -> tuple["StreamingMissionSource", "StreamingMissionSource"]:
        """Partition at the mission level -- only partitions the list of
        mission IDs; no missions are loaded or duplicated on disk."""
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        if len(self.mission_ids) < 2:
            raise ValueError(f"split() requires at least 2 missions, got {len(self.mission_ids)}")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.mission_ids))
        n_val = max(1, int(round(len(self.mission_ids) * val_fraction)))
        val_ids = {self.mission_ids[i] for i in indices[:n_val]}

        train_ids = [m for m in self.mission_ids if m not in val_ids]
        val_ids_list = [m for m in self.mission_ids if m in val_ids]
        train_source = StreamingMissionSource(self.store, self.config, self.master_seed, train_ids, self.cache_size)
        val_source = StreamingMissionSource(self.store, self.config, self.master_seed, val_ids_list, self.cache_size)
        return train_source, val_source

    def all_windows(self, window_length: int, lookahead_s: float, stride: int = 1) -> WindowBatch:
        """Build a batch from every (mission, reference_t_idx) pair across
        every mission this source can see, subsampled by stride -- used
        for deterministic validation evaluation."""
        samples = []
        for mission_id in self.mission_ids:
            mission = self._get(mission_id)
            valid = [i for i, t in enumerate(mission.timestamps)
                     if float(t) + lookahead_s <= float(mission.timestamps[-1]) + 1e-9]
            for t_idx in valid[::stride]:
                samples.append(
                    build_window(
                        mission.mission_id, mission.array, mission.node_ids, mission.node_types,
                        mission.timestamps, mission.events, mission.evaluator, mission.candidate_range_factor,
                        reference_t_idx=t_idx, window_length=window_length, lookahead_s=lookahead_s,
                    )
                )
        return collate_windows(samples)

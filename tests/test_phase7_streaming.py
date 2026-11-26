"""
tests/test_phase7_streaming.py

Pytest suite for the disk-backed streaming mission source
(training/streaming_dataset.py): correctness against an in-memory
reference, LRU cache bounding, mission-level split, config-mismatch
detection, and end-to-end training against a real on-disk corpus.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simulator.event_generator import EventGenerator
from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.mission_generator import MissionGenerator
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.physics_intensity import HazardEvaluator
from simulator.random_manager import RandomManager
from simulator.dataset.window_builder import build_window
from simulator.trajectory_generator import TrajectoryGenerator
from simulator.trajectory_storage import MissionDataStore, TrajectoryWriter
from simulator.training.streaming_dataset import (
    MissionConfigMismatchError,
    StreamingMissionSource,
    load_mission_artifacts,
)
from simulator.training.train import TrainConfig, train


@pytest.fixture
def fast_config():
    base = default_config(master_seed=42)
    small_fleet = FleetConfig(
        usv_count=1, n_auv_min=3, n_auv_max=5, m_iout_min=6, m_iout_max=10,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    short_envelope = MissionEnvelopeConfig(
        duration_s=150.0, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m, depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    config = MissionConfig(
        master_seed=42, split="train", wave=base.wave, current=base.current,
        acoustic=base.acoustic, fleet=small_fleet, envelope=short_envelope,
        hazard=base.hazard, graph=base.graph,
    )
    config.validate()
    return config


@pytest.fixture
def on_disk_corpus(tmp_path, fast_config):
    """Generates a small real corpus directly to disk (not via
    MissionCache) and returns (store, config, master_seed, mission_ids)."""
    rm = RandomManager(master_seed=42)
    writer = TrajectoryWriter(tmp_path)
    mission_ids = []
    for i in range(6):
        mission_id = f"corpus_{i}"
        state = MissionGenerator(fast_config, rm).generate(mission_id)
        array, node_ids = TrajectoryGenerator(fast_config, rm).propagate(state)
        wf = WaveField.sample(fast_config.wave, rm.get_numpy_rng(mission_id, "wave"))
        cf = CurrentField.sample(fast_config.current, rm.get_numpy_rng(mission_id, "current"))
        ch = AcousticChannel(fast_config.acoustic)
        node_types = {n.node_id: n.node_type for n in state.nodes}
        timestamps = np.arange(array.shape[0], dtype=np.float64) / fast_config.envelope.node_sample_rate_hz
        events = EventGenerator(fast_config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)
        writer.write_mission(state, events, array=array, node_ids=node_ids)
        mission_ids.append(mission_id)
    store = MissionDataStore(tmp_path)
    return store, fast_config, 42, mission_ids


# ---------------------------------------------------------------------- #
# load_mission_artifacts correctness against an in-memory reference
# ---------------------------------------------------------------------- #
def test_loaded_artifacts_match_in_memory_reference(on_disk_corpus):
    store, config, master_seed, mission_ids = on_disk_corpus
    mission_id = mission_ids[0]

    rm = RandomManager(master_seed=master_seed)
    state = MissionGenerator(config, rm).generate(mission_id)
    array, node_ids = TrajectoryGenerator(config, rm).propagate(state)
    wf = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
    cf = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
    ch = AcousticChannel(config.acoustic)
    node_types = {n.node_id: n.node_type for n in state.nodes}
    timestamps = np.arange(array.shape[0], dtype=np.float64) / config.envelope.node_sample_rate_hz
    events = EventGenerator(config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)
    evaluator = HazardEvaluator(ch, wf, cf, config.hazard)

    loaded = load_mission_artifacts(store, mission_id, config, master_seed)

    assert loaded.mission_id == mission_id
    assert loaded.node_ids == node_ids
    assert loaded.node_types == node_types
    assert len(loaded.events) == len(events)
    assert np.allclose(loaded.array, array, rtol=5e-3, atol=1e-2)

    sample_ref = build_window(mission_id, array, node_ids, node_types, timestamps, events, evaluator,
                               config.graph.candidate_range_factor, 50, 8, 30.0)
    sample_loaded = build_window(loaded.mission_id, loaded.array, loaded.node_ids, loaded.node_types,
                                  loaded.timestamps, loaded.events, loaded.evaluator,
                                  loaded.candidate_range_factor, 50, 8, 30.0)
    assert torch.equal(sample_ref.edge_index, sample_loaded.edge_index)
    assert torch.equal(sample_ref.edge_link_down_target, sample_loaded.edge_link_down_target)
    assert torch.equal(sample_ref.edge_any_event_target, sample_loaded.edge_any_event_target)
    assert torch.allclose(sample_ref.edge_physics, sample_loaded.edge_physics, rtol=5e-2, atol=1e-3)


def test_config_mismatch_is_detected(on_disk_corpus):
    store, config, master_seed, mission_ids = on_disk_corpus
    bad_config = MissionConfig(
        master_seed=999, split="train", wave=config.wave, current=config.current,
        acoustic=config.acoustic, fleet=config.fleet, envelope=config.envelope,
        hazard=config.hazard, graph=config.graph,
    )
    with pytest.raises(MissionConfigMismatchError):
        load_mission_artifacts(store, mission_ids[0], bad_config, master_seed)


# ---------------------------------------------------------------------- #
# StreamingMissionSource
# ---------------------------------------------------------------------- #
def test_streaming_source_lists_all_missions(on_disk_corpus):
    store, config, master_seed, mission_ids = on_disk_corpus
    source = StreamingMissionSource(store, config, master_seed)
    assert set(source.mission_ids) == set(mission_ids)


def test_streaming_source_rejects_empty_mission_list(on_disk_corpus):
    store, config, master_seed, _ = on_disk_corpus
    with pytest.raises(ValueError):
        StreamingMissionSource(store, config, master_seed, mission_ids=[])


def test_lru_cache_is_bounded(on_disk_corpus):
    store, config, master_seed, mission_ids = on_disk_corpus
    source = StreamingMissionSource(store, config, master_seed, cache_size=2)
    rng = np.random.default_rng(0)
    source.sample_batch(rng, batch_size=20, window_length=6, lookahead_s=20.0)
    assert len(source._cache) <= 2


def test_sample_batch_produces_valid_windows(on_disk_corpus):
    store, config, master_seed, mission_ids = on_disk_corpus
    source = StreamingMissionSource(store, config, master_seed, cache_size=3)
    rng = np.random.default_rng(0)
    batch = source.sample_batch(rng, batch_size=5, window_length=6, lookahead_s=20.0)
    assert batch.node_features.shape[0] == 5
    assert batch.lookahead_s == 20.0


def test_split_is_disjoint_and_reproducible(on_disk_corpus):
    store, config, master_seed, mission_ids = on_disk_corpus
    source = StreamingMissionSource(store, config, master_seed)
    train_src, val_src = source.split(val_fraction=0.34, seed=0)
    assert set(train_src.mission_ids).isdisjoint(set(val_src.mission_ids))
    assert set(train_src.mission_ids) | set(val_src.mission_ids) == set(mission_ids)

    train_src2, val_src2 = source.split(val_fraction=0.34, seed=0)
    assert set(val_src.mission_ids) == set(val_src2.mission_ids)


def test_split_rejects_invalid_fraction(on_disk_corpus):
    store, config, master_seed, _ = on_disk_corpus
    source = StreamingMissionSource(store, config, master_seed)
    with pytest.raises(ValueError):
        source.split(val_fraction=1.5, seed=0)


def test_all_windows_covers_every_mission(on_disk_corpus):
    store, config, master_seed, mission_ids = on_disk_corpus
    source = StreamingMissionSource(store, config, master_seed, cache_size=10)
    batch = source.all_windows(window_length=6, lookahead_s=20.0, stride=20)
    assert set(batch.mission_ids) == set(mission_ids)


# ---------------------------------------------------------------------- #
# End-to-end: training against the streaming source (the real regression test)
# ---------------------------------------------------------------------- #
def test_training_against_streaming_source_loss_decreases(on_disk_corpus):
    """Regression test: the training loop must work identically (loss
    decreases) whether fed a MissionCache or a StreamingMissionSource --
    they are meant to be interchangeable via the shared duck-typed
    interface. This directly exercises an interface bug found and fixed
    while building this phase (train() originally hardcoded
    cache.missions, a MissionCache-only attribute)."""
    store, config, master_seed, mission_ids = on_disk_corpus
    source = StreamingMissionSource(store, config, master_seed, cache_size=4)

    torch.manual_seed(1234)
    model = PGGTPPModel(f_node=11, hidden_dim=16)
    train_config = TrainConfig(n_steps=40, batch_size=4, window_length=6, lookahead_s=20.0,
                                learning_rate=1e-2, log_every=0, seed=1)
    history = train(model, source, train_config)

    first_avg = np.mean(history.total_losses[:5])
    last_avg = np.mean(history.total_losses[-5:])
    assert last_avg < first_avg


def test_training_with_streaming_train_val_split(on_disk_corpus):
    store, config, master_seed, mission_ids = on_disk_corpus
    source = StreamingMissionSource(store, config, master_seed, cache_size=4)
    train_src, val_src = source.split(val_fraction=0.34, seed=0)

    model = PGGTPPModel(f_node=11, hidden_dim=16)
    train_config = TrainConfig(n_steps=20, batch_size=4, window_length=6, lookahead_s=20.0,
                                learning_rate=1e-2, log_every=0, seed=1, val_every=10, val_stride=15)
    history = train(model, train_src, train_config, val_cache=val_src)

    assert history.val_steps == [10, 20]
    assert all(np.isfinite(v) for v in history.val_losses)


def test_mission_cache_and_streaming_source_share_interface():
    """MissionCache must expose mission_ids too, matching
    StreamingMissionSource's interface -- the property train() now
    depends on for both data sources."""
    from simulator.training.train import MissionArtifacts, MissionCache

    fake = MissionArtifacts(
        mission_id="fake", array=np.zeros((1, 1, 11)), node_ids=["USV_0"],
        node_types={}, timestamps=np.array([0.0]), events=[], evaluator=None,
        candidate_range_factor=1.5,
    )
    cache = MissionCache(missions=[fake])
    assert cache.mission_ids == ["fake"]

"""
tests/test_phase1_smoke.py

Minimal pytest suite exercising the Phase 1 pipeline end to end. This is a
starting point, not the full unit-test suite recommended in the Phase 1
delivery notes (per-module test files covering every validation branch) —
those should be added as separate files (test_mission_config.py,
test_random_manager.py, etc.) following the same fixture pattern.

Run with:
    pytest -v
from the repository root, after an editable install (`pip install -e .`).
"""

from __future__ import annotations

import pytest

from simulator.mission_config import ConfigValidationError, default_config
from simulator.mission_generator import MissionGenerator
from simulator.random_manager import RandomManager
from simulator.trajectory_storage import MissionAlreadyExistsError, MissionDataStore, TrajectoryWriter
from simulator.validation_utils import validate_mission_state, validate_storage_budget


@pytest.fixture
def config():
    return default_config(master_seed=7, split="train")


@pytest.fixture
def generator(config):
    return MissionGenerator(config, RandomManager(master_seed=7))


def test_config_validates(config):
    config.validate()  # should not raise


def test_invalid_config_raises():
    from simulator.mission_config import MissionConfig, WaveConfig

    bad = MissionConfig(wave=WaveConfig(n_modes=0))
    with pytest.raises(ConfigValidationError):
        bad.validate()


def test_mission_generation_is_reproducible(config):
    state_a = MissionGenerator(config, RandomManager(master_seed=7)).generate("repro_check")
    state_b = MissionGenerator(config, RandomManager(master_seed=7)).generate("repro_check")
    da, db = state_a.to_dict(), state_b.to_dict()
    da["metadata"].pop("created_at")
    db["metadata"].pop("created_at")
    assert da == db


def test_mission_generation_differs_by_mission_id(generator):
    state_a = generator.generate("mission_a")
    state_b = generator.generate("mission_b")
    assert state_a.nodes[0].x_m != state_b.nodes[0].x_m


def test_generated_mission_passes_validation(generator, config):
    state = generator.generate("mission_valid")
    validate_mission_state(state, config)  # should not raise


def test_write_and_reload_roundtrip(tmp_path, generator, config):
    writer = TrajectoryWriter(tmp_path)
    state = generator.generate("mission_roundtrip")
    validate_mission_state(state, config)
    writer.write_mission(state, events=None)

    store = MissionDataStore(tmp_path)
    assert len(store) == 1

    metadata, array, node_ids, events = store.load_mission("mission_roundtrip")
    assert metadata.mission_id == "mission_roundtrip"
    assert array.shape[1] == len(node_ids) == state.node_count
    assert len(events) == 0


def test_duplicate_write_is_rejected(tmp_path, generator):
    writer = TrajectoryWriter(tmp_path)
    state = generator.generate("mission_dup")
    writer.write_mission(state, events=None)
    with pytest.raises(MissionAlreadyExistsError):
        writer.write_mission(state, events=None)


def test_storage_budget_passes_for_small_corpus(tmp_path, generator, config):
    writer = TrajectoryWriter(tmp_path)
    for i in range(5):
        state = generator.generate(f"mission_budget_{i}")
        validate_mission_state(state, config)
        writer.write_mission(state, events=None)

    result = validate_storage_budget(tmp_path, preferred_limit_gb=30.0, hard_limit_gb=50.0)
    assert result.status.value == "pass"


def test_storage_budget_fails_when_limit_is_tiny(tmp_path, generator):
    writer = TrajectoryWriter(tmp_path)
    state = generator.generate("mission_tiny_budget")
    writer.write_mission(state, events=None)

    # force a Fail verdict with an artificially tiny hard limit
    result = validate_storage_budget(tmp_path, preferred_limit_gb=1e-9, hard_limit_gb=1e-9)
    assert result.status.value == "fail"


def test_write_mission_with_full_array_persists_all_timesteps(tmp_path, generator, config):
    """Regression test: write_mission() used to unconditionally write only
    the length-1 initial-state trajectory, silently discarding any full
    propagated array passed to it (there was no way to pass one at all).
    This directly checks the fix — a full multi-timestep array must
    actually be persisted, not truncated to T=1."""
    from simulator.trajectory_generator import TrajectoryGenerator

    writer = TrajectoryWriter(tmp_path)
    state = generator.generate("mission_full_array")
    array, node_ids = TrajectoryGenerator(config, RandomManager(master_seed=7)).propagate(state)
    assert array.shape[0] > 1  # sanity: the propagated trajectory is genuinely multi-timestep

    writer.write_mission(state, events=None, array=array, node_ids=node_ids)

    store = MissionDataStore(tmp_path)
    _, loaded_array, loaded_node_ids, _ = store.load_mission("mission_full_array")
    assert loaded_array.shape == array.shape
    assert loaded_array.shape[0] == array.shape[0] > 1
    assert loaded_node_ids == node_ids


def test_write_mission_without_array_falls_back_to_initial_state(tmp_path, generator):
    """Phase 1 callers (no propagated trajectory available) must still get
    the old length-1 behavior when array/node_ids are omitted."""
    writer = TrajectoryWriter(tmp_path)
    state = generator.generate("mission_no_array")
    writer.write_mission(state, events=None)

    store = MissionDataStore(tmp_path)
    _, loaded_array, _, _ = store.load_mission("mission_no_array")
    assert loaded_array.shape[0] == 1

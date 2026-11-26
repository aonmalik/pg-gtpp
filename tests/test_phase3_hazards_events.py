"""
tests/test_phase3_hazards_events.py

Pytest suite for the Phase 3 hazard and event generation pipeline:
trajectory propagation, graph building, cause-specific hazards, and the
Ogata-thinning event generator.

All tests use a deliberately small/short mission configuration (2 AUVs, 3
IoUT, 60s duration) to keep runtime low — the O(brackets x edges x probes)
cost of the reference thinning implementation is a documented, known
limitation (see event_generator.py's module docstring and the Phase 3
delivery notes), not something these tests attempt to benchmark at scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.event_types import CauseType, EventType
from simulator.event_generator import EventGenerator
from simulator.graph_builder import build_candidate_edges, theoretical_max_range_m
from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.mission_generator import MissionGenerator
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.physics_intensity import CauseHazards, HazardEvaluator
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator
from simulator.trajectory_storage import MissionDataStore, TrajectoryWriter
from simulator.validation_utils import validate_event_log_schema, validate_node_trajectory_array


@pytest.fixture
def fast_config():
    base = default_config(master_seed=21)
    small_fleet = FleetConfig(
        usv_count=1, n_auv_min=2, n_auv_max=2, m_iout_min=3, m_iout_max=3,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    short_envelope = MissionEnvelopeConfig(
        duration_s=60.0, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m, depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    config = MissionConfig(
        master_seed=21, split="train", wave=base.wave, current=base.current,
        acoustic=base.acoustic, fleet=small_fleet, envelope=short_envelope,
        hazard=base.hazard, graph=base.graph,
    )
    config.validate()
    return config


@pytest.fixture
def pipeline(fast_config):
    """Runs the full generate -> propagate -> events pipeline once and
    returns everything downstream tests need."""
    rm = RandomManager(master_seed=21)
    mission_id = "mission_fast_test"
    state = MissionGenerator(fast_config, rm).generate(mission_id)
    array, node_ids = TrajectoryGenerator(fast_config, rm).propagate(state)
    wf = WaveField.sample(fast_config.wave, rm.get_numpy_rng(mission_id, "wave"))
    cf = CurrentField.sample(fast_config.current, rm.get_numpy_rng(mission_id, "current"))
    ch = AcousticChannel(fast_config.acoustic)
    node_types = {n.node_id: n.node_type for n in state.nodes}
    timestamps = np.arange(array.shape[0], dtype=np.float64) / fast_config.envelope.node_sample_rate_hz
    events = EventGenerator(fast_config, rm, ch, wf, cf).generate(
        mission_id, array, node_ids, node_types, timestamps
    )
    return {
        "config": fast_config, "state": state, "array": array, "node_ids": node_ids,
        "wave_field": wf, "current_field": cf, "channel": ch, "events": events,
        "mission_id": mission_id,
    }


# ---------------------------------------------------------------------- #
# Trajectory generator
# ---------------------------------------------------------------------- #
def test_trajectory_array_passes_validation(pipeline):
    from simulator.trajectory_storage import F_NODE

    validate_node_trajectory_array(
        pipeline["array"], expected_feature_dim=F_NODE, expected_node_count=len(pipeline["node_ids"])
    )


def test_trajectory_is_reproducible(fast_config):
    rm_a = RandomManager(master_seed=21)
    rm_b = RandomManager(master_seed=21)
    state_a = MissionGenerator(fast_config, rm_a).generate("repro_traj")
    state_b = MissionGenerator(fast_config, rm_b).generate("repro_traj")
    array_a, _ = TrajectoryGenerator(fast_config, rm_a).propagate(state_a)
    array_b, _ = TrajectoryGenerator(fast_config, rm_b).propagate(state_b)
    assert np.allclose(array_a, array_b)


def test_iout_nodes_are_static(pipeline):
    array, node_ids = pipeline["array"], pipeline["node_ids"]
    for node_id in node_ids:
        if node_id.startswith("IOUT"):
            idx = node_ids.index(node_id)
            assert np.allclose(array[0, idx, :2], array[-1, idx, :2])


def test_auv_battery_drains_monotonically_nonincreasing(pipeline):
    array, node_ids = pipeline["array"], pipeline["node_ids"]
    battery_col = 6
    for node_id in node_ids:
        if node_id.startswith("AUV"):
            idx = node_ids.index(node_id)
            battery_series = array[:, idx, battery_col]
            assert np.all(np.diff(battery_series) <= 1e-6)


# ---------------------------------------------------------------------- #
# Graph builder
# ---------------------------------------------------------------------- #
def test_theoretical_max_range_positive(pipeline):
    assert theoretical_max_range_m(pipeline["channel"]) > 0.0


def test_candidate_edges_only_usv_auv_and_auv_iout(pipeline):
    edges = build_candidate_edges(
        pipeline["state"].nodes, pipeline["channel"], pipeline["config"].graph.candidate_range_factor
    )
    kinds = {e.edge_kind for e in edges}
    assert kinds <= {"usv_auv", "auv_iout"}


def test_candidate_range_factor_must_exceed_one(pipeline):
    with pytest.raises(ValueError):
        build_candidate_edges(pipeline["state"].nodes, pipeline["channel"], candidate_range_factor=1.0)


# ---------------------------------------------------------------------- #
# Hazard evaluator
# ---------------------------------------------------------------------- #
def test_geometric_hazard_higher_out_of_range(pipeline):
    evaluator = HazardEvaluator(
        pipeline["channel"], pipeline["wave_field"], pipeline["current_field"], pipeline["config"].hazard
    )
    near = evaluator.cause_hazards(100.0, 0, 0, 0, 100, 0, 50, 0, 0, 0, 0, 10.0)
    far = evaluator.cause_hazards(200000.0, 0, 0, 0, 200000, 0, 50, 0, 0, 0, 0, 10.0)
    assert far.geo >= near.geo


def test_recovery_hazard_higher_in_range(pipeline):
    evaluator = HazardEvaluator(
        pipeline["channel"], pipeline["wave_field"], pipeline["current_field"], pipeline["config"].hazard
    )
    assert evaluator.recovery_hazard(100.0) >= evaluator.recovery_hazard(200000.0)


def test_cause_hazards_rejects_negative():
    with pytest.raises(ValueError):
        CauseHazards(geo=-1.0, wave=0.0, doppler=0.0, current=0.0).validate()


def test_dominant_cause_matches_max():
    hazards = CauseHazards(geo=0.1, wave=0.9, doppler=0.2, current=0.05)
    assert hazards.dominant_cause() == CauseType.WAVE


# ---------------------------------------------------------------------- #
# Event generator
# ---------------------------------------------------------------------- #
def test_events_pass_schema_validation(pipeline):
    records = [e.to_record() for e in pipeline["events"]]
    validate_event_log_schema(records)  # should not raise


def test_edge_states_alternate_correctly(pipeline):
    from collections import defaultdict

    per_edge = defaultdict(list)
    for e in pipeline["events"]:
        per_edge[e.edge_key].append(e)
    for key, evs in per_edge.items():
        for i in range(1, len(evs)):
            assert evs[i].event_type != evs[i - 1].event_type, f"edge {key} has consecutive same-type events"


def test_events_are_time_sorted(pipeline):
    timestamps = [e.timestamp_s for e in pipeline["events"]]
    assert timestamps == sorted(timestamps)


def test_link_down_events_always_have_cause(pipeline):
    for e in pipeline["events"]:
        if e.event_type == EventType.LINK_DOWN:
            assert e.cause is not None


def test_link_up_events_have_no_cause(pipeline):
    for e in pipeline["events"]:
        if e.event_type == EventType.LINK_UP:
            assert e.cause is None


def test_marks_are_within_valid_bounds(pipeline):
    for e in pipeline["events"]:
        assert 0.0 <= e.marks.pdr <= 1.0
        assert e.marks.throughput_bps >= 0.0
        assert e.marks.delay_s >= 0.0


def test_event_generation_is_reproducible(fast_config):
    rm_a = RandomManager(master_seed=21)
    rm_b = RandomManager(master_seed=21)
    mission_id = "repro_events"

    def _run(rm):
        state = MissionGenerator(fast_config, rm).generate(mission_id)
        array, node_ids = TrajectoryGenerator(fast_config, rm).propagate(state)
        wf = WaveField.sample(fast_config.wave, rm.get_numpy_rng(mission_id, "wave"))
        cf = CurrentField.sample(fast_config.current, rm.get_numpy_rng(mission_id, "current"))
        ch = AcousticChannel(fast_config.acoustic)
        node_types = {n.node_id: n.node_type for n in state.nodes}
        timestamps = np.arange(array.shape[0], dtype=np.float64) / fast_config.envelope.node_sample_rate_hz
        return EventGenerator(fast_config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)

    events_a = _run(rm_a)
    events_b = _run(rm_b)
    # event_id is an arbitrary unique key (uuid4, real OS randomness) by
    # design (see event_types.py) — it is not part of the reproducibility
    # contract. Reproducibility applies to event *content*: timing, cause,
    # and marks.
    records_a = [{k: v for k, v in e.to_record().items() if k != "event_id"} for e in events_a]
    records_b = [{k: v for k, v in e.to_record().items() if k != "event_id"} for e in events_b]
    assert records_a == records_b


# ---------------------------------------------------------------------- #
# Full storage round-trip with real (non-empty) events
# ---------------------------------------------------------------------- #
def test_full_pipeline_storage_roundtrip(tmp_path, pipeline):
    writer = TrajectoryWriter(tmp_path)
    writer.write_node_trajectory(pipeline["mission_id"], pipeline["node_ids"], pipeline["array"])
    writer.write_event_log(pipeline["mission_id"], pipeline["events"])
    writer.append_mission_index(pipeline["state"].metadata)

    store = MissionDataStore(tmp_path)
    metadata, array2, node_ids2, events_df = store.load_mission(pipeline["mission_id"])

    assert array2.shape == pipeline["array"].shape
    assert node_ids2 == pipeline["node_ids"]
    assert len(events_df) == len(pipeline["events"])

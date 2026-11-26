"""
tests/test_phase4_qa.py

Pytest suite for the Phase 4 QA/verification framework: the time-rescaling
theorem test, the identifiability audit, the shifted-window leakage test,
and the combined certification report.

Uses the same small/fast mission configuration pattern as Phase 3's test
suite to keep runtime low.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simulator.event_generator import EventGenerator
from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.mission_generator import MissionGenerator
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.physics_intensity import HazardEvaluator
from simulator.qa.certification_report import CertificationVerdict, certify_mission
from simulator.qa.identifiability_audit import audit_identifiability
from simulator.qa.leakage_tests import shifted_window_regeneration_test
from simulator.qa.time_rescaling_test import run_time_rescaling_test
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator


@pytest.fixture
def fast_config():
    base = default_config(master_seed=21)
    small_fleet = FleetConfig(
        usv_count=1, n_auv_min=2, n_auv_max=2, m_iout_min=3, m_iout_max=3,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    short_envelope = MissionEnvelopeConfig(
        duration_s=300.0, node_sample_rate_hz=1.0,  # longer than Phase 3's 60s to get more events for the KS test
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
    rm = RandomManager(master_seed=21)
    mission_id = "mission_qa_fixture"
    state = MissionGenerator(fast_config, rm).generate(mission_id)
    array, node_ids = TrajectoryGenerator(fast_config, rm).propagate(state)
    wf = WaveField.sample(fast_config.wave, rm.get_numpy_rng(mission_id, "wave"))
    cf = CurrentField.sample(fast_config.current, rm.get_numpy_rng(mission_id, "current"))
    ch = AcousticChannel(fast_config.acoustic)
    node_types = {n.node_id: n.node_type for n in state.nodes}
    timestamps = np.arange(array.shape[0], dtype=np.float64) / fast_config.envelope.node_sample_rate_hz
    events = EventGenerator(fast_config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)
    evaluator = HazardEvaluator(ch, wf, cf, fast_config.hazard)
    return {
        "config": fast_config, "rm": rm, "mission_id": mission_id, "array": array, "node_ids": node_ids,
        "node_types": node_types, "timestamps": timestamps, "wave_field": wf, "current_field": cf,
        "channel": ch, "events": events, "evaluator": evaluator,
    }


# ---------------------------------------------------------------------- #
# Time-rescaling test
# ---------------------------------------------------------------------- #
def test_time_rescaling_runs_and_returns_valid_result(pipeline):
    result = run_time_rescaling_test(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], pipeline["events"]
    )
    assert result.n_edges_tested >= 0
    if result.n_edges_tested > 0:
        assert 0.0 <= result.binomial_test_p_value <= 1.0
        assert np.all((result.per_edge_p_values >= 0.0) & (result.per_edge_p_values <= 1.0))
    else:
        assert math.isnan(result.binomial_test_p_value)


def test_time_rescaling_uniform_sample_in_unit_interval(pipeline):
    result = run_time_rescaling_test(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], pipeline["events"]
    )
    if len(result.rescaled_uniform_sample) > 0:
        assert np.all(result.rescaled_uniform_sample >= 0.0)
        assert np.all(result.rescaled_uniform_sample <= 1.0)


def test_time_rescaling_empty_events_handled_gracefully(pipeline):
    result = run_time_rescaling_test(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], []
    )
    assert result.n_edges_tested == 0
    assert math.isnan(result.binomial_test_p_value)


# ---------------------------------------------------------------------- #
# Identifiability audit
# ---------------------------------------------------------------------- #
def test_identifiability_audit_correlation_matrix_shape(pipeline):
    result = audit_identifiability(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], pipeline["events"]
    )
    assert result.correlation_matrix.shape == (4, 4)
    assert result.cause_labels == ("geo", "wave", "doppler", "current")


def test_identifiability_ambiguous_fraction_in_valid_range(pipeline):
    result = audit_identifiability(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], pipeline["events"]
    )
    if result.n_events > 0:
        assert 0.0 <= result.ambiguous_fraction <= 1.0


def test_identifiability_no_events_handled_gracefully(pipeline):
    result = audit_identifiability(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], []
    )
    assert result.n_events == 0
    assert math.isnan(result.ambiguous_fraction)


def test_identifiability_degenerate_cause_has_nan_correlations(pipeline):
    result = audit_identifiability(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], pipeline["events"]
    )
    for cause in result.degenerate_causes:
        idx = result.cause_labels.index(cause)
        row = result.correlation_matrix[idx]
        off_diag = np.delete(row, idx)
        assert np.all(np.isnan(off_diag))


def test_flags_identifiability_risk_respects_threshold():
    from simulator.qa.identifiability_audit import IdentifiabilityAuditResult

    result = IdentifiabilityAuditResult(
        n_events=10, ambiguous_fraction=0.1, ambiguity_threshold=1.5,
        correlation_matrix=np.eye(4), cause_labels=("geo", "wave", "doppler", "current"),
        max_off_diagonal_correlation=0.8, degenerate_causes=(),
    )
    assert result.flags_identifiability_risk(correlation_threshold=0.7)
    assert not result.flags_identifiability_risk(correlation_threshold=0.9)


def test_default_config_geo_cause_is_not_degenerate():
    """Regression test for the bandwidth-conversion bug (see
    test_phase2_physics.py): under the old, unrealistic ~90 dB SNR margins,
    'geo' never contributed to any failure and was flagged degenerate by
    every certification run in Phase 4. After the acoustic physics fix and
    recalibration, it should be an active, non-degenerate cause under
    default config on a full-length mission."""
    from simulator.mission_config import default_config
    from simulator.mission_generator import MissionGenerator
    from simulator.random_manager import RandomManager
    from simulator.trajectory_generator import TrajectoryGenerator
    from simulator.event_generator import EventGenerator
    from simulator.physics import AcousticChannel, CurrentField, WaveField
    from simulator.physics_intensity import HazardEvaluator

    config = default_config(master_seed=11)
    rm = RandomManager(master_seed=11)
    mission_id = "geo_regression_check"
    state = MissionGenerator(config, rm).generate(mission_id)
    array, node_ids = TrajectoryGenerator(config, rm).propagate(state)
    wf = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
    cf = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
    ch = AcousticChannel(config.acoustic)
    node_types = {n.node_id: n.node_type for n in state.nodes}
    timestamps = np.arange(array.shape[0], dtype=np.float64) / config.envelope.node_sample_rate_hz
    events = EventGenerator(config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)
    evaluator = HazardEvaluator(ch, wf, cf, config.hazard)

    result = audit_identifiability(evaluator, array, node_ids, timestamps, events)
    assert "geo" not in result.degenerate_causes
    down_causes = {e.cause.value for e in events if e.cause is not None}
    assert "geo" in down_causes


# ---------------------------------------------------------------------- #
# Leakage test — the most important assertion in this suite
# ---------------------------------------------------------------------- #
def test_leakage_test_passes_on_real_implementation(pipeline):
    """This is a regression test: if a future change to event_generator.py
    ever introduces a lookahead dependency, this test should start failing.
    A passing result here is the whole point of Phase 4's leakage-testing
    work — it is not a tautology, since the test is only guaranteed to pass
    if generation genuinely never reads beyond its current bracket (see
    leakage_tests.py's module docstring for the reproducibility argument)."""
    n_timestamps = len(pipeline["timestamps"])
    truncate_idx = n_timestamps // 2
    result = shifted_window_regeneration_test(
        pipeline["config"], pipeline["rm"], pipeline["mission_id"], pipeline["channel"],
        pipeline["wave_field"], pipeline["current_field"], pipeline["array"], pipeline["node_ids"],
        pipeline["node_types"], pipeline["timestamps"], truncate_idx,
    )
    assert result.passed, f"Leakage detected: {result.first_mismatch}"
    assert result.n_events_in_window == result.n_truncated_events


def test_leakage_test_rejects_invalid_truncate_idx(pipeline):
    n_timestamps = len(pipeline["timestamps"])
    with pytest.raises(ValueError):
        shifted_window_regeneration_test(
            pipeline["config"], pipeline["rm"], pipeline["mission_id"], pipeline["channel"],
            pipeline["wave_field"], pipeline["current_field"], pipeline["array"], pipeline["node_ids"],
            pipeline["node_types"], pipeline["timestamps"], n_timestamps - 1,  # too close to the end
        )


# ---------------------------------------------------------------------- #
# Certification report
# ---------------------------------------------------------------------- #
def test_certification_report_produces_valid_verdict(pipeline):
    tr = run_time_rescaling_test(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], pipeline["events"]
    )
    idn = audit_identifiability(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], pipeline["events"]
    )
    truncate_idx = len(pipeline["timestamps"]) // 2
    lk = shifted_window_regeneration_test(
        pipeline["config"], pipeline["rm"], pipeline["mission_id"], pipeline["channel"],
        pipeline["wave_field"], pipeline["current_field"], pipeline["array"], pipeline["node_ids"],
        pipeline["node_types"], pipeline["timestamps"], truncate_idx,
    )
    report = certify_mission(tr, idn, lk)
    assert report.verdict in (CertificationVerdict.PASS, CertificationVerdict.WARNING, CertificationVerdict.FAIL)
    assert len(report.reasons) >= 1


def test_certification_report_fails_when_leakage_fails():
    from simulator.qa.identifiability_audit import IdentifiabilityAuditResult
    from simulator.qa.leakage_tests import LeakageTestResult
    from simulator.qa.time_rescaling_test import TimeRescalingTestResult

    tr = TimeRescalingTestResult(
        per_edge_p_values=np.array([0.6, 0.7, 0.5]), n_edges_tested=3, n_edges_rejecting=0,
        rejection_significance_level=0.05, fraction_rejecting=0.0, binomial_test_p_value=0.9,
        pooled_ks_statistic=0.01, pooled_ks_p_value=0.9, n_pooled_intervals=100,
        rescaled_uniform_sample=np.array([0.5]),
    )
    idn = IdentifiabilityAuditResult(
        n_events=10, ambiguous_fraction=0.1, ambiguity_threshold=1.5,
        correlation_matrix=np.eye(4), cause_labels=("geo", "wave", "doppler", "current"),
        max_off_diagonal_correlation=0.1, degenerate_causes=(),
    )
    lk = LeakageTestResult(passed=False, truncate_at_s=100.0, n_events_in_window=5, n_truncated_events=4)

    report = certify_mission(tr, idn, lk)
    assert report.verdict == CertificationVerdict.FAIL


def test_certification_report_to_dict_is_json_serializable(pipeline):
    import json

    tr = run_time_rescaling_test(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], pipeline["events"]
    )
    idn = audit_identifiability(
        pipeline["evaluator"], pipeline["array"], pipeline["node_ids"], pipeline["timestamps"], pipeline["events"]
    )
    truncate_idx = len(pipeline["timestamps"]) // 2
    lk = shifted_window_regeneration_test(
        pipeline["config"], pipeline["rm"], pipeline["mission_id"], pipeline["channel"],
        pipeline["wave_field"], pipeline["current_field"], pipeline["array"], pipeline["node_ids"],
        pipeline["node_types"], pipeline["timestamps"], truncate_idx,
    )
    report = certify_mission(tr, idn, lk)
    json.dumps(report.to_dict())  # should not raise

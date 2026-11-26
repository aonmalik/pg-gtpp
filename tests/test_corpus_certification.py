"""
tests/test_corpus_certification.py

Pytest suite for corpus-level QA aggregation (``qa/corpus_certification.py``).

Covers both the pure aggregation logic (via synthetic reports, fast) and a
real, small-scale end-to-end batch (via the same small/fast mission config
pattern used in Phase 3/4's test suites, to keep runtime low while still
exercising the actual generation + certification pipeline, not just mocks).
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.event_generator import EventGenerator
from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.mission_generator import MissionGenerator
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.physics_intensity import HazardEvaluator
from simulator.qa.certification_report import CertificationReport, CertificationVerdict, certify_mission
from simulator.qa.corpus_certification import certify_corpus
from simulator.qa.identifiability_audit import IdentifiabilityAuditResult, audit_identifiability
from simulator.qa.leakage_tests import LeakageTestResult, shifted_window_regeneration_test
from simulator.qa.time_rescaling_test import TimeRescalingTestResult, run_time_rescaling_test
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator


def _synthetic_report(passed_leakage: bool = True, ambiguous_fraction: float = 0.1, correlation: float = 0.1) -> CertificationReport:
    tr = TimeRescalingTestResult(
        per_edge_p_values=np.array([0.5]), n_edges_tested=1, n_edges_rejecting=0,
        rejection_significance_level=0.05, fraction_rejecting=0.0, binomial_test_p_value=1.0,
        pooled_ks_statistic=0.01, pooled_ks_p_value=0.9, n_pooled_intervals=10,
        rescaled_uniform_sample=np.array([0.5]),
    )
    idn = IdentifiabilityAuditResult(
        n_events=10, ambiguous_fraction=ambiguous_fraction, ambiguity_threshold=1.5,
        correlation_matrix=np.eye(4), cause_labels=("geo", "wave", "doppler", "current"),
        max_off_diagonal_correlation=correlation, degenerate_causes=(),
    )
    lk = LeakageTestResult(
        passed=passed_leakage, truncate_at_s=100.0, n_events_in_window=5,
        n_truncated_events=5 if passed_leakage else 4,
    )
    return certify_mission(tr, idn, lk)


# ---------------------------------------------------------------------- #
# Pure aggregation logic (synthetic reports — fast)
# ---------------------------------------------------------------------- #
def test_corpus_requires_at_least_one_report():
    with pytest.raises(ValueError):
        certify_corpus([], {})


def test_corpus_all_pass_yields_pass():
    reports = [(f"m{i}", _synthetic_report()) for i in range(10)]
    counts = {mid: 100 for mid, _ in reports}
    result = certify_corpus(reports, counts)
    assert result.verdict == CertificationVerdict.PASS
    assert result.n_pass == 10 and result.n_warning == 0 and result.n_fail == 0
    assert result.pass_rate() == 1.0


def test_corpus_any_fail_propagates_to_corpus_fail():
    reports = [(f"m{i}", _synthetic_report()) for i in range(9)]
    reports.append(("m_bad", _synthetic_report(passed_leakage=False)))
    counts = {mid: 100 for mid, _ in reports}
    result = certify_corpus(reports, counts)
    assert result.verdict == CertificationVerdict.FAIL
    assert "m_bad" in result.failing_mission_ids


def test_corpus_warning_rate_below_tolerance_stays_pass():
    # 1/10 = 10%, below the default 20% tolerance
    reports = [(f"m{i}", _synthetic_report()) for i in range(9)]
    reports.append(("m_warn", _synthetic_report(correlation=0.8)))
    counts = {mid: 100 for mid, _ in reports}
    result = certify_corpus(reports, counts, warning_rate_tolerance=0.2)
    assert result.verdict == CertificationVerdict.PASS


def test_corpus_warning_rate_above_tolerance_becomes_warning():
    # 5/10 = 50%, above the default 20% tolerance
    reports = [(f"w{i}", _synthetic_report(correlation=0.8)) for i in range(5)]
    reports += [(f"p{i}", _synthetic_report()) for i in range(5)]
    counts = {mid: 100 for mid, _ in reports}
    result = certify_corpus(reports, counts, warning_rate_tolerance=0.2)
    assert result.verdict == CertificationVerdict.WARNING


def test_corpus_reports_worst_mission_identity():
    reports = [(f"m{i}", _synthetic_report(ambiguous_fraction=0.05)) for i in range(5)]
    reports.append(("m_worst", _synthetic_report(ambiguous_fraction=0.9)))
    counts = {mid: 100 for mid, _ in reports}
    result = certify_corpus(reports, counts)
    assert result.worst_ambiguous_mission_id == "m_worst"
    assert result.max_ambiguous_fraction == 0.9


def test_corpus_summary_and_pass_rate_consistent():
    reports = [(f"m{i}", _synthetic_report()) for i in range(4)]
    counts = {mid: 50 for mid, _ in reports}
    result = certify_corpus(reports, counts)
    assert "4 total" in result.summary()
    assert result.total_events == 200


# ---------------------------------------------------------------------- #
# Real, small-scale end-to-end batch (exercises the actual pipeline)
# ---------------------------------------------------------------------- #
@pytest.fixture
def small_batch_config():
    base = default_config(master_seed=99)
    small_fleet = FleetConfig(
        usv_count=1, n_auv_min=3, n_auv_max=5, m_iout_min=8, m_iout_max=15,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    short_envelope = MissionEnvelopeConfig(
        duration_s=400.0, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m, depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    config = MissionConfig(
        master_seed=99, split="train", wave=base.wave, current=base.current,
        acoustic=base.acoustic, fleet=small_fleet, envelope=short_envelope,
        hazard=base.hazard, graph=base.graph,
    )
    config.validate()
    return config


def test_real_small_batch_achieves_reasonable_pass_rate(small_batch_config):
    """Regression test: a small, real, freshly generated batch under the
    current (recalibrated) default config should achieve a high corpus
    pass rate. This is the test that should start failing if a future
    change to the acoustic/hazard calibration regresses the fix made after
    Phase 4 (degenerate 'geo' cause, catastrophic ambiguous-event
    fractions, etc.) — it exercises the real generation and certification
    pipeline end to end, not synthetic/mocked reports.
    """
    config = small_batch_config
    rm = RandomManager(master_seed=99)
    n_missions = 6

    mission_reports = []
    event_counts: dict[str, int] = {}

    for i in range(n_missions):
        mission_id = f"corpus_test_{i:02d}"
        state = MissionGenerator(config, rm).generate(mission_id)
        array, node_ids = TrajectoryGenerator(config, rm).propagate(state)
        wf = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
        cf = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
        ch = AcousticChannel(config.acoustic)
        node_types = {n.node_id: n.node_type for n in state.nodes}
        timestamps = np.arange(array.shape[0], dtype=np.float64) / config.envelope.node_sample_rate_hz

        events = EventGenerator(config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)
        evaluator = HazardEvaluator(ch, wf, cf, config.hazard)
        time_rescaling = run_time_rescaling_test(evaluator, array, node_ids, timestamps, events)
        identifiability = audit_identifiability(evaluator, array, node_ids, timestamps, events)
        truncate_idx = len(timestamps) // 2
        leakage = shifted_window_regeneration_test(
            config, rm, mission_id, ch, wf, cf, array, node_ids, node_types, timestamps, truncate_idx
        )
        report = certify_mission(time_rescaling, identifiability, leakage)
        mission_reports.append((mission_id, report))
        event_counts[mission_id] = len(events)

    corpus_result = certify_corpus(mission_reports, event_counts)

    # The regression assertion: leakage must never fail (mechanical, should
    # always hold), and the corpus pass rate should be reasonably high.
    assert corpus_result.n_fail == 0, f"Unexpected FAIL in small batch: {corpus_result.failing_mission_ids}"
    assert corpus_result.pass_rate() >= 0.5, (
        f"Corpus pass rate {corpus_result.pass_rate():.1%} is suspiciously low — "
        f"possible calibration regression. Full summary:\n{corpus_result.summary()}"
    )

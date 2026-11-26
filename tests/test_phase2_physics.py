"""
tests/test_phase2_physics.py

Pytest suite for the closed-form physics foundation: wave field, current
field, and acoustic channel.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.mission_config import AcousticConfig, default_config
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.random_manager import RandomManager


@pytest.fixture
def config():
    return default_config(master_seed=1)


@pytest.fixture
def rm():
    return RandomManager(master_seed=1)


# ---------------------------------------------------------------------- #
# Wave field
# ---------------------------------------------------------------------- #
def test_wave_field_reproducible(config, rm):
    wf_a = WaveField.sample(config.wave, rm.get_numpy_rng("m1", "wave"))
    wf_b = WaveField.sample(config.wave, RandomManager(master_seed=1).get_numpy_rng("m1", "wave"))
    assert wf_a.to_dict() == wf_b.to_dict()


def test_wave_field_roundtrip(config, rm):
    wf = WaveField.sample(config.wave, rm.get_numpy_rng("m1", "wave"))
    wf2 = WaveField.from_dict(wf.to_dict())
    assert wf.to_dict() == wf2.to_dict()


def test_wave_height_rate_matches_finite_difference(config, rm):
    wf = WaveField.sample(config.wave, rm.get_numpy_rng("m1", "wave"))
    eps = 1e-4
    x, y, t = 250.0, 400.0, 33.0
    fd = (wf.height(x, y, t + eps) - wf.height(x, y, t - eps)) / (2 * eps)
    analytic = wf.height_rate(x, y, t)
    assert abs(fd - analytic) / max(abs(analytic), 1e-9) < 1e-6


def test_wave_field_vectorized_shape(config, rm):
    wf = WaveField.sample(config.wave, rm.get_numpy_rng("m1", "wave"))
    xs = np.linspace(0, 1000, 10)
    ts = np.linspace(0, 50, 10)
    h = wf.height(xs, xs, ts)
    assert h.shape == (10,)


# ---------------------------------------------------------------------- #
# Current field
# ---------------------------------------------------------------------- #
def test_current_field_reproducible(config, rm):
    cf_a = CurrentField.sample(config.current, rm.get_numpy_rng("m1", "current"))
    cf_b = CurrentField.sample(
        config.current, RandomManager(master_seed=1).get_numpy_rng("m1", "current")
    )
    assert cf_a.to_dict() == cf_b.to_dict()


def test_current_field_boundary_clamp_is_finite(config, rm):
    cf = CurrentField.sample(config.current, rm.get_numpy_rng("m1", "current"))
    u_lo, v_lo = cf.velocity(depth_m=-1000.0, t_s=0.0)
    u_hi, v_hi = cf.velocity(depth_m=1e6, t_s=0.0)
    assert np.isfinite(u_lo) and np.isfinite(v_lo)
    assert np.isfinite(u_hi) and np.isfinite(v_hi)


def test_current_integrated_drift_matches_numeric_integration(config, rm):
    cf = CurrentField.sample(config.current, rm.get_numpy_rng("m1", "current"))
    du, dv = cf.integrated_drift(depth_m=75.0, t0_s=0.0, t1_s=600.0)
    ts_fine = np.linspace(0, 600, 20001)
    u_fine, v_fine = cf.velocity(depth_m=75.0, t_s=ts_fine)
    du_num = np.trapezoid(u_fine, ts_fine)
    dv_num = np.trapezoid(v_fine, ts_fine)
    assert abs(du_num - du) / max(abs(du), 1e-9) < 1e-6
    assert abs(dv_num - dv) / max(abs(dv), 1e-9) < 1e-6


# ---------------------------------------------------------------------- #
# Acoustic channel
# ---------------------------------------------------------------------- #
def test_snr_decreases_monotonically_with_distance(config):
    channel = AcousticChannel(config.acoustic)
    distances = np.array([10.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0])
    snrs = channel.snr_db(distances)
    assert np.all(np.diff(snrs) < 0)


def test_invalid_acoustic_config_rejected():
    with pytest.raises(Exception):
        AcousticChannel(AcousticConfig(source_level_db=-5))


def test_throughput_positive_and_decreasing(config):
    channel = AcousticChannel(config.acoustic)
    t_near = float(channel.throughput_bps(10.0))
    t_far = float(channel.throughput_bps(2000.0))
    assert t_near > 0 and t_far > 0
    assert t_far < t_near


def test_noise_combination_is_power_domain(config):
    # Sanity check: combining two equal noise sources should raise the
    # total by ~3 dB (power doubling), not by summing dB values directly.
    from simulator.physics.acoustic_channel import _combine_noise_db

    combined = _combine_noise_db(50.0, 50.0)
    assert abs(combined - 53.01) < 0.05


# ---------------------------------------------------------------------- #
# Bandwidth correction regression tests (post-Phase-4 recalibration)
# ---------------------------------------------------------------------- #
def test_total_in_band_noise_includes_bandwidth_term(config):
    channel = AcousticChannel(config.acoustic)
    spectral = channel.ambient_noise_level_db()
    total = channel.total_in_band_noise_db()
    expected_delta = 10.0 * np.log10(config.acoustic.bandwidth_hz)
    assert abs((total - spectral) - expected_delta) < 1e-9


def test_default_config_gives_realistic_link_range(config):
    """Regression test for the bandwidth-conversion bug: under the old,
    dimensionally-inconsistent snr_db, a link at 2000 m had ~90 dB of
    margin (never fails). This checks the corrected physics gives a
    realistic viable/non-viable range instead — short links viable, long
    links (comparable to the mission area's diagonal) not."""
    channel = AcousticChannel(config.acoustic)
    assert bool(channel.is_link_viable(50.0))
    assert not bool(channel.is_link_viable(2000.0))


def test_theoretical_max_range_is_realistic(config):
    """Regression test: theoretical range should be on the order of
    hundreds of meters (a real acoustic modem link budget), not the
    ~100,000 km the pre-fix dimensional bug produced."""
    from simulator.graph_builder import theoretical_max_range_m

    channel = AcousticChannel(config.acoustic)
    max_range = theoretical_max_range_m(channel)
    assert 50.0 < max_range < 5000.0


def test_bandwidth_hz_validation_rejects_nonpositive():
    with pytest.raises(Exception):
        AcousticConfig(bandwidth_hz=0.0).validate()

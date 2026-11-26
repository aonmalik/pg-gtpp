import math
import numpy as np

from simulator.model.gbt_residual import continuous_time_competing_risk_nll


def test_nll_observed_competing_risk():
    physics = np.array([[0.2, 0.3, 0.1, 0.4]], dtype=float)
    residual = np.zeros_like(physics)
    # Event at t=2 from cause 1.
    nll = continuous_time_competing_risk_nll(
        physics, residual, 0.0,
        event_time=np.array([2.0]),
        event_cause=np.array([1]),
        at_risk=np.array([True]),
    )
    expected = 2.0 * 1.0 - math.log(0.3)
    assert math.isclose(nll, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_nll_censored_uses_full_exposure():
    physics = np.array([[0.2, 0.3, 0.1, 0.4]], dtype=float)
    residual = np.zeros_like(physics)
    nll = continuous_time_competing_risk_nll(
        physics, residual, 0.0,
        event_time=np.array([30.0]),
        event_cause=np.array([-1]),
        at_risk=np.array([True]),
    )
    assert math.isclose(nll, 30.0, rel_tol=1e-12, abs_tol=1e-12)


def test_scale_zero_is_exact_physics_likelihood():
    physics = np.array([[0.2, 0.3, 0.1, 0.4], [0.1, 0.1, 0.2, 0.1]])
    residual = np.array([[1.0, -1.0, 0.5, 2.0], [-2.0, 1.0, 1.0, -1.0]])
    kwargs = dict(
        event_time=np.array([2.0, 30.0]),
        event_cause=np.array([1, -1]),
        at_risk=np.array([True, True]),
    )
    a = continuous_time_competing_risk_nll(physics, residual, 0.0, **kwargs)
    b = continuous_time_competing_risk_nll(physics, np.zeros_like(residual), 0.0, **kwargs)
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

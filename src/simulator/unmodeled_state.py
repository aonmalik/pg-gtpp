"""Truth-only operational-state perturbations for robustness benchmarks.

The nominal HazardEvaluator remains unchanged.  These helpers construct a
persistent, partially observed operational stress from trajectory variables
that neural models can observe, then apply a controlled truth-only hazard
perturbation.
"""
from __future__ import annotations

import numpy as np


def operational_stress_components_batch(
    battery_i: np.ndarray, battery_j: np.ndarray,
    fim_i: np.ndarray, fim_j: np.ndarray,
    vx_i: np.ndarray, vy_i: np.ndarray, vx_j: np.ndarray, vy_j: np.ndarray,
    dx: np.ndarray, dy: np.ndarray,
) -> np.ndarray:
    """Return [battery, localization, alignment] stress components in [0,1]."""
    battery_stress = np.clip(1.0 - 0.5 * (np.asarray(battery_i) + np.asarray(battery_j)), 0.0, 1.0)
    fim_mean = 0.5 * (np.asarray(fim_i) + np.asarray(fim_j))
    fim_stress = np.clip(fim_mean / (fim_mean + 0.25 + 1e-8), 0.0, 1.0)

    los_norm = np.hypot(dx, dy)
    ux = np.divide(dx, los_norm, out=np.zeros_like(dx, dtype=float), where=los_norm > 1e-8)
    uy = np.divide(dy, los_norm, out=np.zeros_like(dy, dtype=float), where=los_norm > 1e-8)
    si = np.hypot(vx_i, vy_i)
    sj = np.hypot(vx_j, vy_j)
    ci = np.divide(vx_i * ux + vy_i * uy, si, out=np.zeros_like(si, dtype=float), where=si > 1e-8)
    cj = np.divide(vx_j * (-ux) + vy_j * (-uy), sj, out=np.zeros_like(sj, dtype=float), where=sj > 1e-8)
    moving = (si > 1e-8) | (sj > 1e-8)
    alignment = np.clip(0.5 * (2.0 - ci - cj), 0.0, 1.0) * moving.astype(float)
    return np.stack([battery_stress, fim_stress, alignment], axis=-1)


def operational_stress_batch(*args) -> np.ndarray:
    """Backward-compatible weighted scalar operational stress."""
    components = operational_stress_components_batch(*args)
    return np.clip(0.50 * components[..., 0] + 0.30 * components[..., 1] + 0.20 * components[..., 2], 0.0, 1.0)


# Fixed a-priori sensitivities: rows are causes, columns are
# battery/FIM/alignment.  Each row sums to one, so ``strength`` retains the
# same order of magnitude as the legacy scalar benchmark while changing the
# relative cause risks.  These are benchmark constants, not fitted values.
CAUSE_OPERATIONAL_SENSITIVITY = np.asarray((
    (0.60, 0.20, 0.20),  # geo
    (0.20, 0.30, 0.50),  # wave
    (0.15, 0.55, 0.30),  # doppler
    (0.35, 0.40, 0.25),  # current
), dtype=float)


def cause_specific_truth_multiplier(components: np.ndarray, strength: float) -> np.ndarray:
    """Return [E,4] truth multipliers with fixed cause-specific sensitivity."""
    if strength < 0.0:
        raise ValueError("strength must be >= 0")
    components = np.asarray(components, dtype=float)
    if components.ndim != 2 or components.shape[-1] != 3:
        raise ValueError(f"components must have shape [E,3], got {components.shape}")
    log_multiplier = components @ CAUSE_OPERATIONAL_SENSITIVITY.T
    return np.exp(float(strength) * log_multiplier)


def truth_hazard_multiplier(stress: np.ndarray, strength: float) -> np.ndarray:
    """Legacy scalar truth-only multiplicative hazard."""
    if strength < 0.0:
        raise ValueError("strength must be >= 0")
    return np.exp(float(strength) * np.asarray(stress, dtype=float))

"""
physics/wave_field.py

Closed-form, spectral wave-field model.

Design decisions
-----------------
1. The wave field is represented as a superposition of ``n_modes`` deep-water
   gravity-wave components (amplitude, frequency, phase, propagation
   direction), each drawn once per mission from a seeded RNG stream, rather
   than a spatial grid PDE solve. This is the storage-saving design decision
   documented in the implementation architecture: only the mode coefficients
   need to be persisted (a few dozen floats per mission); the height field at
   any ``(x, y, t)`` is evaluated in closed form on demand.
2. Wavenumber ``k`` is derived from angular frequency ``omega`` via the
   deep-water dispersion relation ``omega**2 = g*k`` — physically standard for
   open-ocean gravity waves and keeps the model to a single free parameter
   per mode (frequency) rather than requiring both frequency and wavenumber
   to be sampled and kept consistent.
3. ``height`` and ``height_rate`` (``d(height)/dt``) are both implemented as
   exact analytic derivatives of the same closed-form expression — not a
   finite-difference approximation — because ``height_rate`` is the quantity
   the wave-hazard cause (a later phase) depends on directly, and an
   analytic derivative is both cheaper and exact, unlike numerical
   differentiation of a noisy sampled signal.
4. ``WaveMode`` and ``WaveField`` are plain dataclasses with explicit
   ``to_dict``/``from_dict`` so a mission's sampled coefficients round-trip
   through the ``env_field_params`` storage table (Phase 4's concern) without
   any custom serialization logic living outside this module.

Data structures
----------------
``WaveMode`` — one spectral component (amplitude_m, frequency_hz, phase_rad,
direction_rad).
``WaveField`` — an ordered collection of ``WaveMode``, with vectorized
``height``/``height_rate`` evaluation.

Storage implications
---------------------
A ``WaveField`` with 8 modes serializes to 8 x 4 = 32 floats (~128 bytes) —
negligible relative to the per-mission storage budget, and the *only* thing
that needs to be written to disk for the wave field to be exactly
reproducible from ``env_field_params`` at load time.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..mission_config import WaveConfig

GRAVITY_MPS2 = 9.81


@dataclass(frozen=True)
class WaveMode:
    """A single deep-water gravity-wave spectral component."""

    amplitude_m: float
    frequency_hz: float
    phase_rad: float
    direction_rad: float

    def validate(self) -> None:
        if self.amplitude_m < 0.0:
            raise ValueError(f"WaveMode.amplitude_m must be >= 0, got {self.amplitude_m}")
        if self.frequency_hz <= 0.0:
            raise ValueError(f"WaveMode.frequency_hz must be > 0, got {self.frequency_hz}")

    @property
    def angular_frequency_rad_s(self) -> float:
        return 2.0 * math.pi * self.frequency_hz

    @property
    def wavenumber_rad_m(self) -> float:
        """Deep-water dispersion relation: omega^2 = g * k."""
        omega = self.angular_frequency_rad_s
        return (omega ** 2) / GRAVITY_MPS2

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "WaveMode":
        return cls(**data)


def sample_wave_modes(config: WaveConfig, rng: np.random.Generator) -> list[WaveMode]:
    """Draw ``config.n_modes`` spectral components from the configured
    sampling ranges, using the caller-supplied (already mission/stream-seeded)
    RNG. This function performs no seeding itself — reproducibility is
    entirely the caller's (``RandomManager``'s) responsibility.
    """
    amp_lo, amp_hi = config.amplitude_range_m
    freq_lo, freq_hi = config.frequency_range_hz
    modes: list[WaveMode] = []
    for _ in range(config.n_modes):
        amplitude = float(rng.uniform(amp_lo, amp_hi))
        frequency = float(rng.uniform(freq_lo, freq_hi))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        direction = (
            float(rng.uniform(0.0, 2.0 * math.pi)) if config.direction_randomized else 0.0
        )
        mode = WaveMode(
            amplitude_m=amplitude, frequency_hz=frequency, phase_rad=phase, direction_rad=direction
        )
        mode.validate()
        modes.append(mode)
    return modes


@dataclass
class WaveField:
    """A mission's full spectral wave field: an ordered set of ``WaveMode``."""

    modes: list[WaveMode]

    def validate(self) -> None:
        if not self.modes:
            raise ValueError("WaveField must contain at least one mode")
        for mode in self.modes:
            mode.validate()

    @classmethod
    def sample(cls, config: WaveConfig, rng: np.random.Generator) -> "WaveField":
        field = cls(modes=sample_wave_modes(config, rng))
        field.validate()
        return field

    def height(self, x_m: float | np.ndarray, y_m: float | np.ndarray, t_s: float | np.ndarray) -> np.ndarray:
        """Closed-form wave elevation eta(x, y, t) [m], vectorized over
        x_m/y_m/t_s (which must be broadcastable to a common shape).
        """
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        t = np.asarray(t_s, dtype=np.float64)
        total = np.zeros(np.broadcast(x, y, t).shape, dtype=np.float64)
        for mode in self.modes:
            k = mode.wavenumber_rad_m
            omega = mode.angular_frequency_rad_s
            projected = x * math.cos(mode.direction_rad) + y * math.sin(mode.direction_rad)
            total = total + mode.amplitude_m * np.cos(k * projected - omega * t + mode.phase_rad)
        return total

    def height_rate(
        self, x_m: float | np.ndarray, y_m: float | np.ndarray, t_s: float | np.ndarray
    ) -> np.ndarray:
        """Exact analytic d(eta)/dt [m/s], vectorized the same way as
        ``height``. This is the quantity the wave-hazard cause consumes.
        """
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        t = np.asarray(t_s, dtype=np.float64)
        total = np.zeros(np.broadcast(x, y, t).shape, dtype=np.float64)
        for mode in self.modes:
            k = mode.wavenumber_rad_m
            omega = mode.angular_frequency_rad_s
            projected = x * math.cos(mode.direction_rad) + y * math.sin(mode.direction_rad)
            # d/dt[cos(k*proj - omega*t + phase)] = omega * sin(k*proj - omega*t + phase)
            total = total + mode.amplitude_m * omega * np.sin(k * projected - omega * t + mode.phase_rad)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {"modes": [m.to_dict() for m in self.modes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WaveField":
        field = cls(modes=[WaveMode.from_dict(m) for m in data["modes"]])
        field.validate()
        return field

"""
physics/current_field.py

Closed-form, depth-layered current-field model.

Design decisions
-----------------
1. The current field is represented as a small number of depth-anchored,
   slowly time-varying sinusoidal (tidal-like) components, each with its own
   depth, amplitude, period, phase, and direction — the same
   coefficients-only, closed-form philosophy as ``wave_field.py``. Velocity
   at an arbitrary depth is obtained by linear interpolation between the two
   nearest layers by depth, not by storing a dense depth grid.
2. ``integrated_drift`` computes the exact analytic integral of a sinusoidal
   velocity component over ``[t0, t1]`` (``∫A·sin(ωt+φ)dt =
   -(A/ω)[cos(ωt1+φ) - cos(ωt0+φ)]``) rather than numerically integrating a
   sampled velocity trace. This is what the current-drift hazard cause (a
   later phase) needs: the AUV's positional drift relative to its commanded
   trajectory over an interval, computed exactly rather than approximated.
3. Depth interpolation uses the two layers bracketing the query depth (or
   clamps to the nearest boundary layer outside the configured depth range)
   — a single, explicit, testable rule rather than an implicit numerical
   scheme.
4. Like ``WaveField``, ``CurrentField`` is trivially serializable
   (``to_dict``/``from_dict``) so only the sampled layer coefficients need to
   persist per mission.

Data structures
----------------
``CurrentLayer`` — one depth-anchored sinusoidal component (depth_m,
amplitude_mps, period_s, phase_rad, direction_rad).
``CurrentField`` — an ordered (by depth) collection of ``CurrentLayer``, with
vectorized ``velocity`` evaluation and analytic ``integrated_drift``.

Storage implications
---------------------
A ``CurrentField`` with 4 layers serializes to 4 x 5 = 20 floats (~80 bytes)
per mission — negligible, and, as with the wave field, the only persisted
artifact needed to exactly reconstruct current velocity/drift at any
``(depth, time)`` at load time.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..mission_config import CurrentConfig


@dataclass(frozen=True)
class CurrentLayer:
    """A single depth-anchored sinusoidal current component."""

    depth_m: float
    amplitude_mps: float
    period_s: float
    phase_rad: float
    direction_rad: float

    def validate(self) -> None:
        if self.depth_m < 0.0:
            raise ValueError(f"CurrentLayer.depth_m must be >= 0, got {self.depth_m}")
        if self.amplitude_mps < 0.0:
            raise ValueError(f"CurrentLayer.amplitude_mps must be >= 0, got {self.amplitude_mps}")
        if self.period_s <= 0.0:
            raise ValueError(f"CurrentLayer.period_s must be > 0, got {self.period_s}")

    @property
    def angular_frequency_rad_s(self) -> float:
        return 2.0 * math.pi / self.period_s

    def _speed_at(self, t_s: np.ndarray) -> np.ndarray:
        omega = self.angular_frequency_rad_s
        return self.amplitude_mps * np.sin(omega * t_s + self.phase_rad)

    def velocity_uv(self, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        speed = self._speed_at(t_s)
        u = speed * math.cos(self.direction_rad)
        v = speed * math.sin(self.direction_rad)
        return u, v

    def integrated_drift_along_speed_axis(self, t0_s: float, t1_s: float) -> float:
        """Exact analytic integral of the along-direction speed component
        over [t0_s, t1_s]: ``integral A*sin(omega*t + phase) dt``.
        """
        omega = self.angular_frequency_rad_s
        a = self.amplitude_mps
        return -(a / omega) * (
            math.cos(omega * t1_s + self.phase_rad) - math.cos(omega * t0_s + self.phase_rad)
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "CurrentLayer":
        return cls(**data)


def sample_current_layers(config: CurrentConfig, rng: np.random.Generator) -> list[CurrentLayer]:
    """Draw ``config.n_layers`` current components, spaced across the
    configured depth range, from the caller-supplied (already mission/stream-
    seeded) RNG. Layer depths are evenly spaced (not randomly drawn) so that
    interpolation coverage across the depth range is guaranteed regardless of
    ``n_layers`` — randomizing depth placement risks leaving large depth gaps
    with no bracketing layer.
    """
    depth_lo, depth_hi = config.depth_range_m
    amp_lo, amp_hi = config.amplitude_range_mps
    period_lo, period_hi = config.period_range_s

    depths = np.linspace(depth_lo, depth_hi, config.n_layers)
    layers: list[CurrentLayer] = []
    for depth in depths:
        amplitude = float(rng.uniform(amp_lo, amp_hi))
        period = float(rng.uniform(period_lo, period_hi))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        direction = float(rng.uniform(0.0, 2.0 * math.pi))
        layer = CurrentLayer(
            depth_m=float(depth),
            amplitude_mps=amplitude,
            period_s=period,
            phase_rad=phase,
            direction_rad=direction,
        )
        layer.validate()
        layers.append(layer)
    return layers


@dataclass
class CurrentField:
    """A mission's full layered current field: layers ordered by depth."""

    layers: list[CurrentLayer]

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("CurrentField must contain at least one layer")
        for layer in self.layers:
            layer.validate()
        depths = [layer.depth_m for layer in self.layers]
        if depths != sorted(depths):
            raise ValueError("CurrentField.layers must be ordered by non-decreasing depth_m")

    @classmethod
    def sample(cls, config: CurrentConfig, rng: np.random.Generator) -> "CurrentField":
        field = cls(layers=sample_current_layers(config, rng))
        field.validate()
        return field

    def _bracketing_layers(self, depth_m: float) -> tuple[CurrentLayer, CurrentLayer, float]:
        """Return (lower_layer, upper_layer, weight) such that the query
        depth is linearly interpolated as
        ``weight * lower + (1 - weight) * upper``... concretely, weight is
        the interpolation fraction toward the upper layer. Depths outside the
        configured range clamp to the nearest boundary layer (weight 0 or 1
        degenerate case handled by returning the same layer twice).
        """
        if depth_m <= self.layers[0].depth_m:
            return self.layers[0], self.layers[0], 0.0
        if depth_m >= self.layers[-1].depth_m:
            return self.layers[-1], self.layers[-1], 0.0
        for lower, upper in zip(self.layers[:-1], self.layers[1:]):
            if lower.depth_m <= depth_m <= upper.depth_m:
                span = upper.depth_m - lower.depth_m
                frac = 0.0 if span == 0.0 else (depth_m - lower.depth_m) / span
                return lower, upper, frac
        # unreachable given the boundary checks above and validate()'s sorted-depth
        # invariant, but kept explicit rather than silently falling through
        raise RuntimeError(f"Failed to bracket depth_m={depth_m} within CurrentField layers")

    def velocity(self, depth_m: float, t_s: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Depth-interpolated (eastward, northward) velocity [m/s] at a
        single depth and one or more query times.
        """
        t = np.asarray(t_s, dtype=np.float64)
        lower, upper, frac = self._bracketing_layers(depth_m)
        u_lo, v_lo = lower.velocity_uv(t)
        if lower is upper:
            return u_lo, v_lo
        u_hi, v_hi = upper.velocity_uv(t)
        u = (1.0 - frac) * u_lo + frac * u_hi
        v = (1.0 - frac) * v_lo + frac * v_hi
        return u, v

    def velocity_batch(
        self, depth_m: np.ndarray, t_s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fully vectorized (eastward, northward) velocity for many
        (depth, time) query pairs at once — the batched counterpart to
        ``velocity``, used by the event generator's hot path (evaluating
        hazards for many candidate edges simultaneously) instead of looping
        over edges in Python.

        ``depth_m`` and ``t_s`` must be broadcastable to a common shape;
        the result has that shape. Depth interpolation/clamping matches
        ``velocity``/``_bracketing_layers`` exactly (verified by a
        dedicated consistency test), computed here via vectorized
        ``searchsorted`` instead of a per-query Python loop.
        """
        depth = np.asarray(depth_m, dtype=np.float64)
        t = np.asarray(t_s, dtype=np.float64)
        depth, t = np.broadcast_arrays(depth, t)
        depth = np.array(depth, dtype=np.float64)  # materialize a writable copy post-broadcast

        layer_depths = np.array([layer.depth_m for layer in self.layers], dtype=np.float64)
        depth_clipped = np.clip(depth, layer_depths[0], layer_depths[-1])

        upper_idx = np.searchsorted(layer_depths, depth_clipped, side="left")
        upper_idx = np.clip(upper_idx, 1, len(layer_depths) - 1)
        lower_idx = upper_idx - 1

        lower_depth = layer_depths[lower_idx]
        upper_depth = layer_depths[upper_idx]
        span = upper_depth - lower_depth
        frac = np.where(span > 0.0, (depth_clipped - lower_depth) / np.where(span > 0.0, span, 1.0), 0.0)

        u = np.zeros_like(depth)
        v = np.zeros_like(depth)
        for layer_index, layer in enumerate(self.layers):
            is_lower = lower_idx == layer_index
            is_upper = upper_idx == layer_index
            if not (is_lower.any() or is_upper.any()):
                continue
            u_layer, v_layer = layer.velocity_uv(t)
            weight = np.where(is_lower, 1.0 - frac, np.where(is_upper, frac, 0.0))
            u += weight * u_layer
            v += weight * v_layer
        return u, v

    def integrated_drift(self, depth_m: float, t0_s: float, t1_s: float) -> tuple[float, float]:
        """Exact analytic (eastward, northward) positional drift [m] at a
        fixed depth over ``[t0_s, t1_s]``, obtained by depth-interpolating
        the per-layer analytic speed integrals (interpolation and
        integration commute for a linear interpolant of linear operators).
        """
        lower, upper, frac = self._bracketing_layers(depth_m)
        drift_lo = lower.integrated_drift_along_speed_axis(t0_s, t1_s)
        du_lo, dv_lo = drift_lo * math.cos(lower.direction_rad), drift_lo * math.sin(lower.direction_rad)
        if lower is upper:
            return du_lo, dv_lo
        drift_hi = upper.integrated_drift_along_speed_axis(t0_s, t1_s)
        du_hi, dv_hi = drift_hi * math.cos(upper.direction_rad), drift_hi * math.sin(upper.direction_rad)
        du = (1.0 - frac) * du_lo + frac * du_hi
        dv = (1.0 - frac) * dv_lo + frac * dv_hi
        return du, dv

    def to_dict(self) -> dict[str, Any]:
        return {"layers": [layer.to_dict() for layer in self.layers]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurrentField":
        field = cls(layers=[CurrentLayer.from_dict(layer) for layer in data["layers"]])
        field.validate()
        return field

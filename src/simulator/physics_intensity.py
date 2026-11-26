"""
physics_intensity.py

Cause-specific competing-risks hazard functions: geometric (range/SNR),
wave-induced, Doppler/mobility-induced, and current-drift-induced, plus a
symmetric recovery hazard for the DOWN -> UP transition.

Design decisions
-----------------
1. Every hazard is a pure, closed-form, non-negative function of the physics
   primitives from Phase 2 (``AcousticChannel``, ``WaveField``,
   ``CurrentField``) and the free ``kappa_c`` constants from
   ``HazardConfig`` — matching the "physics enters at exactly one principled
   point" design from the PG-GTPP research formulation. No hazard function
   here is fit to data; each is a named, physically motivated expression.
2. ``geometric_hazard`` is zero whenever the link has positive SNR margin
   (i.e., is nominally within range) and grows linearly as the margin goes
   negative — modeling range-driven failure as a *soft*, bursty risk near
   and beyond the nominal threshold, not a hard deterministic cutoff. This
   is deliberate: a hard cutoff would make "communication failure" a
   trivial deterministic function of distance, defeating the point of a
   stochastic hazard model.
3. ``wave_hazard`` and ``doppler_hazard`` use the *instantaneous* rate-of-
   change quantities (``WaveField.height_rate``, relative velocity) rather
   than absolute wave height or absolute speed — consistent with the
   research design's framing of these as *disturbance* hazards (rapid
   change destabilizes the acoustic link), not steady-state conditions.
4. ``current_hazard`` uses the *instantaneous current speed* at each node's
   depth (from ``CurrentField.velocity``), not the integrated drift
   (``CurrentField.integrated_drift``) — the integrated drift is a
   diagnostic for displacement over an interval (useful for QA/analysis),
   whereas a hazard *rate* must be an instantaneous quantity, dimensionally
   consistent with the other three cause hazards (all are "per second"
   rates that get integrated over time by the event generator's thinning
   algorithm).
5. ``recovery_hazard`` is deliberately the *only* asymmetry in this module:
   it is a single scalar hazard (no cause decomposition — a recovered link
   does not need cause attribution, matching the schema decision in
   ``event_types.py`` where ``LINK_UP`` events carry ``cause=None``),
   increasing with SNR margin the same way ``geometric_hazard`` decreases
   with it. This keeps the whole two-state process consistent: conditions
   that make failure likely make recovery unlikely, and vice versa.

Data structures
----------------
``CauseHazards`` — a small dataclass bundling all four cause-specific
hazard values plus their sum, returned by ``HazardEvaluator.cause_hazards``.
``HazardEvaluator`` — bundles the three physics primitives and the
``HazardConfig`` constants, exposing hazard evaluation as methods so callers
don't have to thread five objects through every call.

Storage implications
---------------------
None — every function here is a pure, on-demand computation, consistent with
the "recompute, don't store" principle. Only the ``HazardConfig`` constants
(a handful of floats, already part of ``MissionConfig``) and the physics
field coefficients (Phase 2) need to persist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .event_types import CauseType
from .mission_config import HazardConfig
from .physics import AcousticChannel, CurrentField, WaveField


@dataclass(frozen=True)
class CauseHazards:
    """The four cause-specific hazard values [1/s] at a single instant, plus
    their sum (the total failure intensity a competing-risks process uses)."""

    geo: float
    wave: float
    doppler: float
    current: float

    def validate(self) -> None:
        for name, value in (
            ("geo", self.geo),
            ("wave", self.wave),
            ("doppler", self.doppler),
            ("current", self.current),
        ):
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"CauseHazards.{name} must be finite and >= 0, got {value}")

    @property
    def total(self) -> float:
        return self.geo + self.wave + self.doppler + self.current

    def as_dict(self) -> dict[CauseType, float]:
        return {
            CauseType.GEOMETRIC: self.geo,
            CauseType.WAVE: self.wave,
            CauseType.DOPPLER: self.doppler,
            CauseType.CURRENT: self.current,
        }

    def dominant_cause(self) -> CauseType:
        """The cause with the highest instantaneous hazard — used by the
        event generator to assign a cause label to an accepted failure
        event (``argmax_c lambda^c(t)``, per the research design).
        """
        return max(self.as_dict().items(), key=lambda kv: kv[1])[0]

    def ambiguity_ratio(self) -> float:
        """Ratio of the top cause's hazard to the runner-up's — the
        practical, per-event stand-in for the identifiability audit from the
        QA verification framework (Phase 3.3, "ambiguous-event fraction").
        A value near 1.0 means the causes are barely distinguishable at this
        instant; a large value means attribution is unambiguous.
        """
        values = sorted(self.as_dict().values(), reverse=True)
        top, second = values[0], values[1]
        if second <= 0.0:
            return math.inf if top > 0.0 else 1.0
        return top / second


def geometric_hazard(snr_margin_db, kappa_geo: float):
    """Hazard rate from range/SNR margin: zero while the link has positive
    margin, growing linearly as the margin goes negative.

    Accepts a Python float or a numpy array for ``snr_margin_db`` — the same
    function serves both the single-point evaluation path
    (``HazardEvaluator.cause_hazards``) and the vectorized batch path
    (``HazardEvaluator.cause_hazards_batch``), since ``np.maximum`` is a
    strict drop-in for Python's ``max`` on scalars while additionally
    supporting elementwise array operation.
    """
    return kappa_geo * np.maximum(0.0, -snr_margin_db)


def wave_hazard(height_rate_i_mps, height_rate_j_mps, kappa_wave: float):
    """Hazard rate from wave-induced disturbance at either endpoint node."""
    return kappa_wave * (np.abs(height_rate_i_mps) + np.abs(height_rate_j_mps))


def doppler_hazard(relative_speed_mps, kappa_doppler: float):
    """Hazard rate from Doppler/mobility-induced multipath decorrelation,
    proportional to the square of the relative speed between endpoints."""
    return kappa_doppler * (relative_speed_mps ** 2)


def current_hazard(current_speed_i_mps, current_speed_j_mps, kappa_current: float):
    """Hazard rate from ambient current speed at either endpoint's depth."""
    return kappa_current * (current_speed_i_mps + current_speed_j_mps)


def recovery_hazard(snr_margin_db, kappa_recovery: float):
    """Hazard rate for a DOWN -> UP transition: zero while the link has
    negative margin, growing linearly as the margin becomes positive —
    the mirror image of ``geometric_hazard``."""
    return kappa_recovery * np.maximum(0.0, snr_margin_db)


class HazardEvaluator:
    """Bundles the physics primitives and hazard constants needed to
    evaluate cause-specific and recovery hazards for a single edge at a
    single instant.
    """

    def __init__(
        self,
        acoustic_channel: AcousticChannel,
        wave_field: WaveField,
        current_field: CurrentField,
        hazard_config: HazardConfig,
    ) -> None:
        hazard_config.validate()
        self.acoustic_channel = acoustic_channel
        self.wave_field = wave_field
        self.current_field = current_field
        self.hazard_config = hazard_config

    def cause_hazards(
        self,
        distance_m: float,
        x_i: float, y_i: float, z_i: float,
        x_j: float, y_j: float, z_j: float,
        vx_i: float, vy_i: float,
        vx_j: float, vy_j: float,
        t_s: float,
    ) -> CauseHazards:
        """Evaluate all four cause-specific failure hazards for an edge
        between nodes i and j at time ``t_s``, given their instantaneous
        positions and horizontal velocities.
        """
        snr_margin = float(self.acoustic_channel.snr_margin_db(distance_m))

        rate_i = float(self.wave_field.height_rate(x_i, y_i, t_s))
        rate_j = float(self.wave_field.height_rate(x_j, y_j, t_s))

        relative_speed = math.hypot(vx_i - vx_j, vy_i - vy_j)

        cu_i, cv_i = self.current_field.velocity(depth_m=z_i, t_s=t_s)
        cu_j, cv_j = self.current_field.velocity(depth_m=z_j, t_s=t_s)
        current_speed_i = float(math.hypot(cu_i, cv_i))
        current_speed_j = float(math.hypot(cu_j, cv_j))

        hazards = CauseHazards(
            geo=geometric_hazard(snr_margin, self.hazard_config.kappa_geo),
            wave=wave_hazard(rate_i, rate_j, self.hazard_config.kappa_wave),
            doppler=doppler_hazard(relative_speed, self.hazard_config.kappa_doppler),
            current=current_hazard(current_speed_i, current_speed_j, self.hazard_config.kappa_current),
        )
        hazards.validate()
        return hazards

    def recovery_hazard(self, distance_m: float) -> float:
        """Evaluate the DOWN -> UP recovery hazard for an edge at the given
        distance (recovery depends only on the current SNR margin, per the
        module docstring, point 5)."""
        snr_margin = float(self.acoustic_channel.snr_margin_db(distance_m))
        return recovery_hazard(snr_margin, self.hazard_config.kappa_recovery)

    # ------------------------------------------------------------------ #
    # Vectorized batch evaluation — the event generator's hot path
    # ------------------------------------------------------------------ #
    def cause_hazards_batch(
        self,
        distance_m: np.ndarray,
        x_i: np.ndarray, y_i: np.ndarray, z_i: np.ndarray,
        x_j: np.ndarray, y_j: np.ndarray, z_j: np.ndarray,
        vx_i: np.ndarray, vy_i: np.ndarray,
        vx_j: np.ndarray, vy_j: np.ndarray,
        t_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized counterpart to ``cause_hazards``: evaluates the four
        cause-specific failure hazards for many (edge, time) query points at
        once, returning ``(geo, wave, doppler, current)`` arrays instead of
        one ``CauseHazards`` per call. All array arguments must be
        broadcastable to a common shape; used by ``event_generator.py`` to
        replace a per-edge Python loop with a handful of numpy calls per
        thinning round — this is the fix for the O(brackets x edges x
        probes) flat-cost bottleneck identified during Phase 3 profiling.

        Values are *not* individually validated (unlike ``cause_hazards``,
        which validates every single-point result) — validating a large
        array on every batch call would reintroduce per-element Python
        overhead and defeat the purpose of vectorizing. Callers that need
        the guarantee should validate the small, already-accepted-event
        subset instead (see ``event_generator.py``).
        """
        snr_margin = self.acoustic_channel.snr_margin_db(distance_m)

        rate_i = self.wave_field.height_rate(x_i, y_i, t_s)
        rate_j = self.wave_field.height_rate(x_j, y_j, t_s)

        relative_speed = np.hypot(np.asarray(vx_i) - np.asarray(vx_j), np.asarray(vy_i) - np.asarray(vy_j))

        cu_i, cv_i = self.current_field.velocity_batch(z_i, t_s)
        cu_j, cv_j = self.current_field.velocity_batch(z_j, t_s)
        current_speed_i = np.hypot(cu_i, cv_i)
        current_speed_j = np.hypot(cu_j, cv_j)

        geo = geometric_hazard(snr_margin, self.hazard_config.kappa_geo)
        wave = wave_hazard(rate_i, rate_j, self.hazard_config.kappa_wave)
        doppler = doppler_hazard(relative_speed, self.hazard_config.kappa_doppler)
        current = current_hazard(current_speed_i, current_speed_j, self.hazard_config.kappa_current)
        return geo, wave, doppler, current

    def recovery_hazard_batch(self, distance_m: np.ndarray) -> np.ndarray:
        """Vectorized counterpart to ``recovery_hazard``."""
        snr_margin = self.acoustic_channel.snr_margin_db(distance_m)
        return recovery_hazard(snr_margin, self.hazard_config.kappa_recovery)

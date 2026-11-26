"""
mission_config.py

Mission configuration schema for the PG-GTPP simulator.

Design decisions
-----------------
1. Configuration is a tree of small, frozen (immutable) dataclasses — one per
   physics/domain concern (wave field, current field, acoustic channel, fleet
   composition, mission envelope) — rather than one flat dataclass. Later
   phases (wave_field.py, current_field.py, acoustic_channel.py,
   mission_generator.py) each depend only on the sub-config they need.
2. Configs are frozen so that once a ``MissionConfig`` is constructed and
   hashed, it cannot silently drift. The reproducibility framework requires
   that a mission's stored ``config_hash`` always describes the exact config
   object that produced it; mutability would make that guarantee unenforceable.
3. ``MissionConfig.from_yaml`` validates immediately after construction, so a
   malformed config fails fast at load time rather than deep inside mission
   generation, where the failure would be harder to attribute.
4. ``config_hash`` is computed from a canonical (sorted-key, whitespace-free)
   JSON serialization of the fully resolved config. This guarantees that two
   configs with identical field values hash identically regardless of
   construction order, and that any field change (including nested ones)
   changes the hash — the property the storage/reproducibility framework
   depends on to detect config drift between generation time and load time.

Data structures
----------------
``WaveConfig``, ``CurrentConfig``, ``AcousticConfig``, ``FleetConfig``, and
``MissionEnvelopeConfig`` are the five physics/domain sub-configs. They are
composed into a single ``MissionConfig``, which additionally carries the
reproducibility fields (``master_seed``, ``schema_version``) and the dataset
split label.

Storage implications
---------------------
Config objects themselves are tiny (a few dozen floats/ints) and are not part
of the per-mission storage budget analyzed in the implementation architecture
document. What *is* budget-relevant is that ``config_hash`` is stored once per
mission (as a short string) in ``mission_index.parquet`` — this file's output
is exactly what makes that field meaningful and checkable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when a mission configuration fails validation."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigValidationError(message)


@dataclass(frozen=True)
class WaveConfig:
    """Spectral wave-field configuration (closed-form, coefficients-only).

    The wave field is represented as a superposition of ``n_modes`` spectral
    components rather than a spatial grid — see the storage-aware simulator
    design. This config only specifies the *sampling ranges* used to draw
    per-mission mode coefficients (amplitude/frequency/phase/direction); the
    coefficients themselves are drawn at mission-generation time by
    ``mission_generator.py`` using a seeded RNG stream, not stored here.
    """

    n_modes: int = 8
    amplitude_range_m: tuple[float, float] = (0.05, 2.5)
    frequency_range_hz: tuple[float, float] = (0.05, 0.5)
    direction_randomized: bool = True

    def validate(self) -> None:
        _check(self.n_modes >= 1, "WaveConfig.n_modes must be >= 1")
        lo, hi = self.amplitude_range_m
        _check(0.0 <= lo < hi, "WaveConfig.amplitude_range_m must satisfy 0 <= lo < hi")
        lo, hi = self.frequency_range_hz
        _check(0.0 < lo < hi, "WaveConfig.frequency_range_hz must satisfy 0 < lo < hi")


@dataclass(frozen=True)
class CurrentConfig:
    """Layered current-field configuration (closed-form, coefficients-only)."""

    n_layers: int = 4
    depth_range_m: tuple[float, float] = (0.0, 200.0)
    amplitude_range_mps: tuple[float, float] = (0.0, 0.6)
    period_range_s: tuple[float, float] = (600.0, 45000.0)

    def validate(self) -> None:
        _check(self.n_layers >= 1, "CurrentConfig.n_layers must be >= 1")
        lo, hi = self.depth_range_m
        _check(0.0 <= lo < hi, "CurrentConfig.depth_range_m must satisfy 0 <= lo < hi")
        lo, hi = self.amplitude_range_mps
        _check(0.0 <= lo < hi, "CurrentConfig.amplitude_range_mps must satisfy 0 <= lo < hi")
        lo, hi = self.period_range_s
        _check(0.0 < lo < hi, "CurrentConfig.period_range_s must satisfy 0 < lo < hi")


@dataclass(frozen=True)
class AcousticConfig:
    """Urick sonar-equation parameters for the underwater acoustic link
    budget.

    PHYSICS CORRECTION (post-Phase-4). Earlier versions of this config
    (and the resulting ``AcousticChannel.snr_db``) compared a broadband
    source level (``source_level_db``, referenced to the full transmitted
    signal power) directly against the ambient noise level from
    ``ambient_noise_level_db()``, which is a **per-Hz noise spectral
    density** (dB re 1 uPa^2/Hz), with no conversion between them. That is
    an apples-to-oranges comparison and was the actual root cause of the
    unrealistic ~90 dB SNR margins flagged repeatedly since Phase 2 — not
    merely a matter of the specific constants being poorly chosen. The fix
    (``AcousticChannel.snr_db``) now converts the per-Hz noise density to
    total in-band noise power via ``+ 10*log10(bandwidth_hz)``, which is
    why this config now also carries a ``bandwidth_hz`` field it previously
    lacked. ``source_level_db`` has also been recalibrated (175 -> 130 dB)
    now that the comparison is dimensionally correct, chosen so that the
    resulting SNR margin crosses zero at a few hundred meters — comparable
    to the 2000x2000 m mission area — rather than at range so large the
    geometric hazard cause could never activate under any node placement.
    """

    source_level_db: float = 130.0
    frequency_hz: float = 12000.0
    bandwidth_hz: float = 5000.0
    snr_min_db: float = 10.0
    shipping_activity_factor: float = 0.5  # in [0, 1], drives Urick shipping noise term
    wind_speed_mps: float = 5.0
    spreading_loss_exponent: float = 1.5  # 1.0 cylindrical, 2.0 spherical, 1.5 practical

    def validate(self) -> None:
        _check(self.source_level_db > 0.0, "AcousticConfig.source_level_db must be > 0")
        _check(self.frequency_hz > 0.0, "AcousticConfig.frequency_hz must be > 0")
        _check(self.bandwidth_hz > 0.0, "AcousticConfig.bandwidth_hz must be > 0")
        _check(0.0 <= self.shipping_activity_factor <= 1.0,
               "AcousticConfig.shipping_activity_factor must be in [0, 1]")
        _check(self.wind_speed_mps >= 0.0, "AcousticConfig.wind_speed_mps must be >= 0")
        _check(1.0 <= self.spreading_loss_exponent <= 2.0,
               "AcousticConfig.spreading_loss_exponent must be in [1.0, 2.0]")


@dataclass(frozen=True)
class FleetConfig:
    """Fleet composition ranges. USV count is fixed at 1 per the baseline system."""

    usv_count: int = 1
    n_auv_min: int = 3
    n_auv_max: int = 8
    m_iout_min: int = 10
    m_iout_max: int = 30
    max_auv_speed_mps: float = 2.5

    def validate(self) -> None:
        _check(self.usv_count == 1, "FleetConfig.usv_count must be 1 in the current system scope")
        _check(1 <= self.n_auv_min <= self.n_auv_max, "FleetConfig AUV range invalid")
        _check(1 <= self.m_iout_min <= self.m_iout_max, "FleetConfig IoUT range invalid")
        _check(self.max_auv_speed_mps > 0.0, "FleetConfig.max_auv_speed_mps must be > 0")


@dataclass(frozen=True)
class MissionEnvelopeConfig:
    """Mission-level envelope: duration, sampling rate, area, depth, sea state."""

    duration_s: float = 1800.0
    node_sample_rate_hz: float = 1.0
    area_size_m: tuple[float, float] = (2000.0, 2000.0)
    depth_range_m: tuple[float, float] = (20.0, 150.0)
    sea_state: str = "moderate"  # one of {"calm", "moderate", "extreme"}

    _VALID_SEA_STATES = ("calm", "moderate", "extreme")

    def validate(self) -> None:
        _check(self.duration_s > 0.0, "MissionEnvelopeConfig.duration_s must be > 0")
        _check(self.node_sample_rate_hz > 0.0,
               "MissionEnvelopeConfig.node_sample_rate_hz must be > 0")
        w, h = self.area_size_m
        _check(w > 0.0 and h > 0.0, "MissionEnvelopeConfig.area_size_m must be positive")
        lo, hi = self.depth_range_m
        _check(0.0 <= lo < hi, "MissionEnvelopeConfig.depth_range_m must satisfy 0 <= lo < hi")
        _check(self.sea_state in self._VALID_SEA_STATES,
               f"MissionEnvelopeConfig.sea_state must be one of {self._VALID_SEA_STATES}")

    @property
    def expected_samples(self) -> int:
        """Number of node-trajectory samples this envelope implies."""
        return int(round(self.duration_s * self.node_sample_rate_hz))


@dataclass(frozen=True)
class HazardConfig:
    """Cause-specific base-rate constants (kappa_c) for the four named
    competing-risks hazard causes (geo / wave / doppler / current).

    These are the *only* free parameters of the closed-form physics hazard
    model (see ``physics_intensity.py``) — everything else is derived from
    the wave/current/acoustic physics primitives. Exposing them here (rather
    than hardcoding them inside the hazard module) keeps them YAML-
    configurable and calibratable, per the "no hardcoded values" coding
    standard established in Phase 1.

    CALIBRATION STATUS — read before generating any dataset used for
    published results. The defaults below reflect a **second-pass
    calibration** performed after the ``AcousticConfig`` bandwidth-
    conversion bug fix (see that class's docstring): fixing the acoustic
    physics alone made ``geo`` active (previously fully degenerate) and
    dramatically reduced the ambiguous-event fraction (52% -> under 10-20%
    across tested seeds) without changing any kappa here. ``kappa_current``
    was then raised from its Phase-3 value (0.0002, under which ``current``
    never won as the dominant cause in any tested mission despite not
    being strictly zero-variance) to 0.01, verified across multiple seeds
    to give all four causes real representation while keeping the
    ambiguous-event fraction and cause-correlation matrix within the QA
    framework's tolerances. This is still **not validated against real
    underwater acoustic outage statistics** — it is an internally
    consistent, QA-verified calibration, not a literature-derived one. That
    external validation (QA-framework risk #2) remains open.
    """

    kappa_geo: float = 0.01
    kappa_wave: float = 0.0001
    kappa_doppler: float = 0.0002
    kappa_current: float = 0.01
    kappa_recovery: float = 0.05

    def validate(self) -> None:
        for name in ("kappa_geo", "kappa_wave", "kappa_doppler", "kappa_current", "kappa_recovery"):
            value = getattr(self, name)
            _check(value >= 0.0, f"HazardConfig.{name} must be >= 0")


@dataclass(frozen=True)
class GraphConfig:
    """Candidate-edge construction parameters."""

    candidate_range_factor: float = 1.5  # multiplier applied to the theoretical max range
    n_thinning_probes_per_bracket: int = 5  # sub-samples used to bound the dominating rate
    dominating_rate_safety_factor: float = 1.25

    def validate(self) -> None:
        _check(self.candidate_range_factor > 1.0,
               "GraphConfig.candidate_range_factor must be > 1.0 (must exceed the viable range)")
        _check(self.n_thinning_probes_per_bracket >= 2,
               "GraphConfig.n_thinning_probes_per_bracket must be >= 2")
        _check(self.dominating_rate_safety_factor > 1.0,
               "GraphConfig.dominating_rate_safety_factor must be > 1.0")


@dataclass(frozen=True)
class MissionConfig:
    """Top-level, fully resolved mission configuration."""

    schema_version: str = "1.0"
    master_seed: int = 0
    split: str = "train"  # one of {"train", "val", "test"}

    wave: WaveConfig = field(default_factory=WaveConfig)
    current: CurrentConfig = field(default_factory=CurrentConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    envelope: MissionEnvelopeConfig = field(default_factory=MissionEnvelopeConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    _VALID_SPLITS = ("train", "val", "test")

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        _check(isinstance(self.master_seed, int) and self.master_seed >= 0,
               "MissionConfig.master_seed must be a non-negative integer")
        _check(self.split in self._VALID_SPLITS,
               f"MissionConfig.split must be one of {self._VALID_SPLITS}")
        self.wave.validate()
        self.current.validate()
        self.acoustic.validate()
        self.fleet.validate()
        self.envelope.validate()
        self.hazard.validate()
        self.graph.validate()
        logger.debug("MissionConfig validated successfully (hash=%s)", self.config_hash)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionConfig":
        data = dict(data)  # shallow copy, do not mutate caller's dict
        wave = WaveConfig(**data.pop("wave", {}))
        current = CurrentConfig(**data.pop("current", {}))
        acoustic = AcousticConfig(**data.pop("acoustic", {}))
        fleet = FleetConfig(**data.pop("fleet", {}))
        envelope = MissionEnvelopeConfig(**data.pop("envelope", {}))
        hazard = HazardConfig(**data.pop("hazard", {}))
        graph = GraphConfig(**data.pop("graph", {}))
        # tuple fields arrive as lists from YAML/JSON; coerce back to tuples
        envelope = MissionEnvelopeConfig(
            duration_s=envelope.duration_s,
            node_sample_rate_hz=envelope.node_sample_rate_hz,
            area_size_m=tuple(envelope.area_size_m),
            depth_range_m=tuple(envelope.depth_range_m),
            sea_state=envelope.sea_state,
        )
        wave = WaveConfig(
            n_modes=wave.n_modes,
            amplitude_range_m=tuple(wave.amplitude_range_m),
            frequency_range_hz=tuple(wave.frequency_range_hz),
            direction_randomized=wave.direction_randomized,
        )
        current = CurrentConfig(
            n_layers=current.n_layers,
            depth_range_m=tuple(current.depth_range_m),
            amplitude_range_mps=tuple(current.amplitude_range_mps),
            period_range_s=tuple(current.period_range_s),
        )
        config = cls(
            schema_version=data.get("schema_version", "1.0"),
            master_seed=data.get("master_seed", 0),
            split=data.get("split", "train"),
            wave=wave,
            current=current,
            acoustic=acoustic,
            fleet=fleet,
            envelope=envelope,
            hazard=hazard,
            graph=graph,
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MissionConfig":
        path = Path(path)
        _check(path.exists(), f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info("Loaded mission config from %s", path)
        return cls.from_dict(raw)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=True, default_flow_style=False)
        logger.info("Wrote mission config to %s", path)

    # ------------------------------------------------------------------ #
    # Hashing (reproducibility)
    # ------------------------------------------------------------------ #
    @property
    def config_hash(self) -> str:
        """Deterministic content hash of the fully resolved config.

        Excludes ``split`` from the hash: two missions with identical physics
        configuration but assigned to different dataset splits should be
        recognized as physically identical configurations (this matters for
        ablation reuse, which changes model/loader config, not simulator
        physics config).
        """
        payload = self.to_dict()
        payload.pop("split", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def default_config(master_seed: int = 0, split: str = "train") -> MissionConfig:
    """Convenience constructor for a validated, default-parameter config."""
    config = MissionConfig(master_seed=master_seed, split=split)
    config.validate()
    return config

"""
physics/acoustic_channel.py

Closed-form underwater acoustic link-budget model (Urick sonar-equation
family): transmission loss, ambient noise, SNR, and a Shannon-capacity-based
throughput estimate.

IMPORTANT PROVENANCE NOTE — read before using this for publication results
----------------------------------------------------------------------------
This module implements the *standard, textbook* Urick (1983) passive-sonar
link-budget equation: source level, spreading-plus-absorption transmission
loss (Thorp's absorption-coefficient approximation), and ambient noise
(turbulence + shipping + wind + thermal terms, combined in the power domain).
It is written to be structurally consistent with the acoustic channel
described in the PG-GTPP research/simulator design documents (which specify
reusing the baseline paper's `calcSNR`/`calcRate`-style Urick implementation
unmodified). However, the baseline's exact source file was not available in
this coding session to diff against line-by-line, so **this is a standard
reference implementation, not a verified byte-for-bit port of the baseline's
exact constants**. Before this module is used to generate any dataset that
results in published claims, run the Phase 2 simulator-verification
procedure specified in the QA framework document — "golden-value regression
tests against baseline `calcSNR`/`calcRate` outputs" — to confirm or correct
numerical parity. This note should not be deleted until that verification has
actually been performed.

Design decisions
-----------------
1. Every quantity here is a pure, closed-form function of
   ``(distance_m, frequency_hz, AcousticConfig)`` — no state, no randomness,
   no time dependence beyond what the caller passes in through distance. This
   matches the storage-aware design: acoustic-channel quantities are never
   stored, always recomputed.
2. Ambient noise components (turbulence/shipping/wind/thermal) are combined
   in the power domain (``10*log10(sum(10**(NL_i/10)))``), which is the
   physically correct way to combine independent noise sources expressed in
   dB — a common implementation error is to average or sum the dB values
   directly, which is not physically meaningful.
3. Thorp's absorption-coefficient formula is used for frequency-dependent
   absorption loss, valid for the tens-of-kHz range typical of underwater
   acoustic modems, consistent with ``AcousticConfig.frequency_hz``'s
   documented default (12 kHz).
4. ``throughput_bps`` uses the Shannon capacity formula
   (``bandwidth * log2(1 + SNR_linear)``) as a principled, closed-form
   throughput estimate as a function of the computed SNR — not a fitted or
   ad hoc curve. ``bandwidth_hz`` is a parameter (not hardcoded) since actual
   underwater modem bandwidth varies by hardware and is not part of the
   research-design's fixed acoustic constants.

Data structures
----------------
``AcousticChannel`` — a thin, stateless wrapper around ``AcousticConfig``
exposing the link-budget functions as methods (so call sites read as
``channel.snr_db(distance_m)`` rather than free functions requiring the
config to be threaded through every call).

Storage implications
---------------------
None — every method here is a pure function recomputed on demand; this is
exactly the "recompute, don't store" principle the edge-covariate
reconstruction in the storage-aware architecture depends on.
"""

from __future__ import annotations

import math

import numpy as np

from ..mission_config import AcousticConfig


def thorp_absorption_db_per_km(frequency_hz: float) -> float:
    """Thorp's empirical absorption-coefficient approximation [dB/km].
    ``frequency_hz`` is converted internally to kHz, the unit Thorp's formula
    is defined in.
    """
    f_khz = frequency_hz / 1000.0
    f2 = f_khz ** 2
    return (
        0.11 * f2 / (1.0 + f2)
        + 44.0 * f2 / (4100.0 + f2)
        + 2.75e-4 * f2
        + 0.003
    )


def _combine_noise_db(*levels_db: float) -> float:
    """Combine independent noise sources expressed in dB by summing their
    linear (power) equivalents, then converting back to dB — the physically
    correct combination rule (see module docstring, point 2).
    """
    linear_sum = sum(10.0 ** (level / 10.0) for level in levels_db)
    return 10.0 * math.log10(linear_sum)


class AcousticChannel:
    """Stateless Urick-family link-budget model, parameterized by
    ``AcousticConfig``."""

    def __init__(self, config: AcousticConfig) -> None:
        config.validate()
        self.config = config

    # ------------------------------------------------------------------ #
    # Transmission loss
    # ------------------------------------------------------------------ #
    def transmission_loss_db(self, distance_m: float | np.ndarray) -> np.ndarray:
        """Spreading loss (per ``config.spreading_loss_exponent``) plus
        Thorp absorption loss, as a function of distance.
        """
        d = np.asarray(distance_m, dtype=np.float64)
        d_safe = np.maximum(d, 1e-3)  # avoid log(0) at zero range
        spreading_db = self.config.spreading_loss_exponent * 10.0 * np.log10(d_safe)
        absorption_db_per_km = thorp_absorption_db_per_km(self.config.frequency_hz)
        absorption_db = absorption_db_per_km * (d_safe / 1000.0)
        return spreading_db + absorption_db

    # ------------------------------------------------------------------ #
    # Ambient noise
    # ------------------------------------------------------------------ #
    def ambient_noise_level_db(self) -> float:
        """Total ambient noise *spectral density* [dB re 1 uPa^2/Hz],
        combining turbulence, shipping, wind, and thermal components via
        ``_combine_noise_db``.

        This is a per-Hz quantity — see ``total_in_band_noise_db`` for the
        broadband (bandwidth-integrated) noise power actually comparable to
        ``source_level_db`` in the link budget.
        """
        f_khz = self.config.frequency_hz / 1000.0
        shipping_activity = self.config.shipping_activity_factor
        wind_speed = self.config.wind_speed_mps

        nl_turbulence = 17.0 - 30.0 * math.log10(f_khz)
        nl_shipping = (
            40.0
            + 20.0 * (shipping_activity - 0.5)
            + 26.0 * math.log10(f_khz)
            - 60.0 * math.log10(f_khz + 0.03)
        )
        nl_wind = (
            50.0
            + 7.5 * math.sqrt(max(wind_speed, 0.0))
            + 20.0 * math.log10(f_khz)
            - 40.0 * math.log10(f_khz + 0.4)
        )
        nl_thermal = -15.0 + 20.0 * math.log10(f_khz)

        return _combine_noise_db(nl_turbulence, nl_shipping, nl_wind, nl_thermal)

    def total_in_band_noise_db(self) -> float:
        """Total in-band noise power [dB re 1 uPa^2], obtained by
        integrating the per-Hz noise spectral density
        (``ambient_noise_level_db``) over the receiver bandwidth:
        ``NL_spectral + 10*log10(bandwidth_hz)``.

        This conversion was previously missing entirely (see
        ``AcousticConfig``'s docstring for the resulting bug and its fix) —
        ``snr_db`` compared a broadband source level directly against the
        per-Hz spectral density with no bandwidth term, understating total
        noise by ``10*log10(bandwidth_hz)`` (~37 dB for the default 5 kHz
        bandwidth) and producing the unrealistic ~90 dB margins flagged
        since Phase 2.
        """
        return self.ambient_noise_level_db() + 10.0 * math.log10(self.config.bandwidth_hz)

    # ------------------------------------------------------------------ #
    # SNR / link quality
    # ------------------------------------------------------------------ #
    def snr_db(self, distance_m: float | np.ndarray) -> np.ndarray:
        """Received SNR [dB]: source level minus transmission loss minus
        total in-band noise power (see ``total_in_band_noise_db``).
        """
        tl = self.transmission_loss_db(distance_m)
        noise = self.total_in_band_noise_db()
        return self.config.source_level_db - tl - noise

    def is_link_viable(self, distance_m: float | np.ndarray) -> np.ndarray:
        """Boolean mask: True where SNR meets or exceeds
        ``config.snr_min_db``."""
        return self.snr_db(distance_m) >= self.config.snr_min_db

    def snr_margin_db(self, distance_m: float | np.ndarray) -> np.ndarray:
        """SNR minus the minimum viable SNR — positive means margin to
        spare, negative means the link is already below threshold. This is
        the quantity the geometric hazard cause (a later phase) is defined
        in terms of.
        """
        return self.snr_db(distance_m) - self.config.snr_min_db

    # ------------------------------------------------------------------ #
    # Throughput
    # ------------------------------------------------------------------ #
    def throughput_bps(
        self, distance_m: float | np.ndarray, bandwidth_hz: float | None = None
    ) -> np.ndarray:
        """Shannon-capacity throughput estimate [bits/s] at the given
        distance. Defaults to ``config.bandwidth_hz`` (the same bandwidth
        used in the SNR calculation) rather than an independent hardcoded
        value, so throughput and SNR stay dimensionally consistent with
        each other; an explicit override is still accepted for callers that
        need to explore a different receiver bandwidth.
        """
        bw = bandwidth_hz if bandwidth_hz is not None else self.config.bandwidth_hz
        if bw <= 0.0:
            raise ValueError(f"bandwidth_hz must be > 0, got {bw}")
        snr_db = self.snr_db(distance_m)
        snr_linear = np.power(10.0, snr_db / 10.0)
        snr_linear_nonneg = np.maximum(snr_linear, 0.0)
        return bw * np.log2(1.0 + snr_linear_nonneg)

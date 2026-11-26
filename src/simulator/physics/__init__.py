"""
PG-GTPP closed-form physics foundation (Phase 2).

Wave field, current field, and acoustic channel models — all pure, stateless,
closed-form functions of position/time/distance, recomputed on demand rather
than stored (see the storage-aware implementation architecture). No hazard
generation, no event generation, and no trajectory propagation lives here —
those consume these physics primitives in a later phase.
"""

from __future__ import annotations

from .acoustic_channel import AcousticChannel, thorp_absorption_db_per_km
from .current_field import CurrentField, CurrentLayer, sample_current_layers
from .wave_field import WaveField, WaveMode, sample_wave_modes

__all__ = [
    "AcousticChannel",
    "thorp_absorption_db_per_km",
    "CurrentField",
    "CurrentLayer",
    "sample_current_layers",
    "WaveField",
    "WaveMode",
    "sample_wave_modes",
]

"""
PG-GTPP simulator foundation package.

Phase 1 scope only: configuration, state schema, event schema, deterministic
randomness, mission initialization, storage-efficient trajectory persistence,
and validation utilities. No physics propagation, no hazard generation, no
graph construction, no modeling code lives here — those are later phases.
"""

from __future__ import annotations

SIMULATOR_VERSION: str = "0.1.0-phase1"

__all__ = ["SIMULATOR_VERSION"]

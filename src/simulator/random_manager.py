"""
random_manager.py

Deterministic, hierarchical random-number management for the PG-GTPP
simulator.

Design decisions
-----------------
1. There is exactly one source of entropy per dataset build: a single
   ``master_seed``. Every other seed used anywhere in the simulator —
   per-mission, per-physics-module, per-generation-stage — is *derived*
   deterministically from ``(master_seed, mission_id, stream_name)`` via a
   SHA-256 hash, never drawn independently and never stored per-stream. This
   is exactly the reproducibility framework's requirement: the entire corpus
   is regenerable from one number plus the mission index (which itself only
   needs to store ``mission_id`` — the mission seed is always recoverable by
   re-deriving it, so it is not a second source of truth to keep in sync).
2. Global mutable RNG state (``numpy.random.seed()``, ``random.seed()``) is
   never used anywhere in this module or expected to be used by its callers.
   Every RNG returned is an explicit ``numpy.random.Generator`` (PCG64)
   instance owned by the caller — this is what makes multiple concurrent
   generation workers (e.g., a multiprocessing pool generating missions in
   parallel) safe: there is no shared global state to race on.
3. "Streams" are just a second string mixed into the same hash function as
   the mission id, giving each generation sub-concern (node placement, wave
   coefficients, current coefficients, event thinning, ...) its own
   independent, non-overlapping, reproducible sequence — without needing to
   pre-register or store a list of stream names anywhere. Any new module can
   mint a new stream simply by calling
   ``get_numpy_rng(mission_id, "my_new_stream")`` with no changes to this
   file.
4. Torch support is implemented but kept an *optional* dependency: Phase 1's
   own code never requires torch (there is no model here), but the
   modeling/training phases will need reproducible per-mission torch
   generators for e.g. dropout or negative-sampling. Guarding the import
   means this module — and every other Phase 1 module that might transitively
   import it — works in environments without torch installed, while still
   providing full, real (non-placeholder) torch support when it is present.

Data structures
----------------
``RandomManager`` — the sole entry point. Holds only the master seed; carries
no other mutable state.

Storage implications
---------------------
None directly — this module produces RNG objects, not persisted data. Its
role in the storage story is indirect but essential: because every seed is
re-derivable, ``mission_index.parquet`` only needs to store ``mission_id``
(and, for convenience/debuggability, the *derived* ``mission_seed`` — stored
as a redundant, checkable cache, not a required input) rather than a large
table of independently-drawn seeds.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

_SEED_MASK_63BIT = (1 << 63) - 1


class RandomManager:
    """Deterministic seed derivation and reproducible RNG stream factory.

    Example
    -------
    >>> rm = RandomManager(master_seed=42)
    >>> rng_a = rm.get_numpy_rng("mission_0001", "placement")
    >>> rng_b = rm.get_numpy_rng("mission_0001", "wave")
    >>> rng_a is rng_b
    False
    >>> rm.derive_mission_seed("mission_0001") == rm.derive_mission_seed("mission_0001")
    True
    """

    def __init__(self, master_seed: int) -> None:
        if not isinstance(master_seed, int) or master_seed < 0:
            raise ValueError(f"master_seed must be a non-negative integer, got {master_seed!r}")
        self._master_seed = master_seed
        logger.debug("RandomManager initialized with master_seed=%d", master_seed)

    @property
    def master_seed(self) -> int:
        return self._master_seed

    # ------------------------------------------------------------------ #
    # Deterministic derivation
    # ------------------------------------------------------------------ #
    def _derive_seed(self, *parts: Any) -> int:
        """Derive a reproducible, well-mixed 63-bit non-negative integer seed
        from the master seed plus an arbitrary sequence of string/int parts.
        """
        payload = ":".join(str(p) for p in (self._master_seed, *parts))
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) & _SEED_MASK_63BIT

    def derive_mission_seed(self, mission_id: str) -> int:
        """The canonical, top-level seed identifying a mission.

        This value is safe to store redundantly in ``mission_index.parquet``
        for human/debug visibility, but is never required as an *input* to
        reproduce anything — it is always re-derivable from
        ``(master_seed, mission_id)``.
        """
        return self._derive_seed(mission_id)

    def derive_stream_seed(self, mission_id: str, stream: str) -> int:
        """Seed for one named generation sub-concern within a mission."""
        return self._derive_seed(mission_id, stream)

    # ------------------------------------------------------------------ #
    # NumPy RNG streams
    # ------------------------------------------------------------------ #
    def get_numpy_rng(self, mission_id: str, stream: str = "default") -> np.random.Generator:
        """Return a fresh, independent, reproducible NumPy Generator (PCG64)
        for the given mission and named stream.

        Calling this twice with the same arguments yields two Generator
        instances that will produce identical draw sequences (same seed) but
        are independent Python objects — callers must not assume identity,
        only reproducibility of the underlying stream.
        """
        seed = self.derive_stream_seed(mission_id, stream)
        return np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Torch RNG streams (optional dependency)
    # ------------------------------------------------------------------ #
    @staticmethod
    def torch_available() -> bool:
        return _TORCH_AVAILABLE

    def get_torch_generator(
        self, mission_id: str, stream: str = "default", device: str = "cpu"
    ) -> "torch.Generator":
        """Return a fresh, reproducible ``torch.Generator`` for the given
        mission and named stream.

        Raises
        ------
        RuntimeError
            If torch is not installed in the current environment. This is a
            real, actionable error (not a silent no-op) since a caller that
            reaches this method genuinely needs torch-backed randomness.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "RandomManager.get_torch_generator() requires torch to be installed. "
                "Phase 1 (simulator foundation) never calls this method itself; it is "
                "provided for later modeling/training phases."
            )
        seed = self.derive_stream_seed(mission_id, stream) % (2**63 - 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"RandomManager(master_seed={self._master_seed})"

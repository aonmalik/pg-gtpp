"""
qa/trajectory_query.py

Shared position/velocity interpolation utility for QA modules that need to
recompute hazards at arbitrary times across a full mission (not just within
a single generation bracket, unlike ``event_generator.py``'s internal
helpers, which only ever interpolate within the current bracket).

Design decisions
-----------------
1. Assumes a **regular** sample grid (``timestamps[k] == k / sample_rate``),
   which every trajectory ``TrajectoryGenerator`` produces — this lets the
   bracket containing an arbitrary query time be found by direct arithmetic
   (``floor(t * sample_rate)``) instead of a search, and lets the whole
   function vectorize over an array of query times at once.
2. Extrapolation within the located bracket is **exactly** the same
   linear-from-bracket-start-velocity rule ``event_generator.py`` uses
   internally. This consistency is deliberate and important: a QA check
   that recomputed positions a different way than the generator itself
   used would not actually be testing the generator's own implicit
   intensity — it would be testing a different, unrelated intensity that
   happens to look similar.

Storage implications
---------------------
None — pure recomputation from an already-loaded trajectory array.
"""

from __future__ import annotations

import numpy as np

_X_COL, _Y_COL, _Z_COL, _VX_COL, _VY_COL = 0, 1, 2, 3, 4


def interpolate_positions_batch(
    array: np.ndarray, v_idx: int, timestamps: np.ndarray, t_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized (x, y, z, vx, vy) at arbitrary query times ``t_query`` for
    a single node, found by locating each query time's containing bracket
    on the regular sample grid and linearly extrapolating from that
    bracket's start using its stored velocity.

    Returns arrays with the same shape as ``t_query``.
    """
    t_query = np.asarray(t_query, dtype=np.float64)
    sample_rate = 1.0 / (timestamps[1] - timestamps[0])
    t_idx = np.clip(np.floor(t_query * sample_rate).astype(np.int64), 0, len(timestamps) - 1)

    t0 = timestamps[t_idx]
    x0 = array[t_idx, v_idx, _X_COL]
    y0 = array[t_idx, v_idx, _Y_COL]
    z0 = array[t_idx, v_idx, _Z_COL]
    vx0 = array[t_idx, v_idx, _VX_COL]
    vy0 = array[t_idx, v_idx, _VY_COL]

    dt = t_query - t0
    x = x0 + vx0 * dt
    y = y0 + vy0 * dt
    return x, y, z0, vx0, vy0

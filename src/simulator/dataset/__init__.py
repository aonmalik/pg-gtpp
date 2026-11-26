"""
PG-GTPP dataset / tensor pipeline (Phase 5).

Converts already-generated simulator artifacts (a mission's node trajectory
array and event log, as produced by ``trajectory_generator.py`` and
``event_generator.py``) into the fixed-shape tensor batches the model
package consumes. No simulation happens here — this package is a pure
reader/reshaper of existing mission data, matching the "recompute, don't
store" principle: nothing produced here is persisted back to disk.
"""

from __future__ import annotations

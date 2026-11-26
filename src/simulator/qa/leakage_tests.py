"""
qa/leakage_tests.py

The shifted-window regeneration test: a mechanical (not statistical) proof
that event generation never uses trajectory information beyond the current
processing point.

Design decisions
-----------------
1. **Why truncate data, not mission duration.** An earlier design considered
   regenerating a mission with a shorter configured ``duration_s`` and
   comparing prefixes. This is wrong for this codebase specifically: the
   USV's patrol period is defined as a *fraction of the configured mission
   duration* (see ``trajectory_generator.py``), so a shorter-duration config
   produces genuinely different USV motion, not merely a prefix of the same
   motion. This test instead takes an **already-generated, full-length**
   trajectory array and slices it — the same underlying data, just less of
   it visible to a fresh ``EventGenerator.generate()`` call. This isolates
   exactly the property that matters (does event generation depend on
   samples beyond what it's given), with no confound from trajectory
   generation's own duration-dependent behavior.
2. **Why this should produce an exact match, not merely a statistical one.**
   ``RandomManager.get_numpy_rng(mission_id, "event_thinning")`` derives its
   seed from ``(master_seed, mission_id, stream_name)`` only — never from
   trajectory length. If ``event_generator.py`` truly never reads array rows
   beyond the bracket it is currently processing (the property this test
   exists to check), then a truncated run must consume the identical
   sequence of random draws, in the identical order, for every bracket up
   to the truncation point, and therefore must produce byte-identical
   events (excluding the arbitrary ``event_id``, which is uuid4-random by
   design and outside the reproducibility contract — see Phase 3's test
   suite for the same exclusion). Any mismatch is direct, mechanical
   evidence of a lookahead bug.
3. **Truncation must land on a bracket boundary.** Truncating mid-bracket
   would change the *last* bracket's own end time (``t1``), which is a
   legitimate difference in what's being asked of the generator, not a
   leakage signal — the test enforces truncation at an integer sample
   index for exactly this reason.

Data structures
----------------
``LeakageTestResult`` — pass/fail, the compared event counts, and (if
failed) the first mismatching pair, for direct debugging.

Storage implications
---------------------
None — operates entirely on already-generated, in-memory mission artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..event_generator import EventGenerator
from ..mission_config import MissionConfig
from ..mission_state import NodeType
from ..physics import AcousticChannel, CurrentField, WaveField
from ..random_manager import RandomManager


@dataclass
class LeakageTestResult:
    """Result of the shifted-window regeneration test."""

    passed: bool
    truncate_at_s: float
    n_events_in_window: int
    n_truncated_events: int
    first_mismatch: dict[str, Any] | None = field(default=None)

    def summary(self) -> str:
        verdict = "PASS (no leakage detected)" if self.passed else "FAIL (leakage detected)"
        base = (
            f"Shifted-window regeneration test: truncated at t={self.truncate_at_s:.1f}s, "
            f"{self.n_events_in_window} full-run events vs {self.n_truncated_events} "
            f"truncated-run events in the overlapping window -> {verdict}"
        )
        if not self.passed and self.first_mismatch is not None:
            base += f"\n  first mismatch: {self.first_mismatch}"
        return base


def _event_content(event) -> dict[str, Any]:
    """Comparable event content, excluding the arbitrary uuid4 event_id
    (see module docstring, point 2)."""
    return {k: v for k, v in event.to_record().items() if k != "event_id"}


def shifted_window_regeneration_test(
    config: MissionConfig,
    random_manager: RandomManager,
    mission_id: str,
    acoustic_channel: AcousticChannel,
    wave_field: WaveField,
    current_field: CurrentField,
    array: np.ndarray,
    node_ids: list[str],
    node_types: dict[str, NodeType],
    timestamps: np.ndarray,
    truncate_at_idx: int,
) -> LeakageTestResult:
    """Run the shifted-window regeneration test.

    Generates events once on the full trajectory, then again on a
    ``array[:truncate_at_idx+1]`` slice of the *same* trajectory, and checks
    that every full-run event with ``timestamp_s < timestamps[truncate_at_idx]``
    is reproduced exactly by the truncated run.
    """
    if not (1 <= truncate_at_idx < len(timestamps) - 1):
        raise ValueError(
            f"truncate_at_idx must be in [1, {len(timestamps) - 2}] to leave both a non-trivial "
            f"truncated window and at least one bracket after it in the full run, got {truncate_at_idx}"
        )

    full_generator = EventGenerator(config, random_manager, acoustic_channel, wave_field, current_field)
    full_events = full_generator.generate(mission_id, array, node_ids, node_types, timestamps)

    truncated_array = array[: truncate_at_idx + 1]
    truncated_timestamps = timestamps[: truncate_at_idx + 1]
    truncated_generator = EventGenerator(config, random_manager, acoustic_channel, wave_field, current_field)
    truncated_events = truncated_generator.generate(
        mission_id, truncated_array, node_ids, node_types, truncated_timestamps
    )

    cutoff_time = float(timestamps[truncate_at_idx])
    full_events_in_window = sorted(
        (e for e in full_events if e.timestamp_s < cutoff_time),
        key=lambda e: (e.timestamp_s, e.edge_i, e.edge_j),
    )
    truncated_events_sorted = sorted(truncated_events, key=lambda e: (e.timestamp_s, e.edge_i, e.edge_j))

    full_content = [_event_content(e) for e in full_events_in_window]
    truncated_content = [_event_content(e) for e in truncated_events_sorted]

    passed = full_content == truncated_content
    first_mismatch = None
    if not passed:
        for i, (a, b) in enumerate(zip(full_content, truncated_content)):
            if a != b:
                first_mismatch = {"index": i, "full_run": a, "truncated_run": b}
                break
        else:
            first_mismatch = {
                "index": min(len(full_content), len(truncated_content)),
                "note": "event count mismatch beyond the shared prefix",
                "n_full": len(full_content),
                "n_truncated": len(truncated_content),
            }

    return LeakageTestResult(
        passed=passed,
        truncate_at_s=cutoff_time,
        n_events_in_window=len(full_content),
        n_truncated_events=len(truncated_content),
        first_mismatch=first_mismatch,
    )

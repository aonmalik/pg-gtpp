"""
scripts/benchmark_event_generator.py

Standalone benchmark for the vectorized event generator. Runs the full
generate -> propagate -> event-generation pipeline on a default-config,
full-length (1800s) mission and reports timing and event count.

Run with:
    python scripts/benchmark_event_generator.py

Context: the original per-edge-sequential event generator profiled at
~85s for a 27-node, 1800s mission. The current vectorized implementation
(batched hazard evaluation across all candidate edges per thinning round,
plus a flattened probe grid) profiles at ~5s for the same mission — a
~17x speedup, achieved through two rounds of profiling-guided optimization
documented in event_generator.py's module docstring.
"""

from __future__ import annotations

import time

import numpy as np

from simulator.event_generator import EventGenerator
from simulator.mission_config import default_config
from simulator.mission_generator import MissionGenerator
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator


def main() -> None:
    config = default_config(master_seed=11)
    rm = RandomManager(master_seed=11)
    mission_id = "benchmark_mission"

    t0 = time.time()
    state = MissionGenerator(config, rm).generate(mission_id)
    array, node_ids = TrajectoryGenerator(config, rm).propagate(state)
    print(
        f"[benchmark] mission: {state.metadata.n_auv} AUV, {state.metadata.m_iout} IoUT, "
        f"{config.envelope.duration_s:.0f}s duration"
    )
    print(f"[benchmark] generation + trajectory propagation: {time.time() - t0:.2f}s")

    wf = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
    cf = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
    ch = AcousticChannel(config.acoustic)
    node_types = {n.node_id: n.node_type for n in state.nodes}
    timestamps = np.arange(array.shape[0], dtype=np.float64) / config.envelope.node_sample_rate_hz

    generator = EventGenerator(config, rm, ch, wf, cf)
    t0 = time.time()
    events = generator.generate(mission_id, array, node_ids, node_types, timestamps)
    elapsed = time.time() - t0

    print(f"[benchmark] event generation: {len(events)} events in {elapsed:.2f}s")
    print(f"[benchmark] projected wall time for 1000 such missions, single-threaded: {elapsed * 1000 / 60:.1f} min")
    print(f"[benchmark] projected wall time for 1000 such missions, 8-way parallel: {elapsed * 1000 / 60 / 8:.1f} min")


if __name__ == "__main__":
    main()

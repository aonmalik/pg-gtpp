"""
scripts/certify_mission_demo.py

Standalone demonstration of the full Phase 4 QA pipeline: generates one
full-length, default-config mission and runs all three checks against it,
printing the combined certification report.

Run with:
    python scripts/certify_mission_demo.py
"""

from __future__ import annotations

import numpy as np

from simulator.event_generator import EventGenerator
from simulator.mission_config import default_config
from simulator.mission_generator import MissionGenerator
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.physics_intensity import HazardEvaluator
from simulator.qa.certification_report import certify_mission
from simulator.qa.identifiability_audit import audit_identifiability
from simulator.qa.leakage_tests import shifted_window_regeneration_test
from simulator.qa.time_rescaling_test import run_time_rescaling_test
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator


def main() -> None:
    config = default_config(master_seed=11)
    rm = RandomManager(master_seed=11)
    mission_id = "certification_demo_mission"

    state = MissionGenerator(config, rm).generate(mission_id)
    array, node_ids = TrajectoryGenerator(config, rm).propagate(state)
    wf = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
    cf = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
    ch = AcousticChannel(config.acoustic)
    node_types = {n.node_id: n.node_type for n in state.nodes}
    timestamps = np.arange(array.shape[0], dtype=np.float64) / config.envelope.node_sample_rate_hz

    events = EventGenerator(config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)
    print(f"Generated {len(events)} events for mission {mission_id} "
          f"({state.metadata.n_auv} AUV, {state.metadata.m_iout} IoUT, {config.envelope.duration_s:.0f}s)\n")

    evaluator = HazardEvaluator(ch, wf, cf, config.hazard)
    time_rescaling = run_time_rescaling_test(evaluator, array, node_ids, timestamps, events)
    identifiability = audit_identifiability(evaluator, array, node_ids, timestamps, events)

    truncate_idx = int(0.6 * (len(timestamps) - 1))
    leakage = shifted_window_regeneration_test(
        config, rm, mission_id, ch, wf, cf, array, node_ids, node_types, timestamps, truncate_idx
    )

    report = certify_mission(time_rescaling, identifiability, leakage)
    print(report.summary())


if __name__ == "__main__":
    main()

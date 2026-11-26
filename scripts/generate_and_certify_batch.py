"""
scripts/generate_and_certify_batch.py

Generates a small corpus of missions under the default (recalibrated)
config, writes each to disk via the storage-efficient trajectory format,
runs the full QA certification pipeline on each mission, and reports both
the corpus-level certification and the real, measured storage footprint —
the first time the certification framework has been run across more than
one hand-picked mission at a time.

Run with:
    python scripts/generate_and_certify_batch.py [N]

N defaults to 30 if not given. Expect roughly 2-5 seconds per mission
(varies with drawn fleet size), so the default run takes a few minutes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from simulator.event_generator import EventGenerator
from simulator.mission_config import default_config
from simulator.mission_generator import MissionGenerator
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.physics_intensity import HazardEvaluator
from simulator.qa.certification_report import certify_mission
from simulator.qa.corpus_certification import certify_corpus
from simulator.qa.identifiability_audit import audit_identifiability
from simulator.qa.leakage_tests import shifted_window_regeneration_test
from simulator.qa.time_rescaling_test import run_time_rescaling_test
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator
from simulator.trajectory_storage import TrajectoryWriter
from simulator.validation_utils import validate_storage_budget

DEFAULT_N_MISSIONS = 30
MASTER_SEED = 2024
OUTPUT_DIR = Path("qa_batch_output")


def main() -> int:
    n_missions = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_MISSIONS
    print(f"Generating and certifying {n_missions} missions (master_seed={MASTER_SEED})...")
    print(f"Writing to {OUTPUT_DIR.resolve()}\n")

    config = default_config(master_seed=MASTER_SEED)
    rm = RandomManager(master_seed=MASTER_SEED)
    writer = TrajectoryWriter(OUTPUT_DIR)

    mission_reports = []
    event_counts: dict[str, int] = {}
    t_start = time.time()

    for i in range(n_missions):
        mission_id = f"batch_{i:04d}"
        t0 = time.time()

        state = MissionGenerator(config, rm).generate(mission_id)
        array, node_ids = TrajectoryGenerator(config, rm).propagate(state)
        wf = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
        cf = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
        ch = AcousticChannel(config.acoustic)
        node_types = {n.node_id: n.node_type for n in state.nodes}
        timestamps = np.arange(array.shape[0], dtype=np.float64) / config.envelope.node_sample_rate_hz

        events = EventGenerator(config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)
        writer.write_mission(state, events, array=array, node_ids=node_ids)

        evaluator = HazardEvaluator(ch, wf, cf, config.hazard)
        time_rescaling = run_time_rescaling_test(evaluator, array, node_ids, timestamps, events)
        identifiability = audit_identifiability(evaluator, array, node_ids, timestamps, events)
        truncate_idx = int(0.6 * (len(timestamps) - 1))
        leakage = shifted_window_regeneration_test(
            config, rm, mission_id, ch, wf, cf, array, node_ids, node_types, timestamps, truncate_idx
        )
        report = certify_mission(time_rescaling, identifiability, leakage)

        mission_reports.append((mission_id, report))
        event_counts[mission_id] = len(events)

        elapsed = time.time() - t0
        print(
            f"  [{i + 1}/{n_missions}] {mission_id}: {state.metadata.n_auv} AUV, "
            f"{state.metadata.m_iout} IoUT, {len(events)} events, "
            f"verdict={report.verdict.value.upper()} ({elapsed:.1f}s)"
        )

    total_elapsed = time.time() - t_start
    print(f"\nGenerated and certified {n_missions} missions in {total_elapsed:.1f}s "
          f"({total_elapsed / n_missions:.2f}s/mission average)\n")

    corpus_result = certify_corpus(mission_reports, event_counts)
    print(corpus_result.summary())

    print()
    storage_result = validate_storage_budget(OUTPUT_DIR, preferred_limit_gb=30.0, hard_limit_gb=50.0)
    print(storage_result.summary())
    if storage_result.file_count > 0:
        avg_bytes_per_mission = storage_result.total_bytes / n_missions
        print(f"Average storage per mission: {avg_bytes_per_mission / 1024:.1f} KB "
              f"(projected for 10,000 missions: {avg_bytes_per_mission * 10000 / (1024**3):.2f} GB)")

    return 0 if corpus_result.verdict.value != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())

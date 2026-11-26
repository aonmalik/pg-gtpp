"""
scripts/smoke_test.py

Standalone, dependency-light smoke test for the Phase 1 simulator
foundation. Exercises the full pipeline — config -> RNG -> mission
generation -> storage write -> lazy reload -> storage-budget check — and
prints a human-readable pass/fail summary.

Run with:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from simulator.mission_config import default_config
from simulator.mission_generator import MissionGenerator
from simulator.random_manager import RandomManager
from simulator.trajectory_storage import MissionDataStore, TrajectoryWriter
from simulator.validation_utils import validate_mission_state, validate_storage_budget


def main() -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="pg_gtpp_smoke_"))
    print(f"[smoke_test] writing to temporary directory: {work_dir}")

    try:
        config = default_config(master_seed=7, split="train")
        rm = RandomManager(master_seed=7)
        generator = MissionGenerator(config, rm)
        writer = TrajectoryWriter(work_dir)

        mission_ids = [f"mission_{i:04d}" for i in range(15)]
        for mission_id in mission_ids:
            state = generator.generate(mission_id)
            validate_mission_state(state, config)
            writer.write_mission(state, events=None)
        print(f"[smoke_test] generated and wrote {len(mission_ids)} missions")

        store = MissionDataStore(work_dir)
        assert len(store) == len(mission_ids), "mission index count mismatch"

        metadata, array, node_ids, events = store.load_mission("mission_0007")
        assert array.ndim == 3, "node trajectory array must be 3-D [T, |V|, F_node]"
        assert array.shape[1] == len(node_ids), "node count mismatch between array and node_ids"
        print(
            f"[smoke_test] reloaded mission_0007: {metadata.n_auv} AUV, "
            f"{metadata.m_iout} IoUT, array shape={array.shape}, dtype={array.dtype}"
        )

        # reproducibility check: a second generator with the same master seed
        # must produce an identical mission (excluding the wall-clock timestamp)
        state_a = MissionGenerator(config, RandomManager(master_seed=7)).generate("repro_check")
        state_b = MissionGenerator(config, RandomManager(master_seed=7)).generate("repro_check")
        da, db = state_a.to_dict(), state_b.to_dict()
        da["metadata"].pop("created_at")
        db["metadata"].pop("created_at")
        assert da == db, "reproducibility check failed: same master_seed produced different missions"
        print("[smoke_test] reproducibility check passed")

        result = validate_storage_budget(work_dir, preferred_limit_gb=30.0, hard_limit_gb=50.0)
        print(f"[smoke_test] {result.summary()}")
        assert result.status.value == "pass", f"unexpected storage status: {result.status.value}"

        print("\n[smoke_test] ALL CHECKS PASSED")
        return 0

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

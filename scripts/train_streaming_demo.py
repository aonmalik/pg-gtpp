"""
scripts/train_streaming_demo.py

Demonstrates the corpus-scale, disk-backed training workflow: generates a
mission corpus and writes it to disk (via TrajectoryWriter, the same
storage-efficient format every other phase uses), then trains a
PGGTPPModel against it via StreamingMissionSource -- never holding more
than a small, bounded number of missions in memory at once, unlike
scripts/train_demo.py's in-memory MissionCache.

Run with:
    python scripts/train_streaming_demo.py [N_MISSIONS]

N_MISSIONS defaults to 12 if not given.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from simulator.event_generator import EventGenerator
from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.mission_generator import MissionGenerator
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator
from simulator.trajectory_storage import MissionDataStore, TrajectoryWriter
from simulator.training.streaming_dataset import StreamingMissionSource
from simulator.training.train import TrainConfig, train

MODEL_INIT_SEED = 1234
DATA_MASTER_SEED = 7
DEFAULT_N_MISSIONS = 12


def generate_corpus_to_disk(config: MissionConfig, master_seed: int, n_missions: int, output_dir: Path) -> None:
    rm = RandomManager(master_seed=master_seed)
    writer = TrajectoryWriter(output_dir)
    for i in range(n_missions):
        mission_id = f"demo_mission_{i:04d}"
        state = MissionGenerator(config, rm).generate(mission_id)
        array, node_ids = TrajectoryGenerator(config, rm).propagate(state)
        wf = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
        cf = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
        ch = AcousticChannel(config.acoustic)
        node_types = {n.node_id: n.node_type for n in state.nodes}
        timestamps = np.arange(array.shape[0], dtype=np.float64) / config.envelope.node_sample_rate_hz
        events = EventGenerator(config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)
        writer.write_mission(state, events, array=array, node_ids=node_ids)
        print(f"  [{i + 1}/{n_missions}] {mission_id}: {len(node_ids)} nodes, {len(events)} events")


def main() -> None:
    n_missions = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_MISSIONS

    base = default_config(master_seed=DATA_MASTER_SEED)
    fleet = FleetConfig(
        usv_count=1, n_auv_min=3, n_auv_max=6, m_iout_min=8, m_iout_max=15,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    envelope = MissionEnvelopeConfig(
        duration_s=400.0, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m, depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    config = MissionConfig(
        master_seed=DATA_MASTER_SEED, split="train", wave=base.wave, current=base.current,
        acoustic=base.acoustic, fleet=fleet, envelope=envelope,
        hazard=base.hazard, graph=base.graph,
    )

    work_dir = Path(tempfile.mkdtemp(prefix="pg_gtpp_streaming_demo_"))
    print(f"Generating {n_missions}-mission corpus to disk at {work_dir}...")
    try:
        generate_corpus_to_disk(config, DATA_MASTER_SEED, n_missions, work_dir)

        store = MissionDataStore(work_dir)
        print(f"\nCorpus written: {len(store)} missions on disk.")

        source = StreamingMissionSource(store, config, master_seed=DATA_MASTER_SEED, cache_size=4)
        print(f"StreamingMissionSource cache_size=4 (only 4 missions' arrays held in memory at once, "
              f"out of {len(source.mission_ids)} available on disk)\n")

        train_src, val_src = source.split(val_fraction=0.25, seed=0)
        print(f"Split: {len(train_src.mission_ids)} train / {len(val_src.mission_ids)} val missions\n")

        torch.manual_seed(MODEL_INIT_SEED)
        model = PGGTPPModel(f_node=11, hidden_dim=64)
        print(f"Model parameter count: {model.parameter_count():,}\n")

        train_config = TrainConfig(
            n_steps=150, batch_size=16, window_length=10, lookahead_s=30.0,
            learning_rate=5e-3, log_every=25, seed=1,
            lr_schedule="cosine", val_every=25, val_stride=10,
        )
        print("Training against the disk-backed streaming source...")
        history = train(model, train_src, train_config, val_cache=val_src)

        first_avg = float(np.mean(history.total_losses[:10]))
        last_avg = float(np.mean(history.total_losses[-10:]))
        print(f"\nTrain loss: first-10 avg={first_avg:.4f} -> last-10 avg={last_avg:.4f} "
              f"({'decreased' if last_avg < first_avg else 'DID NOT DECREASE'})")
        print(f"Final validation: loss={history.val_losses[-1]:.4f}  AUC={history.val_event_aucs[-1]:.3f}  "
              f"calibration_error={history.val_calibration_errors[-1]:.3f}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"\n(cleaned up temporary corpus directory {work_dir})")


if __name__ == "__main__":
    main()

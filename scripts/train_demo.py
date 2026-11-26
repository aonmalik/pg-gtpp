"""
scripts/train_demo.py

Standalone demonstration of the Phase 6 training pipeline: generates a
small real mission corpus, splits it into train/val AT THE MISSION LEVEL
(see MissionCache.split's docstring for why window-level splitting would
leak), trains a PGGTPPModel with a cosine LR schedule, periodically
evaluates on the genuinely held-out validation missions, and prints a
final summary plus a few sample predictions.

Run with:
    python scripts/train_demo.py
"""

from __future__ import annotations

import numpy as np
import torch

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.model.fusion import CAUSE_NAMES
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.training.train import MissionCache, TrainConfig, train

MODEL_INIT_SEED = 1234  # separate from the data-generation master_seed, but fixed and documented so
# demo runs are fully reproducible end to end -- MissionCache.generate() already seeds all data
# generation via RandomManager, but PGGTPPModel's weight initialization uses PyTorch's own global
# RNG, which is otherwise unseeded and was silently making each demo run start from different
# random weights (same data, different initial parameters, different training trajectory in
# weight-space even when total loss ends up similar).


def main() -> None:
    base = default_config(master_seed=7)
    fleet = FleetConfig(
        usv_count=1, n_auv_min=3, n_auv_max=6, m_iout_min=8, m_iout_max=15,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    envelope = MissionEnvelopeConfig(
        duration_s=600.0, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m, depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    config = MissionConfig(
        master_seed=7, split="train", wave=base.wave, current=base.current,
        acoustic=base.acoustic, fleet=fleet, envelope=envelope,
        hazard=base.hazard, graph=base.graph,
    )

    print("Generating a small mission corpus (10 missions)...")
    cache = MissionCache.generate(config, master_seed=7, n_missions=10)
    total_events = sum(len(m.events) for m in cache.missions)
    print(f"  {len(cache.missions)} missions, {total_events} total events")

    train_cache, val_cache = cache.split(val_fraction=0.3, seed=0)
    print(f"  split (mission-level): {len(train_cache.missions)} train / {len(val_cache.missions)} val\n")

    torch.manual_seed(MODEL_INIT_SEED)
    model = PGGTPPModel(f_node=11, hidden_dim=64)
    print(f"Model parameter count: {model.parameter_count():,}\n")

    train_config = TrainConfig(
        n_steps=150, batch_size=16, window_length=10, lookahead_s=30.0,
        learning_rate=5e-3, log_every=25, seed=1,
        lr_schedule="cosine", val_every=25, val_stride=10,
    )
    print("Training (with cosine LR schedule and periodic validation on held-out missions)...")
    history = train(model, train_cache, train_config, val_cache=val_cache)

    first_avg = float(np.mean(history.total_losses[:10]))
    last_avg = float(np.mean(history.total_losses[-10:]))
    print(f"\nTrain loss: first-10 avg={first_avg:.4f} -> last-10 avg={last_avg:.4f} "
          f"({'decreased' if last_avg < first_avg else 'DID NOT DECREASE'})")
    print(f"Final validation: loss={history.val_losses[-1]:.4f}  "
          f"AUC={history.val_event_aucs[-1]:.3f} (0.5=chance, 1.0=perfect)  "
          f"calibration_error={history.val_calibration_errors[-1]:.3f}")
    print(
        "NOTE: at this toy scale (a handful of missions, ~150 steps), validation AUC "
        "typically stays close to the 0.5 chance baseline even as training loss decreases "
        "-- this is expected, not a bug; see the Phase 6 delivery notes on when this setup "
        "graduates to a real experiment (more data, more capacity, a GPU)."
    )

    print("\nSample predictions on a genuinely held-out validation window:")
    model.eval()
    rng = np.random.default_rng(999)
    batch = val_cache.sample_batch(rng, batch_size=1, window_length=10, lookahead_s=30.0)

    with torch.no_grad():
        out = model(batch.node_features, batch.node_type_ids, batch.node_mask, batch.edge_index, batch.edge_physics)
        total_intensity = out.total_intensity()
        for i in range(min(5, total_intensity.shape[0])):
            gates = {name: round(out.fusion[name]["gate"][i].item(), 2) for name in CAUSE_NAMES}
            print(
                f"  edge {i}: total_intensity={total_intensity[i].item():.4f}  "
                f"pdr_mean={out.marks.pdr_mean()[i].item():.3f}  trust_gates={gates}"
            )


if __name__ == "__main__":
    main()

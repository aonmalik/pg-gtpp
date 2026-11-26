"""
Controlled physics-mismatch benchmark for PG-GTPP.

Purpose
-------
The ordinary simulator is intentionally generated from the same closed-form
physics hazard used by the physics-only baseline.  That is a useful
well-specified control, but it cannot strongly test whether PG-GTPP can learn
useful corrections when the physical prior is imperfect.

This benchmark keeps trajectories, node features, acoustic/wave/current
fields, mission splits, windows, optimizer and seeds fixed.  Only the
*ground-truth event generator* is perturbed by cause-specific hazard
multipliers.  The model still receives the original, unperturbed physics
hazards.  Therefore any improvement of PG-GTPP over physics-only is evidence
that the learned correction is responding to information not captured by the
misspecified physics prior.

The benchmark also includes a genuinely physics-free neural ablation:
``neural_only`` disables both direct physics-intensity fusion and the
physics-biased graph-attention term.

Usage example
-------------
PYTHONPATH=src python scripts/run_physics_mismatch_benchmark.py \
  --n-missions 50 --n-steps 1000 --batch-size 8 \
  --window-length 10 --lookahead-s 30 \
  --mismatch-levels 0,0.05,0.10,0.20,0.30,0.50 \
  --seeds 7,21,42,84,123 \
  --output-dir results/physics_mismatch
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.training.train import MissionCache, TrainConfig, train
from simulator.training.eval import evaluate

F_NODE = 11
MODEL_NAMES = ("physics_only", "neural_only", "pg_gtpp")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(args: argparse.Namespace) -> MissionConfig:
    base = default_config(master_seed=args.corpus_seed)
    fleet = FleetConfig(
        usv_count=1, n_auv_min=3, n_auv_max=8,
        m_iout_min=10, m_iout_max=30,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    envelope = MissionEnvelopeConfig(
        duration_s=args.mission_duration_s,
        node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m,
        depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    return MissionConfig(
        master_seed=args.corpus_seed, split="train",
        wave=base.wave, current=base.current, acoustic=base.acoustic,
        fleet=fleet, envelope=envelope, hazard=base.hazard, graph=base.graph,
    )


def split_three(cache: MissionCache, seed: int):
    train_cache, holdout = cache.split(0.30, seed)
    val_cache, test_cache = holdout.split(0.50, seed + 1)
    return train_cache, val_cache, test_cache


def mismatch_multipliers(level: float) -> tuple[float, float, float, float]:
    """Structured, cause-specific misspecification.

    At level m, the true hazard is changed by:
      geo     1 + m
      wave    1 - 0.5m
      doppler 1 + 0.5m
      current 1 - m/3

    Thus the benchmark contains both under- and over-predicted causes rather
    than merely multiplying the entire total hazard by one constant.
    """
    m = float(level)
    if not (0.0 <= m < 1.0):
        raise ValueError("mismatch level must satisfy 0 <= level < 1")
    return (1.0 + m, 1.0 - 0.5 * m, 1.0 + 0.5 * m, 1.0 - m / 3.0)


def build_model(name: str, hidden_dim: int) -> PGGTPPModel:
    if name == "physics_only":
        return PGGTPPModel(
            f_node=F_NODE, hidden_dim=hidden_dim,
            learned_correction=False,
            use_physics_intensity=True,
            use_fusion_gate=False,
            use_physics_attention=True,
        )
    if name == "neural_only":
        return PGGTPPModel(
            f_node=F_NODE, hidden_dim=hidden_dim,
            learned_correction=True,
            use_physics_intensity=False,
            use_fusion_gate=False,
            use_physics_attention=False,
        )
    if name == "pg_gtpp":
        return PGGTPPModel(
            f_node=F_NODE, hidden_dim=hidden_dim,
            learned_correction=True,
            use_physics_intensity=True,
            use_fusion_gate=True,
            use_physics_attention=True,
        )
    raise ValueError(f"unknown model: {name}")


def train_one(name: str, seed: int, train_cache: MissionCache, args: argparse.Namespace):
    seed_everything(seed)
    model = build_model(name, args.hidden_dim)
    cfg = TrainConfig(
        n_steps=args.n_steps, batch_size=args.batch_size,
        window_length=args.window_length, lookahead_s=args.lookahead_s,
        learning_rate=args.learning_rate, mark_loss_weight=args.mark_loss_weight,
        log_every=args.log_every, checkpoint_every=None, seed=seed,
        lr_schedule=args.lr_schedule,
    )
    train(model, train_cache, cfg)
    return model


def finite_aggregate(rows: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in rows if math.isfinite(float(r[key]))]
    return {
        "mean": mean(vals) if vals else float("nan"),
        "std": stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def metric_aggregate(rows: list[dict]) -> dict:
    keys = [
        "event_auc", "event_auprc", "brier_score", "integrated_brier_score",
        "calibration_error", "survival_calibration_error", "c_index",
        "failure_time_mae_s", "pdr_mae", "pdr_rmse", "throughput_mae",
        "throughput_rmse", "delay_mae", "delay_rmse", "cause_macro_f1",
        "cause_brier",
    ]
    return {k: finite_aggregate(rows, k) for k in keys}


def paired_bootstrap_delta(rows_a: list[dict], rows_b: list[dict], key: str, n_boot: int, seed: int):
    """95% bootstrap CI for paired per-seed metric differences."""
    a = np.asarray([float(r[key]) for r in rows_a], dtype=float)
    b = np.asarray([float(r[key]) for r in rows_b], dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    if d.size == 0:
        return {"mean_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = rng.choice(d, size=d.size, replace=True).mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "mean_delta": float(d.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n": int(d.size),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-missions", type=int, default=50)
    p.add_argument("--corpus-seed", type=int, default=2026)
    p.add_argument("--seeds", type=str, default="7,21,42,84,123")
    p.add_argument("--mismatch-levels", type=str, default="0,0.05,0.10,0.20,0.30,0.50")
    p.add_argument("--mission-duration-s", type=float, default=900.0)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--window-length", type=int, default=10)
    p.add_argument("--lookahead-s", type=float, default=30.0)
    p.add_argument("--mark-loss-weight", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--eval-stride", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--output-dir", type=str, default="results/physics_mismatch")
    p.add_argument("--allow-nondeterministic", action="store_true")
    args = p.parse_args()

    if not args.allow_nondeterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    levels = [float(x.strip()) for x in args.mismatch_levels.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not levels or not seeds:
        raise ValueError("--mismatch-levels and --seeds must not be empty")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every condition regenerates the same deterministic trajectories/fields
    # from (corpus_seed, mission_id). Only the event-generating hazard is
    # changed, so the physical covariates and mission split stay fixed.
    config = make_config(args)
    print(f"Fixed protocol: train/val/test = {int(args.n_missions*0.70)}/{int(args.n_missions*0.15)}/{args.n_missions-int(args.n_missions*0.70)-int(args.n_missions*0.15)}")

    summary = {
        "protocol": {
            "corpus_seed": args.corpus_seed,
            "n_missions": args.n_missions,
            "split": "70/15/15 mission-level",
            "seeds": seeds,
            "mismatch_levels": levels,
            "mismatch_definition": "truth causes scaled as geo=1+m, wave=1-0.5m, doppler=1+0.5m, current=1-m/3; model-visible physics remains unscaled",
            "models": list(MODEL_NAMES),
            "neural_only_definition": "no direct physics intensity and no physics-biased graph attention",
        },
        "conditions": {},
    }

    for level in levels:
        multipliers = mismatch_multipliers(level)
        condition_name = f"mismatch_{level:g}"
        condition_dir = out_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*72}\nCONDITION {condition_name}: truth multipliers={multipliers}\n{'='*72}")

        # The important separation: the stored evaluator continues to expose
        # base physics to the model, while events are generated using the
        # mismatched truth hazard.
        cache = MissionCache.generate(
            config, args.corpus_seed, args.n_missions,
            truth_hazard_multipliers=multipliers,
        )
        train_cache, val_cache, test_cache = split_three(cache, seed=0)
        val_batch = val_cache.all_windows(args.window_length, args.lookahead_s, stride=args.eval_stride)
        test_batch = test_cache.all_windows(args.window_length, args.lookahead_s, stride=args.eval_stride)

        condition = {
            "truth_multipliers": multipliers,
            "models": {},
        }

        trained = {}
        for name in MODEL_NAMES:
            rows = []
            val_rows = []
            print(f"\n--- {name} ---")
            for seed in seeds:
                print(f"seed {seed}")
                model = train_one(name, seed, train_cache, args)
                val = evaluate(model, val_batch, mark_loss_weight=args.mark_loss_weight).to_dict()
                test = evaluate(model, test_batch, mark_loss_weight=args.mark_loss_weight).to_dict()
                rows.append(test)
                val_rows.append(val)
                trained[(name, seed)] = model
                print(
                    f"  test AUROC={test['event_auc']:.4f} "
                    f"AUPRC={test['event_auprc']:.4f} "
                    f"Brier={test['brier_score']:.5f} "
                    f"IBS={test['integrated_brier_score']:.5f} "
                    f"C={test['c_index']:.4f}"
                )
            condition["models"][name] = {
                "parameters": build_model(name, args.hidden_dim).parameter_count(),
                "per_seed_test": rows,
                "per_seed_validation": val_rows,
                "test_aggregate": metric_aggregate(rows),
            }

        pg_rows = condition["models"]["pg_gtpp"]["per_seed_test"]
        phys_rows = condition["models"]["physics_only"]["per_seed_test"]
        neural_rows = condition["models"]["neural_only"]["per_seed_test"]
        condition["paired_deltas"] = {
            "pg_minus_physics": {
                k: paired_bootstrap_delta(pg_rows, phys_rows, k, args.bootstrap, args.corpus_seed + int(level*10000) + 1)
                for k in ["event_auc", "event_auprc", "brier_score", "integrated_brier_score", "c_index"]
            },
            "pg_minus_neural": {
                k: paired_bootstrap_delta(pg_rows, neural_rows, k, args.bootstrap, args.corpus_seed + int(level*10000) + 2)
                for k in ["event_auc", "event_auprc", "brier_score", "integrated_brier_score", "c_index"]
            },
        }
        atomic = condition_dir / "summary.json"
        atomic.write_text(json.dumps(condition, indent=2, allow_nan=True))
        summary["conditions"][condition_name] = condition

        pg_auc = condition["models"]["pg_gtpp"]["test_aggregate"]["event_auc"]["mean"]
        ph_auc = condition["models"]["physics_only"]["test_aggregate"]["event_auc"]["mean"]
        print(f"\nRESULT {condition_name}: PG-GTPP AUROC={pg_auc:.4f}, Physics-only AUROC={ph_auc:.4f}, delta={pg_auc-ph_auc:+.4f}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

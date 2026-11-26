"""Benchmark PG-GTPP with a persistent latent operational communication state.

Truth hazard:
    lambda_true(e,t) = lambda_physics(e,t) * exp(strength * q_e(t))

q_e(t) is a persistent, truth-only operational-quality state updated once per
1-Hz trajectory bracket from observable battery/FIM/motion stress plus a small
AR(1)-style innovation.  The realized q is never stored in the trajectory and
is never passed to HazardEvaluator.  Neural models can only infer it from their
history; physics-only cannot use it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.training.train import MissionCache, TrainConfig, train
from simulator.training.eval import evaluate

F_NODE = 11
MODELS = ("physics_only", "neural_only", "pg_gtpp", "pg_gtpp_no_gate", "pg_gtpp_temporal")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(args):
    base = default_config(master_seed=args.corpus_seed)
    fleet = FleetConfig(
        usv_count=1, n_auv_min=3, n_auv_max=8,
        m_iout_min=10, m_iout_max=30,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    envelope = MissionEnvelopeConfig(
        duration_s=args.mission_duration_s, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m,
        depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    return MissionConfig(
        master_seed=args.corpus_seed, split="train",
        wave=base.wave, current=base.current, acoustic=base.acoustic,
        fleet=fleet, envelope=envelope, hazard=base.hazard, graph=base.graph,
    )


def split_three(cache, seed=0):
    train_cache, holdout = cache.split(0.30, seed)
    val_cache, test_cache = holdout.split(0.50, seed + 1)
    return train_cache, val_cache, test_cache


def build_model(name, hidden_dim):
    if name == "physics_only":
        return PGGTPPModel(f_node=F_NODE, hidden_dim=hidden_dim,
            learned_correction=False, use_physics_intensity=True,
            use_fusion_gate=False, use_physics_attention=True)
    if name == "neural_only":
        return PGGTPPModel(f_node=F_NODE, hidden_dim=hidden_dim,
            learned_correction=True, use_physics_intensity=False,
            use_fusion_gate=False, use_physics_attention=False)
    if name == "pg_gtpp_no_gate":
        return PGGTPPModel(f_node=F_NODE, hidden_dim=hidden_dim,
            learned_correction=True, use_physics_intensity=True,
            use_fusion_gate=False, fixed_fusion_gate=0.0, use_physics_attention=True)
    if name == "pg_gtpp_temporal":
        return PGGTPPModel(f_node=F_NODE, hidden_dim=hidden_dim,
            learned_correction=True, use_physics_intensity=True,
            use_fusion_gate=True, use_physics_attention=True,
            use_operational_temporal=True)
    return PGGTPPModel(f_node=F_NODE, hidden_dim=hidden_dim,
        learned_correction=True, use_physics_intensity=True,
        use_fusion_gate=True, use_physics_attention=True)


def finite(x):
    """Return a JSON-safe finite float, otherwise None."""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fmt_metric(x, digits=3):
    """Format optional metrics without crashing on undefined/NaN values."""
    if x is None:
        return "NA"
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-missions", type=int, default=50)
    p.add_argument("--corpus-seed", type=int, default=2026)
    p.add_argument("--seeds", type=str, default="7,21,42")
    p.add_argument("--strengths", type=str, default="0,0.5,1.0,1.5,2.0")
    p.add_argument("--mission-duration-s", type=float, default=900.0)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--window-length", type=int, default=10)
    p.add_argument("--lookahead-s", type=float, default=30.0)
    p.add_argument("--mark-loss-weight", type=float, default=0.25)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    p.add_argument("--eval-stride", type=int, default=5)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--output-dir", default="results/unmodeled_state_latent")
    p.add_argument("--diagnostic", action="store_true",
                   help="Print an explicit no-gate diagnostic interpretation alongside results.")
    p.add_argument("--mismatch-mode", choices=["cause_specific", "total"], default="cause_specific",
                   help="Truth-only operational mismatch: cause-specific changes relative risks; total preserves the legacy scalar benchmark.")
    args = p.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    config = make_config(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"protocol": vars(args), "models": list(MODELS), "conditions": {}}
    print(f"Mission split: {int(args.n_missions*0.70)}/{int(args.n_missions*0.15)}/{args.n_missions-int(args.n_missions*0.70)-int(args.n_missions*0.15)}")
    if args.mismatch_mode == "cause_specific":
        print("Truth-only residual: persistent q_e(t) with fixed cause-specific sensitivities; relative cause risks can change.")
    else:
        print("Truth-only residual: persistent q_e(t), lambda_true=lambda_physics*exp(strength*q_e(t)).")
    print("q_e is not stored/exposed; neural models infer it only from observable history.")
    print(f"Validation selection: best val AUC every {args.val_every} steps; mark_loss_weight={args.mark_loss_weight}.")
    if args.diagnostic:
        print("Diagnostic: pg_gtpp_no_gate uses fixed gate g=0, so lambda=lambda_physics*exp(residual).")

    for strength in strengths:
        cname = f"strength_{strength:g}"
        print(f"\n{'='*72}\n{cname}\n{'='*72}")
        cache = MissionCache.generate(config, args.corpus_seed, args.n_missions,
                                      unmodeled_state_strength=strength,
                                      unmodeled_state_mode=args.mismatch_mode)
        train_cache, val_cache, test_cache = split_three(cache)
        val_batch = val_cache.all_windows(args.window_length, args.lookahead_s, stride=args.eval_stride)
        test_batch = test_cache.all_windows(args.window_length, args.lookahead_s, stride=args.eval_stride)
        summary["conditions"][cname] = {"strength": strength, "models": {}}

        for name in MODELS:
            rows = []
            for seed in seeds:
                print(f"\n{name} | seed={seed}")
                seed_everything(seed)
                model = build_model(name, args.hidden_dim)
                cfg = TrainConfig(
                    n_steps=args.n_steps, batch_size=args.batch_size,
                    window_length=args.window_length, lookahead_s=args.lookahead_s,
                    learning_rate=args.learning_rate, mark_loss_weight=args.mark_loss_weight,
                    log_every=args.log_every, checkpoint_every=None, seed=seed,
                    lr_schedule=args.lr_schedule, val_every=args.val_every,
                    val_stride=args.eval_stride, restore_best_val=True,
                )
                # physics_only is a fixed analytical evaluator, not a learned
                # model.  Do not enter the optimizer loop for this control.
                if name != "physics_only":
                    train(model, train_cache, cfg, val_cache=val_cache)
                vm = evaluate(model, val_batch, mark_loss_weight=args.mark_loss_weight)
                tm = evaluate(model, test_batch, mark_loss_weight=args.mark_loss_weight)
                row = {
                    "seed": seed,
                    "val_auc": finite(vm.event_auc), "val_auprc": finite(vm.event_auprc),
                    "test_auc": finite(tm.event_auc), "test_auprc": finite(tm.event_auprc),
                    "test_brier": finite(tm.brier_score), "test_ece": finite(tm.calibration_error),
                    "test_cindex": finite(tm.c_index),
                }
                rows.append(row)
                print(
                    "  selected-val AUC=" + fmt_metric(row["val_auc"], 3)
                    + " | test AUC=" + fmt_metric(row["test_auc"], 3)
                    + " | test AUPRC=" + fmt_metric(row["test_auprc"], 3)
                    + " | ECE=" + fmt_metric(row["test_ece"], 4)
                )
            summary["conditions"][cname]["models"][name] = rows

        # Paired per-seed deltas against the physics-only control.
        physics_rows = {r["seed"]: r for r in summary["conditions"][cname]["models"]["physics_only"]}
        for model_name in ("pg_gtpp", "pg_gtpp_no_gate", "pg_gtpp_temporal", "neural_only"):
            for r in summary["conditions"][cname]["models"][model_name]:
                ref = physics_rows.get(r["seed"])
                if ref is not None and r["test_auc"] is not None and ref["test_auc"] is not None:
                    r["delta_test_auc_vs_physics"] = r["test_auc"] - ref["test_auc"]

    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

"""
scripts/run_fair_benchmark.py

Clean publication benchmark for PG-GTPP.

Protocol:
  * one fixed mission corpus and one fixed 70/15/15 mission-level split;
  * identical forecast horizon, window length, optimizer and mark-loss weight;
  * failure-only target: first LINK_DOWN for links UP at reference time;
  * multiple independent training seeds;
  * validation is diagnostic; final numbers are reported on untouched test
    missions;
  * primary metrics are time-dependent AUROC/AUPRC, Brier, ECE, IBS and
    C-index, with communication-mark errors and competing-risk diagnostics
    reported separately.

The baseline names "TGAT-style", "DySAT-style" and "Neural-TPP-style" are
intentional: the repository contains faithful architectural families, not
official reimplementations of the original authors' code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.model.baselines import BASELINE_REGISTRY, physics_regularized_loss
from simulator.training.train import MissionCache, TrainConfig, train
from simulator.training.loss import compute_loss


F_NODE = 11

ABLATIONS = {
    "physics_only": dict(learned_correction=False, use_physics_intensity=True, use_fusion_gate=False),
    "neural_only": dict(learned_correction=True, use_physics_intensity=False, use_fusion_gate=False),
    "pg_gtpp": dict(learned_correction=True, use_physics_intensity=True, use_fusion_gate=True),
}


try:
    from benchmark_checkpoint import atomic_json_dump, completed_runs, save_run_failure, save_run_result, write_progress
except ImportError:
    from scripts.benchmark_checkpoint import atomic_json_dump, completed_runs, save_run_failure, save_run_result, write_progress

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(args: argparse.Namespace) -> MissionConfig:
    base = default_config(master_seed=args.corpus_seed)
    fleet = FleetConfig(
        usv_count=1,
        n_auv_min=3, n_auv_max=8,
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
        master_seed=args.corpus_seed,
        split="train",
        wave=base.wave, current=base.current, acoustic=base.acoustic,
        fleet=fleet, envelope=envelope, hazard=base.hazard, graph=base.graph,
    )


def split_three(cache: MissionCache, seed: int):
    """70/15/15 mission-level split with no window leakage."""
    train_cache, holdout = cache.split(0.30, seed)
    val_cache, test_cache = holdout.split(0.50, seed + 1)
    return train_cache, val_cache, test_cache


def build_model(name: str, hidden_dim: int) -> torch.nn.Module:
    if name in ABLATIONS:
        return PGGTPPModel(f_node=F_NODE, hidden_dim=hidden_dim, **ABLATIONS[name])
    return BASELINE_REGISTRY[name](f_node=F_NODE, hidden_dim=hidden_dim)


def train_physics_soft(model, cache, cfg, reg_weight):
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    model.train()
    for step in range(1, cfg.n_steps + 1):
        batch = cache.sample_batch(rng, cfg.batch_size, cfg.window_length, cfg.lookahead_s)
        if batch.edge_index.shape[1] == 0:
            continue
        opt.zero_grad()
        out = model(batch.node_features, batch.node_type_ids, batch.node_mask,
                    batch.edge_index, batch.edge_physics)
        base = compute_loss(out, batch, mark_loss_weight=cfg.mark_loss_weight)
        risk = batch.edge_at_risk.float()
        physics = batch.edge_physics.clamp_min(0).sum(-1).clamp_min(1e-8)
        neural = out.total_intensity().clamp_min(1e-8)
        reg = ((neural.log() - physics.log()) ** 2 * risk).sum() / risk.sum().clamp_min(1.0)
        total = base.total_loss + reg_weight * reg
        total.backward()
        opt.step()
    return model


def train_one(name, seed, train_cache, val_cache, test_cache, args):
    seed_everything(seed)
    model = build_model(name, args.hidden_dim)
    cfg = TrainConfig(
        n_steps=args.n_steps, batch_size=args.batch_size,
        window_length=args.window_length, lookahead_s=args.lookahead_s,
        learning_rate=args.learning_rate, mark_loss_weight=args.mark_loss_weight,
        log_every=args.log_every, checkpoint_every=None, seed=seed,
        lr_schedule=args.lr_schedule,
    )
    if name == "physics_soft_reg":
        train_physics_soft(model, train_cache, cfg, args.physics_reg_weight)
    else:
        train(model, train_cache, cfg)

    from simulator.training.eval import evaluate
    val_batch = val_cache.all_windows(args.window_length, args.lookahead_s, stride=args.eval_stride)
    test_batch = test_cache.all_windows(args.window_length, args.lookahead_s, stride=args.eval_stride)
    val = evaluate(model, val_batch, mark_loss_weight=args.mark_loss_weight)
    test = evaluate(model, test_batch, mark_loss_weight=args.mark_loss_weight)
    return val.to_dict(), test.to_dict(), model.parameter_count()


def aggregate(rows):
    keys = [
        "event_auc", "event_auprc", "brier_score", "integrated_brier_score",
        "calibration_error", "survival_calibration_error", "c_index",
        "failure_time_mae_s", "pdr_mae", "pdr_rmse", "throughput_mae",
        "throughput_rmse", "delay_mae", "delay_rmse", "cause_macro_f1",
        "cause_brier",
    ]
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
        out[key] = {"mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals)} if vals else {"mean": float("nan"), "std": float("nan"), "n": 0}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-missions", type=int, default=200)
    p.add_argument("--corpus-seed", type=int, default=2026)
    p.add_argument("--seeds", type=str, default="7,17,27,37,47")
    p.add_argument("--mission-duration-s", type=float, default=900.0)
    p.add_argument("--n-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--window-length", type=int, default=10)
    p.add_argument("--lookahead-s", type=float, default=30.0)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--mark-loss-weight", type=float, default=1.0)
    p.add_argument("--physics-reg-weight", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--eval-stride", type=int, default=5)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    p.add_argument("--output-dir", type=str, default="results/fair_benchmark")
    p.add_argument("--include-ablations", action="store_true")
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--allow-nondeterministic", action="store_true")
    args = p.parse_args()

    if not args.allow_nondeterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating fixed corpus: {args.n_missions} missions, corpus_seed={args.corpus_seed}")
    cache = MissionCache.generate(make_config(args), args.corpus_seed, args.n_missions)
    train_cache, val_cache, test_cache = split_three(cache, seed=0)
    print(f"Mission split: train={len(train_cache.missions)}, "
          f"val={len(val_cache.missions)}, test={len(test_cache.missions)}")

    names = list(BASELINE_REGISTRY.keys())
    if args.include_ablations:
        names += list(ABLATIONS.keys())
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        names = [n for n in names if n in wanted]

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    summary = {
        "protocol": {
            "corpus_seed": args.corpus_seed,
            "n_missions": args.n_missions,
            "split": "70/15/15 mission-level",
            "training_seeds": seeds,
            "failure_target": "first LINK_DOWN for links UP at reference time",
            "lookahead_s": args.lookahead_s,
            "metrics": ["AUROC", "AUPRC", "Brier", "ECE", "IBS", "C-index",
                        "failure-time MAE", "PDR/throughput/delay MAE+RMSE"],
        },
        "models": {},
    }

    # ---- crash-safe, resumable benchmark state ----
    _completed_runs = completed_runs(out_dir)
    _checkpoint_config = vars(args).copy()
    _checkpoint_models = list(names)
    _checkpoint_seeds = list(seeds)
    write_progress(out_dir, _checkpoint_models, _checkpoint_seeds, _completed_runs, _checkpoint_config)

    for name in names:
        print(f"\n===== {name} =====")
        # IMPORTANT: include already-persisted runs in the aggregate when resuming.
        seed_rows = []
        val_rows = []
        for seed in seeds:
            saved = _completed_runs.get(name, {}).get(int(seed))
            if isinstance(saved, dict) and "result" in saved:
                seed_rows.append(saved.get("result", {}))
                if saved.get("validation") is not None:
                    val_rows.append(saved.get("validation"))
            elif saved is not None:
                # Backward compatibility with checkpoints created by v2.
                seed_rows.append(saved)
        params = build_model(name, args.hidden_dim).parameter_count()

        for seed in seeds:
            # Resume: skip a model/seed whose test result is already persisted.
            if name in _completed_runs and int(seed) in _completed_runs[name]:
                print(f"  SKIP completed checkpoint: {name} seed {seed}")
                continue
            print(f"--- seed {seed} ---")
            val, test, _params = train_one(name, seed, train_cache, val_cache, test_cache, args)
            params = _params
            val_rows.append(val)
            seed_rows.append(test)
            # Persist immediately after each successful seed; safe against process/terminal loss.
            save_run_result(out_dir, name, int(seed), test, config=_checkpoint_config, validation=val)
            _completed_runs.setdefault(name, {})[int(seed)] = {"result": test, "validation": val}
            write_progress(out_dir, _checkpoint_models, _checkpoint_seeds, _completed_runs, _checkpoint_config)
            print(f"test: AUROC={test['event_auc']:.4f} "
                  f"AUPRC={test['event_auprc']:.4f} "
                  f"Brier={test['brier_score']:.5f} "
                  f"IBS={test['integrated_brier_score']:.5f} "
                  f"C={test['c_index']:.4f}")
        summary["models"][name] = {
            "parameters": params,
            "per_seed_test": seed_rows,
            "test_aggregate": aggregate(seed_rows),
            "per_seed_validation": val_rows,
        }


    atomic_json_dump(summary, out_dir / "summary.json")
    print(f"\nWrote {out_dir / 'summary.json'}")
    write_progress(out_dir, _checkpoint_models, _checkpoint_seeds, _completed_runs, _checkpoint_config, status='completed')


if __name__ == "__main__":
    main()

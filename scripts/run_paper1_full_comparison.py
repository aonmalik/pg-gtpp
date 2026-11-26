"""
scripts/run_paper1_full_comparison.py

Trains and evaluates the 7 literature baselines (``model/baselines/``)
against an IDENTICAL mission corpus, train/val split, and evaluation
protocol as the 3 existing PG-GTPP ablations (``physics_only``,
``neural_only``, ``pg_gtpp``), and writes one results JSON per experiment
in the exact same schema already used by ``paper1_results/*.json`` -- so
the output of this script drops straight into the existing
``make_figures.py`` pipeline with zero changes beyond adding the 7 new
names to its ``ORDER``/``LABELS`` lists.

Why a fresh runner instead of extending ``experiments/run_paper1.py``
-----------------------------------------------------------------------
``run_paper1.py`` in this repository is an orchestration stub (prints
its three stages, does not implement them) -- the actual data-generation
seed, mission count, and step count used to produce the ORIGINAL 3-ablation
results are not recoverable from the uploaded artifacts (checkpoints only
retain model+optimizer state, not the run configuration). This script
does not try to reverse-engineer those exact numbers; instead it is
self-contained and DETERMINISTIC (fixed ``--seed``), so that as long as
you run it once, every one of its 7 (or 10, with ``--include-ablations``)
experiments are directly comparable TO EACH OTHER. If you want these
numbers to also be comparable to your existing physics_only/neural_only/
pg_gtpp results, re-run those three through this same script
(``--include-ablations``) rather than mixing in the old numbers, since
"identical held-out split" is what makes a comparison table defensible to
a reviewer -- mixing runs from two different corpora would reopen exactly
the kind of apples-to-oranges objection this whole exercise is meant to
avoid.

Usage
-----
    python scripts/run_paper1_full_comparison.py \\
        --n-missions 200 --n-steps 10000 --batch-size 16 \\
        --output-dir results/paper1_full_comparison \\
        --include-ablations

Requires torch + this repo's dependencies (see requirements.txt) and a
machine with a GPU recommended for --n-steps in the thousands; every
model here is small (tens of thousands to ~200K parameters, see the
architecture spec doc) so a single mid-range GPU is sufficient, and CPU
is workable at reduced --n-steps for a smoke test.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.model.baselines import BASELINE_REGISTRY, physics_regularized_loss
from simulator.training.train import MissionCache, TrainConfig, save_checkpoint, train
from simulator.training.eval import evaluate

F_NODE = 11  # see tests/test_phase5_model.py::F_NODE (simulator.trajectory_storage.F_NODE)

ABLATION_FLAGS = {
    # Mirrors configs/paper1_baselines.yaml exactly. PGGTPPModel now exposes
    # learned_correction / use_physics_intensity / use_fusion_gate (see
    # model/pg_gtpp_model.py and model/fusion.py), so all three ablations
    # can be constructed and trained through this one script on an
    # identical corpus/split to the 7 baselines -- giving a genuine 10-way
    # comparison in one run. NOTE: this reproduces the *ablation design*
    # faithfully, but will not reproduce your originally-uploaded
    # physics_only.json/neural_only.json bit-for-bit, since those came from
    # a different (unrecoverable) mission corpus, seed, and possibly a
    # different orchestration script. Treat this run's 10 numbers as one
    # self-consistent table; don't mix them with the originally-uploaded
    # physics_only/neural_only numbers in the same table/figure.
    "physics_only": dict(learned_correction=False, use_physics_intensity=True, use_fusion_gate=False),
    "neural_only": dict(learned_correction=True, use_physics_intensity=False, use_fusion_gate=False),
    "pg_gtpp": dict(learned_correction=True, use_physics_intensity=True, use_fusion_gate=True),
}


def build_mission_config(args: argparse.Namespace) -> MissionConfig:
    base = default_config(master_seed=args.seed)
    fleet = FleetConfig(
        usv_count=1, n_auv_min=3, n_auv_max=8, m_iout_min=10, m_iout_max=30,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    envelope = MissionEnvelopeConfig(
        duration_s=args.mission_duration_s, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m, depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    return MissionConfig(
        master_seed=args.seed, split="train", wave=base.wave, current=base.current,
        acoustic=base.acoustic, fleet=fleet, envelope=envelope,
        hazard=base.hazard, graph=base.graph,
    )


def train_physics_soft_reg(model, cache, val_cache, config: TrainConfig, physics_reg_weight: float):
    """Standalone loop mirroring training/train.py::train(), swapping in
    physics_regularized_loss for the one baseline that needs an auxiliary
    loss term. Kept separate rather than adding a hook to train.py, per
    this project's stated preference for not modifying already-verified
    Phase 6 code to accommodate a single downstream caller."""
    import numpy as np

    rng = np.random.default_rng(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    for step in range(1, config.n_steps + 1):
        batch = cache.sample_batch(rng, config.batch_size, config.window_length, config.lookahead_s)
        if batch.edge_index.shape[1] == 0:
            continue
        optimizer.zero_grad()
        output = model(batch.node_features, batch.node_type_ids, batch.node_mask, batch.edge_index, batch.edge_physics)
        loss_out = physics_regularized_loss(
            output, batch, batch.edge_physics,
            mark_loss_weight=config.mark_loss_weight, physics_reg_weight=physics_reg_weight,
        )
        loss_out.total_loss.backward()
        optimizer.step()
        if config.log_every and step % config.log_every == 0:
            print(
                f"  [physics_soft_reg] step {step}/{config.n_steps}  total={loss_out.total_loss.item():.4f}  "
                f"tpp={loss_out.tpp_loss.item():.4f}  mark={loss_out.mark_loss.item():.4f}  "
                f"reg={loss_out.physics_reg_loss.item():.4f}"
            )
        if config.checkpoint_every and config.checkpoint_dir and step % config.checkpoint_every == 0:
            save_checkpoint(config.checkpoint_dir / f"checkpoint_step{step}.pt", model, optimizer, step)
    return model


def run_one(name: str, model, cache: MissionCache, val_cache: MissionCache, args: argparse.Namespace, out_dir: Path):
    print(f"\n=== {name} ({model.parameter_count():,} parameters) ===")
    ckpt_dir = out_dir / "checkpoints" / name
    train_config = TrainConfig(
        n_steps=args.n_steps, batch_size=args.batch_size, window_length=args.window_length,
        lookahead_s=args.lookahead_s, learning_rate=args.learning_rate, log_every=args.log_every,
        checkpoint_every=args.checkpoint_every, checkpoint_dir=ckpt_dir, seed=args.seed,
        lr_schedule="cosine", mark_loss_weight=args.mark_loss_weight,
    )

    if name == "physics_soft_reg":
        train_physics_soft_reg(model, cache, val_cache, train_config, physics_reg_weight=args.physics_reg_weight)
    else:
        # Validation is part of the training protocol when TrainConfig.val_every
        # is enabled.  Passing val_cache is required by train() and keeps the
        # validation missions disjoint from the training missions.
        train(model, cache, train_config, val_cache=val_cache)

    val_batch = val_cache.all_windows(args.window_length, args.lookahead_s, stride=args.val_stride)
    metrics = evaluate(model, val_batch, mark_loss_weight=args.mark_loss_weight)

    result = {
        "experiment": name,
        "parameters": model.parameter_count(),
        "val_loss": metrics.val_loss,
        "auc": metrics.event_auc,
        "ece": metrics.calibration_error,
        "survival_ece": metrics.survival_calibration_error,
        "pdr_mae": metrics.pdr_mae,
        "throughput_mae": metrics.throughput_mae,
        "delay_mae": metrics.delay_mae,
        "n_edges": metrics.n_edges,
        "n_events": metrics.n_events,
        "training_steps": args.n_steps,
    }
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  AUC={metrics.event_auc:.3f}  ECE={metrics.calibration_error:.4f}  -> {out_path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-missions", type=int, default=200)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--mission-duration-s", type=float, default=900.0)
    p.add_argument("--n-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--window-length", type=int, default=10)
    p.add_argument("--lookahead-s", type=float, default=30.0)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument(
        "--mark-loss-weight", type=float, default=1.0,
        help="weight on the mark NLL term relative to the TPP NLL term (loss.py: "
             "total = tpp_loss + mark_loss_weight * mark_loss). Applies identically to "
             "every model in this run -- raising it gives the mark decoder relatively "
             "more gradient signal for every model, not just PG-GTPP, so any comparison "
             "stays apples-to-apples. See docs/BASELINES.md 'Improving PG-GTPP's mark "
             "accuracy' for why this is the first thing worth trying.",
    )
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--physics-reg-weight", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--val-stride", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model-init-seed", type=int, default=1234)
    p.add_argument("--output-dir", type=str, default="results/paper1_full_comparison")
    p.add_argument("--only", type=str, default=None, help="comma-separated subset of baseline names to run")
    p.add_argument(
        "--include-ablations", action="store_true",
        help="also (re)train physics_only/neural_only/pg_gtpp on this run's identical corpus, "
             "using PGGTPPModel's own construction flags if your local pg_gtpp_model.py exposes them.",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="skip any experiment whose <name>.json already exists in --output-dir, instead of "
             "retraining it -- makes a partial/interrupted run resumable rather than starting over.",
    )
    p.add_argument(
        "--decouple-mark-head", action="store_true",
        help="opt-in architecture variant: give PG-GTPP's mark decoder its own small "
             "adapter layer instead of the literal edge_context tensor the 8-head fusion "
             "module also reads (see model/pg_gtpp_model.py docstring). Only affects the "
             "pg_gtpp ablation entry -- physics_only/neural_only and all 7 baselines are "
             "unaffected. Writes to '<name>_decoupled.json' instead of '<name>.json' so it "
             "never silently overwrites the default-architecture result -- report both.",
    )
    p.add_argument(
        "--mark-physics-skip", action="store_true",
        help="opt-in architecture variant: concatenate raw edge_physics (geo/wave/doppler/"
             "current) directly into the mark decoder's input, bypassing the graph "
             "encoder's abstraction for that one signal (see model/pg_gtpp_model.py "
             "docstring for the DySAT-comparison hypothesis this tests). Only affects the "
             "pg_gtpp ablation entry. Writes to 'pg_gtpp_mark_skip.json' -- a distinct "
             "file, never overwrites 'pg_gtpp.json'. Independent of --decouple-mark-head; "
             "run them as separate experiments, not combined, so either result stays "
             "interpretable on its own.",
    )
    p.add_argument(
        "--allow-nondeterministic", action="store_true",
        help="skip forcing deterministic CUDA algorithms (see main() docstring below for why "
             "this matters). Runs faster but results will NOT reproduce run-to-run even at a "
             "fixed --seed/--model-init-seed. Only use this for quick smoke tests where exact "
             "numbers don't matter -- never for results you intend to report.",
    )
    args = p.parse_args()

    if not args.allow_nondeterministic:
        # torch.manual_seed() alone does NOT make GPU training reproducible.
        # graph_encoder.py uses scatter_add/scatter_reduce (non-deterministic
        # CUDA atomic-add thread ordering by default) and neural_hawkes.py
        # uses nn.LSTM (cuDNN's RNN backward pass is non-deterministic by
        # default too). Confirmed empirically: physics_only (whose output
        # never touches the encoder's learned weights) reproduces exactly
        # across reruns of the same seed; every model that routes through
        # the encoder or an LSTM does not -- dysat_style's PDR MAE differed
        # by ~2x between two runs of the nominally identical configuration
        # before this flag existed. warn_only=True below because a small
        # number of CUDA kernels genuinely have no deterministic
        # implementation available; if you see
        # "UserWarning: ... does not have a deterministic implementation"
        # in stderr, that specific op is still a residual noise source and
        # is worth knowing about, not something to silently ignore.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_missions} missions (seed={args.seed})...")
    mission_config = build_mission_config(args)
    cache = MissionCache.generate(mission_config, master_seed=args.seed, n_missions=args.n_missions)
    total_events = sum(len(m.events) for m in cache.missions)
    print(f"  {len(cache.missions)} missions, {total_events} total events")

    train_cache, val_cache = cache.split(val_fraction=args.val_fraction, seed=0)
    print(f"  split (mission-level): {len(train_cache.missions)} train / {len(val_cache.missions)} val")

    names = list(BASELINE_REGISTRY.keys())
    if args.include_ablations:
        names = names + list(ABLATION_FLAGS.keys())
        if args.decouple_mark_head:
            names = names + ["pg_gtpp_decoupled"]
        if args.mark_physics_skip:
            names = names + ["pg_gtpp_mark_skip"]
    if args.only:
        wanted = set(args.only.split(","))
        names = [n for n in names if n in wanted]

    def build_model(name: str):
        torch.manual_seed(args.model_init_seed)
        if name == "pg_gtpp_decoupled":
            # Same construction as pg_gtpp, plus the opt-in mark-head adapter
            # (see model/pg_gtpp_model.py docstring for why). A distinct
            # name -> distinct output file, so this never overwrites the
            # default-architecture pg_gtpp.json.
            return PGGTPPModel(
                f_node=F_NODE, hidden_dim=args.hidden_dim,
                decouple_mark_head=True, **ABLATION_FLAGS["pg_gtpp"],
            )
        if name == "pg_gtpp_mark_skip":
            # Same construction as pg_gtpp, plus the opt-in raw-physics skip
            # connection into the mark decoder (see model/pg_gtpp_model.py
            # docstring: tests whether DySAT-style's mark advantage comes
            # from more direct access to physics-relevant signal than the
            # encoder's abstraction provides). Distinct output file.
            return PGGTPPModel(
                f_node=F_NODE, hidden_dim=args.hidden_dim,
                mark_physics_skip=True, **ABLATION_FLAGS["pg_gtpp"],
            )
        if name in ABLATION_FLAGS:
            return PGGTPPModel(f_node=F_NODE, hidden_dim=args.hidden_dim, **ABLATION_FLAGS[name])
        return BASELINE_REGISTRY[name](f_node=F_NODE, hidden_dim=args.hidden_dim)

    results = {}
    skipped = []
    for name in names:
        out_path = out_dir / f"{name}.json"
        if args.skip_existing and out_path.exists():
            print(f"skip {name}: {out_path} already exists (--skip-existing)")
            results[name] = json.loads(out_path.read_text())
            skipped.append(name)
            continue
        model = build_model(name)
        results[name] = run_one(name, model, train_cache, val_cache, args, out_dir)

    if any(n in ABLATION_FLAGS for n in names) and not skipped:
        print(
            "\nNOTE: physics_only/neural_only/pg_gtpp above were all trained on THIS run's "
            "mission corpus/seed, not the corpus that produced your originally-uploaded "
            "physics_only.json/neural_only.json -- report this run's results as one "
            "self-consistent table (identical held-out split throughout); do not merge them "
            "with the previously-uploaded ablation numbers, which came from a different, "
            "unrecoverable corpus. See docs/BASELINES.md."
        )
    if skipped:
        print(f"\nSkipped (already present, --skip-existing): {', '.join(skipped)}")

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nAll done. Per-experiment JSON + summary written to {out_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def mission_means(records):
    """
    Average overlapping edge/reference-window losses within each mission.
    A mission is therefore the statistical resampling unit.
    """
    grouped = {}
    for r in records:
        grouped.setdefault(str(r["mission_id"]), []).append(float(r["nll"]))

    return {
        mission: float(np.mean(losses))
        for mission, losses in grouped.items()
    }


def paired_delta(records_a, records_b):
    """
    Return mission-level paired differences A - B.
    """
    A = mission_means(records_a)
    B = mission_means(records_b)

    common = sorted(set(A) & set(B))
    if not common:
        raise ValueError("No common missions between paired records")

    return {
        mission: A[mission] - B[mission]
        for mission in common
    }


def bootstrap_mean(values, n_boot, rng):
    """
    Percentile paired bootstrap over missions.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("Cannot bootstrap zero missions")

    if len(values) == 1:
        mean = float(values[0])
        return mean, mean, mean

    idx = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )
    boot_means = values[idx].mean(axis=1)

    mean = float(values.mean())
    lo, hi = np.quantile(boot_means, [0.025, 0.975])

    return float(mean), float(lo), float(hi)


def load_seed_deltas(base, seed):
    pg_path = base / f"pg_gtpp_seed{seed}_test_losses.json"
    physics_path = base / f"physics_only_seed{seed}_test_losses.json"
    gbt_path = base / f"gbt_residual_seed{seed}_test_losses.json"

    if not (pg_path.exists() and physics_path.exists()):
        return None

    pg = load(pg_path)
    physics = load(physics_path)

    pg_minus_physics = paired_delta(pg, physics)

    pg_minus_gbt = None
    if gbt_path.exists():
        gbt = load(gbt_path)
        pg_minus_gbt = paired_delta(pg, gbt)

    return pg_minus_physics, pg_minus_gbt


def main():
    ap = argparse.ArgumentParser(
        description="Mission-level paired bootstrap for PG-GTPP comparisons."
    )
    ap.add_argument(
        "--root",
        default="results/main_fair_comparison",
    )
    ap.add_argument(
        "--strengths",
        default="0,5",
    )
    ap.add_argument(
        "--seeds",
        default="42,43,44",
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=10000,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = ap.parse_args()

    root = Path(args.root)
    rng = np.random.default_rng(args.seed)

    requested_seeds = [
        int(x.strip())
        for x in args.seeds.split(",")
        if x.strip()
    ]

    strengths = [
        x.strip()
        for x in args.strengths.split(",")
        if x.strip()
    ]

    output = {
        "protocol": {
            "root": str(root),
            "strengths": args.strengths,
            "seeds": args.seeds,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "resampling_unit": "mission",
            "within_mission_aggregation": "mean_over_overlapping_test_records",
            "cross_seed_aggregation": "mean_delta_per_mission",
        },
        "conditions": {},
    }

    for strength in strengths:
        base = root / f"strength_{strength}"

        seed_results = {}

        for seed in requested_seeds:
            result = load_seed_deltas(base, seed)

            if result is None:
                continue

            pg_physics, pg_gbt = result

            seed_entry = {
                "n_missions_pg_physics": len(pg_physics),
                "pg_minus_physics_mean": float(
                    np.mean(list(pg_physics.values()))
                ),
            }

            if pg_gbt is not None:
                seed_entry.update({
                    "n_missions_pg_gbt": len(pg_gbt),
                    "pg_minus_gbt_mean": float(
                        np.mean(list(pg_gbt.values()))
                    ),
                })

            seed_results[f"seed_{seed}"] = seed_entry

        # ------------------------------------------------------------
        # Final pooled inference:
        #
        # For each mission:
        #   1. compute PG - baseline separately for each seed
        #   2. average across seeds
        #   3. bootstrap the resulting mission-level differences
        #
        # This prevents the same mission evaluated under different
        # training seeds from being counted as independent missions.
        # ------------------------------------------------------------

        common_physics = None
        common_gbt = None

        per_seed_physics = {}
        per_seed_gbt = {}

        for seed in requested_seeds:
            result = load_seed_deltas(base, seed)

            if result is None:
                continue

            pg_physics, pg_gbt = result

            per_seed_physics[seed] = pg_physics

            if pg_gbt is not None:
                per_seed_gbt[seed] = pg_gbt

        if per_seed_physics:
            mission_sets = [
                set(d.keys())
                for d in per_seed_physics.values()
            ]
            common_physics = sorted(set.intersection(*mission_sets))

            pooled_physics = np.asarray(
                [
                    np.mean(
                        [
                            per_seed_physics[seed][mission]
                            for seed in per_seed_physics
                        ]
                    )
                    for mission in common_physics
                ],
                dtype=float,
            )

            mean, lo, hi = bootstrap_mean(
                pooled_physics,
                args.n_boot,
                rng,
            )

            output["conditions"].setdefault(
                f"strength_{strength}",
                {}
            )["pooled_pg_minus_physics"] = {
                "n_missions": len(common_physics),
                "n_seeds": len(per_seed_physics),
                "delta_nll": mean,
                "ci95": [lo, hi],
            }

        if per_seed_gbt:
            mission_sets = [
                set(d.keys())
                for d in per_seed_gbt.values()
            ]
            common_gbt = sorted(set.intersection(*mission_sets))

            pooled_gbt = np.asarray(
                [
                    np.mean(
                        [
                            per_seed_gbt[seed][mission]
                            for seed in per_seed_gbt
                        ]
                    )
                    for mission in common_gbt
                ],
                dtype=float,
            )

            mean, lo, hi = bootstrap_mean(
                pooled_gbt,
                args.n_boot,
                rng,
            )

            output["conditions"].setdefault(
                f"strength_{strength}",
                {}
            )["pooled_pg_minus_gbt"] = {
                "n_missions": len(common_gbt),
                "n_seeds": len(per_seed_gbt),
                "delta_nll": mean,
                "ci95": [lo, hi],
            }

        output["conditions"].setdefault(
            f"strength_{strength}",
            {}
        )["per_seed"] = seed_results

    output_path = root / "paired_bootstrap.json"
    output_path.write_text(
        json.dumps(output, indent=2)
    )

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

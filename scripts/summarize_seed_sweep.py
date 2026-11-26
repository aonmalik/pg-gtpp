"""
scripts/summarize_seed_sweep.py

Reads results/seed_sweep/seed_<seed>/<model>.json for every seed/model pair
written by run_seed_sweep.sh, and reports mean +/- std per model per metric,
plus a plain-language overlap check on AUC specifically (the question the
sweep exists to answer).

This is deliberately NOT a formal hypothesis test (no t-test, no p-value) --
with 3-5 seeds that would be more decoration than substance. What it does
instead: report the actual spread (mean +/- std, min-max) per model, and
flag in plain language whether the #1 and #2 models' AUC ranges overlap.
Overlapping ranges means "the sweep cannot tell these two apart" -- report
that plainly rather than picking a winner anyway.

Usage:
    python scripts/summarize_seed_sweep.py --input-root results/seed_sweep
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", default="results/seed_sweep")
    args = p.parse_args()

    root = Path(args.input_root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist -- run scripts/run_seed_sweep.sh first")

    seed_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    if not seed_dirs:
        raise SystemExit(f"no seed_* subdirectories found under {root}")

    # per_model[model_name][metric] -> list of values across seeds
    per_model: dict[str, dict[str, list[float]]] = {}
    seeds_seen: dict[str, list[str]] = {}

    for sd in seed_dirs:
        seed_label = sd.name.replace("seed_", "")
        for f in sorted(sd.glob("*.json")):
            if f.name.startswith("_"):
                # _summary.json is written by run_paper1_full_comparison.py
                # itself and gets overwritten by each sequential --only
                # invocation that shares this output dir -- it only ever
                # reflects the LAST model run into this directory, not the
                # full set. Read the individual <model>.json files instead
                # (below); skip anything summary/metadata-like here.
                continue
            model = f.stem
            data = json.loads(f.read_text())
            per_model.setdefault(model, {})
            seeds_seen.setdefault(model, []).append(seed_label)
            for metric in ("auc", "ece", "pdr_mae", "throughput_mae", "delay_mae"):
                per_model[model].setdefault(metric, []).append(data[metric])

    if not per_model:
        raise SystemExit(f"no result JSON files found under {root}/seed_*/")

    print(f"Seed sweep summary ({len(seed_dirs)} seed(s) found)\n")
    header = f"{'model':<18}{'n':>3}  {'AUC (mean+/-std)':<22}{'ECE (mean+/-std)':<22}"
    print(header)
    print("-" * len(header))

    auc_summary = {}
    for model in sorted(per_model, key=lambda m: -statistics.mean(per_model[m]["auc"])):
        aucs = per_model[model]["auc"]
        eces = per_model[model]["ece"]
        n = len(aucs)
        auc_mean = statistics.mean(aucs)
        auc_std = statistics.stdev(aucs) if n > 1 else float("nan")
        ece_mean = statistics.mean(eces)
        ece_std = statistics.stdev(eces) if n > 1 else float("nan")
        auc_summary[model] = (auc_mean, auc_std, n, aucs)
        auc_str = f"{auc_mean:.4f} +/- {auc_std:.4f}" if n > 1 else f"{auc_mean:.4f} (n=1, no std)"
        ece_str = f"{ece_mean:.4f} +/- {ece_std:.4f}" if n > 1 else f"{ece_mean:.4f} (n=1, no std)"
        print(f"{model:<18}{n:>3}  {auc_str:<22}{ece_str:<22}")

    print()
    if any(len(v[3]) == 1 for v in auc_summary.values()):
        print("NOTE: at least one model has only 1 seed so far -- std is undefined for it.")
        print("Run more seeds (or check scripts/run_seed_sweep.sh log for failures) before")
        print("trusting any ranking below.\n")

    # Plain-language overlap check on AUC, using mean +/- 1 std as the range.
    # This is NOT a significance test -- it is a blunt "can we even tell
    # these apart yet" sanity check appropriate for 3-5 seeds.
    ranked = sorted(auc_summary.items(), key=lambda kv: -kv[1][0])
    print("AUC ranking (highest mean first):")
    for i, (model, (mean, std, n, vals)) in enumerate(ranked, 1):
        print(f"  #{i} {model}: {mean:.4f} +/- {std:.4f}" if n > 1 else f"  #{i} {model}: {mean:.4f} (n=1)")

    if len(ranked) >= 2 and all(v[1][2] > 1 for v in ranked[:2]):
        (m1, (mean1, std1, n1, _)), (m2, (mean2, std2, n2, _)) = ranked[0], ranked[1]
        lo1, hi1 = mean1 - std1, mean1 + std1
        lo2, hi2 = mean2 - std2, mean2 + std2
        overlap = not (hi2 < lo1 or hi1 < lo2)
        print()
        if overlap:
            print(f"VERDICT: #1 ({m1}) and #2 ({m2})'s mean+/-1std AUC ranges OVERLAP.")
            print("This sweep cannot yet distinguish them -- do not claim a ranking between")
            print("these two in the paper without more seeds or a proper significance test.")
        else:
            print(f"VERDICT: #1 ({m1}) and #2 ({m2})'s mean+/-1std AUC ranges do NOT overlap.")
            print("This is supportive of a real gap, though still only mean+/-1std on a few")
            print("seeds -- not a substitute for a proper significance test if the paper needs one.")
    print()
    print("Full per-metric detail (mark MAEs) below:")
    print(f"{'model':<18}{'PDR MAE':>20}{'Delay MAE':>20}{'Thpt MAE':>24}")
    for model in per_model:
        def fmt(metric):
            vals = per_model[model][metric]
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            return f"{m:.4f} +/- {s:.4f}" if len(vals) > 1 else f"{m:.4f} (n=1)"
        print(f"{model:<18}{fmt('pdr_mae'):>20}{fmt('delay_mae'):>20}{fmt('throughput_mae'):>24}")


if __name__ == "__main__":
    main()

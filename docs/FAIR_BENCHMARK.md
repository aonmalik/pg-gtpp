# PG-GTPP fair benchmark protocol

## What was fixed

1. **Failure-only target.** The supervised event is the first `LINK_DOWN` after
   the reference time. `LINK_UP` is not a positive failure label.
2. **Risk-set definition.** Edges already down at the reference time are kept
   in the graph for message passing but are excluded from the failure-risk
   likelihood and metrics.
3. **Observed event time.** Every positive edge stores the time from the
   reference instant to the first failure. Censored edges store the full
   forecast horizon.
4. **Cause supervision.** Positive failures carry one of `geo`, `wave`,
   `doppler`, `current`. PG-GTPP trains with the selected cause-specific
   intensity in the competing-risks event term.
5. **Correct likelihood.** The models predict an instantaneous hazard that is
   explicitly assumed constant over the forecast horizon. Under this
   piecewise-constant assumption, the continuous-time first-event likelihood
   is exact: `integral(lambda dt) - log(lambda_c*)` for failures and
   `integral(lambda dt)` for censored observations.
6. **Fair metrics.** The evaluator reports horizon-specific AUROC, AUPRC,
   Brier score and ECE, plus IBS, C-index, failure-time MAE, mark MAE/RMSE,
   and PG-GTPP cause-attribution F1/Brier.
7. **Mission-level split.** The benchmark uses a fixed 70/15/15 split of
   missions. Windows from the same mission never cross train/validation/test.
8. **Multiple seeds.** The benchmark defaults to five training seeds. Report
   mean +/- standard deviation across seeds.
9. **Untouched test set.** Validation metrics are retained for diagnostics;
   final headline numbers are computed only on test missions.
10. **Baseline naming.** `TGAT-style`, `DySAT-style`, and `Neural-TPP-style`
    are explicitly named as architectural families rather than claiming
    official reimplementations of the original authors' code.

## Primary table

Use this order in the paper:

- AUROC@30s
- AUPRC@30s
- Brier@30s (lower is better)
- ECE@30s (lower is better)
- IBS (lower is better)
- C-index (higher is better)

Secondary table:

- failure-time MAE
- PDR MAE/RMSE
- throughput MAE/RMSE
- delay MAE/RMSE

PG-GTPP-only/competing-risk diagnostic:

- cause macro-F1
- cause Brier

## Running

From the repository root:

```bash
PYTHONPATH=src python scripts/run_fair_benchmark.py \
  --n-missions 200 \
  --n-steps 10000 \
  --batch-size 16 \
  --window-length 10 \
  --lookahead-s 30 \
  --seeds 7,17,27,37,47 \
  --include-ablations \
  --output-dir results/fair_benchmark
```

For a quick smoke test:

```bash
PYTHONPATH=src python scripts/run_fair_benchmark.py \
  --n-missions 12 \
  --n-steps 20 \
  --batch-size 4 \
  --seeds 7 \
  --include-ablations
```

The smoke test is for software validation only, not for scientific reporting.

## Important interpretation

The recent UWA channel-prediction papers are **related-work references, not
direct reproductions** in this repository. They generally predict CSI/channel
states rather than first-failure survival probabilities. The benchmark here
therefore compares PG-GTPP against compatible model families on a common
failure-risk task, which is the scientifically fair comparison.

The repository's current TGAT/DySAT/Neural-Hawkes files remain lightweight
style implementations. They should not be described as exact replications
unless their original equations, preprocessing, training objective and
hyperparameters are implemented and verified against the source papers.

# Targeted PG-GTPP improvement benchmark

This revision adds two controlled improvements without changing the default PG-GTPP behavior:

1. **Cause-specific latent operational mismatch** (`--mismatch-mode cause_specific`, default): the hidden persistent operational state now perturbs the four cause hazards with fixed, predeclared sensitivities. Unlike the legacy scalar mismatch, this can change the relative cause ranking and therefore creates a genuine ranking challenge for the learned residual.
2. **Dedicated pair-history temporal residual encoder** (`pg_gtpp_temporal`): a small GRU reads only observable endpoint-pair history (relative position, endpoint velocities, battery and FIM uncertainty) and feeds the fusion/residual pathway. The truth-only latent state is never exposed.

The legacy scalar mismatch remains available with `--mismatch-mode total` for continuity.

## Recommended first run

```bash
cd ~/pg-gtpp
PYTHONPATH=src python scripts/run_unmodeled_state_benchmark.py \
  --n-missions 50 \
  --n-steps 500 \
  --batch-size 8 \
  --strengths 0,1,2,3 \
  --seeds 42 \
  --val-every 100 \
  --mark-loss-weight 0.10 \
  --mismatch-mode cause_specific \
  --output-dir results/targeted_improvement_smoke
```

For a direct legacy comparison, repeat with `--mismatch-mode total`.

## Objective sweep

Do not select the mark-loss weight from the test set. For a fixed corpus/split/seed, run short validation experiments at:

```text
0.00, 0.05, 0.10, 0.25, 0.50
```

Then lock the best validation setting and use it identically for all models in the final comparison.

## Interpretation

A useful result is not simply "PG-GTPP beats physics-only on clean data." The simulator deliberately makes physics-only a strong nominal control. The desired pattern is that PG-GTPP/PG-GTPP-temporal remains competitive when physics is correct and gains ranking and/or calibration as the truth-only operational state increasingly changes the relative cause risks.

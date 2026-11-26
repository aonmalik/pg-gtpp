# Main fair comparison protocol

`run_main_fair_comparison.py` is the main apples-to-apples benchmark runner.

## Models

- `physics_only`
- `neural_only`
- `pg_gtpp`
- `gbt_residual` (physics-offset, cause-specific HistGradientBoosting residual)
- `cox_perlink`
- `rnn_perlink`
- `gnn_generic`
- `tgat_style`
- `dysat_style`
- `neural_hawkes`
- `physics_soft_reg`

All learned neural models use the same mission-level train/validation/test split,
lookahead, training loss, validation selection rule, and seeds. GBT uses its own
continuous-time integrated validation NLL to select residual scale, as required
by its offset-likelihood formulation.

## Continuous-time NLL convention

For physics-aware models, the compensator integrates nominal cause-specific
physics hazards along the known future trajectory. The learned residual ratio is
computed only from the observed history and is held fixed through the forecast.
For models without a future physics path, the reference-time total intensity is
held constant through the forecast.

This distinction is explicit in the JSON and prevents a hidden future-feature
leak. The metric is a model-specific first-event competing-risk likelihood, not a
binary classification surrogate.

## Recommended final run

```bash
cd ~/pg-gtpp
conda activate pg-gtpp

PYTHONPATH=src python scripts/run_main_fair_comparison.py \
  --n-missions 50 \
  --corpus-seed 2026 \
  --seeds 42,43,44 \
  --strengths 0,5 \
  --window-length 10 \
  --lookahead-s 30 \
  --eval-stride 10 \
  --quadrature-points 65 \
  --n-steps 3000 \
  --batch-size 16 \
  --hidden-dim 64 \
  --learning-rate 0.003 \
  --mark-loss-weight 0.25 \
  --val-every 250 \
  --val-stride 10 \
  --output-dir results/main_fair_comparison \
  --skip-existing
```

Run a nominal-only smoke test first with `--n-missions 5 --n-steps 100 --seeds 42`
and `--strengths 0`.

Do not combine these numbers with older `physics_only/neural_only/pg_gtpp`
JSON files generated from a different corpus or split. The table should be built
only from one run of this protocol.

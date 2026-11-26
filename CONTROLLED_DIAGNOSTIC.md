# Controlled PG-GTPP Diagnostic

This version uses the same mission-level 70/15/15 split protocol as
`scripts/run_fair_benchmark.py`:

- train: 70%
- validation: 15%
- test: 15%

It compares `physics_only` and `pg_gtpp`, reports validation metrics,
then diagnoses test-time intensity differences, fusion-gate behavior,
neural correction magnitude, and gradients.

Run from the repository root:

```bash
conda activate pg-gtpp

PYTHONPATH=src python scripts/diagnose_pg_gtpp_fusion.py \
  --n-missions 20 \
  --n-steps 500 \
  --batch-size 4 \
  --window-length 10 \
  --lookahead-s 30 \
  --seed 7
```

Do not use this diagnostic's metrics as final paper results. The purpose
is to verify that PG-GTPP's learned fusion is active under the exact
mission split used by the fair benchmark.

# Controlled Physics-Mismatch Benchmark

This extension adds a benchmark that separates the **model-visible physics prior**
from the **ground-truth event-generating hazard**.

## Why

The original simulator generates failures from the same closed-form hazard used
by the `physics_only` ablation. Therefore `physics_only` is correctly specified
under the original data-generating process and PG-GTPP is not expected to gain
much.

The new benchmark keeps trajectories, wave/current fields, node features,
mission IDs, mission split, windows, and training protocol unchanged. It changes
only the hazard used to generate the ground-truth events:

- geo: `1 + m`
- wave: `1 - 0.5*m`
- doppler: `1 + 0.5*m`
- current: `1 - m/3`

The model still receives the **unscaled** physics hazard. Thus `m=0` is the
well-specified control and increasing `m` introduces controlled physics-model
misspecification.

## Neural-only definition

The new `neural_only` ablation disables both:

1. direct physics-intensity fusion, and
2. physics-biased graph attention.

This is stronger and cleaner than the original ablation, where the graph
encoder still contained a physics attention bias.

## Recommended run

```bash
cd ~/pg-gtpp
conda activate pg-gtpp

PYTHONPATH=src python scripts/run_physics_mismatch_benchmark.py \
  --n-missions 50 \
  --n-steps 1000 \
  --batch-size 8 \
  --window-length 10 \
  --lookahead-s 30 \
  --mismatch-levels 0,0.05,0.10,0.20,0.30,0.50 \
  --seeds 7,21,42,84,123 \
  --output-dir results/physics_mismatch
```

For a quick smoke test:

```bash
PYTHONPATH=src python scripts/run_physics_mismatch_benchmark.py \
  --n-missions 20 --n-steps 100 --batch-size 4 \
  --window-length 10 --lookahead-s 30 \
  --mismatch-levels 0,0.20 --seeds 7 \
  --output-dir results/physics_mismatch_smoke
```

## Interpretation

Do not require PG-GTPP to beat physics-only at `m=0`. The scientifically useful
claim is that PG-GTPP retains or improves performance as the physical prior
becomes misspecified, while physics-only degrades.

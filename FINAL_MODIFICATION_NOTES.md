# PG-GTPP final fusion modification

## What changed

The full PG-GTPP fusion was changed from an absolute-intensity convex blend

`lambda = g * lambda_phys + (1-g) * softplus(delta)`

to a physics-preserving multiplicative log-residual

`lambda = lambda_phys * exp((1-g) * delta)`

where `delta = 2*tanh(MLP(context))` is bounded to `[-2, 2]`.

The correction head is zero-initialized, so the full model starts exactly at
the physics prior. The learned residual can then move the hazard either up or
down, while the gate controls how much of that correction is trusted.

The `physics_only` and `neural_only` ablations remain genuine:
- physics_only returns the physics hazard exactly;
- neural_only disables both direct physics fusion and physics-biased graph attention.

## Why this is the defensible change

The previous additive/convex parameterization forced the learned branch to
learn an absolute hazard scale independently of the physics hazard. In the
controlled diagnostics, the learned contribution was only about 2--3% of the
total hazard and PG-GTPP was almost identical to physics-only. A log-residual
lets the learned model correct multiplicative model error, which is the form
used by many positive-rate models, without destroying the physics prior.

This is NOT a test-set optimization trick: the same architecture and training
objective are used before seeing the held-out test results, and the mismatch
benchmark must still be reported over all predefined mismatch levels and
paired seeds.

## Validation command on the target environment

```bash
cd ~/pg-gtpp
conda activate pg-gtpp
PYTHONPATH=src pytest -q tests/test_phase5_model.py
```

Then run the controlled benchmark:

```bash
PYTHONPATH=src python scripts/run_physics_mismatch_benchmark.py \
  --n-missions 50 \
  --n-steps 1000 \
  --batch-size 8 \
  --window-length 10 \
  --lookahead-s 30 \
  --mismatch-levels 0,0.05,0.10,0.20,0.30,0.50 \
  --seeds 7,21,42,84,123 \
  --output-dir results/physics_mismatch_final_logresidual
```

Do not claim PG-GTPP is better unless the paired held-out results actually
show a statistically and practically meaningful improvement.

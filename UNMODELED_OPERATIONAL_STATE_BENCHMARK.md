# Unmodeled Operational-State Benchmark

## Motivation

The nominal simulator generates failures from four declared physics causes:
`geo`, `wave`, `doppler`, and `current`. The physics-only baseline therefore
has access to the same mechanism used to generate the nominal data.

This benchmark creates a more realistic misspecification without giving the
neural model a privileged hidden label. The simulator already contains three
observable node states that are useful for communication reliability but are
not consumed by `HazardEvaluator`:

- `battery_frac`: energy depletion;
- `fim_uncertainty`: localization uncertainty;
- velocity/heading relative to the instantaneous line of sight: an operational
  alignment proxy.

The truth event generator uses these states to create an additional
multiplicative communication-stress factor, while the model-visible physics
hazard remains exactly the original four-cause evaluator.

## Truth model

For an at-risk edge,

`lambda_true(t) = lambda_physics(t) * exp(s * S_operational(t))`

where `s` is the benchmark strength and `S_operational` is a bounded weighted
combination of battery stress, localization stress, and alignment stress.
The fixed weights are 0.50, 0.30, and 0.20 respectively and are not tuned on
the test set.

Recovery is unchanged. Cause proportions are unchanged because the added
factor is an edge-level reliability multiplier applied to the total failure
hazard. This isolates the question of survival/risk prediction from a new
cause-label mechanism.

## Why this is a fair PG-GTPP test

- Physics-only cannot use the operational states to modify its failure
  intensity; its hazard remains the nominal closed-form physics hazard.
- Neural-only can exploit the ordinary node-history input containing these
  states.
- PG-GTPP receives the same observations as neural-only, but additionally has
  the nominal physics prior and learns a multiplicative residual correction.
- `s=0` is the exact nominal control.
- No hidden state or future target is inserted into the model input.

## Suggested conditions

- `0.0`: nominal control
- `0.5`: weak/moderate operational misspecification
- `1.0`: moderate
- `1.5`: strong
- `2.0`: severe

The benchmark should be run with multiple corpus/training seeds at paper
scale. Do not select the final condition or hyperparameters from the test set.

## Command

```bash
PYTHONPATH=src python scripts/run_unmodeled_state_benchmark.py \
  --n-missions 200 \
  --mission-duration-s 900 \
  --n-steps 3000 \
  --batch-size 8 \
  --window-length 10 \
  --lookahead-s 30 \
  --strengths 0,0.5,1.0,1.5,2.0 \
  --seeds 7,21,42,84,123 \
  --output-dir results/unmodeled_state_200
```

For a quick smoke test:

```bash
PYTHONPATH=src python scripts/run_unmodeled_state_benchmark.py \
  --n-missions 50 --n-steps 500 --batch-size 8 \
  --strengths 0,1.0,2.0 --seeds 42 \
  --output-dir results/unmodeled_state_smoke
```

## Interpretation

A defensible result is a curve where physics-only degrades as `s` increases,
while PG-GTPP retains more performance because its learned residual exploits
information absent from the mechanistic hazard. Neural-only is an important
competitor: if it wins consistently, the experiment does not justify the
physics-fusion claim.

The nominal `s=0` condition should remain close to the existing physics-only
oracle. A large PG-GTPP advantage at `s=0` is not required and should not be
manufactured.

## Simulator integrity fix (September 2026)

The persistent latent-state implementation now has two explicit invariants:

1. **Exact nominal limit:** when `unmodeled_state_strength == 0`, the event
   generator assigns the nominal physics competing-risk hazard directly. The
   latent-state branch is completely bypassed, so the zero-mismatch condition
   is the same physics-only simulator rather than a zero-event special case.
2. **Independent RNG streams:** latent-state innovations use the named
   `operational_state` RNG stream, while Ogata thinning continues to use the
   `event_thinning` stream. Therefore changing mismatch strength does not
   consume or shift thinning random numbers merely because the hidden state is
   being sampled.

The cause-specific perturbation is evaluated as

`lambda_true,c = lambda_physics,c * exp(strength * q_e(t) * s_c^T z_e(t))`,

where `q_e(t)` is the persistent hidden operational state, `z_e(t)` contains
battery/FIM/alignment stress components, and `s_c` is the fixed a-priori
cause-sensitivity row. The realized `q_e(t)` is not stored in the mission
artifacts or supplied to the model.

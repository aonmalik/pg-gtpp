# Final PG-GTPP modifications

## 1. Physics-conditioned residual fusion
`src/simulator/model/fusion.py` now feeds each cause head both the learned edge context and `log1p(physics_intensity)`.
This applies to both the correction head and trust gate. The neural branch therefore has explicit access to the scale of the physics prior it is correcting.

The positive-hazard fusion remains:

`lambda_pg = lambda_phys * exp((1-gate) * bounded_log_residual)`

so `gate=1` and `residual=0` preserve the physics prior exactly.

## 2. Correct fusion diagnostics
`scripts/diagnose_pg_gtpp_fusion.py` no longer reports `gate*physics` as a contribution. That decomposition is invalid for multiplicative residual fusion. It now reports the exact fixed-physics baseline and the signed learned intensity correction (`PG - physics`).

## 3. Important benchmark interpretation
The controlled mismatch experiment currently scales each cause by a constant multiplier. Constant cause-wise scaling changes calibration/rates much more than ranking, so a small AUROC delta is not evidence that the fusion is ineffective. A stronger robustness study should additionally use covariate-dependent misspecification, but that should be declared as a new benchmark condition rather than tuned after seeing the test results.

## 4. Validation
Run the supplied test suite in the project's conda environment. The current inspection environment lacks `pyarrow`, so full pytest collection cannot be completed here; all modified Python files were syntax-checked successfully.

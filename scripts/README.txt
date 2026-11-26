1. Back up your current runner.
2. The supplied run_main_fair_comparison_patch.py is the modified runner based on the uploaded current file. Copy it over scripts/run_main_fair_comparison.py after reviewing it.
3. Run a tiny smoke test with --n-missions 5 --seeds 42 --strengths 0 --n-steps 50 --quadrature-points 9 --only physics_only,pg_gtpp,gbt_residual.
4. Confirm *_test_losses.json files are created.
5. Run the full benchmark (or only the missing models/conditions).
6. Run: PYTHONPATH=src python scripts/paired_bootstrap.py --root results/main_fair_comparison --strengths 0,5 --seeds 42,43,44 --n-boot 10000

Important: the bootstrap is mission-level: overlapping windows from the same mission are first averaged, then missions are resampled. This avoids pretending overlapping windows are independent.

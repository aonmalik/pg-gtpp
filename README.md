

## Final integration pass
- Continuous-time TPP loss interface added.
- Experiment runner scaffold added.
- Regression test for numerical hazard integration added.


## Final integration update
The training path now supports future physics-driven intensity paths and competing-risk evaluation utilities.


## Publication-grade fair benchmark

The corrected benchmark is `scripts/run_fair_benchmark.py`. It uses a fixed
mission-level 70/15/15 split, first-`LINK_DOWN` failure target, proper
piecewise-constant continuous-time survival/competing-risk likelihood,
multi-horizon AUROC/AUPRC/Brier/ECE, IBS, C-index, mark MAE/RMSE, and five
default training seeds. See `docs/FAIR_BENCHMARK.md`.

```bash
PYTHONPATH=src python scripts/run_fair_benchmark.py \
  --n-missions 200 --n-steps 10000 --batch-size 16 \
  --seeds 7,17,27,37,47 --include-ablations
```


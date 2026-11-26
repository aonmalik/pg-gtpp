# PG-GTPP Paper 1 — Baseline Comparison Guide

This adds a **7-model literature comparison** to the existing 3-way ablation
(`physics_only` / `neural_only` / `pg_gtpp`). Every baseline is implemented
under `src/simulator/model/baselines/`, tested in `tests/test_baselines.py`,
and run end-to-end by `scripts/run_paper1_full_comparison.py`.

## READ THIS FIRST: GPU training here was not reproducible before 2026-08-20

Confirmed empirically: rerunning the identical configuration
(`--seed 7 --model-init-seed 1234`, same n-missions/n-steps/batch-size) gave
materially different results for any model touching the graph encoder or an
LSTM -- `dysat_style`'s PDR MAE differed by ~2x (0.093 vs. 0.179) between
two runs of "the same" config. `physics_only` reproduced exactly (its output
never touches learned weights at all -- see the ablation-flags section
below), which is what pinned this down to the encoder/LSTM path specifically
rather than something like non-deterministic mission generation.

Root cause: `graph_encoder.py` uses `scatter_add`/`scatter_reduce` for
attention aggregation, and `neural_hawkes.py` uses `nn.LSTM` -- both are
non-deterministic on CUDA by default regardless of `torch.manual_seed()`
(atomic-add thread ordering for the former, cuDNN's RNN backward pass for
the latter). **This means every single-run result in this document and in
your own uploaded results prior to this fix -- the original 10-way
comparison, the `mark-loss-weight` experiment, the `decouple-mark-head`
experiment -- was subject to this same unmeasured noise floor.** Treat
conclusions drawn from those runs as provisional, not settled.

**Fixed as of this version**: `run_paper1_full_comparison.py` now forces
deterministic CUDA algorithms by default (`torch.use_deterministic_
algorithms(True, warn_only=True)`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG` set before any CUDA
context exists). This costs some speed. If you see `UserWarning: ... does
not have a deterministic implementation` in stderr, that specific op still
has residual noise -- worth noting, not something to ignore. An escape
hatch (`--allow-nondeterministic`) exists for quick smoke tests where exact
numbers don't matter -- **never use it for anything you intend to report.**

**Before trusting anything else in this document**, verify the fix actually
works on your machine:
```bash
python scripts/run_paper1_full_comparison.py --only dysat_style --include-ablations \
    --n-missions 200 --n-steps 10000 --batch-size 16 --output-dir results/repro_check_a
python scripts/run_paper1_full_comparison.py --only dysat_style --include-ablations \
    --n-missions 200 --n-steps 10000 --batch-size 16 --output-dir results/repro_check_b
diff <(cat results/repro_check_a/dysat_style.json) <(cat results/repro_check_b/dysat_style.json)
```
`diff` should show nothing (or only float noise at the last decimal place).
If it still shows large differences, an op without a deterministic CUDA
implementation is the residual cause -- check stderr for the warning it
should have printed, and tell me what it says.

## Why these seven, specifically

Each baseline stands in for one row of the related-work gap-analysis table
from the brutal-review document (Phase 2), plus the classical statistical
control the identifiability discussion (Phase 5 of the same document)
explicitly contrasts PG-GTPP against. This is deliberate: a reviewer's first
question for any "we compare against baselines" claim is *why these
baselines and not others*, and "one per family already named in our own
related-work analysis" is a defensible, non-arbitrary answer.

| # | Registry key        | Class                     | Literature family                          | Graph? | Physics? | Temporal mechanism            |
|---|----------------------|---------------------------|---------------------------------------------|:------:|:--------:|--------------------------------|
| 1 | `cox_perlink`        | `CoxPerLinkModel`         | Classical statistical (Cox-type hazard)     | No     | Yes*     | None (static covariates)       |
| 2 | `rnn_perlink`         | `ShallowRNNPerLinkModel`  | Domain lit., shallow RNN, single-link       | No     | No       | Per-node GRU                   |
| 3 | `gnn_generic`         | `GenericGNNModel`         | Vanilla / TGAT-TGN's "no grounding" half    | Yes    | No       | GRU-pooled window               |
| 4 | `tgat_style`          | `TGATStyleModel`          | TGAT / TGN                                   | Yes    | No       | Time2Vec + attention            |
| 5 | `dysat_style`         | `DySATStyleModel`         | DySAT / Dynamic Graph Transformer            | Yes    | No       | Per-snapshot + temporal attn    |
| 6 | `neural_hawkes`       | `NeuralHawkesPerEdgeModel`| Neural TPP / Neural Hawkes                   | No     | No       | Per-edge joint LSTM             |
| 7 | `physics_soft_reg`    | `PhysicsSoftRegGNNModel`  | Physics-informed GNN (soft loss reg.)       | Yes    | Yes**    | GRU-pooled window                |

\* linear in the physics covariates, not the closed-form physics function itself.
\*\* physics enters only as a loss-level regularizer (`physics_regularized_loss`), never in the forward pass — the direct architectural contrast with PG-GTPP's structurally-separate, gated fusion.

## What is held constant across all 10 experiments (fair-comparison design)

Every one of the 3 ablations + 7 baselines trains against:
- the **same** mission corpus, seed, and mission-level train/val split
- the **same** loss (`training/loss.py::compute_loss` — continuous-time
  competing-risks TPP NLL + Beta/LogNormal mark NLL)
- the **same** mark decoder family (`model/mark_decoder.py::MarkDecoder`,
  reused unmodified by every baseline)
- the **same** evaluation code (`training/eval.py::evaluate` — AUC, ECE,
  survival ECE, PDR/throughput/delay MAE)

Only the *encoder* — how graph structure, time, and physics are (or are
not) used to produce an intensity — differs. This isolates the
architectural/formulation claim being made in the paper, rather than
confounding it with an unrelated difference in loss or mark modeling.

## Running it

```bash
cd pg-gtpp
pip install -r requirements.txt   # needs torch>=2.2; use a GPU if available
pytest tests/test_baselines.py -v   # ~1-2 min on CPU; run this FIRST

python scripts/run_paper1_full_comparison.py \
    --n-missions 200 --n-steps 10000 --batch-size 16 \
    --output-dir results/paper1_full_comparison \
    --include-ablations
```

This writes one `<name>.json` per experiment into `--output-dir`, in the
**exact same schema** as your existing `results/paper1_results/*.json`
(`experiment`, `parameters`, `val_loss`, `auc`, `ece`, `survival_ece`,
`pdr_mae`, `throughput_mae`, `delay_mae`, `n_edges`, `n_events`,
`training_steps`), plus a `_summary.json` collecting all of them. Runtime
scales with `--n-steps`; expect a GPU to be meaningfully faster given
`--n-steps 10000` across 7-10 models, though every model here is small
(tens of thousands to ~200K parameters), so CPU is workable for a smoke
test at reduced step counts.

### Ablation-flag update (resolved)

`PGGTPPModel` now takes `learned_correction` / `use_physics_intensity` /
`use_fusion_gate` construction flags (see `model/pg_gtpp_model.py` and
`model/fusion.py::CauseFusionHead`), mirroring `configs/paper1_baselines.yaml`
exactly:

```
physics_only: learned_correction=False, use_fusion_gate=False   # fused == physics intensity, gate forced to 1
neural_only:  use_physics_intensity=False, use_fusion_gate=False # fused == neural correction, gate forced to 0
pg_gtpp:      all True (default; numerically identical to the pre-flag model)
```

`--include-ablations` now trains and evaluates all three through this
script, on this script's own mission corpus/seed — giving a genuine,
single-corpus 10-way comparison table in one run. Covered by new tests in
`tests/test_baselines.py` (fused-output equivalence checks, parameter-count
checks, and an end-to-end forward+loss pass for all three flag sets) —
**run `pytest tests/test_baselines.py -v` after pulling this update, before
trusting the full run.**

### Remaining limitation, stated plainly

This run's `physics_only`/`neural_only`/`pg_gtpp` will **not** numerically
match your originally-uploaded `physics_only.json`/`neural_only.json`/
`pg_gtpp.json`, because they come from a different mission corpus and seed
(the original corpus/seed that produced those three files is not
recoverable from the uploaded artifacts — checkpoints retain only model +
optimizer state, not the run configuration that generated the data).

**Report this run's 10 results as one self-consistent table** (identical
held-out split throughout — this is the correct, reviewer-defensible
comparison). If you want a second table that includes your original three
numbers, keep it clearly separate and state which corpus/seed each table
uses — never merge results from two different held-out splits into a
single ranked table.

## If a run gets killed partway through (OOM)

Symptom: the process prints `Killed` with no Python traceback, partway
through the 10-model sweep — this is the Linux OOM-killer, not a bug in the
code. It happens because `run_paper1_full_comparison.py` originally trained
all 10 models back-to-back **inside one long-lived process**; PyTorch/glibc's
allocator does not reliably return freed memory to the OS between training
runs, so resident memory climbs model after model until it's killed —
typically right around `evaluate()`'s one-time full-validation-set batch
allocation, which is the largest single allocation in the whole run.

**Fix: run each model in its own fresh process**, via
`scripts/run_all_isolated.sh`:

```bash
bash scripts/run_all_isolated.sh
```

This calls `run_paper1_full_comparison.py --only <name> --skip-existing`
once per experiment, each in a brand-new process, so the OS fully reclaims
memory between models. It's also **resumable** — any `<name>.json` already
present in `--output-dir` is left alone, so if it gets interrupted (or one
model still fails) you can just run it again; only the missing model(s)
retrain. Per-model logs land in `results/paper1_full_comparison/logs/`.

Env vars override the defaults if needed, e.g. a faster smoke test:
```bash
N_STEPS=1000 bash scripts/run_all_isolated.sh
```

Note: each process regenerates the mission corpus from scratch (deterministic,
same `--seed` every time, so this doesn't change results — it's a time cost,
typically well under a minute per model, not a correctness concern).

Once you have `results/paper1_full_comparison/*.json` (7 baselines, plus
`pg_gtpp` and ideally `physics_only`/`neural_only` from the same corpus),
extend the existing `make_figures.py`:

1. Add each new experiment name to `ORDER` and a display name to `LABELS`.
2. Add a color per new bar to `BAR_COLORS` in `ieee_style.py` (10 distinct,
   colorblind-safe, print-safe colors — group by graph/no-graph or
   physics-grounded/not, e.g. via hatching, rather than 10 arbitrary hues,
   for print legibility).
3. Figures 1–5 (AUC, ECE, marks, radar, parameter-efficiency) all iterate
   `ORDER` already, so they extend to 10 bars/points with no structural
   changes — only re-check label rotation and figure width (a 10-bar chart
   needs `COL2`, not `COL1`, and angled tick labels).
4. Figures 6–7 (gate-convergence, trust-prior) are PG-GTPP/ablation-specific
   mechanism-analysis figures and should **not** be extended to the
   baselines (most have no gate to plot) — keep them as-is, scoped to the
   3 original ablations.

## Improving PG-GTPP's mark accuracy (diagnosis, not a guarantee)

Your full 10-way run surfaced a real, consistent pattern: PG-GTPP ranks #1
on AUC but #6/10 on every mark metric (PDR, delay, throughput MAE) and
#7/10 on calibration. The mechanism, found by comparing loss components in
your own training log, not guessed:

```
                tpp_loss   mark_loss
pg_gtpp          1.0050     9.7688
dysat_style      1.0080     7.6763    <- comparable TPP loss, much better marks
```

TPP quality is essentially tied between the two models; mark quality is not
-- PG-GTPP's mark loss is ~2 nats worse. The architectural difference:
every baseline in `model/baselines/` has exactly ONE intensity output
feeding its shared encoder. PG-GTPP's `TrustGatedFusion` has FOUR
cause-specific heads (`correction_mlp` + `gate_mlp` per cause = 8
sub-modules), all reading the identical `edge_context` tensor the mark
decoder also reads, unmodified. The shared encoder representation likely
gets pulled toward what the 8-head fusion module needs at the mark
decoder's expense.

**This is a hypothesis with real evidence behind it, not a proven fix.**
Two things worth trying, in order of cost:

### 1. `--mark-loss-weight` (cheap: hyperparameter only, zero architecture change)

```bash
python scripts/run_paper1_full_comparison.py --only pg_gtpp --include-ablations \
    --mark-loss-weight 2.0 --output-dir results/mark_weight_sweep \
    --n-missions 200 --n-steps 10000 --batch-size 16
```

Applies identically to every model if you run the full sweep with it too
(`compute_loss`'s `mark_loss_weight` is a plain multiplier on the mark NLL
term, `total = tpp_loss + mark_loss_weight * mark_loss`) -- so any
comparison stays apples-to-apples. Try a small sweep (1.0, 2.0, 4.0) on
just `pg_gtpp` first before committing to a full 10-model rerun at a new
weight.

### 2. `--decouple-mark-head` (architecture change: one small adapter layer, opt-in)

```bash
python scripts/run_paper1_full_comparison.py --only pg_gtpp_decoupled --include-ablations \
    --decouple-mark-head --output-dir results/paper1_full_comparison \
    --n-missions 200 --n-steps 10000 --batch-size 16
```

Inserts one `Linear(128,128)+ReLU` adapter between `edge_context` and the
mark decoder (see `model/pg_gtpp_model.py` docstring) -- standard
multi-task "shared trunk, per-task head" practice, not a new capability.
**Off by default** (`decouple_mark_head=False`), verified numerically
identical to the pre-flag model when off
(`test_decouple_mark_head_off_by_default_reproduces_old_behavior`). When
on, it writes to `pg_gtpp_decoupled.json` -- a distinct file, never
overwriting `pg_gtpp.json` -- so you always have both to compare and report.

### 3. `--mark-physics-skip` (architecture change: raw-physics skip connection, opt-in)

A third, independent hypothesis, motivated by the full 10-way comparison
result itself rather than a mechanism inside PG-GTPP: **DySAT-style has no
physics covariates at all and still beats PG-GTPP by a wide, robust margin
on every mark metric** (confirmed under the determinism fix -- see
`RESULTS_FINDINGS_DETERMINISTIC.md`). One explanation: PG-GTPP's mark
decoder only ever sees physics-relevant signal (SNR margin, wave state,
etc.) after it's been compressed through the 2-layer graph encoder's
abstraction, whereas a simpler model's mark accuracy might come from
picking up short-term per-edge dynamics some other way. This flag tests the
narrow version of that idea directly: give the mark decoder a skip
connection straight to the raw `edge_physics` vector (the same 4 values
already fed to the fusion module), bypassing the encoder bottleneck for
that one signal.

```bash
python scripts/run_paper1_full_comparison.py \
    --only pg_gtpp_mark_skip --include-ablations \
    --mark-physics-skip \
    --n-missions 200 --n-steps 10000 --batch-size 16 \
    --output-dir results/paper1_full_comparison_deterministic
```

Writes to `pg_gtpp_mark_skip.json` -- distinct file, never touches
`pg_gtpp.json`. **Off by default**, verified numerically identical to the
pre-flag model when off. Independent of `--decouple-mark-head` (the two can
technically be combined -- tested in
`test_mark_physics_skip_and_decouple_mark_head_compose_cleanly` -- but this
project runs them one at a time so either result stays interpretable on its
own; don't stack speculative changes and then attribute the outcome to just
one of them).

**Run the tests before spending compute on any of the three:**
```bash
pytest tests/test_baselines.py -v -k "decouple or mark_loss or mark_physics_skip"
```

### What NOT to do with these three knobs

- Don't only retrain `pg_gtpp` with a tuned `--mark-loss-weight` and leave
  the other 9 models at the old weight, then put them in the same table --
  that's no longer apples-to-apples. If you adopt a new weight, rerun
  everything at that weight.
- Don't treat "marks improved" as "ready to publish" without at least a
  couple of seeds -- a single-run improvement of a few hundredths of a MAE
  point could easily be noise, especially given `physics_soft_reg`'s
  per-step `reg` loss values alone showed run-to-run swings of ~0.1-0.2 in
  your log at a FIXED configuration.
- Don't run all three fixes and report only whichever one happened to win
  on this validation split without a seed check -- that is test-set
  overfitting via architecture search, exactly what the anti-stacking
  principle in the brutal-review roadmap document exists to prevent.
- If none of the three narrow the gap to DySAT-style, that's still a
  useful, reportable result -- "we investigated the mark-accuracy gap under
  three independent hypotheses and it persisted" is a legitimate,
  citable paragraph, not a dead end to hide.

## Is the AUC ranking real? (seed sweep)

**Important update**: the first seed sweep run (before the determinism fix
above existed) was contaminated by GPU non-determinism -- the between-seed
spread it measured mixed genuine weight-init variance with run-to-run CUDA
noise that would have existed even at a single fixed seed. Re-run the sweep
with the fixed script before trusting its output; the interpretation below
still applies, but only to a sweep run after the fix.

A single run per model cannot tell you whether PG-GTPP's AUC edge over
Neural Hawkes (0.5645 vs. 0.5614, a 0.003 gap) or DySAT-style (0.5507, a
0.014 gap) is real or noise. `scripts/run_seed_sweep.sh` answers this as
cheaply as possible: it reruns a small model set across multiple weight-init
seeds, holding the mission corpus AND training batch order fixed
(`--seed`), so only initialization varies -- see the script's header
comment for the full scoping statement (this is deliberately narrower than
a "vary everything" robustness check, to keep it fast and interpretable).
This axis is only meaningful once run-to-run noise at a FIXED seed is
controlled -- i.e., after the reproducibility check two sections up passes.

```bash
# fast first pass -- do this before the full 10k-step version
N_STEPS=3000 bash scripts/run_seed_sweep.sh

# once that looks sane, the real one (3 seeds x 4 models, full 10k steps)
bash scripts/run_seed_sweep.sh

# more seeds if compute allows -- 5 is better than 3 for the overlap check
MODEL_SEEDS="1234 2345 3456 4567 5678" bash scripts/run_seed_sweep.sh

# then summarize
python scripts/summarize_seed_sweep.py --input-root results/seed_sweep
```

Default model set: `pg_gtpp`, `dysat_style`, `neural_hawkes` (the three
closest AUC contenders) plus `physics_only` as a noise-floor sanity check —
if physics_only's AUC swings across seeds as much as the top three do,
that's a red flag about the corpus/eval setup itself, worth resolving
before trusting any ranking conclusion. Override via `MODELS="..."` if you
want to add more.

The summarizer reports mean ± std per model per metric and a plain-language
overlap check on the top-2 AUC ranges — **not a p-value**, deliberately,
since 3-5 seeds don't support a real significance test. If the #1 and #2
ranges overlap, the honest conclusion is "this sweep can't distinguish
them yet," and that's what the script will tell you — don't override it
with a ranking claim in the paper if it says that.

Runtime: same per-run cost as your full comparison (~1000-1300s per model
at `--n-steps 10000` on your GPU machine) × `len(MODELS)` × `len(MODEL_SEEDS)`
— e.g. 4 models × 3 seeds ≈ 12 runs ≈ 3-4 hours sequential. Use `N_STEPS=3000`
for a same-day first pass; only commit to the full 10k-step version once
that looks directionally sane.

## What to claim, and what not to claim, from this comparison

- **A baseline losing on AUC/ECE to PG-GTPP is expected and worth
  reporting plainly** — that is the point of the comparison.
- **A baseline losing on mark MAE is weaker evidence** — Fig. 3's own
  finding (ablation results) was that mark MAE barely moves across very
  different architectures on this corpus; the same caution applies here.
  Don't lean on mark MAE as your headline baseline-comparison result.
- **`cox_perlink` doing surprisingly well on some metric is not an
  embarrassment to hide** — a strong classical baseline is exactly what a
  Q1 reviewer wants to see included, and it directly supports the
  identifiability discussion (Phase 5, brutal-review doc) rather than
  undermining it. Report it straight.
- **Do not claim statistical significance from a single seeded run per
  model.** Exactly as flagged for the original 3-way ablation
  (`figure_notes.md`), report these as point estimates from one run
  unless you re-run with multiple seeds (the repositioning-heuristic
  document's own recommendation, §5.4, is ≥20 seeded runs per condition —
  a strong target for a camera-ready revision, not a requirement to
  reproduce this exact number here, but at minimum 3-5 seeds per model
  before submission if compute allows).

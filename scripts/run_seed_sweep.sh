#!/usr/bin/env bash
# scripts/run_seed_sweep.sh
#
# Answers one question cheaply before committing to a full 10-model x
# N-seed sweep: does PG-GTPP's AUC win over its closest competitors survive
# multiple random weight initializations, or was the single-run ranking
# noise?
#
# Scope, stated plainly: this varies ONLY the model weight-initialization
# seed (--model-init-seed). The mission corpus and the training batch order
# are both driven by --seed (see build_mission_config() and TrainConfig.seed
# in run_paper1_full_comparison.py), which this script holds FIXED across
# the sweep -- so corpus and batch order are identical for every seed run of
# a given model. That isolates weight-init variance specifically; it does
# NOT test sensitivity to a different data corpus or a different batch
# order. That is a deliberate scoping choice to keep this fast and its
# result easy to interpret, not a claim that weight-init variance is the
# only kind of variance worth checking eventually.
#
# Default model set: pg_gtpp and pg_gtpp_mark_skip (the two variants that
# matter most right now -- pg_gtpp_mark_skip showed a large single-run
# improvement on marks/delay-MAE and needs seed validation before it's
# trusted), plus the two closest literature-baseline AUC contenders
# (neural_hawkes, dysat_style) and physics_only as a noise-floor sanity
# check -- physics_only is expected to reliably score badly regardless of
# seed; if ITS AUC swings across seeds as much as the others do, that is a
# red flag about the corpus/eval setup itself, not just about PG-GTPP, and
# should be resolved before trusting any seed-sweep conclusion.
#
# Safe to re-run / resumable: each (model, seed) pair writes to its own
# results/seed_sweep/seed_<seed>/<model>.json; already-present files are
# skipped, so an interrupted sweep picks up where it left off.
#
# Usage:
#   bash scripts/run_seed_sweep.sh                          # default: 3 seeds, 5 models, full 10k steps
#   N_STEPS=3000 bash scripts/run_seed_sweep.sh              # faster first pass -- do this FIRST
#   MODEL_SEEDS="1234 2345 3456 4567 5678" bash scripts/run_seed_sweep.sh   # 5 seeds instead of 3
#   MODELS="pg_gtpp pg_gtpp_mark_skip" bash scripts/run_seed_sweep.sh      # narrower model set
set -uo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-results/seed_sweep}"
DATA_SEED="${DATA_SEED:-7}"                        # mission corpus + batch order: held FIXED
MODEL_SEEDS="${MODEL_SEEDS:-1234 2345 3456}"        # weight-init seeds to sweep
MODELS="${MODELS:-pg_gtpp pg_gtpp_mark_skip neural_hawkes dysat_style physics_only}"
N_MISSIONS="${N_MISSIONS:-200}"
N_STEPS="${N_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

mkdir -p "$OUTPUT_ROOT"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

TOTAL=0
FAILED=()

# Some model names are opt-in architecture variants of pg_gtpp that only
# get added to run_paper1_full_comparison.py's selectable set when their
# corresponding flag is passed (see that script's --only / --include-
# ablations logic) -- map model name -> extra CLI flag(s) here so this
# script stays a single source of truth for "which models exist."
extra_flags_for() {
    case "$1" in
        pg_gtpp_mark_skip) echo "--mark-physics-skip" ;;
        pg_gtpp_decoupled) echo "--decouple-mark-head" ;;
        *) echo "" ;;
    esac
}

for seed in $MODEL_SEEDS; do
    out_dir="$OUTPUT_ROOT/seed_${seed}"
    mkdir -p "$out_dir"
    for model in $MODELS; do
        out_path="$out_dir/${model}.json"
        TOTAL=$((TOTAL + 1))
        if [ -f "$out_path" ]; then
            echo "skip $model @ model-init-seed=$seed: $out_path already exists"
            continue
        fi
        echo "=============================================================="
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $model @ model-init-seed=$seed (data-seed=$DATA_SEED fixed)"
        echo "=============================================================="
        # shellcheck disable=SC2046
        python scripts/run_paper1_full_comparison.py \
            --only "$model" --include-ablations $(extra_flags_for "$model") \
            --seed "$DATA_SEED" --model-init-seed "$seed" \
            --n-missions "$N_MISSIONS" --n-steps "$N_STEPS" --batch-size "$BATCH_SIZE" \
            --output-dir "$out_dir" \
            2>&1 | tee "$LOG_DIR/${model}_seed${seed}.log"
        status=${PIPESTATUS[0]}
        if [ "$status" -ne 0 ]; then
            echo "WARNING: $model @ seed $seed exited with status $status -- see $LOG_DIR/${model}_seed${seed}.log"
            FAILED+=("${model}@${seed}")
        fi
    done
done

echo ""
echo "=============================================================="
echo "Seed sweep complete: $((TOTAL - ${#FAILED[@]}))/$TOTAL runs succeeded."
echo "=============================================================="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed: ${FAILED[*]} -- check logs, then re-run this script (resumable)."
fi
echo "Results are under $OUTPUT_ROOT/seed_<seed>/<model>.json"
echo "Next: python scripts/summarize_seed_sweep.py --input-root $OUTPUT_ROOT"

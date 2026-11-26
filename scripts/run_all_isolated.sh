#!/usr/bin/env bash
# scripts/run_all_isolated.sh
#
# Runs each of the 7 baselines + 3 ablations in its OWN fresh OS process,
# one at a time, instead of looping over all 10 inside one long-lived
# Python process. This exists specifically because sequential CPU training
# runs inside a single process tend to accumulate resident memory that the
# allocator does not hand back to the OS between runs (glibc/PyTorch
# fragmentation) -- harmless for 2-3 models, but can eventually trigger an
# OOM kill ("Killed", no traceback) partway through a 10-model sweep, as
# happened after model #4 in the original single-process run. A fresh
# process per model means the OS fully reclaims memory before the next one
# starts, at the cost of repeating mission-corpus generation once per model
# (deterministic, same --seed every time, so this does not change results
# -- it is a time cost, not a correctness concern).
#
# Safe to re-run: --skip-existing means any model whose <name>.json already
# exists in --output-dir is left alone, so if this script is interrupted or
# one model still gets killed, just run it again -- completed models are
# skipped and only the missing one(s) retrain.
#
# Usage:
#   bash scripts/run_all_isolated.sh
#   N_STEPS=1000 bash scripts/run_all_isolated.sh          # faster smoke test
#   OUTPUT_DIR=results/foo bash scripts/run_all_isolated.sh
set -uo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/paper1_full_comparison}"
N_MISSIONS="${N_MISSIONS:-200}"
N_STEPS="${N_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# Order matters only for readability of the log -- each is fully independent.
NAMES=(cox_perlink rnn_perlink gnn_generic tgat_style dysat_style neural_hawkes physics_soft_reg physics_only neural_only pg_gtpp)

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

FAILED=()

for name in "${NAMES[@]}"; do
    echo "=============================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $name"
    echo "=============================================================="
    python scripts/run_paper1_full_comparison.py \
        --only "$name" \
        --include-ablations \
        --skip-existing \
        --n-missions "$N_MISSIONS" --n-steps "$N_STEPS" --batch-size "$BATCH_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_DIR/${name}.log"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $name exited with status $status (see $LOG_DIR/${name}.log) -- continuing with the next model."
        FAILED+=("$name")
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $name"
    fi
done

echo ""
echo "=============================================================="
echo "All 10 attempts complete."
echo "=============================================================="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "The following did NOT complete successfully: ${FAILED[*]}"
    echo "Check $LOG_DIR/<name>.log for each, then re-run this script -- it will"
    echo "skip everything that already succeeded and retry only these."
else
    echo "All models completed successfully. Per-experiment JSON files are in $OUTPUT_DIR/"
fi

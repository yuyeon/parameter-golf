#!/usr/bin/env bash
# Master Tier 1 experiment runner
# Runs baseline + 5 experiments sequentially at 64M tokens
# Survives SSH disconnect: run with nohup
#
# Usage: nohup bash experiments/scripts/03_tier1_all.sh > experiments/logs/orchestrator.log 2>&1 &
set -uo pipefail

cd /root/parameter-golf

SOTA_SCRIPT="records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"
RESULTS_BASE="experiments/results"
LOG="experiments/logs/orchestrator.log"

# Common env vars for 64M matched-token budget
# 82 steps × 786,432 = 64.4M tokens
export ITERATIONS=82
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=0
export TTT_ENABLED=0
export WARMDOWN_ITERS=25
export WARMUP_STEPS=5
export VAL_LOSS_EVERY=80
export TRAIN_LOG_EVERY=10
export SEED=42

timestamp() { date -Iseconds; }

run_experiment() {
    local NAME="$1"
    local SCRIPT="$2"
    shift 2
    # Remaining args are extra env overrides: KEY=VAL pairs
    local EXPDIR="${RESULTS_BASE}/${NAME}"
    mkdir -p "$EXPDIR"

    echo ""
    echo "================================================================"
    echo "[$(timestamp)] STARTING: $NAME"
    echo "  Script: $SCRIPT"
    echo "  Extra env: $*"
    echo "================================================================"

    # Apply extra env vars
    for kv in "$@"; do
        export "$kv"
    done

    python3 "$SCRIPT" 2>&1 | tee "${EXPDIR}/train.log"
    local EXIT_CODE=${PIPESTATUS[0]}

    # Unset extra env vars to avoid leaking to next experiment
    for kv in "$@"; do
        local KEY="${kv%%=*}"
        unset "$KEY"
    done

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(timestamp)] COMPLETED: $NAME (exit 0)"
        # Extract metrics
        grep -E "(pre_quant_val|final_int8|step_avg|Serialized|model_params)" \
            "${EXPDIR}/train.log" | tail -10 > "${EXPDIR}/metrics.txt" 2>/dev/null
        echo "DONE" > "${EXPDIR}/.done"
    else
        echo "[$(timestamp)] FAILED: $NAME (exit $EXIT_CODE)"
        echo "FAILED:${EXIT_CODE}" > "${EXPDIR}/.done"
    fi

    # Move any checkpoints into experiment dir
    for f in final_model.pt final_model.int8.ptz final_model.int6.lzma; do
        [ -f "$f" ] && mv "$f" "${EXPDIR}/" 2>/dev/null
    done

    echo ""
    return $EXIT_CODE
}

echo "================================================================"
echo "[$(timestamp)] TIER 1 EXPERIMENT SUITE STARTING"
echo "  Budget: 64M tokens (82 steps × 786432)"
echo "  Seed: 42"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "================================================================"

# ---- Experiment 0: Baseline (SOTA as-is) ----
run_experiment "baseline_64m" "$SOTA_SCRIPT" || true

# ---- Experiment 1: MTP (multi-token prediction) ----
# Already implemented, just enable via env
run_experiment "exp_mtp_64m" "$SOTA_SCRIPT" \
    "MTP_NUM_HEADS=1" "MTP_LOSS_WEIGHT=0.15" || true

# ---- Experiment 2: Larger BigramHash ----
# 3072 buckets vs 2048 default
run_experiment "exp_bigram3k_64m" "$SOTA_SCRIPT" \
    "BIGRAM_VOCAB_SIZE=3072" || true

# ---- Experiment 3: More layers (12) ----
# Test if 12 layers improves at same token budget
# May need reduced mlp_mult or quantization changes to fit 16MB
run_experiment "exp_12layer_64m" "$SOTA_SCRIPT" \
    "NUM_LAYERS=12" || true

# ---- Experiment 4: Wider model (576 dim) ----
# Increase model width from 512 to 576
run_experiment "exp_wider576_64m" "$SOTA_SCRIPT" \
    "MODEL_DIM=576" "NUM_HEADS=9" "NUM_KV_HEADS=3" || true

# ---- Experiment 5: Longer warmdown ----
# 40 steps warmdown (49% of training) vs 25 (30%)
run_experiment "exp_warmdown40_64m" "$SOTA_SCRIPT" \
    "WARMDOWN_ITERS=40" || true

echo ""
echo "================================================================"
echo "[$(timestamp)] TIER 1 SUITE COMPLETE"
echo "================================================================"

# ---- Summary ----
echo ""
echo "=== RESULTS SUMMARY ==="
echo "Experiment               | pre_quant_val_bpb"
echo "-------------------------|------------------"
for dir in ${RESULTS_BASE}/baseline_64m ${RESULTS_BASE}/exp_*_64m; do
    NAME=$(basename "$dir")
    if [ -f "${dir}/metrics.txt" ]; then
        BPB=$(grep -o "pre_quant_val_bpb:[0-9.]*" "${dir}/metrics.txt" 2>/dev/null | tail -1 | cut -d: -f2)
        printf "%-25s| %s\n" "$NAME" "${BPB:-MISSING}"
    else
        printf "%-25s| %s\n" "$NAME" "NO METRICS"
    fi
done | tee "${RESULTS_BASE}/tier1_summary.txt"

echo ""
echo "Full summary saved to ${RESULTS_BASE}/tier1_summary.txt"
echo "TIER1_ALL_DONE" > "${RESULTS_BASE}/.tier1_done"

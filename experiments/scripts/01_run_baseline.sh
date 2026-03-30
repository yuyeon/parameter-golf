#!/usr/bin/env bash
# Run SOTA baseline at 64M tokens on 1 GPU
# Survives SSH disconnect via nohup
set -euo pipefail

cd /root/parameter-golf
EXPDIR="experiments/results/baseline_64m"
mkdir -p "$EXPDIR"
LOG="$EXPDIR/train.log"

echo "=== Baseline 64M started at $(date -Iseconds) ===" | tee "$LOG"

# 64M tokens: 82 steps × 786432 tokens/step = 64.4M
# WARMDOWN_ITERS=25 (~30% of steps)
# TTT disabled for speed
# MAX_WALLCLOCK_SECONDS=0 disables wallclock cap
ITERATIONS=82 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
MAX_WALLCLOCK_SECONDS=0 \
TTT_ENABLED=0 \
WARMDOWN_ITERS=25 \
WARMUP_STEPS=5 \
VAL_LOSS_EVERY=80 \
TRAIN_LOG_EVERY=10 \
SEED=42 \
  python3 records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py \
  2>&1 | tee -a "$LOG"

echo "=== Baseline 64M finished at $(date -Iseconds) ===" | tee -a "$LOG"

# Extract key metrics
echo ""
echo "=== KEY METRICS ==="
grep -E "(pre_quant_val_bpb|pre_quant_val_loss|artifact_bytes|step_avg|Serialized)" "$LOG" | tail -5 | tee "$EXPDIR/metrics.txt"
echo "BASELINE_DONE" > "$EXPDIR/.done"

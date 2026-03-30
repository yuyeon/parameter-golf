#!/usr/bin/env bash
# Quick progress checker — run this anytime to see experiment status
# Usage: bash experiments/scripts/check_progress.sh
cd /root/parameter-golf

echo "=== EXPERIMENT PROGRESS ($(date -Iseconds)) ==="
echo ""

# Check if orchestrator is running
if pgrep -f "run_all.sh" > /dev/null 2>&1; then
    echo "Orchestrator: RUNNING (PID $(pgrep -f 'run_all.sh' | head -1))"
else
    echo "Orchestrator: NOT RUNNING"
fi
echo ""

# Phase status
echo "--- Phase Status ---"
[ -f experiments/logs/.setup_done ] && echo "  Setup:     DONE" || echo "  Setup:     pending"
[ -f experiments/results/baseline_64m/.done ] && echo "  Baseline:  DONE" || echo "  Baseline:  pending"
[ -f experiments/results/tier0_compression/.done ] && echo "  Tier 0:    DONE" || echo "  Tier 0:    pending"
[ -f experiments/results/.tier1_done ] && echo "  Tier 1 (env): DONE" || echo "  Tier 1 (env): pending"
[ -f experiments/results/.code_mods_done ] && echo "  Tier 1 (code): DONE" || echo "  Tier 1 (code): pending"
echo ""

# Results table
echo "--- BPB Results ---"
printf "%-28s| %-18s| %s\n" "Experiment" "pre_quant_val_bpb" "Status"
echo "----------------------------|------------------|--------"
for dir in experiments/results/baseline_64m experiments/results/exp_*_64m; do
    [ -d "$dir" ] || continue
    NAME=$(basename "$dir")
    if [ -f "${dir}/.done" ]; then
        STAT=$(cat "${dir}/.done" | head -1)
    else
        # Check if currently running by looking at train.log modification time
        if [ -f "${dir}/train.log" ]; then
            STAT="RUNNING"
        else
            STAT="pending"
        fi
    fi
    BPB="—"
    if [ -f "${dir}/metrics.txt" ]; then
        BPB=$(grep -o "pre_quant_val_bpb:[0-9.]*" "${dir}/metrics.txt" 2>/dev/null | tail -1 | cut -d: -f2)
    elif [ -f "${dir}/train.log" ]; then
        # Try to get latest BPB from live log
        BPB=$(grep -o "pre_quant_val_bpb:[0-9.]*" "${dir}/train.log" 2>/dev/null | tail -1 | cut -d: -f2)
    fi
    printf "%-28s| %-18s| %s\n" "$NAME" "${BPB:-—}" "$STAT"
done
echo ""

# Current training activity
ACTIVE_LOG=""
for dir in experiments/results/*/; do
    LOG="${dir}train.log"
    if [ -f "$LOG" ] && [ ! -f "${dir}.done" ]; then
        ACTIVE_LOG="$LOG"
        break
    fi
done
if [ -n "$ACTIVE_LOG" ]; then
    echo "--- Active Training (last 5 lines) ---"
    tail -5 "$ACTIVE_LOG"
    echo ""
fi

# GPU utilization
echo "--- GPU Status ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  No GPU info"
echo ""

# Tier 0 summary (if available)
if [ -f experiments/results/tier0_compression/results.json ]; then
    echo "--- Tier 0 Compression Results ---"
    python3 -c "
import json
with open('experiments/results/tier0_compression/results.json') as f:
    r = json.load(f)
bp = r['bitpacking']
ab = r['adaptive_bitwidth']
print(f'  Bitpacking savings (raw): {bp[\"raw_savings_bytes\"]/1e6:.2f} MB')
print(f'  Adaptive bitwidth savings: {ab[\"savings_bytes\"]/1e6:.2f} MB')
print(f'  Bitwidth distribution: {ab[\"bitwidth_distribution\"]}')
" 2>/dev/null
    echo ""
fi

# Status log tail
if [ -f experiments/logs/status.txt ]; then
    echo "--- Recent Status Updates ---"
    tail -5 experiments/logs/status.txt
fi

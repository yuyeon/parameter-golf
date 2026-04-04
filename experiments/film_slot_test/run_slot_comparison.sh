#!/bin/bash
# Compare: no SLOT vs standard SLOT vs causal SLOT on FiLM 5→7+8xMLP
# Runs in isolated working directory to avoid file contention
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export C_INCLUDE_PATH="$HOME/.local/include:$HOME/.local/include/python3.10"
export CPATH="$HOME/.local/include:$HOME/.local/include/python3.10"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO_ROOT/experiments/film_slot/train_gpt.py"
WORKDIR="$REPO_ROOT/experiments/film_slot_test/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Working directory: $WORKDIR ==="

# Common env vars
export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export NUM_SHARED_BLOCKS=5
export NUM_LAYERS=7
export MLP_MULT=8
export SEED=42
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_TOKENS=524288
export TRAIN_LOG_EVERY=200
export VAL_LOSS_EVERY=0
export USE_INT6=1

echo ""
echo "============================================"
echo "  PHASE 1: Train FiLM 5→7+8xMLP (600s)"
echo "  No SLOT — just train + int6 quant eval"
echo "============================================"

export SLOT_ENABLED=0
export RUN_ID="film_noslot_$$"
python3 "$SCRIPT" 2>&1 | tee train.log

echo ""
echo "=== Training complete. Checkpoint saved. ==="
grep -E "step_avg|val_bpb|int6|int8" train.log | tail -10

# The training saved final_model.pt (pre-quant weights with EMA applied)
CKPT="$WORKDIR/final_model.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: final_model.pt not found!"
    exit 1
fi

echo ""
echo "============================================"
echo "  PHASE 2: Standard SLOT eval"
echo "  (optimizes on all scored positions)"
echo "============================================"

# Run in sub-directory to avoid overwriting int6 files
mkdir -p "$WORKDIR/standard_slot"
cd "$WORKDIR/standard_slot"

export SLOT_ENABLED=1
export CAUSAL_SLOT=0
export SLOT_STEPS=24
export SLOT_LR=0.012
export SLOT_LR_MIN=0.001
export SLOT_BATCH_SEQS=32
export EVAL_STRIDE=96
export LOAD_CHECKPOINT="$CKPT"
export RUN_ID="film_standard_slot_$$"
python3 "$SCRIPT" 2>&1 | tee standard_slot.log

echo ""
echo "=== Standard SLOT complete ==="
grep -E "slot|int6" standard_slot.log | tail -5

echo ""
echo "============================================"
echo "  PHASE 3: Causal SLOT eval"
echo "  (optimizes on already-scored context only)"
echo "============================================"

mkdir -p "$WORKDIR/causal_slot"
cd "$WORKDIR/causal_slot"

export CAUSAL_SLOT=1
export RUN_ID="film_causal_slot_$$"
python3 "$SCRIPT" 2>&1 | tee causal_slot.log

echo ""
echo "=== Causal SLOT complete ==="
grep -E "slot|int6" causal_slot.log | tail -5

echo ""
echo "============================================"
echo "  FINAL RESULTS SUMMARY"
echo "============================================"
echo ""
echo "--- No SLOT (int6 post-quant) ---"
grep "final_int6_roundtrip_exact" "$WORKDIR/train.log" 2>/dev/null || grep "final_int6_roundtrip " "$WORKDIR/train.log"
echo ""
echo "--- Standard SLOT (int6 + SLOT) ---"
grep "final_slot" "$WORKDIR/standard_slot/standard_slot.log" | tail -2
echo ""
echo "--- Causal SLOT (int6 + causal SLOT) ---"
grep "final_slot" "$WORKDIR/causal_slot/causal_slot.log" | tail -2

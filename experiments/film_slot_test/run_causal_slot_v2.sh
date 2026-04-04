#!/bin/bash
# Test improved causal SLOT v2 with focal context, warm-start, clamping
# Uses L-BFGS logit-only mode with PR #1350 hyperparameters
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export C_INCLUDE_PATH="$HOME/.local/include:$HOME/.local/include/python3.10"
export CPATH="$HOME/.local/include:$HOME/.local/include/python3.10"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO_ROOT/experiments/film_slot/train_gpt.py"
CKPT="$REPO_ROOT/experiments/film_slot_test/run_20260404_213815/final_model.pt"
WORKDIR="$REPO_ROOT/experiments/film_slot_test/causal_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORKDIR"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

# Common env vars
export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export NUM_SHARED_BLOCKS=5
export NUM_LAYERS=7
export MLP_MULT=8
export SEED=42
export MAX_WALLCLOCK_SECONDS=0
export ITERATIONS=200
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_TOKENS=524288
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=200
export SLOT_ENABLED=1
export CAUSAL_SLOT=1
export SLOT_LR=0.012
export SLOT_LR_MIN=0.001
export SLOT_BATCH_SEQS=32
export EVAL_STRIDE=96
export LOAD_CHECKPOINT="$CKPT"
export USE_INT6=1

run_variant() {
    local name="$1"
    shift
    echo ""
    echo "============================================"
    echo "  $name"
    echo "============================================"
    local dir="$WORKDIR/$name"
    mkdir -p "$dir"
    cd "$dir"
    export RUN_ID="${name}_$$"
    # Apply all extra env vars
    for var in "$@"; do
        export "$var"
    done
    python3 "$SCRIPT" 2>&1 | tee "${name}.log"
    grep -E "final_slot|ERROR" "${name}.log" | tail -3
    echo ""
}

# V2 variants — all use lbfgs_logit as base
# A: Full PR #1350 approach (focal=128, warmstart, clamp=5, steps=25)
run_variant "v2_full" \
    SLOT_MODE=lbfgs_logit SLOT_STEPS=25 SLOT_FOCAL_CTX=128 \
    SLOT_WARMSTART=1 SLOT_CLAMP=5.0 SLOT_LBFGS_HISTORY=20

# B: More steps (50) with same settings
run_variant "v2_50steps" \
    SLOT_MODE=lbfgs_logit SLOT_STEPS=50 SLOT_FOCAL_CTX=128 \
    SLOT_WARMSTART=1 SLOT_CLAMP=5.0 SLOT_LBFGS_HISTORY=20

# C: No focal (all context) for comparison
run_variant "v2_nofocal" \
    SLOT_MODE=lbfgs_logit SLOT_STEPS=25 SLOT_FOCAL_CTX=0 \
    SLOT_WARMSTART=1 SLOT_CLAMP=5.0 SLOT_LBFGS_HISTORY=20

# D: AdamW logit-only with focal+warmstart+clamp (compare optimizer)
run_variant "v2_adamw" \
    SLOT_MODE=logit_only SLOT_STEPS=25 SLOT_FOCAL_CTX=128 \
    SLOT_WARMSTART=1 SLOT_CLAMP=5.0

echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
echo ""
echo "Baseline (no SLOT, int6): 1.3003 BPB"
echo "v1 causal (AdamW delta+bias, 24 steps): 1.3095 BPB (+0.009, HURTS)"
echo "lbfgs_logit (4 steps, no focal/warmstart): 1.2658 BPB (-0.035)"
echo ""
for name in v2_full v2_50steps v2_nofocal v2_adamw; do
    log="$WORKDIR/$name/${name}.log"
    if [ -f "$log" ]; then
        echo "--- $name ---"
        grep "final_slot.*_exact" "$log" 2>/dev/null || grep "final_slot" "$log" 2>/dev/null | tail -1 || echo "FAILED"
    fi
done

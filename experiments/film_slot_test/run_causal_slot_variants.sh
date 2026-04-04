#!/bin/bash
# Test causal SLOT variants on existing FiLM checkpoint
# Compares: v1 (original), logit_only, lbfgs, lbfgs_logit
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export C_INCLUDE_PATH="$HOME/.local/include:$HOME/.local/include/python3.10"
export CPATH="$HOME/.local/include:$HOME/.local/include/python3.10"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO_ROOT/experiments/film_slot/train_gpt.py"
CKPT="$REPO_ROOT/experiments/film_slot_test/run_20260404_213815/final_model.pt"
WORKDIR="$REPO_ROOT/experiments/film_slot_test/causal_variants_$(date +%Y%m%d_%H%M%S)"
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
export SLOT_STEPS=24
export SLOT_LR=0.012
export SLOT_LR_MIN=0.001
export SLOT_BATCH_SEQS=32
export EVAL_STRIDE=96
export LOAD_CHECKPOINT="$CKPT"
export USE_INT6=1

run_variant() {
    local name="$1"
    local mode="$2"
    local extra_env="${3:-}"

    echo ""
    echo "============================================"
    echo "  Causal SLOT: $name (mode=$mode)"
    echo "============================================"

    local dir="$WORKDIR/$name"
    mkdir -p "$dir"
    cd "$dir"
    export SLOT_MODE="$mode"
    export RUN_ID="${name}_$$"
    eval "$extra_env"
    python3 "$SCRIPT" 2>&1 | tee "${name}.log"
    grep -E "final_slot|ERROR" "${name}.log" | tail -3
    echo ""
}

# Also run standard SLOT with logit_only and lbfgs for reference
run_variant "causal_v1" "v1"
run_variant "causal_logit_only" "logit_only"
run_variant "causal_lbfgs" "lbfgs"
run_variant "causal_lbfgs_logit" "lbfgs_logit"

# Also test standard (non-causal) lbfgs_logit for comparison
export CAUSAL_SLOT=0
run_variant "standard_lbfgs_logit" "lbfgs_logit"
run_variant "standard_v1" "v1"

echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
echo ""
echo "Baseline (no SLOT, int6): 1.3003 BPB"
echo ""
for name in causal_v1 causal_logit_only causal_lbfgs causal_lbfgs_logit standard_lbfgs_logit standard_v1; do
    log="$WORKDIR/$name/${name}.log"
    if [ -f "$log" ]; then
        echo "--- $name ---"
        grep "final_slot.*_exact" "$log" 2>/dev/null || grep "final_slot" "$log" 2>/dev/null | tail -1 || echo "FAILED"
    fi
done

#!/bin/bash
# Quick 200-step screen: FiLM 5→7+8xMLP with SP4096 vs SP1024
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export C_INCLUDE_PATH="$HOME/.local/include:$HOME/.local/include/python3.10"
export CPATH="$HOME/.local/include:$HOME/.local/include/python3.10"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO_ROOT/experiments/film_slot/train_gpt.py"
WORKDIR="$REPO_ROOT/experiments/film_slot_test/sp4096_screen_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORKDIR"

# Common settings
export NUM_SHARED_BLOCKS=5
export NUM_LAYERS=7
export MLP_MULT=8
export SEED=42
export ITERATIONS=200
export MAX_WALLCLOCK_SECONDS=0
export TRAIN_BATCH_TOKENS=524288
export TRAIN_LOG_EVERY=50
export VAL_LOSS_EVERY=0
export SLOT_ENABLED=0
export USE_INT6=0

echo "============================================"
echo "  SP1024 control (200 steps)"
echo "============================================"
cd "$WORKDIR" && mkdir -p sp1024 && cd sp1024
export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export TRAIN_SEQ_LEN=1024
export RUN_ID="sp1024_control_$$"
python3 "$SCRIPT" 2>&1 | tee screen.log
echo ""
grep "step:200" screen.log

echo ""
echo "============================================"
echo "  SP4096 (200 steps, same seq_len=1024)"
echo "============================================"
cd "$WORKDIR" && mkdir -p sp4096 && cd sp4096
export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_4096_bpe.model"
export VOCAB_SIZE=4096
export TRAIN_SEQ_LEN=1024
export RUN_ID="sp4096_screen_$$"
python3 "$SCRIPT" 2>&1 | tee screen.log
echo ""
grep "step:200" screen.log

echo ""
echo "============================================"
echo "  SP4096 + higher WD (200 steps)"
echo "============================================"
cd "$WORKDIR" && mkdir -p sp4096_wd && cd sp4096_wd
export MUON_WD=0.085
export EMBED_WD=0.085
export QK_GAIN_INIT=5.0
export RUN_ID="sp4096_wd085_$$"
python3 "$SCRIPT" 2>&1 | tee screen.log
echo ""
grep "step:200" screen.log

echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"
echo "SP1024 control:"
grep "final_int8" "$WORKDIR/sp1024/screen.log" | tail -1
echo "SP4096:"
grep "final_int8" "$WORKDIR/sp4096/screen.log" | tail -1
echo "SP4096 + WD=0.085 + QK_GAIN=5.0:"
grep "final_int8" "$WORKDIR/sp4096_wd/screen.log" | tail -1

#!/bin/bash
# FiLM 5→7+8xMLP submission for 8×H100
# Features: FA3, MuonEq-R, EMA, late QAT, int6+GPTQ, causal SLOT
#
# Session 5 optimized configuration:
# - SP4096 tokenizer (biggest single improvement: -0.22 at 200 steps)
# - QK-Gain 5.0 (-0.039 at 200 steps, free)
# - Causal SLOT with L-BFGS logit-only, 4 steps (converges in 4 steps)
# - Training: 10 min on 8×H100 (~2000-3000 steps depending on step time)
#
# 1×H100 results:
# - Pre-quant: 1.2813 BPB (SP4096+QK-Gain, 600s)
# - Int6+SLOT: 1.2658 BPB (SP1024+SLOT, 600s)
# - Extrapolated 8×H100: ~1.00-1.05 BPB
#
# Usage:
#   bash experiments/film_slot/RUN_8xH100.sh       # Default SP4096
#   VOCAB=1024 bash experiments/film_slot/RUN_8xH100.sh  # SP1024 fallback

VOCAB="${VOCAB:-4096}"

if [ "$VOCAB" = "4096" ]; then
    DATA_PATH="./data/datasets/fineweb10B_sp4096"
    TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
else
    DATA_PATH="./data/datasets/fineweb10B_sp1024"
    TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
fi

DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE="$VOCAB" \
NUM_SHARED_BLOCKS=5 \
NUM_LAYERS=7 \
MLP_MULT=8 \
USE_INT6=1 \
QK_GAIN_INIT=5.0 \
TRAIN_SEQ_LEN=1024 \
SLOT_ENABLED=1 \
SLOT_MODE=lbfgs_logit \
SLOT_STEPS=4 \
SLOT_LR=0.012 \
SLOT_LR_MIN=0.001 \
SLOT_LBFGS_HISTORY=20 \
CAUSAL_SLOT=1 \
EVAL_STRIDE=96 \
SLOT_BATCH_SEQS=32 \
  torchrun --standalone --nproc_per_node=8 experiments/film_slot/train_gpt.py

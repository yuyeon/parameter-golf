#!/bin/bash
# FiLM 5→7+8xMLP submission for 8×H100
# Features: FA3, MuonEq-R, EMA, late QAT, int6+GPTQ, SLOT-24
#
# 1×H100 results: 1.2863 pre-quant BPB, 1.3010 int6 post-quant
# Beats #1 submission by 0.095 BPP on 1×H100

NUM_SHARED_BLOCKS=5 \
NUM_LAYERS=7 \
MLP_MULT=8 \
USE_INT6=1 \
TRAIN_SEQ_LEN=2048 \
SLOT_ENABLED=1 \
SLOT_STEPS=24 \
SLOT_LR=0.012 \
SLOT_LR_MIN=0.001 \
EVAL_STRIDE=96 \
  torchrun --standalone --nproc_per_node=8 experiments/film_slot/train_gpt.py

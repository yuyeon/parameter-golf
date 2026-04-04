# Experiment Plan — Session 4 (2026-04-04)

## Strategic Context

**Merged SOTA**: 1.1147 BPB (PR #1019)
**Non-SLOT frontier**: 1.0897 BPB (PR #1334)
**Best with pre-quant TTT**: 1.0807 BPB (PR #1351)
**Causal SLOT frontier**: 1.0046 BPB (PR #1350)

**Our FiLM best (1×H100)**: 1.2863 pre-quant, 1.3010 int6

The gap to the non-SLOT frontier is ~0.05-0.09 BPP. To close it, we need to adopt
the key techniques the competition converged on AND leverage FiLM's unique advantages.

## Experiment Queue (priority order)

### EXP-1: SLOT Compatibility Test [RUNNING]
**Status**: Running (started 21:38 UTC)
**Goal**: Determine if SLOT (standard + causal) meaningfully improves FiLM on 1×H100
**Config**: FiLM 5→7+8xMLP, SP1024, 600s train, SLOT-24 stride=96
**Expected**: Standard SLOT -0.05 to -0.10 BPP, Causal SLOT -0.02 to -0.05 BPP
**Decision**: If SLOT works on FiLM, it validates FiLM as a competitive SLOT base.
If not, we need to understand why (shared block structure may not produce
diverse enough hidden states for SLOT to exploit).

### EXP-2: SP4096 on FiLM [NEXT]
**Goal**: Biggest expected single improvement — adopt 4096 vocab
**Config**: FiLM 5→7, MLP_MULT=8, SP4096, VOCAB_SIZE=4096
**Changes needed**:
- Set DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE=4096
- May need to adjust MLP_MULT (embedding table grows 4x from 524K to 2.1M params)
- Consider higher WD (0.085-0.10) for quantization friendliness
- Remove BigramHash if present (redundant with 4096 vocab)
**Expected gain**: -0.02 to -0.05 BPP (based on SP4096 impact across other submissions)
**Screening**: 200-step comparison against SP1024 FiLM baseline
**Risk**: Embedding table growth may push artifact over 16MB; need int6 GPTQ to fit

### EXP-3: FiLM + Depth Recurrence (extended block schedule)
**Goal**: Get more virtual depth from our 5 shared blocks
**Idea**: Extend block_schedule with repeated middle entries:
  Current: [0, 1, 2, 3, 4, 0, 1] (7 virtual layers from 5 blocks)
  Proposed: [0, 1, 2, 3, 4, 2, 3, 0, 1] (9 virtual layers) or
           [0, 1, 2, 3, 4, 3, 4, 0, 1, 2] (10 virtual layers)
  FiLM modulation params scale linearly (cheap)
**Screening**: 200-step comparison at multiple depth configurations
**Risk**: More virtual layers = slower steps. Need the depth vs speed tradeoff to be positive.
**Novelty**: FiLM+recurrence is genuinely new — nobody has combined modulation-based
sharing with layer repetition.

### EXP-4: QK-Gain Tuning
**Goal**: Free BPP from simple scalar change
**Config**: QK_GAIN_INIT from 1.5 → 4.0 or 5.0 (what top submissions use)
**Expected gain**: -0.003 BPP (small but free)
**Screening**: 200-step comparison

### EXP-5: Higher Weight Decay for Quantization
**Goal**: Make weights more quantization-friendly
**Config**: WD from 0.04 → 0.085-0.10
**Expected gain**: Indirect — better int6 compression → room for larger model OR lower post-quant loss
**Screening**: 200-step comparison of WD sweep

### EXP-6: Parallel Residuals (from layer 5+)
**Goal**: Separate attn and MLP into parallel residual streams for later layers
**Implementation**: After layer 5, attn and MLP each add independently to residual
  instead of sequentially. x' = x + attn(x) + mlp(x) instead of x' = mlp(x + attn(x))
**Expected gain**: ~-0.005 BPP (based on PR #1334)
**Screening**: 200-step comparison
**Risk**: May not compose with FiLM modulation (resid_mixes need to handle parallel case)

### EXP-7: Pre-Quant Discriminative TTT
**Goal**: Add test-time adaptation that's compatible with GPTQ
**Implementation**: AdamW fine-tuning on val data BEFORE GPTQ quantization.
  Per-block LR: early blocks 0.3x, late blocks 1.0x (PR #1351 approach)
**Expected gain**: -0.005 to -0.009 BPP
**Prerequisite**: Works only on 8×H100 where we have enough training steps for EMA convergence
**Risk**: We're on 1×H100 — may need to defer to 8×H100 run

### EXP-8: Full Stack Integration
**Goal**: Combine all winning modifications
**Config**: FiLM + SP4096 + extended depth + QK-Gain 5.0 + WD=0.09 + Causal SLOT
**Prerequisites**: EXP-2 through EXP-6 completed and positive
**Format**: 600s full run with int6 + SLOT eval

## 200-Step Screening Protocol
- All comparisons use seed=42 on 1×H100
- Budget: 200 steps for quick screening, 600s for confirmation
- Metric: post-int8-quant BPB (pre-quant for speed when quant doesn't affect relative order)
- Decision rule: >0.005 BPP improvement → PROMOTE, 0-0.005 → CONFIRM at 600s, negative → KILL

## Key Uncertainties
1. FiLM's 8×H100 scaling is completely untested
2. SP4096's interaction with FiLM's parameter budget (embedding table grows 4×)
3. Whether FiLM hidden states are diverse enough for SLOT to exploit
4. Whether depth recurrence + FiLM modulation is strictly better than either alone
5. Causal SLOT legality (no ruling yet)

## Time Budget
- EXP-1 running now (~75 min)
- EXP-2 through EXP-5 can be done as 200-step screens (~6 min each)
- EXP-6 requires implementation work
- EXP-7 requires 8×H100
- EXP-8 depends on all prior results

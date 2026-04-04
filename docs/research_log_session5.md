# Research Log — Session 5 (2026-04-04, continued)

## Environment
- GPU: 1× NVIDIA H100 80GB HBM3
- PyTorch 2.11.0+cu128, FA3 3.0.0, Triton OK, CUDA 12.8
- Baseline verified: 353ms/step, 2.597 BPB @ 50 steps

## State at Start of Session

### From Session 4 (same day)
EXP-1 (SLOT compatibility test on FiLM) completed:
- **FiLM baseline (int6)**: 1.3003 BPB
- **Standard SLOT (int6+SLOT24)**: 0.9028 BPB (-0.3975) — works but **ILLEGAL**
- **Causal SLOT v1 (int6+SLOT24)**: 1.3095 BPB (+0.009) — **HURTS performance**
- Multiple runs crashed with torch.compile dtype mismatch (fixed iteratively)

### Root Cause Analysis: Why Causal SLOT v1 Failed

The broadcast delta `[bsz, 1, hdim]` is the core problem:
1. **Standard SLOT**: opt_mask == score_mask (same positions). Delta optimized directly for scored positions → massive improvement.
2. **Causal SLOT**: opt_mask (context positions) and score_mask (new positions) are completely disjoint. A broadcast delta optimized for context can actively hurt new positions.
3. The +0.009 BPB result means the delta HURTS more than it helps on new positions.

### Competition Intelligence (from web research)

PR #1350 achieves 1.0046 BPB with causal SLOT (-0.087 BPP). Key implementation details:
- **L-BFGS optimizer** (max_iter=25, history=20) — much faster convergence than AdamW
- **Logit space** — optimize logit biases, not hidden deltas
- **Focal loss on last 128 context tokens** — nearby context more predictive of new positions
- **Warm-start between windows** — carry bias across consecutive windows
- **Delta clamped to +/-5** — prevent overfitting
- Eval time: ~556s

Per-Sample SLOT (PR #1329) reaches 0.636 BPB but is standard SLOT (illegal).

## Experiments Run

### Quick Smoke Test: L-BFGS logit-only (4 steps)
**Config**: lbfgs_logit mode, 4 steps, no focal/warmstart/clamp (old code)
**Result**: 1.2658 BPB (-0.035 from 1.3003 baseline)
**Significance**: Confirms L-BFGS + logit-only approach works for causal SLOT.
Even with just 4 steps and no focal/warmstart, already -0.035 vs +0.009 for v1.

### L-BFGS logit-only (24 steps) [RUNNING]
**Config**: lbfgs_logit mode, 24 steps, no focal/warmstart/clamp (old code)
**Expected**: ~1.20-1.25 BPB
**Status**: Running (~45 min estimated)

## Implementation Changes

### SLOT Mode System
Added `SLOT_MODE` env var with four modes:
- `v1`: Original AdamW delta+bias (default for backward compat)
- `logit_only`: AdamW logit bias only (no hidden delta)
- `lbfgs`: L-BFGS delta+bias
- `lbfgs_logit`: L-BFGS logit bias only (recommended for causal)

### Causal SLOT v2 Features (matching PR #1350)
- `SLOT_FOCAL_CTX=128`: Focal loss on last 128 context tokens
- `SLOT_WARMSTART=1`: Carry mean logit bias between batches
- `SLOT_CLAMP=5.0`: Clamp logit bias to [-5, 5]
- `SLOT_LBFGS_HISTORY=20`: L-BFGS curvature history

## Untested Novel Ideas

### FiLM-Modulation SLOT (genuinely new)
Instead of optimizing logit biases, optimize FiLM modulation params (attn_scales, mlp_scales, resid_mixes) at test time.
- 14,336 parameters (compact, semantically meaningful)
- Changes HOW the model processes data, not WHAT it outputs
- Requires re-running model forward pass per SLOT step (expensive)
- Could be implemented as "delta to FiLM scales" with clamping
- **Shelved for now** — focus on getting logit-bias approach working first

## Next Steps (prioritized)
1. [RUNNING] Get 24-step L-BFGS baseline result
2. [QUEUED] Run v2 variants (focal+warmstart+clamp) 
3. [QUEUED] SP4096 200-step screen
4. [QUEUED] QK-Gain 5.0 + WD 0.085 screen
5. [IDEA] Extended depth recurrence (7 → 9-10 virtual layers)
6. [IDEA] FiLM-modulation SLOT

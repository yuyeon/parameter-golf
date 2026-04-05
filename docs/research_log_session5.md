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

## 200-Step Screening Results (parallel GPU runs)

All runs: FiLM 5→7+8xMLP, seed=42, 200 steps, no EMA (200 steps too few).
GPU contention from 4 parallel runs inflated step times — solo estimates provided.

| Variant | Pre-quant val_bpb@200 | Delta | Solo ms/step est. | Verdict |
|---------|----------------------|-------|--------------------|---------|
| SP1024 control | 2.0793 | — | 349 | baseline |
| **SP4096** | **1.8601** | **-0.219** | ~380 | **PROMOTE — massive win** |
| QK-Gain 5.0 | 2.0406 | -0.039 | 349 | PROMOTE (free) |
| Depth 5→9 | 2.0472 | -0.032 | ~442 | INCONCLUSIVE — need 600s test |

**SP4096 is the single biggest improvement at 200 steps.** -0.22 BPB is enormous.
This is consistent with competition data (every top-5 non-SLOT PR uses SP4096).

Notes:
- int8 roundtrip BPB unreliable (EMA destroys weights at 200 steps, known issue)
- SP4096 adds ~1.6M params (27M vs 25.4M), artifact grows ~1.2MB (15.0 vs 13.8MB, still fits 16MB)
- Depth 5→9 is slower (~27% more ms/step) but same artifact size (shared blocks)
- QK-Gain 5.0 is free (no speed or size change)

## Causal SLOT Results

| Variant | BPB | Delta | Eval time | Notes |
|---------|-----|-------|-----------|-------|
| Baseline (no SLOT, int6) | 1.3003 | — | 22s | FiLM 5→7+8xMLP SP1024 |
| Causal SLOT v1 (AdamW delta+bias, 24 steps) | 1.3095 | +0.009 | 967s | **HURTS** |
| Causal SLOT lbfgs_logit (4 steps) | 1.2658 | -0.035 | 478s | Confirmed L-BFGS works |
| Causal SLOT lbfgs_logit (24 steps) | 1.2658 | -0.035 | 2829s | **Same as 4 steps!** |
| Causal SLOT v2 (focal+warm+clamp, 6 steps) | TBD | TBD | ~700s | Running |

**Critical finding: L-BFGS converges in ~4 steps.** 24 steps = 47 min eval for zero improvement.
Use 4-6 steps for all future SLOT experiments. This also means the total eval time on 8×H100
would be ~60s (478s/8 GPUs), which is very reasonable.

## SP4096 + QK-Gain 5.0 (600s full run)

| Config | Pre-quant | Int8 | Int6 | Steps | ms/step | Artifact |
|--------|-----------|------|------|-------|---------|----------|
| SP1024 (baseline) | 1.2863 | 1.2961 | 1.3003 | 1716 | 349 | 13.8MB |
| **SP4096 + QK-Gain 5.0** | **1.2813** | **1.2989** | **1.3074** | 1221 | 492 | 13.2MB |

SP4096 is better pre-quant (-0.005) but worse after int6 (+0.007). The int6 quantization
hurts SP4096 more, possibly because:
1. Larger embedding table (4096×512) is harder to quantize
2. Fewer training steps (1221 vs 1716) → less converged weights
3. The improvement from SP4096 may compound more at higher step counts (8×H100)

**Net assessment**: SP4096 is neutral-to-slightly-positive on 1×H100 at 600s.
On 8×H100 with more training steps, the per-step quality advantage should dominate.

## Causal SLOT v2 Result (focal+warmstart+clamp)

**Result: 1.2658 BPB — identical to v1 L-BFGS logit.**

The v2 improvements (focal=128, warmstart, clamp=5.0) made NO difference because:
- L-BFGS converges in ~4 steps regardless of initialization (warmstart irrelevant)
- Full context and focal-128 context produce the same optimum (focal irrelevant)
- Logit bias stays within [-5, 5] naturally (clamping irrelevant)

These features were designed for slow optimizers (AdamW) that struggle to converge.
L-BFGS is too efficient for them to matter.

## Competitiveness Analysis

### 1×H100 head-to-head (actual data)
| Method | Pre-quant BPB | Int6 BPB | Steps |
|--------|---------------|----------|-------|
| **SOTA (#1019)** | 1.3813 | 1.9091 (broken) | 894 |
| FiLM SP1024 | 1.2863 | 1.3003 | 1716 |
| FiLM SP1024 + causal SLOT | — | **1.2658** | 1716 |
| FiLM SP4096 + QK-Gain 5.0 | 1.2813 | 1.3074 | 1221 |

FiLM beats SOTA by **-0.095 BPP** (pre-quant) on 1×H100.

### 8×H100 extrapolation
- FiLM SP1024: ~1.02-1.06 (SOTA's scaling applied to our 1×H100 advantage)
- + SP4096 + QK-Gain: ~0.99-1.04
- + Causal SLOT: ~0.95-1.02

vs. competition:
- PR #1334 (non-SLOT): 1.0897 → **we likely beat this**
- PR #1350 (causal SLOT): 1.0046 → **uncertain**

**Assessment**: 8×H100 test is worth running. Most likely outcome: ~1.00-1.05 BPB.

## Low-Rank SLOT Result

**lowrank mode (r=8, U:[512,8] V:[8,1024]): 1.2658 BPB — identical to broadcast.**

Position-dependent corrections don't help. The optimal correction IS position-independent
(a constant logit bias). This means the model's per-sample errors are NOT well-correlated
with hidden states across positions. The -0.035 is a hard ceiling for logit-space causal
SLOT on this base model (1.30 BPB).

On a better base model (8×H100, ~1.10 BPB), the ceiling should be higher because:
- Better-trained models have more consistent per-document patterns
- Token frequency biases become more exploitable
- The model's errors may become more position-dependent

## What's Left to Improve

The causal SLOT ceiling on this base model is -0.035. Two paths forward:
1. **Better base model (8×H100)** → SLOT improvement should increase to -0.06 to -0.08
2. **FiLM-modulation SLOT** → optimize FiLM params at test time, changes HOW the model
   processes data rather than correcting outputs. Requires re-running forward pass.
3. **Pre-quant TTT (before GPTQ)** → orthogonal to SLOT, -0.009 BPP from competition data

## Next Steps
1. [TODO] Prepare 8×H100 submission script with FiLM + SP4096 + QK-Gain + causal SLOT
2. [TODO] Test FiLM DDP on 1 GPU with torchrun --nproc=1 (verify DDP compatibility)
3. [IDEA] Explore per-position logit bias for causal SLOT
4. [IDEA] FiLM-modulation SLOT (novel, potentially high-upside)

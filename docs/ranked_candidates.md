# Ranked Candidates

*Last updated: 2026-04-03 — first round of experiments complete*

## Summary of Results (200-step screening on H100)

| Rank | Config | Post-quant BPB (avg) | ms/step | Delta vs Baseline | Status |
|------|--------|---------------------|---------|-------------------|--------|
| 1 | **MuonEq-R** | **1.6332** | 333 | **-0.024** | **PROMOTE** |
| 2 | MuonEq-R + XSA9 | 1.6357 | 351 | -0.021 | PROMOTE |
| 3 | MuonEq-R + 2 Conv layers | 1.6360 | 319 | -0.018 | SALVAGE |
| 4 | MuonEq-R + XSA4 | 1.6368 | 341 | -0.017 | HOLD |
| 5 | MuonEq-R + 2 Conv + XSA | 1.6407 | 332 | -0.016 | HOLD |
| 6 | MuonEq-R + MTP | 1.6426 | 340 | -0.014 | KILL |
| 7 | MTP k=3 alone | 1.6512 | 339 | -0.003 | KILL |
| — | Baseline (control) | 1.6569 | 330 | — | — |
| — | Asymmetric 3/6 split | 1.6540 | 329 | -0.000 | KILL |

## Promoted Candidates

### 1. MuonEq-R (Row-Normalized Gradient Before Newton-Schulz)
- **Thesis**: Row-normalizing the gradient matrix before Newton-Schulz orthogonalization makes Muon invariant to row-wise weight scaling, producing more balanced updates.
- **Novelty claim**: MuonEq-R was introduced in PR #1279 but not isolated or tested on the baseline. We show it's the single most impactful change.
- **Best result**: 1.6287 post-quant BPB (seed 1337), 1.6376 (seed 42). Average improvement: **-0.024 BPB**.
- **Implementation**: 3 lines added to Muon.step() — row-normalize grad before zeropower_via_newtonschulz5.
- **Step time overhead**: ~0% (333 vs 330 ms/step)
- **Risks**: May interact differently with parameter banking on 8xH100. Needs testing on strong stacks.
- **Before full run**: Test on SOTA stack with banking, confirm composes with GPTQ, verify at 7000+ steps.

### 2. MuonEq-R + XSA (All Layers)
- **Thesis**: XSA subtracts self-value projection from attention output, encouraging cross-position mixing. Composes additively with MuonEq-R.
- **Best result**: 1.6357 post-quant BPB (seed 42).
- **Step time overhead**: +5% (351 vs 333 ms/step)
- **Before full run**: Confirm on strong stack, verify XSA-all is better than XSA-4 at longer runs.

## Salvage Candidates

### 3. Hybrid Conv-Attention (2 Conv + 7 Attention)
- **Thesis**: Early layers don't need full attention. Causal depthwise convolution is 4% faster per step.
- **Result**: 1.6360 post-quant BPB at 319 ms/step — faster than baseline but doesn't compose well with XSA (adding XSA loses the speed advantage).
- **Salvage path**: Try with more attention layers (1 conv + 8 attn) or wider convolution kernels. The step time savings could fund ~280 extra steps at full budget.

## Killed Ideas

### Asymmetric U-Net Split
- **Result**: No improvement at 9 layers. 1/8 split is actively harmful (+0.028 BPB, 2x slower).
- **Why**: At 9 layers, the U-Net structure may be too shallow for asymmetry to matter. PR #1275's results may only apply at higher depth.

### Multi-Token Prediction (MTP)
- **Result alone**: -0.003 BPB (marginal, consistent)
- **Result with MuonEq-R**: Hurts (+0.009 BPB vs MuonEq-R alone)
- **Why killed**: Doesn't compose with better optimization. The auxiliary loss interferes with MuonEq-R's improved gradient dynamics.

## Full Research Agenda (15 ideas)

*See original agenda below. Updated status reflects experiment results.*

### Tier 1: Tested
1. ~~Asymmetric U-Net Split~~ → **KILL** (no effect at 9L)
2. ~~Depth Recurrence~~ → **HOLD** (slower step time, but small artifact; needs better base)
3. ~~MuonEq-R~~ → **PROMOTE** (dominant effect)
4. ~~Multi-Token Prediction~~ → **KILL** (doesn't compose)
5. ~~Conv-Attention Hybrid~~ → **SALVAGE** (fast but doesn't compose with XSA)
6. ~~XSA (added to baseline)~~ → **PROMOTE** (known technique, confirmed composable)

### Tier 2: Next to test
7. Per-Layer Quantization Bitwidth Allocation
8. Causal Word-Boundary Features (WARP-Pos/Type)
9. Learned Position-Dependent Gating (Dynamic Depth)
10. JEPA Auxiliary Objective

### Tier 3: Exploratory
11. Single SSM Layer Hybrid
12. Sliding Window Train / Full Context Eval
13. Mixture of Low-Rank Experts
14. Compressibility-Aware Training
15. Ternary + Scale Recovery at 3x Model Size

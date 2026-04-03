# Ranked Candidates

*Last updated: 2026-04-03 — round 2 complete*

## Master Results Table (200-step screening, H100, post-quant BPB)

| Rank | Config | Seed 42 | Seed 1337 | Avg | ms/step | Delta vs Baseline |
|------|--------|---------|-----------|-----|---------|-------------------|
| 1 | **Kitchen Sink 3x** (MuonEq-R+XSA9+LeakyReLU²+SmearGate+3xMLP) | **1.6159** | *pending* | — | 385 | **-0.038** |
| 2 | MuonEq-R + LeakyReLU² | 1.6245 | — | — | 333 | -0.030 |
| 3 | MuonEq-R + XSA9 | 1.6357 | — | — | 351 | -0.018 |
| 4 | **MuonEq-R** | **1.6376** | **1.6287** | **1.6332** | 333 | **-0.024** |
| 5 | Dynamic Depth + MuonEq-R | 1.6325 | — | — | 353 | -0.022 |
| 6 | MuonEq-R + 2 Conv layers | 1.6360 | — | — | 319 | -0.018 |
| 7 | MuonEq-R + XSA4 | 1.6368 | — | — | 341 | -0.017 |
| 8 | MuonEq-R + 2 Conv + XSA | 1.6407 | 1.6404 | 1.6406 | 332 | -0.014 |
| 9 | MuonEq-R + MTP | 1.6409 | 1.6442 | 1.6426 | 340 | -0.014 |
| 10 | MTP k=3 alone | 1.6512 | — | — | 339 | -0.003 |
| — | Baseline | 1.6541 | 1.6596 | 1.6569 | 330 | — |
| — | MuonEq-R + SmearGate (alone) | 1.6566 | — | — | 336 | +0.000 |
| — | Asymmetric 3/6 | 1.6540 | — | — | 329 | +0.000 |

### On SOTA stack (pre-quant only, GPTQ fails at 200 steps)
| Config | Pre-quant 200-step | ms/step | Delta |
|--------|-------------------|---------|-------|
| SOTA baseline | 1.9295 | 669 | — |
| SOTA + MuonEq-R | **1.7401** | 669 | **-0.189** |

### Longer run validation (500 steps, seed 42)
| Config | 500-step post-quant | Delta |
|--------|-------------------|-------|
| Baseline | 1.4834 | — |
| MuonEq-R | **1.4687** | **-0.015** |

## Promoted Candidates (Rank-ordered)

### 1. Kitchen Sink: MuonEq-R + XSA-all + LeakyReLU² + SmearGate + 3x MLP
- **Thesis**: Combining the best orthogonal improvements from the community into a clean baseline stack
- **Novelty claim**: This specific combination has not been tested. MuonEq-R is the novel ingredient.
- **Best observed**: 1.6159 post-quant BPB (-0.038 vs baseline)
- **Parameters**: 21.8M (fits in 16MB with int6+GPTQ)
- **Step time**: 385 ms/step (~17% slower than baseline, but much faster than SOTA's 669ms)
- **Risk**: SmearGate's contribution unclear (helps here, hurt in isolation). Needs multi-seed verification.
- **Before full run**: Verify seed 1337, add BigramHash, try 11 layers

### 2. MuonEq-R (standalone)
- **Thesis**: Row-normalizing Muon gradients is universally beneficial
- **Best observed**: 1.6287 post-quant (seed 1337)
- **Avg improvement**: -0.024 BPB across 2 seeds, -0.015 at 500 steps
- **On SOTA stack**: -0.189 pre-quant (massive effect on banked model)
- **Risk**: None identified. Pure win.
- **Before full run**: Already validated. Apply to SOTA stack and run with GPTQ.

### 3. LeakyReLU-squared (with MuonEq-R)
- **Thesis**: Leaky negative slope preserves gradient flow through squaring, improving optimization
- **Best observed**: 1.6245 post-quant (-0.013 vs MuonEq-R alone)
- **Risk**: None. Zero overhead.
- **Before full run**: Already in SOTA stack. Just confirm composability.

### 4. Dynamic Depth Gating
- **Thesis**: Per-token gates let easy tokens skip layers, improving capacity allocation
- **Best observed**: 1.6325 post-quant (-0.005 vs MuonEq-R alone)
- **Risk**: 20ms/step overhead may not justify the gain. Gate adds parameters.
- **Before full run**: Test with lighter gate (no hidden layer), test on strong stack.

## Killed Ideas
- **Asymmetric U-Net Split**: No effect at 9L
- **Multi-Token Prediction**: Doesn't compose with MuonEq-R
- **SmearGate alone**: Hurts on baseline without other components

## Key Learnings

1. **MuonEq-R is the dominant improvement.** Every experiment should use it as the new baseline.
2. **Orthogonal improvements compose**: XSA + LeakyReLU² + 3x MLP all add on top of MuonEq-R.
3. **Some techniques only work in combination**: SmearGate hurts alone but helps in the kitchen sink.
4. **MTP doesn't compose**: Auxiliary losses interfere with MuonEq-R's gradient dynamics.
5. **Step time is critical**: Dynamic depth's 20ms overhead is significant. Conv mixer's 14ms savings is significant. Every ms = ~2 training steps over 10 minutes.

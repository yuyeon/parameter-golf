# Parameter Golf Landscape Map

*Last updated: 2026-04-03*

## Current SOTA

**1.1147 BPB** (abaybektursun, PR #1019) — AR Self-Gen GPTQ + XSA-all + BigramHash 3072x112.

**Unmerged frontier**: PR #1279 claims **1.0924 BPB** with MuonEq-R + depth recurrence + N61 mixed GPTQ (no TTT, no SLOT — fully clean). If verified, this is the real target to beat.

## What's Crowded (diminishing returns)

| Technique | Status | Why crowded |
|-----------|--------|-------------|
| Int6 quantization + GPTQ variants | Saturated | Every top-10 submission uses it. AR self-gen GPTQ is near-optimal. |
| Sliding window eval (stride=64) | Universal | Free ~0.034 BPB, already in every serious submission. |
| EMA + SWA weight averaging | Universal | Marginal gains from tuning decay/frequency. |
| SmearGate + BigramHash embedding | Universal | Bucket size (2048-3072) is the only remaining knob. |
| Muon optimizer | Universal | MuonEq-R (PR #1279) is the newest variant but incremental. |
| 11L/512d/8H/4KV architecture | Converged | The community has converged on this shape. |
| LeakyReLU(0.5)-squared MLP 3x | Converged | Marginal over ReLU-squared. |
| XSA (all layers) | Near-saturated | Applying to all 11 layers is already done in SOTA. |
| Warmdown tuning (3000-4000 iters) | Saturated | Little room left. |
| LR / WD / momentum sweeps | Saturated | Heavily searched across 25+ submissions. |

## What's Underexplored (high potential)

### Architecture
1. **Depth recurrence / weight sharing** — PR #1279 shows +0.022 BPB from repeating layers 4-5 with untied MLPs. PR #363 said it doesn't help, but that was before MuonEq-R and noisy QAT. The key insight: recurrence saves *parameters* (more room for bigger models under 16MB) but costs *step time* (fewer steps in 10 min). The tradeoff is now more favorable with faster step times.

2. **Asymmetric encoder-decoder split** — PR #1275 found monotonic improvement from shifting layers to decoder (1/10 split instead of 5/5). One-line change, never tested before because everyone copied the image U-Net convention. Incomplete H100 data.

3. **Parallel residual streams** — PR #1274 uses parallel residuals from layer 7+. Compositional with depth recurrence. Unexplored in isolation.

4. **State-space model hybrids** — PR #1268 (Mamba3) showed SSMs win early but lose due to 108ms/step vs 86ms for attention. The right idea might be: 1-2 SSM layers for cheap long-range mixing + attention for the rest, not a full SSM model.

5. **Hierarchical Shared Attention (HSA)** — PR #1264 showed modest gains from multi-level head sharing. Needs testing on strong stacks.

### Training
6. **MuonEq-R optimizer** — Row-normalizes gradients before Newton-Schulz. Used in the 1.0924 submission. Not yet tested in isolation or on other stacks.

7. **Auxiliary predictive objectives** — JEPA (PR #1277, in progress), auxiliary next-token from intermediate layers. Could help tiny-model representations without increasing model size.

8. **Coprime-stride data loading** — PR #1272 mentions this as "actually matters." Non-sequential shard access pattern that improves data diversity per step.

### Evaluation / Test-Time
9. **SLOT (Sparse Latent Optimization at Test-time)** — PR #1263 claims 0.9354 BPB, a massive 0.18 BPB improvement. Currently contested for causal violations (PR #1272). If a *causal-legal* variant exists, this is by far the highest-upside direction.

10. **Legal TTT improvements** — Score-first TTT gave -0.0025 BPB (PR #549). Most TTT variants have been negative on strong stacks. But the design space is large (what to adapt, learning rate, number of steps).

### Compression / Quantization
11. **Mixed INT5/INT6 quantization** — PR #1279 uses 61 int6 layers with some int5. Finer-grained allocation could save bytes for larger models.

12. **1-bit / ternary with scale recovery** — PR #640 got 1.1570 with ternary 73.7M params. The key question: can ternary + much larger model beat int6 + smaller model?

### Tokenizer
13. **Alternative tokenizers** — Scylla (998 tokens) in PR #1274 had buggy byte accounting, but the idea of optimized tokenizer design remains open. BPE 4096/8192 are used in some submissions but not systematically compared.

## What's Overrated

1. **N-gram caches / online logit bias** — PR #1272 proves these don't help on strong models. "Scale deception": helps weak models, neutral/negative on strong ones.

2. **KNN hidden state retrieval** — PR #1259 confirms scale deception: -2 to -4% on weak models but +1.5% (hurts) on strong models.

3. **TTT as currently implemented** — The SOTA dropped TTT because Full Hessian GPTQ was worth more (-0.0047 vs -0.0025). TTT may still have upside but the current implementations are too slow or too weak.

4. **Full SSM models** — Step time kills them in a time-budgeted competition. Only viable as 1-2 hybrid layers.

5. **EMA at short step counts** — PR #1252 found EMA hurts when step count < 3000. Many submissions blindly include EMA without checking this crossover.

## What's Underrated

1. **Depth recurrence** — Dismissed by PR #363 but resurrected by PR #1279 (+0.022 BPB). The trick is noisy QAT + untied MLPs per repeat.

2. **Asymmetric U-Net split** — PR #1275: one-line change, monotonic improvement, nobody ever tested it because the 50/50 split was cargo-culted from vision.

3. **Parameter budget reallocation** — Everyone uses 11L/512d. But with better compression (GPTQ + ternary + LZMA), you could fit a *much* larger model in 16MB. The 73.7M ternary model (PR #640) shows this path exists.

4. **MuonEq-R** — A genuine optimizer improvement that the 1.0924 submission depends on. Not yet widely adopted or tested.

5. **Causal SLOT variants** — If someone can make SLOT causal-legal, the gains are enormous (0.18 BPB). Even a fraction of that would be a massive win.

6. **Word-boundary features (WARP)** — PR #1252 found gains from word-length and word-position features, but had to close due to a causality bug in WARP-Len. WARP-Pos and WARP-Type remain valid and untested on strong stacks.

## Key Strategic Insights

1. **The bottleneck is step time, not model quality.** In a 10-minute budget, anything that slows step time by 10% costs ~700 training steps. You need to *save* steps (via parameter sharing, smaller models) or *buy* steps (via faster kernels).

2. **Compression is the second bottleneck.** 16MB allows ~21M params at int6 or ~73M at ternary. Better compression = bigger model = better quality, if the compression doesn't hurt too much.

3. **The strongest improvements in the last 2 weeks came from quantization (GPTQ variants), not architecture.** This suggests architecture is undertested — the community optimized the easy knobs first.

4. **Negative results are extremely informative.** PR #1272's systematic failures on strong models (n-grams, KNN, adapters, complementary training) saves us from several dead ends.

5. **The gap between merged SOTA (1.1147) and unmerged frontier (1.0924) is 0.022 BPB.** This gap comes from MuonEq-R + depth recurrence + mixed quantization. These are the techniques to build on.

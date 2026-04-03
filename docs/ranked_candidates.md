# Ranked Candidates

*Last updated: 2026-04-03 — round 4 complete (25+ experiments)*

## Best Known Configuration

**Kitchen Sink + BigramHash + seq2048 + Muon WD**
- MuonEq-R + XSA-all + LeakyReLU² + SmearGate + 3x MLP + BigramHash 2048 + Muon WD 0.04 + seq_len 2048
- **Script**: `experiments/kitchen_sink_bigram/train_gpt.py`
- **200-step**: 1.5971 post-quant BPB (int8+zlib), 448ms/step, 22.1M params
- **600s single-H100** (seq1024): 1.3656 BPB, 13.66MB artifact
- **600s single-H100** (seq2048): running now — expected ~1.34 BPB

## Complete Results Table (200-step screening, sorted)

| Rank | Config | BPB (post-quant) | ms/step | Delta vs Baseline |
|------|--------|-----------------|---------|-------------------|
| 1 | **KS + BigramHash + seq2048** | **1.5971** | 448 | **-0.057** |
| 2 | KS + BigramHash + WD | 1.6077 | 391 | -0.046 |
| 3 | KS + BigramHash | 1.6085 | 387 | -0.046 |
| 4 | KS 3x (avg 2 seeds) | 1.6151 | 385 | -0.042 |
| 5 | KS + Ortho Init | 1.6187 | 390 | -0.035 |
| 6 | MuonEq-R + LeakyReLU² | 1.6245 | 333 | -0.030 |
| 7 | Dynamic Depth + MuonEq-R | 1.6325 | 353 | -0.022 |
| 8 | MuonEq-R (avg 2 seeds) | 1.6332 | 333 | -0.024 |
| 9 | MuonEq-R + XSA9 | 1.6357 | 351 | -0.018 |
| 10 | MuonEq-R + 2 Conv | 1.6360 | 319 | -0.018 |
| — | Baseline (avg 2 seeds) | 1.6569 | 330 | — |

## 600s Full-Budget Results

| Config | Steps | BPB | Artifact | Notes |
|--------|-------|-----|----------|-------|
| KS + BigramHash (seq1024) | 709 | **1.3656** | 13.66 MB | Working, int8+zlib |
| KS + BigramHash (seq2048) | ~1200? | *running* | ~13.7 MB | Best config |
| SOTA + MuonEq-R v3 | 494 | 1.6686 (pre-q) | — | GPTQ int6 fails on single GPU |

## Key Findings Summary

### What works (composable stack)
1. **MuonEq-R** (-0.024 BPB, 0% overhead) — the foundation
2. **LeakyReLU²** (-0.013 BPB, 0% overhead) — better activation gradient flow
3. **XSA-all** (-0.002 BPB, +5% overhead) — cross-position information mixing
4. **3x MLP** (-0.010 BPB, +15% overhead) — more capacity per layer
5. **BigramHash** (-0.007 BPB, +1% overhead) — token pair features
6. **seq_len 2048** (-0.011 BPB, +15% overhead) — longer training context

### What doesn't work
- Asymmetric U-Net splits (no effect at 9L)
- MTP auxiliary loss (conflicts with MuonEq-R)
- Orthogonal init (conflicts with MuonEq-R)
- SmearGate alone (hurts without other components)
- 11L without banking (3x slower, kills step budget)
- Naive int6 (catastrophic without GPTQ error compensation)

### Unresolved
- GPTQ int6 on single GPU (needs 7000+ steps for good Hessian)
- EMA requires 2000+ steps to converge (skip for short runs)
- Conv-attention hybrid (speed win but doesn't compose with XSA)
- Dynamic depth (promising but overhead needs optimization)

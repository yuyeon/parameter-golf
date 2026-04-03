# Full Run Queue (8xH100 Promotion Candidates)

*Last updated: 2026-04-03 — round 2 results*

## Queue (ordered by priority)

### 1. MuonEq-R on SOTA Stack (Highest Priority)
- **What**: Apply MuonEq-R (3-line change) to the merged SOTA submission
- **Evidence**: -0.189 BPB pre-quant at 200 steps on SOTA stack. -0.024 avg on baseline.
- **Expected on leaderboard**: 1.10-1.11 BPB (if composable with GPTQ)
- **Implementation**: Single edit to Muon.step() in SOTA train_gpt.py
- **Remaining**: Verify GPTQ works, run 3 seeds

### 2. Kitchen Sink Stack (Full Budget)
- **What**: MuonEq-R + XSA9 + LeakyReLU² + SmearGate + 3x MLP on clean 9L baseline
- **Evidence**: 1.6159 post-quant at 200 steps (-0.038 vs baseline)
- **Expected on leaderboard**: ~1.18-1.20 BPB with full training + GPTQ
- **Remaining**: Multi-seed verification, add BigramHash, try 11L

### 3. Kitchen Sink + BigramHash + 11 Layers
- **What**: Extend kitchen sink to 11L + BigramHash 2048 + GPTQ
- **Evidence**: Individual components all validated. Not yet tested together.
- **Expected**: ~1.14-1.16 BPB
- **Remaining**: Implement and screen at 200 steps

## Deferred (need more screening)
- Dynamic Depth Gating: promising but step time overhead needs optimization
- Conv-Attention Hybrid: interesting speed angle but doesn't compose with XSA

## Promotion Criteria
1. Clear positive signal (>0.005 BPB over appropriate control)
2. Robust across 2+ seeds
3. Simple, correct implementation
4. Fits 16MB artifact budget
5. Likely to survive scale-up
6. Distinct from existing leaderboard entries

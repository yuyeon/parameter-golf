# Full Run Queue (8xH100 Promotion Candidates)

*Last updated: 2026-04-03*

## Queue (ordered by priority)

### 1. MuonEq-R on SOTA Stack
- **What**: Apply MuonEq-R (3-line Muon modification) to the current SOTA submission (2026-03-25 ValCalib_GPTQ_XSA_BigramHash3072)
- **Why**: MuonEq-R showed -0.024 BPB on the baseline — the single largest improvement from our experiments. The SOTA stack already uses Muon, so this is a drop-in change.
- **Expected improvement**: 0.01-0.02 BPB on SOTA (effects may be smaller on stronger stacks, but even 0.005 would be significant)
- **Risk**: May interact with parameter banking or GPTQ calibration. Needs testing.
- **Remaining validation**: (a) Test on SOTA script at 200+ steps, (b) Verify GPTQ still works, (c) Run 2+ seeds

### 2. MuonEq-R + XSA-all on Baseline (Full Budget)
- **What**: Run the clean MuonEq-R + XSA9 combination for full 10-minute budget on 8xH100 with GPTQ
- **Why**: XSA composes with MuonEq-R (additive -0.002 BPB). Combined stack on baseline may beat some existing leaderboard entries.
- **Expected improvement**: val_bpb ~1.18-1.20 at full scale (vs baseline 1.2244)
- **Risk**: Baseline architecture lacks many SOTA tricks (BigramHash, SmearGate, banking, 3x MLP). May not be competitive on its own.
- **Remaining validation**: 500-step run confirms trend holds

### 3. Hybrid Conv-Attention Architecture (if conv mixer signal strengthens)
- **What**: 1-2 causal conv layers + remaining attention layers with XSA
- **Why**: 4% step time savings = ~280 extra training steps. Quality matches MuonEq-R alone.
- **Expected improvement**: More training steps → lower BPB at same wallclock
- **Risk**: Conv layers may not scale to larger models/longer sequences. Needs testing at scale.
- **Remaining validation**: (a) Try 1 conv + 8 attn, (b) Test at seq_len=2048, (c) Verify quantization compatibility

## Promotion Criteria
1. Clear positive signal on screening run (>0.005 BPB improvement over control)
2. Robust across 2+ seeds
3. Simple implementation (low risk of bugs at scale)
4. Compatible with 16MB artifact budget
5. Likely to survive scale-up
6. Distinct from existing leaderboard ideas

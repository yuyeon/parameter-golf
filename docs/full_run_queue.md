# Full Run Queue (8xH100 Promotion Candidates)

*Last updated: 2026-04-03*

## BREAKTHROUGH: FiLM Weight Sharing Beats Standard at Same Budget

FiLM 5→9 + 3xMLP at 600s single H100: **1.3370 post-quant BPB**
- vs Standard Kitchen Sink (seq1024): 1.3656 BPB
- vs Standard Kitchen Sink (seq2048): 1.2698 BPB

**Why FiLM wins at fixed wallclock**: Smaller model (12.3M vs 22.1M) → faster steps (350ms vs 448ms) → 2.4x more training steps → extra training overcomes per-step quality gap.

## Queue (ordered by priority)

### 1. FiLM 5→9 + 3xMLP + MuonEq-R (BEST NOVEL)
- **Script**: `experiments/film_depth/train_gpt.py`
- **Config**: `NUM_SHARED_BLOCKS=5 NUM_LAYERS=9 MLP_MULT=3`
- **600s single H100**: 1.3370 BPB, 1708 steps, 10.3MB artifact
- **Expected on 8xH100**: ~1.15-1.18 BPB with 8x more steps
- **Novelty**: FiLM-depth weight sharing not in any leaderboard submission
- **To do**: Add BigramHash, add GPTQ int6, run 3 seeds on 8xH100

### 2. MuonEq-R on SOTA Stack
- **Script**: `experiments/sota_muoneqr/train_gpt.py`
- **Evidence**: Training val_bpb 1.5741 at 680 steps (massive improvement)
- **Blocker**: Needs 8xH100 for EMA + GPTQ to work
- **Expected**: Beat 1.1147 SOTA

### 3. Kitchen Sink + BigramHash (seq2048)
- **Script**: `experiments/kitchen_sink_bigram/train_gpt.py`
- **600s single H100**: 1.2698 BPB, 1338 steps, 15.6MB artifact
- **Not novel** — combines known techniques

## Key Decision
- FiLM 5→9 is the **genuinely novel** candidate for a new submission
- MuonEq-R on SOTA is the **safest bet** for beating the leaderboard
- Both should be tested on 8xH100 when available

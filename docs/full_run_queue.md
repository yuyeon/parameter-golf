# Full Run Queue (8xH100 Promotion Candidates)

*Last updated: 2026-04-03 — full-budget single-H100 results available*

## Full-Budget Single-H100 Results (600s wallclock)

| Script | Steps | Pre-quant BPB | Post-quant BPB | Artifact Size | GPTQ |
|--------|-------|---------------|----------------|---------------|------|
| **Kitchen Sink + BigramHash** | 709 | 1.3597 | **1.3656** | 13.66 MB | N/A (int8) |
| SOTA + MuonEq-R | 427 | 1.7636 | *GPTQ failed* | — | Cholesky fail |

**Note**: The SOTA script is designed for 8xH100 (parallel banking, DDP). Single-GPU performance is not representative. The Kitchen Sink script works well on single GPU.

## Queue (ordered by priority)

### 1. MuonEq-R on SOTA Stack (8xH100 only)
- **Implementation**: `experiments/sota_muoneqr/train_gpt.py`
- **Evidence**: -0.189 BPB pre-quant at 200 steps (single GPU). GPTQ needs 8xH100 for proper training length.
- **Expected on leaderboard**: 1.10-1.11 BPB (beating current 1.1147 SOTA)
- **Blocker**: Requires 8xH100 to test properly. EMA and GPTQ don't work on single GPU at low step count.

### 2. Kitchen Sink + BigramHash (clean submission)
- **Implementation**: `experiments/kitchen_sink_bigram/train_gpt.py`
- **Config**: 9L/512d, MuonEq-R + XSA9 + LeakyReLU² + SmearGate + 3xMLP + BigramHash 2048 + WD 0.04
- **Evidence**: 1.3656 post-quant BPB at 709 steps (single H100, 600s)
- **Artifact**: 13.66 MB (fits 16MB budget with room for int6 GPTQ)
- **Expected on 8xH100**: ~1.18-1.22 BPB (5-7x more steps, GPTQ compression)
- **To submit**: Add GPTQ int6, add sliding window eval, run 3 seeds on 8xH100

### 3. Kitchen Sink Transplanted to SOTA Banking
- **What**: Use SOTA's parallel Muon banking + our improvements
- **Expected**: Best possible result — faster steps (banking) + better optimization (MuonEq-R) + better arch (kitchen sink)
- **Effort**: Significant implementation — merge banking into kitchen sink

## Key Decision: What to run on 8xH100

If you have 8xH100 access:
1. **Fastest win**: Apply MuonEq-R to SOTA script (3-line change), run 3 seeds → submit
2. **Highest upside**: Kitchen Sink + BigramHash with GPTQ → needs int6 quantization added
3. **Maximum effort**: Transplant kitchen sink to banking architecture → days of work

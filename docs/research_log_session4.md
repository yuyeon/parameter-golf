# Research Log — Session 4 (2026-04-04)

## Environment
- GPU: 1× NVIDIA H100 80GB HBM3 (single GPU instance)
- PyTorch 2.11.0+cu128, FA3 3.0.0, Triton OK, CUDA 12.8
- Baseline verified: 327ms/step, 2.577 BPB @ 50 steps ✓

## Full State Reconstruction

### Our Best Submission (FiLM)
| Config | 1×H100 600s | Expected 8×H100 |
|--------|-------------|-----------------|
| FiLM 5→7+8xMLP FA3+EMA+QAT | 1.2863 pre-quant, 1.3010 int6 | ~1.14-1.18 (speculative) |
| FiLM 5→7+8xMLP+SLOT24 | Not tested (needs 8×H100) | ~1.05-1.10 (very speculative) |

Architecture: 5 shared blocks, 7 virtual layers, 8× MLP expansion, 512d, 8H/4KV.
Novelty: FiLM-depth weight sharing (per-layer modulation of shared blocks).
Advantage: 349ms/step on 1×H100 vs 669ms for SOTA → 2× more training steps.

### Current Competition Landscape (2026-04-04)

#### Non-SLOT, Non-TTT Frontier
| PR | BPB | Key Stack |
|----|-----|-----------|
| **#1334** | **1.0897** | SP4096 + Depth Recurrence(4,5) + Parallel Residuals(7+) + MuonEq-R + QK-Gain 5.0 |
| #1331 | 1.0900 | MuonEq-R + 3-Layer Recurrence + WD=0.095 |
| #1344 | 1.0923 | SP4096 + Polar Express + MuonEq-R + Depth Recurrence(3,4,5) |
| #1279 | 1.0924 | MuonEq-R + Depth Recurrence + N61 Mixed GPTQ |

#### Pre-Quant TTT (GPTQ-compatible — adapts before quantization)
| PR | BPB | Delta vs no-TTT | Method |
|----|-----|-----------------|--------|
| **#1351** | **1.0807** | -0.009 | Discriminative TTT: per-block AdamW LR (0.3x early, 1.0x late), 10 epochs |
| #1326 | 1.0896 | -0.003 | Legal TTT (SGD, freeze early blocks) |

#### Causal SLOT (legality pending, strong legal argument)
| PR | BPB | SLOT delta | Method |
|----|-----|------------|--------|
| **#1350** | **1.0046** | -0.088 | L-BFGS Causal SLOT (25 iter, logit space, context-only loss) |
| #1333 | 1.0766 | -0.013 | Causal SLOT-16 |

#### Full SLOT (likely illegal — uses future token info)
| PR | BPB | Note |
|----|-----|------|
| #1329 | 0.636 | Per-Sample SLOT, 24 steps |
| #1324 | 0.727 | SLOT-48 + VRL |

### SLOT Legality Assessment
- **Standard SLOT (optimize on all positions)**: Almost certainly illegal. PR #1240 proved 100% causal violation.
- **Causal SLOT (optimize only on already-scored positions)**: Strong legal argument — identical principle to legal score-first TTT. No official ruling. Issue #1336 filed, no maintainer response.
- **Our implementation**: Correct stride-based masking, frozen model, per-sample delta. Would need causal variant for safety.

### GPTQ + TTT Incompatibility (Confirmed)
PR #1341 systematic analysis:
- **Post-quant TTT on GPTQ weights**: +0.03 BPP WORSE. GPTQ's column-wise Hessian error compensation creates fragile weight structure that gradient updates destroy.
- **Pre-quant TTT (before GPTQ)**: -0.009 BPP WORKS. Adapts full-precision weights, then quantizes adapted weights.
- **Implication**: TTT and GPTQ are compatible IF TTT happens before quantization. The "incompatibility" is specifically about updating quantized weights.

### Gap Analysis: FiLM vs Non-SLOT Frontier

Our best extrapolated: ~1.14-1.18 BPB
Non-SLOT frontier: 1.0897 BPB
Gap: **~0.05-0.09 BPB**

Sources of the gap (techniques we haven't adopted):
1. **SP4096 tokenizer**: Every top-5 non-SLOT PR uses 4096 vocab. Bigger vocab = more bits per token = better compression of natural language. We use SP1024.
2. **Depth recurrence with untied patterns**: Repeat layers 3-5 (or 4-5), getting 13-14 virtual layers from 11 physical. We have FiLM depth sharing, but it's a different mechanism.
3. **Parallel residuals (layer 7+)**: Separate attention and MLP residual streams. Not tested on FiLM.
4. **QK-Gain 5.0**: Simple scalar multiplication on attention logits. Proven at -0.003 BPP.
5. **Higher WD (0.09-0.10)**: Quantization-friendly weight regularization. We use 0.04.
6. **Pre-quant discriminative TTT**: Per-block AdamW fine-tuning before GPTQ. -0.009 BPP.
7. **4× MLP (with SP4096)**: SP4096 frees embedding params, allowing wider MLP.
8. **Polar Express NS**: 4-step minimax-optimal Newton-Schulz (vs standard 5-step).

### Can FiLM Close the Gap?

**Favorable factors:**
- FiLM's step-time advantage (349ms vs ~106ms×8 GPUs... wait — 8×H100 changes the picture significantly)
- FiLM's parameter efficiency (shared blocks = smaller model = more room in 16MB)
- SLOT/Causal-SLOT is architecture-agnostic — should work with FiLM

**Unfavorable factors:**
- On 8×H100, data parallelism gives SOTA ~5500 steps in 600s. FiLM on 8×H100 might get ~10000 steps, but the per-step quality difference narrows.
- SP4096 requires significant code changes for FiLM
- Several techniques (depth recurrence, parallel residuals) may not compose well with FiLM's weight sharing
- FiLM was optimized for 1×H100 screening; the 8×H100 scaling behavior is unknown

### Critical Uncertainty
We have never run FiLM on 8×H100. The extrapolation is highly uncertain.
The non-SLOT frontier uses techniques that are proven at 8×H100 scale.
FiLM's advantage is from faster steps, but 8×H100 data parallelism may reduce that advantage.

## Strategic Assessment

### Path 1: FiLM + Latest Techniques (Novel, High Risk)
- Add SP4096, QK-Gain, higher WD, pre-quant TTT to FiLM
- Add Causal SLOT for eval
- Risk: Unknown 8×H100 scaling, many untested compositions
- Upside: Genuinely novel submission with potentially unique architecture

### Path 2: Adopt Best Non-SLOT Stack + Our Innovations (Lower Risk)
- Start from PR #1334 stack (SP4096, depth recurrence, parallel residuals)
- Add MuonEq-R (already ours), pre-quant discriminative TTT
- Add Causal SLOT
- Risk: Not novel (stacking known techniques)
- Upside: More likely to place well

### Path 3: FiLM as Alternative Architecture for SLOT (Novel, Medium Risk)
- FiLM's shared blocks might work especially well with SLOT because:
  - Shared blocks create a "compressed" hidden representation
  - SLOT's per-sample delta can exploit this compressed structure
  - Fewer unique parameters = potentially better SLOT optimization landscape
- Test: FiLM vs standard architecture as SLOT base
- Risk: SLOT legality uncertain

## Immediate Priorities
1. Run FiLM 5→7+8xMLP on THIS H100 for 600s to re-verify our baseline
2. Verify FiLM+Causal SLOT works on 1×H100 (even if slow)
3. Test SP4096 tokenizer with FiLM
4. Profile the non-SLOT frontier techniques individually on 1×H100

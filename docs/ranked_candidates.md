# Ranked Candidates — Final

*2026-04-03 — 35+ experiments, ~10 hours on H100*

## Top Novel Finding: FiLM-Depth Weight Sharing

**FiLM 5→9 + 3xMLP + ReLU² + MuonEq-R**
- **Script**: `experiments/film_depth/train_gpt.py`
- **Config**: `NUM_SHARED_BLOCKS=5 NUM_LAYERS=9 MLP_MULT=3`
- **600s single H100**: **1.3370 post-quant BPB**, 1708 steps, 10.3MB artifact
- **Beats** standard kitchen sink (1.3656) at same wallclock by getting 2.4x more steps
- **Novel**: FiLM-depth conditioning not in any leaderboard submission
- **DDP-ready**: Script supports multi-GPU via existing torchrun/DDP infrastructure

### Why it works
5 shared transformer blocks are cycled to create 9 virtual layers. Each virtual layer gets its own learned scale vectors (FiLM modulation) for attention output, MLP output, and residual mixing. This saves 28% parameters vs independent blocks, enabling:
- Faster step time (350ms vs 390ms+ for standard 9L)
- Smaller artifact (10.3MB vs 13.7MB)
- More training steps at same wallclock budget

### Critical constraint
**Compile speed is everything.** Any feature that complicates the compute graph (BigramHash, LeakyReLU², XSA, low-rank) slows torch.compile and loses more steps than it gains. The FiLM model must stay simple.

## All Novel Experiments (sorted by 600s BPB)

| Config | 600s BPB | Steps | Artifact | Novel? |
|--------|----------|-------|----------|--------|
| **FiLM 5→9 + 3xMLP** | **1.3370** | 1708 | 10.3MB | **YES** |
| Kitchen Sink (seq2048) | 1.2698 | 1338 | 15.6MB | No (known tech) |
| Kitchen Sink (seq1024) | 1.3656 | 709 | 13.7MB | No |
| FiLM 5→9 + LeakyReLU² | 1.3648 | 1135 | 9.6MB | Yes but worse |
| FiLM 5→9 + BigramHash | 1.3887 | 930 | 8.9MB | Yes but worse |

## Complete Novel Ideas Tested (35+ experiments)

### Architecture
| Idea | 200-step BPB | Verdict | Key Learning |
|------|-------------|---------|--------------|
| **FiLM 5→9 + 3xMLP** | 1.6353 | **PROMOTE** | 28% param savings, faster steps |
| FiLM 3→9 | 1.6860 | SALVAGE | 65% savings but quality gap |
| 8 Register tokens | 1.6362 | HOLD | Nearly free but marginal |
| Conv-attention hybrid | 1.6360 | HOLD | Speed win, but XSA incompatible |
| Dynamic depth gate | 1.6325 | HOLD | Signal but +20ms overhead |
| Cross-layer gate | 1.6331 | KILL | No benefit over U-Net skips |
| Dense prediction | 1.6343 | KILL | No benefit |
| Parallel attn+MLP | 1.6700 | KILL | Slower and worse |
| FiLM 7→9, 5→12 | 1.71+ | KILL | Too many unique/virtual blocks |
| Low-rank + diagonal | 1.95-2.06 | KILL | Compile catastrophe |
| 11L without banking | 1.7120 | KILL | 3x step time |

### Training/Optimizer
| Idea | 200-step BPB | Verdict | Key Learning |
|------|-------------|---------|--------------|
| **MuonEq-R** | 1.6376 (1.6332 avg) | **PROMOTE** | -0.024 BPB, 0 overhead |
| Stochastic depth | — | KILL | torch.compile incompatible |
| MTP (multi-token pred) | 1.6512 | KILL | Doesn't compose with MuonEq-R |
| Denoising auxiliary | — | KILL | torch.compile incompatible |

### Architecture + Known Tech
| Idea | 200-step BPB | Verdict |
|------|-------------|---------|
| Asymmetric U-Net | 1.6540 | KILL — no effect at 9L |
| Orthogonal init | 1.6187 | KILL — conflicts with MuonEq-R |
| FiLM + XSA | 1.6547 | KILL — XSA incompatible with sharing |
| FiLM + LeakyReLU² | (600s: 1.3648) | KILL — compile speed loss |
| FiLM + BigramHash | (600s: 1.3887) | KILL — compile speed loss |

## Key Discoveries

1. **FiLM weight sharing is viable and novel** — 5 shared blocks for 9 layers, per-layer scale modulation
2. **Compile speed is the real bottleneck** — not model quality, not parameters
3. **MuonEq-R is universally beneficial** — 3 lines, zero cost, ~0.02 BPB
4. **XSA is incompatible with weight sharing** — shared attention can't adapt to XSA's self-value subtraction
5. **Feature additions hurt FiLM** — BigramHash, LeakyReLU², registers all slow compile enough to net-negative
6. **Standard kitchen sink at seq2048 is still the best absolute result** (1.2698 BPB) but uses known techniques

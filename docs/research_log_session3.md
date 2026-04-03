# Research Log — Session 3 (2026-04-03)

## Environment
- GPU: NVIDIA H100 80GB HBM3 (single GPU)
- PyTorch 2.11.0+cu128, FA3 3.0.0, Triton OK
- Full H100 setup verified: 326ms/step baseline, 2.577 BPB @ 50 steps

## Key State Reconstruction

### 1×H100 Comparison Framework
We're using this GPU to screen ideas, comparing against #1 on the same hardware.
Winners go to 8×H100 for full evaluation.

### Known 1×H100 results:
| Model | Steps@200 | BPB@200 | ms/step | Steps@600s | BPB@600s |
|-------|-----------|---------|---------|------------|----------|
| #1 (clean) | 200 | 1.9295 | 669 | ~680-900 | ~1.57-1.67* |
| #1 + MuonEq-R | 200 | — | — | 680 | 1.5741 |
| FiLM 5→7+8xMLP | 200 | 1.6255 | 379 | 1601 | 1.2912 |
| Kitchen Sink | 200 | 1.6077 | 385 | ~1560 | 1.3656 |

*extrapolated from MuonEq-R runs; clean 600s running now

### Scaling extrapolation (speculative)
FiLM 200→600s: 1.6255 → 1.2912 = -0.334 BPB for ~8× more steps
If FiLM gets ~10-13K steps on 8×H100, rough extrapolation: ~1.09-1.14 BPB
#1 on 8×H100: 1.1147 BPB (actual, ~7000 steps)

This is speculative but the 1×H100 signal strongly favors FiLM.

### Blockers identified from previous sessions:
- SOTA stack needs 8×H100 for EMA convergence (>2000 steps)
- GPTQ needs well-trained weights (Cholesky fails at ≤200 steps)
- FiLM is at a local optimum for modifications (Round 4: all 6 novel ideas killed)
- XSA incompatible with FiLM weight sharing
- LeakyReLU², BigramHash hurt FiLM (compile overhead eats step budget)

## Running Experiments
- [ ] #1 clean baseline on 1×H100 for 600s (in progress)
- [ ] FiLM 5→7+8xMLP with int6 for 600s (queued)

## Novel Ideas Under Investigation
(see below — being developed while GPU experiments run)

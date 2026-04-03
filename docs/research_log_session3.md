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
| **#1 (clean)** | 200 | 1.9295 | 669 | **894** | **1.3813** |
| #1 + MuonEq-R | 200 | — | — | 680 | 1.5741 |
| **FiLM 5→7+8xMLP** | 200 | 1.6255 | 379 | **1601** | **1.2912** |
| Kitchen Sink | 200 | 1.6077 | 385 | ~1560 | 1.3656 |

**KEY FINDING: FiLM beats #1 by 0.090 BPB on 1×H100 at 600s.**
FiLM gets 1.79× more steps (375ms vs 671ms/step) and that speed advantage
translates directly to better BPB. EMA is broken for both on 1 GPU (too few steps).

Note: #1 uses BigramHash=3072, XSA-all, SmearGate, VE, EMA, banking, 786K batch.
FiLM uses none of these but wins on raw training efficiency.

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

## Novel Ideas Implemented (ready to screen)

### 1. FiLM + FA3 + Partial RoPE (`experiments/film_fa3/`)
- **FA3**: Replace PyTorch SDPA with Flash Attention 3 (Hopper-native). Free speedup.
- **Partial RoPE (16/64 dims)**: Only first 16 of 64 head dims get position encoding.
  48 dims attend position-free (can learn position-invariant patterns).
  Zero parameter cost, proven on #1 submission.
- **Rationale**: More steps in 600s (FA3 speed) + better attention capacity (Partial RoPE).
  Neither was tried on FiLM before. XSA was shown incompatible with FiLM, but Partial
  RoPE is fundamentally different (changes encoding, not output processing).

### 2. Differential Attention FiLM (`experiments/film_diffattn/`)
- **Mechanism**: Each head splits into 2 sub-heads at head_dim/2. Two FA3 calls,
  outputs subtracted with learnable lambda. Cancels attention noise.
- **Parameter cost**: ~4 floats/head for lambda + GroupNorm ≈ negligible
- **Compute cost**: 2× FA3 calls at half head_dim. Net similar FLOPs but more kernel overhead.
- **Risk**: FA3 at head_dim=32 may be less efficient than at 64. Untested at <1B scale.
- **Novel**: No submission or PR uses Differential Attention. This is genuinely new.

### Research notes on rejected ideas
- **Sigmoid Attention**: Can't use FA3 (requires custom kernel). Skip.
- **Linear Attention / GLA / DeltaNet**: All need custom kernels slower than FA3. Skip.
- **SwiGLU MLP**: 3 weight matrices vs 2 for same hidden dim = worse param efficiency at 8×MLP. Skip.
- **LeakyReLU² for FiLM**: Already killed (compile overhead costs 50% more ms/step).

## Experiment Queue (after SOTA baseline completes)
1. FiLM 5→7+8xMLP with int6 (600s) — confirm artifact fits 16MB
2. FiLM + FA3 + Partial RoPE 5→7+8xMLP (200 steps) — compare ms/step and BPP vs baseline FiLM
3. FiLM + Differential Attention 5→7+8xMLP (200 steps) — novel mechanism screen
4. If FA3/DiffAttn help: full 600s runs with int6

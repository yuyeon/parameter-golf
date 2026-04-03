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

## Completed Experiments (Session 3)

### 1×H100 head-to-head (600s, pre-quant BPB)
| Model | Steps | ms/step | Pre-quant | Int6 post-quant | Artifact |
|-------|-------|---------|-----------|-----------------|----------|
| #1 submission (clean) | 894 | 671 | 1.3813 | 1.9091 (broken) | 7.5MB |
| FiLM SDPA | 1577 | 381 | 1.2930 | 1.3002 | 13.7MB |
| **FiLM FA3+EMA+QAT** | **1718** | **349** | **1.2863** | **1.3010** | **13.8MB** |

### 200-step screening results
| Config | BPB@200 | ms/step | Verdict |
|--------|---------|---------|---------|
| FiLM FA3 seq1024 524K (baseline) | 1.6196 | 353 | **BEST on 1 GPU** |
| FiLM FA3 seq2048 524K | 1.6049 | 374 | marginal on 1 GPU, promising for 8×H100 |
| FiLM FA3 seq2048 786K | 1.5921 | 551 | best per-step, too slow on 1 GPU |
| FiLM Partial RoPE 16/64 | 1.6381 | 359 | **KILL** — hurts FiLM |
| FiLM Differential Attention | 1.6594 | 477 | **KILL** — +0.040 worse, 35% slower |

### Previously tried (from experiment log)
- FiLM 5→7+10xMLP: 1.6257 BPB, 429ms, 16.8MB — marginal vs 8x, exceeded 16MB with int8 (might fit int6)
- FiLM 3→7+12xMLP: 1.6444 BPB — KILLED (too few unique blocks)

### Not yet tried
- FiLM 5→7+9xMLP (wider MLP, between 8x and 10x)
- FiLM 4→7 or 6→7 (different shared/unique block ratios)
- Warmdown tuning (SOTA uses 3500 vs our 1200)
- Optimizer hyperparameter tuning (matrix_lr, muon_momentum, etc.)

### Key observations
- MuonEq-R already in film_fa3 (inherited from earlier work)
- Partial RoPE hurts FiLM (shared blocks need full position info)
- Differential Attention killed (FA3 at head_dim=32 too slow)
- EMA doesn't help on 1 GPU (not enough steps), expected to help on 8×H100
- Late QAT triggered at step 1538 on 1 GPU

### Session 3 novel experiments killed (5 total)
1. Partial RoPE: -0.019 BPP worse (shared blocks need full position info)
2. Differential Attention: +0.040 BPP worse, 35% slower (FA3 at head_dim=32 inefficient)
3. Seq-len curriculum: torch.compile dynamic=True 40s overhead; MLP-bound arch gains little
4. Cross-layer KV sharing: +0.100 BPP worse (destroys layer specialization)
5. Factored MLP: +0.008 BPP worse, 4% slower (deeper nonlinearity doesn't help vs wide relu²)

### Competition landscape update (2026-04-03)
SLOT (Scored-position Learnable Optimization at Test-time) is the new paradigm:
- PR #1313: 0.8637 BPB (SLOT-24, stride=96) — best pending
- PR #1229: 0.9300 BPB (SLOT-16, stride=64)
- Merged #1: 1.1147 BPB (no SLOT)

SLOT is test-time only: optimizes per-sample delta + logit_bias on frozen hidden states.
Architecture-agnostic. ~0.25 BPP gain. Implemented for FiLM in experiments/film_slot/.

Running: SLOT24 (PR #1313) baseline on 1×H100 for comparison.

### Recommended 8×H100 submission
```bash
NUM_SHARED_BLOCKS=5 NUM_LAYERS=7 MLP_MULT=8 USE_INT6=1 TRAIN_SEQ_LEN=2048 SLOT_ENABLED=1 \
  torchrun --nproc_per_node=8 experiments/film_slot/train_gpt.py
```

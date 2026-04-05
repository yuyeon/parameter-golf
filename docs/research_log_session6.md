# Research Log — Session 6 (2026-04-05)

## Environment
- GPU: 1× NVIDIA H100 80GB HBM3
- PyTorch 2.11.0+cu128, FA3 3.0.0, Triton OK, CUDA 12.8
- Fresh environment setup (pip bootstrap, torch, FA3, deps)

## Competition Frontier (reconstructed 2026-04-05)

### Merged SOTA
- **1.1147 BPB** — PR #1019 (abaybektursun): AR Self-Gen GPTQ + XSA-all + BigramHash3072

### Best Open Record PRs (legal, 10min 8×H100)
| PR | BPB | Method | Key Innovation |
|----|-----|--------|---------------|
| #1350 | **1.0046** | L-BFGS Causal SLOT | Logit-space adaptation with causal masking |
| #1372 | 1.0050 | Cascaded 2-Phase L-BFGS SLOT | Faster variant of #1350 |
| #1333 | 1.0766 | AdamW Causal SLOT-16 | Weaker SLOT variant on same base |
| #1351 | 1.0807 | Discriminative TTT | Per-block LR, no SLOT |
| #1334 | 1.0897 | SP4096+DepthRecur+ParallelResid+MuonEq-R | Clean, no SLOT/TTT |
| #1344 | 1.0923 | SP4096+PolarExpress+MuonEq-R+DepthRecur | Similar to #1334 |
| #1354 | 1.1093 | Varlen+FusedMLP+TTT | Speed improvements |
| #1364 | 1.1025 | Pre-quant AdamW TTT + QK-Gain 4.0 | |
| #1322 | 1.0854 | Lucky V (SLOT-32) | AdamW SLOT hidden-space |

### Non-Record (unlimited compute)
| PR | BPB | Method | Significance |
|----|-----|--------|-------------|
| #1370 | **1.003** | 10L Gated DeltaNet (pure GDN) | Softmax-free, 7k steps 2×A100 |
| #1371 | 1.399 @16k | GDN-Attention Hybrid | Wins over attention at long context |
| #1368 | 0.850 | Mean-delta SLOT + depth recurrence | Very strong SLOT |

### Key Observations
1. **The true frontier is ~1.005 BPB** (PR #1350 L-BFGS causal SLOT)
2. **Pre-SLOT base model frontier is ~1.09 BPB** (PR #1334)
3. **SLOT adds -0.087 BPB** on the #1019 base (L-BFGS logit-only, causal)
4. **GDN achieves 1.003 non-record** — competitive with best legal record!
5. **Our FiLM approach**: 1.286 pre-quant 1×H100 600s (extrapolated ~1.02-1.06 on 8×H100)

## Gap Analysis

To beat 1.005 BPB (PR #1350), we need EITHER:
1. A pre-SLOT base significantly better than 1.09 (so SLOT pushes below 1.005)
2. A better test-time adaptation than L-BFGS logit SLOT
3. A fundamentally different approach

### Decomposition of PR #1350's 1.005 BPB
- Base model training: ~1.09 BPB (SP4096, depth recurrence, etc.)
- Pre-quant TTT: -0.022 BPB → ~1.07
- GPTQ + compression: ~neutral
- L-BFGS causal SLOT: -0.087 BPB → ~1.005

## Novel Directions (Bitter Lesson Aligned)

### Direction 1: GDN Hybrid (HIGHEST PRIORITY)

**Thesis**: Replace most attention layers with Gated DeltaNet (linear attention). This is:
- Bitter Lesson aligned: learns token mixing instead of hardcoding softmax
- Proven: 1.003 BPB non-record (PR #1370)
- Faster per step: O(n) vs O(n²) → more training steps in 600s
- Enables longer training contexts (8k-32k viable)

**Evidence**:
- PR #1370: Pure GDN achieves 1.003 BPB (29.9M params, 7k steps, 2×A100)
  - Uses BigramHash(3072,112) + trigram, SwiGLU 3× MLP, RMSNorm
  - EMA + SWA + Late-QAT + GPTQ int6 + zstd-22
  - Score-first TTT: SGD momentum 0.9, 3 epochs/chunk
- PR #1371: GDN-Attention hybrid wins at 16k+ context
  - 3:1 ratio GDN:attention (Olmo Hybrid design)
  - Crossover: hybrid loses at 8k, wins at 16k+
  - At 32k: hybrid 1.471 vs attention 1.706 (-0.235 BPB!)

**Plan**:
1. Install flash-linear-attention (FLA) library for GDN kernels
2. Implement hybrid: 8 GDN layers + 3 attention layers (or similar ratio)
3. Add proven stack: SP4096, MuonEq-R, depth recurrence, GPTQ int6
4. Add causal SLOT (L-BFGS logit-only)
5. Screen on 1×H100 at 200 steps and 600s
6. If promising, run full 8×H100 submission

**Expected**: If GDN hybrid trains 30-50% faster (more steps) and matches quality per step, the base model could reach ~1.05-1.08 BPB. With SLOT: ~0.97-1.00 BPP.

**Risk**: GDN may need different optimizer recipe. Torch.compile compatibility unclear. 10-min budget may be tight with GDN+GPTQ+SLOT.

**UPDATE (tested 2026-04-05)**: GDN is 3-16× SLOWER than FA3 on H100 at all sequence lengths tested (1k-16k). FA3 on Hopper is too well optimized. **GDN path is DEAD for record track on H100.**

| Seq Length | GDN ms | FA3 ms | GDN/FA3 ratio |
|-----------|--------|--------|---------------|
| 1024 | 1.43 | 0.09 | 16.0× slower |
| 4096 | 1.40 | 0.15 | 9.5× slower |
| 8192 | 1.39 | 0.26 | 5.4× slower |
| 16384 | 1.40 | 0.49 | 2.9× slower |

The non-record GDN result (PR #1370, 1.003 BPB) was achieved on A100 where FA3 is unavailable. On H100, pure attention with FA3 is strictly dominant.

### Direction 2: Multi-Scale Causal SLOT (NOVEL)

**Thesis**: Current SLOT optimizes a single 1024-dim (or vocab_size) broadcast logit bias. This is position-independent. What if we optimize at multiple scales?

**Ideas**:
- **Layer-wise hidden deltas with L-BFGS**: Instead of only logit bias, optimize additive deltas at multiple intermediate layers (e.g., every 3rd layer). Still causal (only use scored positions for optimization). More expressive adaptation.
- **Frequency-domain SLOT**: Decompose the logit bias into frequency components. Low-frequency components capture document-level biases, high-frequency capture local patterns. Optimize coefficients of a small DCT basis.
- **Vocabulary-cluster SLOT**: Group tokens into clusters (by embedding similarity). Optimize per-cluster biases instead of per-token. More structured, fewer parameters.
- **Per-position logit modulation**: Instead of position-independent bias, learn a position-dependent scaling. Use a small linear model: `bias(pos) = W @ pos_encoding + b` where W is tiny.

**Expected gain**: If the best broadcast logit SLOT gives -0.087, multi-scale should give -0.10 to -0.12.

**Risk**: L-BFGS may not scale well to many parameters. Eval time budget (~550s) limits complexity.

### Direction 3: Ternary Quantization with Larger Models

**Thesis**: int6 fits ~25M params in 16MB. Ternary fits 73M (3×). If we can train 73M well in 600s, the larger model should be better.

**Evidence**:
- PR #640: 73M ternary achieves 1.1570 BPB (competitive)
- PR #641: 106M binary achieves 1.1239 BPB (non-record, 2hr)
- Scaling law: 3× params should give significant BPB improvement

**Risk**: Ternary training harder. May not converge in 600s. Step time with 73M params on 8×H100?

### Direction 4: Self-Distillation Pipeline

**Thesis**: Use the 10-minute budget more efficiently:
1. Train a large model for 7 minutes (unconstrained by 16MB)
2. Distill to a small model for 3 minutes
3. Quantize and compress the small model

The teacher can be much larger and better. The student inherits the knowledge.

**Risk**: 3 minutes may not be enough for distillation. Teacher may not be good enough in 7 min.

### Direction 5: Continuous Depth (Neural ODE style)

**Thesis**: Instead of discrete layers, use a neural ODE formulation where depth is continuous. This naturally enables "depth recurrence" as a special case and can be more parameter-efficient.

**Risk**: Adaptive solvers are slow. Fixed-step may not be better than standard layers.

## Prioritized Experiment Plan

### Phase 1: Baseline Verification (today)
1. ✅ Environment setup
2. Download SP1024 + SP4096 data
3. Verify baseline training (50 steps)
4. Time baseline steps on 1×H100

### Phase 2: Strong Base Implementation (today)
1. Get PR #1334 stack working (SP4096 + depth recur + parallel resid + MuonEq-R)
2. Verify it reaches ~1.30-1.35 BPB on 1×H100 600s (extrapolated from #1334's 8×H100 result)
3. Add causal SLOT (L-BFGS logit-only)
4. This is our "known-good" baseline for comparison

### Phase 3: GDN Hybrid (today-tomorrow) ← MAIN BET
1. Install FLA library
2. Implement GDN hybrid architecture
3. 200-step screening vs attention baseline
4. If step time is faster: 600s full run
5. Add SLOT, compare against pure-attention + SLOT

### Phase 4: Novel SLOT Improvements (tomorrow)
1. Test layer-wise L-BFGS SLOT
2. Test frequency-domain SLOT
3. Test per-position logit modulation
4. If any beats broadcast SLOT by >0.01 BPB: promote to 8×H100

### Phase 5: 8×H100 Submission
1. Combine best base + best SLOT + best novel techniques
2. Get 8×H100 access (RunPod)
3. Run 3-seed validation
4. Submit PR

## Experiment Results

### GDN Hybrid — KILLED
GDN is 3-16× slower than FA3 on H100 at all sequence lengths (1k-16k). FA3 on Hopper is too well optimized. GDN direction dead for record track.

### Adaptive Computation Time (ACT) — KILLED
Tested 3 configurations against baseline:

| Config | ms/step | val_bpb@200 | Params | Verdict |
|--------|---------|-------------|--------|---------|
| Baseline 9L | 331 | 1.722 | ~17M | Reference |
| ACT 3×5 (512d) | 517 | 1.893 | ~6M | WORSE |
| ACT 3×5 (768d) | 923 | ~2.08 | ~16M | WORSE |

**Root cause**: During training, all iterations run for gradient flow — NO training speedup.
ACT's only benefit is eval-time adaptive depth, but the model is already worse from training.
The halting mechanism + iteration conditioning add overhead without reducing effective compute.

**Lesson**: ACT only helps when the model can actually skip computation during training
(requires specialized hardware support or gradient approximation). With standard backprop
and torch.compile, all iterations must execute regardless of halting probability.

### Mixture of Depths (MoD) — KILLED
Soft per-token routing adds overhead without reducing compute. 732ms/step (2.2x slower than baseline 328ms). Compile-safe but useless — the routing gate adds computation, it doesn't remove any.

### HyperGPT (Shared Blocks + FiLM Modulation) — KILLED  
3 shared blocks → 9 or 12 virtual layers. Weight sharing saves artifact size but NOT step time.

| Config | ms/step | val_bpb@steps | Steps (120s) |
|--------|---------|---------------|--------------|
| Baseline 9L (512d) | 328 | 1.721@200 | 365 |
| HyperGPT 3→12 (512d) | 888 | 2.595@136 | 135 |
| HyperGPT 3→9 (768d, 4xMLP) | 1279 | 2.996@100 | 94 |

**Root cause**: Step time ∝ number of layer forward passes, not number of unique parameters.
Sharing parameters saves artifact space but every virtual layer still runs a full forward pass.

### Key Meta-Lesson
**This competition is step-time-limited, not parameter-limited.** The 16MB budget accommodates
~25M params (int6), and a 9-layer 512d model uses exactly ~17M. There's even room for more params!
But 600s at 328ms/step = only ~1800 steps. Any architecture change that increases step time
is strictly worse because fewer steps > smaller model.

**Corollary**: Novel training-time architectures must be FASTER per step, not just more
parameter-efficient. FA3 + torch.compile + Muon set a very tight speed baseline.

## SP4096 Results (1×H100)

SP4096 tokenizer validation (200 steps):
- **SP4096: 1.616 BPB** vs SP1024: 1.737 BPB → **-0.121 BPB improvement**
- Step time: 344ms (only 5% slower than SP1024 at 328ms)
- This is the single largest free improvement available

600s SP4096 result:
- **SP4096 600s: 1.274 BPB** (int8+zlib) at 1742 steps, 344ms/step
- **SP1024 600s: 1.342 BPB** (int8+zlib) at 1197 steps, 502ms/step
- **Delta: -0.068 BPB** from tokenizer alone
- Artifact: 16.0 MB (int8+zlib, needs int6 for competition)
- This is the new best 1×H100 baseline

| Config | Steps | ms/step | val_bpb | int8_bpb | Artifact |
|--------|-------|---------|---------|----------|----------|
| SP1024 600s | 1197 | 502 | — | 1.342 | 8.9 MB |
| SP4096 600s | 1742 | 345 | 1.272 | 1.274 | 16.0 MB |

## 600s Baseline (1×H100)
- **1.342 BPB** (int8+zlib) at 1197 steps, 502ms/step
- Model: 9 layers, 512d, 2x MLP, SP1024
- This is the reference for SLOT beam testing

## Summary of All Novel Architecture Experiments

| Idea | Status | Root Cause of Failure |
|------|--------|----------------------|
| GDN Hybrid | KILLED | FA3 is 3-16x faster than GDN on H100 |
| ACT Transformer | KILLED | All iterations run for gradients → no speedup |
| Progressive Depth | KILLED | torch.compile needs static graphs |
| Mixture of Depths | KILLED | Soft routing adds computation, doesn't remove any |
| HyperGPT | KILLED | Weight sharing saves artifact size, not step time |
| Bottleneck MLP | KILLED | Kernel launch overhead dominates (2% speedup) |

## Key Meta-Lessons

1. **Competition is step-time-limited**: 600s / 328ms = ~1800 steps max. Any slowdown is fatal.
2. **FA3 + torch.compile define the speed ceiling**: anything that disrupts these is penalized.
3. **Parameter budget is NOT the bottleneck**: 16MB fits ~25M params, model only uses ~17M.
4. **Novel architectures must be strictly faster per step** to win — no "quality for speed" tradeoffs work because the quality gap from fewer steps is always larger.
5. **The only viable novel contribution space for training is**: better optimizer, better loss function, better data loading — things that improve quality WITHOUT changing the computation graph.
6. **Test-time novelty is the most practical avenue**: no torch.compile constraints, unlimited compute budget.

## Path to Winning Submission

### Current Best (1×H100, 600s)
**1.274 BPB** with plain SP4096 baseline (9L, 512d, 2x MLP)

### Extrapolated 8×H100 Performance
On 8×H100, we get ~6000 steps (3.4x more than 1×H100's 1742).
Assuming similar scaling to competition entries:
- SP4096 baseline (our 1.274): → ~1.05-1.10 BPB on 8×H100
- + SOTA techniques (depth recur, parallel resid, MuonEq-R): → ~1.00-1.05
- + L-BFGS causal SLOT: → ~0.92-0.97
- + Novel improvements: → potentially sub-0.92

### Proven SOTA Stack to Adopt
From PR #1334 + #1350 + #1351:
1. SP4096 tokenizer (**verified -0.068 BPB, our biggest win**)
2. Depth recurrence layers 4,5 (13 virtual from 11 physical)
3. Parallel residuals from layer 7
4. MuonEq-R (row-normalize before NS orthogonalization)
5. QK-Gain 5.0
6. Higher weight decay 0.09-0.10
7. Full Hessian GPTQ int6 + brotli compression
8. EMA + tight SWA
9. Late QAT
10. Discriminative per-block TTT (pre-quant)
11. L-BFGS causal SLOT (logit-only, 25 steps)

### Our Novel Contributions
1. **SLOT-Aware Training (SAT)**: train model awareness of test-time SLOT
   - Neutral at 200 steps, needs full-budget testing
2. **Learned Residual SLOT**: error predictor for SLOT warm-start
   - 100K params (98KB), fits in artifact
   - Needs testing on mature model
3. **Parallel SLOT Beams**: multi-strategy SLOT selection
   - Run K beams, pick best per window
   - Novel test-time compute scaling

### Next Steps
1. Build combined submission script with ALL proven techniques
2. Test SAT and learned residual SLOT at full 600s budget on 1×H100
3. Get 8×H100 access and run 3-seed validation
4. Submit PR

## Decision Criteria
- Novel technique must beat broadcast SLOT by >0.005 BPB on 1×H100 to justify further investment
- Any technique must fit within 600s training + 600s eval budget on 8×H100

# Novel Ideas — Bitter Lesson Aligned (Session 6, 2026-04-05)

## The Problem in First Principles

The competition measures **bits per byte** on held-out text. The constraint is:
- 16MB artifact (model + code)
- 10 minutes training on 8xH100
- Unlimited eval time (within reason)

Every existing top submission uses essentially the same recipe:
1. Train an 11-layer 512d transformer with Muon for ~6000 steps
2. Hand-engineered tricks: BigramHash, SmearGate, XSA, depth recurrence, etc.
3. Compress to int6 via GPTQ (~25M params in 16MB)
4. Test-time optimization: L-BFGS SLOT on logit biases (-0.087 BPP)

These are **engineering** contributions, not **learning** contributions. The Bitter
Lesson says: general methods that leverage computation beat hand-engineering.

## What the Bitter Lesson Implies Here

Three computation budgets exist: training compute, artifact size, eval compute.
The winning approach should **learn** how to use all three optimally, not hardcode it.

---

## IDEA 1: Adaptive Computation Time (ACT) Transformer

### Core Insight
Fixed-depth transformers spend the same compute on "the" as on "juxtaposition."
This is wasteful. Easy tokens need 2-3 layers. Hard tokens need 10+.

### Architecture
- 2-3 shared transformer blocks (cycle like depth recurrence, but adaptive)
- Per-iteration conditioning: learned iteration embedding added to residual
- Halting mechanism: each block outputs a scalar halt probability per token
- Cumulative halt probability reaches 1.0 → stop iterating for that token
- ACT loss (Graves 2016): auxiliary penalty on total computation

### Why This Wins at Parameter Golf
- **Faster training**: average 5-6 iterations vs 11 fixed layers → ~2x faster steps → 
  ~2x more training steps in 600s. More training = better model.
- **Better eval quality**: at eval time, use MORE iterations (no time constraint).
  Hard tokens that got 5 iterations in training now get 15-20.
- **Massive parameter efficiency**: 2-3 shared blocks = ~40% of standard params.
  Freed space → wider model or more params for other components.
- **Composes with SLOT**: SLOT works on any architecture.

### Why This Is Bitter Lesson Aligned
- The model LEARNS computation allocation, not hand-engineered
- More compute (deeper iterations) → strictly better predictions
- Scales to any model size: this is the Universal Transformer principle
- Google Brain proved this works (Universal Transformer, 2018)

### Implementation Plan
1. Modify baseline train_gpt.py: replace 11 unique layers with 3 shared blocks
2. Add iteration embedding (learned, per-block, per-iteration)
3. Add halting mechanism (sigmoid gate → accumulated probability)
4. ACT loss: λ * mean(computation_time) with λ=0.01
5. Training: torch.compile with fixed K_max iterations (compile-friendly)
6. Eval: adaptive depth (no compile needed, run until halt)

### Risk Assessment
- Halting policy hard to train in limited steps (MEDIUM risk)
- torch.compile may not handle early exit well (LOW risk — use fixed K_max in training)
- May not converge before training budget expires (MEDIUM risk)

### Expected Impact
- Training speedup: 30-50% faster steps → -0.03 to -0.06 BPP from more training
- Eval quality: adaptive depth → -0.01 to -0.03 BPP on hard tokens
- Total: -0.04 to -0.09 BPP improvement over fixed-depth equivalent

---

## IDEA 2: Amortized Test-Time Adaptation (Learn the SLOT)

### Core Insight
L-BFGS SLOT re-discovers the same patterns window after window. "This is a technical
document, boost technical vocabulary" is discovered by L-BFGS 50,000 times during eval.
This is the opposite of the Bitter Lesson: it's search at test time when learning should
be used.

### Architecture
- Train a small "adaptation network" A during training
- A takes: averaged hidden states of context → predicts logit bias
- A is trained to predict what L-BFGS would converge to
- At eval: A(context) → instant logit bias (one forward pass, no optimization)
- Optional: use A's output as initialization for 2-3 L-BFGS steps (best of both worlds)

### Why This Wins
- **Speed**: replaces 25 L-BFGS steps (~556s) with 1 forward pass (~30s)
- **Quality**: A sees ALL training data patterns; L-BFGS sees only one window
- **Generalization**: A learns cross-document patterns that per-window L-BFGS can't
- **Saved eval time**: with 500s freed up, can do TTT, longer SLOT, ensemble

### Implementation
1. During training (last 100s), for each batch:
   a. Forward pass → hidden states
   b. Run L-BFGS SLOT → optimal logit bias (the "target")
   c. Train A: loss = ||A(avg_hidden) - optimal_bias||^2
2. A architecture: LayerNorm → Linear(512, 512) → GELU → Linear(512, vocab_size)
3. A parameters: ~800K (~3.2KB at int6, negligible artifact size)
4. At eval: A(context) → bias_init, then 3-5 L-BFGS steps for refinement

### Why This Is Bitter Lesson Aligned
- Replaces search (L-BFGS at test time) with learning (trained predictor)
- More training data for A → better initial guesses → better final quality
- This is literally the "learning beats search" principle

### Risk Assessment
- A must generalize from training → eval distribution (MEDIUM risk)
- L-BFGS targets may be noisy during training (LOW risk — average over batches)
- Overhead during training (LOW risk — only last 100s, small model)

### Expected Impact
- Speed: 10-20× faster SLOT eval (30s vs 556s)
- Quality: equivalent or slightly better than pure L-BFGS (better initialization)
- Freed eval time: enables stacking multiple techniques
- Total: same or better SLOT quality, massive speed advantage

---

## IDEA 3: Learned Weight Compression (Neural Codec for Model Weights)

### Core Insight
int6 quantization is uniform and ignores weight structure. A Gaussian distribution
centered at zero has entropy < 6 bits. Learned compression can exploit:
- Non-uniform weight distribution → variable-length coding
- Cross-layer similarity → shared codebooks
- Low-rank structure → factored representations

### Architecture
- Each weight matrix W = Codebook(indices) + Residual
- Codebook: shared across all layers, learned during training
- Indices: per-weight pointer into codebook (low bits)
- Residual: per-weight correction (very low bits)
- Total bits per weight: 3-4 (vs 6 for int6) → 40-50M params in 16MB

### Why This Wins
- **60-100% more parameters**: 40-50M vs 25M at int6
- Scaling law: sqrt(2) more params → ~0.05 BPP improvement
- More parameters with same training = better model

### Implementation
- Product quantization: split each weight into sub-vectors of size 4
- Shared codebook per sub-vector: 256 entries (8 bits per sub-vector, 2 bits per scalar)
- Residual: 2-bit correction per scalar
- Total: ~4 bits per weight → ~33M params in 16MB
- Training: codebook updated via straight-through estimator
- Differentiable quantization throughout training (like QAT but with codebook)

### Why This Is Bitter Lesson Aligned
- LEARNED compression instead of hand-designed quantization
- The codebook discovers weight structure automatically
- Scales: larger models benefit even more from learned compression
- This is the principle behind VQ-VAE applied to weights

### Risk Assessment
- Codebook training might not converge (MEDIUM risk)
- Product quantization overhead (LOW risk — lookup is fast)
- Codebook needs to be in artifact (LOW risk — 256 × 4 × float16 = 2KB)

### Expected Impact
- Parameter count: 33M vs 25M (+32%)
- Quality: -0.03 to -0.05 BPP from more parameters
- Compose with all other techniques

---

## IDEA 4: Test-Time Compute Scaling via Parallel SLOT Beams

### Core Insight
Current SLOT runs one optimization per window. But the eval budget is large.
What if we ran K parallel optimizations with different parameterizations and
picked the best per window?

### Architecture
- Beam 1: L-BFGS on logit biases (standard, proven -0.087 BPP)
- Beam 2: L-BFGS on hidden-space additive deltas at layer 6 (non-convex)
- Beam 3: L-BFGS on attention temperature scaling per head (calibration)
- Beam 4: L-BFGS on residual stream scaling factors per layer
- For each window: run all K beams, score on context, pick best beam for new tokens

### Why This Wins
- Each beam captures different aspects of adaptation:
  - Logit bias: vocabulary distribution
  - Hidden delta: representation correction
  - Temperature: confidence calibration
  - Residual scaling: layer importance weighting
- Best-of-K is strictly better than any single approach
- Non-convex beams (hidden delta) can find corrections logit-bias can't

### Why This Is Bitter Lesson Aligned
- More test-time compute → strictly better results
- Search over adaptation strategies (which beam works best for this window)
- Scales: more beams = better, unlimited by compute

### Implementation
- Implement each beam as a separate SLOT variant
- Run all beams in parallel (GPU parallelism across beams)
- Per window: evaluate each beam on context loss, select argmin
- Score new tokens with the winning beam's delta

### Expected Impact
- Additional -0.01 to -0.03 BPP beyond best single SLOT
- Linear scaling with eval compute
- Novel: no one in competition does multi-strategy SLOT

---

## IDEA 5: Progressive/Curriculum Depth During Training

### Core Insight
Training starts with random weights where deep layers add noise. Early in training,
only shallow layers contribute meaningfully. Using 11 layers from step 0 wastes compute.

### Architecture
- Start with 4 active layers (fast steps, ~150ms)
- Every 1000 steps, activate one more layer
- By step 5000: all 11 layers active (or 13 with recurrence)
- Early steps run 2-3x faster → 30-50% more total steps in 600s

### Why This Wins
- More total optimizer steps in same wallclock
- Early layers get 2x more gradient updates → better foundations
- Late layers only train once early layers have good representations
- This is standard in curriculum learning literature

### Why This Is Bitter Lesson Aligned
- Optimizes the use of compute over training trajectory
- General principle: scale computation with model readiness
- Proven at scale: GPT-3 used progressive training

### Implementation
- Simple masking: set inactive layers to identity (skip connection only)
- Unfreeze in order: layers 0-3 first, then 4-6, then 7-10
- No architecture changes needed

### Expected Impact
- 20-40% more total steps in 600s
- -0.01 to -0.03 BPP from better early training
- Zero risk (layers 0-3 already exist, we just activate more over time)

---

## Prioritization

| Idea | Novelty | Expected BPP Gain | Implementation Effort | Risk | Priority |
|------|---------|-------------------|----------------------|------|----------|
| 1. ACT Transformer | Very High | -0.04 to -0.09 | High | Medium | **1** |
| 2. Amortized SLOT | Very High | eval speed + -0.01 | Medium | Medium | **2** |
| 3. Learned Compression | High | -0.03 to -0.05 | High | Medium | 3 |
| 4. Parallel SLOT Beams | Medium-High | -0.01 to -0.03 | Low | Low | **4** |
| 5. Progressive Depth | Medium | -0.01 to -0.03 | Low | Very Low | **5** |

**Recommended order**: Start with 5 (low risk, fast to implement, immediate benefit),
then 1 (highest upside, needs careful implementation), then 4 (fast to implement),
then 2 (medium effort, high payoff).

Idea 3 is highest effort and should only be attempted if we have time.

# Novel Optimizer Ideas (from first principles)

## Background: Muon's linear algebra

For weight matrix W ∈ R^{M×N}, gradient G = ∂L/∂W.
SVD: G = UΣV^T where U ∈ R^{M×k}, Σ = diag(σ₁,...,σ_k), V ∈ R^{N×k}

**Muon** computes: update = UV^T (polar factor, all σᵢ → 1)
**MuonEq-R** adds: row-normalize G before NS5 (equivariance fix)

## The spectrum of update strategies

| Method | Singular values of update | Information preserved |
|--------|--------------------------|---------------------|
| SGD | σᵢ (raw) | All — but poorly conditioned |
| Muon | 1 (uniform) | Directions only, magnitudes discarded |
| Spectral norm | σᵢ/σ₁ | Relative magnitudes, bounded |
| Matrix sqrt | √σᵢ | Partial magnitudes, softer than SGD |
| Partial NS5 | between σᵢ and 1 | Gradually discard magnitudes |

## Novel proposals

### 1. Spectral Muon (G / ||G||₂)
Normalize by largest singular value (spectral norm) instead of orthogonalizing.
Preserves full singular value structure, just bounds the max.
Cheap: power iteration for top singular value (2 iterations).

### 2. Muon-Soft (2 NS5 steps instead of 5)  
Fewer NS5 steps = "partial" orthogonalization.
The singular values converge to 1 gradually through the iteration.
At 2 steps, they're partway between original and uniform.

### 3. Column-Equivariant Muon (MuonEq-C)
MuonEq-R normalizes rows. MuonEq-C normalizes columns.
Rows = output features. Columns = input features.
Column normalization makes the optimizer equivariant to input-feature scaling.
Could help when input features have very different scales.

### 4. Bi-Equivariant Muon (MuonEq-RC)
Normalize BOTH rows and columns (Sinkhorn-like alternation).
Makes optimizer equivariant to both input and output feature scaling.
1 round: row-normalize, then column-normalize.

## Results (200 steps, FiLM 5→7+8xMLP, seed=42)

| Optimizer | BPB@200 | ms/step | vs MuonEq-R |
|-----------|---------|---------|-------------|
| **MuonEq-C** | **1.6098** | 354 | **-0.0095 BETTER** |
| MuonEq-R | 1.6193 | 353 | baseline |
| MuonEq-RC | 1.6201 | 354 | +0.0008 (noise) |
| Muon-Soft | 1.9398 | 350 | +0.320 KILL |
| Spectral | 2.3530 | 358 | +0.734 KILL |

### Analysis
- **Full NS5 orthogonalization is essential.** Partial (2 steps) or spectral normalization
  both catastrophically worse. The gradient's singular values must be flattened.
- **Column normalization > row normalization for FiLM.** Shared blocks see heterogeneous
  inputs from different virtual layers (FiLM modulation). Column normalization equalizes
  input-feature scales, which is the dominant conditioning issue.
- **Bi-equivariant is neutral.** Row normalization after column normalization partially
  undoes the column benefit.
- **Novel insight:** The choice of pre-NS5 normalization axis (row vs column) matters
  and depends on the architecture's input distribution characteristics.
- **MuonEq-C killed at convergence:** Early advantage (-0.013 at step 200) evaporates
  to +0.0005 worse at step 1722. Row-equivariance wins at long training.

## v2: Weight-aware optimizers (ALL KILLED)

Three optimizers that use W (current weights) — information Muon ignores:
- **NSG (null-space gradient)**: +0.008 worse. Projecting away W's column space doesn't help.
- **WIP (weight-inverse precond)**: crashed (Cholesky instability).
- **Procrustes (multiplicative rotation)**: diverged to 18.29 BPB.

## Deep theoretical analysis

### Why Muon is provably hard to beat

Key identity: G @ (G^TG)^{-1/2} = UV^T (the polar factor).

This means Muon computes the EXACT optimal right-preconditioning of G
by the gradient's own covariance structure. It's the solution to:

  argmax_X tr(G^T X) subject to ||X||_2 ≤ 1

To beat Muon at the same cost, we need information BEYOND the current
gradient. The gradient already encodes weight information via chain rule.

### GPU performance constraints

NS5 (5 matrix multiplications) is perfectly suited to GPU SIMD:
- 512×512: NS5=0.32ms, QR=4.01ms (NS5 is 12× faster)
- 4096×512: NS5=3.69ms, QR=9.90ms (NS5 is 2.7× faster)
- SVD: 53ms for 512×512 (166× slower than NS5)

Any alternative must be expressible as a SHORT SEQUENCE OF MATRIX
MULTIPLICATIONS to compete with NS5 on GPU. QR, SVD, Cholesky all
use sequential operations that GPUs handle poorly.

### What COULD beat Muon (theoretical)

Information sources beyond the current gradient:
1. Gradient HISTORY (momentum does this, but in element space not spectral space)
2. Loss landscape curvature (2nd-order info — too expensive)
3. Cross-layer gradient correlations (unexplored, could help)
4. Data distribution statistics (batch-level information)

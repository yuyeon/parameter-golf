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

# Experiment Log

*Chronological record of all experiments — 2026-04-03*

## Environment
- GPU: NVIDIA H100 80GB HBM3
- CUDA: 12.8, Driver 580.126.09
- PyTorch: 2.11.0+cu128, Flash Attention 3: 3.0.0
- All experiments: single H100, post-int8-quant BPB on FineWeb validation

---

## Controls (200 steps)

| ID | Script | Config | Seed 42 BPB | Seed 1337 BPB | ms/step |
|----|--------|--------|-------------|---------------|---------|
| C1 | train_gpt.py | 9L/512d/8H/4KV, ReLU², 2x MLP | 1.6541 | 1.6596 | 330 |
| C2 | SOTA (2026-03-25) | 11L, banking, XSA-all, BigramHash3072 | *(GPTQ failed)* | — | 669 |
| C3 | Depth Recurrence | 3 unique blocks, 9 eff depth, LoRA r=4 | 1.8151 | — | 546 |

---

## Round 1: Basic Ideas

### Exp 1: Asymmetric U-Net Split
**Script**: `experiments/asymmetric_split/train_gpt.py`

| Split | BPB | ms/step | Verdict |
|-------|-----|---------|---------|
| 4/5 (default) | 1.6541 | 330 | control |
| 3/6 | 1.6540 | 329 | noise |
| 2/7 | 1.6543 | 328 | noise |
| 1/8 | 1.6824 | 571 | **KILL** |

### Exp 2: Multi-Token Prediction (MTP)
**Script**: `experiments/mtp/train_gpt.py`

| Config | BPB | ms/step | Verdict |
|--------|-----|---------|---------|
| k=2 w=0.15 | 1.6514 | 339 | small |
| k=2 w=0.30 | 1.6516 | 340 | small |
| k=3 w=0.15 | 1.6512 | 339 | small |

**KILL** — doesn't compose with MuonEq-R.

### Exp 3: MuonEq-R
**Script**: `experiments/muoneqr/train_gpt.py` | **Diff**: 3 lines in Muon.step()

| Seed | Baseline | MuonEq-R | Delta |
|------|----------|----------|-------|
| 42 | 1.6541 | **1.6376** | **-0.0165** |
| 1337 | 1.6596 | **1.6287** | **-0.0309** |

**PROMOTE** — largest single improvement.

### Exp 4: MuonEq-R + MTP
| Seed | MuonEq-R | +MTP | Delta |
|------|----------|------|-------|
| 42 | 1.6376 | 1.6409 | +0.003 worse |
| 1337 | 1.6287 | 1.6442 | +0.016 worse |

**KILL** — interference.

### Exp 5: MuonEq-R + XSA
**Script**: `experiments/muoneqr_xsa/train_gpt.py`

| Config | BPB | ms/step |
|--------|-----|---------|
| +XSA4 | 1.6368 | 341 |
| +XSA9 | **1.6357** | 351 |

**PROMOTE** — composes with MuonEq-R.

### Exp 6: Hybrid Conv-Attention
**Script**: `experiments/linear_mixer/train_gpt.py`, `experiments/hybrid_mixer_xsa/train_gpt.py`

| Config | BPB | ms/step |
|--------|-----|---------|
| 2 conv + 7 attn | 1.6360 | **319** (fastest) |
| 2 conv + 7 attn + XSA | 1.6407 | 332 |

**SALVAGE** — fast but XSA doesn't compose on fewer attention layers.

---

## Round 2: Stacking Improvements

### Exp 7: MuonEq-R + LeakyReLU²
**Script**: `experiments/muoneqr_leakyrelu2/train_gpt.py`
**Result**: **1.6245** BPB, 333ms/step. **-0.013 vs MuonEq-R alone. PROMOTE.**

### Exp 8: MuonEq-R + SmearGate
**Script**: `experiments/muoneqr_smeargate/train_gpt.py`
**Result**: 1.6566 BPB. **Worse** alone. Context-dependent — helps in kitchen sink.

### Exp 9: Kitchen Sink 3x (MuonEq-R + XSA9 + LeakyReLU² + SmearGate + 3x MLP)
**Script**: `experiments/kitchen_sink/train_gpt.py`

| Seed | BPB | ms/step | Params |
|------|-----|---------|--------|
| 42 | **1.6159** | 385 | 21.8M |
| 1337 | **1.6143** | 386 | 21.8M |

**PROMOTE** — best combined stack on baseline architecture.

### Exp 10: Kitchen Sink 2x (with SmearGate)
**Result**: 1.6356 BPB. SmearGate needs 3x MLP to be useful.

### Exp 11: Dynamic Depth Gating
**Script**: `experiments/dynamic_depth/train_gpt.py`
**Result**: **1.6325** BPB, 353ms/step. -0.005 vs MuonEq-R, but +20ms overhead.
**SALVAGE** — promising but overhead needs reduction.

### Exp 12: MuonEq-R on SOTA Stack
**Script**: `experiments/sota_muoneqr/train_gpt.py`
**Result**: **1.7401 pre-quant** (GPTQ failed). -0.189 vs SOTA baseline (1.9295). Massive effect.
**PROMOTE** — highest priority for full run.

---

## Round 3: BigramHash + Validation

### Exp 13: Kitchen Sink + BigramHash 2048
**Script**: `experiments/kitchen_sink_bigram/train_gpt.py`
**Result**: **1.6085** BPB, 387ms/step, 22.1M params. **PROMOTE.**

### Exp 14: Kitchen Sink + BigramHash + Muon WD=0.04
**Result**: **1.6077** BPB. Marginal additional improvement.

### Exp 15: Kitchen Sink + BigramHash + 11 Layers
**Result**: 1.7120 BPB, **989ms/step**. **KILL** — 11L without banking is too slow.

### Exp 16: Kitchen Sink + BigramHash @ 500 steps
**Result**: **1.4200** BPB. Strong validation of the stack at longer training.

### Exp 17: MuonEq-R @ 500 steps
**Result**: 1.4687 BPB. Improvement holds: -0.015 vs baseline (1.4834).

### Exp 18: SOTA + MuonEq-R @ 600s (running)
Full-budget run in progress.

### Exp 19: Kitchen Sink + BigramHash @ 600s (running)
Full-budget run in progress.

---

## Summary Statistics

- Total experiments: 19 (18 complete, 2 running)
- **Promoted**: 5 ideas (MuonEq-R, LeakyReLU², XSA, Kitchen Sink, BigramHash)
- **Killed**: 4 ideas (asymmetric split, MTP, 11L without banking, SmearGate alone)
- **Salvaged**: 2 ideas (conv mixer, dynamic depth)
- **Best 200-step result**: 1.6077 BPB (Kitchen Sink + BigramHash + WD)
- **Best 500-step result**: 1.4200 BPB (Kitchen Sink + BigramHash)
- **Single most impactful change**: MuonEq-R (-0.024 avg, 3 lines, zero overhead)

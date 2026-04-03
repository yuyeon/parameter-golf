# Experiment Log

*Chronological record of all experiments*

## 2026-04-03: Session Start

### Environment
- GPU: NVIDIA H100 80GB HBM3
- CUDA: 12.8, Driver 580.126.09
- PyTorch: 2.11.0+cu128
- Flash Attention 3: 3.0.0

### Orientation Complete
- Read full repo, README, leaderboard, top 10 submissions, 30+ PRs
- Identified current SOTA: 1.1147 BPB (merged), 1.0924 BPB (unmerged PR #1279)
- Built landscape_map.md with 15 ideas

---

## Control Baselines (200 steps)

### CTRL-1: Naive Baseline
- **Script**: `train_gpt.py` (unmodified)
- **Config**: 9L/512d/8H/4KV, ReLU², 2x MLP, seq_len=1024
- **Seed 42**: val_bpb 1.6505 pre-quant, **1.6541 post-quant**, 330ms/step
- **Seed 1337**: val_bpb 1.6568 pre-quant, **1.6596 post-quant**, 330ms/step
- **VRAM**: 10,940 MiB
- **Artifact**: 10.9 MB

### CTRL-2: SOTA Stack (2026-03-25)
- **Script**: `records/track_10min_16mb/2026-03-25.../train_gpt.py`
- **Config**: 11L/512d/8H/4KV, LeakyReLU², 3x MLP, BigramHash 3072, XSA-all, banking
- **Seed 42**: val_bpb 1.9295 pre-quant, GPTQ failed (Cholesky on undertrained model)
- **Step time**: 669 ms/step (2x baseline)
- **Note**: GPTQ requires longer training for meaningful calibration

### CTRL-3: Depth Recurrence
- **Script**: `records/track_non_record_16mb/2026-03-21.../train_gpt.py`
- **Config**: 3 unique blocks, 9 effective depth, LoRA rank 4
- **Seed 42**: val_bpb 1.8114 pre-quant, **1.8151 post-quant**, 546ms/step
- **Artifact**: 4.3 MB (massive headroom — 11.7 MB unused)
- **Note**: Step time degrades from ~380ms to ~546ms over training (torch.compile issue with cycled blocks)

---

## Experiment 1: Asymmetric U-Net Split

### Exp 1a: Various encoder/decoder splits
- **Script**: `experiments/asymmetric_split/train_gpt.py`
- **Diff**: Single env var `NUM_ENCODER_LAYERS` controls split
- **Novelty claim**: Standard 50/50 split is cargo-culted from vision. PR #1275 found decoder-heavy splits help.

| Split | val_bpb | post-quant | ms/step | Delta |
|-------|---------|------------|---------|-------|
| 4/5 (default) | 1.6505 | 1.6541 | 330 | — |
| 3/6 | 1.6517 | 1.6540 | 329 | -0.0001 |
| 2/7 | 1.6520 | 1.6543 | 328 | +0.0002 |
| 1/8 | 1.6726 | 1.6824 | 571 | +0.028 |

**Verdict: KILL** — No effect at 9 layers. 1/8 split is degenerate (step time doubles). PR #1275's result may only apply at higher depth.

---

## Experiment 2: Multi-Token Prediction (MTP)

### Exp 2a-c: Various MTP configs
- **Script**: `experiments/mtp/train_gpt.py`
- **Diff**: Added `mtp_head` CastedLinear at mid-layer, predicts token+k during training, discarded at eval
- **Novelty claim**: MTP (DeepSeek-V3 style) not attempted in parameter-golf

| Config | val_bpb | post-quant | ms/step | Delta |
|--------|---------|------------|---------|-------|
| k=2 w=0.15 | 1.6490 | 1.6514 | 339 | -0.003 |
| k=2 w=0.30 | 1.6504 | 1.6516 | 340 | -0.003 |
| k=3 w=0.15 | 1.6488 | 1.6512 | 339 | -0.003 |

**Verdict: HOLD (alone), KILL (with MuonEq-R)** — Small consistent improvement alone, but interferes with MuonEq-R.

---

## Experiment 3: MuonEq-R (Row-Normalized Gradient)

- **Script**: `experiments/muoneqr/train_gpt.py`
- **Diff**: 3 lines in Muon.step() — `row_norms = g.norm(dim=-1, keepdim=True).clamp(min=1e-8); g = g / row_norms` before Newton-Schulz
- **Novelty claim**: MuonEq-R from PR #1279, isolated on baseline

| Seed | Baseline post-q | MuonEq-R post-q | Delta |
|------|-----------------|-----------------|-------|
| 42 | 1.6541 | **1.6376** | **-0.0165** |
| 1337 | 1.6596 | **1.6287** | **-0.0309** |

**Verdict: PROMOTE** — Largest single improvement found. Zero overhead. Robust across seeds.

---

## Experiment 4: MuonEq-R + MTP Combined

- **Script**: `experiments/combined/train_gpt_muoneqr_mtp.py`

| Seed | MuonEq-R only | MuonEq-R + MTP | Delta |
|------|---------------|----------------|-------|
| 42 | 1.6376 | 1.6409 | +0.003 (worse) |
| 1337 | 1.6287 | 1.6442 | +0.016 (worse) |

**Verdict: KILL MTP** — Auxiliary loss interferes with MuonEq-R's gradient dynamics.

---

## Experiment 5: MuonEq-R + XSA

- **Script**: `experiments/muoneqr_xsa/train_gpt.py`
- **Diff**: Added XSA (subtract self-value projection) to attention layers

| Config | post-quant BPB | ms/step | Delta vs MuonEq-R |
|--------|---------------|---------|-------------------|
| MuonEq-R | 1.6376 | 333 | — |
| + XSA last 4 | 1.6368 | 341 | -0.0008 |
| + XSA all 9 | **1.6357** | 351 | **-0.0019** |

**Verdict: PROMOTE** — XSA composes with MuonEq-R. XSA-all > XSA-4.

---

## Experiment 6: Hybrid Conv-Attention + MuonEq-R

- **Script**: `experiments/linear_mixer/train_gpt.py` (conv only), `experiments/hybrid_mixer_xsa/train_gpt.py` (conv + XSA)
- **Diff**: CausalConvMixer replaces first N attention layers with depthwise causal conv + gate

| Config | post-quant BPB | ms/step | Delta vs MuonEq-R |
|--------|---------------|---------|-------------------|
| MuonEq-R | 1.6376 | 333 | — |
| + 2 Conv | 1.6360 | **319** | -0.0016, **14ms faster** |
| + 2 Conv + XSA7 | 1.6407 | 332 | +0.003 |

**Verdict: SALVAGE** — Conv layers are faster but adding XSA to the remaining attention layers doesn't compose well (XSA-on-fewer-layers loses the benefit). Worth trying 1 conv + 8 attn + XSA-all.

---

## Pending: Longer MuonEq-R Run (500 steps)
Running in background. Will validate that the improvement holds at higher step counts.

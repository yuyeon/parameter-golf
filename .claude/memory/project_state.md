---
name: project_state
description: Current state of parameter-golf proxy harness and search as of 2026-03-24
type: project
---

## Frozen proxy harness (OPERATIONALLY READY)
- **proxy_train**: dispersed_5sh_seed42 (shards [0,1,4,6,9]) — proxy_data/proxy_train.json
- **Screening budget**: 32M matched tokens (976 steps × 32768 batch)
- **proxy_val_tune**: 2000 seqs, mixed strategy, fp=0cd8500a7500de94 — proxy_data/proxy_val_tune.json
- **proxy_val_audit**: 2000 seqs, disjoint, fp=61917f1d3687c31c — proxy_data/proxy_val_audit.json
- **proxy_val_long**: 500 seqs, top bytes, fp=1311cb667c9091d0 — proxy_data/proxy_val_long.json
- **Backup train**: odd_5sh (shards [1,3,5,7,9])

## Screening policy
- 32M matched tokens primary screening
- Record: train-proxy, eval-tune, eval-audit (for promoted candidates)
- **Proxy-fragile**: cross-family comparisons within ~0.02 BPB gap. Do NOT trust as final.
- Require confirmation stage for proxy-fragile finalists
- Tune vs audit agreement: Spearman=+1.000 (verified stable)

## Proxy-fidelity limitation
- SmearGate/MixedQuant inversion persists at 32M (gap −0.013 train-proxy)
- Scaling study: correct at 8M, inverted at 16M+, gap halves 16M→32M but doesn't cross
- Cross-family comparisons within 0.02 BPB are unreliable at proxy scale

## Architecture search — STARTING NOW
- Prioritize within-family improvements first
- Track family labels explicitly
- Flag cross-family near-ties automatically

## Known anchor baselines at 32M on dispersed_5sh_seed42
- NaiveBaseline (ref 1.2244): train=1.6801, eval-tune=4.0832
- Baseline10L (ref ~1.20): train=1.6770, eval-tune=missing (env override arch)
- SmearGate3x (ref 1.1458): train=1.6733, eval-tune=4.0620
- MixedQuant (ref 1.1630): train=1.6604, eval-tune=4.0249

## Environment
- GPU: NVIDIA A40 (44.4 GB), micromamba NOT available, use plain python
- 10 train shards + 1 val shard, 219 tests passing
- Key scripts: _eval_proxy_subprocess.py (submission-native), sweep_train_subsets.py
- Timeout for 32M: use 1200s sequential (plenty for 976 steps + eval-proxy)

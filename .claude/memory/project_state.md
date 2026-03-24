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
- Train-proxy + eval-proxy (bits-per-token, NOT BPB) recorded for each candidate
- **Proxy-fragile**: cross-family comparisons within ~0.02 BPB gap. Require confirmation stage.
- Proxy is good at within-family relative ordering, poor at cross-family (Spearman +0.08 across 12 models)
- Submissions winning via post-training techniques (QAT, int6, sliding eval) rank poorly in proxy

## Proxy-fidelity limitation
- SmearGate/MixedQuant inversion: correct at 8M, inverted at 16M+, gap halves 16M→32M
- Budget-limited inversion, not framework bug
- Scaling study data in artifacts/scaling_study/results_v3.json

## Architecture search completed (2026-03-24)
- 12 of 13 submissions screened at 32M tokens on dispersed_5sh_seed42
- Results in artifacts/arch_search/search_results.json
- 1 failure: Seq2048_FP16Emb_TunedLR (0 steps, likely seq_len conflict)
- 1 missing: 10L_Int5MLP_MuonWD04_SWA50 (no ref BPB, wasn't in run list)
- Top by train-proxy: TrainingOptSeq4096 (1.6567), LongContextSeq2048 (1.6586), MixedQuant (1.6603)
- Overall ranking fidelity: Spearman +0.077 (poor for cross-family)

## Upstream sync (2026-03-24)
- Merged 19 upstream commits from openai/parameter-golf
- 5 new submissions added, including new SOTA: LeakyReLU_LegalTTT_ParallelMuon (1.1183 BPB)
- New submissions NOT yet screened through proxy: should be next task
- New submissions in records/track_10min_16mb/:
  - 2026-03-20_11L_EfficientPartialXSA_FA3_SWA120
  - 2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271
  - 2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248
  - 2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233
  - 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon

## Anchor baselines at 32M on dispersed_5sh_seed42
| Model | Train-proxy | Eval-tune | Ref BPB | VRAM |
|-------|------------|-----------|---------|------|
| NaiveBaseline | 1.6843 | 4.0888 | 1.2244 | 0.878 GB |
| Baseline10L | 1.6770 | — | ~1.20 | 0.971 GB |
| SmearGate3x | 1.6737 | 4.0605 | 1.1458 | 1.052 GB |
| MixedQuant | 1.6603 | 4.0278 | 1.1630 | 1.037 GB |

## Git state
- Latest commit: merged upstream + our proxy harness (pushed to origin/main)
- All framework code, scripts, tests committed
- artifacts/, logs/, proxy_data/ are gitignored (large files / generated data)

## Environment
- GPU: NVIDIA A40 (44.4 GB), micromamba NOT available, use plain python
- 10 train shards + 1 val shard in data/datasets/fineweb10B_sp1024/
- 219 tests passing
- Key scripts: _eval_proxy_subprocess.py, sweep_train_subsets.py, _run_arch_search.sh

## Bugs found and fixed (complete list)
1. micromamba not found → auto-fallback to plain python
2. EVAL_SEQ_LEN mismatch → sweep forces EVAL_SEQ_LEN=TRAIN_SEQ_LEN
3. ExperimentSpec.from_dict double-pop → fixed
4. SubsetManifest.fingerprint ignored shard_ids → fixed
5. Broken test assertion (assert a != b or True) → fixed
6. NaiveBaseline/LowerLR identical scripts → use architecturally-different anchors
7. run_summary not written on timeout → finally block always writes
8. VRAM not logged on timeout → vram_status.txt + measured/unknown tracking
9. train-proxy vs eval-proxy conflated → separated as distinct metrics
10. 10L_SW confounds architecture with eval mode → split into Baseline10L
11. eval-proxy build_model() fails for custom arch → subprocess-based eval
12. eval-proxy metric mislabeled as BPB → renamed to bits_per_token
13. Scaling study timeout too short → budget-proportional timeout formula
14. Checkpoint lost on timeout → eager copy on "Serialized model" log line
15. vram_guard.py used total_mem instead of total_memory → fixed

---
name: project_state
description: Current state of the parameter-golf proxy framework project as of 2026-03-22
type: project
---

Fork of openai/parameter-golf. Upstream remote configured.
**Why:** User is participating in the Parameter Golf competition (ends 2026-04-30).
**How to apply:** Framework is for screening model ideas locally before committing to expensive H100 runs.

Proxy framework implemented and committed (commit 3154f3c) but not yet pushed.
- Git push failed due to SSH key issue — user needs to run `gh auth login`
- 26 files, 6367 lines added across proxy_framework/, scripts/, configs/, docs/, tests/, tools/
- 174 tests all passing (140 original + 34 parallel module tests)
- All 80 training shards + val shard downloaded (16GB in data/datasets/fineweb10B_sp1024/)

**Parallel screening added (2026-03-22):**
- User now has access to an A40 (48GB VRAM) in addition to RTX 3080
- Added proxy_framework/parallel.py, scripts/run_parallel_screen.py, tests/test_parallel.py
- A40 auto-detects 4 parallel workers (each capped at 10GB to match 3080 constraints)
- Memory isolation via torch.cuda.set_per_process_memory_fraction() wrapper
- Artifact collisions avoided by unique temp working directories per worker
- Supports: submission discovery, JSON manifests, multi-seed, dry-run, leaderboard comparison

Key VRAM measurements on RTX 3080 12GB (uncompiled baseline model):
- 65K batch: 13.7GB peak (CRASHES the PC)
- 32K batch: 6.9GB peak (safe, default)
- 16K batch: 3.5GB peak
- 8K batch: 1.8GB peak

Baseline quick-eval result: 863 steps in 5 min, post-quant val_bpb = 1.5333 (vs leaderboard 1.2244).
10L_MixedPrecision quick-eval: 367 steps, post-quant val_bpb = 1.7145 (vs leaderboard 1.2147).
Ranking was inverted for this pair — needs more data points to validate framework.

Next steps: push to GitHub, run parallel screening on A40 across all submissions, measure rank correlation.

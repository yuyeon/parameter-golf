# CLAUDE.md

## Project overview

Fork of [openai/parameter-golf](https://github.com/openai/parameter-golf) — a competition to train the best language model fitting in a 16MB artifact, trained in under 10 minutes on 8×H100 GPUs. Evaluated by bits-per-byte (BPB) on the FineWeb validation set.

We added a **local proxy training/evaluation framework** for screening model ideas cheaply on an RTX 3080 12GB before committing to expensive H100 runs. Supports **parallel screening** on larger GPUs (A40/A100) to test many ideas concurrently.

## Repository structure

```
train_gpt.py              # Official baseline training script
data/                     # Dataset shards + tokenizer (fineweb10B_sp1024)
records/                  # Competition submissions with training scripts + logs

# Our additions:
proxy_framework/          # Core framework package
  budget.py               #   Budget modes (tokens/steps/wallclock), RunSummary, matched-budget comparison
  vram_guard.py           #   VRAM enforcement (10GB cap), GPU detection, parallel worker calculation
  parallel.py             #   Parallel experiment execution (multi-worker on large GPUs)
  train_subset_search.py  #   Train subset candidate generation + shard-dir isolation
  provisional_val.py      #   Provisional validation lens for train-subset comparison
  finalist_selection.py   #   Rank candidates by fidelity metrics, select finalists
  metrics.py              #   Ranking fidelity (Spearman, Kendall, pairwise, top-k, bootstrap CI)
  data_utils.py           #   Shard I/O, sequence enumeration, subset manifests
  config.py               #   Dataclass configs + YAML/JSON
  model_utils.py          #   Model import, per-sequence eval via F.cross_entropy patching

scripts/                  # Runnable scripts
  generate_train_candidates.py # Stage 1: generate train subset search space
  sweep_train_subsets.py  #   Stage 2-3: sweep candidates on anchors, select finalists
  run_parallel_screen.py  #   Parallel screening on large GPUs (A40/A100)
  profile_full_val.py     #   Profile anchor models on full val set (per-sequence JSONL)
  build_proxy_val.py      #   Stage 4: build final proxy_val (accepts --train-finalists)
  build_proxy_train.py    #   Define proxy train subsets (shard selection)
  run_local_screen.py     #   Quick screening runs (~10 min, VRAM-safe)
  run_local_audit.py      #   Higher-fidelity audit runs (multi-seed, full-val)
  analyze_proxy_rankings.py # Compare proxy rankings vs leaderboard

artifacts/                # Generated artifacts (gitignored)
  train_subsets/          #   Candidate train subset manifests
  train_subset_sweep/     #   Sweep results + selection/finalists.json

tools/                    # Earlier quick-eval utilities
  quick_eval.py           #   First-gen screening runner
  eval_only.py            #   Standalone quantization + eval from checkpoint

configs/                  # Example YAML configs
tests/                    # pytest tests (140 tests)
docs/                     # Framework documentation
```

## Key constraints

- **VRAM cap: 10.0 GB strict per experiment** — enforced by VRAMGuard, hard fail if exceeded
- **Default batch: 32K tokens** — measured at 6.9GB peak on RTX 3080 (65K = 13.7GB, crashes)
- **Default budget mode: matched tokens** — fairer than wall-clock for comparing different architectures
- **Parallel workers**: auto-detected from GPU VRAM (A40 48GB → 4 workers, A100 80GB → 6 workers)

## Common commands

```bash
# Environment
micromamba run -n parameter-golf python <script>

# Run tests
micromamba run -n parameter-golf python -m pytest tests/ -v

# --- Staged train-subset selection (Stages 0-3) ---

# Stage 1: Generate candidate train subsets
python scripts/generate_train_candidates.py \
    --output-dir artifacts/train_subsets

# Stage 2-3: Sweep candidates on anchor models + select finalists
python scripts/sweep_train_subsets.py \
    --candidates-dir artifacts/train_subsets \
    --records-dir records/track_10min_16mb \
    --output-dir artifacts/train_subset_sweep

# --- Stage 4: Build final proxy val subsets ---

# (Requires profiled anchor models first)
python scripts/build_proxy_val.py \
    --profile-dir profiling_results/ \
    --output-dir proxy_data/ \
    --train-finalists artifacts/train_subset_sweep/selection/finalists.json

# --- Stage 5: Screen and audit candidates ---

# Screen a single submission (matched-token, 16M tokens)
python scripts/run_local_screen.py \
    --script records/track_10min_16mb/<submission>/train_gpt.py \
    --budget-mode tokens --budget-value 16000000

# Parallel screening of ALL submissions (auto-detects GPU, e.g. 4x on A40)
python scripts/run_parallel_screen.py \
    --records-dir records/track_10min_16mb \
    --output-dir parallel_results/batch_001

# Analyze proxy rankings vs leaderboard
python scripts/analyze_proxy_rankings.py \
    --results-dir proxy_results/ --calibrate
```

## Environment

- Python env: `micromamba run -n parameter-golf`
- Primary GPU: RTX 3080 12GB (target for individual experiments)
- Parallel GPU: A40 48GB or similar (for running multiple experiments concurrently)
- Data: 80 train shards + 1 val shard in data/datasets/fineweb10B_sp1024/

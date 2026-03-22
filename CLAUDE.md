# CLAUDE.md

## Project overview

Fork of [openai/parameter-golf](https://github.com/openai/parameter-golf) — a competition to train the best language model fitting in a 16MB artifact, trained in under 10 minutes on 8×H100 GPUs. Evaluated by bits-per-byte (BPB) on the FineWeb validation set.

We added a **local proxy training/evaluation framework** for screening model ideas cheaply on an RTX 3080 12GB before committing to expensive H100 runs.

## Repository structure

```
train_gpt.py              # Official baseline training script
data/                     # Dataset shards + tokenizer (fineweb10B_sp1024)
records/                  # Competition submissions with training scripts + logs

# Our additions:
proxy_framework/          # Core framework package
  budget.py               #   Budget modes (tokens/steps/wallclock), RunSummary, matched-budget comparison
  vram_guard.py           #   VRAM enforcement (10GB cap, background monitor)
  metrics.py              #   Ranking fidelity (Spearman, Kendall, pairwise, top-k, bootstrap CI)
  data_utils.py           #   Shard I/O, sequence enumeration, subset manifests
  config.py               #   Dataclass configs + YAML/JSON
  model_utils.py          #   Model import, per-sequence eval via F.cross_entropy patching

scripts/                  # Runnable scripts
  profile_full_val.py     #   Profile anchor models on full val set (per-sequence JSONL)
  build_proxy_val.py      #   Build proxy_val_tune/audit/long subsets from profiles
  build_proxy_train.py    #   Define proxy train subsets (shard selection)
  run_local_screen.py     #   Quick screening runs (~10 min, VRAM-safe)
  run_local_audit.py      #   Higher-fidelity audit runs (multi-seed, full-val)
  analyze_proxy_rankings.py # Compare proxy rankings vs leaderboard

tools/                    # Earlier quick-eval utilities
  quick_eval.py           #   First-gen screening runner
  eval_only.py            #   Standalone quantization + eval from checkpoint

configs/                  # Example YAML configs
tests/                    # pytest tests (140 tests)
docs/                     # Framework documentation
```

## Key constraints

- **VRAM cap: 10.0 GB strict** — enforced by VRAMGuard, hard fail if exceeded
- **Default batch: 32K tokens** — measured at 6.9GB peak on RTX 3080 (65K = 13.7GB, crashes)
- **Default budget mode: matched tokens** — fairer than wall-clock for comparing different architectures

## Common commands

```bash
# Environment
micromamba run -n parameter-golf python <script>

# Run tests
micromamba run -n parameter-golf python -m pytest tests/ -v

# Screen a submission (matched-token, 16M tokens)
micromamba run -n parameter-golf python scripts/run_local_screen.py \
    --script records/track_10min_16mb/<submission>/train_gpt.py \
    --budget-mode tokens --budget-value 16000000

# Analyze proxy rankings vs leaderboard
micromamba run -n parameter-golf python scripts/analyze_proxy_rankings.py \
    --results-dir proxy_results/ --calibrate
```

## Environment

- Python env: `micromamba run -n parameter-golf`
- GPU: RTX 3080 12GB (WSL2)
- Data: 80 train shards + 1 val shard in data/datasets/fineweb10B_sp1024/

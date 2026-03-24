# CLAUDE.md

## Project overview

Fork of [openai/parameter-golf](https://github.com/openai/parameter-golf) — a competition to train the best language model fitting in a 16MB artifact, trained in under 10 minutes on 8×H100 GPUs. Evaluated by bits-per-byte (BPB) on the FineWeb validation set.

We built a **local proxy training/evaluation framework** for screening model ideas cheaply before committing to expensive H100 runs. The proxy harness is now **operationally frozen** and used for architecture/hyperparameter search.

## Frozen proxy harness

```
proxy_train:      dispersed_5sh_seed42 (shards [0,1,4,6,9])  → proxy_data/proxy_train.json
proxy_val_tune:   2000 seqs, mixed strategy                   → proxy_data/proxy_val_tune.json
proxy_val_audit:  2000 seqs, disjoint from tune                → proxy_data/proxy_val_audit.json
proxy_val_long:   500 seqs, longest byte counts                → proxy_data/proxy_val_long.json
screening budget: 32M matched tokens (976 steps × 32768 batch)
VRAM cap:         10.0 GB strict per experiment
```

## Screening policy

- **Primary metric**: train-proxy (pre_quant_val_bpb from the training script's built-in eval)
- **Secondary metric**: eval-proxy (bits-per-token on proxy_val_tune via submission-native subprocess)
- **Proxy-fragile**: cross-family comparisons within ~0.02 BPB gap. Do NOT trust as final.
- **Confirmation required**: proxy-fragile finalists need higher-budget or multi-seed confirmation
- Proxy is good at **within-family relative ordering** but poor at cross-family ranking (Spearman +0.08 across 12 models)
- Submissions that win via post-training techniques (QAT, int6, sliding eval) rank poorly in the proxy because those techniques are invisible during training

## Proxy-fidelity limitations

The SmearGate/MixedQuant inversion (gap 0.013 BPB, leaderboard says SmearGate should win) is a known proxy-fidelity limitation:
- Correct at 8M tokens, inverted at 16M+, gap halves 16M→32M but doesn't cross
- This is a budget-limited inversion, not a framework bug
- Cross-family comparisons within ~0.02 BPB of the leaderboard are unreliable at proxy scale

## Repository structure

```
train_gpt.py              # Official baseline training script
data/                     # Dataset shards + tokenizer (fineweb10B_sp1024)
records/                  # Competition submissions with training scripts + logs

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
  generate_train_candidates.py # Generate train subset search space
  sweep_train_subsets.py  #   Sweep candidates on anchors, select finalists
  _eval_proxy_subprocess.py #  Submission-native eval (works for custom architectures)
  profile_full_val.py     #   Profile anchor models on full val set (per-sequence JSONL)
  build_proxy_val.py      #   Build final proxy_val (accepts --train-finalists)
  build_proxy_train.py    #   Define proxy train subsets (shard selection)
  run_parallel_screen.py  #   Parallel screening on large GPUs (A40/A100)
  run_local_screen.py     #   Quick screening runs (VRAM-safe)
  run_local_audit.py      #   Higher-fidelity audit runs (multi-seed, full-val)
  analyze_proxy_rankings.py # Compare proxy rankings vs leaderboard
  _run_arch_search.sh     #   Architecture search sweep (nohup-safe)
  _run_scaling_study.sh   #   Budget-scaling study (nohup-safe)

proxy_data/               # Frozen proxy manifests
  proxy_train.json        #   Frozen train subset (dispersed_5sh_seed42)
  proxy_val_tune.json     #   2000-seq screening subset
  proxy_val_audit.json    #   2000-seq confirmation subset (disjoint)
  proxy_val_long.json     #   500-seq long-context subset

artifacts/                # Generated artifacts (gitignored)
  train_subsets/          #   21 candidate train subset manifests
  phase2_5_sweep/         #   Train-subset confirmation sweep results
  scaling_study/          #   SmearGate vs MixedQuant budget-scaling data
  arch_search/            #   Full architecture search results
  profiling/              #   Per-sequence profiling JSONL
  profiling_32m/          #   32M checkpoint profiling data (used for final proxy_val)

tools/                    # Earlier quick-eval utilities
configs/                  # Example YAML configs
tests/                    # pytest tests (219 tests)
docs/                     # Framework documentation
```

## Common commands

```bash
# Run tests
python -m pytest tests/ -v

# --- Screen a submission at 32M matched tokens ---
python scripts/sweep_train_subsets.py \
    --candidates-dir artifacts/train_subsets \
    --records-dir records/track_10min_16mb \
    --candidate-ids dispersed_5sh_seed42 \
    --anchor-scripts records/track_10min_16mb/<submission>/train_gpt.py \
    --budget-mode tokens --budget-value 32000000 \
    --proxy-val-manifest proxy_data/proxy_val_tune.json \
    --timeout 1200 --max-workers 1 \
    --output-dir artifacts/screen_results

# --- Eval-proxy on a checkpoint (submission-native, works for all architectures) ---
python scripts/_eval_proxy_subprocess.py \
    --script records/track_10min_16mb/<submission>/train_gpt.py \
    --checkpoint <path/to/final_model.pt> \
    --manifest proxy_data/proxy_val_tune.json \
    --output /tmp/eval_result.json

# --- Full architecture search (all submissions, nohup-safe) ---
nohup bash scripts/_run_arch_search.sh > logs/arch_search_nohup.log 2>&1 &

# --- Generate train subset candidates (already done, 21 candidates) ---
python scripts/generate_train_candidates.py --output-dir artifacts/train_subsets

# --- Build proxy_val from profiling data ---
python scripts/build_proxy_val.py \
    --profile-dir artifacts/profiling_32m \
    --output-dir proxy_data \
    --train-finalists artifacts/phase2_5_sweep/selection/phase2_5_report.json
```

## Key operational notes

- **micromamba is NOT available** on the current A40 instance. Scripts auto-fallback to plain `python`.
- **Timeout for 32M runs**: use 1200s for sequential single-worker runs. With parallel workers, torch.compile contention requires longer timeouts.
- **Submissions with sliding-window eval** have slow post-training phases (30-60 min). The sweep uses eager checkpoint copying + timeout to avoid blocking.
- **Eval-proxy metric is bits-per-token** (loss/log2), NOT bits-per-byte. Ordering is preserved but absolute values differ from train-proxy BPB.
- **Most submissions share identical training code**. Only NaiveBaseline and LowerLR are byte-for-byte identical, but many others only differ in post-training (quantization, sliding eval) which is invisible to the proxy.

## Anchor baselines at 32M tokens on dispersed_5sh_seed42

| Model | Train-proxy | Eval-tune | Ref BPB | VRAM |
|-------|------------|-----------|---------|------|
| NaiveBaseline | 1.6843 | 4.0888 | 1.2244 | 0.878 GB |
| Baseline10L | 1.6770 | — | ~1.20 | 0.971 GB |
| SmearGate3x | 1.6737 | 4.0605 | 1.1458 | 1.052 GB |
| MixedQuant | 1.6603 | 4.0278 | 1.1630 | 1.037 GB |

## Environment

- Python env: plain `python` (micromamba not available on A40 instance)
- Primary GPU: RTX 3080 12GB (target hardware for individual experiments)
- Parallel GPU: A40 48GB (current development/search machine)
- Data: 10 train shards + 1 val shard in data/datasets/fineweb10B_sp1024/

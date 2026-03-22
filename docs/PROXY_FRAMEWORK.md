# Proxy Framework for Parameter Golf

## Goal

Preserve model rankings under cheap local runs on a consumer GPU (RTX 3080 12GB / 10 GB usable VRAM) so that ideas can be screened in minutes rather than burning H100 hours. A proxy run that takes 5 minutes should, ideally, produce the same top-k ordering of candidate submissions as the full 10-minute 8xH100 evaluation, despite training on fewer tokens, evaluating on fewer sequences, and fitting inside a fraction of the compute budget.

## Why We Need Both Proxy-Train and Proxy-Val

**Proxy-train** controls how the model is trained during the screening run. We use a subset of training shards, shorter wall-clock time, and tighter VRAM constraints. The question proxy-train answers is: "if I train this architecture/hyperparameter variant cheaply, does the relative ordering of its loss align with the ordering I would see after full training?"

**Proxy-val** controls how the trained model is evaluated. We use a subset of the official validation set to compute per-sequence BPB. The question proxy-val answers is: "if I evaluate on fewer validation sequences, does the aggregate score preserve the same ranking as evaluating on the full set?"

These two proxies are orthogonal. A good proxy-train setup with a bad proxy-val (or vice versa) will produce unreliable rankings. Both must be validated independently and jointly.

## Why proxy_val_tune and proxy_val_audit Are Disjoint

We split the proxy validation set into two non-overlapping subsets:

- **proxy_val_tune** (default 2000 sequences): Used repeatedly during rapid iteration. Every candidate idea is evaluated against this set during the screening phase.
- **proxy_val_audit** (default 2000 disjoint sequences): Held out and used only once per candidate, after the screening phase selects the top-N most promising ideas.

The disjoint split prevents overfitting the proxy. If we used the same sequences for both screening and confirmation, we could inadvertently optimize for sequences that happen to favor certain architectures. The audit set acts as a fresh, unbiased check.

This is analogous to train/val splitting in ML: tune on one set, confirm on another.

## How proxy_val_long Rescues Long-Context / Sliding-Eval Ideas

Some competition submissions use longer sequence lengths (2048+) and sliding-window evaluation at inference time. These ideas systematically improve BPB on the full validation set but may not show any benefit when evaluated at `seq_len=1024` on random sequences.

**proxy_val_long** (default 500 sequences at `seq_len=2048`) is a third validation subset specifically designed to capture the signal from long-context modifications. When sliding-window evaluation is enabled (`sliding_window=true`, `sliding_stride=64`), the proxy still evaluates within the 10 GB VRAM cap by reducing the batch size (`val_batch_seqs=4`).

Without this subset, the proxy framework would be blind to an entire class of improvements.

## Why Matched-Budget Comparison Matters

Comparing models after "5 minutes of wall-clock on a 3080" is **misleading**.  Different architectures have different per-step times: a 10-layer model with 3x MLP expansion might process 300 steps in 5 minutes, while the 9-layer baseline processes 1000 steps.  The baseline "wins" not because it is architecturally better, but because it saw 3x more training data.

**Matched-budget comparison** controls for this by ensuring all models are compared after processing the same amount of training compute, measured as:

| Budget mode | What it matches | When to use |
|---|---|---|
| **tokens** (recommended default) | Total training tokens processed | Fairest comparison — equalizes data exposure |
| **optimizer_steps** | Number of parameter updates | When step count matters more than data volume |
| **wallclock** | Wall-clock training time | Practical constraint; secondary comparison |

The framework supports all three modes.  The recommended workflow is:

1. **Primary**: compare under `--budget-mode tokens` at a fixed token count (e.g. 16M)
2. **Secondary**: also report `optimizer_steps` and `wallclock` for context
3. **Calibrate**: use `--calibrate` to find which mode best predicts reference rankings

### Why matched wall-clock alone is misleading

Wall-clock conflates three things:
- **Architecture speed** (how many tokens/second the model processes)
- **Data exposure** (how many tokens the model has seen)
- **Optimization quality** (how efficiently the model learns from each token)

A model that is slower per step but learns more per token will look worse under wall-clock comparison but better under matched-token comparison.  Since the competition evaluates quality (BPB) not speed, matched-token comparison is more predictive of final rankings.

## Predicted Quality at Target Budget

When you have multiple proxy runs of the same model at different budgets (e.g. 8M, 16M, 32M tokens), the framework can extrapolate to estimate quality at a larger target budget using log-linear fitting:

```
loss ≈ a × log(budget) + b
```

This is a simple, robust estimator — NOT a sophisticated scaling law.  Predictions are flagged with confidence levels:
- **high**: ≤2x extrapolation with ≥3 data points
- **moderate**: ≤4x extrapolation with ≥2 data points
- **low**: >4x extrapolation or sparse data

Raw observed values are always reported alongside predictions.  Use predictions as directional signals, not precise estimates.

## Ranking Fidelity Metrics

We judge proxy quality by how well proxy rankings agree with reference (full-compute) rankings:

| Metric | What it measures | Range |
|---|---|---|
| **Spearman rho** | Rank correlation between proxy and reference orderings | [-1, +1] |
| **Kendall tau** | Fraction of concordant vs discordant pairs, robust to outliers | [-1, +1] |
| **Pairwise accuracy** | Fraction of model pairs whose relative order is preserved | [0, 1] |
| **Top-1 agreement** | Whether proxy and reference agree on the single best model | {True, False} |
| **Top-k overlap** | Jaccard overlap of the top-k models in proxy vs reference | [0, 1] |

All metrics assume lower score = better model (as with BPB / loss). Bootstrap confidence intervals (default 1000 samples) are computed for all continuous metrics.

A proxy is considered useful when Spearman rho > 0.8 and pairwise accuracy > 0.85.

## How the VRAM Cap Is Enforced

All local runs operate under a strict peak VRAM cap (default 10.0 GB out of the physical 12 GB on an RTX 3080, leaving headroom for the OS, display, and CUDA overhead).

The enforcement mechanism has three layers:

1. **`check_vram(max_gb)`**: A point-in-time check that reads `torch.cuda.max_memory_allocated()` and raises `RuntimeError` if the peak exceeds the cap.

2. **`VRAMGuard` context manager**: Resets peak memory stats on entry, checks on exit, and provides a `.check()` method for intermediate checkpoints. Any violation raises `RuntimeError`, cleanly aborting the run.

3. **Background monitor thread (`VRAMGuard.start_monitor()`)**: A daemon thread that calls `check_vram` every N seconds (default 5). If a violation is detected asynchronously, it prints to stderr and calls `os._exit(1)` for a hard fail, preventing the process from continuing with silently corrupted results.

Additionally, `safe_batch_size()` provides a conservative estimate of the maximum micro-batch size given model VRAM, per-sequence overhead, and the VRAM cap.

## How to Reproduce Subset Definitions

All data subsets are defined deterministically:

1. **Seeds**: Every subset is generated with an explicit random seed (default 42), recorded in the `SubsetManifest`.
2. **Manifest JSONs**: Each subset definition is serialized as a JSON manifest containing the exact sequence IDs, sequence length, strategy name, seed, and a SHA-256 fingerprint.
3. **Fingerprints**: The `SubsetManifest.fingerprint` property computes a deterministic hash from `(name, sorted(seq_ids), seq_len)`, enabling dedup and cache invalidation.

To reproduce a subset:
```bash
# Load and verify
python -c "
from proxy_framework.data_utils import load_manifest
m = load_manifest('manifests/proxy_val_tune.json')
print(f'{m.name}: {m.n_seqs} seqs, fingerprint={m.fingerprint}')
"
```

To regenerate from scratch, use the same seed and strategy parameters. The manifest fingerprint should match exactly.

## Workflow: What Scripts to Run in What Order

### Step 1: Build subset manifests

Generate the proxy validation and training subsets:

```bash
# Build proxy_val_tune and proxy_val_audit manifests (disjoint)
python -m scripts.build_proxy_val \
    --config configs/proxy_val_build.yaml \
    --output-dir manifests/

# Build proxy_val_long manifest
python -m scripts.build_proxy_val_long \
    --config configs/long_context_val.yaml \
    --output-dir manifests/
```

### Step 2: Screen candidates (matched-token mode, recommended)

Run each candidate with a fixed token budget so comparisons are fair:

```bash
# Matched-token screening (default: 16M tokens)
python scripts/run_local_screen.py \
    --script records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
    --budget-mode tokens --budget-value 16000000 \
    --val-tune-manifest manifests/proxy_val_tune.json \
    --output-dir proxy_results/screen_tokens/

# Matched-step screening (500 optimizer updates)
python scripts/run_local_screen.py \
    --script records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
    --budget-mode optimizer_steps --budget-value 500 \
    --output-dir proxy_results/screen_steps/

# Wall-clock screening (5 minutes, for throughput measurement)
python scripts/run_local_screen.py \
    --script records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
    --budget-mode wallclock --budget-value 300 \
    --output-dir proxy_results/screen_wallclock/
```

Each run saves a `run_summary.json` with full budget accounting (tokens processed, steps, wall-clock, throughput, VRAM peak).

### Step 3: Audit top candidates

Re-evaluate on the disjoint audit set with a larger budget:

```bash
python scripts/run_local_audit.py \
    --script records/track_10min_16mb/BEST_CANDIDATE/train_gpt.py \
    --budget-mode tokens --budget-value 64000000 \
    --val-audit-manifest manifests/proxy_val_audit.json \
    --seeds 1337 42 7 \
    --output-dir proxy_results/audit/
```

### Step 4: Analyze ranking fidelity

Compare proxy rankings against the official leaderboard:

```bash
# Basic analysis with matched-token comparison
python scripts/analyze_proxy_rankings.py \
    --results-dir proxy_results/screen_tokens/ \
    --budget-mode tokens \
    --k 3

# Calibration: which budget mode best predicts the leaderboard?
python scripts/analyze_proxy_rankings.py \
    --results-dir proxy_results/ \
    --calibrate

# Predicted quality at target budget (extrapolate from multiple runs)
python scripts/analyze_proxy_rankings.py \
    --results-dir proxy_results/ \
    --predict --target-budget 7000000000
```

### Step 5: Iterate on the proxy itself

If ranking fidelity is low, adjust the proxy configuration (more tokens, different validation subset strategy, different budget mode) and repeat from Step 1.

## Challenge-Specific vs Generic Assumptions

### Challenge-specific

- Binary shard format (magic=20240520, header=256 int32s, body=uint16 tokens)
- BPB (bits-per-byte) as the primary evaluation metric
- Specific model architecture (GPT with CastedLinear, muP-style scaling)
- Tokenizer: SP-1024 BPE
- Competition constraint: 10 minutes on 8xH100, 16 MB checkpoint

### Generic (reusable for other ranking-preservation tasks)

- Ranking fidelity metrics (Spearman, Kendall, pairwise, top-k)
- VRAM guard and safe batch-size estimation
- Disjoint tune/audit validation split to prevent proxy overfitting
- Subset manifest system with deterministic fingerprints
- Bootstrap confidence intervals for metric uncertainty

## Design Rationale

The proxy framework design is informed by several lines of research on how small-scale decisions can transfer to larger settings:

- **DataDecide** (arXiv:2504.11393): Demonstrates that model rankings established at small scale can be predictive of rankings at larger scale, provided the evaluation protocol is carefully designed. This motivates our belief that a 5-minute proxy run on an RTX 3080 can produce rankings correlated with full H100 training.

- **DoReMi** (arXiv:2305.10429): Shows that data mixing decisions made at proxy scale (a smaller model) can transfer upward to improve larger models. This justifies our use of reduced training data (10 shards instead of all shards) and shorter training time while still expecting the relative ordering of architectural ideas to be preserved.

- **PreSelect** (arXiv:2503.00808): Finds that some training documents are more predictive of downstream quality than others, and that selecting documents carefully can improve efficiency. This motivates our "mixed" strategy for validation subset construction, where we sample across difficulty quantiles rather than uniformly at random.

- **SparseEval** (arXiv:2602.07909): Demonstrates that a carefully chosen subset of evaluation items can preserve ranking signal with far fewer evaluation examples. This directly supports our use of 2000-sequence proxy validation sets instead of the full validation corpus, and motivates future work on importance-weighted sequence selection.

## Quick Start

```bash
# Install dependencies
pip install torch numpy pyyaml

# Run the test suite
python -m pytest tests/ -v

# Analyze rankings (assuming proxy results exist)
python -m scripts.analyze_proxy_rankings \
    --results-dir proxy_results/screen_baseline/ \
    --k 3

# Load a config programmatically
python -c "
from proxy_framework.config import load_config
cfg = load_config('configs/screen_baseline.yaml')
print(f'Config: {cfg.name}')
print(f'  VRAM cap: {cfg.vram.max_gb} GB')
print(f'  Train batch: {cfg.train.train_batch_tokens} tokens')
print(f'  Seq len: {cfg.data.seq_len}')
"
```

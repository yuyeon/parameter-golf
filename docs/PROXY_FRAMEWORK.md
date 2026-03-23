# Proxy Framework for Parameter Golf

## Goal

Preserve model rankings under cheap local runs on a consumer GPU (RTX 3080 12GB / 10 GB usable VRAM) so that ideas can be screened in minutes rather than burning H100 hours. A proxy run that takes 5 minutes should, ideally, produce the same top-k ordering of candidate submissions as the full 10-minute 8xH100 evaluation, despite training on fewer tokens, evaluating on fewer sequences, and fitting inside a fraction of the compute budget.

## Staged Procedure: Train-First-But-Not-Train-In-Isolation

We do NOT build the final train subset completely independently of validation. Instead we follow a staged procedure that uses a provisional validation lens to compare train subsets, then builds the final validation subsets after narrowing the train subset candidates.

### Stage 0: Establish a provisional validation lens

Before building the final proxy validation subsets, create a TEMPORARY evaluation lens good enough to compare candidate train subsets.  Options:

1. **Full official validation** — the training script already evaluates on the full val set at end of training; we parse `val_bpb` from logs (free, highest fidelity).
2. **Simple provisional subset** — random or stratified-by-length subset of ~2000 val sequences (faster, no profiling required).

The provisional lens is NOT final.  Its purpose is to compare *relative model orderings* across different train subsets.

### Stage 1: Generate multiple candidate train subsets

Rather than committing to a single train subset, generate a search space of candidates from the official challenge train shards:

| Family       | Description                                  |
|-------------|----------------------------------------------|
| `contiguous`  | First N shards in order (simple baseline)    |
| `uniform`     | Every K-th shard for maximum spread          |
| `dispersed`   | Random shard sample (seed-controlled)        |
| `single`      | Just one shard (tests data-sensitivity)      |
| `bookend`     | First half + last half of corpus             |
| `odd` / `even`| Alternating shard indices                    |

Each candidate has a unique ID, exact shard IDs, sampling seed, and token count.  All are saved as deterministic `SubsetManifest` JSON files.

```bash
python scripts/generate_train_candidates.py \
    --data-dir data/datasets/fineweb10B_sp1024 \
    --output-dir artifacts/train_subsets
```

### Stage 2: Compare train subsets using anchor models

Choose a small pool of anchor models spanning plausible good and bad ideas.  For each candidate train subset:

1. Train all anchor models under the SAME matched-budget protocol
2. Evaluate using the provisional validation lens
3. Compare induced rankings against a reference ranking (leaderboard or higher-fidelity runs)

Ranking metrics computed for each train subset:
- Kendall tau
- Spearman rho
- Pairwise accuracy
- Top-1 agreement
- Top-k overlap

"Best train subset" means: the subset whose induced rankings most closely match the higher-fidelity reference ranking.  NOT the subset producing the lowest absolute loss or the best result for only one model.

```bash
python scripts/sweep_train_subsets.py \
    --candidates-dir artifacts/train_subsets \
    --records-dir records/track_10min_16mb \
    --output-dir artifacts/train_subset_sweep \
    --budget-mode tokens --budget-value 16000000
```

### Stage 3: Narrow train subsets to finalists

After evaluating all candidates, keep the top 1–3 that best preserve ranking signal.  Finalists are selected by a composite score of Spearman, Kendall, and pairwise accuracy.  A selection report explains why each finalist was chosen.

Output:
- `artifacts/train_subset_sweep/selection/finalists.json`
- `artifacts/train_subset_sweep/selection/report.json`

### Stage 4: Build final proxy validation subsets

ONLY AFTER train-subset finalists are chosen, build the final proxy validation subsets:

- `proxy_val_tune` — for tuning proxy parameters (2000 sequences)
- `proxy_val_audit` — disjoint from tune, unbiased eval (2000 sequences)
- `proxy_val_long` — longest byte-count sequences (500 sequences)

The final validation subsets are optimized relative to the train-subset finalists, not in isolation.

```bash
# Profile anchor models on full val set
python scripts/profile_full_val.py \
    --manifest models.json \
    --output-dir profiling_results/

# Build final val subsets with finalist context
python scripts/build_proxy_val.py \
    --profile-dir profiling_results/ \
    --output-dir proxy_data/ \
    --train-finalists artifacts/train_subset_sweep/selection/finalists.json
```

### Stage 5: Choose the final joint proxy harness

Select the final combination:
- `proxy_train` (from finalists)
- `proxy_val_tune`
- `proxy_val_audit`
- `proxy_val_long`

The selected harness should maximize rank fidelity under local budget constraints.

## Why We Need Both Proxy-Train and Proxy-Val

**Proxy-train** controls how the model is trained during the screening run. We use a subset of training shards, shorter wall-clock time, and tighter VRAM constraints. The question proxy-train answers is: "if I train this architecture/hyperparameter variant cheaply, does the relative ordering of its loss align with the ordering I would see after full training?"

**Proxy-val** controls how the trained model is evaluated. We use a subset of the official validation set to compute per-sequence BPB. The question proxy-val answers is: "if I evaluate on fewer validation sequences, does the aggregate score preserve the same ranking as evaluating on the full set?"

These two proxies are orthogonal. A good proxy-train setup with a bad proxy-val (or vice versa) will produce unreliable rankings. Both must be validated independently and jointly.

## Why proxy_val_tune and proxy_val_audit Are Disjoint

We split the proxy validation set into two non-overlapping subsets:

- **proxy_val_tune** (default 2000 sequences): Used repeatedly during rapid iteration. Every candidate idea is evaluated against this set during the screening phase.
- **proxy_val_audit** (default 2000 disjoint sequences): Held out and used only once per candidate, after the screening phase selects the top-N most promising ideas.

The disjoint split prevents overfitting the proxy. If we used the same sequences for both screening and confirmation, we could inadvertently optimize for sequences that happen to favor certain architectures. The audit set acts as a fresh, unbiased check.

## How proxy_val_long Rescues Long-Context / Sliding-Eval Ideas

Some competition submissions use longer sequence lengths (2048+) and sliding-window evaluation at inference time. These ideas systematically improve BPB on the full validation set but may not show any benefit when evaluated at `seq_len=1024` on random sequences.

**proxy_val_long** (default 500 sequences at `seq_len=2048`) is a third validation subset specifically designed to capture the signal from long-context modifications.

## Why Matched-Budget Comparison Matters

Comparing models after "5 minutes of wall-clock on a 3080" is **misleading**.  Different architectures have different per-step times.  A 10-layer model with 3x MLP expansion might process 300 steps in 5 minutes, while the 9-layer baseline processes 1000 steps.  The baseline "wins" not because it is architecturally better, but because it saw 3x more training data.

**Matched-budget comparison** controls for this by ensuring all models are compared after processing the same amount of training compute:

| Budget mode | What it matches | When to use |
|---|---|---|
| **tokens** (recommended default) | Total training tokens processed | Fairest comparison — equalizes data exposure |
| **optimizer_steps** | Number of parameter updates | When step count matters more than data volume |
| **wallclock** | Wall-clock training time | Practical constraint; secondary comparison |

## Predicted Quality at Target Budget

When you have multiple proxy runs at different budgets, the framework can extrapolate using log-linear fitting: `loss ≈ a × log(budget) + b`.  Predictions are flagged with confidence levels (high/moderate/low).

## Ranking Fidelity Metrics

| Metric | What it measures | Range |
|---|---|---|
| **Spearman rho** | Rank correlation between proxy and reference orderings | [-1, +1] |
| **Kendall tau** | Fraction of concordant vs discordant pairs | [-1, +1] |
| **Pairwise accuracy** | Fraction of model pairs whose relative order is preserved | [0, 1] |
| **Top-1 agreement** | Whether proxy and reference agree on the single best model | {True, False} |
| **Top-k overlap** | Jaccard overlap of the top-k models in proxy vs reference | [0, 1] |

A proxy is considered useful when Spearman rho > 0.8 and pairwise accuracy > 0.85.

## How the VRAM Cap Is Enforced

All local runs operate under a strict peak VRAM cap (default 10.0 GB).  Enforcement has three layers:

1. **`check_vram(max_gb)`**: Point-in-time check via `torch.cuda.max_memory_allocated()`.
2. **`VRAMGuard` context manager**: Resets peak stats on entry, checks on exit.
3. **Background monitor thread**: Daemon polling every 5s, hard-kills via `os._exit(1)` on violation.

## Parallel Execution on Large GPUs

When a larger GPU is available (A40 48 GB, A100 80 GB), the framework runs multiple experiments concurrently while keeping each individual experiment within RTX 3080 constraints.

- Auto-detects GPU and calculates safe parallelism (A40 → 4 workers)
- Per-process CUDA memory isolation via `torch.cuda.set_per_process_memory_fraction()`
- Artifact collision avoided by unique temp working directories

## How to Reproduce Subset Definitions

All data subsets are defined deterministically:

1. **Seeds**: Every subset is generated with an explicit random seed, recorded in the `SubsetManifest`.
2. **Manifest JSONs**: Each subset definition is serialized as a JSON manifest containing exact shard IDs / sequence IDs, seed, strategy, and a SHA-256 fingerprint.
3. **Fingerprints**: `SubsetManifest.fingerprint` computes a deterministic hash from `(name, sorted(seq_ids), seq_len)`.

## Workflow: What Scripts to Run in What Order

### Stage 0–3: Select train subset

```bash
# Generate candidate train subsets
python scripts/generate_train_candidates.py \
    --output-dir artifacts/train_subsets

# Sweep candidates on anchor models (parallel on A40)
python scripts/sweep_train_subsets.py \
    --candidates-dir artifacts/train_subsets \
    --records-dir records/track_10min_16mb \
    --output-dir artifacts/train_subset_sweep

# View finalists
cat artifacts/train_subset_sweep/selection/finalists.json
```

### Stage 4: Build final proxy validation subsets

```bash
# Profile anchor models on full val set (needs trained checkpoints)
python scripts/profile_full_val.py \
    --manifest models.json \
    --output-dir profiling_results/

# Build final val subsets with train-finalist context
python scripts/build_proxy_val.py \
    --profile-dir profiling_results/ \
    --output-dir proxy_data/ \
    --train-finalists artifacts/train_subset_sweep/selection/finalists.json
```

### Stage 5: Screen and audit candidates

```bash
# Screen with matched-token budget
python scripts/run_local_screen.py \
    --script records/.../train_gpt.py \
    --budget-mode tokens --budget-value 16000000 \
    --val-tune-manifest proxy_data/proxy_val_tune.json

# Parallel screening on large GPU
python scripts/run_parallel_screen.py \
    --records-dir records/track_10min_16mb \
    --output-dir parallel_results/batch_001

# Audit top candidates with multi-seed
python scripts/run_local_audit.py \
    --script records/.../train_gpt.py \
    --budget-mode tokens --budget-value 64000000 \
    --val-audit-manifest proxy_data/proxy_val_audit.json \
    --seeds 1337 42 7

# Analyze ranking fidelity
python scripts/analyze_proxy_rankings.py \
    --results-dir proxy_results/ --calibrate
```

## Design Rationale

The proxy framework design is informed by:

- **DataDecide** (arXiv:2504.11393): Rankings at matched compute budgets transfer more reliably than rankings at arbitrary budgets.
- **DoReMi** (arXiv:2305.10429): Data mixing decisions at proxy scale can transfer upward.
- **PreSelect** (arXiv:2503.00808): Some documents are more predictive of downstream quality than others.
- **SparseEval** (arXiv:2602.07909): Carefully chosen eval subsets preserve ranking signal with far fewer examples.

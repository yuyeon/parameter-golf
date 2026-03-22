#!/usr/bin/env python3
"""
Build proxy validation subsets from per-sequence profiling data.

Consumes the JSONL output of profile_full_val.py and produces three frozen
SubsetManifest JSON files:

  proxy_val_tune   - for tuning proxy parameters           (~2000 sequences)
  proxy_val_audit  - disjoint from tune, unbiased eval     (~2000 sequences)
  proxy_val_long   - longest byte-count sequences          (~500 sequences)

Strategies for selecting tune/audit sequences:
  random              - uniform random
  stratified_length   - stratified by byte count
  stratified_difficulty - stratified by mean loss across anchors
  stratified_discriminative - prioritize high-variance sequences
  mixed               - combination of difficulty + discriminativeness
  greedy_forward      - greedy forward-selection maximizing ranking agreement

Design informed by:
  - SparseEval (arXiv:2602.07909): a carefully chosen subset of evaluation
    items can preserve ranking signal with far fewer examples.
  - PreSelect (arXiv:2503.00808): some documents are more predictive of
    downstream quality than others; stratified/discriminative selection
    exploits this structure.

Usage:
    python scripts/build_proxy_val.py \
        --profile-dir profiling_results/ \
        --output-dir  proxy_val_subsets/ \
        --strategy mixed \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from proxy_framework.data_utils import SubsetManifest, save_manifest


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_profile_records(profile_dir: Path) -> list[dict]:
    """Load all JSONL profile records from directory."""
    records: list[dict] = []
    jsonl_files = sorted(profile_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files in {profile_dir}")
    for jf in jsonl_files:
        print(f"  Loading {jf.name} ...")
        with open(jf, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError as e:
                    print(f"  WARNING: skipping malformed line {line_num} in {jf.name}: {e}")
    print(f"  Loaded {len(records)} total records")
    return records


def compute_seq_stats(records: list[dict]) -> dict[int, dict]:
    """Aggregate per-sequence statistics across models.

    Returns {seq_id: {bytes, tokens, mean_loss, std_loss, mean_bpb, std_bpb,
                      n_models, losses, bpbs}}.
    """
    by_seq: dict[int, dict] = defaultdict(lambda: {
        "losses": [],
        "bpbs": [],
        "bytes": 0,
        "tokens": 0,
    })
    for rec in records:
        sid = rec["seq_id"]
        entry = by_seq[sid]
        entry["losses"].append(rec["loss"])
        entry["bpbs"].append(rec["bpb"])
        entry["bytes"] = rec["bytes"]
        entry["tokens"] = rec["tokens"]

    stats: dict[int, dict] = {}
    for sid, entry in by_seq.items():
        losses = np.array(entry["losses"], dtype=np.float64)
        bpbs = np.array(entry["bpbs"], dtype=np.float64)
        stats[sid] = {
            "seq_id": sid,
            "bytes": entry["bytes"],
            "tokens": entry["tokens"],
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)) if len(losses) > 1 else 0.0,
            "mean_bpb": float(np.mean(bpbs)),
            "std_bpb": float(np.std(bpbs)) if len(bpbs) > 1 else 0.0,
            "n_models": len(losses),
        }
    return stats


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def _stratified_sample(
    seq_ids: list[int],
    values: np.ndarray,
    n: int,
    n_strata: int,
    rng: np.random.Generator,
) -> list[int]:
    """Stratified sampling: divide into n_strata bins by value, sample
    proportionally from each bin."""
    order = np.argsort(values)
    strata: list[list[int]] = []
    for s in range(n_strata):
        lo = len(order) * s // n_strata
        hi = len(order) * (s + 1) // n_strata
        strata.append([seq_ids[order[i]] for i in range(lo, hi)])

    selected: list[int] = []
    # Allocate per stratum proportionally
    per_stratum = max(1, n // n_strata)
    remainder = n - per_stratum * n_strata

    for s_idx, stratum in enumerate(strata):
        take = per_stratum + (1 if s_idx < remainder else 0)
        take = min(take, len(stratum))
        if take > 0:
            chosen = rng.choice(stratum, size=take, replace=False).tolist()
            selected.extend(chosen)

    # If still short (edge case), fill randomly from remaining pool
    if len(selected) < n:
        pool = set(seq_ids) - set(selected)
        extra = rng.choice(list(pool), size=min(n - len(selected), len(pool)), replace=False)
        selected.extend(extra.tolist())

    return selected[:n]


def select_random(
    stats: dict[int, dict],
    n: int,
    rng: np.random.Generator,
    exclude: set[int] | None = None,
) -> list[int]:
    """Uniform random selection."""
    pool = sorted(set(stats.keys()) - (exclude or set()))
    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return sorted(chosen.tolist())


def select_stratified_length(
    stats: dict[int, dict],
    n: int,
    rng: np.random.Generator,
    exclude: set[int] | None = None,
    n_strata: int = 10,
) -> list[int]:
    """Stratified by byte count."""
    pool = sorted(set(stats.keys()) - (exclude or set()))
    values = np.array([stats[sid]["bytes"] for sid in pool], dtype=np.float64)
    return sorted(_stratified_sample(pool, values, n, n_strata, rng))


def select_stratified_difficulty(
    stats: dict[int, dict],
    n: int,
    rng: np.random.Generator,
    exclude: set[int] | None = None,
    n_strata: int = 10,
) -> list[int]:
    """Stratified by mean loss across anchors (difficulty)."""
    pool = sorted(set(stats.keys()) - (exclude or set()))
    values = np.array([stats[sid]["mean_loss"] for sid in pool], dtype=np.float64)
    return sorted(_stratified_sample(pool, values, n, n_strata, rng))


def select_stratified_discriminative(
    stats: dict[int, dict],
    n: int,
    rng: np.random.Generator,
    exclude: set[int] | None = None,
    n_strata: int = 10,
) -> list[int]:
    """Prioritize high-variance sequences (discriminativeness).

    Stratified by std_loss, with heavier sampling from higher-variance strata.
    """
    pool = sorted(set(stats.keys()) - (exclude or set()))
    values = np.array([stats[sid]["std_loss"] for sid in pool], dtype=np.float64)

    # Weight strata by increasing index (higher variance = more samples)
    order = np.argsort(values)
    n_strata_actual = min(n_strata, len(pool))
    strata: list[list[int]] = []
    for s in range(n_strata_actual):
        lo = len(order) * s // n_strata_actual
        hi = len(order) * (s + 1) // n_strata_actual
        strata.append([pool[order[i]] for i in range(lo, hi)])

    # Weights: linear ramp favoring high-variance strata
    weights = np.arange(1, n_strata_actual + 1, dtype=np.float64)
    weights /= weights.sum()
    allocations = np.round(weights * n).astype(int)
    # Adjust to sum to n
    diff = n - allocations.sum()
    allocations[-1] += diff

    selected: list[int] = []
    for s_idx, stratum in enumerate(strata):
        take = min(int(allocations[s_idx]), len(stratum))
        if take > 0:
            chosen = rng.choice(stratum, size=take, replace=False).tolist()
            selected.extend(chosen)

    # Fill if short
    if len(selected) < n:
        remaining_pool = set(pool) - set(selected)
        extra = rng.choice(
            list(remaining_pool),
            size=min(n - len(selected), len(remaining_pool)),
            replace=False,
        )
        selected.extend(extra.tolist())

    return sorted(selected[:n])


def select_mixed(
    stats: dict[int, dict],
    n: int,
    rng: np.random.Generator,
    exclude: set[int] | None = None,
) -> list[int]:
    """Mixed strategy: 50% difficulty-stratified, 50% discriminative-stratified."""
    n_difficulty = n // 2
    n_discriminative = n - n_difficulty

    selected_difficulty = select_stratified_difficulty(
        stats, n_difficulty, rng, exclude=exclude
    )
    exclude_combined = (exclude or set()) | set(selected_difficulty)
    selected_discriminative = select_stratified_discriminative(
        stats, n_discriminative, rng, exclude=exclude_combined
    )
    return sorted(selected_difficulty + selected_discriminative)


def select_greedy_forward(
    stats: dict[int, dict],
    records: list[dict],
    n: int,
    rng: np.random.Generator,
    exclude: set[int] | None = None,
) -> list[int]:
    """Greedy forward-selection maximizing ranking agreement with full val.

    For each candidate sequence, greedily pick the one that, when added to the
    selected set, maximizes Spearman correlation of per-model mean-BPB
    between the subset and the full set.

    NOTE: This can be very slow for large sequence pools.  The implementation
    uses a random subsample of candidates at each step if the pool is large.
    """
    from proxy_framework.metrics import spearman_rho

    pool = sorted(set(stats.keys()) - (exclude or set()))
    if len(pool) <= n:
        return sorted(pool)

    # Pre-compute per-model scores on full val
    model_seq_bpb: dict[str, dict[int, float]] = defaultdict(dict)
    for rec in records:
        sid = rec["seq_id"]
        if sid in (exclude or set()):
            continue
        key = rec["model_name"]
        model_seq_bpb[key][sid] = rec["bpb"]

    models = sorted(model_seq_bpb.keys())
    if len(models) < 2:
        print("  WARNING: greedy_forward needs >= 2 models. Falling back to mixed.")
        return select_mixed(stats, n, rng, exclude=exclude)

    # Full-val reference scores per model
    ref_scores = []
    for m in models:
        all_bpbs = [model_seq_bpb[m][sid] for sid in pool if sid in model_seq_bpb[m]]
        ref_scores.append(float(np.mean(all_bpbs)))

    selected: list[int] = []
    selected_set: set[int] = set()
    remaining = set(pool)

    # Limit candidate evaluation per step for speed
    max_candidates_per_step = min(500, len(pool))

    print(f"  Greedy forward selection: {n} sequences from {len(pool)} candidates")
    t0 = time.perf_counter()

    for step in range(n):
        best_sid = -1
        best_corr = -2.0

        # Random subsample of candidates for efficiency
        candidates = list(remaining)
        if len(candidates) > max_candidates_per_step:
            candidates = rng.choice(
                candidates, size=max_candidates_per_step, replace=False
            ).tolist()

        for sid in candidates:
            trial = selected + [sid]
            trial_scores = []
            for m in models:
                bpbs = [
                    model_seq_bpb[m][s]
                    for s in trial
                    if s in model_seq_bpb[m]
                ]
                trial_scores.append(float(np.mean(bpbs)) if bpbs else 0.0)
            corr = spearman_rho(trial_scores, ref_scores)
            if corr > best_corr:
                best_corr = corr
                best_sid = sid

        if best_sid < 0:
            break
        selected.append(best_sid)
        selected_set.add(best_sid)
        remaining.discard(best_sid)

        if (step + 1) % 100 == 0 or step + 1 == n:
            elapsed = time.perf_counter() - t0
            print(
                f"    Step {step+1}/{n}  best_corr={best_corr:.6f}  "
                f"elapsed={elapsed:.1f}s"
            )

    return sorted(selected)


STRATEGIES = {
    "random": select_random,
    "stratified_length": select_stratified_length,
    "stratified_difficulty": select_stratified_difficulty,
    "stratified_discriminative": select_stratified_discriminative,
    "mixed": select_mixed,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build proxy validation subsets from profiling data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        required=True,
        help="Input directory containing profile JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for SubsetManifest JSON files.",
    )
    parser.add_argument(
        "--n-tune",
        type=int,
        default=2000,
        help="Number of sequences for proxy_val_tune (default: 2000).",
    )
    parser.add_argument(
        "--n-audit",
        type=int,
        default=2000,
        help="Number of sequences for proxy_val_audit (default: 2000).",
    )
    parser.add_argument(
        "--n-long",
        type=int,
        default=500,
        help="Number of sequences for proxy_val_long (default: 500).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="mixed",
        choices=list(STRATEGIES.keys()) + ["greedy_forward"],
        help="Strategy for selecting tune/audit subsets (default: mixed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection (default: 42).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length (default: 1024).",
    )
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------------
    # Load profiling data
    # -----------------------------------------------------------------------
    print("Loading profiling data ...")
    records = load_profile_records(profile_dir)

    model_names = sorted(set(r["model_name"] for r in records))
    print(f"\nModels found: {len(model_names)}")
    for mn in model_names:
        n_recs = sum(1 for r in records if r["model_name"] == mn)
        print(f"  - {mn}  ({n_recs} records)")

    # -----------------------------------------------------------------------
    # Compute per-sequence statistics
    # -----------------------------------------------------------------------
    print("\nComputing per-sequence statistics ...")
    stats = compute_seq_stats(records)
    total_seqs = len(stats)
    print(f"  Total unique sequences: {total_seqs}")

    if total_seqs < args.n_tune + args.n_audit:
        print(
            f"  WARNING: Only {total_seqs} sequences available, but "
            f"n_tune={args.n_tune} + n_audit={args.n_audit} = {args.n_tune + args.n_audit} "
            f"requested. Reducing subset sizes proportionally."
        )
        ratio = total_seqs / (args.n_tune + args.n_audit)
        args.n_tune = max(1, int(args.n_tune * ratio * 0.95))
        args.n_audit = max(1, int(args.n_audit * ratio * 0.95))
        print(f"  Adjusted: n_tune={args.n_tune}, n_audit={args.n_audit}")

    # Print summary statistics
    all_mean_loss = [s["mean_loss"] for s in stats.values()]
    all_std_loss = [s["std_loss"] for s in stats.values()]
    all_bytes = [s["bytes"] for s in stats.values()]
    print(f"  Mean loss across sequences: {np.mean(all_mean_loss):.4f} "
          f"(std={np.std(all_mean_loss):.4f})")
    print(f"  Mean cross-model std(loss): {np.mean(all_std_loss):.4f}")
    print(f"  Byte counts: min={min(all_bytes)}, max={max(all_bytes)}, "
          f"mean={np.mean(all_bytes):.0f}")

    # -----------------------------------------------------------------------
    # Step 1: Select proxy_val_long (longest byte counts)
    # -----------------------------------------------------------------------
    print(f"\nSelecting proxy_val_long ({args.n_long} sequences) ...")
    sorted_by_bytes = sorted(stats.keys(), key=lambda sid: stats[sid]["bytes"], reverse=True)
    long_ids = sorted(sorted_by_bytes[:args.n_long])
    long_set = set(long_ids)
    min_long_bytes = min(stats[s]["bytes"] for s in long_ids) if long_ids else 0
    print(f"  Selected {len(long_ids)} sequences with byte counts "
          f">= {min_long_bytes}")

    # -----------------------------------------------------------------------
    # Step 2: Select proxy_val_tune (disjoint from long)
    # -----------------------------------------------------------------------
    print(f"\nSelecting proxy_val_tune ({args.n_tune} sequences, strategy={args.strategy}) ...")
    t0 = time.perf_counter()

    if args.strategy == "greedy_forward":
        tune_ids = select_greedy_forward(
            stats, records, args.n_tune, rng, exclude=long_set
        )
    else:
        strategy_fn = STRATEGIES[args.strategy]
        tune_ids = strategy_fn(stats, args.n_tune, rng, exclude=long_set)

    tune_set = set(tune_ids)
    elapsed = time.perf_counter() - t0
    print(f"  Selected {len(tune_ids)} sequences in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Step 3: Select proxy_val_audit (disjoint from tune AND long)
    # -----------------------------------------------------------------------
    print(f"\nSelecting proxy_val_audit ({args.n_audit} sequences, strategy={args.strategy}) ...")
    exclude_audit = long_set | tune_set
    t0 = time.perf_counter()

    if args.strategy == "greedy_forward":
        audit_ids = select_greedy_forward(
            stats, records, args.n_audit, rng, exclude=exclude_audit
        )
    else:
        strategy_fn = STRATEGIES[args.strategy]
        audit_ids = strategy_fn(stats, args.n_audit, rng, exclude=exclude_audit)

    audit_set = set(audit_ids)
    elapsed = time.perf_counter() - t0
    print(f"  Selected {len(audit_ids)} sequences in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Verify disjointness
    # -----------------------------------------------------------------------
    assert tune_set.isdisjoint(audit_set), "BUG: tune and audit overlap!"
    assert tune_set.isdisjoint(long_set), "BUG: tune and long overlap!"
    assert audit_set.isdisjoint(long_set), "BUG: audit and long overlap!"
    print("\nDisjointness verified: tune, audit, and long are all disjoint.")

    # -----------------------------------------------------------------------
    # Build and save manifests
    # -----------------------------------------------------------------------
    metadata_common = {
        "strategy": args.strategy,
        "seed": args.seed,
        "n_models_profiled": len(model_names),
        "models_profiled": model_names,
        "total_pool_sequences": total_seqs,
    }

    # proxy_val_tune
    tune_manifest = SubsetManifest(
        name="proxy_val_tune",
        split="val",
        strategy=args.strategy,
        seq_len=args.seq_len,
        seq_ids=sorted(tune_ids),
        seed=args.seed,
        metadata={
            **metadata_common,
            "purpose": "tuning proxy parameters",
            "mean_loss_of_subset": float(np.mean([stats[s]["mean_loss"] for s in tune_ids])),
            "mean_std_loss_of_subset": float(np.mean([stats[s]["std_loss"] for s in tune_ids])),
        },
    )
    tune_path = output_dir / "proxy_val_tune.json"
    save_manifest(tune_manifest, tune_path)
    print(f"\nSaved proxy_val_tune: {tune_path}")
    print(f"  {tune_manifest.n_seqs} seqs, {tune_manifest.n_tokens:,} tokens, "
          f"fingerprint={tune_manifest.fingerprint}")

    # proxy_val_audit
    audit_manifest = SubsetManifest(
        name="proxy_val_audit",
        split="val",
        strategy=args.strategy,
        seq_len=args.seq_len,
        seq_ids=sorted(audit_ids),
        seed=args.seed,
        metadata={
            **metadata_common,
            "purpose": "unbiased evaluation (disjoint from tune)",
            "mean_loss_of_subset": float(np.mean([stats[s]["mean_loss"] for s in audit_ids])),
            "mean_std_loss_of_subset": float(np.mean([stats[s]["std_loss"] for s in audit_ids])),
        },
    )
    audit_path = output_dir / "proxy_val_audit.json"
    save_manifest(audit_manifest, audit_path)
    print(f"\nSaved proxy_val_audit: {audit_path}")
    print(f"  {audit_manifest.n_seqs} seqs, {audit_manifest.n_tokens:,} tokens, "
          f"fingerprint={audit_manifest.fingerprint}")

    # proxy_val_long
    long_manifest = SubsetManifest(
        name="proxy_val_long",
        split="val",
        strategy="top_bytes",
        seq_len=args.seq_len,
        seq_ids=sorted(long_ids),
        seed=args.seed,
        metadata={
            **metadata_common,
            "purpose": "long-context validation (longest byte counts)",
            "min_bytes": int(min(stats[s]["bytes"] for s in long_ids)) if long_ids else 0,
            "max_bytes": int(max(stats[s]["bytes"] for s in long_ids)) if long_ids else 0,
            "mean_bytes": float(np.mean([stats[s]["bytes"] for s in long_ids])) if long_ids else 0,
        },
    )
    long_path = output_dir / "proxy_val_long.json"
    save_manifest(long_manifest, long_path)
    print(f"\nSaved proxy_val_long: {long_path}")
    print(f"  {long_manifest.n_seqs} seqs, {long_manifest.n_tokens:,} tokens, "
          f"fingerprint={long_manifest.fingerprint}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Strategy:      {args.strategy}")
    print(f"  Seed:          {args.seed}")
    print(f"  Total pool:    {total_seqs} sequences")
    print(f"  proxy_val_tune:  {len(tune_ids)} seqs  "
          f"(mean_loss={np.mean([stats[s]['mean_loss'] for s in tune_ids]):.4f})")
    print(f"  proxy_val_audit: {len(audit_ids)} seqs  "
          f"(mean_loss={np.mean([stats[s]['mean_loss'] for s in audit_ids]):.4f})")
    print(f"  proxy_val_long:  {len(long_ids)} seqs  "
          f"(mean_bytes={np.mean([stats[s]['bytes'] for s in long_ids]):.0f})")
    coverage = (len(tune_ids) + len(audit_ids) + len(long_ids)) / total_seqs * 100
    print(f"  Total coverage:  {len(tune_ids) + len(audit_ids) + len(long_ids)} seqs "
          f"({coverage:.1f}% of pool)")
    print(f"\nOutputs in: {output_dir}")


if __name__ == "__main__":
    main()

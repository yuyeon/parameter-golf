#!/usr/bin/env python3
"""
Build proxy training subset recipes.

Lists available training shards and generates a SubsetManifest JSON
describing which shards to use for proxy training runs.

Strategies:
  contiguous  -- first N shards (simple, deterministic)
  uniform     -- every K-th shard for diversity across the corpus
  random      -- randomly selected shards (seed-controlled)

Design informed by:
  - DoReMi (arXiv:2305.10429): data mixing decisions made at proxy scale
    can transfer upward.  Subset selection here aims to preserve training
    distribution characteristics at reduced volume.
  - DataDecide (arXiv:2504.11393): model rankings from small-scale
    experiments can predict full-scale rankings when data is chosen
    carefully.

Example:
    python scripts/build_proxy_train.py \
        --data-dir data/datasets/fineweb10B_sp1024 \
        --output-dir proxy_data \
        --n-shards 5 \
        --strategy uniform
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from proxy_framework.data_utils import (
    SubsetManifest,
    list_train_shards,
    save_manifest,
    shard_token_count,
)


def _select_shards(
    all_shards: list[Path],
    n_shards: int,
    strategy: str,
    seed: int,
) -> list[int]:
    """Return shard indices according to the chosen strategy."""
    total = len(all_shards)
    if n_shards > total:
        print(f"[warn] Requested {n_shards} shards but only {total} available. Using all.")
        n_shards = total

    if strategy == "contiguous":
        return list(range(n_shards))

    elif strategy == "uniform":
        # Pick every K-th shard to spread across the corpus
        if n_shards >= total:
            return list(range(total))
        step = total / n_shards
        indices = []
        for i in range(n_shards):
            idx = int(i * step)
            indices.append(min(idx, total - 1))
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        return sorted(deduped)

    elif strategy == "random":
        import random
        rng = random.Random(seed)
        indices = rng.sample(range(total), n_shards)
        return sorted(indices)

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: contiguous, uniform, random")


def main():
    parser = argparse.ArgumentParser(
        description="Build a proxy training subset manifest."
    )
    parser.add_argument(
        "--data-dir",
        default="data/datasets/fineweb10B_sp1024",
        help="Directory containing fineweb_train_*.bin shards",
    )
    parser.add_argument(
        "--output-dir",
        default="proxy_data",
        help="Directory to save the manifest JSON",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=5,
        help="Number of shards to include (default 5)",
    )
    parser.add_argument(
        "--strategy",
        choices=["contiguous", "uniform", "random"],
        default="contiguous",
        help="Shard selection strategy (default contiguous)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the 'random' strategy",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Manifest name (auto-generated if omitted)",
    )
    args = parser.parse_args()

    # Resolve data directory
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()

    print(f"[build] Data directory: {data_dir}")

    # List available shards
    try:
        all_shards = list_train_shards(data_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"[build] Found {len(all_shards)} training shards")
    for i, shard in enumerate(all_shards):
        tok_count = shard_token_count(shard)
        print(f"  shard {i:3d}: {shard.name}  ({tok_count:,} tokens)")

    # Select shards
    selected_indices = _select_shards(all_shards, args.n_shards, args.strategy, args.seed)
    print(f"\n[build] Strategy: {args.strategy}")
    print(f"[build] Selected {len(selected_indices)} shards: {selected_indices}")

    # Compute total tokens
    total_tokens = 0
    shard_details = []
    for idx in selected_indices:
        shard_path = all_shards[idx]
        tok_count = shard_token_count(shard_path)
        total_tokens += tok_count
        shard_details.append({
            "shard_id": idx,
            "filename": shard_path.name,
            "token_count": tok_count,
        })
        print(f"  shard {idx}: {shard_path.name} ({tok_count:,} tokens)")

    print(f"\n[build] Total tokens: {total_tokens:,}")

    # Build manifest
    manifest_name = args.name or f"proxy_train_{args.strategy}_{len(selected_indices)}shards"
    manifest = SubsetManifest(
        name=manifest_name,
        split="train",
        strategy=args.strategy,
        seq_len=1024,
        seq_ids=[],  # Train subsets are defined by shard_ids, not seq_ids
        shard_ids=selected_indices,
        n_seqs=0,
        n_tokens=total_tokens,
        seed=args.seed,
        metadata={
            "data_dir": str(data_dir),
            "total_shards_available": len(all_shards),
            "n_shards_selected": len(selected_indices),
            "shard_details": shard_details,
        },
    )

    # Save
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / f"{manifest_name}.json"
    save_manifest(manifest, manifest_path)

    print(f"\n[build] Manifest saved to {manifest_path}")
    print(f"[build] Fingerprint: {manifest.fingerprint}")

    # Also print summary
    summary = {
        "name": manifest_name,
        "strategy": args.strategy,
        "n_shards": len(selected_indices),
        "shard_ids": selected_indices,
        "total_tokens": total_tokens,
        "seed": args.seed,
        "manifest_path": str(manifest_path),
        "fingerprint": manifest.fingerprint,
    }
    print(f"\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()

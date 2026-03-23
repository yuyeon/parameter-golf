#!/usr/bin/env python3
"""
Generate multiple candidate train subsets for the proxy ranking sweep.

Creates a search space of train subset recipes, each as a SubsetManifest
JSON.  These candidates are then evaluated by sweep_train_subsets.py to
find which one best preserves model rankings.

Example:
    python scripts/generate_train_candidates.py \
        --data-dir data/datasets/fineweb10B_sp1024 \
        --output-dir artifacts/train_subsets \
        --shard-counts 5 10

    # List generated candidates
    ls artifacts/train_subsets/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from proxy_framework.train_subset_search import (
    generate_candidate_grid,
    save_candidates,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate train subsets for ranking sweep."
    )
    parser.add_argument(
        "--data-dir",
        default="data/datasets/fineweb10B_sp1024",
        help="Directory containing fineweb_train_*.bin shards",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/train_subsets",
        help="Output directory for candidate manifests",
    )
    parser.add_argument(
        "--shard-counts",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Shard counts to try for multi-shard families (default: 5 10)",
    )
    parser.add_argument(
        "--dispersed-seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Seeds for dispersed-random family (default: 42 123 456)",
    )
    parser.add_argument(
        "--single-shard-ids",
        type=int,
        nargs="*",
        default=None,
        help="Specific single-shard candidates (default: auto-spread)",
    )
    parser.add_argument(
        "--contiguous-offsets",
        type=int,
        nargs="*",
        default=None,
        help="Starting offsets for contiguous family (default: auto)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()

    print(f"[generate] Data directory: {data_dir}")
    print(f"[generate] Output directory: {args.output_dir}")

    candidates = generate_candidate_grid(
        data_dir=data_dir,
        shard_counts=args.shard_counts,
        dispersed_seeds=args.dispersed_seeds,
        single_shard_ids=args.single_shard_ids,
        contiguous_offsets=args.contiguous_offsets,
    )

    print(f"\n[generate] Generated {len(candidates)} candidates:")
    for cand in candidates:
        print(f"  {cand.candidate_id:<35} shards={cand.shard_ids}  "
              f"tokens={cand.total_tokens:,}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    paths = save_candidates(candidates, output_dir, data_dir=data_dir)

    print(f"\n[generate] Saved {len(paths)} manifests to {output_dir}")

    # Summary
    families = {}
    for c in candidates:
        families.setdefault(c.family, []).append(c.candidate_id)
    print("\nFamilies:")
    for fam, ids in sorted(families.items()):
        print(f"  {fam}: {len(ids)} candidates")


if __name__ == "__main__":
    main()

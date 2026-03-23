"""
Provisional validation lens for train-subset comparison.

Before building the FINAL proxy validation subsets we need a temporary
evaluation lens that is good enough to rank candidate train subsets.
This provisional lens is intentionally simple -- its only job is to
let us compare *relative model orderings* across different train
subsets, not to be a perfect proxy for competition evaluation.

Two modes are supported:

``full_train_val``
    Use the training script's built-in validation evaluation.  Every
    training run already computes val_bpb on the FULL official
    validation set at the end of training, so this comes for free --
    we just parse val_bpb from the logs.  This is the highest-fidelity
    provisional lens available.

``random_subset``
    Build a simple random or stratified-by-length subset of validation
    sequences (no profiling data required).  Useful when full-val
    evaluation is too slow or when we want a quicker feedback loop.

The provisional lens is NOT the final proxy_val.  Final proxy_val
subsets are built in Stage 4 after train-subset finalists are chosen.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from enum import Enum
from pathlib import Path
from typing import Sequence

from proxy_framework.data_utils import SubsetManifest, save_manifest


class ProvisionalValMode(str, Enum):
    FULL_TRAIN_VAL = "full_train_val"
    RANDOM_SUBSET = "random_subset"


def build_provisional_val_subset(
    data_dir: str | Path,
    n_seqs: int = 2000,
    seq_len: int = 1024,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> SubsetManifest:
    """Build a simple random provisional validation subset.

    Does NOT require profiling data -- just counts total val sequences
    and samples uniformly.

    Args:
        data_dir: Directory containing ``fineweb_val_*.bin`` shards.
        n_seqs: Number of sequences to include.
        seq_len: Sequence length.
        seed: Random seed for reproducibility.
        output_path: If provided, save the manifest here.

    Returns:
        A SubsetManifest with random seq_ids.
    """
    import random as stdlib_random

    data_dir = Path(data_dir)

    # Count total validation tokens to determine sequence count
    total_val_tokens = 0
    for val_shard in sorted(data_dir.glob("fineweb_val_*.bin")):
        with open(val_shard, "rb") as f:
            header = struct.unpack("<256i", f.read(256 * 4))
            assert header[0] == 20240520, f"Bad magic: {header[0]}"
            total_val_tokens += header[2]

    if total_val_tokens == 0:
        raise FileNotFoundError(f"No validation shards in {data_dir}")

    total_seqs = (total_val_tokens - 1) // seq_len
    n_seqs = min(n_seqs, total_seqs)

    rng = stdlib_random.Random(seed)
    seq_ids = sorted(rng.sample(range(total_seqs), n_seqs))

    manifest = SubsetManifest(
        name="provisional_val",
        split="val",
        strategy="random",
        seq_len=seq_len,
        seq_ids=seq_ids,
        seed=seed,
        metadata={
            "purpose": "provisional validation lens for train-subset comparison",
            "total_val_sequences": total_seqs,
            "total_val_tokens": total_val_tokens,
            "is_provisional": True,
        },
    )

    if output_path:
        save_manifest(manifest, output_path)

    return manifest


def extract_val_bpb_from_log(log_text: str) -> float | None:
    """Parse the final val_bpb from a training log.

    The training script prints lines like:
        step:488 | val_loss:4.1234 | val_bpb:5.9523 | ...

    Returns the last val_bpb found, or None.
    """
    import re
    val_bpb = None
    for line in reversed(log_text.splitlines()):
        if "val_bpb:" in line and "step:" in line:
            m = re.search(r"val_bpb:([\d.]+)", line)
            if m:
                val_bpb = float(m.group(1))
                break
    return val_bpb


def extract_post_quant_bpb_from_log(log_text: str) -> float | None:
    """Parse the post-quantization val_bpb from a training log."""
    import re
    for line in log_text.splitlines():
        if "final_int8_zlib_roundtrip " in line:
            m = re.search(r"val_bpb:([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def collect_sweep_scores(
    results_dir: str | Path,
    mode: ProvisionalValMode = ProvisionalValMode.FULL_TRAIN_VAL,
) -> dict[str, dict[str, float]]:
    """Collect per-(candidate, model) scores from a sweep output directory.

    Expected directory structure::

        results_dir/
          <candidate_id>/
            <model_name>/
              train.log
              run_summary.json

    Returns ``{candidate_id: {model_name: val_bpb}}``.
    """
    results_dir = Path(results_dir)
    scores: dict[str, dict[str, float]] = {}

    for cand_dir in sorted(results_dir.iterdir()):
        if not cand_dir.is_dir():
            continue
        cand_id = cand_dir.name
        cand_scores: dict[str, float] = {}

        for model_dir in sorted(cand_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            # Try run_summary.json first
            summary_path = model_dir / "run_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    bpb = (summary.get("post_quant_val_bpb")
                           or summary.get("pre_quant_val_bpb")
                           or summary.get("proxy_val_tune_bpb"))
                    if bpb and bpb > 0:
                        cand_scores[model_name] = bpb
                        continue
                except Exception:
                    pass

            # Fall back to log parsing
            log_path = model_dir / "train.log"
            if log_path.exists():
                log_text = log_path.read_text(errors="replace")
                bpb = extract_post_quant_bpb_from_log(log_text)
                if bpb is None:
                    bpb = extract_val_bpb_from_log(log_text)
                if bpb is not None:
                    cand_scores[model_name] = bpb

        if cand_scores:
            scores[cand_id] = cand_scores

    return scores

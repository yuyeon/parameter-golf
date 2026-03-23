"""
Train subset candidate generation for proxy ranking fidelity.

The most relevant train subset is the one whose cheap local training
runs induce model rankings that best match higher-fidelity rankings
under the target competition setting.

This module generates multiple candidate training subsets from the
available shard pool.  Each candidate is a SubsetManifest specifying
exact shard IDs, sampling seed, and metadata.

Candidate families
------------------
contiguous   First N shards in order (simple baseline).
uniform      Every K-th shard for maximum spread.
dispersed    Random shard sample (seed-controlled).
single       Just one shard (tests data-sensitivity).
bookend      First N//2 + last N//2 shards.
odd / even   Alternating shard indices.

Since train_gpt.py reads shards sequentially via TokenStream, the
*order* of shards in DATA_PATH determines what the model actually
sees.  With a 16 M-token budget at 32 K tokens/step the model reads
~16 % of the first shard, so the identity of the first shard is the
dominant variable at screening budgets.  At audit budgets (64 M+) or
with warmdown phases that re-read earlier data, shard diversity matters
more.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from proxy_framework.data_utils import (
    SubsetManifest,
    list_train_shards,
    save_manifest,
    shard_token_count,
)


# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainSubsetCandidate:
    """One candidate train subset with full provenance."""

    candidate_id: str
    family: str               # contiguous | uniform | dispersed | single | bookend | odd | even
    shard_ids: list[int]
    n_shards: int = 0
    total_tokens: int = 0
    seed: int = 42
    metadata: dict = field(default_factory=dict)
    manifest_path: str = ""

    def __post_init__(self):
        if not self.n_shards:
            self.n_shards = len(self.shard_ids)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainSubsetCandidate":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Family generators
# ---------------------------------------------------------------------------

def _contiguous(n_available: int, n_shards: int, offset: int = 0) -> list[int]:
    """First ``n_shards`` shards starting from ``offset``."""
    return [(offset + i) % n_available for i in range(min(n_shards, n_available))]


def _uniform(n_available: int, n_shards: int) -> list[int]:
    """Every K-th shard for maximum spread."""
    if n_shards >= n_available:
        return list(range(n_available))
    step = n_available / n_shards
    indices = sorted(set(min(int(i * step), n_available - 1) for i in range(n_shards)))
    return indices


def _dispersed(n_available: int, n_shards: int, seed: int) -> list[int]:
    """Randomly sampled shards (seed-controlled)."""
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_available), min(n_shards, n_available)))


def _single(shard_id: int) -> list[int]:
    """Just one shard."""
    return [shard_id]


def _bookend(n_available: int, n_shards: int) -> list[int]:
    """First half + last half for corpus-boundary coverage."""
    if n_shards >= n_available:
        return list(range(n_available))
    first_half = n_shards // 2
    second_half = n_shards - first_half
    ids = list(range(first_half)) + list(range(n_available - second_half, n_available))
    return sorted(set(ids))


def _odd(n_available: int) -> list[int]:
    return [i for i in range(n_available) if i % 2 == 1]


def _even(n_available: int) -> list[int]:
    return [i for i in range(n_available) if i % 2 == 0]


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_candidate_grid(
    data_dir: str | Path,
    shard_counts: Sequence[int] = (5, 10),
    dispersed_seeds: Sequence[int] = (42, 123, 456),
    single_shard_ids: Sequence[int] | None = None,
    contiguous_offsets: Sequence[int] | None = None,
) -> list[TrainSubsetCandidate]:
    """Generate the full search space of train subset candidates.

    Args:
        data_dir: Directory containing ``fineweb_train_*.bin`` shards.
        shard_counts: Sizes to try for contiguous/uniform/dispersed/bookend.
        dispersed_seeds: Seeds for the dispersed-random family.
        single_shard_ids: Which single-shard candidates to create.
            Defaults to a spread across available shards.
        contiguous_offsets: Starting offsets for contiguous candidates.
            Defaults to [0] plus a mid-point offset.

    Returns:
        List of ``TrainSubsetCandidate`` objects (unsaved).
    """
    data_dir = Path(data_dir)
    all_shards = list_train_shards(data_dir)
    n_avail = len(all_shards)

    if n_avail == 0:
        raise FileNotFoundError(f"No training shards found in {data_dir}")

    # Token counts per shard
    shard_tokens = {i: shard_token_count(all_shards[i]) for i in range(n_avail)}

    # Defaults
    if single_shard_ids is None:
        # Spread across corpus: first, ~1/4, ~1/2, ~3/4, last
        single_shard_ids = sorted(set([
            0,
            max(0, n_avail // 4),
            max(0, n_avail // 2),
            max(0, 3 * n_avail // 4),
            n_avail - 1,
        ]))

    if contiguous_offsets is None:
        contiguous_offsets = [0, n_avail // 2] if n_avail > 1 else [0]

    candidates: list[TrainSubsetCandidate] = []

    def _make(family: str, shard_ids: list[int], suffix: str = "",
              seed: int = 42, extra_meta: dict | None = None) -> None:
        cid = f"{family}_{suffix}" if suffix else family
        tokens = sum(shard_tokens.get(s, 0) for s in shard_ids)
        meta = {"n_available_shards": n_avail}
        if extra_meta:
            meta.update(extra_meta)
        candidates.append(TrainSubsetCandidate(
            candidate_id=cid,
            family=family,
            shard_ids=shard_ids,
            total_tokens=tokens,
            seed=seed,
            metadata=meta,
        ))

    # --- Family: contiguous ---
    for n in shard_counts:
        if n > n_avail:
            continue
        for off in contiguous_offsets:
            ids = _contiguous(n_avail, n, offset=off)
            _make("contiguous", ids, f"{n}sh_off{off}", extra_meta={"offset": off})

    # --- Family: uniform ---
    for n in shard_counts:
        if n > n_avail:
            continue
        ids = _uniform(n_avail, n)
        _make("uniform", ids, f"{n}sh")

    # --- Family: dispersed ---
    for n in shard_counts:
        if n > n_avail:
            continue
        for seed in dispersed_seeds:
            ids = _dispersed(n_avail, n, seed)
            _make("dispersed", ids, f"{n}sh_seed{seed}", seed=seed)

    # --- Family: single ---
    for sid in single_shard_ids:
        if sid < n_avail:
            _make("single", _single(sid), f"shard{sid}")

    # --- Family: bookend ---
    for n in shard_counts:
        if n > n_avail:
            continue
        ids = _bookend(n_avail, n)
        _make("bookend", ids, f"{n}sh")

    # --- Family: odd / even ---
    if n_avail >= 4:
        _make("odd", _odd(n_avail), f"{len(_odd(n_avail))}sh")
        _make("even", _even(n_avail), f"{len(_even(n_avail))}sh")

    return candidates


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def save_candidates(
    candidates: list[TrainSubsetCandidate],
    output_dir: str | Path,
    data_dir: str | Path | None = None,
) -> list[Path]:
    """Save each candidate as a SubsetManifest JSON.

    Returns list of saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for cand in candidates:
        manifest = SubsetManifest(
            name=cand.candidate_id,
            split="train",
            strategy=cand.family,
            seq_len=1024,
            seq_ids=[],         # train subsets defined by shard_ids
            shard_ids=cand.shard_ids,
            n_seqs=0,
            n_tokens=cand.total_tokens,
            seed=cand.seed,
            metadata={
                "family": cand.family,
                "candidate_id": cand.candidate_id,
                **({"data_dir": str(data_dir)} if data_dir else {}),
                **cand.metadata,
            },
        )
        manifest_path = output_dir / f"{cand.candidate_id}.json"
        save_manifest(manifest, manifest_path)
        cand.manifest_path = str(manifest_path)
        paths.append(manifest_path)

    return paths


def load_candidates(input_dir: str | Path) -> list[TrainSubsetCandidate]:
    """Load all candidate manifests from a directory."""
    input_dir = Path(input_dir)
    candidates = []
    for p in sorted(input_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        candidates.append(TrainSubsetCandidate(
            candidate_id=meta.get("candidate_id", data.get("name", p.stem)),
            family=meta.get("family", data.get("strategy", "unknown")),
            shard_ids=data.get("shard_ids", []),
            total_tokens=data.get("n_tokens", 0),
            seed=data.get("seed", 42),
            metadata=meta,
            manifest_path=str(p),
        ))
    return candidates


# ---------------------------------------------------------------------------
# DATA_PATH isolation via symlinks
# ---------------------------------------------------------------------------

def prepare_shard_dir(
    shard_ids: list[int],
    source_data_dir: str | Path,
    work_dir: str | Path | None = None,
) -> Path:
    """Create a temporary directory with symlinks to specific shards.

    The training script reads shards from DATA_PATH via glob on
    ``fineweb_train_*.bin``.  By symlinking only the desired shards
    into a temp directory, we control exactly which data is used.

    Validation shards and tokenizer files are also symlinked so the
    training script's eval phase works normally.

    Args:
        shard_ids: Which shard indices to include.
        source_data_dir: Original data directory with all shards.
        work_dir: If provided, create the shard dir here (caller owns
            cleanup).  Otherwise a temp directory is created.

    Returns:
        Path to the prepared directory.
    """
    source = Path(source_data_dir).resolve()

    if work_dir is not None:
        shard_dir = Path(work_dir) / "shard_subset"
        shard_dir.mkdir(parents=True, exist_ok=True)
    else:
        shard_dir = Path(tempfile.mkdtemp(prefix="pgolf_shards_"))

    # Symlink selected training shards
    for sid in shard_ids:
        src = source / f"fineweb_train_{sid:06d}.bin"
        dst = shard_dir / src.name
        if src.exists() and not dst.exists():
            os.symlink(str(src), str(dst))

    # Symlink ALL validation shards (needed for eval)
    for val_shard in source.glob("fineweb_val_*.bin"):
        dst = shard_dir / val_shard.name
        if not dst.exists():
            os.symlink(str(val_shard), str(dst))

    return shard_dir


def cleanup_shard_dir(shard_dir: str | Path) -> None:
    """Remove a temporary shard directory created by ``prepare_shard_dir``."""
    shutil.rmtree(str(shard_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# Reference ranking loader
# ---------------------------------------------------------------------------

def load_reference_ranking(
    records_dir: str | Path,
    metric: str = "val_bpb",
) -> dict[str, float]:
    """Load official reference ranking from submission.json files.

    Handles the various submission.json formats in the repository.
    Returns ``{model_name: bpb_score}`` (lower is better).
    """
    records_dir = Path(records_dir)
    scores: dict[str, float] = {}

    for sub_dir in sorted(records_dir.iterdir()):
        sub_json = sub_dir / "submission.json"
        if not sub_json.exists():
            continue
        try:
            with open(sub_json) as f:
                data = json.load(f)
            name = sub_dir.name
            # Try multiple possible keys
            bpb = (data.get("val_bpb")
                   or data.get("mean_val_bpb")
                   or data.get("bpb"))
            if bpb is not None:
                scores[name] = float(bpb)
        except Exception:
            continue

    return scores

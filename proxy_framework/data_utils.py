"""
Data utilities for proxy framework.

Handles loading challenge-native binary shards, defining subsets by
sequence ID, and converting between subset manifests and token tensors.
The binary shard format is defined by the Parameter Golf repo:
  header: 256 x int32 (magic=20240520, version=1, num_tokens, ...)
  body:   num_tokens x uint16

Subset manifest design inspired by:
  - PreSelect (arXiv:2503.00808): reproducible document-level selection
  - DoReMi (arXiv:2305.10429): proxy-scale data recipes that transfer
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Binary shard I/O (mirrors the repo's load_data_shard)
# ---------------------------------------------------------------------------

SHARD_MAGIC = 20240520
HEADER_INTS = 256


def load_shard(path: Path | str) -> Tensor:
    """Load a binary shard and return a 1-D uint16 tensor of token IDs."""
    path = Path(path)
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {path}")
    n = int(header[2])
    offset = HEADER_INTS * np.dtype("<i4").itemsize
    tokens = np.fromfile(path, dtype="<u2", count=n, offset=offset)
    if tokens.size != n:
        raise ValueError(f"Short read: {path}")
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


def load_all_val_tokens(pattern_or_dir: str | Path, seq_len: int = 1024) -> Tensor:
    """Load all validation tokens, trimmed to a multiple of seq_len + 1."""
    import glob
    p = str(pattern_or_dir)
    if "*" not in p:
        p = str(Path(p) / "fineweb_val_*.bin")
    files = sorted(glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No val files: {p}")
    tokens = torch.cat([load_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


# ---------------------------------------------------------------------------
# Sequence-level views of the validation set
# ---------------------------------------------------------------------------

@dataclass
class SeqInfo:
    """Metadata for one fixed-length sequence (our notion of 'document')."""
    seq_id: int
    token_offset: int      # start index in the flat token array
    num_tokens: int        # always == seq_len for full sequences
    num_bytes: int = 0     # UTF-8 bytes this sequence represents


def enumerate_sequences(
    total_tokens: int, seq_len: int
) -> list[SeqInfo]:
    """Return SeqInfo list for all non-overlapping sequences."""
    n_seqs = (total_tokens - 1) // seq_len
    return [
        SeqInfo(seq_id=i, token_offset=i * seq_len, num_tokens=seq_len)
        for i in range(n_seqs)
    ]


def extract_sequences(
    tokens: Tensor, seq_ids: list[int], seq_len: int
) -> tuple[Tensor, Tensor]:
    """Extract (x, y) pairs for specific sequence IDs.

    Returns:
        x: (len(seq_ids), seq_len) input token IDs
        y: (len(seq_ids), seq_len) target token IDs
    """
    xs, ys = [], []
    for sid in seq_ids:
        start = sid * seq_len
        chunk = tokens[start : start + seq_len + 1].long()
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return torch.stack(xs), torch.stack(ys)


def iter_batches(
    tokens: Tensor,
    seq_ids: list[int],
    seq_len: int,
    batch_seqs: int,
) -> Iterator[tuple[Tensor, Tensor, list[int]]]:
    """Yield (x, y, batch_seq_ids) batches for given sequence IDs."""
    for i in range(0, len(seq_ids), batch_seqs):
        batch_ids = seq_ids[i : i + batch_seqs]
        x, y = extract_sequences(tokens, batch_ids, seq_len)
        yield x, y, batch_ids


# ---------------------------------------------------------------------------
# Subset manifests
# ---------------------------------------------------------------------------

@dataclass
class SubsetManifest:
    """Deterministic, reproducible definition of a data subset."""
    name: str
    split: str                          # "train" or "val"
    strategy: str                       # e.g. "random", "stratified_difficulty", ...
    seq_len: int
    seq_ids: list[int] = field(default_factory=list)
    shard_ids: list[int] | None = None  # for train subsets
    n_seqs: int = 0
    n_tokens: int = 0
    seed: int = 42
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_seqs = len(self.seq_ids) if self.seq_ids else self.n_seqs
        self.n_tokens = self.n_seqs * self.seq_len

    @property
    def fingerprint(self) -> str:
        """Deterministic hash for cache-busting / dedup."""
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(str(sorted(self.seq_ids)).encode())
        h.update(str(self.seq_len).encode())
        return h.hexdigest()[:16]


def save_manifest(manifest: SubsetManifest, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_manifest(path: Path | str) -> SubsetManifest:
    with open(path) as f:
        d = json.load(f)
    return SubsetManifest(**d)


# ---------------------------------------------------------------------------
# Train shard helpers
# ---------------------------------------------------------------------------

def list_train_shards(data_dir: str | Path) -> list[Path]:
    """List available training shard files in order."""
    d = Path(data_dir)
    shards = sorted(d.glob("fineweb_train_*.bin"))
    if not shards:
        raise FileNotFoundError(f"No training shards in {d}")
    return shards


def shard_token_count(path: Path) -> int:
    """Read token count from shard header without loading data."""
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    return int(header[2])

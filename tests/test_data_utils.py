"""
Tests for proxy_framework.data_utils data handling utilities.

Run with:
    python -m pytest tests/test_data_utils.py -v

Uses small synthetic tensors; no real data files needed.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from proxy_framework.data_utils import (
    SubsetManifest,
    enumerate_sequences,
    extract_sequences,
    load_manifest,
    save_manifest,
)


# ---------------------------------------------------------------------------
# enumerate_sequences
# ---------------------------------------------------------------------------

class TestEnumerateSequences:
    def test_basic_count(self):
        # 1025 tokens with seq_len=1024 -> (1025-1)//1024 = 1 sequence
        seqs = enumerate_sequences(total_tokens=1025, seq_len=1024)
        assert len(seqs) == 1
        assert seqs[0].seq_id == 0
        assert seqs[0].token_offset == 0
        assert seqs[0].num_tokens == 1024

    def test_multiple_sequences(self):
        # 10241 tokens with seq_len=1024 -> (10241-1)//1024 = 10 sequences
        seqs = enumerate_sequences(total_tokens=10241, seq_len=1024)
        assert len(seqs) == 10
        for i, s in enumerate(seqs):
            assert s.seq_id == i
            assert s.token_offset == i * 1024
            assert s.num_tokens == 1024

    def test_exact_boundary(self):
        # 2049 tokens with seq_len=1024 -> (2049-1)//1024 = 2 sequences
        seqs = enumerate_sequences(total_tokens=2049, seq_len=1024)
        assert len(seqs) == 2

    def test_not_enough_for_one(self):
        # 1024 tokens with seq_len=1024 -> (1024-1)//1024 = 0 sequences
        seqs = enumerate_sequences(total_tokens=1024, seq_len=1024)
        assert len(seqs) == 0

    def test_short_seq_len(self):
        # 21 tokens, seq_len=4 -> (21-1)//4 = 5 sequences
        seqs = enumerate_sequences(total_tokens=21, seq_len=4)
        assert len(seqs) == 5
        assert seqs[4].token_offset == 16

    def test_seq_ids_are_sequential(self):
        seqs = enumerate_sequences(total_tokens=5001, seq_len=100)
        ids = [s.seq_id for s in seqs]
        assert ids == list(range(len(seqs)))


# ---------------------------------------------------------------------------
# SubsetManifest serialization roundtrip
# ---------------------------------------------------------------------------

class TestSubsetManifest:
    def test_roundtrip(self, tmp_path):
        manifest = SubsetManifest(
            name="test_subset",
            split="val",
            strategy="random",
            seq_len=1024,
            seq_ids=[0, 5, 10, 15, 20],
            seed=42,
            metadata={"source": "unit_test"},
        )
        path = tmp_path / "manifest.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)

        assert loaded.name == manifest.name
        assert loaded.split == manifest.split
        assert loaded.strategy == manifest.strategy
        assert loaded.seq_len == manifest.seq_len
        assert loaded.seq_ids == manifest.seq_ids
        assert loaded.seed == manifest.seed
        assert loaded.metadata == manifest.metadata
        assert loaded.n_seqs == 5
        assert loaded.n_tokens == 5 * 1024

    def test_roundtrip_empty(self, tmp_path):
        manifest = SubsetManifest(
            name="empty",
            split="train",
            strategy="none",
            seq_len=512,
            seq_ids=[],
        )
        path = tmp_path / "empty.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)
        assert loaded.seq_ids == []
        assert loaded.n_seqs == 0
        assert loaded.n_tokens == 0

    def test_saved_json_is_valid(self, tmp_path):
        manifest = SubsetManifest(
            name="json_check",
            split="val",
            strategy="random",
            seq_len=256,
            seq_ids=[1, 2, 3],
        )
        path = tmp_path / "check.json"
        save_manifest(manifest, path)
        # Should be valid JSON
        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "json_check"
        assert data["seq_ids"] == [1, 2, 3]

    def test_shard_ids_roundtrip(self, tmp_path):
        manifest = SubsetManifest(
            name="train_sub",
            split="train",
            strategy="contiguous",
            seq_len=1024,
            shard_ids=[0, 1, 2, 3, 4],
        )
        path = tmp_path / "train.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)
        assert loaded.shard_ids == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# SubsetManifest fingerprint
# ---------------------------------------------------------------------------

class TestSubsetManifestFingerprint:
    def test_deterministic(self):
        m1 = SubsetManifest(
            name="fp_test", split="val", strategy="random",
            seq_len=1024, seq_ids=[0, 1, 2, 3],
        )
        m2 = SubsetManifest(
            name="fp_test", split="val", strategy="random",
            seq_len=1024, seq_ids=[0, 1, 2, 3],
        )
        assert m1.fingerprint == m2.fingerprint

    def test_different_ids_different_fingerprint(self):
        m1 = SubsetManifest(
            name="fp_test", split="val", strategy="random",
            seq_len=1024, seq_ids=[0, 1, 2],
        )
        m2 = SubsetManifest(
            name="fp_test", split="val", strategy="random",
            seq_len=1024, seq_ids=[0, 1, 3],
        )
        assert m1.fingerprint != m2.fingerprint

    def test_different_name_different_fingerprint(self):
        m1 = SubsetManifest(
            name="subset_a", split="val", strategy="random",
            seq_len=1024, seq_ids=[0, 1, 2],
        )
        m2 = SubsetManifest(
            name="subset_b", split="val", strategy="random",
            seq_len=1024, seq_ids=[0, 1, 2],
        )
        assert m1.fingerprint != m2.fingerprint

    def test_different_seq_len_different_fingerprint(self):
        m1 = SubsetManifest(
            name="fp_test", split="val", strategy="random",
            seq_len=1024, seq_ids=[0, 1, 2],
        )
        m2 = SubsetManifest(
            name="fp_test", split="val", strategy="random",
            seq_len=2048, seq_ids=[0, 1, 2],
        )
        assert m1.fingerprint != m2.fingerprint

    def test_order_invariance(self):
        """Fingerprint should be the same regardless of seq_ids order
        because it sorts internally."""
        m1 = SubsetManifest(
            name="fp_test", split="val", strategy="random",
            seq_len=1024, seq_ids=[3, 1, 2, 0],
        )
        m2 = SubsetManifest(
            name="fp_test", split="val", strategy="random",
            seq_len=1024, seq_ids=[0, 1, 2, 3],
        )
        assert m1.fingerprint == m2.fingerprint

    def test_fingerprint_length(self):
        m = SubsetManifest(
            name="test", split="val", strategy="random",
            seq_len=1024, seq_ids=[0],
        )
        assert len(m.fingerprint) == 16  # first 16 hex chars of SHA-256


# ---------------------------------------------------------------------------
# extract_sequences
# ---------------------------------------------------------------------------

class TestExtractSequences:
    def test_basic_extraction(self):
        # Create a simple token tensor: [0, 1, 2, ..., 20]
        tokens = torch.arange(21, dtype=torch.long)
        seq_len = 4

        # Extract sequence 0: tokens[0:5] -> x=[0,1,2,3], y=[1,2,3,4]
        x, y = extract_sequences(tokens, seq_ids=[0], seq_len=seq_len)
        assert x.shape == (1, 4)
        assert y.shape == (1, 4)
        assert x[0].tolist() == [0, 1, 2, 3]
        assert y[0].tolist() == [1, 2, 3, 4]

    def test_multiple_sequences(self):
        tokens = torch.arange(21, dtype=torch.long)
        seq_len = 4

        x, y = extract_sequences(tokens, seq_ids=[0, 2], seq_len=seq_len)
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)
        # Seq 0: start=0 -> x=[0,1,2,3], y=[1,2,3,4]
        assert x[0].tolist() == [0, 1, 2, 3]
        assert y[0].tolist() == [1, 2, 3, 4]
        # Seq 2: start=8 -> x=[8,9,10,11], y=[9,10,11,12]
        assert x[1].tolist() == [8, 9, 10, 11]
        assert y[1].tolist() == [9, 10, 11, 12]

    def test_output_dtype_is_long(self):
        tokens = torch.arange(11, dtype=torch.int16)
        x, y = extract_sequences(tokens, seq_ids=[0], seq_len=4)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_single_token_seq(self):
        tokens = torch.arange(5, dtype=torch.long)
        x, y = extract_sequences(tokens, seq_ids=[0, 1], seq_len=1)
        assert x.shape == (2, 1)
        assert y.shape == (2, 1)
        assert x[0].item() == 0
        assert y[0].item() == 1
        assert x[1].item() == 1
        assert y[1].item() == 2

    def test_contiguous_sequences_cover_tokens(self):
        """Extracting all sequences should tile the token range."""
        tokens = torch.arange(13, dtype=torch.long)
        seq_len = 3
        # (13-1)//3 = 4 sequences
        x, y = extract_sequences(tokens, seq_ids=[0, 1, 2, 3], seq_len=seq_len)
        assert x.shape == (4, 3)
        # Each sequence's x starts at i*seq_len
        for i in range(4):
            assert x[i, 0].item() == i * seq_len

"""
Tests for proxy_framework.train_subset_search, provisional_val, and
finalist_selection.

Run with:
    python -m pytest tests/test_train_subset_search.py -v
"""

import json
import os
import struct
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from proxy_framework.train_subset_search import (
    TrainSubsetCandidate,
    _bookend,
    _contiguous,
    _dispersed,
    _even,
    _odd,
    _single,
    _uniform,
    generate_candidate_grid,
    load_candidates,
    load_reference_ranking,
    prepare_shard_dir,
    save_candidates,
)
from proxy_framework.provisional_val import (
    ProvisionalValMode,
    build_provisional_val_subset,
    extract_post_quant_bpb_from_log,
    extract_val_bpb_from_log,
)
from proxy_framework.finalist_selection import (
    CandidateEvaluation,
    SelectionReport,
    build_selection_report,
    evaluate_candidate,
    format_evaluation_table,
    load_finalists,
    save_selection_report,
    select_finalists,
)


# ---------------------------------------------------------------------------
# Helpers: create fake shard files
# ---------------------------------------------------------------------------

def _write_fake_shard(path: Path, n_tokens: int = 1000):
    """Write a minimal valid shard file (magic + header + tokens)."""
    header = [0] * 256
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = n_tokens
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{256}i", *header))
        # Write dummy tokens (uint16)
        f.write(b"\x00\x01" * n_tokens)


def _make_shard_dir(tmp_path: Path, n_shards: int = 10,
                    n_val: int = 1, tokens_per: int = 1000) -> Path:
    """Create a temp directory with fake train + val shards."""
    d = tmp_path / "data"
    d.mkdir()
    for i in range(n_shards):
        _write_fake_shard(d / f"fineweb_train_{i:06d}.bin", tokens_per)
    for i in range(n_val):
        _write_fake_shard(d / f"fineweb_val_{i:06d}.bin", tokens_per)
    return d


# ===========================================================================
# train_subset_search tests
# ===========================================================================

class TestFamilyGenerators:
    def test_contiguous_basic(self):
        assert _contiguous(10, 5) == [0, 1, 2, 3, 4]

    def test_contiguous_offset(self):
        assert _contiguous(10, 3, offset=7) == [7, 8, 9]

    def test_contiguous_wrap(self):
        # offset=8, 5 shards wraps around
        assert _contiguous(10, 5, offset=8) == [8, 9, 0, 1, 2]

    def test_uniform_spread(self):
        ids = _uniform(10, 5)
        assert len(ids) == 5
        assert ids == sorted(ids)
        # Should cover 0 and 8 (roughly evenly spaced)
        assert 0 in ids

    def test_uniform_all(self):
        assert _uniform(5, 10) == [0, 1, 2, 3, 4]

    def test_dispersed_deterministic(self):
        a = _dispersed(10, 5, seed=42)
        b = _dispersed(10, 5, seed=42)
        assert a == b
        assert len(a) == 5

    def test_dispersed_different_seeds(self):
        a = _dispersed(10, 5, seed=42)
        b = _dispersed(10, 5, seed=123)
        assert a != b

    def test_single(self):
        assert _single(3) == [3]

    def test_bookend(self):
        ids = _bookend(10, 4)
        # Should include first 2 and last 2
        assert 0 in ids
        assert 1 in ids
        assert 8 in ids
        assert 9 in ids

    def test_odd(self):
        assert _odd(10) == [1, 3, 5, 7, 9]

    def test_even(self):
        assert _even(10) == [0, 2, 4, 6, 8]


class TestTrainSubsetCandidate:
    def test_basic(self):
        c = TrainSubsetCandidate(
            candidate_id="test_1",
            family="contiguous",
            shard_ids=[0, 1, 2],
        )
        assert c.n_shards == 3
        assert c.candidate_id == "test_1"

    def test_to_dict(self):
        c = TrainSubsetCandidate(
            candidate_id="test_1",
            family="single",
            shard_ids=[5],
            total_tokens=100000,
        )
        d = c.to_dict()
        assert d["candidate_id"] == "test_1"
        assert d["shard_ids"] == [5]


class TestGenerateCandidateGrid:
    def test_basic(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=10)
        candidates = generate_candidate_grid(
            data_dir,
            shard_counts=[5],
            dispersed_seeds=[42],
        )
        assert len(candidates) > 0
        # Check families
        families = {c.family for c in candidates}
        assert "contiguous" in families
        assert "uniform" in families
        assert "dispersed" in families
        assert "single" in families
        assert "bookend" in families

    def test_odd_even_with_enough_shards(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=6)
        candidates = generate_candidate_grid(
            data_dir,
            shard_counts=[3],
            dispersed_seeds=[42],
        )
        families = {c.family for c in candidates}
        assert "odd" in families
        assert "even" in families

    def test_no_odd_even_with_few_shards(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=3)
        candidates = generate_candidate_grid(
            data_dir,
            shard_counts=[2],
            dispersed_seeds=[42],
        )
        families = {c.family for c in candidates}
        # 3 shards < 4 threshold for odd/even
        assert "odd" not in families

    def test_all_unique_ids(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=10)
        candidates = generate_candidate_grid(data_dir)
        ids = [c.candidate_id for c in candidates]
        assert len(ids) == len(set(ids)), "Duplicate candidate IDs"

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            generate_candidate_grid(empty)


class TestSaveCandidates:
    def test_save_load_roundtrip(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=5)
        candidates = generate_candidate_grid(
            data_dir, shard_counts=[3], dispersed_seeds=[42],
        )
        out = tmp_path / "manifests"
        save_candidates(candidates, out, data_dir)

        loaded = load_candidates(out)
        assert len(loaded) == len(candidates)
        assert {c.candidate_id for c in loaded} == {c.candidate_id for c in candidates}


class TestPrepareShardDir:
    def test_symlinks_correct_shards(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=10)
        shard_dir = prepare_shard_dir([2, 5, 7], data_dir, work_dir=tmp_path / "work")

        train_files = sorted(shard_dir.glob("fineweb_train_*.bin"))
        assert len(train_files) == 3
        names = {f.name for f in train_files}
        assert "fineweb_train_000002.bin" in names
        assert "fineweb_train_000005.bin" in names
        assert "fineweb_train_000007.bin" in names

    def test_val_shards_symlinked(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=5, n_val=1)
        shard_dir = prepare_shard_dir([0], data_dir, work_dir=tmp_path / "work")

        val_files = list(shard_dir.glob("fineweb_val_*.bin"))
        assert len(val_files) == 1

    def test_missing_shard_skipped(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=3)
        # Shard 5 doesn't exist -- should not crash
        shard_dir = prepare_shard_dir([0, 5], data_dir, work_dir=tmp_path / "work")
        train_files = list(shard_dir.glob("fineweb_train_*.bin"))
        assert len(train_files) == 1  # Only shard 0


class TestLoadReferenceRanking:
    def test_loads_scores(self, tmp_path):
        for name, bpb in [("model_A", 1.23), ("model_B", 1.45)]:
            d = tmp_path / name
            d.mkdir()
            (d / "submission.json").write_text(json.dumps({"val_bpb": bpb}))

        scores = load_reference_ranking(tmp_path)
        assert scores["model_A"] == pytest.approx(1.23)
        assert scores["model_B"] == pytest.approx(1.45)

    def test_handles_missing_fields(self, tmp_path):
        d = tmp_path / "model_X"
        d.mkdir()
        (d / "submission.json").write_text(json.dumps({"some_other_field": 42}))

        scores = load_reference_ranking(tmp_path)
        assert "model_X" not in scores

    def test_handles_mean_val_bpb(self, tmp_path):
        d = tmp_path / "model_Y"
        d.mkdir()
        (d / "submission.json").write_text(
            json.dumps({"mean_val_bpb": 1.19})
        )

        scores = load_reference_ranking(tmp_path)
        assert scores["model_Y"] == pytest.approx(1.19)


# ===========================================================================
# provisional_val tests
# ===========================================================================

class TestProvisionalVal:
    def test_build_provisional_subset(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=1, n_val=1,
                                   tokens_per=10000)
        manifest = build_provisional_val_subset(
            data_dir, n_seqs=50, seq_len=100, seed=42,
        )
        assert len(manifest.seq_ids) == 50
        assert manifest.seq_len == 100
        assert manifest.metadata["is_provisional"] is True

    def test_deterministic(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=1, n_val=1,
                                   tokens_per=10000)
        a = build_provisional_val_subset(data_dir, n_seqs=50, seed=42)
        b = build_provisional_val_subset(data_dir, n_seqs=50, seed=42)
        assert a.seq_ids == b.seq_ids

    def test_saves_to_file(self, tmp_path):
        data_dir = _make_shard_dir(tmp_path, n_shards=1, n_val=1,
                                   tokens_per=5000)
        out = tmp_path / "prov_val.json"
        build_provisional_val_subset(data_dir, n_seqs=10, output_path=out)
        assert out.exists()

    def test_extract_val_bpb(self):
        log = "step:100 | val_loss:3.5 | val_bpb:5.05 | train_time:5000ms"
        assert extract_val_bpb_from_log(log) == pytest.approx(5.05)

    def test_extract_val_bpb_empty(self):
        assert extract_val_bpb_from_log("") is None

    def test_extract_post_quant_bpb(self):
        log = "final_int8_zlib_roundtrip val_bpb:5.23 val_loss:3.6"
        assert extract_post_quant_bpb_from_log(log) == pytest.approx(5.23)

    def test_extract_post_quant_bpb_missing(self):
        assert extract_post_quant_bpb_from_log("no quant here") is None


# ===========================================================================
# finalist_selection tests
# ===========================================================================

class TestEvaluateCandidate:
    def test_perfect_correlation(self):
        proxy = {"A": 1.0, "B": 2.0, "C": 3.0}
        ref = {"A": 1.0, "B": 2.0, "C": 3.0}
        ev = evaluate_candidate("test", proxy, ref)
        assert ev.spearman_rho == pytest.approx(1.0)
        assert ev.kendall_tau == pytest.approx(1.0)
        assert ev.pairwise_accuracy == pytest.approx(1.0)
        assert ev.top_1_agreement is True

    def test_inverse_correlation(self):
        proxy = {"A": 3.0, "B": 2.0, "C": 1.0}
        ref = {"A": 1.0, "B": 2.0, "C": 3.0}
        ev = evaluate_candidate("test", proxy, ref)
        assert ev.spearman_rho == pytest.approx(-1.0)
        assert ev.pairwise_accuracy == pytest.approx(0.0)

    def test_insufficient_models(self):
        proxy = {"A": 1.0}
        ref = {"A": 1.0}
        ev = evaluate_candidate("test", proxy, ref)
        assert len(ev.warnings) > 0

    def test_partial_overlap(self):
        proxy = {"A": 1.0, "B": 2.0, "X": 3.0}
        ref = {"A": 1.0, "B": 2.0, "Y": 3.0}
        ev = evaluate_candidate("test", proxy, ref)
        # Only A and B are common
        assert ev.n_models_compared == 2


class TestSelectFinalists:
    def test_selects_top_k(self):
        evals = [
            CandidateEvaluation(
                candidate_id=f"c{i}",
                n_models_compared=3,
                composite_score=float(i),
            )
            for i in range(5)
        ]
        finalists = select_finalists(evals, n_finalists=2)
        assert len(finalists) == 2
        assert finalists[0].candidate_id == "c4"  # highest composite
        assert finalists[1].candidate_id == "c3"

    def test_excludes_insufficient_models(self):
        evals = [
            CandidateEvaluation(
                candidate_id="good", n_models_compared=3, composite_score=0.9,
            ),
            CandidateEvaluation(
                candidate_id="bad", n_models_compared=1, composite_score=1.0,
            ),
        ]
        finalists = select_finalists(evals, n_finalists=2, min_models=2)
        assert len(finalists) == 1
        assert finalists[0].candidate_id == "good"

    def test_empty_input(self):
        assert select_finalists([], n_finalists=3) == []


class TestSelectionReport:
    def test_roundtrip(self, tmp_path):
        evals = [
            CandidateEvaluation(
                candidate_id="c1", n_models_compared=3,
                composite_score=0.8, family="contiguous",
            ),
        ]
        finalists = [evals[0]]
        report = build_selection_report(
            evals, finalists,
            reference_source="leaderboard",
            anchor_models=["A", "B", "C"],
        )
        assert report.n_finalists == 1
        assert report.n_candidates_evaluated == 1

        rp = tmp_path / "report.json"
        fp = tmp_path / "finalists.json"
        save_selection_report(report, rp, fp)

        assert rp.exists()
        assert fp.exists()

        loaded = load_finalists(fp)
        assert len(loaded) == 1
        assert loaded[0]["candidate_id"] == "c1"


class TestFormatEvaluationTable:
    def test_formats_without_error(self):
        evals = [
            CandidateEvaluation(
                candidate_id="c1", n_models_compared=3,
                spearman_rho=0.8, kendall_tau=0.7,
                pairwise_accuracy=0.85, top_1_agreement=True,
                composite_score=0.78,
            ),
            CandidateEvaluation(
                candidate_id="c2", n_models_compared=3,
                spearman_rho=0.6, kendall_tau=0.5,
                pairwise_accuracy=0.7, top_1_agreement=False,
                composite_score=0.6,
            ),
        ]
        text = format_evaluation_table(evals, finalists=[evals[0]])
        assert "c1" in text
        assert "FINALIST" in text

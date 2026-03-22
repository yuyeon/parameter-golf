"""
Tests for proxy_framework.metrics ranking fidelity functions.

Run with:
    python -m pytest tests/test_metrics.py -v
"""

import math

import pytest

from proxy_framework.metrics import (
    bootstrap_ci,
    format_report,
    kendall_tau,
    pairwise_accuracy,
    ranking_report,
    spearman_rho,
    top_1_agreement,
    top_k_overlap,
)


# ---------------------------------------------------------------------------
# spearman_rho
# ---------------------------------------------------------------------------

class TestSpearmanRho:
    def test_perfect_correlation(self):
        # Identical orderings -> rho = +1
        proxy = [1.0, 2.0, 3.0, 4.0, 5.0]
        ref = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert spearman_rho(proxy, ref) == pytest.approx(1.0)

    def test_perfect_anticorrelation(self):
        # Reversed orderings -> rho = -1
        proxy = [5.0, 4.0, 3.0, 2.0, 1.0]
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert spearman_rho(proxy, ref) == pytest.approx(-1.0)

    def test_no_correlation(self):
        # Specific permutation that gives rho = 0
        # ranks: proxy [1,2,3,4,5], ref [3,4,5,1,2]
        # d^2: (1-3)^2 + (2-4)^2 + (3-5)^2 + (4-1)^2 + (5-2)^2 = 4+4+4+9+9 = 30
        # rho = 1 - 6*30/(5*24) = 1 - 180/120 = 1 - 1.5 = -0.5
        proxy = [1.0, 2.0, 3.0, 4.0, 5.0]
        ref = [3.0, 4.0, 5.0, 1.0, 2.0]
        assert spearman_rho(proxy, ref) == pytest.approx(-0.5)

    def test_single_element(self):
        assert math.isnan(spearman_rho([1.0], [2.0]))

    def test_two_elements_same_order(self):
        assert spearman_rho([1.0, 2.0], [3.0, 4.0]) == pytest.approx(1.0)

    def test_two_elements_reversed(self):
        assert spearman_rho([2.0, 1.0], [3.0, 4.0]) == pytest.approx(-1.0)

    def test_ties_in_scores(self):
        # Ties should still produce a finite value
        proxy = [1.0, 1.0, 2.0]
        ref = [1.0, 2.0, 3.0]
        result = spearman_rho(proxy, ref)
        assert not math.isnan(result)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# kendall_tau
# ---------------------------------------------------------------------------

class TestKendallTau:
    def test_perfect_concordance(self):
        proxy = [1.0, 2.0, 3.0, 4.0]
        ref = [10.0, 20.0, 30.0, 40.0]
        assert kendall_tau(proxy, ref) == pytest.approx(1.0)

    def test_perfect_discordance(self):
        proxy = [4.0, 3.0, 2.0, 1.0]
        ref = [1.0, 2.0, 3.0, 4.0]
        assert kendall_tau(proxy, ref) == pytest.approx(-1.0)

    def test_partial_concordance(self):
        # proxy order: A < B < C, ref order: A < C < B
        # pairs: (A,B) concordant, (A,C) concordant, (B,C) discordant
        # tau = (2-1)/3 = 1/3
        proxy = [1.0, 2.0, 3.0]
        ref = [1.0, 3.0, 2.0]
        assert kendall_tau(proxy, ref) == pytest.approx(1.0 / 3.0)

    def test_single_element(self):
        assert math.isnan(kendall_tau([1.0], [2.0]))

    def test_all_ties(self):
        # All pairs tied -> denom = 0 -> nan
        proxy = [1.0, 1.0, 1.0]
        ref = [2.0, 2.0, 2.0]
        assert math.isnan(kendall_tau(proxy, ref))


# ---------------------------------------------------------------------------
# pairwise_accuracy
# ---------------------------------------------------------------------------

class TestPairwiseAccuracy:
    def test_perfect(self):
        proxy = [1.0, 2.0, 3.0]
        ref = [10.0, 20.0, 30.0]
        assert pairwise_accuracy(proxy, ref) == pytest.approx(1.0)

    def test_worst(self):
        proxy = [3.0, 2.0, 1.0]
        ref = [1.0, 2.0, 3.0]
        assert pairwise_accuracy(proxy, ref) == pytest.approx(0.0)

    def test_half(self):
        # proxy ranks: A=1, B=2, C=3
        # ref ranks:   A=1, B=3, C=2
        # pairs: (A,B) correct, (A,C) correct, (B,C) wrong -> 2/3
        proxy = [1.0, 2.0, 3.0]
        ref = [1.0, 3.0, 2.0]
        assert pairwise_accuracy(proxy, ref) == pytest.approx(2.0 / 3.0)

    def test_single_element(self):
        assert math.isnan(pairwise_accuracy([1.0], [2.0]))

    def test_tied_ref_pair_skipped(self):
        # If ref scores are tied for a pair, that pair is skipped
        proxy = [1.0, 2.0, 3.0]
        ref = [1.0, 1.0, 3.0]
        # Pairs: (0,1) ref tied -> skip, (0,2) correct, (1,2) correct -> 2/2
        assert pairwise_accuracy(proxy, ref) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# top_k_overlap
# ---------------------------------------------------------------------------

class TestTopKOverlap:
    def test_perfect_overlap(self):
        proxy = [1.0, 2.0, 3.0, 4.0, 5.0]
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert top_k_overlap(proxy, ref, k=3) == pytest.approx(1.0)

    def test_no_overlap(self):
        # proxy top-2: indices 0,1 (scores 1,2)
        # ref top-2:   indices 3,4 (scores 1,2)
        proxy = [1.0, 2.0, 5.0, 6.0, 7.0]
        ref = [7.0, 6.0, 5.0, 2.0, 1.0]
        assert top_k_overlap(proxy, ref, k=2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # proxy top-2: indices 0,1 (scores 1,2)
        # ref top-2:   indices 0,2 (scores 1,2)
        # intersection = {0}, union = {0,1,2}, jaccard = 1/3
        proxy = [1.0, 2.0, 3.0, 4.0]
        ref = [1.0, 3.0, 2.0, 4.0]
        assert top_k_overlap(proxy, ref, k=2) == pytest.approx(1.0 / 3.0)

    def test_k_larger_than_n(self):
        proxy = [1.0, 2.0]
        ref = [2.0, 1.0]
        # k gets clamped to n=2, all models in top-k -> overlap=1.0
        assert top_k_overlap(proxy, ref, k=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# top_1_agreement
# ---------------------------------------------------------------------------

class TestTop1Agreement:
    def test_agrees(self):
        proxy = [1.0, 5.0, 3.0]
        ref = [2.0, 8.0, 4.0]
        # Both say index 0 is best (lowest)
        assert top_1_agreement(proxy, ref) is True

    def test_disagrees(self):
        proxy = [5.0, 1.0, 3.0]
        ref = [1.0, 5.0, 3.0]
        assert top_1_agreement(proxy, ref) is False

    def test_empty(self):
        assert top_1_agreement([], []) is False


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_mean_of_constants(self):
        values = [5.0] * 100
        pt, lo, hi = bootstrap_ci(values, n_bootstrap=500)
        assert pt == pytest.approx(5.0)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_ci_contains_point(self):
        values = [float(i) for i in range(50)]
        pt, lo, hi = bootstrap_ci(values, n_bootstrap=1000)
        assert lo <= pt <= hi

    def test_ci_width_reasonable(self):
        values = [float(i) for i in range(50)]
        pt, lo, hi = bootstrap_ci(values, n_bootstrap=1000)
        # The mean is ~24.5, CI should be within a few units
        assert pt == pytest.approx(24.5)
        assert hi - lo < 15.0  # generous bound

    def test_custom_stat_fn(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        pt, lo, hi = bootstrap_ci(values, stat_fn=max, n_bootstrap=500)
        assert pt == 5.0
        assert lo <= pt <= hi

    def test_deterministic_with_seed(self):
        values = [float(i) for i in range(20)]
        r1 = bootstrap_ci(values, n_bootstrap=200, seed=123)
        r2 = bootstrap_ci(values, n_bootstrap=200, seed=123)
        assert r1 == r2


# ---------------------------------------------------------------------------
# ranking_report and format_report
# ---------------------------------------------------------------------------

class TestRankingReport:
    def test_basic_report(self):
        proxy = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        ref = {"A": 1.1, "B": 2.1, "C": 3.1, "D": 4.1}
        report = ranking_report(proxy, ref, k=2)

        assert report["n_models"] == 4
        assert report["spearman_rho"] == pytest.approx(1.0)
        assert report["kendall_tau"] == pytest.approx(1.0)
        assert report["pairwise_accuracy"] == pytest.approx(1.0)
        assert report["top_1_agreement"] is True
        assert report["top_k_overlap"] == pytest.approx(1.0)
        assert len(report["per_model"]) == 4

    def test_insufficient_models(self):
        report = ranking_report({"A": 1.0}, {"A": 1.0})
        assert "error" in report

    def test_partial_overlap(self):
        proxy = {"A": 1.0, "B": 2.0, "C": 3.0}
        ref = {"A": 1.1, "B": 2.1, "D": 4.1}
        report = ranking_report(proxy, ref, k=2)
        assert report["n_models"] == 2
        assert set(report["models"]) == {"A", "B"}

    def test_format_report_runs(self):
        proxy = {"A": 1.0, "B": 2.0, "C": 3.0}
        ref = {"A": 1.1, "B": 2.1, "C": 3.1}
        report = ranking_report(proxy, ref, k=2)
        text = format_report(report)
        assert "Spearman" in text
        assert "Kendall" in text

    def test_format_report_error(self):
        report = ranking_report({"A": 1.0}, {"A": 1.0})
        text = format_report(report)
        assert "Error" in text

    def test_per_model_rank_deltas(self):
        proxy = {"A": 3.0, "B": 1.0, "C": 2.0}
        ref = {"A": 1.0, "B": 2.0, "C": 3.0}
        report = ranking_report(proxy, ref, k=2)
        deltas = {m["name"]: m["rank_delta"] for m in report["per_model"]}
        # A: proxy_rank=3, ref_rank=1 -> delta=2
        assert deltas["A"] == 2
        # B: proxy_rank=1, ref_rank=2 -> delta=-1
        assert deltas["B"] == -1

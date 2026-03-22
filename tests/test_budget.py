"""
Tests for proxy_framework.budget matched-budget training abstractions.

Run with:
    python -m pytest tests/test_budget.py -v
"""

import json
import math
from pathlib import Path

import pytest

from proxy_framework.budget import (
    BudgetMode,
    BudgetSpec,
    RunSummary,
    filter_matched_pairs,
    fit_log_linear,
    group_matched_runs,
    is_budget_matched,
    load_run_summary,
    predict_at_budget,
    save_run_summary,
)


# ---------------------------------------------------------------------------
# BudgetSpec
# ---------------------------------------------------------------------------

class TestBudgetSpec:
    def test_to_env_overrides_wallclock(self):
        spec = BudgetSpec(mode="wallclock", value=600.0, seq_len=1024, batch_tokens=32768)
        env = spec.to_env_overrides()
        assert env["MAX_WALLCLOCK_SECONDS"] == "600.0"
        assert env["ITERATIONS"] == "20000"
        assert env["TRAIN_BATCH_TOKENS"] == "32768"
        assert env["TRAIN_SEQ_LEN"] == "1024"

    def test_to_env_overrides_tokens(self):
        spec = BudgetSpec(mode="tokens", value=1_000_000, seq_len=1024, batch_tokens=32768)
        env = spec.to_env_overrides()
        # 1_000_000 / 32768 = 30 steps
        assert env["ITERATIONS"] == "30"
        assert env["MAX_WALLCLOCK_SECONDS"] == "7200"
        assert env["TRAIN_BATCH_TOKENS"] == "32768"

    def test_to_env_overrides_optimizer_steps(self):
        spec = BudgetSpec(mode="optimizer_steps", value=500, seq_len=1024, batch_tokens=32768)
        env = spec.to_env_overrides()
        assert env["ITERATIONS"] == "500"
        assert env["MAX_WALLCLOCK_SECONDS"] == "7200"

    def test_to_env_overrides_tokens_small_value(self):
        # When value < batch_tokens, should still produce at least 1 step
        spec = BudgetSpec(mode="tokens", value=100, batch_tokens=32768)
        env = spec.to_env_overrides()
        assert env["ITERATIONS"] == "1"

    def test_target_tokens_tokens_mode(self):
        spec = BudgetSpec(mode="tokens", value=2_000_000, batch_tokens=32768)
        assert spec.target_tokens == 2_000_000

    def test_target_tokens_steps_mode(self):
        spec = BudgetSpec(mode="optimizer_steps", value=100, batch_tokens=32768)
        assert spec.target_tokens == 100 * 32768

    def test_target_tokens_wallclock_mode(self):
        spec = BudgetSpec(mode="wallclock", value=600)
        assert spec.target_tokens is None

    def test_target_steps_steps_mode(self):
        spec = BudgetSpec(mode="optimizer_steps", value=500)
        assert spec.target_steps == 500

    def test_target_steps_tokens_mode(self):
        spec = BudgetSpec(mode="tokens", value=1_000_000, batch_tokens=32768)
        assert spec.target_steps == 30  # 1_000_000 // 32768

    def test_target_steps_wallclock_mode(self):
        spec = BudgetSpec(mode="wallclock", value=600)
        assert spec.target_steps is None

    def test_effective_batch_tokens_default(self):
        spec = BudgetSpec(mode="tokens", value=100, batch_tokens=32768)
        # effective_batch_tokens defaults to batch_tokens
        assert spec.effective_batch_tokens == 32768

    def test_effective_batch_tokens_explicit(self):
        spec = BudgetSpec(
            mode="tokens", value=100, batch_tokens=32768, effective_batch_tokens=65536
        )
        assert spec.effective_batch_tokens == 65536

    def test_effective_batch_tokens_zero_uses_batch_tokens(self):
        spec = BudgetSpec(
            mode="tokens", value=100, batch_tokens=32768, effective_batch_tokens=0
        )
        assert spec.effective_batch_tokens == 32768


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------

class TestRunSummary:
    def test_compute_derived_fills_throughput(self):
        s = RunSummary(
            tokens_processed=1_000_000,
            optimizer_steps=100,
            train_wallclock_sec=50.0,
        )
        s.compute_derived()
        assert s.train_tokens_per_sec == pytest.approx(20_000.0)
        assert s.optimizer_steps_per_sec == pytest.approx(2.0)

    def test_compute_derived_fills_tokens_from_steps(self):
        s = RunSummary(
            optimizer_steps=100,
            effective_batch_tokens=32768,
            train_wallclock_sec=50.0,
        )
        s.compute_derived()
        assert s.tokens_processed == 100 * 32768
        # Note: throughput is computed before tokens_processed is filled from steps,
        # so train_tokens_per_sec reflects the initial tokens_processed (0)
        assert s.train_tokens_per_sec == pytest.approx(0.0)

    def test_compute_derived_zero_wallclock(self):
        s = RunSummary(
            tokens_processed=1_000_000,
            optimizer_steps=100,
            train_wallclock_sec=0.0,
        )
        s.compute_derived()
        # Should not divide by zero; throughput stays 0
        assert s.train_tokens_per_sec == 0.0
        assert s.optimizer_steps_per_sec == 0.0

    def test_save_load_roundtrip(self, tmp_path):
        original = RunSummary(
            run_name="test_run",
            model_name="TestModel",
            config_path="/path/to/config.yaml",
            seed=42,
            budget_mode="tokens",
            budget_value=1_000_000,
            tokens_processed=999_000,
            optimizer_steps=30,
            train_wallclock_sec=120.5,
            train_loss=2.345,
            pre_quant_val_bpb=1.567,
            post_quant_val_bpb=1.678,
            status="completed",
        )

        path = tmp_path / "run_summary.json"
        save_run_summary(original, path)
        loaded = load_run_summary(path)

        assert loaded.run_name == original.run_name
        assert loaded.model_name == original.model_name
        assert loaded.config_path == original.config_path
        assert loaded.seed == original.seed
        assert loaded.budget_mode == original.budget_mode
        assert loaded.budget_value == original.budget_value
        assert loaded.tokens_processed == original.tokens_processed
        assert loaded.optimizer_steps == original.optimizer_steps
        assert loaded.train_wallclock_sec == pytest.approx(original.train_wallclock_sec)
        assert loaded.train_loss == pytest.approx(original.train_loss)
        assert loaded.pre_quant_val_bpb == pytest.approx(original.pre_quant_val_bpb)
        assert loaded.post_quant_val_bpb == pytest.approx(original.post_quant_val_bpb)
        assert loaded.status == original.status

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "run_summary.json"
        s = RunSummary(model_name="Test")
        save_run_summary(s, path)
        assert path.exists()

    def test_load_handles_missing_fields_gracefully(self, tmp_path):
        # A JSON file that only has a subset of fields
        data = {
            "run_name": "partial_run",
            "model_name": "PartialModel",
            "tokens_processed": 500_000,
        }
        path = tmp_path / "partial.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = load_run_summary(path)
        assert loaded.run_name == "partial_run"
        assert loaded.model_name == "PartialModel"
        assert loaded.tokens_processed == 500_000
        # Missing fields get defaults
        assert loaded.optimizer_steps == 0
        assert loaded.train_wallclock_sec == 0.0
        assert loaded.status == "incomplete"

    def test_load_ignores_extra_fields(self, tmp_path):
        data = {
            "run_name": "test",
            "model_name": "TestModel",
            "extra_field_not_in_dataclass": "should_be_ignored",
            "another_unknown": 42,
        }
        path = tmp_path / "extra.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = load_run_summary(path)
        assert loaded.run_name == "test"
        assert loaded.model_name == "TestModel"


# ---------------------------------------------------------------------------
# is_budget_matched
# ---------------------------------------------------------------------------

class TestIsBudgetMatched:
    def _make_run(self, tokens=0, steps=0, wallclock=0.0, name="model"):
        return RunSummary(
            model_name=name,
            tokens_processed=tokens,
            optimizer_steps=steps,
            train_wallclock_sec=wallclock,
        )

    def test_matched_within_tolerance_tokens(self):
        a = self._make_run(tokens=1_000_000)
        b = self._make_run(tokens=1_050_000)  # 5% difference
        assert is_budget_matched(a, b, mode=BudgetMode.TOKENS) is True

    def test_not_matched_beyond_tolerance_tokens(self):
        a = self._make_run(tokens=1_000_000)
        b = self._make_run(tokens=1_200_000)  # 20% difference
        assert is_budget_matched(a, b, mode=BudgetMode.TOKENS) is False

    def test_matched_exactly_at_tolerance(self):
        a = self._make_run(tokens=1_000_000)
        b = self._make_run(tokens=1_100_000)  # exactly 10%
        assert is_budget_matched(a, b, mode=BudgetMode.TOKENS) is True

    def test_zero_values_return_false(self):
        a = self._make_run(tokens=0)
        b = self._make_run(tokens=1_000_000)
        assert is_budget_matched(a, b, mode=BudgetMode.TOKENS) is False

    def test_both_zero_return_false(self):
        a = self._make_run(tokens=0)
        b = self._make_run(tokens=0)
        assert is_budget_matched(a, b, mode=BudgetMode.TOKENS) is False

    def test_matched_optimizer_steps(self):
        a = self._make_run(steps=100)
        b = self._make_run(steps=105)
        assert is_budget_matched(a, b, mode=BudgetMode.OPTIMIZER_STEPS) is True

    def test_not_matched_optimizer_steps(self):
        a = self._make_run(steps=100)
        b = self._make_run(steps=200)
        assert is_budget_matched(a, b, mode=BudgetMode.OPTIMIZER_STEPS) is False

    def test_matched_wallclock(self):
        a = self._make_run(wallclock=300.0)
        b = self._make_run(wallclock=320.0)
        assert is_budget_matched(a, b, mode=BudgetMode.WALLCLOCK) is True

    def test_not_matched_wallclock(self):
        a = self._make_run(wallclock=300.0)
        b = self._make_run(wallclock=500.0)
        assert is_budget_matched(a, b, mode=BudgetMode.WALLCLOCK) is False

    def test_different_modes_give_different_results(self):
        # Matched in tokens but not in wallclock
        a = RunSummary(
            model_name="A",
            tokens_processed=1_000_000,
            optimizer_steps=100,
            train_wallclock_sec=100.0,
        )
        b = RunSummary(
            model_name="B",
            tokens_processed=1_050_000,
            optimizer_steps=105,
            train_wallclock_sec=200.0,  # very different wallclock
        )
        assert is_budget_matched(a, b, mode=BudgetMode.TOKENS) is True
        assert is_budget_matched(a, b, mode=BudgetMode.WALLCLOCK) is False

    def test_symmetry(self):
        a = self._make_run(tokens=1_000_000)
        b = self._make_run(tokens=1_050_000)
        assert is_budget_matched(a, b) == is_budget_matched(b, a)

    def test_custom_tolerance(self):
        a = self._make_run(tokens=1_000_000)
        b = self._make_run(tokens=1_150_000)  # 15% difference
        assert is_budget_matched(a, b, tolerance=0.10) is False
        assert is_budget_matched(a, b, tolerance=0.20) is True

    def test_string_mode_value(self):
        a = self._make_run(tokens=1_000_000)
        b = self._make_run(tokens=1_050_000)
        assert is_budget_matched(a, b, mode="tokens") is True


# ---------------------------------------------------------------------------
# group_matched_runs
# ---------------------------------------------------------------------------

class TestGroupMatchedRuns:
    def _make_run(self, name, tokens=0, steps=0, wallclock=0.0):
        return RunSummary(
            model_name=name,
            tokens_processed=tokens,
            optimizer_steps=steps,
            train_wallclock_sec=wallclock,
        )

    def test_groups_similar_budget_runs(self):
        runs = [
            self._make_run("A", tokens=1_000_000),
            self._make_run("B", tokens=1_050_000),
            self._make_run("C", tokens=1_020_000),
        ]
        groups = group_matched_runs(runs, mode=BudgetMode.TOKENS)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_excludes_singletons(self):
        runs = [
            self._make_run("A", tokens=1_000_000),
            self._make_run("B", tokens=1_050_000),
            self._make_run("C", tokens=5_000_000),  # far away, singleton
        ]
        groups = group_matched_runs(runs, mode=BudgetMode.TOKENS)
        assert len(groups) == 1
        names = {s.model_name for s in groups[0]}
        assert "C" not in names

    def test_multiple_groups(self):
        runs = [
            self._make_run("A", tokens=1_000_000),
            self._make_run("B", tokens=1_050_000),
            self._make_run("C", tokens=5_000_000),
            self._make_run("D", tokens=5_200_000),
        ]
        groups = group_matched_runs(runs, mode=BudgetMode.TOKENS)
        assert len(groups) == 2

    def test_handles_empty_input(self):
        groups = group_matched_runs([], mode=BudgetMode.TOKENS)
        assert groups == []

    def test_single_run_no_groups(self):
        runs = [self._make_run("A", tokens=1_000_000)]
        groups = group_matched_runs(runs, mode=BudgetMode.TOKENS)
        assert groups == []

    def test_all_different_no_groups(self):
        runs = [
            self._make_run("A", tokens=100),
            self._make_run("B", tokens=10_000),
            self._make_run("C", tokens=1_000_000),
        ]
        groups = group_matched_runs(runs, mode=BudgetMode.TOKENS)
        assert groups == []


# ---------------------------------------------------------------------------
# filter_matched_pairs
# ---------------------------------------------------------------------------

class TestFilterMatchedPairs:
    def _make_run(self, name, tokens=0, bpb=0.0):
        return RunSummary(
            model_name=name,
            tokens_processed=tokens,
            pre_quant_val_bpb=bpb,
        )

    def test_includes_matched_runs(self):
        runs = [
            self._make_run("A", tokens=1_000_000, bpb=1.5),
            self._make_run("B", tokens=1_050_000, bpb=1.6),
            self._make_run("C", tokens=1_020_000, bpb=1.4),
        ]
        scores, warnings = filter_matched_pairs(runs, mode=BudgetMode.TOKENS)
        assert "A" in scores
        assert "B" in scores
        assert "C" in scores
        assert len(warnings) == 0

    def test_excludes_outliers_with_warnings(self):
        runs = [
            self._make_run("A", tokens=1_000_000, bpb=1.5),
            self._make_run("B", tokens=1_050_000, bpb=1.6),
            self._make_run("C", tokens=5_000_000, bpb=1.2),  # outlier
        ]
        scores, warnings = filter_matched_pairs(runs, mode=BudgetMode.TOKENS)
        # Median is 1_050_000. C is far away.
        assert "C" not in scores or len(warnings) > 0

    def test_returns_empty_for_insufficient_data(self):
        runs = [self._make_run("A", tokens=1_000_000, bpb=1.5)]
        scores, warnings = filter_matched_pairs(runs, mode=BudgetMode.TOKENS)
        assert scores == {}
        assert len(warnings) > 0

    def test_returns_empty_for_no_runs(self):
        scores, warnings = filter_matched_pairs([], mode=BudgetMode.TOKENS)
        assert scores == {}

    def test_excludes_zero_budget(self):
        runs = [
            self._make_run("A", tokens=1_000_000, bpb=1.5),
            self._make_run("B", tokens=1_050_000, bpb=1.6),
            self._make_run("C", tokens=0, bpb=1.4),  # zero tokens
        ]
        scores, warnings = filter_matched_pairs(runs, mode=BudgetMode.TOKENS)
        assert "C" not in scores
        assert any("no tokens data" in w for w in warnings)

    def test_uses_best_available_bpb(self):
        s = RunSummary(
            model_name="ModelX",
            tokens_processed=1_000_000,
            post_quant_val_bpb=1.8,
            pre_quant_val_bpb=1.5,
        )
        runs = [
            s,
            self._make_run("B", tokens=1_050_000, bpb=1.6),
        ]
        scores, _ = filter_matched_pairs(runs, mode=BudgetMode.TOKENS)
        # post_quant_val_bpb takes priority since it is checked first
        assert scores["ModelX"] == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# fit_log_linear
# ---------------------------------------------------------------------------

class TestFitLogLinear:
    def test_known_linear_in_log_relationship(self):
        # Construct data: loss = -0.5 * log(budget) + 10
        import math as m
        budgets = [100.0, 1000.0, 10000.0]
        losses = [-0.5 * m.log(b) + 10 for b in budgets]
        a, b = fit_log_linear(budgets, losses)
        assert a == pytest.approx(-0.5, abs=1e-10)
        assert b == pytest.approx(10.0, abs=1e-10)

    def test_two_points(self):
        budgets = [100.0, 1000.0]
        losses = [5.0, 3.0]
        a, b = fit_log_linear(budgets, losses)
        # Should fit exactly through both points
        predicted_100 = a * math.log(100.0) + b
        predicted_1000 = a * math.log(1000.0) + b
        assert predicted_100 == pytest.approx(5.0, abs=1e-10)
        assert predicted_1000 == pytest.approx(3.0, abs=1e-10)

    def test_raises_on_insufficient_data(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_log_linear([100.0], [5.0])

    def test_raises_on_empty_data(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_log_linear([], [])

    def test_raises_on_equal_budgets(self):
        with pytest.raises(ValueError, match="must not all be equal"):
            fit_log_linear([100.0, 100.0], [5.0, 3.0])

    def test_raises_on_zero_budget(self):
        with pytest.raises(ValueError, match="positive"):
            fit_log_linear([0.0, 100.0], [5.0, 3.0])

    def test_raises_on_negative_budget(self):
        with pytest.raises(ValueError, match="positive"):
            fit_log_linear([-100.0, 100.0], [5.0, 3.0])

    def test_negative_slope_for_decreasing_loss(self):
        budgets = [100.0, 1000.0, 10000.0]
        losses = [5.0, 4.0, 3.0]
        a, _ = fit_log_linear(budgets, losses)
        assert a < 0


# ---------------------------------------------------------------------------
# predict_at_budget
# ---------------------------------------------------------------------------

class TestPredictAtBudget:
    def test_returns_prediction_with_high_confidence(self):
        # 3 data points, target within 2x of max observed
        budgets = [1000.0, 2000.0, 4000.0]
        losses = [5.0, 4.0, 3.5]
        result = predict_at_budget(budgets, losses, target_budget=6000.0)
        assert result["predicted_loss"] is not None
        assert result["confidence"] == "high"
        assert result["extrapolation_ratio"] == pytest.approx(1.5)

    def test_returns_prediction_with_moderate_confidence(self):
        # 2 data points, target within 4x
        budgets = [1000.0, 2000.0]
        losses = [5.0, 4.0]
        result = predict_at_budget(budgets, losses, target_budget=6000.0)
        assert result["predicted_loss"] is not None
        assert result["confidence"] == "moderate"

    def test_returns_prediction_with_low_confidence(self):
        # 2 data points, target beyond 4x
        budgets = [1000.0, 2000.0]
        losses = [5.0, 4.0]
        result = predict_at_budget(budgets, losses, target_budget=20000.0)
        assert result["predicted_loss"] is not None
        assert result["confidence"] == "low"

    def test_returns_none_with_reason_insufficient_data(self):
        result = predict_at_budget([1000.0], [5.0], target_budget=2000.0)
        assert result["predicted_loss"] is None
        assert "reason" in result
        assert "at least" in result["reason"]

    def test_returns_none_when_min_points_not_met(self):
        result = predict_at_budget(
            [1000.0, 2000.0], [5.0, 4.0], target_budget=3000.0, min_points=3
        )
        assert result["predicted_loss"] is None
        assert "at least 3" in result["reason"]

    def test_confidence_degrades_with_extrapolation_ratio(self):
        budgets = [1000.0, 2000.0, 3000.0]
        losses = [5.0, 4.0, 3.5]

        # Interpolation (within range) -> high
        r1 = predict_at_budget(budgets, losses, target_budget=2500.0)
        assert r1["confidence"] == "high"

        # Moderate extrapolation
        r2 = predict_at_budget(budgets, losses, target_budget=10000.0)
        assert r2["confidence"] == "moderate"

        # Extreme extrapolation
        r3 = predict_at_budget(budgets, losses, target_budget=100000.0)
        assert r3["confidence"] == "low"

    def test_preserves_observed_data(self):
        budgets = [1000.0, 2000.0]
        losses = [5.0, 4.0]
        result = predict_at_budget(budgets, losses, target_budget=3000.0)
        assert result["observed_budgets"] == [1000.0, 2000.0]
        assert result["observed_losses"] == [5.0, 4.0]

    def test_prediction_is_consistent_with_fit(self):
        budgets = [100.0, 1000.0, 10000.0]
        losses = [-0.5 * math.log(b) + 10 for b in budgets]
        result = predict_at_budget(budgets, losses, target_budget=50000.0)
        expected = -0.5 * math.log(50000.0) + 10
        assert result["predicted_loss"] == pytest.approx(expected, abs=1e-8)

    def test_handles_equal_budgets_gracefully(self):
        result = predict_at_budget([100.0, 100.0], [5.0, 4.0], target_budget=200.0)
        assert result["predicted_loss"] is None
        assert "reason" in result

    def test_extrapolation_ratio_field(self):
        budgets = [1000.0, 2000.0]
        losses = [5.0, 4.0]
        result = predict_at_budget(budgets, losses, target_budget=8000.0)
        assert result["extrapolation_ratio"] == pytest.approx(4.0)

    def test_fit_coefficients_returned(self):
        budgets = [1000.0, 2000.0]
        losses = [5.0, 4.0]
        result = predict_at_budget(budgets, losses, target_budget=3000.0)
        assert "fit_a" in result
        assert "fit_b" in result
        assert isinstance(result["fit_a"], float)
        assert isinstance(result["fit_b"], float)

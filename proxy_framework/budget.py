"""
Matched-budget training abstractions.

The key insight: comparing models after "5 minutes on a 3080" is misleading
because different architectures have different step times.  Model A might
process 500 steps (16M tokens) while model B processes 300 steps (10M tokens)
in the same wall-clock.  Matched-budget comparison controls for this.

Three budget modes:
  wallclock        -- stop at a wall-clock limit (practical constraint)
  tokens           -- stop after processing N training tokens (fairest comparison)
  optimizer_steps  -- stop after N optimizer updates

DataDecide (arXiv:2504.11393) shows that rankings at matched compute budgets
transfer more reliably than rankings at arbitrary budgets.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence


class BudgetMode(str, Enum):
    WALLCLOCK = "wallclock"
    TOKENS = "tokens"
    OPTIMIZER_STEPS = "optimizer_steps"


@dataclass
class BudgetSpec:
    """Specifies the training budget for a run."""
    mode: str  # BudgetMode value
    value: float  # seconds / token count / step count
    seq_len: int = 1024
    batch_tokens: int = 32768
    # These are computed from the training script's defaults
    grad_accum_steps: int = 8
    effective_batch_tokens: int = 0  # batch_tokens (since single GPU)

    def __post_init__(self):
        if not self.effective_batch_tokens:
            self.effective_batch_tokens = self.batch_tokens

    def to_env_overrides(self) -> dict[str, str]:
        """Convert budget spec to train_gpt.py environment variable overrides."""
        env = {
            "TRAIN_BATCH_TOKENS": str(self.batch_tokens),
            "TRAIN_SEQ_LEN": str(self.seq_len),
        }
        mode = BudgetMode(self.mode)
        if mode == BudgetMode.WALLCLOCK:
            env["MAX_WALLCLOCK_SECONDS"] = str(self.value)
            env["ITERATIONS"] = "20000"  # won't be reached
        elif mode == BudgetMode.TOKENS:
            target_steps = max(1, int(self.value / self.batch_tokens))
            env["ITERATIONS"] = str(target_steps)
            env["MAX_WALLCLOCK_SECONDS"] = "7200"  # 2h safety valve
        elif mode == BudgetMode.OPTIMIZER_STEPS:
            env["ITERATIONS"] = str(int(self.value))
            env["MAX_WALLCLOCK_SECONDS"] = "7200"
        return env

    @property
    def target_tokens(self) -> int | None:
        """Target tokens if mode is tokens, else estimated from steps."""
        mode = BudgetMode(self.mode)
        if mode == BudgetMode.TOKENS:
            return int(self.value)
        if mode == BudgetMode.OPTIMIZER_STEPS:
            return int(self.value) * self.batch_tokens
        return None

    @property
    def target_steps(self) -> int | None:
        mode = BudgetMode(self.mode)
        if mode == BudgetMode.OPTIMIZER_STEPS:
            return int(self.value)
        if mode == BudgetMode.TOKENS:
            return max(1, int(self.value / self.batch_tokens))
        return None


@dataclass
class RunSummary:
    """Standardized accounting for every training/eval run."""

    # Identity
    run_name: str = ""
    model_name: str = ""
    config_path: str = ""
    seed: int = 0
    git_commit: str = ""

    # Budget
    budget_mode: str = ""
    budget_value: float = 0
    target_seq_len: int = 1024
    microbatch_seqs: int = 0
    grad_accum_steps: int = 8
    effective_batch_tokens: int = 0

    # Training progress
    tokens_processed: int = 0
    optimizer_steps: int = 0
    train_wallclock_sec: float = 0
    train_tokens_per_sec: float = 0
    optimizer_steps_per_sec: float = 0
    train_loss: float = 0
    budget_exhausted: str = ""  # "wallclock" | "iterations" | "incomplete"

    # Evaluation
    eval_wallclock_sec: float = 0
    eval_mode: str = ""  # "proxy_val_tune" | "proxy_val_audit" | "full_val" | ...
    eval_seq_len: int = 1024
    eval_stride: int = 0
    pre_quant_val_bpb: float = 0
    post_quant_val_bpb: float = 0
    proxy_val_tune_bpb: float = 0
    proxy_val_long_bpb: float = 0
    proxy_val_audit_bpb: float = 0
    artifact_bytes: int = 0

    # Memory
    peak_vram_allocated_gb: float = 0
    peak_vram_reserved_gb: float = 0

    # Status
    status: str = "incomplete"  # "completed" | "failed" | "memory_guard_triggered"
    failure_reason: str = ""

    def compute_derived(self):
        """Fill derived fields from primary data."""
        if self.train_wallclock_sec > 0:
            self.train_tokens_per_sec = self.tokens_processed / self.train_wallclock_sec
            self.optimizer_steps_per_sec = self.optimizer_steps / self.train_wallclock_sec
        if self.optimizer_steps > 0 and self.effective_batch_tokens > 0:
            self.tokens_processed = self.optimizer_steps * self.effective_batch_tokens

    @staticmethod
    def _get_git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return ""


def save_run_summary(summary: RunSummary, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(summary), f, indent=2)


def load_run_summary(path: Path | str) -> RunSummary:
    with open(path) as f:
        d = json.load(f)
    return RunSummary(**{k: v for k, v in d.items() if k in RunSummary.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Matched-budget comparison
# ---------------------------------------------------------------------------

MATCH_TOLERANCE = 0.10  # 10% tolerance for "matched" budgets


def is_budget_matched(
    a: RunSummary,
    b: RunSummary,
    mode: str | BudgetMode = BudgetMode.TOKENS,
    tolerance: float = MATCH_TOLERANCE,
) -> bool:
    """Check if two runs have matched budgets under the given mode."""
    mode = BudgetMode(mode)
    if mode == BudgetMode.TOKENS:
        va, vb = a.tokens_processed, b.tokens_processed
    elif mode == BudgetMode.OPTIMIZER_STEPS:
        va, vb = a.optimizer_steps, b.optimizer_steps
    elif mode == BudgetMode.WALLCLOCK:
        va, vb = a.train_wallclock_sec, b.train_wallclock_sec
    else:
        return False
    if va == 0 or vb == 0:
        return False
    ratio = max(va, vb) / min(va, vb)
    return ratio <= 1.0 + tolerance


def group_matched_runs(
    summaries: list[RunSummary],
    mode: str | BudgetMode = BudgetMode.TOKENS,
    tolerance: float = MATCH_TOLERANCE,
) -> list[list[RunSummary]]:
    """Group runs into clusters where all pairs are budget-matched.

    Returns list of groups, each group containing runs that can be
    fairly compared under the given budget mode.
    """
    mode = BudgetMode(mode)
    if not summaries:
        return []

    def _budget_val(s: RunSummary) -> float:
        if mode == BudgetMode.TOKENS:
            return s.tokens_processed
        elif mode == BudgetMode.OPTIMIZER_STEPS:
            return s.optimizer_steps
        return s.train_wallclock_sec

    # Sort by budget value, then greedily assign to groups
    sorted_runs = sorted(summaries, key=_budget_val)
    groups: list[list[RunSummary]] = []

    for run in sorted_runs:
        placed = False
        for group in groups:
            # Check if run matches all members of the group
            if all(is_budget_matched(run, g, mode, tolerance) for g in group):
                group.append(run)
                placed = True
                break
        if not placed:
            groups.append([run])

    return [g for g in groups if len(g) >= 2]


def filter_matched_pairs(
    summaries: list[RunSummary],
    mode: str | BudgetMode = BudgetMode.TOKENS,
    tolerance: float = MATCH_TOLERANCE,
) -> tuple[dict[str, float], list[str]]:
    """Extract scores from budget-matched runs for ranking comparison.

    Returns (scores_dict, warnings) where scores_dict maps model_name to
    the primary BPB score, and warnings lists any unmatched runs that were
    excluded.
    """
    mode = BudgetMode(mode)
    if len(summaries) < 2:
        return {}, ["need at least 2 runs"]

    # Find the median budget value and filter to runs within tolerance
    def _val(s):
        if mode == BudgetMode.TOKENS:
            return s.tokens_processed
        elif mode == BudgetMode.OPTIMIZER_STEPS:
            return s.optimizer_steps
        return s.train_wallclock_sec

    vals = sorted(_val(s) for s in summaries if _val(s) > 0)
    if not vals:
        return {}, ["no runs with budget data"]
    median_val = vals[len(vals) // 2]

    scores = {}
    warnings = []
    for s in summaries:
        v = _val(s)
        if v == 0:
            warnings.append(f"{s.model_name}: no {mode.value} data, excluded")
            continue
        ratio = max(v, median_val) / min(v, median_val)
        if ratio > 1.0 + tolerance:
            warnings.append(
                f"{s.model_name}: {mode.value}={v} vs median={median_val} "
                f"(ratio={ratio:.2f}), excluded from matched comparison"
            )
            continue
        # Use the best available BPB score
        bpb = (s.post_quant_val_bpb or s.pre_quant_val_bpb or
               s.proxy_val_tune_bpb or s.train_loss)
        if bpb and bpb > 0:
            scores[s.model_name] = bpb

    return scores, warnings


# ---------------------------------------------------------------------------
# Predicted quality at target budget (simple log-linear extrapolation)
# ---------------------------------------------------------------------------

def fit_log_linear(budgets: Sequence[float], losses: Sequence[float]) -> tuple[float, float]:
    """Fit loss = a * log(budget) + b.  Returns (a, b).

    Simple and robust.  Assumes monotonically decreasing loss with budget
    (more compute = lower loss).  Uses least-squares on log(budget) vs loss.
    """
    n = len(budgets)
    if n < 2:
        raise ValueError("need at least 2 data points")
    log_b = [math.log(b) for b in budgets if b > 0]
    if len(log_b) != n:
        raise ValueError("all budgets must be positive")
    mean_x = sum(log_b) / n
    mean_y = sum(losses) / n
    ss_xx = sum((x - mean_x) ** 2 for x in log_b)
    if ss_xx == 0:
        raise ValueError("budgets must not all be equal")
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_b, losses))
    a = ss_xy / ss_xx
    b = mean_y - a * mean_x
    return a, b


def predict_at_budget(
    observed_budgets: Sequence[float],
    observed_losses: Sequence[float],
    target_budget: float,
    min_points: int = 2,
) -> dict:
    """Predict loss at a target budget from observed (budget, loss) pairs.

    Returns dict with:
        predicted_loss: the extrapolated value
        observed_budgets: input budgets
        observed_losses: input losses
        fit_a, fit_b: log-linear coefficients
        extrapolation_ratio: target_budget / max(observed_budgets)
        confidence: "high" / "moderate" / "low" based on data and ratio

    Caveats:
        - Assumes log-linear relationship (loss = a*log(budget) + b)
        - Only produces predictions when enough data points exist
        - Extrapolation beyond 4x observed budget is flagged as low confidence
        - Raw observed results are always preserved alongside predictions
    """
    if len(observed_budgets) < min_points:
        return {
            "predicted_loss": None,
            "reason": f"need at least {min_points} observed budgets, got {len(observed_budgets)}",
            "observed_budgets": list(observed_budgets),
            "observed_losses": list(observed_losses),
        }

    try:
        a, b = fit_log_linear(observed_budgets, observed_losses)
    except ValueError as e:
        return {
            "predicted_loss": None,
            "reason": str(e),
            "observed_budgets": list(observed_budgets),
            "observed_losses": list(observed_losses),
        }

    predicted = a * math.log(target_budget) + b
    max_obs = max(observed_budgets)
    ratio = target_budget / max_obs

    if ratio <= 2.0 and len(observed_budgets) >= 3:
        confidence = "high"
    elif ratio <= 4.0 and len(observed_budgets) >= 2:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "predicted_loss": predicted,
        "observed_budgets": list(observed_budgets),
        "observed_losses": list(observed_losses),
        "fit_a": a,
        "fit_b": b,
        "target_budget": target_budget,
        "extrapolation_ratio": ratio,
        "confidence": confidence,
    }

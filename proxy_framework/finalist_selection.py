"""
Finalist selection for train subset candidates.

After sweeping multiple candidate train subsets on anchor models,
this module ranks candidates by ranking-fidelity metrics and selects
the top finalists.

"Best" means: the candidate whose induced model rankings most closely
match the higher-fidelity reference ranking.  It does NOT mean lowest
absolute loss or highest average document quality.

The most relevant train subset is the one whose cheap local training
runs induce model rankings that best match higher-fidelity rankings
under the target competition setting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from proxy_framework.metrics import (
    kendall_tau,
    pairwise_accuracy,
    ranking_report,
    spearman_rho,
    top_1_agreement,
    top_k_overlap,
)


# ---------------------------------------------------------------------------
# Per-candidate evaluation
# ---------------------------------------------------------------------------

@dataclass
class CandidateEvaluation:
    """Ranking-fidelity evaluation of one train subset candidate."""

    candidate_id: str
    family: str = ""
    shard_ids: list[int] = field(default_factory=list)
    n_models_compared: int = 0
    # Core ranking-fidelity metrics
    spearman_rho: float = float("nan")
    kendall_tau: float = float("nan")
    pairwise_accuracy: float = float("nan")
    top_1_agreement: bool = False
    top_k_overlap: float = float("nan")
    top_k: int = 3
    # Composite score for sorting
    composite_score: float = float("nan")
    # Per-model detail
    proxy_scores: dict[str, float] = field(default_factory=dict)
    ref_scores: dict[str, float] = field(default_factory=dict)
    # Status
    n_successful_runs: int = 0
    n_failed_runs: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_candidate(
    candidate_id: str,
    proxy_scores: dict[str, float],
    ref_scores: dict[str, float],
    family: str = "",
    shard_ids: list[int] | None = None,
    top_k: int = 3,
    weights: dict[str, float] | None = None,
) -> CandidateEvaluation:
    """Evaluate one candidate's ranking fidelity against a reference.

    Args:
        candidate_id: Unique identifier for this candidate.
        proxy_scores: ``{model_name: proxy_bpb}`` from training on
            this candidate's shards.
        ref_scores: ``{model_name: reference_bpb}`` from higher-fidelity
            evaluation (e.g. official leaderboard).
        family: Candidate family name.
        shard_ids: Shard indices used by this candidate.
        top_k: K for top-k overlap.
        weights: Optional metric weights for composite score.
            Defaults to equal weighting of spearman, kendall, pairwise.

    Returns:
        CandidateEvaluation with all metrics filled in.
    """
    if weights is None:
        weights = {
            "spearman_rho": 1.0,
            "kendall_tau": 1.0,
            "pairwise_accuracy": 1.0,
        }

    common = sorted(set(proxy_scores) & set(ref_scores))
    ev = CandidateEvaluation(
        candidate_id=candidate_id,
        family=family,
        shard_ids=shard_ids or [],
        n_models_compared=len(common),
        top_k=top_k,
        proxy_scores=dict(proxy_scores),
        ref_scores=dict(ref_scores),
    )

    if len(common) < 2:
        ev.warnings.append(
            f"Only {len(common)} common models; need >= 2 for ranking metrics"
        )
        return ev

    p = [proxy_scores[m] for m in common]
    r = [ref_scores[m] for m in common]

    ev.spearman_rho = spearman_rho(p, r)
    ev.kendall_tau = kendall_tau(p, r)
    ev.pairwise_accuracy = pairwise_accuracy(p, r)
    ev.top_1_agreement = top_1_agreement(p, r)
    ev.top_k_overlap = top_k_overlap(p, r, min(top_k, len(common)))

    # Composite score: weighted sum (higher = better)
    total_weight = sum(weights.values())
    if total_weight > 0:
        score = 0.0
        if "spearman_rho" in weights:
            score += weights["spearman_rho"] * ev.spearman_rho
        if "kendall_tau" in weights:
            score += weights["kendall_tau"] * ev.kendall_tau
        if "pairwise_accuracy" in weights:
            score += weights["pairwise_accuracy"] * ev.pairwise_accuracy
        ev.composite_score = score / total_weight

    return ev


# ---------------------------------------------------------------------------
# Finalist selection
# ---------------------------------------------------------------------------

def select_finalists(
    evaluations: list[CandidateEvaluation],
    n_finalists: int = 3,
    min_models: int = 2,
) -> list[CandidateEvaluation]:
    """Select the top train subset finalists by composite score.

    Candidates with fewer than ``min_models`` compared are excluded.
    Returns up to ``n_finalists`` candidates, sorted best-first.
    """
    eligible = [
        ev for ev in evaluations
        if ev.n_models_compared >= min_models
        and not (ev.composite_score != ev.composite_score)  # not NaN
    ]
    ranked = sorted(eligible, key=lambda ev: ev.composite_score, reverse=True)
    return ranked[:n_finalists]


# ---------------------------------------------------------------------------
# Report I/O
# ---------------------------------------------------------------------------

@dataclass
class SelectionReport:
    """Full report from the train-subset selection process."""

    n_candidates_evaluated: int = 0
    n_anchor_models: int = 0
    n_finalists: int = 0
    reference_source: str = ""     # e.g. "leaderboard" or "full_val"
    metric_weights: dict[str, float] = field(default_factory=dict)
    finalists: list[dict] = field(default_factory=list)
    all_evaluations: list[dict] = field(default_factory=list)
    anchor_models: list[str] = field(default_factory=list)
    methodology: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_selection_report(
    evaluations: list[CandidateEvaluation],
    finalists: list[CandidateEvaluation],
    reference_source: str = "leaderboard",
    metric_weights: dict[str, float] | None = None,
    anchor_models: list[str] | None = None,
) -> SelectionReport:
    """Build a complete selection report."""
    return SelectionReport(
        n_candidates_evaluated=len(evaluations),
        n_anchor_models=len(anchor_models) if anchor_models else 0,
        n_finalists=len(finalists),
        reference_source=reference_source,
        metric_weights=metric_weights or {},
        finalists=[f.to_dict() for f in finalists],
        all_evaluations=[e.to_dict() for e in evaluations],
        anchor_models=anchor_models or [],
        methodology=(
            "The most relevant train subset is the one whose cheap local "
            "training runs induce model rankings that best match higher-"
            "fidelity rankings under the target competition setting. "
            "Candidates are ranked by a composite score of Spearman rho, "
            "Kendall tau, and pairwise accuracy."
        ),
    )


def save_selection_report(
    report: SelectionReport,
    report_path: str | Path,
    finalists_path: str | Path | None = None,
) -> None:
    """Save the selection report and optionally a finalists manifest."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    if finalists_path and report.finalists:
        finalists_path = Path(finalists_path)
        finalists_path.parent.mkdir(parents=True, exist_ok=True)
        finalists_data = {
            "n_finalists": len(report.finalists),
            "reference_source": report.reference_source,
            "methodology": report.methodology,
            "finalists": [
                {
                    "rank": i + 1,
                    "candidate_id": f["candidate_id"],
                    "family": f["family"],
                    "shard_ids": f["shard_ids"],
                    "composite_score": f["composite_score"],
                    "spearman_rho": f["spearman_rho"],
                    "kendall_tau": f["kendall_tau"],
                    "pairwise_accuracy": f["pairwise_accuracy"],
                }
                for i, f in enumerate(report.finalists)
            ],
        }
        with open(finalists_path, "w") as f:
            json.dump(finalists_data, f, indent=2)


def load_finalists(path: str | Path) -> list[dict]:
    """Load finalists from a finalists.json manifest."""
    with open(path) as f:
        data = json.load(f)
    return data.get("finalists", [])


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_evaluation_table(
    evaluations: list[CandidateEvaluation],
    finalists: list[CandidateEvaluation] | None = None,
) -> str:
    """Pretty-print evaluation results as a ranked table."""
    finalist_ids = {f.candidate_id for f in finalists} if finalists else set()

    ranked = sorted(
        [e for e in evaluations
         if e.n_models_compared >= 2
         and not (e.composite_score != e.composite_score)],
        key=lambda e: e.composite_score,
        reverse=True,
    )

    lines = []
    lines.append(f"{'Rank':<6} {'Candidate':<35} {'Spear':>7} {'Kend':>7} "
                 f"{'Pair%':>7} {'Top1':>5} {'Comp':>7} {'Note':>8}")
    lines.append(f"{'-'*6} {'-'*35} {'-'*7} {'-'*7} "
                 f"{'-'*7} {'-'*5} {'-'*7} {'-'*8}")

    for i, ev in enumerate(ranked, 1):
        note = "FINALIST" if ev.candidate_id in finalist_ids else ""
        t1 = "Y" if ev.top_1_agreement else "N"
        lines.append(
            f"#{i:<5} {ev.candidate_id:<35} "
            f"{ev.spearman_rho:>+7.3f} {ev.kendall_tau:>+7.3f} "
            f"{ev.pairwise_accuracy:>7.3f} {t1:>5} "
            f"{ev.composite_score:>7.3f} {note:>8}"
        )

    return "\n".join(lines)

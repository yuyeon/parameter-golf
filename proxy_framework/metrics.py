"""
Ranking fidelity metrics for proxy evaluation.

Measures how well a proxy ranking preserves the ordering from a reference
(higher-fidelity) ranking.  All metrics assume lower score = better model
(as with BPB / loss).

Metric selection informed by DataDecide (arXiv:2504.11393) which uses
rank correlations to validate small-scale predictors of large-scale
outcomes, and SparseEval (arXiv:2602.07909) which uses pairwise accuracy
and top-k overlap to evaluate sparse evaluation sets.
"""

from __future__ import annotations
import math
import random
from typing import Callable, Sequence


def _ranks(scores: Sequence[float]) -> list[int]:
    """Convert scores to ranks (1 = best = lowest score)."""
    indexed = sorted(enumerate(scores), key=lambda t: t[1])
    ranks = [0] * len(scores)
    for rank, (idx, _) in enumerate(indexed):
        ranks[idx] = rank + 1
    return ranks


def spearman_rho(proxy: Sequence[float], ref: Sequence[float]) -> float:
    """Spearman rank correlation.  +1 = perfect agreement."""
    n = len(proxy)
    if n < 2:
        return float("nan")
    rp, rr = _ranks(proxy), _ranks(ref)
    d2 = sum((a - b) ** 2 for a, b in zip(rp, rr))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def kendall_tau(proxy: Sequence[float], ref: Sequence[float]) -> float:
    """Kendall's tau-b rank correlation.  +1 = perfect agreement."""
    n = len(proxy)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dp = proxy[i] - proxy[j]
            dr = ref[i] - ref[j]
            if dp * dr > 0:
                concordant += 1
            elif dp * dr < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def pairwise_accuracy(proxy: Sequence[float], ref: Sequence[float]) -> float:
    """Fraction of model pairs whose relative order is preserved."""
    n = len(proxy)
    if n < 2:
        return float("nan")
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dr = ref[i] - ref[j]
            if dr == 0:
                continue
            dp = proxy[i] - proxy[j]
            total += 1
            if dp * dr > 0:
                correct += 1
    return correct / total if total > 0 else float("nan")


def top_1_agreement(proxy: Sequence[float], ref: Sequence[float]) -> bool:
    """Whether the proxy and reference agree on the best model."""
    if len(proxy) < 1:
        return False
    return _ranks(proxy).index(1) == _ranks(ref).index(1)


def top_k_overlap(
    proxy: Sequence[float], ref: Sequence[float], k: int
) -> float:
    """Jaccard overlap of the top-k models in proxy vs reference."""
    if len(proxy) < k:
        k = len(proxy)
    if k < 1:
        return float("nan")
    rp, rr = _ranks(proxy), _ranks(ref)
    top_proxy = {i for i, r in enumerate(rp) if r <= k}
    top_ref = {i for i, r in enumerate(rr) if r <= k}
    return len(top_proxy & top_ref) / len(top_proxy | top_ref)


def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[Sequence[float]], float] | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic.

    Returns (point_estimate, ci_lower, ci_upper).
    Default stat_fn is the mean.
    """
    if stat_fn is None:
        stat_fn = lambda xs: sum(xs) / len(xs)
    rng = random.Random(seed)
    n = len(values)
    vals = list(values)
    point = stat_fn(vals)
    boots = []
    for _ in range(n_bootstrap):
        sample = [vals[rng.randint(0, n - 1)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    alpha = 1.0 - ci
    lo = boots[int(n_bootstrap * alpha / 2)]
    hi = boots[int(n_bootstrap * (1 - alpha / 2))]
    return point, lo, hi


def ranking_report(
    proxy_scores: dict[str, float],
    ref_scores: dict[str, float],
    k: int = 3,
) -> dict:
    """Comprehensive ranking comparison between proxy and reference.

    Args:
        proxy_scores: {model_name: proxy_bpb}
        ref_scores: {model_name: reference_bpb}
        k: for top-k overlap

    Returns dict with all metrics.
    """
    common = sorted(set(proxy_scores) & set(ref_scores))
    if len(common) < 2:
        return {"error": "need at least 2 common models", "n": len(common)}

    p = [proxy_scores[n] for n in common]
    r = [ref_scores[n] for n in common]

    report = {
        "n_models": len(common),
        "models": common,
        "spearman_rho": spearman_rho(p, r),
        "kendall_tau": kendall_tau(p, r),
        "pairwise_accuracy": pairwise_accuracy(p, r),
        "top_1_agreement": top_1_agreement(p, r),
        "top_k_overlap": top_k_overlap(p, r, k),
        "top_k": k,
    }

    # Per-model detail
    rp, rr = _ranks(p), _ranks(r)
    report["per_model"] = [
        {
            "name": common[i],
            "proxy_score": p[i],
            "ref_score": r[i],
            "proxy_rank": rp[i],
            "ref_rank": rr[i],
            "rank_delta": rp[i] - rr[i],
        }
        for i in range(len(common))
    ]

    return report


def format_report(report: dict) -> str:
    """Pretty-print a ranking report."""
    if "error" in report:
        return f"Error: {report['error']}"
    lines = [
        f"Ranking comparison ({report['n_models']} models):",
        f"  Spearman rho:       {report['spearman_rho']:+.4f}",
        f"  Kendall tau:        {report['kendall_tau']:+.4f}",
        f"  Pairwise accuracy:  {report['pairwise_accuracy']:.4f}",
        f"  Top-1 agreement:    {report['top_1_agreement']}",
        f"  Top-{report['top_k']} overlap:      {report['top_k_overlap']:.4f}",
        "",
        f"  {'Model':<45} {'Proxy':>8} {'Ref':>8} {'P-Rank':>7} {'R-Rank':>7}",
        f"  {'-'*45} {'-'*8} {'-'*8} {'-'*7} {'-'*7}",
    ]
    for m in sorted(report["per_model"], key=lambda x: x["ref_rank"]):
        lines.append(
            f"  {m['name']:<45} {m['proxy_score']:>8.4f} {m['ref_score']:>8.4f} "
            f"#{m['proxy_rank']:<6} #{m['ref_rank']:<6}"
        )
    return "\n".join(lines)

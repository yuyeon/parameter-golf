#!/usr/bin/env python3
"""
Analyze proxy ranking fidelity against official leaderboard scores.

Loads results from multiple screening runs (JSON files produced by
run_local_screen.py) and compares them against official submission scores
in records/track_10min_16mb/*/submission.json.

Computes:
  - Spearman rho, Kendall tau, pairwise accuracy
  - Top-1 agreement and top-k overlap
  - Bootstrap confidence intervals on all ranking metrics
  - Comparison across multiple proxy configurations

Methodology follows DataDecide (arXiv:2504.11393) in validating that
small-scale proxy rankings are predictive of full-scale outcomes.

Usage:
    python -m scripts.analyze_proxy_rankings \\
        --results-dir proxy_results/ \\
        --k 3 \\
        --bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proxy_framework.budget import (
    BudgetMode,
    RunSummary,
    filter_matched_pairs,
    load_run_summary,
    predict_at_budget,
)
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
# Loading helpers
# ---------------------------------------------------------------------------

def load_leaderboard(records_dir: Path) -> dict[str, float]:
    """Load official BPB scores from records/track_10min_16mb/*/submission.json.

    Returns:
        {submission_name: val_bpb}
    """
    scores: dict[str, float] = {}
    pattern = records_dir / "track_10min_16mb" / "*" / "submission.json"
    for sub_json in sorted(records_dir.glob("track_10min_16mb/*/submission.json")):
        with open(sub_json) as f:
            data = json.load(f)
        # Use the directory name as the canonical submission name
        name = sub_json.parent.name
        bpb = data.get("val_bpb")
        if bpb is not None:
            scores[name] = float(bpb)
    return scores


def load_screening_results(results_dir: Path) -> dict[str, dict[str, float]]:
    """Load proxy screening results from a directory of JSON files.

    Each JSON is expected to have structure:
        {
            "config_name": "...",
            "scores": { "submission_name": proxy_bpb, ... }
        }
    or a flat mapping:
        { "submission_name": proxy_bpb, ... }

    Returns:
        {config_name: {submission_name: proxy_bpb}}
    """
    configs: dict[str, dict[str, float]] = {}
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {results_dir}")

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        # Determine config name and scores dict
        if "config_name" in data and "scores" in data:
            config_name = data["config_name"]
            scores = data["scores"]
        elif "name" in data and "scores" in data:
            config_name = data["name"]
            scores = data["scores"]
        else:
            # Treat the whole file as a flat {name: score} mapping
            config_name = jf.stem
            scores = {k: v for k, v in data.items() if isinstance(v, (int, float))}

        if scores:
            configs[config_name] = {k: float(v) for k, v in scores.items()}

    return configs


def load_run_summaries(results_dir: Path) -> list[RunSummary]:
    """Scan results_dir for run_summary*.json files and load them.

    Looks for files matching:
      - run_summary.json
      - run_summary_seed*.json
    in results_dir and all immediate subdirectories.
    """
    summaries: list[RunSummary] = []
    patterns = ["run_summary.json", "run_summary_seed*.json"]
    for pattern in patterns:
        # Search top-level
        for p in sorted(results_dir.glob(pattern)):
            summaries.append(load_run_summary(p))
        # Search one level down
        for p in sorted(results_dir.glob(f"*/{pattern}")):
            summaries.append(load_run_summary(p))
    return summaries


# ---------------------------------------------------------------------------
# Matched-budget comparison
# ---------------------------------------------------------------------------

def format_budget_report(
    summaries: list[RunSummary],
    mode: BudgetMode,
) -> tuple[str, dict[str, float], list[str]]:
    """Format a matched-budget comparison report."""
    lines = []

    # Show per-model budget info
    lines.append(f"Budget comparison (mode: {mode.value}):")
    lines.append(
        f"  {'Model':<35} {'Tokens':>12} {'Steps':>8} {'Wallclock':>10} {'Status':>10}"
    )
    lines.append(
        f"  {'-'*35} {'-'*12} {'-'*8} {'-'*10} {'-'*10}"
    )
    for s in sorted(summaries, key=lambda x: x.model_name):
        lines.append(
            f"  {s.model_name:<35} {s.tokens_processed:>12,} {s.optimizer_steps:>8} "
            f"{s.train_wallclock_sec:>10.1f} {'matched' if True else 'excluded':>10}"
        )
    lines.append("")

    # Run filter_matched_pairs
    scores, warnings = filter_matched_pairs(summaries, mode=mode)

    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append("")

    matched_count = len(scores)
    total_count = len(summaries)
    lines.append(f"Matched runs: {matched_count}/{total_count}")
    lines.append("")

    return "\n".join(lines), scores, warnings


def format_calibration_table(
    summaries: list[RunSummary],
    ref_scores: dict[str, float],
    k: int = 3,
) -> str:
    """Run comparison under all three budget modes and report which gives best correlation."""
    modes = [BudgetMode.TOKENS, BudgetMode.OPTIMIZER_STEPS, BudgetMode.WALLCLOCK]

    lines = [
        "Budget calibration (all modes):",
        f"  {'Budget Mode':<22} {'Spearman':>9} {'Kendall':>9} {'Pairwise':>9} {'Matched':>10}",
        f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9} {'-'*10}",
    ]

    best_mode = None
    best_spearman = -2.0

    for mode in modes:
        scores, _ = filter_matched_pairs(summaries, mode=mode)
        total = len(summaries)
        matched = len(scores)

        common = sorted(set(scores) & set(ref_scores))
        if len(common) < 2:
            lines.append(
                f"  {mode.value:<22} {'N/A':>9} {'N/A':>9} {'N/A':>9} {matched}/{total:>7}"
            )
            continue

        p = [scores[m] for m in common]
        r = [ref_scores[m] for m in common]

        sp = spearman_rho(p, r)
        kt = kendall_tau(p, r)
        pa = pairwise_accuracy(p, r)

        lines.append(
            f"  {mode.value:<22} {sp:>+9.4f} {kt:>+9.4f} {pa:>9.4f} {matched}/{total:>7}"
        )

        if sp > best_spearman:
            best_spearman = sp
            best_mode = mode

    lines.append("")
    if best_mode is not None:
        lines.append(f"Best budget mode by Spearman: {best_mode.value} ({best_spearman:+.4f})")
    lines.append("")

    return "\n".join(lines)


def format_prediction_table(
    summaries: list[RunSummary],
    target_budget: float,
    mode: BudgetMode = BudgetMode.TOKENS,
) -> str:
    """For models with multiple runs at different budgets, predict BPB at target budget."""
    # Group summaries by model_name
    by_model: dict[str, list[RunSummary]] = {}
    for s in summaries:
        by_model.setdefault(s.model_name, []).append(s)

    def _budget_val(s: RunSummary) -> float:
        if mode == BudgetMode.TOKENS:
            return float(s.tokens_processed)
        elif mode == BudgetMode.OPTIMIZER_STEPS:
            return float(s.optimizer_steps)
        return s.train_wallclock_sec

    lines = [
        f"Predictions at target budget = {target_budget:,.0f} {mode.value}:",
        f"  {'Model':<35} {'Observed BPB':>13} {'Predicted BPB':>14} {'Confidence':>12}",
        f"  {'-'*35} {'-'*13} {'-'*14} {'-'*12}",
    ]

    for model_name in sorted(by_model):
        runs = by_model[model_name]
        # Get observed BPB (best available)
        best_bpb = None
        for s in runs:
            bpb = (s.post_quant_val_bpb or s.pre_quant_val_bpb or
                   s.proxy_val_tune_bpb or s.train_loss)
            if bpb and bpb > 0:
                if best_bpb is None or bpb < best_bpb:
                    best_bpb = bpb

        observed_str = f"{best_bpb:.4f}" if best_bpb else "N/A"

        # Collect (budget, loss) pairs for prediction
        budgets = []
        losses = []
        for s in runs:
            b = _budget_val(s)
            bpb = (s.post_quant_val_bpb or s.pre_quant_val_bpb or
                   s.proxy_val_tune_bpb or s.train_loss)
            if b > 0 and bpb and bpb > 0:
                budgets.append(b)
                losses.append(bpb)

        if len(budgets) >= 2:
            result = predict_at_budget(budgets, losses, target_budget)
            if result["predicted_loss"] is not None:
                lines.append(
                    f"  {model_name:<35} {observed_str:>13} "
                    f"{result['predicted_loss']:>14.4f} {result['confidence']:>12}"
                )
            else:
                lines.append(
                    f"  {model_name:<35} {observed_str:>13} {'N/A':>14} "
                    f"{'(' + result.get('reason', '') + ')':>12}"
                )
        else:
            lines.append(
                f"  {model_name:<35} {observed_str:>13} {'N/A':>14} "
                f"{'(need 2+ runs)':>12}"
            )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bootstrap metric wrappers
# ---------------------------------------------------------------------------

def _bootstrap_ranking_metric(
    proxy_scores: dict[str, float],
    ref_scores: dict[str, float],
    metric_fn,
    n_bootstrap: int = 1000,
    k: int = 3,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap a ranking metric by resampling the set of models.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    import random

    common = sorted(set(proxy_scores) & set(ref_scores))
    n = len(common)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))

    p = [proxy_scores[m] for m in common]
    r = [ref_scores[m] for m in common]

    point = metric_fn(p, r) if "k" not in metric_fn.__code__.co_varnames else metric_fn(p, r, k)

    rng = random.Random(seed)
    boot_vals = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        bp = [p[i] for i in indices]
        br = [r[i] for i in indices]
        try:
            if "k" in metric_fn.__code__.co_varnames:
                val = metric_fn(bp, br, k)
            else:
                val = metric_fn(bp, br)
        except Exception:
            continue
        if isinstance(val, bool):
            val = float(val)
        if not math.isnan(val):
            boot_vals.append(val)

    if not boot_vals:
        return (point if not isinstance(point, bool) else float(point), float("nan"), float("nan"))

    boot_vals.sort()
    alpha = 0.05
    lo = boot_vals[int(len(boot_vals) * alpha / 2)]
    hi = boot_vals[int(len(boot_vals) * (1 - alpha / 2))]
    point_val = point if not isinstance(point, bool) else float(point)
    return (point_val, lo, hi)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    proxy_scores: dict[str, float],
    ref_scores: dict[str, float],
    config_name: str,
    k: int = 3,
    n_bootstrap: int = 1000,
) -> str:
    """Generate a formatted report for one proxy config vs the leaderboard."""

    # Basic ranking report
    report = ranking_report(proxy_scores, ref_scores, k=k)
    if "error" in report:
        return f"[{config_name}] {report['error']}"

    lines = [
        f"{'=' * 72}",
        f"Proxy Configuration: {config_name}",
        f"{'=' * 72}",
        "",
        format_report(report),
        "",
    ]

    # Bootstrap confidence intervals
    lines.append("Bootstrap confidence intervals (95%, n={:,}):".format(n_bootstrap))
    lines.append(f"  {'Metric':<22} {'Point':>9} {'95% CI':>22}")
    lines.append(f"  {'-'*22} {'-'*9} {'-'*22}")

    metrics = [
        ("Spearman rho", spearman_rho),
        ("Kendall tau", kendall_tau),
        ("Pairwise accuracy", pairwise_accuracy),
    ]
    for metric_name, metric_fn in metrics:
        pt, lo, hi = _bootstrap_ranking_metric(
            proxy_scores, ref_scores, metric_fn,
            n_bootstrap=n_bootstrap, k=k,
        )
        lines.append(f"  {metric_name:<22} {pt:>+9.4f} [{lo:>+9.4f}, {hi:>+9.4f}]")

    # Top-k overlap bootstrap
    pt, lo, hi = _bootstrap_ranking_metric(
        proxy_scores, ref_scores, top_k_overlap,
        n_bootstrap=n_bootstrap, k=k,
    )
    lines.append(f"  {'Top-' + str(k) + ' overlap':<22} {pt:>9.4f} [{lo:>9.4f}, {hi:>9.4f}]")

    # Top-1 agreement (just point estimate, it is binary)
    common = sorted(set(proxy_scores) & set(ref_scores))
    p = [proxy_scores[m] for m in common]
    r = [ref_scores[m] for m in common]
    t1 = top_1_agreement(p, r)
    lines.append(f"  {'Top-1 agreement':<22} {'True' if t1 else 'False':>9}")

    lines.append("")
    return "\n".join(lines)


def compare_configs(
    all_proxy: dict[str, dict[str, float]],
    ref_scores: dict[str, float],
    k: int = 3,
) -> str:
    """Generate a comparison table across multiple proxy configurations."""
    if not all_proxy:
        return "No proxy configurations to compare."

    header = (
        f"  {'Config':<35} {'n':>3} {'Spearman':>9} {'Kendall':>9} "
        f"{'Pairwise':>9} {'Top-1':>6} {'Top-' + str(k):>6}"
    )
    sep = f"  {'-'*35} {'-'*3} {'-'*9} {'-'*9} {'-'*9} {'-'*6} {'-'*6}"

    lines = [
        "Comparison across proxy configurations:",
        header,
        sep,
    ]

    for config_name, proxy_scores in sorted(all_proxy.items()):
        common = sorted(set(proxy_scores) & set(ref_scores))
        if len(common) < 2:
            lines.append(f"  {config_name:<35} {len(common):>3} {'(insufficient data)':>50}")
            continue
        p = [proxy_scores[m] for m in common]
        r = [ref_scores[m] for m in common]

        sp = spearman_rho(p, r)
        kt = kendall_tau(p, r)
        pa = pairwise_accuracy(p, r)
        t1 = top_1_agreement(p, r)
        tk = top_k_overlap(p, r, k)

        lines.append(
            f"  {config_name:<35} {len(common):>3} {sp:>+9.4f} {kt:>+9.4f} "
            f"{pa:>9.4f} {'Y' if t1 else 'N':>6} {tk:>6.3f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze proxy ranking fidelity against official leaderboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing proxy screening result JSON files.",
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=PROJECT_ROOT / "records",
        help="Directory containing official records (default: records/).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="k for top-k overlap metric (default: 3).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals (default: 1000).",
    )
    parser.add_argument(
        "--budget-mode",
        type=str,
        choices=["wallclock", "tokens", "optimizer_steps"],
        default="tokens",
        help="Budget mode for matched comparison (default: tokens).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run comparison under all three budget modes and report best.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Predict BPB at target budget for models with multiple runs.",
    )
    parser.add_argument(
        "--target-budget",
        type=float,
        default=None,
        help="Target budget value for --predict (required with --predict).",
    )
    args = parser.parse_args()

    # Load reference scores
    ref_scores = load_leaderboard(args.records_dir)
    if not ref_scores:
        print("ERROR: No leaderboard scores found in", args.records_dir, file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(ref_scores)} official leaderboard entries.\n")

    # Load proxy results
    all_proxy = load_screening_results(args.results_dir)
    print(f"Loaded {len(all_proxy)} proxy configuration(s).\n")

    # Per-config detailed reports
    for config_name, proxy_scores in sorted(all_proxy.items()):
        report = generate_report(
            proxy_scores, ref_scores, config_name,
            k=args.k, n_bootstrap=args.bootstrap,
        )
        print(report)

    # Summary comparison table
    if len(all_proxy) > 1:
        print(compare_configs(all_proxy, ref_scores, k=args.k))
        print()

    # -----------------------------------------------------------------------
    # Matched-budget analysis (requires RunSummary files)
    # -----------------------------------------------------------------------
    summaries = load_run_summaries(args.results_dir)
    if summaries:
        budget_mode = BudgetMode(args.budget_mode)
        print(f"\n{'=' * 72}")
        print(f"Matched-Budget Analysis ({len(summaries)} run summaries loaded)")
        print(f"{'=' * 72}\n")

        report_text, matched_scores, warnings = format_budget_report(
            summaries, budget_mode
        )
        print(report_text)

        # Compute ranking metrics on matched runs only
        if len(matched_scores) >= 2:
            common = sorted(set(matched_scores) & set(ref_scores))
            if len(common) >= 2:
                p = [matched_scores[m] for m in common]
                r = [ref_scores[m] for m in common]

                print("Ranking metrics (matched runs only):")
                print(f"  Spearman rho:       {spearman_rho(p, r):+.4f}")
                print(f"  Kendall tau:        {kendall_tau(p, r):+.4f}")
                print(f"  Pairwise accuracy:  {pairwise_accuracy(p, r):.4f}")
                print()
            else:
                print("Insufficient overlap between matched runs and leaderboard.\n")

        if warnings:
            print("Excluded runs:")
            for w in warnings:
                print(f"  WARNING: {w}")
            print()

        # Calibration mode
        if args.calibrate:
            print(format_calibration_table(summaries, ref_scores, k=args.k))

        # Prediction mode
        if args.predict:
            if args.target_budget is None:
                print(
                    "ERROR: --target-budget is required when using --predict.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(format_prediction_table(
                summaries, args.target_budget, mode=budget_mode
            ))
    else:
        if args.calibrate or args.predict:
            print(
                "WARNING: No run_summary*.json files found; "
                "--calibrate and --predict require RunSummary data.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()

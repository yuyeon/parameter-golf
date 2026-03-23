#!/usr/bin/env python3
"""
Run multiple candidate submissions through proxy screening in parallel.

Automatically detects the GPU and calculates how many experiments can
run concurrently.  Each experiment uses the same VRAM cap and budget as
a single RTX 3080 screening run, so results transfer directly.

Examples:

    # Screen all submissions in a track directory
    python scripts/run_parallel_screen.py \
        --records-dir records/track_10min_16mb \
        --output-dir parallel_results/batch_001

    # Screen specific submissions
    python scripts/run_parallel_screen.py \
        --scripts records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
                  records/track_10min_16mb/2026-03-18_LowerLR/train_gpt.py \
        --output-dir parallel_results/compare_001

    # Use a manifest file with custom configs per experiment
    python scripts/run_parallel_screen.py \
        --manifest experiments.json \
        --output-dir parallel_results/manifest_001

    # Override parallelism (e.g. to leave headroom for other work)
    python scripts/run_parallel_screen.py \
        --records-dir records/track_10min_16mb \
        --max-workers 2 \
        --output-dir parallel_results/batch_002

    # Multi-seed screening (runs each submission N times with different seeds)
    python scripts/run_parallel_screen.py \
        --records-dir records/track_10min_16mb \
        --seeds 1337 42 7 \
        --output-dir parallel_results/multiseed_001
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from proxy_framework.parallel import (
    ExperimentSpec,
    ParallelRunner,
    discover_submissions,
    load_experiment_manifest,
    save_experiment_manifest,
)
from proxy_framework.vram_guard import detect_gpu, max_parallel_workers
from proxy_framework.metrics import ranking_report, format_report
from proxy_framework.budget import load_run_summary


def _build_specs_from_args(args: argparse.Namespace) -> list[ExperimentSpec]:
    """Build experiment specs from CLI arguments."""
    specs: list[ExperimentSpec] = []

    if args.manifest:
        specs = load_experiment_manifest(args.manifest)
    elif args.records_dir:
        specs = discover_submissions(args.records_dir)
    elif args.scripts:
        for script_path in args.scripts:
            path = Path(script_path).resolve()
            name = path.parent.name
            specs.append(ExperimentSpec(name=name, script=str(path)))
    else:
        print("[ERROR] Provide --records-dir, --scripts, or --manifest")
        sys.exit(1)

    if not specs:
        print("[ERROR] No experiments found")
        sys.exit(1)

    # Apply CLI overrides to all specs
    for spec in specs:
        spec.budget_mode = args.budget_mode
        spec.budget_value = args.budget_value
        spec.batch_tokens = args.batch_tokens
        spec.max_gb = args.max_gb

    # Expand seeds: create one spec per (submission, seed) pair
    seeds = args.seeds
    if len(seeds) > 1:
        expanded = []
        for spec in specs:
            for seed in seeds:
                s = ExperimentSpec(
                    name=f"{spec.name}_seed{seed}",
                    script=spec.script,
                    seed=seed,
                    budget_mode=spec.budget_mode,
                    budget_value=spec.budget_value,
                    batch_tokens=spec.batch_tokens,
                    max_gb=spec.max_gb,
                    extra_env=dict(spec.extra_env),
                )
                expanded.append(s)
        specs = expanded
    else:
        for spec in specs:
            spec.seed = seeds[0]

    return specs


def _compare_with_leaderboard(
    results_dir: Path,
    records_dir: str | None,
) -> None:
    """If leaderboard data is available, compute ranking fidelity."""
    if not records_dir:
        return

    records_path = Path(records_dir)

    # Load official scores
    ref_scores: dict[str, float] = {}
    for sub_json in records_path.glob("*/submission.json"):
        try:
            with open(sub_json) as f:
                data = json.load(f)
            name = sub_json.parent.name
            bpb = data.get("val_bpb") or data.get("bpb")
            if bpb:
                ref_scores[name] = float(bpb)
        except Exception:
            continue

    if not ref_scores:
        return

    # Load proxy scores from RunSummary files
    proxy_scores: dict[str, float] = {}
    for summary_path in results_dir.rglob("run_summary.json"):
        try:
            summary = load_run_summary(summary_path)
            # Strip _seedN suffix if present
            model_name = summary.model_name
            bpb = (summary.post_quant_val_bpb or summary.pre_quant_val_bpb
                   or summary.proxy_val_tune_bpb or summary.train_loss)
            if bpb and bpb > 0:
                # If multiple seeds, keep the best
                if model_name not in proxy_scores or bpb < proxy_scores[model_name]:
                    proxy_scores[model_name] = bpb
        except Exception:
            continue

    if len(set(proxy_scores) & set(ref_scores)) < 2:
        print("\n[ranking] Not enough common models for ranking comparison")
        return

    report = ranking_report(proxy_scores, ref_scores)
    print(f"\n{'='*80}")
    print("RANKING FIDELITY vs LEADERBOARD")
    print(f"{'='*80}")
    print(format_report(report))


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel proxy screening experiments."
    )

    # Input sources (mutually preferred, not exclusive)
    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "--records-dir",
        help="Directory containing submission dirs (each with train_gpt.py)"
    )
    input_group.add_argument(
        "--scripts", nargs="+",
        help="Explicit list of train_gpt.py paths to screen"
    )
    input_group.add_argument(
        "--manifest",
        help="Path to experiment manifest JSON (overrides --records-dir/--scripts)"
    )

    # Budget & training
    budget_group = parser.add_argument_group("budget")
    budget_group.add_argument(
        "--budget-mode",
        choices=["wallclock", "tokens", "optimizer_steps"],
        default="tokens",
        help="Budget mode (default: tokens)"
    )
    budget_group.add_argument(
        "--budget-value", type=float, default=16_000_000,
        help="Budget value (default: 16M tokens)"
    )
    budget_group.add_argument(
        "--batch-tokens", type=int, default=32768,
        help="TRAIN_BATCH_TOKENS per step (default: 32768)"
    )
    budget_group.add_argument(
        "--seeds", type=int, nargs="+", default=[1337],
        help="Seeds to run (default: 1337). Multiple seeds = multi-seed screening"
    )

    # GPU & parallelism
    gpu_group = parser.add_argument_group("GPU & parallelism")
    gpu_group.add_argument(
        "--max-workers", type=int, default=None,
        help="Max parallel workers (auto-detected from GPU if omitted)"
    )
    gpu_group.add_argument(
        "--max-gb", type=float, default=10.0,
        help="VRAM cap per experiment in GB (default: 10.0, matches 3080)"
    )

    # Output
    parser.add_argument(
        "--output-dir", default="parallel_results",
        help="Output directory for all results"
    )
    parser.add_argument(
        "--conda-env", default="parameter-golf",
        help="Conda/micromamba env name"
    )
    parser.add_argument(
        "--save-manifest", default=None,
        help="Save experiment manifest to this path (useful for reproducibility)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without actually running"
    )

    args = parser.parse_args()

    specs = _build_specs_from_args(args)

    # Optionally save manifest for reproducibility
    if args.save_manifest:
        save_experiment_manifest(specs, args.save_manifest)
        print(f"[info] Manifest saved to {args.save_manifest}")

    if args.dry_run:
        gpu = detect_gpu()
        workers = args.max_workers or max_parallel_workers(gpu.total_vram_gb, args.max_gb)
        print(f"GPU: {gpu.name} ({gpu.total_vram_gb:.1f} GB)")
        print(f"Workers: {workers}")
        print(f"Experiments: {len(specs)}")
        print()
        for i, spec in enumerate(specs, 1):
            print(f"  {i:3}. {spec.name}")
            print(f"       script: {spec.script}")
            print(f"       seed={spec.seed}, budget={spec.budget_mode}={spec.budget_value}")
        return

    # Run experiments
    runner = ParallelRunner(
        output_dir=args.output_dir,
        per_worker_gb=args.max_gb,
        max_workers=args.max_workers,
        conda_env=args.conda_env,
        repo_root=REPO_ROOT,
    )

    wall_t0 = time.time()
    results = runner.run(specs)
    total_wall = time.time() - wall_t0

    runner.print_summary(results)

    # Timing comparison
    n_successful = sum(1 for r in results if r.success)
    total_experiment_time = sum(r.wallclock_sec for r in results)
    if n_successful > 0:
        print(f"Total wall-clock time: {total_wall:.0f}s")
        print(f"Sum of experiment times: {total_experiment_time:.0f}s")
        if total_experiment_time > 0:
            speedup = total_experiment_time / total_wall
            print(f"Effective speedup: {speedup:.1f}x")

    # Compare with leaderboard if possible
    _compare_with_leaderboard(
        Path(args.output_dir),
        args.records_dir,
    )


if __name__ == "__main__":
    main()

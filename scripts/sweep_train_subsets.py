#!/usr/bin/env python3
"""
Sweep train subset candidates on anchor models.

For each (candidate_train_subset × anchor_model) pair:
  1. Train the anchor under matched budget using only that candidate's shards
  2. Evaluate using the provisional validation lens (training script's
     built-in full-val evaluation by default)
  3. Compute ranking-fidelity metrics vs a reference ranking

Outputs:
  - Per-run summaries in  <output-dir>/<candidate>/<model>/
  - Aggregated comparison report
  - Finalist selection report

The provisional validation lens used here is NOT the final proxy_val.
Final proxy_val subsets are built only after train-subset finalists are
chosen (Stage 4 of the staged procedure).

Example:
    # Generate candidates first
    python scripts/generate_train_candidates.py

    # Sweep all candidates × all submissions
    python scripts/sweep_train_subsets.py \\
        --candidates-dir artifacts/train_subsets \\
        --records-dir records/track_10min_16mb \\
        --output-dir artifacts/train_subset_sweep \\
        --budget-mode tokens --budget-value 16000000

    # Sweep with specific anchor models
    python scripts/sweep_train_subsets.py \\
        --candidates-dir artifacts/train_subsets \\
        --anchor-scripts records/.../train_gpt.py records/.../train_gpt.py \\
        --output-dir artifacts/train_subset_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from proxy_framework.budget import BudgetSpec, RunSummary, save_run_summary
from proxy_framework.finalist_selection import (
    CandidateEvaluation,
    SelectionReport,
    build_selection_report,
    evaluate_candidate,
    format_evaluation_table,
    save_selection_report,
    select_finalists,
)
from proxy_framework.provisional_val import (
    ProvisionalValMode,
    collect_sweep_scores,
    extract_post_quant_bpb_from_log,
    extract_val_bpb_from_log,
)
from proxy_framework.train_subset_search import (
    TrainSubsetCandidate,
    cleanup_shard_dir,
    load_candidates,
    load_reference_ranking,
    prepare_shard_dir,
)
from proxy_framework.vram_guard import (
    detect_gpu,
    max_parallel_workers,
    memory_fraction_for_worker,
)


# ---------------------------------------------------------------------------
# Log parsing (reused from parallel.py)
# ---------------------------------------------------------------------------

def _parse_metrics(log_text: str) -> dict:
    """Extract key metrics from a training log."""
    metrics: dict = {}
    for line in reversed(log_text.splitlines()):
        if "val_bpb:" in line and "step:" in line:
            m = re.search(r"val_loss:([\d.]+)", line)
            if m:
                metrics["val_loss"] = float(m.group(1))
            m = re.search(r"val_bpb:([\d.]+)", line)
            if m:
                metrics["val_bpb"] = float(m.group(1))
            m = re.search(r"step:(\d+)", line)
            if m:
                metrics["steps_completed"] = int(m.group(1))
            break
    for line in reversed(log_text.splitlines()):
        if "train_loss:" in line:
            m = re.search(r"train_loss:([\d.]+)", line)
            if m:
                metrics["train_loss"] = float(m.group(1))
            break
    for line in log_text.splitlines():
        if "peak memory allocated:" in line:
            m = re.search(r"peak memory allocated:\s*(\d+)\s*MiB", line)
            if m:
                metrics["peak_vram_mib"] = int(m.group(1))
    for line in log_text.splitlines():
        if "final_int8_zlib_roundtrip " in line:
            m = re.search(r"val_bpb:([\d.]+)", line)
            if m:
                metrics["post_quant_bpb"] = float(m.group(1))
    for line in log_text.splitlines():
        if "Serialized model int8+zlib:" in line:
            m = re.search(r"int8\+zlib:\s*(\d+)\s*bytes", line)
            if m:
                metrics["artifact_bytes"] = int(m.group(1))
    return metrics


# ---------------------------------------------------------------------------
# Memory-limiting wrapper
# ---------------------------------------------------------------------------

_WRAPPER_TEMPLATE = """\
import os, sys, torch
torch.cuda.set_per_process_memory_fraction({fraction:.3f})
exec(open(sys.argv[1]).read())
"""


def _write_wrapper(dest: Path, fraction: float) -> Path:
    w = dest / "_mem_wrapper.py"
    w.write_text(_WRAPPER_TEMPLATE.format(fraction=fraction))
    return w


# ---------------------------------------------------------------------------
# Single (candidate, model) run
# ---------------------------------------------------------------------------

def _run_one(
    candidate: TrainSubsetCandidate,
    script_path: str,
    model_name: str,
    output_dir: Path,
    source_data_dir: Path,
    budget_spec: BudgetSpec,
    mem_fraction: float,
    conda_env: str,
    seed: int,
) -> dict:
    """Train one model on one candidate's shards and return metrics."""
    run_dir = output_dir / candidate.candidate_id / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create isolated shard directory
    work_dir = Path(tempfile.mkdtemp(prefix=f"sweep_{candidate.candidate_id}_{model_name}_"))
    shard_dir = prepare_shard_dir(
        shard_ids=candidate.shard_ids,
        source_data_dir=source_data_dir,
        work_dir=work_dir,
    )

    tokenizer_path = (source_data_dir.parent.parent / "tokenizers" / "fineweb_1024_bpe.model").resolve()
    # Fall back: try the standard location
    if not tokenizer_path.exists():
        tokenizer_path = (REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve()

    budget_env = budget_spec.to_env_overrides()

    env = os.environ.copy()
    env.update({
        "TRAIN_BATCH_TOKENS": str(budget_spec.batch_tokens),
        "WARMUP_STEPS": "5",
        "WARMDOWN_ITERS": "150",
        "VAL_LOSS_EVERY": "100",
        "TRAIN_LOG_EVERY": "25",
        "VAL_BATCH_SIZE": "32768",
        "PYTHONUNBUFFERED": "1",
        "DATA_PATH": str(shard_dir),
        "TOKENIZER_PATH": str(tokenizer_path),
        "SEED": str(seed),
    })
    env.update(budget_env)

    script_abs = str(Path(script_path).resolve())

    # Build command: use conda_env wrapper if available, else plain python
    if conda_env == "none":
        python_cmd = ["python", "-u"]
    else:
        import shutil as _shutil
        _mamba = _shutil.which("micromamba")
        if _mamba:
            python_cmd = ["micromamba", "run", "-n", conda_env, "python", "-u"]
        else:
            python_cmd = ["python", "-u"]

    if mem_fraction < 1.0:
        wrapper = _write_wrapper(work_dir, mem_fraction)
        cmd = [*python_cmd, str(wrapper), script_abs]
    else:
        cmd = [*python_cmd, script_abs]

    log_path = run_dir / "train.log"
    wall_t0 = time.time()
    proc = None
    try:
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(work_dir),
            )
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
            proc.wait()
    except Exception as e:
        wall_elapsed = time.time() - wall_t0
        cleanup_shard_dir(work_dir)
        return {
            "candidate_id": candidate.candidate_id,
            "model_name": model_name,
            "success": False,
            "error": str(e),
            "wallclock_sec": round(wall_elapsed, 2),
        }

    wall_elapsed = time.time() - wall_t0
    rc = proc.returncode if proc else -1

    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    parsed = _parse_metrics(log_text)

    # Copy artifacts
    for artifact in ("final_model.pt", "final_model.int8.ptz"):
        src = work_dir / artifact
        if src.exists():
            shutil.copy2(str(src), str(run_dir / artifact))

    # Build RunSummary
    steps = parsed.get("steps_completed", 0)
    tokens = budget_spec.batch_tokens * steps
    budget_iters = int(budget_env.get("ITERATIONS", "0"))
    budget_wc = float(budget_env.get("MAX_WALLCLOCK_SECONDS", "0"))

    if steps >= budget_iters and budget_iters > 0:
        exhausted = "iterations"
    elif wall_elapsed >= budget_wc * 0.95:
        exhausted = "wallclock"
    else:
        exhausted = "incomplete"

    summary = RunSummary(
        run_name=f"{candidate.candidate_id}__{model_name}",
        model_name=model_name,
        config_path=script_abs,
        seed=seed,
        git_commit=RunSummary._get_git_commit(),
        budget_mode=budget_spec.mode,
        budget_value=budget_spec.value,
        target_seq_len=1024,
        effective_batch_tokens=budget_spec.batch_tokens,
        tokens_processed=tokens,
        optimizer_steps=steps,
        train_wallclock_sec=round(wall_elapsed, 2),
        train_loss=parsed.get("train_loss", 0),
        budget_exhausted=exhausted,
        eval_mode="full_train_val",
        pre_quant_val_bpb=parsed.get("val_bpb", 0),
        post_quant_val_bpb=parsed.get("post_quant_bpb", 0),
        artifact_bytes=parsed.get("artifact_bytes", 0),
        peak_vram_allocated_gb=round(parsed.get("peak_vram_mib", 0) / 1024, 3),
        status="completed" if rc == 0 else "failed",
        failure_reason="" if rc == 0 else f"exit code {rc}",
    )
    summary.compute_derived()
    save_run_summary(summary, run_dir / "run_summary.json")

    cleanup_shard_dir(work_dir)

    return {
        "candidate_id": candidate.candidate_id,
        "model_name": model_name,
        "success": rc == 0,
        "val_bpb": parsed.get("val_bpb"),
        "post_quant_bpb": parsed.get("post_quant_bpb"),
        "train_loss": parsed.get("train_loss"),
        "steps_completed": steps,
        "tokens_processed": tokens,
        "wallclock_sec": round(wall_elapsed, 2),
        "peak_vram_mib": parsed.get("peak_vram_mib", 0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep train subset candidates on anchor models."
    )

    # Inputs
    parser.add_argument(
        "--candidates-dir", required=True,
        help="Directory of candidate train subset manifests (from generate_train_candidates.py)",
    )
    parser.add_argument(
        "--records-dir", default=None,
        help="Records directory (auto-discovers anchor models from train_gpt.py files)",
    )
    parser.add_argument(
        "--anchor-scripts", nargs="+", default=None,
        help="Explicit anchor model scripts (overrides --records-dir)",
    )
    parser.add_argument(
        "--candidate-ids", nargs="+", default=None,
        help="Only sweep these candidate IDs (default: all)",
    )

    # Budget
    parser.add_argument("--budget-mode", default="tokens",
                        choices=["wallclock", "tokens", "optimizer_steps"])
    parser.add_argument("--budget-value", type=float, default=16_000_000)
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=1337)

    # GPU
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-gb", type=float, default=10.0)
    parser.add_argument("--conda-env", default="parameter-golf")

    # Reference ranking
    parser.add_argument(
        "--reference-dir", default=None,
        help="Directory with submission.json files for reference ranking "
             "(defaults to --records-dir)",
    )

    # Output
    parser.add_argument("--output-dir", default="artifacts/train_subset_sweep")
    parser.add_argument("--n-finalists", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load candidates
    # -----------------------------------------------------------------------
    candidates_dir = Path(args.candidates_dir)
    if not candidates_dir.is_absolute():
        candidates_dir = (REPO_ROOT / candidates_dir).resolve()

    all_candidates = load_candidates(candidates_dir)
    if args.candidate_ids:
        all_candidates = [c for c in all_candidates
                          if c.candidate_id in args.candidate_ids]

    if not all_candidates:
        print("[ERROR] No candidates found")
        sys.exit(1)

    print(f"[sweep] Loaded {len(all_candidates)} train subset candidates")

    # -----------------------------------------------------------------------
    # Discover anchor models
    # -----------------------------------------------------------------------
    anchor_scripts: list[tuple[str, str]] = []   # (name, path)

    if args.anchor_scripts:
        for s in args.anchor_scripts:
            p = Path(s).resolve()
            anchor_scripts.append((p.parent.name, str(p)))
    elif args.records_dir:
        rdir = Path(args.records_dir)
        if not rdir.is_absolute():
            rdir = (REPO_ROOT / rdir).resolve()
        for script in sorted(rdir.glob("*/train_gpt.py")):
            anchor_scripts.append((script.parent.name, str(script)))
    else:
        print("[ERROR] Provide --records-dir or --anchor-scripts")
        sys.exit(1)

    if not anchor_scripts:
        print("[ERROR] No anchor models found")
        sys.exit(1)

    print(f"[sweep] Anchor models: {len(anchor_scripts)}")
    for name, path in anchor_scripts:
        print(f"  - {name}")

    # -----------------------------------------------------------------------
    # Load reference ranking
    # -----------------------------------------------------------------------
    ref_dir = Path(args.reference_dir or args.records_dir or "")
    if not ref_dir.is_absolute():
        ref_dir = (REPO_ROOT / ref_dir).resolve()

    ref_scores = load_reference_ranking(ref_dir)
    if ref_scores:
        print(f"\n[sweep] Reference ranking ({len(ref_scores)} models):")
        for name, bpb in sorted(ref_scores.items(), key=lambda x: x[1]):
            print(f"  {name}: {bpb:.4f}")
    else:
        print("\n[sweep] WARNING: No reference ranking found. "
              "Will use full-val scores as reference.")

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    source_data_dir = (REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024").resolve()
    budget_spec = BudgetSpec(
        mode=args.budget_mode,
        value=args.budget_value,
        batch_tokens=args.batch_tokens,
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(all_candidates) * len(anchor_scripts)
    print(f"\n[sweep] Total runs: {total_runs} "
          f"({len(all_candidates)} candidates × {len(anchor_scripts)} anchors)")
    print(f"[sweep] Budget: {args.budget_mode}={args.budget_value}")

    # GPU detection
    gpu = detect_gpu()
    n_workers = args.max_workers or max_parallel_workers(
        gpu.total_vram_gb, args.max_gb
    ) or 1
    mem_frac = memory_fraction_for_worker(gpu.total_vram_gb, n_workers)
    use_frac = mem_frac if n_workers > 1 else 1.0

    print(f"[sweep] GPU: {gpu.name} ({gpu.total_vram_gb:.1f} GB), "
          f"{n_workers} workers, mem_fraction={use_frac:.3f}")

    if args.dry_run:
        print("\n[dry-run] Would run:")
        for cand in all_candidates:
            for name, _ in anchor_scripts:
                print(f"  {cand.candidate_id} × {name}")
        return

    # -----------------------------------------------------------------------
    # Run the sweep
    # -----------------------------------------------------------------------
    all_results: list[dict] = []
    completed = 0

    # Build work items: list of (candidate, model_name, script_path)
    work_items = [
        (cand, name, path)
        for cand in all_candidates
        for name, path in anchor_scripts
    ]

    wall_t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for cand, name, path in work_items:
            fut = pool.submit(
                _run_one,
                candidate=cand,
                script_path=path,
                model_name=name,
                output_dir=output_dir,
                source_data_dir=source_data_dir,
                budget_spec=budget_spec,
                mem_fraction=use_frac,
                conda_env=args.conda_env,
                seed=args.seed,
            )
            futures[fut] = (cand.candidate_id, name)

        for fut in as_completed(futures):
            cid, mname = futures[fut]
            completed += 1
            try:
                result = fut.result()
            except Exception as e:
                result = {
                    "candidate_id": cid,
                    "model_name": mname,
                    "success": False,
                    "error": str(e),
                }
            all_results.append(result)

            status = "OK" if result.get("success") else "FAIL"
            bpb = result.get("val_bpb")
            bpb_s = f"BPB={bpb:.4f}" if bpb else "no BPB"
            wc = result.get("wallclock_sec", 0)
            print(f"[sweep] [{completed}/{total_runs}] "
                  f"{cid} × {mname}: {status} ({bpb_s}, {wc:.0f}s)")

    total_wall = time.time() - wall_t0
    print(f"\n[sweep] All runs completed in {total_wall:.0f}s")

    # -----------------------------------------------------------------------
    # Collect scores and evaluate candidates
    # -----------------------------------------------------------------------
    # Group results: {candidate_id: {model_name: best_bpb}}
    sweep_scores: dict[str, dict[str, float]] = {}
    for r in all_results:
        if not r.get("success"):
            continue
        cid = r["candidate_id"]
        mname = r["model_name"]
        bpb = r.get("post_quant_bpb") or r.get("val_bpb")
        if bpb:
            sweep_scores.setdefault(cid, {})[mname] = bpb

    if not sweep_scores:
        print("[ERROR] No successful runs. Cannot evaluate candidates.")
        sys.exit(1)

    # Determine reference ranking
    # Prefer leaderboard scores; fall back to best available from sweep
    anchor_names = [n for n, _ in anchor_scripts]
    if ref_scores:
        reference = {n: ref_scores[n] for n in anchor_names if n in ref_scores}
        ref_source = "leaderboard"
    else:
        # Use the median score across all candidates as a pseudo-reference
        from collections import defaultdict
        import statistics
        all_model_scores: dict[str, list[float]] = defaultdict(list)
        for cid, scores in sweep_scores.items():
            for m, s in scores.items():
                all_model_scores[m].append(s)
        reference = {m: statistics.median(ss)
                     for m, ss in all_model_scores.items() if ss}
        ref_source = "median_across_candidates"

    if len(reference) < 2:
        print("[ERROR] Need at least 2 models in reference ranking")
        sys.exit(1)

    print(f"\n[sweep] Reference ranking source: {ref_source}")
    print(f"[sweep] Reference models: {len(reference)}")

    # Evaluate each candidate
    evaluations: list[CandidateEvaluation] = []
    for cand in all_candidates:
        cid = cand.candidate_id
        if cid not in sweep_scores:
            continue
        ev = evaluate_candidate(
            candidate_id=cid,
            proxy_scores=sweep_scores[cid],
            ref_scores=reference,
            family=cand.family,
            shard_ids=cand.shard_ids,
        )
        ev.n_successful_runs = sum(
            1 for r in all_results
            if r.get("candidate_id") == cid and r.get("success")
        )
        ev.n_failed_runs = sum(
            1 for r in all_results
            if r.get("candidate_id") == cid and not r.get("success")
        )
        evaluations.append(ev)

    # -----------------------------------------------------------------------
    # Select finalists
    # -----------------------------------------------------------------------
    finalists = select_finalists(evaluations, n_finalists=args.n_finalists)

    print(f"\n{'='*80}")
    print("TRAIN SUBSET SWEEP RESULTS")
    print(f"{'='*80}\n")
    print(format_evaluation_table(evaluations, finalists))

    if finalists:
        print(f"\n{'='*80}")
        print(f"TOP {len(finalists)} FINALISTS")
        print(f"{'='*80}")
        for i, f in enumerate(finalists, 1):
            print(f"\n  #{i} {f.candidate_id}")
            print(f"      Family: {f.family}")
            print(f"      Shards: {f.shard_ids}")
            print(f"      Spearman: {f.spearman_rho:+.3f}")
            print(f"      Kendall:  {f.kendall_tau:+.3f}")
            print(f"      Pairwise: {f.pairwise_accuracy:.3f}")
            print(f"      Top-1:    {'YES' if f.top_1_agreement else 'NO'}")
            print(f"      Composite: {f.composite_score:.3f}")

    # -----------------------------------------------------------------------
    # Save reports
    # -----------------------------------------------------------------------
    report = build_selection_report(
        evaluations=evaluations,
        finalists=finalists,
        reference_source=ref_source,
        anchor_models=anchor_names,
    )

    selection_dir = output_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)

    save_selection_report(
        report,
        report_path=selection_dir / "report.json",
        finalists_path=selection_dir / "finalists.json",
    )

    # Save raw sweep results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump({
            "n_candidates": len(all_candidates),
            "n_anchors": len(anchor_scripts),
            "n_total_runs": total_runs,
            "n_successful": sum(1 for r in all_results if r.get("success")),
            "budget": {"mode": args.budget_mode, "value": args.budget_value},
            "reference_source": ref_source,
            "total_wallclock_sec": round(total_wall, 2),
            "results": all_results,
        }, f, indent=2)

    print(f"\n[sweep] Reports saved to {selection_dir}")
    print(f"  report.json    - full evaluation details")
    print(f"  finalists.json - top {len(finalists)} train subset finalists")
    print(f"  ../sweep_results.json - raw per-run results")


if __name__ == "__main__":
    main()

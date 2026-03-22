#!/usr/bin/env python3
"""
Quick evaluation framework for parameter-golf submissions.

Runs a submission's train_gpt.py with scaled-down parameters to produce
a proxy BPB score in ~10-15 minutes on a single consumer GPU (e.g. RTX 3080 12GB).

The proxy score should correlate with the full 8xH100 leaderboard ranking,
allowing rapid iteration on ideas without expensive compute.

Usage:
    # Evaluate a single submission
    python tools/quick_eval.py records/track_10min_16mb/2026-03-17_NaiveBaseline/

    # Evaluate all record submissions and compare with leaderboard
    python tools/quick_eval.py --all

    # Custom time budget and batch size
    python tools/quick_eval.py --time-budget 600 --batch-tokens 32768 path/to/submission/
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIR = ROOT / "records" / "track_10min_16mb"
RESULTS_DIR = ROOT / "tools" / "results"
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Leaderboard reference
# ---------------------------------------------------------------------------

def load_leaderboard():
    """Load official scores from submission.json files."""
    entries = []
    for d in sorted(RECORDS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "submission.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        # Some submissions use val_bpb, others only have val_loss (which is actually bpb)
        bpb = meta.get("val_bpb", meta.get("val_loss"))
        if bpb is None:
            continue
        entries.append({"name": d.name, "val_bpb": float(bpb)})
    entries.sort(key=lambda x: x["val_bpb"])
    return entries

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def build_env(time_budget, batch_tokens):
    """Build environment variables for quick evaluation on a consumer GPU.

    Strategy: reduce TRAIN_BATCH_TOKENS so we get more optimizer steps in the
    time budget.  With 65 536 tokens/step on a 3080 we get ~1 200 steps in
    15 minutes (vs ~158 steps with the default 524 288 batch).  Muon's
    Newton-Schulz normalization makes it relatively batch-size-insensitive,
    and Adam adapts per-parameter, so we do NOT scale learning rates by
    default.  This keeps the framework simple and avoids interfering with
    submission-specific LR tuning.
    """
    env = os.environ.copy()
    env.update({
        # Core scaling
        "MAX_WALLCLOCK_SECONDS": str(time_budget),
        "TRAIN_BATCH_TOKENS":   str(batch_tokens),
        # Keep torch.compile (uncompiled is too slow on consumer GPUs).
        # Compile overhead is ~5 min one-time cost; steps are fast after.
        # Reduce warmup steps to minimize compile-phase time.
        "WARMUP_STEPS":         "5",
        # Validation / logging cadence (tuned for shorter runs)
        "VAL_LOSS_EVERY":       "100",
        "TRAIN_LOG_EVERY":      "25",
        # Warmdown scaled for shorter run
        "WARMDOWN_ITERS":       "150",
        # Reduce val batch to speed up roundtrip eval (uncompiled model is slow)
        "VAL_BATCH_SIZE":       str(batch_tokens),
        # Absolute data paths so scripts work from any cwd
        "DATA_PATH":            str(DATA_DIR / "datasets" / "fineweb10B_sp1024"),
        "TOKENIZER_PATH":       str(DATA_DIR / "tokenizers" / "fineweb_1024_bpe.model"),
    })
    return env

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_output(text):
    """Extract metrics from training script stdout/stderr."""
    m = {}

    # Validation checkpoints
    m["val_checkpoints"] = []
    for match in re.finditer(
        r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+).*?train_time:(\d+)ms",
        text,
    ):
        m["val_checkpoints"].append({
            "step":          int(match.group(1)),
            "val_loss":      float(match.group(2)),
            "val_bpb":       float(match.group(3)),
            "train_time_ms": int(match.group(4)),
        })

    # Post-quantization roundtrip (the official submission metric)
    match = re.search(
        r"final_int8_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+)", text
    )
    if match:
        m["post_quant_val_loss"] = float(match.group(1))
        m["post_quant_val_bpb"]  = float(match.group(2))

    match = re.search(
        r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", text
    )
    if match:
        m["post_quant_val_bpb_exact"] = float(match.group(2))

    # Early stopping
    match = re.search(
        r"stopping_early.*?train_time:(\d+)ms.*?step:(\d+)/(\d+)", text
    )
    if match:
        m["total_train_time_ms"] = int(match.group(1))
        m["stopped_at_step"]     = int(match.group(2))
        m["max_iterations"]      = int(match.group(3))

    # Artifact size
    match = re.search(r"Serialized model int8\+zlib: (\d+) bytes", text)
    if match:
        m["artifact_bytes"] = int(match.group(1))

    match = re.search(r"model_params:(\d+)", text)
    if match:
        m["model_params"] = int(match.group(1))

    # Convenience: last pre-quant val checkpoint
    if m["val_checkpoints"]:
        last = m["val_checkpoints"][-1]
        m["final_val_bpb"]  = last["val_bpb"]
        m["final_val_loss"] = last["val_loss"]

    # The "quick score" is the post-quant bpb if available, else last pre-quant
    m["quick_score"] = m.get("post_quant_val_bpb", m.get("final_val_bpb"))

    return m

# ---------------------------------------------------------------------------
# Submission runner
# ---------------------------------------------------------------------------

def find_script(submission_dir):
    """Find the training script in a submission directory."""
    for name in ("train_gpt.py", "train_gpt_v5.py"):
        p = submission_dir / name
        if p.exists():
            return p
    return None


def run_submission(submission_dir, time_budget, batch_tokens, conda_env, log_dir):
    """Run one submission with quick-eval parameters.  Returns a metrics dict."""
    script = find_script(submission_dir)
    if script is None:
        return {"submission": submission_dir.name, "error": "no training script found"}

    env = build_env(time_budget, batch_tokens)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = ["micromamba", "run", "-n", conda_env, "python", "-u", str(script)]

    print(f"\n{'=' * 60}")
    print(f"  {submission_dir.name}")
    print(f"  script={script.name}  budget={time_budget}s  batch={batch_tokens}")
    print(f"{'=' * 60}")

    log_path = log_dir / f"{submission_dir.name}.log"
    start = time.time()

    try:
        # Write directly to log file so we can `tail -f` for progress
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                # Budget for: compile (~5 min) + training + recompile after
                # weight reload (~5 min) + quantization + roundtrip eval
                timeout=time_budget + 1800,
            )
    except subprocess.TimeoutExpired:
        return {
            "submission": submission_dir.name,
            "error": "timed out",
            "elapsed_seconds": round(time.time() - start, 1),
        }

    elapsed = time.time() - start
    output = log_path.read_text(encoding="utf-8")

    # Preserve model checkpoint for eval-only reruns if training saved one
    model_src = ROOT / "final_model.pt"
    if model_src.exists():
        model_dst = log_dir / f"{submission_dir.name}.pt"
        model_src.rename(model_dst)
        print(f"  Saved checkpoint: {model_dst}")
    # Also move the quantized artifact if it exists
    quant_src = ROOT / "final_model.int8.ptz"
    if quant_src.exists():
        quant_src.rename(log_dir / f"{submission_dir.name}.int8.ptz")

    metrics = parse_output(output)
    metrics["submission"]      = submission_dir.name
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["returncode"]      = result.returncode
    metrics["log_file"]        = str(log_path)

    if result.returncode != 0:
        stderr_tail = output[-1500:]
        metrics["error"] = stderr_tail
        # If training completed but quant/eval crashed, note the checkpoint
        model_path = log_dir / f"{submission_dir.name}.pt"
        if model_path.exists():
            metrics["checkpoint"] = str(model_path)
            metrics["hint"] = (
                "Training completed but post-training phase failed. "
                "Re-run with: python tools/eval_only.py "
                f"--script {find_script(submission_dir)} --checkpoint {model_path}"
            )

    # Print one-liner
    score = metrics.get("quick_score", "FAIL")
    steps = metrics.get("stopped_at_step", "?")
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"  [{status}] quick_score={score}  steps={steps}  time={elapsed:.0f}s")

    return metrics

# ---------------------------------------------------------------------------
# Ranking comparison
# ---------------------------------------------------------------------------

def spearman_rho(xs, ys):
    """Spearman rank correlation coefficient."""
    n = len(xs)
    if n < 2:
        return float("nan")
    d2 = sum((x - y) ** 2 for x, y in zip(xs, ys))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def compare_rankings(results, leaderboard):
    """Compare quick-eval ranking with the official leaderboard."""
    lb = {e["name"]: e["val_bpb"] for e in leaderboard}

    paired = []
    for r in results:
        name = r["submission"]
        qs = r.get("quick_score")
        if qs is not None and name in lb:
            paired.append((name, qs, lb[name]))

    if len(paired) < 2:
        print("\nNot enough paired results for ranking comparison.")
        return None

    # Rank by each metric (lower bpb = better = rank 1)
    by_quick = sorted(paired, key=lambda t: t[1])
    by_lb    = sorted(paired, key=lambda t: t[2])
    qr = {name: i + 1 for i, (name, _, _) in enumerate(by_quick)}
    lr = {name: i + 1 for i, (name, _, _) in enumerate(by_lb)}

    names = [name for name, _, _ in by_lb]
    rho = spearman_rho([qr[n] for n in names], [lr[n] for n in names])

    print(f"\n{'=' * 74}")
    print(f"  RANKING COMPARISON  (Spearman rho = {rho:.4f})")
    print(f"{'=' * 74}")
    print(f"  {'Submission':<46} {'Quick':>7} {'LBoard':>7}")
    print(f"  {'-' * 46} {'-' * 7} {'-' * 7}")
    for name in names:
        marker = " " if qr[name] == lr[name] else "*"
        print(f" {marker}{name:<46} #{qr[name]:<6} #{lr[name]:<6}")
    print()

    # Also show absolute scores side by side
    print(f"  {'Submission':<46} {'Q-BPB':>8} {'L-BPB':>8}")
    print(f"  {'-' * 46} {'-' * 8} {'-' * 8}")
    for name in names:
        qs = next(t[1] for t in paired if t[0] == name)
        ls = next(t[2] for t in paired if t[0] == name)
        print(f"  {name:<46} {qs:>8.4f} {ls:>8.4f}")

    return rho


def _save_results(path, run_id, args, results):
    """Save intermediate results after each submission completes."""
    with open(path, "w") as f:
        json.dump({
            "run_id": run_id,
            "time_budget": args.time_budget,
            "batch_tokens": args.batch_tokens,
            "results": results,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quick evaluation framework for parameter-golf submissions"
    )
    parser.add_argument(
        "submission", nargs="?",
        help="Path to a submission directory (containing train_gpt.py)",
    )
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all record submissions")
    parser.add_argument("--time-budget", type=int, default=900,
                        help="Training time budget in seconds (default: 900)")
    parser.add_argument("--batch-tokens", type=int, default=65536,
                        help="Tokens per training step (default: 65536)")
    parser.add_argument("--conda-env", default="parameter-golf",
                        help="Micromamba environment name (default: parameter-golf)")
    parser.add_argument("--output", default=None,
                        help="Path for the JSON results file")
    parser.add_argument("--resume", default=None,
                        help="Resume from a previous results.json or log directory")
    parser.add_argument("--submissions", nargs="+", default=None,
                        help="Only run these submissions (substring match on dir name)")
    args = parser.parse_args()

    if not args.submission and not args.all:
        parser.error("Provide a submission directory or use --all")

    # Prepare output directory
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = RESULTS_DIR / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Find all runnable submissions
        submissions = sorted([
            d for d in RECORDS_DIR.iterdir()
            if d.is_dir() and find_script(d) is not None
        ])
        # Filter to specific submissions if requested
        if args.submissions:
            submissions = [
                d for d in submissions
                if any(s in d.name for s in args.submissions)
            ]
        print(f"Found {len(submissions)} submissions to evaluate")

        out = args.output or str(log_dir / "results.json")

        # Resume support: load previous results and skip already-completed runs.
        # Also scans for orphaned log files from crashed runs.
        results = []
        completed = set()
        if args.resume:
            # First check the results JSON
            resume_path = Path(args.resume)
            if resume_path.is_file():
                prev = json.loads(resume_path.read_text())
                prev_results = prev.get("results", prev) if isinstance(prev, dict) else prev
                for r in prev_results:
                    if r.get("quick_score") is not None:
                        results.append(r)
                        completed.add(r["submission"])
                # Use the same log dir as the resumed run
                prev_log_dir = None
                for r in prev_results:
                    lf = r.get("log_file")
                    if lf:
                        prev_log_dir = Path(lf).parent
                        break
                if prev_log_dir and prev_log_dir.is_dir():
                    log_dir = prev_log_dir

            # Also recover from orphaned log files (crash left log but no JSON entry)
            if resume_path.is_dir():
                log_dir = resume_path
            for log_file in log_dir.glob("*.log"):
                sub_name = log_file.stem
                if sub_name in completed:
                    continue
                log_text = log_file.read_text(encoding="utf-8")
                recovered = parse_output(log_text)
                if recovered.get("quick_score") is not None:
                    recovered["submission"] = sub_name
                    recovered["log_file"] = str(log_file)
                    recovered["recovered_from_log"] = True
                    results.append(recovered)
                    completed.add(sub_name)
                    print(f"  Recovered {sub_name}: quick_score={recovered['quick_score']}")

            if completed:
                print(f"Resuming: {len(completed)} submissions already done, "
                      f"{len(submissions) - len(completed)} remaining")

        for sub_dir in submissions:
            if sub_dir.name in completed:
                continue
            r = run_submission(
                sub_dir, args.time_budget, args.batch_tokens,
                args.conda_env, log_dir,
            )
            results.append(r)

            # Save after every run so progress survives crashes
            _save_results(out, run_id, args, results)

        # Final ranking comparison
        leaderboard = load_leaderboard()
        rho = compare_rankings(results, leaderboard)

        summary = {
            "run_id": run_id,
            "time_budget": args.time_budget,
            "batch_tokens": args.batch_tokens,
            "spearman_rho": rho,
            "results": results,
        }
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {out}")

    else:
        sub_dir = Path(args.submission).resolve()
        if not sub_dir.is_dir():
            parser.error(f"Not a directory: {sub_dir}")

        r = run_submission(
            sub_dir, args.time_budget, args.batch_tokens,
            args.conda_env, log_dir,
        )
        out = args.output or str(log_dir / f"{sub_dir.name}.json")
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

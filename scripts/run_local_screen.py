#!/usr/bin/env python3
"""
Run a candidate submission through a proxy training + evaluation cycle.

Drives a submission's train_gpt.py with reduced hyperparameters suitable
for a local RTX 3080, captures logs, and optionally evaluates the resulting
checkpoint on proxy_val_tune sequences.

Rank-fidelity motivation (DataDecide, arXiv:2504.11393): short proxy runs
can predict full-scale rankings when the evaluation protocol is consistent.
VRAM cap enforced per SparseEval principle of efficient evaluation.

Example:
    python scripts/run_local_screen.py \
        --script records/my_idea/train_gpt.py \
        --output-dir proxy_results/screen_001 \
        --time-budget 300 \
        --val-tune-manifest proxy_data/proxy_val_tune.json
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from proxy_framework.data_utils import (
    extract_sequences,
    iter_batches,
    load_all_val_tokens,
    load_manifest,
)
from proxy_framework.model_utils import (
    build_model,
    eval_per_sequence,
    import_submission,
)
from proxy_framework.budget import BudgetSpec, RunSummary, save_run_summary
from proxy_framework.vram_guard import VRAMGuard, check_vram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_final_metrics(log_text: str) -> dict:
    """Scrape training log for key metrics emitted by the submission."""
    metrics: dict = {}

    # Find the last step line with val metrics
    for line in reversed(log_text.splitlines()):
        if "val_bpb:" in line and "step:" in line:
            m = re.search(r"val_loss:([\d.]+)", line)
            if m:
                metrics["train_val_loss"] = float(m.group(1))
            m = re.search(r"val_bpb:([\d.]+)", line)
            if m:
                metrics["train_val_bpb"] = float(m.group(1))
            m = re.search(r"step:(\d+)", line)
            if m:
                metrics["steps_completed"] = int(m.group(1))
            m = re.search(r"train_time:(\d+)ms", line)
            if m:
                metrics["train_time_ms"] = int(m.group(1))
            m = re.search(r"step_avg:([\d.]+)ms", line)
            if m:
                metrics["step_avg_ms"] = float(m.group(1))
            break

    # Last train_loss
    for line in reversed(log_text.splitlines()):
        if "train_loss:" in line:
            m = re.search(r"train_loss:([\d.]+)", line)
            if m:
                metrics["train_loss"] = float(m.group(1))
            break

    # Peak memory
    for line in log_text.splitlines():
        if "peak memory allocated:" in line:
            m = re.search(r"peak memory allocated:\s*(\d+)\s*MiB", line)
            if m:
                metrics["train_vram_peak_mib"] = int(m.group(1))

    # Int8+zlib roundtrip bpb
    for line in log_text.splitlines():
        if "final_int8_zlib_roundtrip " in line:
            m = re.search(r"val_bpb:([\d.]+)", line)
            if m:
                metrics["post_quant_val_bpb"] = float(m.group(1))
            m = re.search(r"val_loss:([\d.]+)", line)
            if m:
                metrics["post_quant_val_loss"] = float(m.group(1))

    # Artifact sizes
    for line in log_text.splitlines():
        if "Serialized model int8+zlib:" in line:
            m = re.search(r"int8\+zlib:\s*(\d+)\s*bytes", line)
            if m:
                metrics["artifact_bytes_int8_zlib"] = int(m.group(1))
        elif "Serialized model:" in line and "int8" not in line:
            m = re.search(r"Serialized model:\s*(\d+)\s*bytes", line)
            if m:
                metrics["artifact_bytes_raw"] = int(m.group(1))

    return metrics


def _compute_proxy_eval_bpb(
    script_path: str,
    checkpoint_path: str,
    val_manifest_path: str,
    max_gb: float,
    device_str: str = "cuda",
) -> dict:
    """Load checkpoint and evaluate on proxy_val sequences, returning BPB."""
    import torch

    device = torch.device(device_str)
    print(f"[eval] Loading submission module from {script_path}")
    mod = import_submission(script_path)
    model, hp = build_model(mod, device)

    print(f"[eval] Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)

    manifest = load_manifest(val_manifest_path)
    seq_len = manifest.seq_len if manifest.seq_len > 0 else hp.train_seq_len
    seq_ids = manifest.seq_ids

    print(f"[eval] Evaluating on {len(seq_ids)} sequences (seq_len={seq_len})")

    data_dir = Path(hp.data_path).resolve() if hasattr(hp, "data_path") else REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"
    val_tokens = load_all_val_tokens(str(data_dir), seq_len=seq_len)

    # Batch eval with VRAM guard
    guard = VRAMGuard(max_gb=max_gb)
    guard.__enter__()
    guard.start_monitor(interval_s=5.0)

    batch_seqs = max(1, 32768 // seq_len)  # ~32k tokens per micro-batch
    all_losses = []

    compiled_model.eval()
    with torch.inference_mode():
        for x, y, batch_ids in iter_batches(val_tokens, seq_ids, seq_len, batch_seqs):
            x = x.to(device=device, dtype=torch.int64)
            y = y.to(device=device, dtype=torch.int64)
            per_seq = eval_per_sequence(compiled_model, x, y, seq_len)
            all_losses.append(per_seq.cpu())

    guard.stop_monitor()
    vram_info = guard.check()
    guard.__exit__(None, None, None)

    all_losses_t = torch.cat(all_losses)
    mean_loss = all_losses_t.mean().item()
    mean_bpb = mean_loss / math.log(2.0)  # approximate BPB (ignoring byte counts)

    return {
        "proxy_val_loss": round(mean_loss, 6),
        "proxy_val_bpb_approx": round(mean_bpb, 6),
        "proxy_val_n_seqs": len(seq_ids),
        "eval_vram_peak_gb": vram_info.get("peak_gb", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a candidate submission through proxy screening."
    )
    parser.add_argument("--script", required=True, help="Path to submission train_gpt.py")
    parser.add_argument("--output-dir", default="proxy_results/screen", help="Output directory")
    parser.add_argument("--time-budget", type=int, default=300, help="MAX_WALLCLOCK_SECONDS for training (default 300)")
    parser.add_argument("--batch-tokens", type=int, default=32768, help="TRAIN_BATCH_TOKENS (default 32768)")
    parser.add_argument("--max-gb", type=float, default=10.0, help="VRAM cap in GB for eval phase")
    parser.add_argument("--conda-env", default="parameter-golf", help="Conda/micromamba env name")
    parser.add_argument("--train-manifest", default=None, help="Path to proxy train manifest JSON")
    parser.add_argument("--val-tune-manifest", default=None, help="Path to proxy_val_tune manifest JSON")
    parser.add_argument("--val-long-manifest", default=None, help="Path to proxy_val_long manifest JSON")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--budget-mode", choices=["wallclock", "tokens", "optimizer_steps"],
                        default="tokens", help="Budget mode (default: tokens)")
    parser.add_argument("--budget-value", type=float, default=16000000,
                        help="Budget value (default: 16000000 for tokens mode)")
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build absolute data paths
    data_dir = (REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024").resolve()
    tokenizer_path = (REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve()

    # Build budget spec and derive env overrides for MAX_WALLCLOCK_SECONDS / ITERATIONS
    budget_spec = BudgetSpec(
        mode=args.budget_mode,
        value=args.budget_value,
        batch_tokens=args.batch_tokens,
    )
    budget_env = budget_spec.to_env_overrides()

    # Construct env vars for the training subprocess
    env = os.environ.copy()
    env.update({
        "TRAIN_BATCH_TOKENS": str(args.batch_tokens),
        "WARMUP_STEPS": "5",
        "WARMDOWN_ITERS": "150",
        "VAL_LOSS_EVERY": "100",
        "TRAIN_LOG_EVERY": "25",
        "VAL_BATCH_SIZE": "32768",
        "PYTHONUNBUFFERED": "1",
        "DATA_PATH": str(data_dir),
        "TOKENIZER_PATH": str(tokenizer_path),
        "SEED": str(args.seed),
    })
    # Apply budget-derived overrides (MAX_WALLCLOCK_SECONDS, ITERATIONS, etc.)
    env.update(budget_env)

    # If a train manifest specifies which shards to use, we point DATA_PATH
    # at that subset (the submission reads TRAIN_FILES from DATA_PATH).
    if args.train_manifest:
        manifest = load_manifest(args.train_manifest)
        if manifest.shard_ids is not None:
            # The train manifest records shard indices; the submission still
            # reads from the same data_dir but we log what subset was used.
            print(f"[info] Train manifest: {len(manifest.shard_ids)} shards selected")
            env["TRAIN_MANIFEST_INFO"] = json.dumps({
                "path": str(args.train_manifest),
                "n_shards": len(manifest.shard_ids),
            })

    # Run training subprocess
    log_path = out_dir / "train.log"
    cmd = [
        "micromamba", "run", "-n", args.conda_env,
        "python", "-u", str(script_path),
    ]

    print(f"[screen] Running training: {' '.join(cmd)}")
    print(f"[screen] Budget: {args.budget_mode}={args.budget_value}, batch tokens: {args.batch_tokens}")
    print(f"[screen] Log file: {log_path}")

    wall_t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(REPO_ROOT),
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                log_f.write(line)
                log_f.flush()
            proc.wait()
    except Exception as e:
        print(f"[ERROR] Training subprocess failed: {e}")
        proc = None

    wall_elapsed = time.time() - wall_t0
    return_code = proc.returncode if proc else -1
    print(f"[screen] Training finished in {wall_elapsed:.1f}s (exit code {return_code})")

    # Read log for metrics
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    parsed = _parse_final_metrics(log_text)

    results = {
        "script": str(script_path),
        "seed": args.seed,
        "time_budget_s": args.time_budget,
        "batch_tokens": args.batch_tokens,
        "wall_elapsed_s": round(wall_elapsed, 2),
        "return_code": return_code,
        "max_gb": args.max_gb,
        **parsed,
    }

    # Copy artifacts to output dir & record sizes
    for artifact_name in ("final_model.pt", "final_model.int8.ptz"):
        src = REPO_ROOT / artifact_name
        if src.exists():
            dst = out_dir / artifact_name
            shutil.copy2(str(src), str(dst))
            results[f"artifact_size_{artifact_name}"] = os.path.getsize(str(dst))
            print(f"[screen] Copied {artifact_name} -> {dst} ({os.path.getsize(str(dst))} bytes)")
            # Clean up source
            src.unlink()
        else:
            print(f"[screen] Artifact not found: {src}")

    # Optional: proxy_val_tune evaluation
    if args.val_tune_manifest:
        checkpoint = out_dir / "final_model.pt"
        if checkpoint.exists():
            print(f"[screen] Running proxy_val_tune evaluation...")
            try:
                eval_results = _compute_proxy_eval_bpb(
                    script_path=str(script_path),
                    checkpoint_path=str(checkpoint),
                    val_manifest_path=args.val_tune_manifest,
                    max_gb=args.max_gb,
                )
                results["proxy_val_tune"] = eval_results
                print(f"[screen] proxy_val_tune BPB: {eval_results.get('proxy_val_bpb_approx', 'N/A')}")
            except Exception as e:
                print(f"[ERROR] proxy_val_tune eval failed: {e}")
                results["proxy_val_tune_error"] = str(e)
        else:
            print("[screen] No checkpoint found, skipping proxy_val_tune eval")

    # Optional: proxy_val_long evaluation
    if args.val_long_manifest:
        checkpoint = out_dir / "final_model.pt"
        if checkpoint.exists():
            print(f"[screen] Running proxy_val_long evaluation...")
            try:
                eval_results = _compute_proxy_eval_bpb(
                    script_path=str(script_path),
                    checkpoint_path=str(checkpoint),
                    val_manifest_path=args.val_long_manifest,
                    max_gb=args.max_gb,
                )
                results["proxy_val_long"] = eval_results
                print(f"[screen] proxy_val_long BPB: {eval_results.get('proxy_val_bpb_approx', 'N/A')}")
            except Exception as e:
                print(f"[ERROR] proxy_val_long eval failed: {e}")
                results["proxy_val_long_error"] = str(e)
        else:
            print("[screen] No checkpoint found, skipping proxy_val_long eval")

    # Build and save RunSummary for matched-budget tracking
    steps_completed = parsed.get("steps_completed", 0)
    batch_tokens = args.batch_tokens
    tokens_processed = batch_tokens * steps_completed
    budget_iterations = int(budget_env.get("ITERATIONS", "0"))
    budget_wallclock = float(budget_env.get("MAX_WALLCLOCK_SECONDS", "0"))

    # Determine which budget limit was hit
    if steps_completed >= budget_iterations and budget_iterations > 0:
        budget_exhausted = "iterations"
    elif wall_elapsed >= budget_wallclock * 0.95:
        budget_exhausted = "wallclock"
    else:
        budget_exhausted = "incomplete"

    summary = RunSummary(
        run_name=out_dir.name,
        model_name=script_path.parent.name,
        config_path=str(script_path),
        seed=args.seed,
        git_commit=RunSummary._get_git_commit(),
        budget_mode=args.budget_mode,
        budget_value=args.budget_value,
        target_seq_len=1024,
        effective_batch_tokens=batch_tokens,
        tokens_processed=tokens_processed,
        optimizer_steps=steps_completed,
        train_wallclock_sec=round(wall_elapsed, 2),
        train_loss=parsed.get("train_loss", 0),
        budget_exhausted=budget_exhausted,
        eval_mode="proxy_val_tune",
        pre_quant_val_bpb=parsed.get("train_val_bpb", 0),
        post_quant_val_bpb=parsed.get("post_quant_val_bpb", 0),
        proxy_val_tune_bpb=results.get("proxy_val_tune", {}).get("proxy_val_bpb_approx", 0),
        proxy_val_long_bpb=results.get("proxy_val_long", {}).get("proxy_val_bpb_approx", 0),
        artifact_bytes=parsed.get("artifact_bytes_int8_zlib", 0),
        peak_vram_allocated_gb=round(parsed.get("train_vram_peak_mib", 0) / 1024, 3),
        status="completed" if return_code == 0 else "failed",
        failure_reason="" if return_code == 0 else f"exit code {return_code}",
    )
    summary.compute_derived()
    save_run_summary(summary, out_dir / "run_summary.json")
    print(f"[screen] RunSummary saved to {out_dir / 'run_summary.json'}")

    # Save results
    results_path = out_dir / "screen_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[screen] Results saved to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

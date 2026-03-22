#!/usr/bin/env python3
"""
Rerun promoted candidates with a larger local budget for audit.

Similar to run_local_screen but designed for deeper evaluation:
- Longer time budget (default 600s)
- Evaluates on proxy_val_audit (separate from proxy_val_tune to avoid overfitting)
- Optionally evaluates on the full validation set
- Supports multi-seed runs for variance estimation

Example:
    python scripts/run_local_audit.py \
        --script records/my_idea/train_gpt.py \
        --output-dir proxy_results/audit_001 \
        --val-audit-manifest proxy_data/proxy_val_audit.json \
        --seeds 1337 42 7 \
        --full-val
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


def _eval_checkpoint_on_manifest(
    script_path: str,
    checkpoint_path: str,
    val_manifest_path: str,
    max_gb: float,
    device_str: str = "cuda",
) -> dict:
    """Load checkpoint and evaluate on a proxy_val manifest, returning BPB."""
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

    guard = VRAMGuard(max_gb=max_gb)
    guard.__enter__()
    guard.start_monitor(interval_s=5.0)

    batch_seqs = max(1, 32768 // seq_len)
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
    mean_bpb = mean_loss / math.log(2.0)

    return {
        "proxy_val_loss": round(mean_loss, 6),
        "proxy_val_bpb_approx": round(mean_bpb, 6),
        "proxy_val_n_seqs": len(seq_ids),
        "eval_vram_peak_gb": vram_info.get("peak_gb", 0),
    }


def _eval_checkpoint_full_val(
    script_path: str,
    checkpoint_path: str,
    max_gb: float,
    device_str: str = "cuda",
) -> dict:
    """Load checkpoint and evaluate on the full validation set.

    Uses the submission's own eval_val for a proper BPB computation with
    the sentencepiece byte-counting logic.
    """
    import torch
    import sentencepiece as spm

    device = torch.device(device_str)
    mod = import_submission(script_path)
    model, hp = build_model(mod, device)

    print(f"[eval] Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)

    # Load full val tokens using the submission's own paths
    data_dir = Path(hp.data_path).resolve() if hasattr(hp, "data_path") else REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"
    val_pattern = str(data_dir / "fineweb_val_*.bin")
    val_tokens = mod.load_validation_tokens(val_pattern, hp.train_seq_len)

    tokenizer_path = hp.tokenizer_path if hasattr(hp, "tokenizer_path") else str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = mod.build_sentencepiece_luts(
        sp, hp.vocab_size, device
    )

    guard = VRAMGuard(max_gb=max_gb)
    guard.__enter__()
    guard.start_monitor(interval_s=5.0)

    print(f"[eval] Evaluating on full validation set ({val_tokens.numel() - 1} tokens)")
    val_loss, val_bpb = mod.eval_val(
        hp,
        compiled_model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=8,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )

    guard.stop_monitor()
    vram_info = guard.check()
    guard.__exit__(None, None, None)

    return {
        "full_val_loss": round(val_loss, 6),
        "full_val_bpb": round(val_bpb, 6),
        "eval_vram_peak_gb": vram_info.get("peak_gb", 0),
    }


def _run_single_seed(
    script_path: Path,
    seed: int,
    args: argparse.Namespace,
    seed_dir: Path,
    budget_spec: BudgetSpec,
) -> dict:
    """Run training + evaluation for a single seed."""
    seed_dir.mkdir(parents=True, exist_ok=True)

    data_dir = (REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024").resolve()
    tokenizer_path = (REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve()

    budget_env = budget_spec.to_env_overrides()

    env = os.environ.copy()
    env.update({
        "TRAIN_BATCH_TOKENS": "32768",
        "WARMUP_STEPS": "5",
        "WARMDOWN_ITERS": "150",
        "VAL_LOSS_EVERY": "100",
        "TRAIN_LOG_EVERY": "25",
        "VAL_BATCH_SIZE": "65536",  # Larger for audit
        "PYTHONUNBUFFERED": "1",
        "DATA_PATH": str(data_dir),
        "TOKENIZER_PATH": str(tokenizer_path),
        "SEED": str(seed),
    })
    # Apply budget-derived overrides (MAX_WALLCLOCK_SECONDS, ITERATIONS, etc.)
    env.update(budget_env)

    log_path = seed_dir / "train.log"
    cmd = [
        "micromamba", "run", "-n", args.conda_env,
        "python", "-u", str(script_path),
    ]

    print(f"[audit] Seed {seed}: running training...")
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
    print(f"[audit] Seed {seed}: training finished in {wall_elapsed:.1f}s (exit code {return_code})")

    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    parsed = _parse_final_metrics(log_text)

    result = {
        "seed": seed,
        "wall_elapsed_s": round(wall_elapsed, 2),
        "return_code": return_code,
        **parsed,
    }

    # Copy artifacts
    for artifact_name in ("final_model.pt", "final_model.int8.ptz"):
        src = REPO_ROOT / artifact_name
        if src.exists():
            dst = seed_dir / artifact_name
            shutil.copy2(str(src), str(dst))
            result[f"artifact_size_{artifact_name}"] = os.path.getsize(str(dst))
            src.unlink()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit a promoted candidate with larger budget and separate val split."
    )
    parser.add_argument("--script", required=True, help="Path to submission train_gpt.py")
    parser.add_argument("--checkpoint", default=None, help="Path to existing checkpoint (skip training)")
    parser.add_argument("--output-dir", default="proxy_results/audit", help="Output directory")
    parser.add_argument("--time-budget", type=int, default=600, help="MAX_WALLCLOCK_SECONDS (default 600)")
    parser.add_argument("--val-audit-manifest", default=None, help="Path to proxy_val_audit manifest JSON")
    parser.add_argument("--full-val", action="store_true", help="Also evaluate on the full validation set")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337], help="Seeds for multi-seed runs")
    parser.add_argument("--max-gb", type=float, default=10.0, help="VRAM cap in GB for eval phase")
    parser.add_argument("--conda-env", default="parameter-golf", help="Conda/micromamba env name")
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

    # Build budget spec for matched-budget support
    budget_spec = BudgetSpec(
        mode=args.budget_mode,
        value=args.budget_value,
        batch_tokens=32768,
    )

    all_seed_results = []

    if args.checkpoint:
        # Skip training, evaluate an existing checkpoint
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.exists():
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        result = {"seed": None, "checkpoint": str(checkpoint_path), "training_skipped": True}

        # Evaluate on proxy_val_audit
        if args.val_audit_manifest:
            print(f"[audit] Evaluating existing checkpoint on proxy_val_audit...")
            try:
                eval_results = _eval_checkpoint_on_manifest(
                    script_path=str(script_path),
                    checkpoint_path=str(checkpoint_path),
                    val_manifest_path=args.val_audit_manifest,
                    max_gb=args.max_gb,
                )
                result["proxy_val_audit"] = eval_results
                print(f"[audit] proxy_val_audit BPB: {eval_results.get('proxy_val_bpb_approx', 'N/A')}")
            except Exception as e:
                print(f"[ERROR] proxy_val_audit eval failed: {e}")
                result["proxy_val_audit_error"] = str(e)

        # Evaluate on full val
        if args.full_val:
            print(f"[audit] Evaluating existing checkpoint on full validation set...")
            try:
                full_results = _eval_checkpoint_full_val(
                    script_path=str(script_path),
                    checkpoint_path=str(checkpoint_path),
                    max_gb=args.max_gb,
                )
                result["full_val"] = full_results
                print(f"[audit] Full val BPB: {full_results.get('full_val_bpb', 'N/A')}")
            except Exception as e:
                print(f"[ERROR] full val eval failed: {e}")
                result["full_val_error"] = str(e)

        all_seed_results.append(result)

    else:
        # Multi-seed training + evaluation
        for seed in args.seeds:
            seed_dir = out_dir / f"seed_{seed}"
            print(f"\n{'='*60}")
            print(f"[audit] Starting seed {seed}")
            print(f"{'='*60}")

            result = _run_single_seed(script_path, seed, args, seed_dir, budget_spec)

            checkpoint_path = seed_dir / "final_model.pt"

            # Evaluate on proxy_val_audit (NOT proxy_val_tune)
            if args.val_audit_manifest and checkpoint_path.exists():
                print(f"[audit] Seed {seed}: evaluating on proxy_val_audit...")
                try:
                    eval_results = _eval_checkpoint_on_manifest(
                        script_path=str(script_path),
                        checkpoint_path=str(checkpoint_path),
                        val_manifest_path=args.val_audit_manifest,
                        max_gb=args.max_gb,
                    )
                    result["proxy_val_audit"] = eval_results
                    print(f"[audit] proxy_val_audit BPB: {eval_results.get('proxy_val_bpb_approx', 'N/A')}")
                except Exception as e:
                    print(f"[ERROR] proxy_val_audit eval failed: {e}")
                    result["proxy_val_audit_error"] = str(e)

            # Evaluate on full val
            if args.full_val and checkpoint_path.exists():
                print(f"[audit] Seed {seed}: evaluating on full validation set...")
                try:
                    full_results = _eval_checkpoint_full_val(
                        script_path=str(script_path),
                        checkpoint_path=str(checkpoint_path),
                        max_gb=args.max_gb,
                    )
                    result["full_val"] = full_results
                    print(f"[audit] Full val BPB: {full_results.get('full_val_bpb', 'N/A')}")
                except Exception as e:
                    print(f"[ERROR] full val eval failed: {e}")
                    result["full_val_error"] = str(e)

            # Build and save RunSummary for this seed
            steps_completed = result.get("steps_completed", 0)
            batch_tokens = 32768
            tokens_processed = batch_tokens * steps_completed
            budget_env = budget_spec.to_env_overrides()
            budget_iterations = int(budget_env.get("ITERATIONS", "0"))
            budget_wallclock = float(budget_env.get("MAX_WALLCLOCK_SECONDS", "0"))
            wall_elapsed_s = result.get("wall_elapsed_s", 0)

            if steps_completed >= budget_iterations and budget_iterations > 0:
                budget_exhausted = "iterations"
            elif wall_elapsed_s >= budget_wallclock * 0.95:
                budget_exhausted = "wallclock"
            else:
                budget_exhausted = "incomplete"

            summary = RunSummary(
                run_name=seed_dir.name,
                model_name=script_path.parent.name,
                config_path=str(script_path),
                seed=seed,
                git_commit=RunSummary._get_git_commit(),
                budget_mode=args.budget_mode,
                budget_value=args.budget_value,
                target_seq_len=1024,
                effective_batch_tokens=batch_tokens,
                tokens_processed=tokens_processed,
                optimizer_steps=steps_completed,
                train_wallclock_sec=round(wall_elapsed_s, 2),
                train_loss=result.get("train_loss", 0),
                budget_exhausted=budget_exhausted,
                eval_mode="proxy_val_audit" if args.val_audit_manifest else "full_val" if args.full_val else "",
                pre_quant_val_bpb=result.get("train_val_bpb", 0),
                post_quant_val_bpb=result.get("post_quant_val_bpb", 0),
                proxy_val_audit_bpb=result.get("proxy_val_audit", {}).get("proxy_val_bpb_approx", 0),
                artifact_bytes=result.get("artifact_bytes_int8_zlib", 0),
                peak_vram_allocated_gb=round(result.get("train_vram_peak_mib", 0) / 1024, 3),
                status="completed" if result.get("return_code", -1) == 0 else "failed",
                failure_reason="" if result.get("return_code", -1) == 0 else f"exit code {result.get('return_code', -1)}",
            )
            if args.full_val:
                full_bpb = result.get("full_val", {}).get("full_val_bpb", 0)
                if full_bpb:
                    summary.pre_quant_val_bpb = full_bpb
            summary.compute_derived()
            summary_path = out_dir / f"run_summary_seed{seed}.json"
            save_run_summary(summary, summary_path)
            print(f"[audit] RunSummary saved to {summary_path}")

            all_seed_results.append(result)

    # Aggregate multi-seed results
    aggregate = {
        "script": str(script_path),
        "time_budget_s": args.time_budget,
        "max_gb": args.max_gb,
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "per_seed": all_seed_results,
    }

    # Compute aggregate stats if multiple seeds
    if len(all_seed_results) > 1:
        def _safe_mean(values):
            valid = [v for v in values if v is not None]
            return round(sum(valid) / len(valid), 6) if valid else None

        def _safe_std(values):
            valid = [v for v in values if v is not None]
            if len(valid) < 2:
                return None
            mean = sum(valid) / len(valid)
            var = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
            return round(var ** 0.5, 6)

        # Aggregate training metrics
        train_bpb_values = [r.get("train_val_bpb") for r in all_seed_results]
        post_quant_bpb_values = [r.get("post_quant_val_bpb") for r in all_seed_results]

        aggregate["aggregate"] = {
            "train_val_bpb_mean": _safe_mean(train_bpb_values),
            "train_val_bpb_std": _safe_std(train_bpb_values),
            "post_quant_val_bpb_mean": _safe_mean(post_quant_bpb_values),
            "post_quant_val_bpb_std": _safe_std(post_quant_bpb_values),
        }

        # Aggregate proxy_val_audit
        audit_bpb_values = [
            r.get("proxy_val_audit", {}).get("proxy_val_bpb_approx")
            for r in all_seed_results
        ]
        if any(v is not None for v in audit_bpb_values):
            aggregate["aggregate"]["proxy_val_audit_bpb_mean"] = _safe_mean(audit_bpb_values)
            aggregate["aggregate"]["proxy_val_audit_bpb_std"] = _safe_std(audit_bpb_values)

        # Aggregate full val
        full_bpb_values = [
            r.get("full_val", {}).get("full_val_bpb")
            for r in all_seed_results
        ]
        if any(v is not None for v in full_bpb_values):
            aggregate["aggregate"]["full_val_bpb_mean"] = _safe_mean(full_bpb_values)
            aggregate["aggregate"]["full_val_bpb_std"] = _safe_std(full_bpb_values)

    # Save results
    results_path = out_dir / "audit_results.json"
    with open(results_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n[audit] Results saved to {results_path}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

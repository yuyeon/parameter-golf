#!/usr/bin/env python3
"""
Profile anchor models over the FULL official validation split.

Records per-sequence metrics (seq_id, bytes, tokens, loss, bpb, model_name, seed)
and writes them as JSONL (one line per sequence per model).  Supports resumability:
already-profiled (model_name, seed) combos are skipped automatically.

Per-document profiling enables SparseEval-style (arXiv:2602.07909) subset
selection: identify which sequences are most discriminative across models,
then build compact proxy-val sets that preserve ranking signal.
Also supports PreSelect-style (arXiv:2503.00808) difficulty/variance analysis.

Usage (single model):
    python scripts/profile_full_val.py \
        --model-script records/.../train_gpt.py \
        --checkpoint  records/.../final_model.pt \
        --output-dir  profiling_results/

Usage (multiple models via JSON manifest):
    python scripts/profile_full_val.py \
        --manifest models.json \
        --output-dir profiling_results/

    models.json format:
    [
      {"name": "baseline_s1337", "model_script": "...", "checkpoint": "...", "seed": 1337},
      ...
    ]
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path so proxy_framework is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import sentencepiece as spm
import torch

from proxy_framework.data_utils import (
    enumerate_sequences,
    extract_sequences,
    iter_batches,
    load_all_val_tokens,
)
from proxy_framework.model_utils import (
    build_model,
    capture_per_token_loss,
    compute_bpb_per_sequence,
    count_bytes_per_sequence,
    import_submission,
)
from proxy_framework.vram_guard import VRAMGuard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
):
    """Build byte-counting LUTs from a SentencePiece model.

    Mirrors train_gpt.py's build_sentencepiece_luts.
    """
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def _load_completed_combos(output_path: Path) -> set[tuple[str, int]]:
    """Scan existing JSONL to find (model_name, seed) combos already done."""
    completed: set[tuple[str, int]] = set()
    if not output_path.exists():
        return completed
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add((rec["model_name"], rec["seed"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def _parse_model_entries(args) -> list[dict]:
    """Build a list of model entries from CLI args or a JSON manifest."""
    entries: list[dict] = []

    if args.manifest:
        manifest_path = Path(args.manifest)
        with open(manifest_path, "r") as f:
            entries = json.load(f)
        # Validate
        for entry in entries:
            if "model_script" not in entry or "checkpoint" not in entry:
                raise ValueError(
                    "Each manifest entry needs 'model_script' and 'checkpoint' keys"
                )
            entry.setdefault("seed", 1337)
            entry.setdefault(
                "name",
                Path(entry["model_script"]).parent.name,
            )
        return entries

    # CLI pairs --model-script / --checkpoint
    scripts = args.model_script or []
    checkpoints = args.checkpoint or []
    if len(scripts) != len(checkpoints):
        raise ValueError(
            f"Need equal number of --model-script ({len(scripts)}) "
            f"and --checkpoint ({len(checkpoints)}) arguments"
        )
    if not scripts:
        raise ValueError(
            "Provide at least one --model-script/--checkpoint pair or --manifest"
        )
    for script, ckpt in zip(scripts, checkpoints):
        entries.append({
            "name": Path(script).parent.name,
            "model_script": script,
            "checkpoint": ckpt,
            "seed": 1337,
        })
    return entries


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def profile_model(
    entry: dict,
    val_tokens: torch.Tensor,
    seq_infos: list,
    seq_len: int,
    batch_seqs: int,
    max_gb: float,
    output_path: Path,
    device: torch.device,
    use_compile: bool = True,
):
    """Profile a single model over all validation sequences."""
    model_name = entry["name"]
    seed = entry["seed"]
    script_path = entry["model_script"]
    ckpt_path = entry["checkpoint"]

    print(f"\n{'='*70}")
    print(f"Profiling: {model_name}  (seed={seed})")
    print(f"  Script:     {script_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*70}")

    # Import submission module and build model using submission-native constructor
    mod = import_submission(script_path)
    hp = mod.Hyperparameters()
    # Use inspect.signature to discover GPT constructor params, supporting
    # custom architectures (SmearGate, BigramHash, etc.)
    import inspect
    gpt_sig = inspect.signature(mod.GPT.__init__)
    gpt_params = {}
    for param_name in gpt_sig.parameters:
        if param_name == "self":
            continue
        if hasattr(hp, param_name):
            gpt_params[param_name] = getattr(hp, param_name)
    model = mod.GPT(**gpt_params).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, mod.CastedLinear):
            module.float()
    if hasattr(mod, "restore_low_dim_params_to_fp32"):
        mod.restore_low_dim_params_to_fp32(model)

    # Load checkpoint weights
    print(f"  Loading checkpoint ...")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Compile for speed
    if use_compile:
        print(f"  Compiling model with torch.compile ...")
        model = torch.compile(model)

    # Build tokenizer LUTs for byte counting
    print(f"  Building tokenizer LUTs ...")
    tokenizer_path = hp.tokenizer_path if hasattr(hp, "tokenizer_path") else None
    if tokenizer_path is None:
        # Fall back to default
        tokenizer_path = str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model")
    sp = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_path).resolve()))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        _build_sentencepiece_luts(sp, hp.vocab_size, device)
    )

    # All sequence IDs
    all_seq_ids = [s.seq_id for s in seq_infos]
    total_seqs = len(all_seq_ids)
    total_batches = math.ceil(total_seqs / batch_seqs)

    print(f"  Total sequences: {total_seqs}  |  Batch size: {batch_seqs}")
    print(f"  Total batches:   {total_batches}")

    t0 = time.perf_counter()
    n_done = 0

    with VRAMGuard(max_gb=max_gb) as guard:
        guard.start_monitor(interval_s=10.0)
        try:
            with open(output_path, "a") as fout:
                with torch.no_grad():
                    for x, y, batch_ids in iter_batches(
                        val_tokens, all_seq_ids, seq_len, batch_seqs
                    ):
                        x = x.to(device=device, dtype=torch.long)
                        y = y.to(device=device, dtype=torch.long)

                        bsz = x.shape[0]

                        # Capture per-token losses
                        with capture_per_token_loss() as captured:
                            with torch.autocast(
                                device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=True,
                            ):
                                _ = model(x, y)
                        per_token = captured["loss"]  # (bsz * seq_len,)
                        per_seq_loss = per_token.reshape(bsz, seq_len).mean(dim=1)

                        # Byte counts
                        byte_counts = count_bytes_per_sequence(
                            y, x,
                            base_bytes_lut,
                            has_leading_space_lut,
                            is_boundary_token_lut,
                            seq_len,
                        )

                        # BPB
                        per_seq_bpb = compute_bpb_per_sequence(
                            per_seq_loss, byte_counts, seq_len
                        )

                        # Write JSONL records
                        for i, sid in enumerate(batch_ids):
                            rec = {
                                "seq_id": sid,
                                "bytes": int(byte_counts[i].item()),
                                "tokens": seq_len,
                                "loss": float(per_seq_loss[i].item()),
                                "bpb": float(per_seq_bpb[i].item()),
                                "model_name": model_name,
                                "seed": seed,
                            }
                            fout.write(json.dumps(rec) + "\n")

                        n_done += bsz
                        if n_done % (batch_seqs * 50) == 0 or n_done >= total_seqs:
                            elapsed = time.perf_counter() - t0
                            seqs_per_sec = n_done / elapsed if elapsed > 0 else 0
                            vram = guard.check()
                            print(
                                f"  [{model_name}] {n_done}/{total_seqs} seqs "
                                f"({100*n_done/total_seqs:.1f}%)  "
                                f"{seqs_per_sec:.1f} seq/s  "
                                f"VRAM: {vram['allocated_gb']:.2f}GB alloc / "
                                f"{vram['peak_gb']:.2f}GB peak"
                            )
                            fout.flush()
        finally:
            guard.stop_monitor()

    elapsed = time.perf_counter() - t0
    print(f"  Finished {model_name} in {elapsed:.1f}s  ({n_done/elapsed:.1f} seq/s)")

    # Cleanup
    del model, state_dict, sp
    del base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile anchor models on the full validation split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-script",
        nargs="*",
        default=None,
        help="Path(s) to train_gpt.py submission scripts.",
    )
    parser.add_argument(
        "--checkpoint",
        nargs="*",
        default=None,
        help="Path(s) to .pt checkpoint files (one per --model-script).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to a JSON manifest listing models to profile.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write profiling JSONL output.",
    )
    parser.add_argument(
        "--batch-seqs",
        type=int,
        default=4,
        help="Number of sequences per micro-batch (default: 4).",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=10.0,
        help="VRAM cap in GB (default: 10.0).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory for validation shards.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length (default: 1024).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (for debugging).",
    )
    args = parser.parse_args()

    # Resolve output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "profile_full_val.jsonl"

    # Parse model entries
    entries = _parse_model_entries(args)
    print(f"Models to profile: {len(entries)}")
    for e in entries:
        print(f"  - {e['name']}  seed={e['seed']}")

    # Check resumability
    completed = _load_completed_combos(output_path)
    if completed:
        print(f"\nAlready completed ({len(completed)} combos):")
        for name, seed in sorted(completed):
            print(f"  - {name}  seed={seed}")

    # Filter to only remaining entries
    remaining = [
        e for e in entries if (e["name"], e["seed"]) not in completed
    ]
    if not remaining:
        print("\nAll models already profiled. Nothing to do.")
        return
    print(f"\nModels remaining: {len(remaining)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No CUDA device found. Running on CPU (will be very slow).")

    # Load validation tokens
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024")
    print(f"\nLoading validation tokens from {data_dir} ...")
    val_tokens = load_all_val_tokens(data_dir, seq_len=args.seq_len)
    total_tokens = val_tokens.numel() - 1
    print(f"  Total validation tokens: {total_tokens:,}")

    # Enumerate sequences
    seq_infos = enumerate_sequences(val_tokens.numel(), args.seq_len)
    print(f"  Total sequences: {len(seq_infos)}")

    # Profile each remaining model
    for entry in remaining:
        try:
            profile_model(
                entry=entry,
                val_tokens=val_tokens,
                seq_infos=seq_infos,
                seq_len=args.seq_len,
                batch_seqs=args.batch_seqs,
                max_gb=args.max_gb,
                output_path=output_path,
                device=device,
                use_compile=not args.no_compile,
            )
        except Exception as exc:
            print(f"\nERROR profiling {entry['name']}: {exc}")
            import traceback
            traceback.print_exc()
            # Continue with remaining models
            continue

    print(f"\nDone. Results written to: {output_path}")


if __name__ == "__main__":
    main()

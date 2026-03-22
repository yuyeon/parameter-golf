#!/usr/bin/env python3
"""
Eval-only mode: load a saved model checkpoint and run quantization + BPB eval.

Use this when training completed but the process crashed before finishing
quantization or eval (e.g. OOM during the roundtrip validation).

This works by dynamically importing the submission's train_gpt.py and
reusing its model definition, quantization, and eval functions.

Usage:
    python tools/eval_only.py \\
        --script records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \\
        --checkpoint final_model.pt
"""

import argparse
import importlib.util
import io
import os
import sys
import time
import zlib
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def import_submission(script_path):
    """Import a submission's train_gpt.py as a module."""
    spec = importlib.util.spec_from_file_location("submission", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Eval-only: quantize + evaluate a saved model")
    parser.add_argument("--script", required=True, help="Path to submission's train_gpt.py")
    parser.add_argument("--checkpoint", required=True, help="Path to saved model state_dict (.pt)")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    args_cli = parser.parse_args()

    script_path = Path(args_cli.script).resolve()
    checkpoint_path = Path(args_cli.checkpoint).resolve()

    if not script_path.exists():
        sys.exit(f"Script not found: {script_path}")
    if not checkpoint_path.exists():
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args_cli.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set data paths
    os.environ.setdefault("DATA_PATH", str(DATA_DIR / "datasets" / "fineweb10B_sp1024"))
    os.environ.setdefault("TOKENIZER_PATH", str(DATA_DIR / "tokenizers" / "fineweb_1024_bpe.model"))

    print(f"Loading submission module: {script_path.name}")
    mod = import_submission(script_path)
    hp = mod.Hyperparameters()

    # Load tokenizer + validation data
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    val_tokens = mod.load_validation_tokens(hp.val_files, hp.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = mod.build_sentencepiece_luts(
        sp, hp.vocab_size, device
    )
    print(f"Validation tokens: {val_tokens.numel() - 1}")

    # Build model and load checkpoint
    print("Building model...")
    base_model = mod.GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, mod.CastedLinear):
            module.float()
    mod.restore_low_dim_params_to_fp32(base_model)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    base_model.load_state_dict(state_dict, strict=True)

    # Compile for fast eval (uncompiled eval on 62M tokens is way too slow)
    print("Compiling model (this takes a few minutes)...")
    compiled_model = torch.compile(base_model, dynamic=False)

    grad_accum_steps = 8  # single GPU

    # Pre-quant eval
    print("Running pre-quant eval...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = mod.eval_val(
        hp, compiled_model, 0, 1, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    print(f"pre_quant val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
          f"eval_time:{1000 * (time.perf_counter() - t0):.0f}ms")

    # Quantize using the submission's own quantization functions
    print("Quantizing...")
    quant_fn_name = None
    for name in ("quantize_state_dict", "quantize_state_dict_int8"):
        if hasattr(mod, name):
            quant_fn_name = name
            break
    if quant_fn_name is None:
        sys.exit("No quantization function found in submission")

    quant_fn = getattr(mod, quant_fn_name)
    dequant_fn_name = quant_fn_name.replace("quantize", "dequantize")
    if not hasattr(mod, dequant_fn_name):
        sys.exit(f"No {dequant_fn_name} found in submission")
    dequant_fn = getattr(mod, dequant_fn_name)

    # Quantize from base_model (not compiled wrapper) to get clean state_dict
    quant_obj, quant_stats = quant_fn(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()

    # Detect compression: zstd if submission uses it, else zlib
    compress_fn = lambda data: zlib.compress(data, 9)
    decompress_fn = zlib.decompress
    compress_label = "zlib"
    try:
        import pyzstd
        if "zstd" in script_path.read_text():
            compress_fn = lambda data: pyzstd.compress(data, 22)
            decompress_fn = pyzstd.decompress
            compress_label = "zstd"
    except ImportError:
        pass

    quant_blob = compress_fn(quant_raw)

    artifact_bytes = len(quant_blob)
    code_bytes = len(script_path.read_bytes())
    print(f"Artifact: {artifact_bytes} bytes ({compress_label})")
    print(f"Code: {code_bytes} bytes")
    print(f"Total: {artifact_bytes + code_bytes} bytes "
          f"({'OVER' if artifact_bytes + code_bytes > 16_000_000 else 'under'} 16MB)")

    # Roundtrip: decompress, dequantize, load into base_model, eval via compiled_model
    print("Running post-quant roundtrip eval...")
    quant_state = torch.load(io.BytesIO(decompress_fn(quant_blob)), map_location="cpu")
    # Load into base_model — the compiled wrapper sees the updated weights
    base_model.load_state_dict(dequant_fn(quant_state), strict=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    q_val_loss, q_val_bpb = mod.eval_val(
        hp, compiled_model, 0, 1, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    print(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
          f"eval_time:{1000 * (time.perf_counter() - t0):.0f}ms")
    print(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()

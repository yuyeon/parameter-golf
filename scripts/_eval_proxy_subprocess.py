#!/usr/bin/env python3
"""
Submission-native eval-proxy subprocess.

Evaluates a checkpoint on a proxy_val manifest using the submission's OWN
model construction code, not a generic build_model().  This makes it work
for custom architectures (SmearGate, BigramHash, etc.).

Called as a subprocess by the sweep runner:
    python scripts/_eval_proxy_subprocess.py \
        --script <submission_train_gpt.py> \
        --checkpoint <final_model.pt> \
        --manifest <proxy_val_tune.json> \
        --output <result.json> \
        [--max-gb 10.0]

Output JSON:
    {
        "proxy_val_mean_loss": float,
        "proxy_val_bits_per_token": float,
        "proxy_val_n_seqs": int,
        "eval_vram_peak_gb": float,
        "status": "ok" | "error",
        "error": str | null
    }

Note: the metric is BITS PER TOKEN (loss / log(2)), NOT bits-per-byte.
True BPB requires byte counting with the tokenizer, which would require
matching each submission's tokenizer logic.  For ranking purposes,
bits-per-token preserves the same ordering as BPB.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _import_fresh(script_path: str):
    """Import a submission module from scratch (no caching)."""
    # Use a unique module name to avoid cache collisions
    import uuid
    mod_name = f"submission_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-gb", type=float, default=10.0)
    args = parser.parse_args()

    result = {
        "proxy_val_mean_loss": 0,
        "proxy_val_bits_per_token": 0,
        "proxy_val_n_seqs": 0,
        "eval_vram_peak_gb": 0,
        "status": "error",
        "error": None,
    }

    try:
        import torch
        from proxy_framework.data_utils import (
            iter_batches,
            load_all_val_tokens,
            load_manifest,
        )
        from proxy_framework.vram_guard import VRAMGuard

        device = torch.device("cuda")

        # Import the submission module fresh
        mod = _import_fresh(args.script)
        hp = mod.Hyperparameters()

        # Build model using the submission's OWN GPT constructor
        # by inspecting __init__ signature and passing matching hp attrs
        import inspect
        gpt_sig = inspect.signature(mod.GPT.__init__)
        gpt_params = {}
        for param_name in gpt_sig.parameters:
            if param_name == "self":
                continue
            if hasattr(hp, param_name):
                gpt_params[param_name] = getattr(hp, param_name)

        model = mod.GPT(**gpt_params).to(device).bfloat16()

        # Restore precision
        for module in model.modules():
            if isinstance(module, mod.CastedLinear):
                module.float()
        if hasattr(mod, "restore_low_dim_params_to_fp32"):
            mod.restore_low_dim_params_to_fp32(model)

        # Load checkpoint
        state_dict = torch.load(args.checkpoint, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = torch.compile(model, dynamic=False, fullgraph=True)

        # Load manifest and val tokens
        manifest = load_manifest(args.manifest)
        seq_len = manifest.seq_len if manifest.seq_len > 0 else hp.train_seq_len
        seq_ids = manifest.seq_ids

        data_dir = (Path(hp.data_path).resolve()
                    if hasattr(hp, "data_path")
                    else REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024")
        val_tokens = load_all_val_tokens(str(data_dir), seq_len=seq_len)

        guard = VRAMGuard(max_gb=args.max_gb)
        guard.__enter__()
        guard.start_monitor(interval_s=5.0)

        # Eval: monkey-patch F.cross_entropy to get per-token losses
        from proxy_framework.model_utils import capture_per_token_loss

        batch_seqs = max(1, 32768 // seq_len)
        all_losses = []

        model.eval()
        with torch.inference_mode():
            for x, y, batch_ids in iter_batches(
                val_tokens, seq_ids, seq_len, batch_seqs
            ):
                x = x.to(device=device, dtype=torch.int64)
                y = y.to(device=device, dtype=torch.int64)
                bsz = x.shape[0]
                with capture_per_token_loss() as captured:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _ = model(x, y)
                per_token = captured["loss"]
                per_seq = per_token.reshape(bsz, seq_len).mean(dim=1)
                all_losses.append(per_seq.cpu())

        guard.stop_monitor()
        try:
            vram_info = guard.check()
        except RuntimeError:
            vram_info = {"peak_gb": args.max_gb}
        guard.__exit__(None, None, None)

        all_losses_t = torch.cat(all_losses)
        mean_loss = all_losses_t.mean().item()
        bits_per_token = mean_loss / math.log(2.0)

        result.update({
            "proxy_val_mean_loss": round(mean_loss, 6),
            "proxy_val_bits_per_token": round(bits_per_token, 6),
            "proxy_val_n_seqs": len(seq_ids),
            "eval_vram_peak_gb": vram_info.get("peak_gb", 0),
            "status": "ok",
        })

        # Cleanup
        del model, state_dict, val_tokens
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        result["error"] = f"{e}\n{traceback.format_exc()}"

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # Also print for log capture
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

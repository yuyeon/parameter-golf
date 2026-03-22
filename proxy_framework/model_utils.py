"""
Model loading and per-sequence evaluation utilities.

The key challenge: submission models only return mean loss from forward().
We need per-sequence losses for validation profiling.  Solution: temporarily
monkey-patch torch.nn.functional.cross_entropy to capture per-token losses,
then reshape into per-sequence losses.  This works with ANY model architecture
since all submissions ultimately call F.cross_entropy.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import Tensor


def import_submission(script_path: str | Path) -> ModuleType:
    """Import a submission's train_gpt.py as a module without running main()."""
    script_path = Path(script_path).resolve()
    spec = importlib.util.spec_from_file_location("submission", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    # Temporarily block __main__ execution
    old_name = None
    spec.loader.exec_module(mod)
    return mod


def build_model(mod: ModuleType, device: torch.device):
    """Build a model from a submission module's Hyperparameters + GPT class."""
    hp = mod.Hyperparameters()
    model = mod.GPT(
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

    # Restore precision for linear weights and control params
    for module in model.modules():
        if isinstance(module, mod.CastedLinear):
            module.float()
    mod.restore_low_dim_params_to_fp32(model)
    return model, hp


@contextmanager
def capture_per_token_loss():
    """Context manager that patches F.cross_entropy to capture per-token losses.

    Yields a dict with key "loss" containing the per-token loss tensor
    after each forward pass.

    Usage::

        with capture_per_token_loss() as captured:
            mean_loss = model(x, y)
            per_token = captured["loss"]  # shape: (batch * seq_len,)
    """
    original_ce = torch.nn.functional.cross_entropy
    captured = {"loss": None}

    def _patched_ce(input, target, *args, reduction="mean", **kwargs):
        per_token = original_ce(input, target, *args, reduction="none", **kwargs)
        captured["loss"] = per_token.detach()
        if reduction == "mean":
            return per_token.mean()
        elif reduction == "sum":
            return per_token.sum()
        return per_token

    torch.nn.functional.cross_entropy = _patched_ce
    try:
        yield captured
    finally:
        torch.nn.functional.cross_entropy = original_ce


def eval_per_sequence(
    model,
    x: Tensor,
    y: Tensor,
    seq_len: int,
) -> Tensor:
    """Run model forward on a batch and return per-sequence mean loss.

    Args:
        model: GPT model (compiled or not)
        x: (batch, seq_len) input IDs
        y: (batch, seq_len) target IDs
        seq_len: sequence length

    Returns:
        per_seq_loss: (batch,) mean cross-entropy per sequence
    """
    bsz = x.shape[0]
    with capture_per_token_loss() as captured:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            _ = model(x, y)
    per_token = captured["loss"]  # (bsz * seq_len,)
    return per_token.reshape(bsz, seq_len).mean(dim=1)


def compute_bpb_per_sequence(
    per_seq_loss: Tensor,
    byte_counts: Tensor,
    seq_len: int,
) -> Tensor:
    """Convert per-sequence loss to bits-per-byte.

    Args:
        per_seq_loss: (n_seqs,) mean cross-entropy per sequence (nats)
        byte_counts: (n_seqs,) number of UTF-8 bytes per sequence
        seq_len: tokens per sequence

    Returns:
        bpb: (n_seqs,) bits-per-byte per sequence
    """
    bits_per_token = per_seq_loss / math.log(2.0)
    tokens_per_byte = seq_len / byte_counts.float()
    return bits_per_token * tokens_per_byte


def count_bytes_per_sequence(
    y: Tensor,
    x: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len: int,
) -> Tensor:
    """Count UTF-8 bytes per sequence using tokenizer LUTs.

    Mirrors the byte-counting logic from eval_val in train_gpt.py.
    """
    bsz = y.shape[0]
    prev_ids = x.reshape(-1)
    tgt_ids = y.reshape(-1)
    token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int32)
    token_bytes += (
        has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
    ).to(dtype=torch.int32)
    return token_bytes.reshape(bsz, seq_len).sum(dim=1)

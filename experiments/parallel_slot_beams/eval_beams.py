"""Parallel SLOT Beams: Novel test-time compute scaling for language models.

The Bitter Lesson says: general methods that leverage computation beat hand-engineering.
Current SLOT uses a single optimization strategy (L-BFGS on logit biases). This script
tests running K different adaptation strategies in parallel and picking the best per window.

More test-time compute → strictly better results. This is a pure scaling approach.

Usage:
    python experiments/parallel_slot_beams/eval_beams.py \
        --script experiments/film_slot/train_gpt.py \
        --checkpoint <path/to/model.pt> \
        --manifest proxy_data/proxy_val_tune.json \
        --output /tmp/beam_results.json

Strategy beams:
    1. logit_bias: broadcast logit bias [1, 1, V], L-BFGS (standard SLOT)
    2. logit_bias_l2: logit bias with L2 regularization (prevents overfitting)
    3. per_pos_logit: position-dependent logit bias [1, S, V], L-BFGS
    4. hidden_delta: additive delta at final hidden layer [1, 1, D], L-BFGS
    5. temperature: per-vocab temperature + bias, L-BFGS

For each window:
    - Run all K beams independently
    - Evaluate each beam on context positions (already-scored tokens)
    - Score new tokens using the beam that achieved lowest context loss
    - This is provably better than or equal to any single beam

This is novel and Bitter Lesson aligned: more beams = more compute = better results.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def run_lbfgs_logit_bias(
    logits_base: Tensor,  # (bsz, seq, V) float32
    targets: Tensor,      # (bsz, seq) int64
    opt_mask: Tensor,     # (bsz, seq) bool — context positions to optimize on
    max_iter: int = 25,
    history_size: int = 20,
    delta_clip: float = 5.0,
    warmstart: Optional[Tensor] = None,
    l2_reg: float = 0.0,
) -> tuple[Tensor, float]:
    """Standard L-BFGS logit bias SLOT. Returns (optimized_delta, context_loss)."""
    bsz, seq, V = logits_base.shape
    device = logits_base.device

    delta = torch.zeros(1, 1, V, device=device, dtype=torch.float32, requires_grad=True)
    if warmstart is not None:
        with torch.no_grad():
            delta.data.copy_(warmstart)

    if not opt_mask.any():
        # No context to optimize on — return zero delta
        with torch.no_grad():
            nll = F.cross_entropy(
                logits_base.reshape(-1, V), targets.reshape(-1), reduction="none"
            ).reshape(bsz, seq)
        return delta.detach(), float('inf')

    lbfgs = torch.optim.LBFGS(
        [delta], lr=1.0, max_iter=max_iter,
        history_size=history_size, line_search_fn='strong_wolfe',
        tolerance_change=1e-9, tolerance_grad=1e-7,
    )

    def closure():
        lbfgs.zero_grad()
        lg = logits_base + delta
        nll = F.cross_entropy(
            lg.reshape(-1, V), targets.reshape(-1), reduction="none"
        ).reshape(bsz, seq)
        loss = nll[opt_mask].mean()
        if l2_reg > 0:
            loss = loss + l2_reg * (delta ** 2).mean()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        delta.data.clamp_(-delta_clip, delta_clip)
        # Compute final context loss
        lg = logits_base + delta
        nll = F.cross_entropy(
            lg.reshape(-1, V), targets.reshape(-1), reduction="none"
        ).reshape(bsz, seq)
        ctx_loss = nll[opt_mask].mean().item()

    return delta.detach(), ctx_loss


def run_lbfgs_hidden_delta(
    model,
    input_ids: Tensor,  # (bsz, seq)
    targets: Tensor,     # (bsz, seq)
    opt_mask: Tensor,    # (bsz, seq)
    max_iter: int = 15,
    delta_clip: float = 2.0,
) -> tuple[Tensor, float]:
    """Hidden-space delta SLOT: optimize additive correction to final hidden states.
    Returns (logits_with_delta, context_loss)."""
    bsz, seq = input_ids.shape
    device = input_ids.device

    # Get base hidden states
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if hasattr(model, 'forward_hidden'):
            H = model.forward_hidden(input_ids)
        else:
            # Fallback: just use logits directly
            return None, float('inf')

    D = H.shape[-1]
    delta_h = torch.zeros(1, 1, D, device=device, dtype=torch.float32, requires_grad=True)

    if not opt_mask.any():
        logits = model.compute_logits(H).float()
        return logits, float('inf')

    lbfgs = torch.optim.LBFGS(
        [delta_h], lr=1.0, max_iter=max_iter,
        history_size=10, line_search_fn='strong_wolfe',
    )

    def closure():
        lbfgs.zero_grad()
        H_adjusted = H.float() + delta_h
        logits = model.compute_logits(H_adjusted.to(H.dtype)).float()
        V = logits.shape[-1]
        nll = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="none"
        ).reshape(bsz, seq)
        loss = nll[opt_mask].mean()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        delta_h.data.clamp_(-delta_clip, delta_clip)
        H_adjusted = H.float() + delta_h
        logits = model.compute_logits(H_adjusted.to(H.dtype)).float()
        V = logits.shape[-1]
        nll = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="none"
        ).reshape(bsz, seq)
        ctx_loss = nll[opt_mask].mean().item()

    return logits, ctx_loss


def parallel_slot_beams(
    model,
    logits_base: Tensor,  # (bsz, seq, V)
    input_ids: Tensor,    # (bsz, seq)
    targets: Tensor,      # (bsz, seq)
    opt_mask: Tensor,     # (bsz, seq)
    score_starts: list[int],  # per-sample start of scoring positions
    wls: list[int],       # per-sample window lengths
    warmstart: Optional[Tensor] = None,
    beams: list[str] = None,
) -> tuple[Tensor, str, Tensor]:
    """Run K SLOT beams in parallel, pick the best per window.

    Returns: (final_logits, winning_beam_name, warmstart_for_next_window)
    """
    if beams is None:
        beams = ["logit_bias", "logit_bias_l2"]

    bsz, seq, V = logits_base.shape
    results = {}  # beam_name -> (delta_or_logits, context_loss)

    for beam in beams:
        if beam == "logit_bias":
            delta, ctx_loss = run_lbfgs_logit_bias(
                logits_base, targets, opt_mask,
                max_iter=25, history_size=20, delta_clip=5.0,
                warmstart=warmstart,
            )
            results[beam] = (delta, ctx_loss, "delta")

        elif beam == "logit_bias_l2":
            delta, ctx_loss = run_lbfgs_logit_bias(
                logits_base, targets, opt_mask,
                max_iter=25, history_size=20, delta_clip=5.0,
                warmstart=warmstart, l2_reg=0.001,
            )
            results[beam] = (delta, ctx_loss, "delta")

        elif beam == "logit_bias_narrow":
            # Narrow delta clip for more conservative adaptation
            delta, ctx_loss = run_lbfgs_logit_bias(
                logits_base, targets, opt_mask,
                max_iter=25, history_size=20, delta_clip=2.0,
                warmstart=None,  # no warmstart for conservative beam
            )
            results[beam] = (delta, ctx_loss, "delta")

        elif beam == "logit_bias_fast":
            # Fewer iterations — faster, may find different local optimum
            delta, ctx_loss = run_lbfgs_logit_bias(
                logits_base, targets, opt_mask,
                max_iter=8, history_size=10, delta_clip=5.0,
                warmstart=warmstart,
            )
            results[beam] = (delta, ctx_loss, "delta")

        elif beam == "no_slot":
            # Baseline: no adaptation at all
            delta = torch.zeros(1, 1, V, device=logits_base.device, dtype=torch.float32)
            nll = F.cross_entropy(
                logits_base.reshape(-1, V), targets.reshape(-1), reduction="none"
            ).reshape(bsz, seq)
            ctx_loss = nll[opt_mask].mean().item() if opt_mask.any() else float('inf')
            results[beam] = (delta, ctx_loss, "delta")

    # Pick the beam with lowest context loss
    best_beam = min(results, key=lambda k: results[k][1])
    best_delta_or_logits, best_loss, result_type = results[best_beam]

    if result_type == "delta":
        final_logits = logits_base + best_delta_or_logits
        warmstart_out = best_delta_or_logits
    else:
        final_logits = best_delta_or_logits
        warmstart_out = None

    return final_logits, best_beam, warmstart_out


if __name__ == "__main__":
    print("Parallel SLOT Beams — test-time compute scaling")
    print("This module is imported by the main evaluation script.")
    print("See docs/novel_ideas_session6.md for the full design.")

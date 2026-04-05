"""Temperature-Augmented SLOT: jointly optimize temperature + logit bias at test time.

Novel contribution: current SLOT (PR #1350) only optimizes a logit bias delta.
We add a per-window temperature parameter that captures confidence calibration.

logits_corrected = logits / tau + delta

where tau ∈ R (scalar) and delta ∈ R^V (per-vocab bias).

This is Bitter Lesson aligned: more expressive parameterization of the
test-time adaptation, with only 1 extra parameter per window.

The temperature captures: "is the model overconfident or underconfident for this window?"
The bias captures: "which specific tokens does the model over/under-predict?"

Together, they should provide better adaptation than bias alone.
"""

import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path


def eval_with_temp_slot(
    model_forward_fn,  # function(input_ids) -> logits [bsz, seq, V]
    val_tokens: Tensor,
    vocab_size: int,
    seq_len: int = 1024,
    stride: int = 64,
    slot_max_iter: int = 25,
    slot_history: int = 20,
    delta_clip: float = 5.0,
    use_temperature: bool = True,
    use_bias: bool = True,
    focal_tokens: int = 128,
    device: str = "cuda",
):
    """Evaluate with temperature-augmented causal SLOT.

    Returns dict with val_loss and val_bpb for each configuration:
    - no_slot: baseline (no adaptation)
    - bias_only: standard SLOT (logit bias only)
    - temp_only: temperature only (no bias)
    - temp_bias: temperature + bias (our novel contribution)
    """
    total_tok = val_tokens.numel() - 1
    ws_list = list(range(0, total_tok, stride))
    ws_list = [ws for ws in ws_list if min(ws + seq_len, total_tok) - ws >= 1]

    results = {}

    for mode in ["no_slot", "bias_only", "temp_bias"]:
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        token_count = torch.zeros((), device=device, dtype=torch.float64)

        warmstart_delta = None
        warmstart_tau = None

        t0 = time.perf_counter()
        batch_size = 32

        for bi in range(0, len(ws_list), batch_size):
            bws = ws_list[bi:bi+batch_size]
            bsz = len(bws)

            xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wls = []

            for i, ws in enumerate(bws):
                end = min(ws + seq_len, total_tok)
                wl = end - ws
                wls.append(wl)
                ct = val_tokens[ws:end+1].to(dtype=torch.int64, device=device)
                xb[i, :wl] = ct[:-1]
                yb[i, :wl] = ct[1:]

            # Forward pass
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_base = model_forward_fn(xb).float()

            V = logits_base.shape[-1]

            if mode == "no_slot":
                # No adaptation
                nll = F.cross_entropy(
                    logits_base.reshape(-1, V), yb.reshape(-1), reduction="none"
                ).reshape(bsz, seq_len)
            else:
                # Build causal optimization mask
                focal_start = max(seq_len - focal_tokens, 0)
                opt_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
                has_opt = False
                for i, ws in enumerate(bws):
                    wl = wls[i]
                    s = 0 if ws == 0 else max(wl - stride, 0)
                    if s > focal_start:
                        opt_mask[i, focal_start:s] = True
                        has_opt = True

                if mode == "bias_only" and has_opt:
                    # Standard SLOT: logit bias only
                    delta = torch.zeros(1, 1, V, device=device, dtype=torch.float32, requires_grad=True)
                    if warmstart_delta is not None:
                        delta.data.copy_(warmstart_delta)

                    lbfgs = torch.optim.LBFGS(
                        [delta], lr=1.0, max_iter=slot_max_iter,
                        history_size=slot_history, line_search_fn='strong_wolfe',
                    )
                    def closure():
                        lbfgs.zero_grad()
                        lg = logits_base + delta
                        nll_all = F.cross_entropy(
                            lg.reshape(-1, V), yb.reshape(-1), reduction="none"
                        ).reshape(bsz, seq_len)
                        loss = nll_all[opt_mask].mean()
                        loss.backward()
                        return loss
                    lbfgs.step(closure)
                    delta.data.clamp_(-delta_clip, delta_clip)
                    warmstart_delta = delta.detach().clone()

                    with torch.no_grad():
                        nll = F.cross_entropy(
                            (logits_base + delta.detach()).reshape(-1, V),
                            yb.reshape(-1), reduction="none"
                        ).reshape(bsz, seq_len)

                elif mode == "temp_bias" and has_opt:
                    # NOVEL: Temperature + bias
                    delta = torch.zeros(1, 1, V, device=device, dtype=torch.float32, requires_grad=True)
                    # tau starts at 1.0 (no temperature scaling), optimized alongside delta
                    log_tau = torch.zeros(1, device=device, dtype=torch.float32, requires_grad=True)
                    if warmstart_delta is not None:
                        delta.data.copy_(warmstart_delta)
                    if warmstart_tau is not None:
                        log_tau.data.copy_(warmstart_tau)

                    lbfgs = torch.optim.LBFGS(
                        [delta, log_tau], lr=1.0, max_iter=slot_max_iter,
                        history_size=slot_history, line_search_fn='strong_wolfe',
                    )
                    def closure_temp():
                        lbfgs.zero_grad()
                        tau = torch.exp(log_tau).clamp(0.5, 2.0)  # bound temperature
                        lg = logits_base / tau + delta
                        nll_all = F.cross_entropy(
                            lg.reshape(-1, V), yb.reshape(-1), reduction="none"
                        ).reshape(bsz, seq_len)
                        loss = nll_all[opt_mask].mean()
                        loss.backward()
                        return loss
                    lbfgs.step(closure_temp)
                    delta.data.clamp_(-delta_clip, delta_clip)
                    log_tau.data.clamp_(math.log(0.5), math.log(2.0))
                    warmstart_delta = delta.detach().clone()
                    warmstart_tau = log_tau.detach().clone()

                    with torch.no_grad():
                        tau = torch.exp(log_tau).clamp(0.5, 2.0)
                        nll = F.cross_entropy(
                            (logits_base / tau + delta.detach()).reshape(-1, V),
                            yb.reshape(-1), reduction="none"
                        ).reshape(bsz, seq_len)
                else:
                    # No opt positions (first window) — use base logits
                    nll = F.cross_entropy(
                        logits_base.reshape(-1, V), yb.reshape(-1), reduction="none"
                    ).reshape(bsz, seq_len)

            # Score new positions
            for i, ws in enumerate(bws):
                wl = wls[i]
                s = 0 if ws == 0 else max(wl - stride, 0)
                loss_sum += nll[i, s:wl].to(torch.float64).sum()
                token_count += float(wl - s)

        elapsed = time.perf_counter() - t0
        avg_loss = (loss_sum / token_count).item()
        bpb_approx = avg_loss / math.log(2.0)  # approximate BPB (without byte counting)

        results[mode] = {
            "val_loss": avg_loss,
            "bpt": bpb_approx,
            "time_s": elapsed,
        }
        print(f"  {mode}: loss={avg_loss:.4f} bpt={bpb_approx:.4f} time={elapsed:.1f}s")

    return results


if __name__ == "__main__":
    print("Temperature-Augmented SLOT test")
    print("Requires a trained model checkpoint. Run on the 600s baseline.")
    print("Usage: integrate into the main train_gpt.py eval loop.")

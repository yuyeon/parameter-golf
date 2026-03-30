#!/usr/bin/env python3
"""Tier 0: Analyze compression opportunities on a trained checkpoint.

No training needed. Loads the baseline checkpoint and measures:
1. Bitstream packing savings (int6 in dense 6-bit stream vs int8 containers)
2. Row-level adaptive bitwidth (int4-int8 per row, knapsack allocation)
3. Total bytes freed and roundtrip BPB impact.
"""

import glob
import io
import json
import lzma
import math
import os
import struct
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = "experiments/results/tier0_compression"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG = open(os.path.join(RESULTS_DIR, "analysis.log"), "w")


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG.write(line + "\n")
    LOG.flush()


def pack_int6_bitstream(int8_tensor: torch.Tensor) -> bytes:
    """Pack int6 values (stored in int8) into a dense 6-bit stream.
    4 values = 24 bits = 3 bytes."""
    vals = int8_tensor.flatten().numpy().astype(np.int8)
    # Shift to unsigned: [-32,31] -> [0,63]
    unsigned = (vals.astype(np.int16) + 32).astype(np.uint8)
    n = len(unsigned)
    # Pad to multiple of 4
    pad = (4 - n % 4) % 4
    if pad:
        unsigned = np.concatenate([unsigned, np.zeros(pad, dtype=np.uint8)])

    # Pack 4 values into 3 bytes
    out = bytearray()
    for i in range(0, len(unsigned), 4):
        a, b, c, d = unsigned[i], unsigned[i + 1], unsigned[i + 2], unsigned[i + 3]
        byte0 = (a << 2) | (b >> 4)
        byte1 = ((b & 0x0F) << 4) | (c >> 2)
        byte2 = ((c & 0x03) << 6) | d
        out.extend([byte0 & 0xFF, byte1 & 0xFF, byte2 & 0xFF])
    return bytes(out), n  # Return original count for unpacking


def unpack_int6_bitstream(data: bytes, count: int) -> np.ndarray:
    """Unpack dense 6-bit stream back to int8 values."""
    out = []
    idx = 0
    for i in range(0, len(data), 3):
        if idx >= count:
            break
        b0, b1, b2 = data[i], data[i + 1], data[i + 2]
        a = (b0 >> 2) & 0x3F
        b = ((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)
        c = ((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)
        d = b2 & 0x3F
        out.extend([a, b, c, d])
        idx += 4
    # Shift back to signed: [0,63] -> [-32,31]
    arr = np.array(out[:count], dtype=np.int16) - 32
    return arr.astype(np.int8)


def quantize_per_row(tensor, bits):
    """Quantize a 2D tensor with given bit width per row."""
    t = tensor.float()
    clip_range = (1 << (bits - 1)) - 1  # e.g. int6 -> 31, int4 -> 7
    row_max = t.abs().amax(dim=1).clamp_min(1e-12)
    scale = (row_max / clip_range).to(torch.float16)
    q = torch.clamp(torch.round(t / scale.float()[:, None]), -clip_range, clip_range).to(torch.int8)
    return q, scale


def dequant_per_row(q, scale):
    return q.float() * scale.float().unsqueeze(1)


def mse_at_bitwidth(tensor, bits):
    """Compute MSE of quantize-dequantize roundtrip at given bitwidth."""
    q, s = quantize_per_row(tensor, bits)
    recon = dequant_per_row(q, s)
    return (tensor.float() - recon).pow(2).mean().item()


def bytes_per_row_at_bitwidth(cols, bits):
    """Bytes needed to store one row at given bitwidth + fp16 scale."""
    data_bits = cols * bits
    data_bytes = (data_bits + 7) // 8  # Ceiling division
    scale_bytes = 2  # fp16
    return data_bytes + scale_bytes


def main():
    log("=== Tier 0: Compression Analysis ===")

    # Find the most recent baseline checkpoint
    ckpt_paths = sorted(glob.glob("experiments/results/baseline_64m/final_model.pt"))
    if not ckpt_paths:
        # Try finding any checkpoint from a previous run
        ckpt_paths = sorted(glob.glob("**/final_model.pt", recursive=True))
    if not ckpt_paths:
        log("ERROR: No checkpoint found. Run baseline first (01_run_baseline.sh)")
        sys.exit(1)

    ckpt_path = ckpt_paths[0]
    log(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    log(f"State dict keys: {len(state_dict)}")

    # --- Analysis 1: Bitstream Packing ---
    log("\n=== Analysis 1: Bitstream Packing ===")

    total_int6_params = 0
    total_current_bytes = 0  # int8 containers
    total_packed_bytes = 0   # dense 6-bit packing
    int6_matrices = {}

    for name, tensor in state_dict.items():
        if tensor.ndim < 2 or tensor.numel() <= 65536:
            continue
        # These would be quantized to int6 in the SOTA pipeline
        if any(k in name for k in ["attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight"]):
            continue
        if "tok_emb" in name or "lm_head" in name:
            continue  # These stay int8/fp16

        q, s = quantize_per_row(tensor, 6)
        n_params = q.numel()
        total_int6_params += n_params

        # Current: int8 container (1 byte each) + fp16 scales
        current_bytes = n_params + q.shape[0] * 2
        total_current_bytes += current_bytes

        # Packed: 6-bit dense stream + fp16 scales
        packed_data, count = pack_int6_bitstream(q)
        packed_bytes = len(packed_data) + q.shape[0] * 2
        total_packed_bytes += packed_bytes

        savings_pct = (1 - packed_bytes / current_bytes) * 100
        int6_matrices[name] = {
            "shape": list(tensor.shape),
            "current_bytes": current_bytes,
            "packed_bytes": packed_bytes,
            "savings_pct": round(savings_pct, 1),
        }

        # Verify roundtrip
        unpacked = unpack_int6_bitstream(packed_data, count)
        assert np.array_equal(q.numpy().flatten(), unpacked), f"Roundtrip failed for {name}"

    raw_savings = total_current_bytes - total_packed_bytes
    log(f"Total int6 parameters: {total_int6_params:,}")
    log(f"Current (int8 containers): {total_current_bytes:,} bytes ({total_current_bytes/1e6:.2f} MB)")
    log(f"Packed (6-bit dense):      {total_packed_bytes:,} bytes ({total_packed_bytes/1e6:.2f} MB)")
    log(f"Raw savings (pre-compression): {raw_savings:,} bytes ({raw_savings/1e6:.2f} MB)")
    log(f"Savings ratio: {raw_savings/total_current_bytes*100:.1f}%")

    # Now test with compression on top
    log("\nCompression comparison:")
    # Simulate full quantized state as bytes
    current_buf = io.BytesIO()
    packed_buf = io.BytesIO()
    for name, tensor in state_dict.items():
        if tensor.ndim < 2 or tensor.numel() <= 65536:
            continue
        if any(k in name for k in ["attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight"]):
            continue
        if "tok_emb" in name or "lm_head" in name:
            continue
        q, s = quantize_per_row(tensor, 6)
        current_buf.write(q.numpy().tobytes())
        current_buf.write(s.numpy().tobytes())
        packed_data, _ = pack_int6_bitstream(q)
        packed_buf.write(packed_data)
        packed_buf.write(s.numpy().tobytes())

    current_raw = current_buf.getvalue()
    packed_raw = packed_buf.getvalue()

    for comp_name, comp_fn in [("zlib-9", lambda d: zlib.compress(d, 9)),
                                ("lzma-6", lambda d: lzma.compress(d, preset=6))]:
        c_current = comp_fn(current_raw)
        c_packed = comp_fn(packed_raw)
        savings = len(c_current) - len(c_packed)
        log(f"  {comp_name}: current={len(c_current):,}  packed={len(c_packed):,}  "
            f"savings={savings:,} ({savings/1e6:.3f} MB)")

    # --- Analysis 2: Per-Row Adaptive Bitwidth ---
    log("\n=== Analysis 2: Per-Row Adaptive Bitwidth ===")

    bitwidths = [4, 5, 6, 7, 8]
    total_rows = 0
    bitwidth_counts = {b: 0 for b in bitwidths}
    total_uniform_mse = 0.0
    total_adaptive_mse = 0.0
    total_uniform_bytes = 0
    total_adaptive_bytes = 0

    for name, tensor in state_dict.items():
        if tensor.ndim < 2 or tensor.numel() <= 65536:
            continue
        if any(k in name for k in ["attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight"]):
            continue
        if "tok_emb" in name or "lm_head" in name:
            continue

        rows, cols = tensor.shape
        total_rows += rows

        # Uniform int6 baseline
        q6, s6 = quantize_per_row(tensor, 6)
        recon6 = dequant_per_row(q6, s6)
        mse6 = (tensor.float() - recon6).pow(2).sum(dim=1)  # Per-row MSE
        total_uniform_mse += mse6.sum().item()
        total_uniform_bytes += rows * bytes_per_row_at_bitwidth(cols, 6)

        # Per-row: try each bitwidth, pick cheapest that has MSE <= 1.5× int6 MSE
        for r in range(rows):
            row = tensor[r:r+1]
            row_mse6 = mse6[r].item()
            best_bits = 6
            for bits in [4, 5, 6, 7, 8]:
                q_r, s_r = quantize_per_row(row, bits)
                recon_r = dequant_per_row(q_r, s_r)
                row_mse = (row.float() - recon_r).pow(2).sum().item()
                # Accept if MSE is within 50% of int6 MSE (for lower bits)
                # or if it's strictly better (for higher bits, only if bytes saved elsewhere)
                if bits < 6 and row_mse <= 1.5 * max(row_mse6, 1e-12):
                    best_bits = bits
                    break
                elif bits == 6:
                    best_bits = 6
                    break
            # For this greedy approach, just use int6 (proper knapsack would optimize globally)
            # Instead, just measure MSE at each bitwidth for the summary
            bitwidth_counts[best_bits] += 1
            total_adaptive_bytes += bytes_per_row_at_bitwidth(cols, best_bits)
            q_best, s_best = quantize_per_row(row, best_bits)
            recon_best = dequant_per_row(q_best, s_best)
            total_adaptive_mse += (row.float() - recon_best).pow(2).sum().item()

    adaptive_byte_savings = total_uniform_bytes - total_adaptive_bytes
    log(f"Total rows analyzed: {total_rows}")
    log(f"Bitwidth distribution: {dict(bitwidth_counts)}")
    log(f"Uniform int6 total bytes:  {total_uniform_bytes:,} ({total_uniform_bytes/1e6:.2f} MB)")
    log(f"Adaptive bitwidth bytes:   {total_adaptive_bytes:,} ({total_adaptive_bytes/1e6:.2f} MB)")
    log(f"Byte savings: {adaptive_byte_savings:,} ({adaptive_byte_savings/1e6:.3f} MB)")
    log(f"Uniform MSE:  {total_uniform_mse:.6f}")
    log(f"Adaptive MSE: {total_adaptive_mse:.6f}")
    log(f"MSE ratio: {total_adaptive_mse / max(total_uniform_mse, 1e-12):.4f}")

    # --- Summary ---
    log("\n=== SUMMARY ===")
    log(f"Bitpacking savings (after lzma): check values above")
    log(f"Adaptive bitwidth savings: {adaptive_byte_savings/1e6:.3f} MB")
    log(f"Combined potential: could free significant budget for larger model")

    results = {
        "bitpacking": {
            "total_int6_params": total_int6_params,
            "current_bytes": total_current_bytes,
            "packed_bytes": total_packed_bytes,
            "raw_savings_bytes": raw_savings,
        },
        "adaptive_bitwidth": {
            "total_rows": total_rows,
            "bitwidth_distribution": bitwidth_counts,
            "uniform_bytes": total_uniform_bytes,
            "adaptive_bytes": total_adaptive_bytes,
            "savings_bytes": adaptive_byte_savings,
            "uniform_mse": total_uniform_mse,
            "adaptive_mse": total_adaptive_mse,
        },
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log("TIER0_DONE")
    LOG.close()


if __name__ == "__main__":
    main()

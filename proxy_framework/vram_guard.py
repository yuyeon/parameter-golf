"""
VRAM guard utilities for RTX 3080 12GB.

Enforces a strict peak VRAM cap (default 10.0 GB) on all local runs.
Provides a context manager, a decorator, and a periodic-check helper.
"""

import threading
import time

import torch

DEFAULT_MAX_GB = 10.0


def _bytes_to_gb(b: int) -> float:
    return b / (1024 ** 3)


def check_vram(max_gb: float = DEFAULT_MAX_GB, reset: bool = False) -> dict:
    """Check current VRAM usage.  Raises RuntimeError if over cap."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0}
    if reset:
        torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    info = {
        "allocated_gb": round(_bytes_to_gb(allocated), 3),
        "reserved_gb": round(_bytes_to_gb(reserved), 3),
        "peak_gb": round(_bytes_to_gb(peak), 3),
    }
    if _bytes_to_gb(peak) > max_gb:
        raise RuntimeError(
            f"VRAM cap exceeded: peak {info['peak_gb']:.3f} GB > {max_gb:.1f} GB cap. "
            f"(allocated={info['allocated_gb']:.3f} GB, reserved={info['reserved_gb']:.3f} GB)"
        )
    return info


class VRAMGuard:
    """Context manager that enforces a peak VRAM cap.

    Usage::

        with VRAMGuard(max_gb=10.0):
            model = build_model()
            train(model)
        # Raises RuntimeError if peak VRAM exceeded at any check point

    The guard resets peak stats on entry and checks on exit.
    Call ``guard.check()`` manually for intermediate checks.
    """

    def __init__(self, max_gb: float = DEFAULT_MAX_GB):
        self.max_gb = max_gb
        self._monitor_thread = None
        self._stop_event = None

    def check(self) -> dict:
        return check_vram(self.max_gb, reset=False)

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.check()
        return False

    def start_monitor(self, interval_s: float = 5.0):
        """Start a background thread that checks VRAM every interval_s seconds."""
        if self._monitor_thread is not None:
            return
        self._stop_event = threading.Event()

        def _monitor():
            while not self._stop_event.is_set():
                try:
                    check_vram(self.max_gb)
                except RuntimeError as e:
                    import sys
                    print(f"\n[VRAMGuard] FATAL: {e}", file=sys.stderr)
                    import os
                    os._exit(1)
                self._stop_event.wait(interval_s)

        self._monitor_thread = threading.Thread(target=_monitor, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2)
            self._monitor_thread = None
            self._stop_event = None


def safe_batch_size(
    model_vram_gb: float,
    overhead_gb: float = 2.0,
    per_seq_gb: float = 0.1,
    max_gb: float = DEFAULT_MAX_GB,
) -> int:
    """Estimate a safe micro-batch size given VRAM constraints."""
    available = max_gb - model_vram_gb - overhead_gb
    if available <= 0:
        return 1
    return max(1, int(available / per_seq_gb))

"""
VRAM guard utilities for local GPU training.

Enforces a strict peak VRAM cap (default 10.0 GB) on all local runs.
Provides a context manager, a periodic-check helper, and GPU detection
for parallel experiment scheduling.
"""

import threading
import time
from dataclasses import dataclass

import torch

DEFAULT_MAX_GB = 10.0
PARALLEL_HEADROOM_GB = 4.0  # Reserve for CUDA context overhead per process


@dataclass
class GPUInfo:
    """Detected GPU information."""
    name: str
    total_vram_gb: float
    compute_capability: tuple[int, int] = (0, 0)
    multi_processor_count: int = 0


def detect_gpu(device: int = 0) -> GPUInfo:
    """Detect GPU type and total VRAM.

    Returns GPUInfo with name and total VRAM.  If CUDA is unavailable,
    returns a placeholder with 0 VRAM.
    """
    if not torch.cuda.is_available():
        return GPUInfo(name="none", total_vram_gb=0.0)
    props = torch.cuda.get_device_properties(device)
    return GPUInfo(
        name=props.name,
        total_vram_gb=round(props.total_memory / (1024 ** 3), 1),
        compute_capability=(props.major, props.minor),
        multi_processor_count=props.multi_processor_count,
    )


def max_parallel_workers(
    total_vram_gb: float,
    per_worker_gb: float = DEFAULT_MAX_GB,
    headroom_gb: float = PARALLEL_HEADROOM_GB,
) -> int:
    """Calculate how many parallel experiments fit on this GPU.

    Each worker needs ``per_worker_gb`` for model/data plus a share of
    ``headroom_gb`` for CUDA context overhead (driver buffers, etc.).
    """
    if total_vram_gb <= 0 or per_worker_gb <= 0:
        return 0
    # Each process needs per_worker_gb + a per-process CUDA context cost (~0.5-1GB)
    per_process_context_gb = 1.0
    usable = total_vram_gb - headroom_gb
    if usable <= 0:
        return 0
    workers = int(usable / (per_worker_gb + per_process_context_gb))
    return max(1, workers) if total_vram_gb >= per_worker_gb else 0


def memory_fraction_for_worker(
    total_vram_gb: float,
    n_workers: int,
    headroom_gb: float = PARALLEL_HEADROOM_GB,
) -> float:
    """Compute per-process CUDA memory fraction for ``n_workers`` on this GPU.

    Returns a fraction (0.0, 1.0] suitable for
    ``torch.cuda.set_per_process_memory_fraction()``.
    """
    if n_workers <= 1:
        return 1.0
    # Divide usable VRAM equally among workers
    usable = total_vram_gb - headroom_gb
    per_worker = usable / n_workers
    fraction = per_worker / total_vram_gb
    # Clamp to a reasonable range
    return max(0.1, min(1.0, round(fraction, 3)))


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

"""
Tests for proxy_framework.vram_guard VRAM enforcement utilities.

Run with:
    python -m pytest tests/test_vram_guard.py -v

Most tests mock CUDA to avoid requiring a GPU. Tests that need real CUDA
are skipped when torch.cuda.is_available() returns False.
"""

from unittest import mock

import pytest
import torch

from proxy_framework.vram_guard import (
    DEFAULT_MAX_GB,
    VRAMGuard,
    _bytes_to_gb,
    check_vram,
    safe_batch_size,
)


HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# _bytes_to_gb
# ---------------------------------------------------------------------------

class TestBytesToGb:
    def test_zero(self):
        assert _bytes_to_gb(0) == 0.0

    def test_one_gb(self):
        assert _bytes_to_gb(1024 ** 3) == pytest.approx(1.0)

    def test_fractional(self):
        assert _bytes_to_gb(512 * 1024 * 1024) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# check_vram (mocked)
# ---------------------------------------------------------------------------

class TestCheckVRAM:
    def test_no_cuda_returns_zeros(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = False
            info = check_vram(max_gb=10.0)
            assert info["allocated_gb"] == 0
            assert info["reserved_gb"] == 0
            assert info["peak_gb"] == 0

    def test_under_cap_passes(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.memory_allocated.return_value = 1 * 1024 ** 3  # 1 GB
            mock_cuda.memory_reserved.return_value = 2 * 1024 ** 3   # 2 GB
            mock_cuda.max_memory_allocated.return_value = 2 * 1024 ** 3  # 2 GB peak
            info = check_vram(max_gb=10.0)
            assert info["peak_gb"] == pytest.approx(2.0)

    def test_over_cap_raises(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.memory_allocated.return_value = 11 * 1024 ** 3
            mock_cuda.memory_reserved.return_value = 12 * 1024 ** 3
            mock_cuda.max_memory_allocated.return_value = 11 * 1024 ** 3
            with pytest.raises(RuntimeError, match="VRAM cap exceeded"):
                check_vram(max_gb=10.0)

    def test_reset_flag(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.memory_allocated.return_value = 0
            mock_cuda.memory_reserved.return_value = 0
            mock_cuda.max_memory_allocated.return_value = 0
            check_vram(max_gb=10.0, reset=True)
            mock_cuda.reset_peak_memory_stats.assert_called_once()

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device available")
    def test_real_cuda_high_cap(self):
        """With a very high cap, check_vram should always pass."""
        info = check_vram(max_gb=1000.0)
        assert info["peak_gb"] >= 0

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device available")
    def test_real_cuda_low_cap(self):
        """With a 0 GB cap, any allocation should fail."""
        # Allocate a small tensor to ensure peak > 0
        t = torch.zeros(1, device="cuda")
        with pytest.raises(RuntimeError, match="VRAM cap exceeded"):
            check_vram(max_gb=0.0)
        del t


# ---------------------------------------------------------------------------
# VRAMGuard context manager (mocked)
# ---------------------------------------------------------------------------

class TestVRAMGuard:
    def test_context_manager_no_cuda(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = False
            with VRAMGuard(max_gb=10.0) as guard:
                info = guard.check()
                assert info["peak_gb"] == 0

    def test_context_manager_under_cap(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.memory_allocated.return_value = 1 * 1024 ** 3
            mock_cuda.memory_reserved.return_value = 1 * 1024 ** 3
            mock_cuda.max_memory_allocated.return_value = 1 * 1024 ** 3
            with VRAMGuard(max_gb=10.0) as guard:
                pass  # Should not raise

    def test_context_manager_over_cap_on_exit(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.memory_allocated.return_value = 11 * 1024 ** 3
            mock_cuda.memory_reserved.return_value = 12 * 1024 ** 3
            mock_cuda.max_memory_allocated.return_value = 11 * 1024 ** 3
            with pytest.raises(RuntimeError, match="VRAM cap exceeded"):
                with VRAMGuard(max_gb=10.0):
                    pass

    def test_context_manager_suppresses_check_on_exception(self):
        """If an exception occurs inside the block, VRAMGuard should not mask it."""
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.memory_allocated.return_value = 11 * 1024 ** 3
            mock_cuda.memory_reserved.return_value = 12 * 1024 ** 3
            mock_cuda.max_memory_allocated.return_value = 11 * 1024 ** 3
            with pytest.raises(ValueError, match="test error"):
                with VRAMGuard(max_gb=10.0):
                    raise ValueError("test error")

    def test_resets_peak_on_entry(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.memory_allocated.return_value = 0
            mock_cuda.memory_reserved.return_value = 0
            mock_cuda.max_memory_allocated.return_value = 0
            with VRAMGuard(max_gb=10.0):
                pass
            mock_cuda.reset_peak_memory_stats.assert_called()

    def test_default_max_gb(self):
        guard = VRAMGuard()
        assert guard.max_gb == DEFAULT_MAX_GB


# ---------------------------------------------------------------------------
# safe_batch_size
# ---------------------------------------------------------------------------

class TestSafeBatchSize:
    def test_basic_calculation(self):
        # available = 10 - 3 - 2 = 5 GB, per_seq = 0.1 -> 50 seqs
        bs = safe_batch_size(model_vram_gb=3.0, overhead_gb=2.0,
                             per_seq_gb=0.1, max_gb=10.0)
        assert bs == 50

    def test_no_room_returns_one(self):
        # Model alone exceeds the cap minus overhead
        bs = safe_batch_size(model_vram_gb=9.0, overhead_gb=2.0,
                             per_seq_gb=0.1, max_gb=10.0)
        assert bs == 1

    def test_exact_fit(self):
        # available = 10 - 5 - 2 = 3 GB, per_seq = 1.0 -> 3 seqs
        bs = safe_batch_size(model_vram_gb=5.0, overhead_gb=2.0,
                             per_seq_gb=1.0, max_gb=10.0)
        assert bs == 3

    def test_large_per_seq(self):
        # available = 10 - 3 - 2 = 5, per_seq = 10 -> int(0.5) = 0 -> clamped to 1
        bs = safe_batch_size(model_vram_gb=3.0, overhead_gb=2.0,
                             per_seq_gb=10.0, max_gb=10.0)
        assert bs == 1

    def test_fractional_rounds_down(self):
        # available = 10 - 4 - 2 = 4, per_seq = 0.3 -> int(13.33) = 13
        bs = safe_batch_size(model_vram_gb=4.0, overhead_gb=2.0,
                             per_seq_gb=0.3, max_gb=10.0)
        assert bs == 13

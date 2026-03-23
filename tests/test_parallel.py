"""
Tests for proxy_framework.parallel and proxy_framework.vram_guard GPU detection.

Run with:
    python -m pytest tests/test_parallel.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from proxy_framework.vram_guard import (
    GPUInfo,
    detect_gpu,
    max_parallel_workers,
    memory_fraction_for_worker,
)
from proxy_framework.parallel import (
    ExperimentSpec,
    ExperimentResult,
    ParallelRunner,
    _parse_metrics_from_log,
    discover_submissions,
    load_experiment_manifest,
    save_experiment_manifest,
)


# ---------------------------------------------------------------------------
# GPU detection (mocked)
# ---------------------------------------------------------------------------

class TestDetectGPU:
    def test_no_cuda(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = False
            info = detect_gpu()
            assert info.name == "none"
            assert info.total_vram_gb == 0.0

    def test_rtx_3080(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            props = mock.MagicMock()
            props.name = "NVIDIA GeForce RTX 3080"
            props.total_memory = 12 * 1024 ** 3  # 12 GB
            props.major = 8
            props.minor = 6
            props.multi_processor_count = 68
            mock_cuda.get_device_properties.return_value = props
            info = detect_gpu()
            assert info.name == "NVIDIA GeForce RTX 3080"
            assert info.total_vram_gb == pytest.approx(12.0)
            assert info.compute_capability == (8, 6)

    def test_a40(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            props = mock.MagicMock()
            props.name = "NVIDIA A40"
            props.total_memory = 48 * 1024 ** 3
            props.major = 8
            props.minor = 6
            props.multi_processor_count = 84
            mock_cuda.get_device_properties.return_value = props
            info = detect_gpu()
            assert info.name == "NVIDIA A40"
            assert info.total_vram_gb == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# max_parallel_workers
# ---------------------------------------------------------------------------

class TestMaxParallelWorkers:
    def test_a40_default(self):
        # 48 GB total, 4 GB headroom = 44 usable, each needs 10 + 1 = 11 GB
        # 44 / 11 = 4 workers
        workers = max_parallel_workers(48.0, per_worker_gb=10.0, headroom_gb=4.0)
        assert workers == 4

    def test_rtx_3080(self):
        # 12 GB total, 4 GB headroom = 8 usable, each needs 10 + 1 = 11 GB
        # 8 / 11 = 0, but clamped to 1 since 12 >= 10
        workers = max_parallel_workers(12.0, per_worker_gb=10.0, headroom_gb=4.0)
        assert workers == 1

    def test_a100_80gb(self):
        # 80 GB total, 4 headroom = 76 usable, each 11 -> 6 workers
        workers = max_parallel_workers(80.0, per_worker_gb=10.0, headroom_gb=4.0)
        assert workers == 6

    def test_zero_vram(self):
        assert max_parallel_workers(0.0) == 0

    def test_tiny_gpu(self):
        # 4 GB GPU can't fit 10 GB worker
        assert max_parallel_workers(4.0, per_worker_gb=10.0) == 0

    def test_small_worker(self):
        # 12 GB, 4 headroom = 8 usable, each 4 + 1 = 5 -> 1 worker
        workers = max_parallel_workers(12.0, per_worker_gb=4.0, headroom_gb=4.0)
        assert workers == 1

    def test_custom_headroom(self):
        # 48 GB, 8 headroom = 40 usable, each 11 -> 3 workers
        workers = max_parallel_workers(48.0, per_worker_gb=10.0, headroom_gb=8.0)
        assert workers == 3


# ---------------------------------------------------------------------------
# memory_fraction_for_worker
# ---------------------------------------------------------------------------

class TestMemoryFraction:
    def test_single_worker(self):
        assert memory_fraction_for_worker(48.0, 1) == 1.0

    def test_four_workers_a40(self):
        # (48 - 4) / 4 = 11 per worker; 11 / 48 = 0.229
        frac = memory_fraction_for_worker(48.0, 4, headroom_gb=4.0)
        assert 0.2 <= frac <= 0.3

    def test_two_workers(self):
        frac = memory_fraction_for_worker(48.0, 2, headroom_gb=4.0)
        assert 0.4 <= frac <= 0.5

    def test_clamped_above_zero(self):
        # Even with ridiculous params, minimum is 0.1
        frac = memory_fraction_for_worker(10.0, 100, headroom_gb=9.0)
        assert frac >= 0.1


# ---------------------------------------------------------------------------
# ExperimentSpec
# ---------------------------------------------------------------------------

class TestExperimentSpec:
    def test_defaults(self):
        spec = ExperimentSpec(name="test", script="/path/to/train.py")
        assert spec.seed == 1337
        assert spec.budget_mode == "tokens"
        assert spec.budget_value == 16_000_000
        assert spec.batch_tokens == 32768
        assert spec.max_gb == 10.0
        assert spec.extra_env == {}

    def test_to_dict_roundtrip(self):
        spec = ExperimentSpec(
            name="test_exp",
            script="/path/to/train.py",
            seed=42,
            budget_mode="wallclock",
            budget_value=300,
            extra_env={"FOO": "bar"},
        )
        d = spec.to_dict()
        assert d["name"] == "test_exp"
        assert d["seed"] == 42
        assert d["extra_env"] == {"FOO": "bar"}

    def test_from_dict_preserves_extra_env(self):
        d = {
            "name": "test_exp",
            "script": "/path/to/train.py",
            "seed": 42,
            "budget_mode": "tokens",
            "budget_value": 16_000_000,
            "batch_tokens": 32768,
            "max_gb": 10.0,
            "extra_env": {"NUM_LAYERS": "12", "FOO": "bar"},
        }
        spec = ExperimentSpec.from_dict(d)
        assert spec.extra_env == {"NUM_LAYERS": "12", "FOO": "bar"}
        assert spec.name == "test_exp"
        assert spec.seed == 42

    def test_from_dict_empty_extra_env(self):
        d = {"name": "t", "script": "/p.py"}
        spec = ExperimentSpec.from_dict(d)
        assert spec.extra_env == {}

    def test_custom_env(self):
        spec = ExperimentSpec(
            name="test",
            script="/path/to/train.py",
            extra_env={"NUM_LAYERS": "12", "MLP_MULT": "3"},
        )
        assert spec.extra_env["NUM_LAYERS"] == "12"


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------

class TestExperimentResult:
    def test_defaults(self):
        r = ExperimentResult(name="test", spec={})
        assert r.success is False
        assert r.val_bpb == float("inf")
        assert r.steps_completed == 0

    def test_to_dict(self):
        r = ExperimentResult(
            name="test",
            spec={"name": "test"},
            success=True,
            val_bpb=1.234,
            steps_completed=500,
        )
        d = r.to_dict()
        assert d["val_bpb"] == 1.234
        assert d["steps_completed"] == 500


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

class TestParseMetricsFromLog:
    def test_full_log(self):
        log = """\
step:100 | train_loss:4.5678 | val_loss:4.1234 | val_bpb:5.9523 | train_time:50000ms | step_avg:100.0ms
step:200 | train_loss:3.8765 | val_loss:3.5678 | val_bpb:5.1456 | train_time:100000ms | step_avg:100.0ms
peak memory allocated: 6789 MiB
final_int8_zlib_roundtrip val_bpb:5.2345 val_loss:3.6789
Serialized model int8+zlib: 4567890 bytes
"""
        m = _parse_metrics_from_log(log)
        assert m["val_loss"] == pytest.approx(3.5678)
        assert m["val_bpb"] == pytest.approx(5.1456)
        assert m["steps_completed"] == 200
        assert m["train_loss"] == pytest.approx(3.8765)
        assert m["peak_vram_mib"] == 6789
        assert m["post_quant_bpb"] == pytest.approx(5.2345)
        assert m["artifact_bytes"] == 4567890

    def test_empty_log(self):
        assert _parse_metrics_from_log("") == {}

    def test_partial_log(self):
        log = "step:50 | train_loss:5.0 | val_loss:4.5 | val_bpb:6.5 | train_time:25000ms"
        m = _parse_metrics_from_log(log)
        assert m["val_bpb"] == pytest.approx(6.5)
        assert m["steps_completed"] == 50


# ---------------------------------------------------------------------------
# discover_submissions
# ---------------------------------------------------------------------------

class TestDiscoverSubmissions:
    def test_discover(self, tmp_path):
        # Create mock submission directories
        for name in ["sub_A", "sub_B", "sub_C"]:
            d = tmp_path / name
            d.mkdir()
            (d / "train_gpt.py").write_text("# mock training script")

        specs = discover_submissions(tmp_path)
        assert len(specs) == 3
        names = [s.name for s in specs]
        assert "sub_A" in names
        assert "sub_B" in names
        assert "sub_C" in names

    def test_empty_dir(self, tmp_path):
        specs = discover_submissions(tmp_path)
        assert specs == []

    def test_nested_pattern(self, tmp_path):
        # Different glob pattern
        d = tmp_path / "track" / "sub_X"
        d.mkdir(parents=True)
        (d / "train_gpt.py").write_text("# mock")

        specs = discover_submissions(tmp_path / "track")
        assert len(specs) == 1
        assert specs[0].name == "sub_X"


# ---------------------------------------------------------------------------
# Experiment manifest I/O
# ---------------------------------------------------------------------------

class TestExperimentManifest:
    def test_roundtrip(self, tmp_path):
        specs = [
            ExperimentSpec(name="exp1", script="/path/a.py", seed=42),
            ExperimentSpec(name="exp2", script="/path/b.py", seed=7),
        ]
        manifest_path = tmp_path / "manifest.json"
        save_experiment_manifest(specs, manifest_path)
        loaded = load_experiment_manifest(manifest_path)
        assert len(loaded) == 2
        assert loaded[0].name == "exp1"
        assert loaded[0].seed == 42
        assert loaded[1].name == "exp2"

    def test_with_defaults(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        data = {
            "defaults": {"budget_mode": "wallclock", "budget_value": 300, "seed": 42},
            "experiments": [
                {"name": "exp1", "script": "/a.py"},
                {"name": "exp2", "script": "/b.py", "seed": 7},
            ],
        }
        with open(manifest_path, "w") as f:
            json.dump(data, f)

        loaded = load_experiment_manifest(manifest_path)
        assert len(loaded) == 2
        assert loaded[0].budget_mode == "wallclock"
        assert loaded[0].budget_value == 300
        assert loaded[0].seed == 42  # from defaults
        assert loaded[1].seed == 7   # overridden


# ---------------------------------------------------------------------------
# ParallelRunner construction
# ---------------------------------------------------------------------------

class TestParallelRunner:
    def test_init_no_cuda(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = False
            runner = ParallelRunner(output_dir="/tmp/test_par")
            assert runner.max_workers == 1  # fallback
            assert runner.gpu.total_vram_gb == 0.0

    def test_init_a40(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            props = mock.MagicMock()
            props.name = "NVIDIA A40"
            props.total_memory = 48 * 1024 ** 3
            props.major = 8
            props.minor = 6
            props.multi_processor_count = 84
            mock_cuda.get_device_properties.return_value = props
            runner = ParallelRunner(output_dir="/tmp/test_par")
            assert runner.max_workers == 4
            assert runner.gpu.name == "NVIDIA A40"

    def test_max_workers_override(self):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            props = mock.MagicMock()
            props.name = "NVIDIA A40"
            props.total_memory = 48 * 1024 ** 3
            props.major = 8
            props.minor = 6
            props.multi_processor_count = 84
            mock_cuda.get_device_properties.return_value = props
            runner = ParallelRunner(output_dir="/tmp/test_par", max_workers=2)
            assert runner.max_workers == 2


# ---------------------------------------------------------------------------
# ParallelRunner.print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_with_results(self, capsys):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = False
            runner = ParallelRunner(output_dir="/tmp/test_par")
            results = [
                ExperimentResult(
                    name="exp_A", spec={}, success=True,
                    val_bpb=5.2, post_quant_bpb=5.3,
                    steps_completed=500, wallclock_sec=120,
                    peak_vram_mib=6500,
                ),
                ExperimentResult(
                    name="exp_B", spec={}, success=True,
                    val_bpb=5.0, post_quant_bpb=5.1,
                    steps_completed=500, wallclock_sec=130,
                    peak_vram_mib=7000,
                ),
                ExperimentResult(
                    name="exp_C", spec={}, success=False,
                    error="OOM",
                ),
            ]
            text = runner.print_summary(results)
            assert "exp_B" in text  # should be ranked #1 (lower BPB)
            assert "exp_A" in text
            assert "Failed" in text
            assert "OOM" in text

    def test_all_failed(self, capsys):
        with mock.patch("proxy_framework.vram_guard.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = False
            runner = ParallelRunner(output_dir="/tmp/test_par")
            results = [
                ExperimentResult(name="exp_X", spec={}, success=False, error="crash"),
            ]
            text = runner.print_summary(results)
            assert "Failed" in text


# ---------------------------------------------------------------------------
# GPUInfo dataclass
# ---------------------------------------------------------------------------

class TestGPUInfo:
    def test_defaults(self):
        info = GPUInfo(name="test", total_vram_gb=24.0)
        assert info.compute_capability == (0, 0)
        assert info.multi_processor_count == 0

    def test_all_fields(self):
        info = GPUInfo(
            name="NVIDIA A40",
            total_vram_gb=48.0,
            compute_capability=(8, 6),
            multi_processor_count=84,
        )
        assert info.name == "NVIDIA A40"
        assert info.total_vram_gb == 48.0

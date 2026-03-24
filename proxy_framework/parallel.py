"""
Parallel experiment execution for large-GPU environments.

When a GPU has enough VRAM to host multiple experiments simultaneously
(e.g. A40 48 GB can run 4x 10 GB experiments), this module schedules
them as concurrent subprocesses with per-process memory limits.

Each individual experiment still respects RTX 3080 constraints
(10 GB VRAM, ~20 min runtime) so results transfer directly to the
target hardware.

Design:
    - Each experiment runs as a separate *process* (not thread) so that
      PyTorch's CUDA memory tracking is per-process.
    - ``torch.cuda.set_per_process_memory_fraction()`` is injected via
      a tiny wrapper script to hard-limit each process's GPU memory.
    - Artifact collisions (final_model.pt) are avoided by giving each
      worker a unique working directory.
    - A ``ThreadPoolExecutor`` manages the subprocess pool (each thread
      babysits one subprocess, reading its stdout).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from proxy_framework.budget import BudgetSpec, RunSummary, save_run_summary
from proxy_framework.vram_guard import (
    DEFAULT_MAX_GB,
    detect_gpu,
    max_parallel_workers,
    memory_fraction_for_worker,
)


# ---------------------------------------------------------------------------
# Experiment specification & results
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSpec:
    """Definition of a single screening experiment."""

    name: str
    script: str                     # absolute path to train_gpt.py
    seed: int = 1337
    budget_mode: str = "tokens"     # wallclock | tokens | optimizer_steps
    budget_value: float = 16_000_000
    batch_tokens: int = 32768
    max_gb: float = DEFAULT_MAX_GB
    extra_env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentSpec:
        d = dict(d)
        extra = d.pop("extra_env", {})
        return cls(**d, extra_env=extra)


@dataclass
class ExperimentResult:
    """Outcome of a completed experiment."""

    name: str
    spec: dict                      # ExperimentSpec as dict
    success: bool = False
    return_code: int = -1
    # Training metrics
    val_loss: float = float("inf")
    val_bpb: float = float("inf")
    post_quant_bpb: float = float("inf")
    train_loss: float = float("inf")
    steps_completed: int = 0
    tokens_processed: int = 0
    # Resources
    peak_vram_mib: int = 0
    wallclock_sec: float = 0.0
    artifact_bytes: int = 0
    # Paths
    output_dir: str = ""
    log_path: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Log parsing (shared with run_local_screen.py)
# ---------------------------------------------------------------------------

def _parse_metrics_from_log(log_text: str) -> dict:
    """Extract key metrics from a training log."""
    metrics: dict = {}

    for line in reversed(log_text.splitlines()):
        if "val_bpb:" in line and "step:" in line:
            m = re.search(r"val_loss:([\d.]+)", line)
            if m:
                metrics["val_loss"] = float(m.group(1))
            m = re.search(r"val_bpb:([\d.]+)", line)
            if m:
                metrics["val_bpb"] = float(m.group(1))
            m = re.search(r"step:(\d+)", line)
            if m:
                metrics["steps_completed"] = int(m.group(1))
            m = re.search(r"train_time:(\d+)ms", line)
            if m:
                metrics["train_time_ms"] = int(m.group(1))
            break

    for line in reversed(log_text.splitlines()):
        if "train_loss:" in line:
            m = re.search(r"train_loss:([\d.]+)", line)
            if m:
                metrics["train_loss"] = float(m.group(1))
            break

    for line in log_text.splitlines():
        if "peak memory allocated:" in line:
            m = re.search(r"peak memory allocated:\s*(\d+)\s*MiB", line)
            if m:
                metrics["peak_vram_mib"] = int(m.group(1))

    for line in log_text.splitlines():
        if "final_int8_zlib_roundtrip " in line:
            m = re.search(r"val_bpb:([\d.]+)", line)
            if m:
                metrics["post_quant_bpb"] = float(m.group(1))

    for line in log_text.splitlines():
        if "Serialized model int8+zlib:" in line:
            m = re.search(r"int8\+zlib:\s*(\d+)\s*bytes", line)
            if m:
                metrics["artifact_bytes"] = int(m.group(1))

    return metrics


# ---------------------------------------------------------------------------
# Memory-limiting wrapper
# ---------------------------------------------------------------------------

_WRAPPER_TEMPLATE = """\
import os, sys, torch
# Limit this process to {fraction:.3f} of total GPU memory
torch.cuda.set_per_process_memory_fraction({fraction:.3f})
# Execute the actual training script
exec(open(sys.argv[1]).read())
"""


def _write_mem_wrapper(dest_dir: Path, fraction: float) -> Path:
    """Write a tiny Python wrapper that sets CUDA memory fraction."""
    wrapper = dest_dir / "_mem_limit_wrapper.py"
    wrapper.write_text(_WRAPPER_TEMPLATE.format(fraction=fraction))
    return wrapper


# ---------------------------------------------------------------------------
# Single experiment worker
# ---------------------------------------------------------------------------

def _run_single_experiment(
    spec: ExperimentSpec,
    output_dir: Path,
    repo_root: Path,
    mem_fraction: float = 1.0,
    conda_env: str = "parameter-golf",
) -> ExperimentResult:
    """Run one experiment in a subprocess with memory isolation.

    The training script runs in a unique temp directory to avoid
    artifact file collisions (final_model.pt) between parallel workers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"pgolf_{spec.name}_"))

    # Build budget spec -> env overrides
    budget = BudgetSpec(
        mode=spec.budget_mode,
        value=spec.budget_value,
        batch_tokens=spec.batch_tokens,
    )
    budget_env = budget.to_env_overrides()

    data_dir = (repo_root / "data" / "datasets" / "fineweb10B_sp1024").resolve()
    tokenizer_path = (repo_root / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve()

    env = os.environ.copy()
    env.update({
        "TRAIN_BATCH_TOKENS": str(spec.batch_tokens),
        "WARMUP_STEPS": "5",
        "WARMDOWN_ITERS": "150",
        "VAL_LOSS_EVERY": "100",
        "TRAIN_LOG_EVERY": "25",
        "VAL_BATCH_SIZE": "32768",
        "PYTHONUNBUFFERED": "1",
        "DATA_PATH": str(data_dir),
        "TOKENIZER_PATH": str(tokenizer_path),
        "SEED": str(spec.seed),
        # Force consistent eval seq_len so BPB is comparable across submissions
        "EVAL_SEQ_LEN": str(budget.seq_len),
    })
    env.update(budget_env)
    env.update(spec.extra_env)

    # Build command: use memory-limiting wrapper if running parallel
    script_path = str(Path(spec.script).resolve())
    if conda_env == "none":
        python_cmd = ["python", "-u"]
    else:
        _mamba = shutil.which("micromamba")
        if _mamba:
            python_cmd = ["micromamba", "run", "-n", conda_env, "python", "-u"]
        else:
            python_cmd = ["python", "-u"]

    if mem_fraction < 1.0:
        wrapper_path = _write_mem_wrapper(work_dir, mem_fraction)
        cmd = [*python_cmd, str(wrapper_path), script_path]
    else:
        cmd = [*python_cmd, script_path]

    log_path = output_dir / "train.log"

    wall_t0 = time.time()
    proc = None
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(work_dir),
            )
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
            proc.wait()
    except Exception as e:
        wall_elapsed = time.time() - wall_t0
        return ExperimentResult(
            name=spec.name,
            spec=spec.to_dict(),
            success=False,
            error=str(e),
            wallclock_sec=round(wall_elapsed, 2),
            output_dir=str(output_dir),
            log_path=str(log_path),
        )

    wall_elapsed = time.time() - wall_t0
    return_code = proc.returncode if proc else -1

    # Parse log
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    parsed = _parse_metrics_from_log(log_text)

    # Copy artifacts from work_dir to output_dir
    artifact_sizes = {}
    for artifact_name in ("final_model.pt", "final_model.int8.ptz"):
        src = work_dir / artifact_name
        if src.exists():
            dst = output_dir / artifact_name
            shutil.copy2(str(src), str(dst))
            artifact_sizes[artifact_name] = os.path.getsize(str(dst))

    # Clean up work dir
    shutil.rmtree(work_dir, ignore_errors=True)

    # Build RunSummary
    steps = parsed.get("steps_completed", 0)
    tokens = spec.batch_tokens * steps
    budget_iterations = int(budget_env.get("ITERATIONS", "0"))
    budget_wallclock = float(budget_env.get("MAX_WALLCLOCK_SECONDS", "0"))

    if steps >= budget_iterations and budget_iterations > 0:
        budget_exhausted = "iterations"
    elif wall_elapsed >= budget_wallclock * 0.95:
        budget_exhausted = "wallclock"
    else:
        budget_exhausted = "incomplete"

    summary = RunSummary(
        run_name=spec.name,
        model_name=Path(spec.script).parent.name,
        config_path=spec.script,
        seed=spec.seed,
        git_commit=RunSummary._get_git_commit(),
        budget_mode=spec.budget_mode,
        budget_value=spec.budget_value,
        target_seq_len=1024,
        effective_batch_tokens=spec.batch_tokens,
        tokens_processed=tokens,
        optimizer_steps=steps,
        train_wallclock_sec=round(wall_elapsed, 2),
        train_loss=parsed.get("train_loss", 0),
        budget_exhausted=budget_exhausted,
        eval_mode="screen",
        pre_quant_val_bpb=parsed.get("val_bpb", 0),
        post_quant_val_bpb=parsed.get("post_quant_bpb", 0),
        artifact_bytes=parsed.get("artifact_bytes", 0),
        peak_vram_allocated_gb=round(parsed.get("peak_vram_mib", 0) / 1024, 3),
        status="completed" if return_code == 0 else "failed",
        failure_reason="" if return_code == 0 else f"exit code {return_code}",
    )
    summary.compute_derived()
    save_run_summary(summary, output_dir / "run_summary.json")

    return ExperimentResult(
        name=spec.name,
        spec=spec.to_dict(),
        success=(return_code == 0),
        return_code=return_code,
        val_loss=parsed.get("val_loss", float("inf")),
        val_bpb=parsed.get("val_bpb", float("inf")),
        post_quant_bpb=parsed.get("post_quant_bpb", float("inf")),
        train_loss=parsed.get("train_loss", float("inf")),
        steps_completed=steps,
        tokens_processed=tokens,
        peak_vram_mib=parsed.get("peak_vram_mib", 0),
        wallclock_sec=round(wall_elapsed, 2),
        artifact_bytes=parsed.get("artifact_bytes", 0),
        output_dir=str(output_dir),
        log_path=str(log_path),
    )


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

class ParallelRunner:
    """Run multiple experiments concurrently on a single large GPU.

    Auto-detects available VRAM and calculates safe parallelism.
    Each experiment enforces the same VRAM cap as an RTX 3080 run,
    so results are directly comparable.

    Usage::

        runner = ParallelRunner(
            output_dir="parallel_results/batch_001",
            per_worker_gb=10.0,
        )
        results = runner.run(experiments)
        runner.print_summary(results)
    """

    def __init__(
        self,
        output_dir: str | Path = "parallel_results",
        per_worker_gb: float = DEFAULT_MAX_GB,
        max_workers: int | None = None,
        conda_env: str = "parameter-golf",
        repo_root: str | Path | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.per_worker_gb = per_worker_gb
        self.conda_env = conda_env
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent

        # Detect GPU and compute parallelism
        self.gpu = detect_gpu()
        if max_workers is not None:
            self.max_workers = max_workers
        else:
            self.max_workers = max_parallel_workers(
                self.gpu.total_vram_gb, per_worker_gb
            )
            # Fall back to 1 if detection fails
            if self.max_workers == 0:
                self.max_workers = 1

        self.mem_fraction = memory_fraction_for_worker(
            self.gpu.total_vram_gb, self.max_workers
        )

    def run(
        self,
        experiments: Sequence[ExperimentSpec],
        progress_callback: callable | None = None,
    ) -> list[ExperimentResult]:
        """Run all experiments, returning results in completion order.

        Args:
            experiments: List of experiments to run.
            progress_callback: Optional ``fn(completed, total, result)``
                called after each experiment finishes.

        Returns:
            List of ExperimentResult in the same order as ``experiments``.
        """
        n = len(experiments)
        use_fraction = self.mem_fraction if self.max_workers > 1 else 1.0

        print(f"[parallel] GPU: {self.gpu.name} ({self.gpu.total_vram_gb:.1f} GB)")
        print(f"[parallel] Workers: {self.max_workers}, "
              f"VRAM/worker: {self.per_worker_gb:.1f} GB, "
              f"mem fraction: {use_fraction:.3f}")
        print(f"[parallel] Experiments: {n}")
        print(f"[parallel] Output: {self.output_dir}")
        print()

        results_map: dict[str, ExperimentResult] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_name = {}
            for spec in experiments:
                exp_dir = self.output_dir / spec.name
                future = pool.submit(
                    _run_single_experiment,
                    spec=spec,
                    output_dir=exp_dir,
                    repo_root=self.repo_root,
                    mem_fraction=use_fraction,
                    conda_env=self.conda_env,
                )
                future_to_name[future] = spec.name

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = ExperimentResult(
                        name=name,
                        spec={},
                        success=False,
                        error=str(e),
                    )
                results_map[name] = result
                completed += 1

                status = "OK" if result.success else "FAIL"
                bpb_str = (
                    f"BPB={result.val_bpb:.4f}"
                    if result.val_bpb < float("inf")
                    else "no BPB"
                )
                print(
                    f"[parallel] [{completed}/{n}] {name}: {status} "
                    f"({bpb_str}, {result.wallclock_sec:.0f}s, "
                    f"VRAM={result.peak_vram_mib}MiB)"
                )

                if progress_callback:
                    progress_callback(completed, n, result)

        # Return in original order
        return [results_map[spec.name] for spec in experiments]

    def print_summary(self, results: list[ExperimentResult]) -> str:
        """Print a ranked summary table and return it as a string."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"PARALLEL SCREENING RESULTS")
        lines.append(f"GPU: {self.gpu.name} ({self.gpu.total_vram_gb:.1f} GB), "
                      f"{self.max_workers} workers")
        lines.append(f"{'='*80}")

        if successful:
            # Sort by val_bpb (lower is better)
            ranked = sorted(successful, key=lambda r: r.val_bpb)
            lines.append(f"\n{'Rank':<6} {'Name':<45} {'Val BPB':>9} "
                          f"{'PQ BPB':>9} {'Steps':>7} {'Time':>7} {'VRAM':>7}")
            lines.append(f"{'-'*6} {'-'*45} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")

            for i, r in enumerate(ranked, 1):
                pq = f"{r.post_quant_bpb:.4f}" if r.post_quant_bpb < float("inf") else "N/A"
                vbpb = f"{r.val_bpb:.4f}" if r.val_bpb < float("inf") else "N/A"
                lines.append(
                    f"#{i:<5} {r.name:<45} {vbpb:>9} {pq:>9} "
                    f"{r.steps_completed:>7} {r.wallclock_sec:>6.0f}s "
                    f"{r.peak_vram_mib:>5}Mi"
                )

        if failed:
            lines.append(f"\nFailed ({len(failed)}):")
            for r in failed:
                lines.append(f"  {r.name}: {r.error or f'exit code {r.return_code}'}")

        lines.append("")
        summary_text = "\n".join(lines)
        print(summary_text)

        # Save summary
        summary_path = self.output_dir / "parallel_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "gpu": self.gpu.name,
                "gpu_vram_gb": self.gpu.total_vram_gb,
                "max_workers": self.max_workers,
                "mem_fraction": self.mem_fraction,
                "per_worker_gb": self.per_worker_gb,
                "n_experiments": len(results),
                "n_successful": len(successful),
                "n_failed": len(failed),
                "results": [r.to_dict() for r in results],
            }, f, indent=2)

        return summary_text


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def discover_submissions(
    records_dir: str | Path,
    pattern: str = "*/train_gpt.py",
) -> list[ExperimentSpec]:
    """Build ExperimentSpecs from submission directories.

    Scans ``records_dir`` for directories matching ``pattern`` and
    creates one ExperimentSpec per submission.
    """
    records_dir = Path(records_dir)
    specs = []
    for script in sorted(records_dir.glob(pattern)):
        name = script.parent.name
        specs.append(ExperimentSpec(
            name=name,
            script=str(script.resolve()),
        ))
    return specs


def load_experiment_manifest(path: str | Path) -> list[ExperimentSpec]:
    """Load experiment specs from a JSON manifest file.

    Expected format::

        {
          "defaults": {"budget_mode": "tokens", "budget_value": 16000000, ...},
          "experiments": [
            {"name": "exp1", "script": "path/to/train_gpt.py", ...},
            ...
          ]
        }
    """
    with open(path) as f:
        data = json.load(f)

    defaults = data.get("defaults", {})
    specs = []
    for exp in data["experiments"]:
        merged = {**defaults, **exp}
        specs.append(ExperimentSpec(
            name=merged["name"],
            script=merged["script"],
            seed=merged.get("seed", 1337),
            budget_mode=merged.get("budget_mode", "tokens"),
            budget_value=merged.get("budget_value", 16_000_000),
            batch_tokens=merged.get("batch_tokens", 32768),
            max_gb=merged.get("max_gb", DEFAULT_MAX_GB),
            extra_env=merged.get("extra_env", {}),
        ))
    return specs


def save_experiment_manifest(
    specs: Sequence[ExperimentSpec],
    path: str | Path,
    defaults: dict | None = None,
) -> None:
    """Save experiment specs to a JSON manifest file."""
    data = {
        "defaults": defaults or {},
        "experiments": [s.to_dict() for s in specs],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

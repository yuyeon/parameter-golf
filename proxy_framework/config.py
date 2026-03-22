"""
Configuration system using dataclasses + YAML.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    data_dir: str = "data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model"
    train_shards: int = 10       # how many train shards to use
    seq_len: int = 1024


@dataclass
class ModelConfig:
    """Mirrors Hyperparameters from train_gpt.py but only the model shape."""
    vocab_size: int = 1024
    num_layers: int = 9
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 2
    tie_embeddings: bool = True
    rope_base: float = 10000.0
    logit_softcap: float = 30.0
    qk_gain_init: float = 1.5
    tied_embed_init_std: float = 0.005


@dataclass
class TrainConfig:
    budget_mode: str = "tokens"  # "wallclock" | "tokens" | "optimizer_steps"
    budget_value: float = 16_000_000  # default: 16M tokens
    train_batch_tokens: int = 32768
    max_wallclock_seconds: float = 300.0
    iterations: int = 20000
    warmup_steps: int = 5
    warmdown_iters: int = 150
    matrix_lr: float = 0.04
    scalar_lr: float = 0.04
    embed_lr: float = 0.6
    tied_embed_lr: float = 0.05
    head_lr: float = 0.008
    muon_momentum: float = 0.95
    seed: int = 1337


@dataclass
class EvalConfig:
    val_batch_seqs: int = 8        # micro-batch size in sequences
    eval_compiled: bool = True     # use torch.compile for eval
    sliding_window: bool = False
    sliding_stride: int = 64


@dataclass
class VRAMConfig:
    max_gb: float = 10.0
    monitor_interval_s: float = 5.0
    enable_monitor: bool = True


@dataclass
class ProxyConfig:
    """Top-level config for a proxy framework run."""
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    vram: VRAMConfig = field(default_factory=VRAMConfig)

    # Subset references (paths to manifest JSONs)
    proxy_train_manifest: str | None = None
    proxy_val_tune_manifest: str | None = None
    proxy_val_audit_manifest: str | None = None
    proxy_val_long_manifest: str | None = None

    # Output
    output_dir: str = "proxy_results"


def load_config(path: str | Path) -> ProxyConfig:
    """Load config from YAML or JSON."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            raw = yaml.safe_load(text)
        except ImportError:
            raise ImportError("pip install pyyaml to use YAML configs")
    else:
        raw = json.loads(text)
    return _dict_to_config(raw)


def save_config(config: ProxyConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(config)
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path, "w") as f:
                yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fall back to JSON
            path = path.with_suffix(".json")
            with open(path, "w") as f:
                json.dump(d, f, indent=2)
    else:
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


def _dict_to_config(d: dict[str, Any]) -> ProxyConfig:
    """Recursively build ProxyConfig from a flat/nested dict."""
    kw = {}
    field_types = {
        "data": DataConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "eval": EvalConfig,
        "vram": VRAMConfig,
    }
    for k, v in d.items():
        if k in field_types and isinstance(v, dict):
            kw[k] = field_types[k](**v)
        else:
            kw[k] = v
    return ProxyConfig(**kw)

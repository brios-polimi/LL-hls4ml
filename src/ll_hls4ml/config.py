"""Load experiment configuration from YAML with env overrides."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "default.yaml"


def _repo_root() -> Path:
    return _REPO_ROOT


def _resolve_path(value: str, base: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _substitute_vars(obj, variables: dict):
    if isinstance(obj, dict):
        return {k: _substitute_vars(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_vars(v, variables) for v in obj]
    if isinstance(obj, str):
        pattern = re.compile(r"\$\{(\w+)\}")

        def repl(match):
            key = match.group(1)
            if key not in variables:
                raise KeyError(f"Unknown config variable: {key}")
            return str(variables[key])

        return pattern.sub(repl, obj)
    return obj


@dataclass
class Config:
    data_root: Path
    graph_dir: Path
    tensor_dir: Path
    target_label: str
    vocab_path: Path
    checkpoint_dir: Path
    exports_dir: Path
    repo_root: Path

    @classmethod
    def from_dict(cls, raw: dict, repo_root: Path) -> Config:
        data_root = _resolve_path(raw["data_root"], repo_root)
        graph_dir = _resolve_path(raw["graph_dir"], repo_root)
        tensor_dir = _resolve_path(raw["tensor_dir"], repo_root)
        vocab_path = _resolve_path(raw["vocab_path"], repo_root)
        checkpoint_dir = _resolve_path(raw["checkpoint_dir"], repo_root)
        exports_dir = _resolve_path(raw["exports_dir"], repo_root)
        return cls(
            data_root=data_root,
            graph_dir=graph_dir,
            tensor_dir=tensor_dir,
            target_label=raw.get("target_label", "lut"),
            vocab_path=vocab_path,
            checkpoint_dir=checkpoint_dir,
            exports_dir=exports_dir,
            repo_root=repo_root,
        )


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML. ``LL_HLS4ML_DATA_ROOT`` overrides ``data_root``."""
    repo_root = _repo_root()
    config_path = Path(path) if path else _DEFAULT_CONFIG
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    if env_data := os.environ.get("LL_HLS4ML_DATA_ROOT"):
        raw["data_root"] = env_data

    variables = {"data_root": raw["data_root"]}
    raw = _substitute_vars(raw, variables)
    return Config.from_dict(raw, repo_root)

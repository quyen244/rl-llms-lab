"""YAML config loading with `--set a.b.c=value` command-line overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _parse_value(raw)
    return cfg


def _parse_value(raw: str) -> Any:
    # YAML 1.1 reads "2e-5" as a string, so try numeric coercion first.
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def registry_name(cfg: dict[str, Any]) -> str:
    """rl-lab-<experiment>-<model>, lowercase, no dots. Used for the MLflow registry and the Hub repo."""
    model = cfg["model"]["name"].split("/")[-1].lower().replace(".", "-")
    return f"rl-lab-{cfg['experiment']}-{model}"


def hub_model_id(cfg: dict[str, Any], username: str) -> str:
    return f"{username}/{registry_name(cfg)}"


def flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config so every value can be logged as an MLflow param."""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{name}."))
        else:
            out[name] = v
    return out

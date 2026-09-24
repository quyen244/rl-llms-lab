"""Run records: one self-describing JSON file per experiment.

The record is the source of truth for reports. It captures what was run (config, code commit,
model and dataset revisions), what happened (metrics) and what it cost (time, memory, dollars).
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REQUIRED = ["schema_version", "run_id", "created_at", "experiment", "method", "git", "env",
            "model", "dataset", "config", "metrics", "cost", "status"]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def git_info() -> dict[str, Any]:
    return {"commit": _run(["git", "rev-parse", "HEAD"]), "dirty": bool(_run(["git", "status", "--porcelain"]))}


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes", "mlflow"):
        try:
            info[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info[pkg] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["gpu_count"], info["gpu_name"] = 0, None
    return info


def hub_revision(kind: str, name: str) -> str | None:
    """Commit sha of a model or dataset repo on the Hub, so the exact version is pinned."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        return (api.dataset_info(name) if kind == "dataset" else api.model_info(name)).sha
    except Exception:
        return None


def cost_info(wall_seconds: float, gpu_count: int, peak_mem_gb: float | None, hourly_usd: float | None) -> dict[str, Any]:
    gpu_hours = wall_seconds / 3600 * max(gpu_count, 1)
    return {
        "wall_seconds": round(wall_seconds, 1),
        "gpu_hours": round(gpu_hours, 4),
        "gpu_hourly_usd": hourly_usd,
        "cost_usd": round(gpu_hours * hourly_usd, 4) if hourly_usd else None,
        "peak_gpu_mem_gb": round(peak_mem_gb, 2) if peak_mem_gb is not None else None,
    }


def build_record(
    *, run_id: str, cfg: dict[str, Any], dataset: dict[str, Any], metrics: dict[str, Any],
    cost: dict[str, Any], mlflow_run_id: str | None = None, artifacts: dict[str, Any] | None = None,
    status: str = "finished",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": cfg["experiment"],
        "method": cfg["method"],
        "git": git_info(),
        "env": env_info(),
        "model": {
            "name": cfg["model"]["name"],
            "revision": hub_revision("model", cfg["model"]["name"]),
            "quantization": {k: v for k, v in cfg["model"].items() if k.startswith(("load_in", "bnb"))},
            "lora": cfg.get("lora"),
        },
        "dataset": dataset,
        "config": cfg,
        "metrics": metrics,
        "cost": cost,
        "artifacts": artifacts or {},
        "status": status,
    }


def validate(rec: dict[str, Any]) -> list[str]:
    errors = [f"missing field: {k}" for k in REQUIRED if k not in rec]
    if rec.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {rec.get('schema_version')}")
    return errors


def write_record(rec: dict[str, Any], directory: str | Path = "outputs/records") -> Path:
    errors = validate(rec)
    if errors:
        raise ValueError("invalid run record: " + "; ".join(errors))
    path = Path(directory) / f"{rec['experiment']}-{rec['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=str))
    return path


def load_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    records = []
    for p in paths:
        rec = json.loads(Path(p).read_text())
        errors = validate(rec)
        if errors:
            raise ValueError(f"{p}: " + "; ".join(errors))
        records.append(rec)
    return records

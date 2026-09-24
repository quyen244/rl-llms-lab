"""Dataset loading. Datasets are in TRL's standard conversational formats."""
from __future__ import annotations

from typing import Any

from lab.record import hub_revision


def load_train_dataset(data_cfg: dict[str, Any], seed: int = 42):
    """Return (dataset, info). info pins the exact data used, for the run record."""
    from datasets import load_dataset

    ds = load_dataset(data_cfg["name"], data_cfg.get("config"), split=data_cfg.get("split", "train"))
    total = len(ds)
    n = data_cfg.get("max_samples")
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, total)))
    info = {
        "name": data_cfg["name"],
        "config": data_cfg.get("config"),
        "split": data_cfg.get("split", "train"),
        "revision": hub_revision("dataset", data_cfg["name"]),
        "fingerprint": getattr(ds, "_fingerprint", None),
        "rows_total": total,
        "rows_used": len(ds),
        "sample_seed": seed,
    }
    return ds, info

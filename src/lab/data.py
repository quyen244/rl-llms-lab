"""Dataset loading. Datasets are in TRL's standard conversational formats."""
from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset


def load_train_dataset(data_cfg: dict[str, Any], seed: int = 42) -> Dataset:
    ds = load_dataset(data_cfg["name"], data_cfg.get("config"), split=data_cfg.get("split", "train"))
    n = data_cfg.get("max_samples")
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return ds

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


def to_gsm8k_solutions(ds, tok):
    """GSM8K rows -> prompt/completion with the human-written solution, in the same format as distillation.

    Prompts match eval and the teacher prompts; calculator annotations like <<48/2=24>> are removed.
    """
    import re

    from lab.evals import PROMPT_SUFFIX

    def fmt(r):
        prompt = tok.apply_chat_template([{"role": "user", "content": r["question"] + PROMPT_SUFFIX}],
                                         tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "completion": re.sub(r"<<[^>]*>>", "", r["answer"])}

    return ds.map(fmt, remove_columns=ds.column_names)

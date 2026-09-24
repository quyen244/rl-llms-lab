"""GSM8K evaluation (greedy decoding, exact match on the final number).

torch and datasets are imported lazily so the answer-extraction logic is unit-testable anywhere.
"""
from __future__ import annotations

import re
from typing import Any

PROMPT_SUFFIX = "\nSolve step by step, then give the final answer on its own line as '#### <number>'."
_NUM = r"-?\d[\d,]*\.?\d*"


def _clean(s: str) -> str | None:
    s = s.replace(",", "").strip().rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def extract_answer(text: str) -> tuple[str | None, bool]:
    """Return (answer, followed_format). Prefers '#### n', then \\boxed{n}, then the last number."""
    m = re.findall(r"####\s*\$?(" + _NUM + ")", text)
    if m:
        return _clean(m[-1]), True
    m = re.findall(r"\\boxed\{\s*\$?(" + _NUM + r")", text)
    if m:
        return _clean(m[-1]), False
    m = re.findall(_NUM, text)
    return (_clean(m[-1]), False) if m else (None, False)


def gold_answer(answer_field: str) -> str | None:
    return _clean(answer_field.split("####")[-1])


def evaluate_gsm8k(model, tok, n: int = 1000, batch_size: int = 16, max_new_tokens: int = 384) -> dict[str, Any]:
    """Evaluate on the first n GSM8K test questions (deterministic subset)."""
    import torch
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(n, len(ds))
    ds = ds.select(range(n))
    tok.padding_side = "left"
    model.eval()
    correct = formatted = new_tokens = 0
    for i in range(0, n, batch_size):
        batch = ds[i : i + batch_size]
        prompts = [
            tok.apply_chat_template([{"role": "user", "content": q + PROMPT_SUFFIX}], tokenize=False, add_generation_prompt=True)
            for q in batch["question"]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
        gen = out[:, enc["input_ids"].shape[1]:]
        new_tokens += int((gen != tok.pad_token_id).sum())
        for text, gold in zip(tok.batch_decode(gen, skip_special_tokens=True), batch["answer"]):
            pred, fmt = extract_answer(text)
            correct += int(pred is not None and pred == gold_answer(gold))
            formatted += int(fmt)
    return {
        "gsm8k_acc": correct / n,
        "gsm8k_n": n,
        "gsm8k_format_rate": formatted / n,
        "gsm8k_mean_new_tokens": new_tokens / n,
    }

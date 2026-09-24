"""GRPO on GSM8K with TRL's GRPOTrainer and QLoRA.

Prompts use the same chat format and suffix as lab.evals, so training and evaluation ask the same
question. Rewards are rule-based (no reward model):

- correctness_reward: 1.0 if the extracted final number equals the GSM8K gold answer, else 0.0
- format_reward: FORMAT_BONUS if the completion ends its answer with '#### <number>', else 0.0

They are separate named functions so TRL logs rewards/correctness_reward/mean and
rewards/format_reward/mean separately. TRL passes every extra dataset column (here 'answer')
to each reward function as a keyword argument, one value per completion.

Written against trl 1.13 (GRPOConfig has no max_prompt_length; generation uses model.generate unless use_vllm).
"""
from __future__ import annotations

from typing import Any

from lab.evals import PROMPT_SUFFIX, extract_answer, gold_answer

FORMAT_BONUS = 0.1


def build_prompt(question: str) -> list[dict[str, str]]:
    """Conversational prompt, identical to the one lab.evals.evaluate_gsm8k builds."""
    return [{"role": "user", "content": question + PROMPT_SUFFIX}]


def _text(completion: Any) -> str:
    """TRL passes a message list for conversational prompts and a plain string otherwise."""
    if isinstance(completion, str):
        return completion
    return "".join(m.get("content") or "" for m in completion if m.get("role", "assistant") == "assistant")


def correctness_reward(completions: list, answer: list[str], **kwargs) -> list[float]:
    out = []
    for c, gold in zip(completions, answer, strict=True):
        pred, _ = extract_answer(_text(c))
        out.append(1.0 if pred is not None and pred == gold_answer(gold) else 0.0)
    return out


def format_reward(completions: list, **kwargs) -> list[float]:
    return [FORMAT_BONUS if extract_answer(_text(c))[1] else 0.0 for c in completions]


REWARD_FUNCS = [correctness_reward, format_reward]


def format_gsm8k(example: dict[str, str]) -> dict[str, Any]:
    return {"prompt": build_prompt(example["question"]), "answer": example["answer"]}


def load_grpo_dataset(data_cfg: dict[str, Any], seed: int = 42):
    """GSM8K train prompts for GRPO. Returns (dataset, info) like lab.data.load_train_dataset.

    Columns: 'prompt' (chat messages) and 'answer' (full GSM8K solution; the gold number is after '####').
    """
    from lab.data import load_train_dataset

    cfg = {"name": "openai/gsm8k", "config": "main", "split": "train", **data_cfg}
    ds, info = load_train_dataset(cfg, seed)
    ds = ds.map(format_gsm8k, remove_columns=[c for c in ds.column_names if c not in ("answer",)])
    info["fingerprint"] = getattr(ds, "_fingerprint", None)
    info["format"] = "chat prompt = question + lab.evals.PROMPT_SUFFIX"
    return ds, info


def run_grpo(cfg, model, tok, ds, lora):
    """Build a GRPOTrainer. Same contract as run_sft in lab.train."""
    from trl import GRPOConfig, GRPOTrainer

    from lab.train import _common_args

    train_cfg = dict(cfg["train"])
    weights = train_cfg.pop("reward_weights", None)
    args = GRPOConfig(**_common_args(cfg), **train_cfg, reward_weights=weights)
    tok.padding_side = "left"  # decoder-only generation needs left padding
    return GRPOTrainer(
        model=model,
        reward_funcs=list(REWARD_FUNCS),
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
    )

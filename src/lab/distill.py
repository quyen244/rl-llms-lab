"""Knowledge distillation: a 4-bit teacher, a QLoRA student, two stages.

    Stage 1, sequence-level: the teacher generates answers to GSM8K train questions (using
    lab.evals.PROMPT_SUFFIX so answers end '#### n'), cached to a jsonl file, then the student is
    SFT-trained on those (prompt, completion) pairs. method: distill_seq -> run_distill_seq.

    Stage 2, logit-level KL: the student is trained on the *same* cached (prompt, completion) pairs,
    with an added token-level KL-to-teacher loss on the completion tokens. Qwen2.5 sizes share a
    tokenizer, so the teacher's forward pass reuses the student's own input_ids directly: no separate
    teacher tokenizer or vocabulary mapping needed at training time. method: distill_kl -> run_distill_kl.

Both stages read the same on-disk cache, built lazily by generate_teacher_completions/load_distill_dataset:
if the cache already holds >= data.max_samples rows, no teacher is loaded and no GPU time is spent.
Generation time is measured separately (dataset_info["teacher_generation_seconds"]) so it can be added to
training wall time by the caller, e.g. `wall += dataset_info.get("teacher_generation_seconds", 0.0)`.

Standalone use (pre-generate the cache without training, e.g. overnight, then train later):

    python -m lab.distill --config configs/distill_qlora.yaml [--set a.b=c]
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from lab.data import load_train_dataset
from lab.evals import PROMPT_SUFFIX, extract_answer, gold_answer
from lab.record import hub_revision


def _common_args(cfg: dict) -> dict:
    """Precision, MLflow and Hugging Face Hub arguments shared by every trainer config.

    Duplicated from lab.train._common_args (not imported: lab.train imports lab.distill to register
    METHODS, so importing back would be circular). Keep the two in sync if either changes.
    """
    from lab.config import hub_model_id
    from lab.modeling import resolve_dtype

    dtype = resolve_dtype(cfg["model"].get("compute_dtype", "auto"))
    args = {
        "report_to": "mlflow",
        "run_name": cfg["experiment"],
        "bf16": dtype == "bfloat16",
        "fp16": dtype == "float16",
    }
    username = os.environ.get("HF_USERNAME")
    if cfg.get("hub", {}).get("push", False):
        if username and os.environ.get("HF_TOKEN"):
            args.update(
                push_to_hub=True,
                hub_model_id=hub_model_id(cfg, username),
                hub_private_repo=True,
                hub_strategy="checkpoint",
            )
        else:
            print("hub.push is on but HF_USERNAME/HF_TOKEN are not set in .env; not pushing.")
    return args


# --------------------------------------------------------------------------------------
# Teacher generation and caching
# --------------------------------------------------------------------------------------


def _teacher_cache_path(cfg: dict) -> Path:
    d = cfg.get("distill", {})
    teacher = cfg["teacher"]["name"].split("/")[-1]
    n = cfg["data"].get("max_samples", 1500)
    return Path(d.get("cache_dir", "outputs/distill")) / f"teacher-{teacher}-n{n}.jsonl"


def _load_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Teacher's own GSM8K accuracy and format-compliance rate on its cached generations."""
    n = len(rows)
    if n == 0:
        return {"teacher_gsm8k_acc": None, "teacher_format_rate": None}
    correct = formatted = 0
    for r in rows:
        pred, is_fmt = extract_answer(r["completion"])
        correct += int(pred is not None and r.get("gold_answer") is not None and pred == r["gold_answer"])
        formatted += int(is_fmt)
    return {"teacher_gsm8k_acc": correct / n, "teacher_format_rate": formatted / n}


def generate_teacher_completions(cfg: dict) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (rows, info) for the first data.max_samples GSM8K train prompts.

    Reuses whatever is already cached at outputs/distill/teacher-<name>-n<N>.jsonl and only generates the
    missing rows. If the cache is already full, no teacher model is loaded (no GPU needed). Each row is
    {"prompt": <chat-templated question>, "completion": <teacher's raw generation>, "gold_answer": <str>}.
    """
    n_target = cfg["data"].get("max_samples", 1500)
    cache_path = _teacher_cache_path(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_cache(cache_path)

    info: dict[str, Any] = {
        "teacher_model": cfg["teacher"]["name"],
        "teacher_revision": hub_revision("model", cfg["teacher"]["name"]),
        "cache_path": str(cache_path),
        "n_cached_at_start": len(rows),
        "teacher_generation_seconds": 0.0,
        "n_generated": 0,
    }

    if len(rows) >= n_target:
        rows = rows[:n_target]
        info.update(_score_rows(rows))
        return rows, info

    import torch

    from lab.modeling import build_model, build_tokenizer

    teacher_tok = build_tokenizer(cfg["teacher"]["name"])
    teacher_tok.padding_side = "left"
    teacher = build_model(cfg["teacher"])
    teacher.eval()

    data_cfg = dict(cfg["data"], max_samples=n_target)
    ds, _ = load_train_dataset(data_cfg, cfg["train"].get("seed", 42))
    todo = ds.select(range(len(rows), n_target))

    d = cfg.get("distill", {})
    batch_size = d.get("gen_batch_size", 8)
    max_new_tokens = d.get("max_new_tokens", 384)
    do_sample = d.get("do_sample", False)

    t0 = time.time()
    with cache_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            prompts = [
                teacher_tok.apply_chat_template(
                    [{"role": "user", "content": q + PROMPT_SUFFIX}], tokenize=False, add_generation_prompt=True
                )
                for q in batch["question"]
            ]
            enc = teacher_tok(prompts, return_tensors="pt", padding=True).to(teacher.device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": teacher_tok.pad_token_id,
                "use_cache": True,
            }
            if do_sample:
                gen_kwargs["temperature"] = d.get("temperature", 1.0)
            with torch.no_grad():
                out = teacher.generate(**enc, **gen_kwargs)
            gen = out[:, enc["input_ids"].shape[1] :]
            texts = teacher_tok.batch_decode(gen, skip_special_tokens=True)
            for prompt, answer_field, completion in zip(prompts, batch["answer"], texts):
                row = {"prompt": prompt, "completion": completion, "gold_answer": gold_answer(answer_field)}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
            done = min(i + batch_size, len(todo))
            print(f"[distill-teacher] {done}/{len(todo)} generated", flush=True)

    info["teacher_generation_seconds"] = round(time.time() - t0, 1)
    info["n_generated"] = len(todo)

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    info.update(_score_rows(rows))
    return rows, info


def load_distill_dataset(cfg: dict):
    """(dataset, dataset_info) with "prompt"/"completion" columns, for run_distill_seq/run_distill_kl.

    Mirrors lab.data.load_train_dataset's return shape (a dataset plus an info dict for the run record),
    but the rows come from generate_teacher_completions instead of a Hub dataset.
    """
    from datasets import Dataset

    rows, info = generate_teacher_completions(cfg)
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    dataset_info = {
        "name": "openai/gsm8k (teacher-generated completions)",
        "config": "main",
        "split": "train",
        "revision": hub_revision("dataset", "openai/gsm8k"),
        "rows_total": None,
        "rows_used": len(ds),
        "sample_seed": cfg["train"].get("seed", 42),
        **info,
    }
    return ds, dataset_info


# --------------------------------------------------------------------------------------
# Stage 1: sequence-level distillation (plain SFT on teacher generations)
# --------------------------------------------------------------------------------------


def run_distill_seq(cfg, model, tok, ds, lora):
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(**_common_args(cfg), **cfg["train"])
    return SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)


# --------------------------------------------------------------------------------------
# Stage 2: logit-level KL distillation (SFT + a token-level KL-to-teacher term)
# --------------------------------------------------------------------------------------


def kl_distillation_loss(student_logits, teacher_logits, labels, temperature: float = 2.0):
    """Token-level KL(teacher || student) on next-token distributions, masked to completion tokens.

    Uses the same shift-by-one and (labels != -100) mask that SFTTrainer's own cross-entropy loss uses,
    so the KL term and the CE term are computed over exactly the same tokens. Scaled by temperature**2
    (standard KD scaling, keeps gradient magnitude roughly temperature-independent).
    """
    import torch
    import torch.nn.functional as F

    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError(
            f"student vocab size {student_logits.shape[-1]} != teacher vocab size {teacher_logits.shape[-1]}; "
            "logit KL distillation needs student and teacher to share a tokenizer/vocabulary."
        )
    shift_student = student_logits[..., :-1, :].float()
    shift_teacher = teacher_logits[..., :-1, :].float()
    shift_labels = labels[..., 1:]
    mask = shift_labels != -100
    n = mask.sum()
    if n == 0:
        return shift_student.new_zeros(())
    student_logp = F.log_softmax(shift_student / temperature, dim=-1)
    teacher_p = F.softmax(shift_teacher / temperature, dim=-1)
    kl = F.kl_div(student_logp, teacher_p, reduction="none").sum(-1)  # (batch, seq-1)
    kl = (kl * mask).sum() / n
    return kl * (temperature**2)


class _KLTrainer:
    """Mixin applied to SFTTrainer at construction time (see run_distill_kl) so the class only needs to
    exist once TRL's SFTTrainer is importable. Adds one teacher forward pass and one loss term per step;
    everything else (tokenization, label masking, packing, TRL bookkeeping) stays SFTTrainer's own.
    """

    def __init__(self, *args, teacher, kl_temperature: float = 2.0, kl_alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.kl_temperature = kl_temperature
        self.kl_alpha = kl_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        ce_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        with torch.no_grad():
            teacher_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        kl = kl_distillation_loss(outputs.logits, teacher_out.logits, labels, self.kl_temperature)
        loss = (1 - self.kl_alpha) * ce_loss + self.kl_alpha * kl

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["kl_loss"].append(kl.detach().item())
        self._metrics[mode]["ce_loss"].append(ce_loss.detach().item())

        return (loss, outputs) if return_outputs else loss


def run_distill_kl(cfg, model, tok, ds, lora):
    from trl import SFTConfig, SFTTrainer

    from lab.modeling import build_model

    teacher_cfg = cfg["teacher"]
    if not teacher_cfg.get("load_in_4bit", False):
        print("warning: teacher.load_in_4bit is not set; a full-precision teacher likely will not fit "
              "alongside the student on a 16 GB GPU.")
    teacher = build_model(teacher_cfg)
    teacher.eval()
    teacher.requires_grad_(False)

    d = cfg.get("distill", {})
    train_kwargs = dict(cfg["train"])
    # KL needs the dense (batch, seq_len, vocab) logits SFTTrainer's default "chunked_nll" loss_type does
    # not guarantee (it projects the lm_head on non-ignored tokens only); force the plain path.
    train_kwargs["loss_type"] = "nll"
    train_kwargs["use_liger_kernel"] = False
    args = SFTConfig(**_common_args(cfg), **train_kwargs)

    KLTrainer = type("KLTrainer", (_KLTrainer, SFTTrainer), {})
    return KLTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
        teacher=teacher,
        kl_temperature=d.get("kl_temperature", 2.0),
        kl_alpha=d.get("kl_alpha", 0.5),
    )


METHODS = {"distill_seq": run_distill_seq, "distill_kl": run_distill_kl}


def main() -> None:
    """Pre-generate and cache teacher completions without training (e.g. to run overnight)."""
    import argparse

    from dotenv import load_dotenv

    from lab.config import load_config

    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], dest="overrides")
    a = p.parse_args()
    cfg = load_config(a.config, a.overrides)

    ds, info = load_distill_dataset(cfg)
    print(f"cached {len(ds)} teacher completions -> {info['cache_path']}")
    for k in (
        "teacher_model", "teacher_revision", "n_cached_at_start", "n_generated",
        "teacher_generation_seconds", "teacher_gsm8k_acc", "teacher_format_rate",
    ):
        print(f"  {k}: {info.get(k)}")


if __name__ == "__main__":
    main()

"""lab.distill tests. No GPU, no model downloads: the cache-hit paths never build a model, and the
KL loss math is checked with plain CPU tensors.
"""
import json

import pytest
import torch

from lab.distill import (
    _score_rows,
    _teacher_cache_path,
    generate_teacher_completions,
    kl_distillation_loss,
    load_distill_dataset,
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr("lab.distill.hub_revision", lambda kind, name: "deadbeefcafe")


def _cfg(tmp_path, n=3):
    return {
        "experiment": "distill-smoke",
        "teacher": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "load_in_4bit": True},
        "data": {"name": "openai/gsm8k", "config": "main", "split": "train", "max_samples": n},
        "distill": {"cache_dir": str(tmp_path)},
        "train": {"seed": 42},
    }


def _write_cache(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_cache_path_names_by_teacher_and_n(tmp_path):
    cfg = _cfg(tmp_path, n=5)
    p = _teacher_cache_path(cfg)
    assert p.parent == tmp_path
    assert p.name == "teacher-Qwen2.5-1.5B-Instruct-n5.jsonl"


def test_score_rows_accuracy_and_format_rate():
    rows = [
        {"completion": "reasoning...\n#### 4", "gold_answer": "4"},   # correct, formatted
        {"completion": "reasoning...\n#### 5", "gold_answer": "4"},   # wrong, formatted
        {"completion": "the answer is 4, no marker", "gold_answer": "4"},  # correct (last number), not formatted
    ]
    scored = _score_rows(rows)
    assert scored["teacher_gsm8k_acc"] == pytest.approx(2 / 3)
    assert scored["teacher_format_rate"] == pytest.approx(2 / 3)
    assert _score_rows([]) == {"teacher_gsm8k_acc": None, "teacher_format_rate": None}


def test_generate_teacher_completions_cache_hit_needs_no_gpu(tmp_path):
    """A fully populated cache must be served without importing torch/transformers/building a model."""
    cfg = _cfg(tmp_path, n=2)
    cache_path = _teacher_cache_path(cfg)
    rows = [
        {"prompt": "Q1?", "completion": "work\n#### 1", "gold_answer": "1"},
        {"prompt": "Q2?", "completion": "work\n#### 2", "gold_answer": "2"},
        {"prompt": "Q3?", "completion": "work\n#### 3", "gold_answer": "3"},  # extra row, should be ignored
    ]
    _write_cache(cache_path, rows)

    out_rows, info = generate_teacher_completions(cfg)

    assert len(out_rows) == 2
    assert out_rows == rows[:2]
    assert info["n_generated"] == 0
    assert info["n_cached_at_start"] == 3
    assert info["teacher_generation_seconds"] == 0.0
    assert info["teacher_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert info["teacher_gsm8k_acc"] == 1.0
    assert info["teacher_format_rate"] == 1.0


def test_load_distill_dataset_from_cache(tmp_path):
    cfg = _cfg(tmp_path, n=2)
    cache_path = _teacher_cache_path(cfg)
    rows = [
        {"prompt": "Q1?", "completion": "work\n#### 1", "gold_answer": "1"},
        {"prompt": "Q2?", "completion": "work\n#### 2", "gold_answer": "2"},
    ]
    _write_cache(cache_path, rows)

    ds, dataset_info = load_distill_dataset(cfg)

    assert len(ds) == 2
    assert set(ds.column_names) == {"prompt", "completion"}
    assert ds[0]["prompt"] == "Q1?"
    assert dataset_info["rows_used"] == 2
    assert dataset_info["revision"] == "deadbeefcafe"
    assert dataset_info["teacher_generation_seconds"] == 0.0
    assert dataset_info["sample_seed"] == 42


def test_kl_distillation_loss_zero_when_student_matches_teacher():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 32)
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[:, 2:] = 5  # last 4 positions are "completion" tokens
    loss = kl_distillation_loss(logits, logits.clone(), labels, temperature=2.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_distillation_loss_positive_when_distributions_differ():
    torch.manual_seed(0)
    student = torch.randn(2, 6, 32)
    teacher = torch.randn(2, 6, 32)
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[:, 2:] = 5
    loss = kl_distillation_loss(student, teacher, labels, temperature=2.0)
    assert loss.item() > 0.0


def test_kl_distillation_loss_ignores_fully_masked_batch():
    logits = torch.randn(1, 4, 16)
    labels = torch.full((1, 4), -100, dtype=torch.long)
    loss = kl_distillation_loss(logits, torch.randn(1, 4, 16), labels)
    assert loss.item() == 0.0


def test_kl_distillation_loss_rejects_vocab_mismatch():
    labels = torch.full((1, 4), -100, dtype=torch.long)
    labels[:, 2:] = 1
    with pytest.raises(ValueError, match="vocab size"):
        kl_distillation_loss(torch.randn(1, 4, 16), torch.randn(1, 4, 32), labels)

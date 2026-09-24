import json

import pytest

from lab.evals import extract_answer, gold_answer
from lab.governance import is_better
from lab.methods import METHOD_PROFILES, complexity_score
from lab.record import build_record, cost_info, load_records, validate, write_record
from lab.report import build_report, pareto_front, wilson_ci


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr("lab.record.hub_revision", lambda kind, name: "deadbeefcafe")


def _rec(method, acc, hours, run_id, model="Qwen/Qwen2.5-1.5B-Instruct", price=0.5, n=1000):
    cfg = {"experiment": method, "method": method, "model": {"name": model}, "train": {"seed": 42}}
    rec = build_record(
        run_id=run_id, cfg=cfg,
        dataset={"train": {"revision": "abc12345", "rows_used": 5000}, "eval": {"rows_used": n}},
        metrics={"train": {}, "eval": {"gsm8k_acc": acc, "gsm8k_n": n}},
        cost=cost_info(hours * 3600, 1, 12.0, price),
    )
    return rec


def test_extract_answer_formats():
    assert extract_answer("so 5+3=8\n#### 8") == ("8", True)
    assert extract_answer("#### 1,234") == ("1234", True)
    assert extract_answer("the answer is \\boxed{42}") == ("42", False)
    assert extract_answer("about 3.50 dollars") == ("3.5", False)
    assert extract_answer("no digits here") == (None, False)
    assert extract_answer("9" * 400) == (None, False)  # overflows float to inf
    assert gold_answer("lots of text #### 72") == "72"


def test_complexity_ordering():
    assert set(METHOD_PROFILES) >= {"sft", "dpo", "grpo", "ppo"}
    assert complexity_score("sft") < complexity_score("dpo") < complexity_score("grpo") < complexity_score("ppo")


def test_cost_info():
    c = cost_info(7200, 1, 10.123, 0.5)
    assert c["gpu_hours"] == 2.0 and c["cost_usd"] == 1.0
    assert cost_info(3600, 1, None, None)["cost_usd"] is None


def test_is_better():
    assert is_better(0.6, None)
    assert not is_better(None, 0.5)
    assert not is_better(0.505, 0.5, min_gain=0.01)
    assert is_better(0.52, 0.5, min_gain=0.01)


def test_wilson_and_pareto():
    lo, hi = wilson_ci(0.5, 1000)
    assert 0.46 < lo < 0.5 < hi < 0.54
    pts = [("a", 1.0, 0.5), ("b", 2.0, 0.4), ("c", 3.0, 0.7)]
    assert pareto_front(pts) == {"a", "c"}


def test_record_roundtrip_and_validation(tmp_path):
    rec = _rec("sft", 0.55, 1.0, "r1")
    path = write_record(rec, tmp_path)
    assert load_records([path])[0]["run_id"] == "r1"
    bad = dict(rec)
    del bad["metrics"]
    assert validate(bad)
    with pytest.raises(ValueError):
        write_record(bad, tmp_path)


def test_report_end_to_end(tmp_path):
    recs = [
        _rec("baseline", 0.40, 0.0, "b0", price=None),
        _rec("sft", 0.45, 1.0, "s1"),
        _rec("dpo", 0.50, 2.0, "d1"),
        _rec("grpo", 0.62, 5.0, "g1"),
    ]
    paths = [write_record(r, tmp_path / "records") for r in recs]
    md = build_report(load_records(paths), tmp_path / "report").read_text(encoding="utf-8")
    for needle in ("Leaderboard", "Pareto", "grpo", "Group relative policy optimisation", "Reproducibility", "Wilson"):
        assert needle in md
    assert "yes" in md  # grpo gain over baseline is significant
    assert json.loads((tmp_path / "records" / "sft-s1.json").read_text())["schema_version"] == 1

from lab.config import flatten, hub_model_id, load_config


def test_overrides_and_flatten(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("method: sft\ntrain:\n  learning_rate: 1.0e-4\n")
    cfg = load_config(f, ["train.learning_rate=2e-5", "data.max_samples=10"])
    assert cfg["train"]["learning_rate"] == 2e-5
    assert flatten(cfg)["data.max_samples"] == 10


def test_hub_model_id():
    cfg = {"experiment": "dpo-qlora", "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"}}
    assert hub_model_id(cfg, "me") == "me/rl-lab-dpo-qlora-qwen2-5-1-5b-instruct"

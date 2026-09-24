from pathlib import Path

from lab.config import load_config
from lab.evals import PROMPT_SUFFIX
from lab.grpo import FORMAT_BONUS, REWARD_FUNCS, build_prompt, correctness_reward, format_gsm8k, format_reward

GOLD = "Natalia sold 48/2 = 24 clips in May.\n48+24 = 72\n#### 72"


def _msg(text):
    return [{"role": "assistant", "content": text}]


def test_prompt_matches_eval_format():
    p = build_prompt("How many clips?")
    assert p == [{"role": "user", "content": "How many clips?" + PROMPT_SUFFIX}]
    row = format_gsm8k({"question": "How many clips?", "answer": GOLD})
    assert row == {"prompt": p, "answer": GOLD}


def test_correctness_reward_conversational_and_plain():
    comps = [_msg("48+24=72\n#### 72"), _msg("the answer is 72"), _msg("#### 71"), _msg("no idea"), "#### 72"]
    assert correctness_reward(comps, answer=[GOLD] * 5) == [1.0, 1.0, 0.0, 0.0, 1.0]


def test_correctness_reward_normalises_numbers():
    gold = "... #### 1234"
    assert correctness_reward([_msg("#### 1,234"), _msg("#### 1234.00")], answer=[gold, gold]) == [1.0, 1.0]


def test_format_reward():
    comps = [_msg("so #### 72"), _msg("72"), _msg("\\boxed{72}"), _msg("")]
    assert format_reward(comps, answer=[GOLD] * 4) == [FORMAT_BONUS, 0.0, 0.0, 0.0]


def test_rewards_accept_trl_kwargs():
    # TRL 1.13 calls reward_func(prompts=..., completions=..., completion_ids=..., <dataset columns>, trainer_state=...)
    kw = dict(prompts=[build_prompt("q")], completions=[_msg("#### 72")], completion_ids=[[1, 2]],
              answer=[GOLD], trainer_state=None, log_extra=None, log_metric=None)
    assert [f(**kw) for f in REWARD_FUNCS] == [[1.0], [FORMAT_BONUS]]
    assert [f.__name__ for f in REWARD_FUNCS] == ["correctness_reward", "format_reward"]


def test_grpo_config():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "grpo_qlora.yaml")
    t = cfg["train"]
    assert cfg["method"] == "grpo" and cfg["data"]["name"] == "openai/gsm8k"
    assert cfg["data"]["max_samples"] == 1500 and cfg["eval"]["gsm8k_samples"] == 1000
    assert (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]) % t["num_generations"] == 0
    assert "warmup_ratio" not in t

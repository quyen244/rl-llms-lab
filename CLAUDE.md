# rl-lab: handoff for the GPU-server session

Read this first. It records decisions, what is verified, and what to do next.
The owner works directly in this project; do not use any "firstmate" supervisor workflow here.
Style: no em dashes, concise reports (observed, cause, change, verification), report verification honestly.

## Goal

Learn and compare post-training algorithms (SFT, DPO, GRPO, PPO, distillation, rejection-sampling self-improvement) with QLoRA on rented GPUs.
Every run must produce MLflow tracking plus a JSON run record, and `lab.report` turns records into a comparison report on performance, cost and complexity.
The report is meant to showcase the work, so keep the rigor (baselines, confidence intervals, pinned revisions).

## Decisions made by the owner

- Hardware: one 16 GB GPU. 4-bit quantization plus LoRA (QLoRA) for everything. QLoRA is the priority.
- Data sizes: about 5,000 training rows per experiment and 1,000 test questions (first 1,000 of GSM8K test, which has 1,319).
- Models: Qwen2.5 family only (0.5B and 1.5B main, 7B for one capstone run and as teacher in 4-bit). Base for SFT, Instruct for DPO and GRPO.
- Datasets: SFT `trl-lib/Capybara` or `HuggingFaceTB/smoltalk` (7B config uses subset `smol-magpie-ultra`), DPO `trl-lib/ultrafeedback_binarized`, RL `openai/gsm8k`.
- Hugging Face: `HF_USERNAME` and `HF_TOKEN` go in `.env` (owner fills in later). Pushing is off by default (`--set hub.push=true`). Repos private, adapters only.
- GPU price is unknown: `cost.gpu_hourly_usd` is null in configs. The report falls back to GPU hours until it is filled in.
- The owner will push the git repo to a remote later; commits so far are local.

Full plan: `docs/experiment-plan.md`.

## What exists

- `src/lab/train.py`: entry point for `sft`, `dpo` and `baseline` (eval only). Logs to MLflow, evaluates GSM8K, writes `outputs/records/<experiment>-<run_id>.json`, registers the adapter as registry alias `candidate`.
- `src/lab/record.py`: run-record schema v1 (git commit and dirty flag, environment, model and dataset revisions from the Hub, config, metrics, cost, peak memory).
- `src/lab/evals.py`: GSM8K greedy exact-match evaluation.
- `src/lab/report.py`: report.md plus tradeoff.png with leaderboard, Wilson 95% CIs, significance vs baseline, Pareto front, per-method table, pros and cons, reproducibility table, limitations.
- `src/lab/methods.py`: qualitative profile and complexity score per algorithm.
- `src/lab/governance.py`: `promote` sets alias `champion` only if the candidate beats the champion by more than `--min-gain` (default 0.01); `show` lists versions.
- `src/lab/tracking.py`: MLflow dataset lineage and registry helpers. Default tracking store is `sqlite:///mlflow.db` (registry needs a DB).
- Configs: `baseline.yaml`, `sft_qlora.yaml`, `dpo_qlora.yaml`, `sft_qlora_7b.yaml`. `compute_dtype: auto` picks bf16 or fp16.
- Tests: `python -m pytest -q` (9 pass locally; they cover config, run records, report generation, answer extraction, promotion rule; none need a GPU).

## Verification status (be honest about this)

- Tested locally on Windows without GPU: config parsing, record write and validation, report generation from synthetic records, GSM8K answer extraction, promotion logic.
- NOT tested, never run on a GPU: everything that loads a model or calls TRL, MLflow, or the Hub. Expect small fixes on first run.
- Specific unverified assumptions to check first:
  1. TRL argument names (`SFTConfig`/`DPOConfig` accepting `max_length`, `processing_class=`, `hub_strategy`) against the installed TRL version.
  2. `mlflow.register_model("runs:/<id>/adapter", ...)` on a plain artifact folder (may need a model flavour or a different registration path).
  3. `mlflow.data.from_huggingface` dataset lineage; it is wrapped in try/except and only warns.
  4. Batched generation in `evaluate_gsm8k` with a 4-bit PEFT model, and gradient checkpointing versus `use_cache`.
  5. 7B QLoRA fitting in 16 GB at batch 1, length 1024. Measure peak memory (it is in the record) and lower `train.max_length` or `lora.r` if it runs out of memory.
  6. fp16 QLoRA stability on cards without bf16.
- The demo report seen during development used synthetic numbers. No real result exists yet.

## Next steps on the GPU server

1. `cp .env.example .env`, fill values (the owner does the secrets), then `bash scripts/bootstrap.sh`.
2. Smoke test: `uv run python -m lab.train --config configs/sft_qlora.yaml --set data.max_samples=64 --set train.max_steps=5 --set eval.gsm8k_samples=32`. Confirm MLflow run, `outputs/records/*.json`, registry alias, and fix anything from the unverified list.
3. Same smoke test for `dpo_qlora.yaml`.
4. E0 baselines: `configs/baseline.yaml` with `--set model.name=...` for Qwen2.5-0.5B-Instruct and 1.5B-Instruct. Set `cost.gpu_hourly_usd` from the rental price.
5. Real runs: E1 SFT (QLoRA versus plain LoRA), E2 DPO beta sweep 0.05/0.1/0.3.
6. Build the remaining pieces in this order: GRPO on GSM8K (needs a reward function and vLLM or HF generation), PPO on the same task, distillation (sequence-level then logit KL, teacher 4-bit), rejection-sampling loop, IFEval and MATH-500 evals, then from-scratch PyTorch DPO/GRPO/PPO checked against TRL. Register each new method in `src/lab/methods.py` and `train.py` `METHODS`.
7. After runs: `uv run python -m lab.report`, then `python -m lab.governance promote --model <registry name>`.

## Not built yet

GRPO, PPO, distillation, rejection sampling, IFEval and MATH-500 evals, vLLM generation, multi-seed aggregation in the report (single seed per config today, flagged in the report limitations).

# rl-lab

Scratch lab for learning post-training: SFT, DPO, GRPO/PPO and distillation, all with QLoRA and tracked in MLflow.
Iterate on 0.5B-1.5B models, then run a 7B on a rented GPU with the same code.

## Layout

- `configs/` one YAML per experiment (`--set a.b=c` overrides any value)
- `src/lab/` config, QLoRA model setup, data loading, training entry point
- `scripts/` rented-server bootstrap and MLflow server

## Quick start (Linux GPU box)

    bash scripts/bootstrap.sh
    bash scripts/mlflow_server.sh            # or set MLFLOW_TRACKING_URI to a remote server
    uv run python -m lab.train --config configs/sft_qlora.yaml
    uv run python -m lab.train --config configs/dpo_qlora.yaml

## Roadmap

1. SFT (QLoRA) - scaffolded
2. DPO (QLoRA) - scaffolded
3. GRPO with a verifiable reward (GSM8K)
4. PPO
5. Distillation (logit KL and sequence-level)
6. From-scratch PyTorch versions of DPO/GRPO/PPO, checked against TRL

## Notes

- `bitsandbytes` 4-bit is Linux only, so QLoRA runs on the rented server. Local Windows is for unit tests and tiny CPU smoke tests.
- `compute_dtype: auto` picks bf16 when the GPU supports it, else float16.
- Copy `.env.example` to `.env` and fill in `HF_USERNAME`, `HF_TOKEN` and optionally `MLFLOW_TRACKING_URI`.
  Pushing to the Hub is off by default; enable with `--set hub.push=true`.
- Target hardware is one 16 GB GPU. See `docs/experiment-plan.md` for models, experiments and datasets.
- Status: scaffold only. Nothing here has been run against a GPU yet.

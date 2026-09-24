# Experiment plan

Goal: learn SFT, DPO, GRPO, PPO, distillation and self-improvement loops with QLoRA on rented GPUs.
Every experiment answers one question and has a metric, so results are comparable in MLflow.

Verified on the Hub (ungated, Apache-2.0 or MIT): Qwen2.5 0.5B, 1.5B and 7B (base and Instruct), `trl-lib/Capybara`, `trl-lib/ultrafeedback_binarized`, `HuggingFaceH4/ultrafeedback_binarized`, `openai/gsm8k`, `HuggingFaceTB/smoltalk`, `HuggingFaceH4/MATH-500`, `Anthropic/hh-rlhf`.

## Model choice

Use one family, Qwen2.5, for everything.

- 0.5B: debugging and from-scratch implementations. A run takes minutes.
- 1.5B: the main experiment model. Big enough that RL shows real signal, small enough for many runs.
- 7B-Instruct: distillation teacher (loaded in 4-bit on 16 GB) and one QLoRA capstone run only.
- Same tokenizer across sizes, so logit distillation from 7B to 0.5B/1.5B works without vocabulary mapping.
- Base checkpoints for SFT experiments (to see what SFT adds), Instruct checkpoints for DPO and GRPO (they start from a working chat model, so rewards are not all zero).

## Datasets

| Purpose | Dataset | Why |
|---|---|---|
| SFT | `HuggingFaceTB/smoltalk` (subset), `trl-lib/Capybara` for smoke tests | Standard chat format, small subsets are enough |
| DPO | `trl-lib/ultrafeedback_binarized` | Preference pairs in TRL's format, no conversion code |
| GRPO, PPO, self-improvement | `openai/gsm8k` train | Exact-match reward, so no reward model is needed |
| Evaluation | GSM8K test, `HuggingFaceH4/MATH-500`, IFEval via lm-evaluation-harness | Fixed held-out sets, same for every run |
| Distillation prompts | Prompts from smoltalk and GSM8K train, answered by the 7B teacher | Teacher data stays private in a HF dataset repo |

## Experiments (in order)

Each experiment has a baseline measured first, otherwise numbers mean nothing.

- **E0 Baselines.** Evaluate untouched 0.5B, 1.5B and 7B Instruct on GSM8K test, MATH-500 and IFEval. Build the eval script once and reuse it. Deliverable: a baseline table in MLflow.
- **E1 SFT and the cost of QLoRA.** Qwen2.5-1.5B base, smoltalk subset. Compare full-precision LoRA against QLoRA (4-bit). Question: what does 4-bit quantization cost in quality, memory and speed. Metrics: eval loss, IFEval, peak GPU memory, tokens/s.
- **E2 DPO.** Qwen2.5-1.5B-Instruct on ultrafeedback, sweep beta over 0.05, 0.1 and 0.3. Question: how does beta trade preference fit against drift. Metrics: reward accuracy, reward margin, KL to reference, mean response length, IFEval. Watch length: DPO often wins by getting longer, and this is the main lesson.
- **E3 GRPO.** Qwen2.5-1.5B-Instruct on GSM8K, reward = exact-match answer plus a format bonus. Question: how much does RL with a verifiable reward move GSM8K accuracy. Metrics: mean reward, GSM8K test accuracy, completion length, KL. Needs vLLM for generation.
- **E4 PPO.** Same task, model and reward as E3, so PPO and GRPO are compared directly. Start with a from-scratch minimal PPO on 0.5B, then the TRL trainer. Expect this to be the most fragile experiment (value head, KL control).
- **E5 Distillation.** Teacher 7B-Instruct, student 0.5B or 1.5B. Compare three students: (a) SFT on original data, (b) sequence-level distillation (SFT on teacher outputs), (c) logit distillation with KL loss. Metrics: GSM8K, IFEval, agreement with teacher.
- **E6 Self-improvement.** Rejection sampling: sample n answers per GSM8K prompt, keep the correct ones, SFT on them, repeat. Then build DPO pairs from correct versus incorrect samples. Compare against E3 GRPO.
- **E7 Capstone.** One 7B QLoRA run of the best recipe from E1 to E3, published to the Hub.

## Compute

Target hardware: one 16 GB GPU, with 4-bit quantization and LoRA for everything (decided by the owner).

- 0.5B and 1.5B QLoRA fit comfortably in 16 GB.
- 7B QLoRA should fit with batch 1, gradient accumulation, gradient checkpointing, 8-bit paged optimizer and max length 1024 (`configs/sft_qlora_7b.yaml`).
  This is an expectation, not a measurement. Confirm peak memory on the first run and lower `max_length` or `lora.r` if it runs out of memory.
- A 7B bf16 teacher (about 15 GB of weights) does not fit in 16 GB with room for generation.
  Run the teacher in 4-bit or 8-bit for E5, or use Qwen2.5-3B-Instruct as the teacher.
- GRPO and PPO hold generation and training in memory together, so keep them at 0.5B and 1.5B, short completions, small group sizes. 7B RL is out of scope on 16 GB.
- Many 16 GB cards (T4, V100) have no bf16. `compute_dtype: auto` picks bf16 when supported and float16 otherwise.
- Prices and times are not estimated here. Measure the first E1 run and extrapolate.

## Hugging Face configuration

- Namespace and token live in `.env` (copy `.env.example`): `HF_USERNAME` and a fine-grained write token `HF_TOKEN`. `.env` is gitignored.
- Pushing is off by default. Set `hub.push: true` in a config (or `--set hub.push=true`) to push to a private repo.
- Repos are private by default and named `<namespace>/rl-lab-<experiment>-<model>-<method>`.
- Push LoRA adapters only, with the resolved config and the MLflow run id in the model card. Merge to 16-bit weights only for final models.
- Use `hub_strategy="checkpoint"` so the last checkpoint is on the Hub if a rented server disappears, and resume from it.
- Generated datasets (teacher outputs, preference pairs) go to private dataset repos.

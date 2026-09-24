# Experiment plan

Goal: learn SFT, DPO, GRPO, PPO, distillation and self-improvement loops with QLoRA on rented GPUs.
Every experiment answers one question and has a metric, so results are comparable in MLflow.

Verified on the Hub (ungated, Apache-2.0 or MIT): Qwen2.5 0.5B, 1.5B and 7B (base and Instruct), `trl-lib/Capybara`, `trl-lib/ultrafeedback_binarized`, `HuggingFaceH4/ultrafeedback_binarized`, `openai/gsm8k`, `HuggingFaceTB/smoltalk`, `HuggingFaceH4/MATH-500`, `Anthropic/hh-rlhf`.

## Model choice

Use one family, Qwen2.5, for everything.

- 0.5B: debugging and from-scratch implementations. A run takes minutes.
- 1.5B: the main experiment model. Big enough that RL shows real signal, small enough for many runs.
- 7B-Instruct: distillation teacher (bf16 inference) and one QLoRA capstone run only.
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

- 0.5B and 1.5B QLoRA fit on a single 24 GB GPU (RTX 4090 or A5000). 7B QLoRA and 7B bf16 teacher inference also fit on 24 GB.
- GRPO and PPO need generation plus training in memory, so prefer 40 GB or more for E3 and E4 at 1.5B if 24 GB is tight.
- Prices and times are not estimated here. Measure the first E1 run and extrapolate.

## Hugging Face configuration

- Namespace: your personal account or an org, decided before the first push.
- Token: a fine-grained token with write access, only as `HF_TOKEN` on the server, never in git (`.env` is ignored).
- Repos are private by default and named `<namespace>/rl-lab-<experiment>-<model>-<method>`.
- Push LoRA adapters only, with the resolved config and the MLflow run id in the model card. Merge to 16-bit weights only for final models.
- Use `hub_strategy="checkpoint"` so the last checkpoint is on the Hub if a rented server disappears, and resume from it.
- Generated datasets (teacher outputs, preference pairs) go to private dataset repos.

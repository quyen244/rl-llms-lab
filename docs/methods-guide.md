# Methods guide

Short reference for each technique used in this lab: the idea, the math, how it is wired here, what to watch, and common mistakes.
Covers what is built and smoke-tested so far: QLoRA, the baseline eval, SFT, DPO. GRPO and distillation are added when they run.

All runs go through one entry point:

```bash
docker compose run --rm lab python -m lab.train --config configs/<file>.yaml [--set key=value ...]
```

Each run logs to MLflow, evaluates GSM8K, writes `outputs/records/<experiment>-<id>.json`, and (for trained methods) registers the adapter with alias `candidate`.

---

## 1. QLoRA (the training setup under every method)

**Idea.** Freeze the base model in 4-bit, train a small low-rank add-on (the adapter). A 0.5B to 7B model then fits in 16 GB.

**Math.** For a frozen weight `W` (d x k), LoRA learns `B` (d x r) and `A` (r x k), r much smaller than d:

```
W' = W + (alpha / r) * B A
```

Only `A` and `B` get gradients. `B` starts at zero, so training starts from the unchanged model.

QLoRA adds: `W` stored as 4-bit NF4 (a 4-bit format fitted to normally distributed weights), double quantization (the quantization constants are quantized too), and computation in bf16. Gradients flow through the dequantized `W` into `A` and `B`.

**Here.** [src/lab/modeling.py](../src/lab/modeling.py) `build_model` and `build_lora`. Config block:

| key | value | meaning |
|---|---|---|
| `load_in_4bit` | true | QLoRA on; false gives plain LoRA (the E1 comparison) |
| `bnb_4bit_quant_type` | nf4 | 4-bit format |
| `bnb_4bit_use_double_quant` | true | saves about 0.4 bits per parameter |
| `compute_dtype` | auto | bf16 on this GPU |
| `lora.r` / `lora.alpha` | 16 / 32 | rank and scale (effective scale alpha/r = 2) |
| `lora.target_modules` | all-linear | adapters on every linear layer (attention and MLP) |
| `gradient_checkpointing` | true | recompute activations in backward: less memory, about 20 to 30% slower |

**Watch.** `peak_gpu_mem_gb` in the run record. If out of memory: lower `train.max_length`, then batch size (raise `gradient_accumulation_steps` to keep the effective batch), then `lora.r`.

**Pitfalls.** Only the adapter is saved (`outputs/<method>/`), so loading it needs the same base model and revision. 4-bit generation is slower than bf16 for small models. Effective batch = `per_device_train_batch_size x gradient_accumulation_steps` (2 x 8 = 16 here).

---

## 2. Baseline (evaluation only, no training)

**Idea.** Measure the untouched model with exactly the same eval as every trained run. Every claim is "method X changed accuracy by N points versus this".

**Eval protocol.** [src/lab/evals.py](../src/lab/evals.py) `evaluate_gsm8k`:
- First 1000 questions of the GSM8K test split (fixed, not random).
- Chat template, question plus the instruction to end with `#### <number>`.
- Greedy decoding (deterministic), `max_new_tokens=384`, batch 96 (`eval.batch_size`).
- Answer extraction: `#### n` first, then `\boxed{n}`, then the last number in the text. Exact match against the gold number.

**Metrics.**
- `gsm8k_acc`: exact-match accuracy. The report adds a Wilson 95% interval; at n=1000 and about 30% that is roughly plus or minus 3 points.
- `gsm8k_format_rate`: share of answers using `#### n`. Separates "learned the format" from "got better at math".
- `gsm8k_mean_new_tokens`: answer length.

**Result so far.** Qwen2.5-0.5B-Instruct, 4-bit: accuracy 31.1%, format rate 0.1% (`outputs/records/baseline-e1f9399b.json`). The model solves problems but ignores the format instruction, so the last-number fallback does most of the work.

**Pitfalls.** Keep eval settings identical across runs (n, batch size, max tokens, 4-bit), or the comparison is not fair. Batch size can change greedy outputs slightly through padding numerics, so it is fixed at 96.

---

## 3. SFT (supervised fine-tuning)

**Idea.** Show the model good responses and train it to imitate them, token by token.

**Math.** Next-token cross-entropy on the target tokens:

```
L_SFT = - sum_t log p_theta(y_t | x, y_<t)
```

**Data.** `trl-lib/Capybara`, 1500 rows (shuffled with seed 42, then the first 1500). Multi-turn chat conversations, formatted with the model's chat template. Loading is in [src/lab/data.py](../src/lab/data.py); the dataset revision and fingerprint go into the run record.

**Here.** [src/lab/train.py](../src/lab/train.py) `run_sft` builds TRL `SFTConfig` and `SFTTrainer` with the LoRA config. Config [configs/sft_qlora.yaml](../configs/sft_qlora.yaml):

| key | value | why |
|---|---|---|
| `learning_rate` | 2e-4 | LoRA tolerates about 10x the full fine-tuning rate |
| `num_train_epochs` | 1 | about 94 optimizer steps at effective batch 16 |
| `warmup_steps` | 0.03 | below 1 means a fraction of total steps (transformers 5 removed `warmup_ratio`) |
| `lr_scheduler_type` | cosine | |
| `max_length` | 1024 | longer conversations are truncated |

**Watch.**
- `loss` should fall early then flatten (smoke run: about 1.49, full run about 1.4 at 64% of the epoch).
- `mean_token_accuracy`: share of target tokens predicted correctly (about 0.66).
- `grad_norm`: spikes mean the learning rate is too high.

**Pitfalls.**
- Capybara is general chat, not math. SFT on it can leave GSM8K flat or lower it. That is a real finding, not a bug. For a math gain, SFT on GSM8K-style solutions (or on teacher outputs, which is sequence-level distillation).
- Check whether the loss covers only the assistant tokens or the prompt too (TRL `assistant_only_loss` / `completion_only_loss`); this changes what is learned.
- Training on a Base model and on an Instruct model are different experiments. This config uses Instruct to compare directly with the baseline.

---

## 4. DPO (direct preference optimization)

**Idea.** Given a prompt with a preferred answer `y_w` and a rejected answer `y_l`, raise the probability of `y_w` relative to `y_l`, while staying close to a frozen reference model. No reward model and no sampling during training.

**Math.**

```
L_DPO = - log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )
```

- `pi` is the model being trained, `pi_ref` the frozen starting model.
- `beta` sets how far the model may move from the reference: small beta allows larger changes, large beta keeps it close.
- It comes from the RLHF objective (maximize reward minus beta x KL to the reference), solved in closed form so the reward is implicit: `r(x,y) = beta * log(pi(y|x) / pi_ref(y|x))`.

**Reference model with LoRA.** No second model copy: the reference is the same base with the adapter turned off. That is why DPO fits in memory here.

**Data.** `trl-lib/ultrafeedback_binarized`, 1500 rows of (prompt, chosen, rejected).

**Here.** `run_dpo` in [src/lab/train.py](../src/lab/train.py), TRL `DPOConfig` / `DPOTrainer`. Config [configs/dpo_qlora.yaml](../configs/dpo_qlora.yaml): `beta 0.1`, `learning_rate 5e-6` (much lower than SFT: DPO is easy to over-optimize), `warmup_steps 0.1`. The planned sweep is beta 0.05 / 0.1 / 0.3:

```bash
docker compose run --rm lab python -m lab.train --config configs/dpo_qlora.yaml --set train.beta=0.3 --set train.output_dir=outputs/dpo-b0.3
```

**Watch.**
- `rewards/chosen`, `rewards/rejected`: implicit rewards (beta x log ratio). Both often go down; what matters is the gap.
- `rewards/margins`: chosen minus rejected, should rise above 0.
- `rewards/accuracies`: share of pairs where chosen scores higher. Starts near 0.5 and should climb. At 5 smoke steps it was 0.31, which is just noise.
- `loss` starts at log 2 = 0.693 (smoke: 0.695) and should fall.

**Pitfalls.**
- About 6x slower per step than SFT: every pair runs 4 forward passes (chosen and rejected, policy and reference). Smoke: about 40 s/step while sharing the GPU with SFT.
- UltraFeedback is general helpfulness, not math, so a GSM8K gain is not expected. Report it honestly as off-task preference data.
- The chosen log-probability falling while the margin grows is normal DPO behavior, but if accuracy on the task drops, beta is too low or training too long.

---

## Reproducing and comparing

- Environment: `Dockerfile` + `uv.lock` pin every package (torch 2.13, trl 1.13.0, transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2).
- Each record stores the git commit and dirty flag, model and dataset revisions, full config, metrics, wall time, GPU hours and peak memory.
- Smoke-test records go to `outputs/smoke-records/` so they never enter the report.
- Report: `docker compose run --rm lab python -m lab.report`.
- MLflow UI: `docker compose run --rm -p 5000:5000 lab mlflow ui --backend-store-uri sqlite:////workspace/mlflow.db --host 0.0.0.0`, then open http://localhost:5000.

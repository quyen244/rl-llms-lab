"""Upload cached teacher generations as a private Hugging Face dataset with a dataset card.

Usage: python scripts/push_teacher_data.py <teacher jsonl> <distill run record json>
"""
import json
import os
import sys
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

CARD = """---
license: mit
tags: [rl-lab, distillation, gsm8k]
source_datasets: [openai/gsm8k]
---

# {repo}

Step-by-step solutions written by **{teacher}** (4-bit, revision `{teacher_rev}`) for {n} questions of the
GSM8K `main` train split ({data_rev}), sampled with shuffle seed {seed}. Used for sequence-level and logit-KL
distillation into Qwen2.5-0.5B-Instruct in the rl-lab project.

| column | content |
|---|---|
| `prompt` | chat-templated question plus the instruction to end with `#### <number>` |
| `completion` | the teacher's raw greedy generation (max 384 new tokens) |
| `gold_answer` | the GSM8K reference answer |

Teacher quality on these rows: {acc:.1%} correct, {fmt:.1%} used the `#### n` format.
Generation was greedy; rows were produced in batches of 96 and 128, which can cause tiny numeric differences.
Completions are not filtered for correctness.
"""

if __name__ == "__main__":
    load_dotenv()
    jsonl, record = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    info = record["dataset"]["train"]
    teacher = info["teacher_model"]
    repo = f"{os.environ['HF_USERNAME']}/rl-lab-gsm8k-teacher-{teacher.split('/')[-1].lower()}"
    token = os.environ["HF_TOKEN"]
    Dataset.from_list(rows).push_to_hub(repo, private=True, token=token)
    card = CARD.format(repo=repo, teacher=teacher, teacher_rev=info.get("teacher_revision"),
                       n=len(rows), data_rev=info.get("revision"), seed=info.get("sample_seed"),
                       acc=info.get("teacher_gsm8k_acc") or 0, fmt=info.get("teacher_format_rate") or 0)
    HfApi(token=token).upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                                   repo_id=repo, repo_type="dataset", commit_message="dataset card")
    print(f"https://huggingface.co/datasets/{repo} ({len(rows)} rows)")

"""Upload a finished run's LoRA adapter, run record and model card to a private Hub repo.

Usage: python scripts/push_adapter.py outputs/records/<run>.json [...]
Writes the repo id and Hub commit back into the record's artifacts.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from lab.config import hub_model_id

CARD = """---
base_model: {base}
library_name: peft
tags: [rl-lab, qlora, {method}]
---

# {repo}

QLoRA adapter for `{base}` trained with **{method}** in the rl-lab project (learning-scale run).

| metric | value |
|---|---|
| GSM8K accuracy (first {n} test questions, greedy) | {acc} |
| `#### n` format rate | {fmt} |
| training GPU hours | {gpu_h} |
| peak GPU memory (GB) | {mem} |

Run `{run_id}`, git commit `{commit}`. Full config, dataset revisions and metrics: `run_record.json`.

```python
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("{repo}")
```
"""


def push(record_path: Path, api: HfApi, user: str) -> str:
    rec = json.loads(record_path.read_text())
    cfg = rec["config"]
    out_dir = Path(rec["artifacts"]["output_dir"])
    repo = hub_model_id(cfg, user)
    api.create_repo(repo, private=True, exist_ok=True)
    ev = rec["metrics"]["eval"]
    card = CARD.format(base=cfg["model"]["name"], method=cfg["method"], repo=repo,
                       n=ev.get("gsm8k_n"), acc=ev.get("gsm8k_acc"), fmt=ev.get("gsm8k_format_rate"),
                       gpu_h=rec["cost"].get("gpu_hours"), mem=rec["cost"].get("peak_gpu_mem_gb"),
                       run_id=rec["run_id"], commit=(rec.get("git", {}).get("commit") or "n/a")[:8])
    (out_dir / "README.md").write_text(card)
    (out_dir / "run_record.json").write_text(record_path.read_text())
    info = api.upload_folder(repo_id=repo, folder_path=str(out_dir),
                             allow_patterns=["adapter_*", "*.jinja", "tokenizer*", "README.md", "run_record.json"],
                             commit_message=f"{cfg['method']} run {rec['run_id']}")
    rec["artifacts"]["hub_model_id"] = repo
    rec["artifacts"]["hub_commit"] = info.oid
    record_path.write_text(json.dumps(rec, indent=2, default=str))
    return f"https://huggingface.co/{repo}"


if __name__ == "__main__":
    load_dotenv()
    user = os.environ["HF_USERNAME"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    for p in sys.argv[1:]:
        print(push(Path(p), api, user), flush=True)

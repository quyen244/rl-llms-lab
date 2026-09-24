"""Single entry point.

    python -m lab.train --config configs/sft_qlora.yaml [--set a.b=c] [--resume]
    python -m lab.train --config configs/baseline.yaml      # evaluation only, no training

Every run: logs to MLflow (params, dataset lineage, metrics, adapter, registry candidate),
evaluates on GSM8K, and writes outputs/records/<experiment>-<run_id>.json for the report.
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from lab.config import flatten, hub_model_id, load_config, registry_name
from lab.data import load_train_dataset
from lab.modeling import build_lora, build_model, build_tokenizer, resolve_dtype
from lab.record import build_record, cost_info, git_info, hub_revision, write_record
from lab.tracking import DEFAULT_TRACKING_URI, log_dataset, register_adapter


def _common_args(cfg) -> dict:
    """Precision, MLflow and Hugging Face Hub arguments shared by every trainer config."""
    dtype = resolve_dtype(cfg["model"].get("compute_dtype", "auto"))
    args = {
        "report_to": "mlflow",
        "run_name": cfg["experiment"],
        "bf16": dtype == "bfloat16",
        "fp16": dtype == "float16",
    }
    username = os.environ.get("HF_USERNAME")
    if cfg.get("hub", {}).get("push", False):
        if username and os.environ.get("HF_TOKEN"):
            args.update(
                push_to_hub=True,
                hub_model_id=hub_model_id(cfg, username),
                hub_private_repo=True,
                # Rented servers vanish: keep the last checkpoint on the Hub.
                hub_strategy="checkpoint",
            )
        else:
            print("hub.push is on but HF_USERNAME/HF_TOKEN are not set in .env; not pushing.")
    return args


def run_sft(cfg, model, tok, ds, lora):
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(**_common_args(cfg), **cfg["train"])
    return SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)


def run_dpo(cfg, model, tok, ds, lora):
    from trl import DPOConfig, DPOTrainer

    args = DPOConfig(**_common_args(cfg), **cfg["train"])
    return DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)


METHODS = {"sft": run_sft, "dpo": run_dpo}


def _train_metrics(trainer) -> dict:
    """Last logged value of every numeric training metric."""
    out: dict = {}
    for entry in trainer.state.log_history:
        for k, v in entry.items():
            if isinstance(v, (int, float)) and k not in ("epoch", "step"):
                out[k] = v
    out["global_step"] = trainer.state.global_step
    return out


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], dest="overrides")
    p.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output_dir")
    a = p.parse_args()
    cfg = load_config(a.config, a.overrides)
    method = cfg["method"]

    import mlflow
    import torch

    from lab.evals import evaluate_gsm8k

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(cfg["experiment"])
    run_id = uuid.uuid4().hex[:8]

    tok = build_tokenizer(cfg["model"]["name"])
    model = build_model(cfg["model"])

    with mlflow.start_run(run_name=f"{cfg['experiment']}-{run_id}") as run:
        mlflow.set_tags({"method": method, "base_model": cfg["model"]["name"], "lab_run_id": run_id})
        mlflow.log_params({f"cfg.{k}": v for k, v in flatten(cfg).items()})

        wall, peak_gb, train_metrics, dataset_info, eval_model = 0.0, None, {}, None, model
        out_dir = Path(cfg.get("train", {}).get("output_dir", "outputs/baseline"))
        if method != "baseline":
            ds, dataset_info = load_train_dataset(cfg["data"], cfg["train"].get("seed", 42))
            log_dataset(ds, dataset_info["name"], dataset_info["split"], "training")
            trainer = METHODS[method](cfg, model, tok, ds, build_lora(cfg["lora"]))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            trainer.train(resume_from_checkpoint=True if a.resume else None)
            wall = time.time() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
            trainer.save_model(str(out_dir))
            if trainer.args.push_to_hub:
                trainer.push_to_hub()
            train_metrics = _train_metrics(trainer)
            eval_model = trainer.model

        eval_cfg = cfg.get("eval", {})
        eval_metrics = evaluate_gsm8k(eval_model, tok, n=eval_cfg.get("gsm8k_samples", 1000)) if eval_cfg.get("gsm8k_samples", 1000) else {}
        mlflow.log_metrics({f"final.{k}": v for k, v in eval_metrics.items()})
        mlflow.log_metrics({f"train_summary.{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))})

        artifacts: dict = {"output_dir": str(out_dir)}
        if method != "baseline":
            for f in out_dir.iterdir():  # adapter files only, not intermediate checkpoint folders
                if f.is_file():
                    mlflow.log_artifact(str(f), "adapter")
            name = registry_name(cfg)
            artifacts["mlflow_model_uri"] = register_adapter(
                run.info.run_id, "adapter", name,
                {"lab_run_id": run_id, "git_commit": git_info()["commit"], "gsm8k_acc": eval_metrics.get("gsm8k_acc")},
            )
            if os.environ.get("HF_USERNAME") and cfg.get("hub", {}).get("push"):
                artifacts["hub_model_id"] = hub_model_id(cfg, os.environ["HF_USERNAME"])

        eval_info = {"name": "openai/gsm8k", "config": "main", "split": "test",
                     "revision": hub_revision("dataset", "openai/gsm8k"),
                     "rows_used": eval_metrics.get("gsm8k_n")}
        n_gpu = torch.cuda.device_count()
        rec = build_record(
            run_id=run_id, mlflow_run_id=run.info.run_id, cfg=cfg,
            dataset={"train": dataset_info, "eval": eval_info},
            metrics={"train": train_metrics, "eval": eval_metrics},
            cost=cost_info(wall, n_gpu, peak_gb, cfg.get("cost", {}).get("gpu_hourly_usd")),
            artifacts=artifacts,
        )
        path = write_record(rec)
        mlflow.log_artifact(str(path), "record")
        print(f"run record: {path}")


if __name__ == "__main__":
    main()

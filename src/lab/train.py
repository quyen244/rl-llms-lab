"""Single entry point: python -m lab.train --config configs/sft_qlora.yaml [--set a.b=c] [--resume]"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from lab.config import flatten, hub_model_id, load_config
from lab.data import load_train_dataset
from lab.modeling import build_lora, build_model, build_tokenizer, resolve_dtype


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


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], dest="overrides")
    p.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output_dir")
    a = p.parse_args()
    cfg = load_config(a.config, a.overrides)

    import mlflow

    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", cfg["experiment"])
    tok = build_tokenizer(cfg["model"]["name"])
    model = build_model(cfg["model"])
    ds = load_train_dataset(cfg["data"], cfg["train"].get("seed", 42))
    trainer = METHODS[cfg["method"]](cfg, model, tok, ds, build_lora(cfg["lora"]))

    trainer.train(resume_from_checkpoint=True if a.resume else None)
    trainer.save_model(cfg["train"]["output_dir"])
    if cfg.get("hub", {}).get("push", False) and trainer.args.push_to_hub:
        trainer.push_to_hub()
    # Log the resolved config to the run TRL's MLflow callback opened.
    if mlflow.active_run():
        mlflow.log_params(flatten(cfg))
        mlflow.log_artifact(a.config)


if __name__ == "__main__":
    main()

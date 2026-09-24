"""Single entry point: python -m lab.train --config configs/sft_qlora.yaml [--set a.b=c]"""
from __future__ import annotations

import argparse
import os

from lab.config import flatten, load_config
from lab.data import load_train_dataset
from lab.modeling import build_lora, build_model, build_tokenizer


def _precision(cfg) -> dict:
    dtype = cfg["model"]["compute_dtype"]
    return {"bf16": dtype == "bfloat16", "fp16": dtype == "float16"}


def run_sft(cfg, model, tok, ds, lora):
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(report_to="mlflow", run_name=cfg["experiment"], **_precision(cfg), **cfg["train"])
    return SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)


def run_dpo(cfg, model, tok, ds, lora):
    from trl import DPOConfig, DPOTrainer

    args = DPOConfig(report_to="mlflow", run_name=cfg["experiment"], **_precision(cfg), **cfg["train"])
    return DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)


METHODS = {"sft": run_sft, "dpo": run_dpo}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], dest="overrides")
    a = p.parse_args()
    cfg = load_config(a.config, a.overrides)

    import mlflow

    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", cfg["experiment"])
    tok = build_tokenizer(cfg["model"]["name"])
    model = build_model(cfg["model"])
    ds = load_train_dataset(cfg["data"], cfg["train"].get("seed", 42))
    trainer = METHODS[cfg["method"]](cfg, model, tok, ds, build_lora(cfg["lora"]))

    trainer.train()
    trainer.save_model(cfg["train"]["output_dir"])
    # Log the resolved config to the run TRL's MLflow callback opened.
    if mlflow.active_run():
        mlflow.log_params(flatten(cfg))
        mlflow.log_artifact(a.config)


if __name__ == "__main__":
    main()

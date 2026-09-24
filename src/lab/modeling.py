"""Model, tokenizer and QLoRA/LoRA setup shared by all methods."""
from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def resolve_dtype(name: str) -> str:
    """'auto' picks bfloat16 when the GPU supports it, else float16 (T4, V100)."""
    if name != "auto":
        return name
    return "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"


def build_model(model_cfg: dict[str, Any]):
    dtype = getattr(torch, resolve_dtype(model_cfg.get("compute_dtype", "auto")))
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if model_cfg.get("load_in_4bit", False):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}
    return AutoModelForCausalLM.from_pretrained(model_cfg["name"], **kwargs)


def build_lora(lora_cfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

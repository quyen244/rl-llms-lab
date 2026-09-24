"""Download models and datasets into the HF cache so GPU runs never wait on the network.

Usage: python scripts/prefetch.py [--with-7b]
"""
import argparse

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
]
MODELS_7B = ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"]
DATASETS = ["openai/gsm8k", "trl-lib/Capybara", "trl-lib/ultrafeedback_binarized"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--with-7b", action="store_true")
    a = p.parse_args()
    for repo in MODELS + (MODELS_7B if a.with_7b else []):
        print("model", repo, flush=True)
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
    for repo in DATASETS:
        print("dataset", repo, flush=True)
        snapshot_download(repo, repo_type="dataset")
    print("done", flush=True)

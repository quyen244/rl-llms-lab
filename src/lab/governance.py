"""Model governance: promote a 'candidate' model version to 'champion' only if it measurably wins.

    python -m lab.governance promote --model rl-lab-dpo-qlora-qwen2-5-1-5b-instruct --metric final.gsm8k_acc
    python -m lab.governance show --model <name>

Aliases: 'candidate' is set automatically after every training run; 'champion' is set only here.
"""
from __future__ import annotations

import argparse
import os


def is_better(new: float | None, old: float | None, min_gain: float = 0.0, higher_is_better: bool = True) -> bool:
    """True if new beats old by strictly more than min_gain. No old value means new wins; no new value never wins."""
    if new is None:
        return False
    if old is None:
        return True
    gain = (new - old) if higher_is_better else (old - new)
    return gain > min_gain


def _client():
    import mlflow
    from lab.tracking import DEFAULT_TRACKING_URI

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
    return mlflow.MlflowClient()


def _metric(client, version, metric: str) -> float | None:
    return client.get_run(version.run_id).data.metrics.get(metric)


def promote(model: str, metric: str, min_gain: float = 0.01, higher_is_better: bool = True) -> str:
    client = _client()
    cand = client.get_model_version_by_alias(model, "candidate")
    try:
        champ = client.get_model_version_by_alias(model, "champion")
    except Exception:
        champ = None
    new = _metric(client, cand, metric)
    old = _metric(client, champ, metric) if champ else None
    if is_better(new, old, min_gain, higher_is_better):
        client.set_registered_model_alias(model, "champion", cand.version)
        client.set_model_version_tag(model, cand.version, "promoted_on", metric)
        return f"promoted v{cand.version} to champion ({metric}: {old} -> {new})"
    return f"kept champion (candidate v{cand.version} {metric}={new} does not beat {old} by more than {min_gain})"


def show(model: str) -> str:
    client = _client()
    lines = []
    for mv in client.search_model_versions(f"name='{model}'"):
        lines.append(f"v{mv.version} aliases={list(mv.aliases)} run={mv.run_id} tags={dict(mv.tags)}")
    return "\n".join(lines) or "no versions"


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("promote")
    pp.add_argument("--model", required=True)
    pp.add_argument("--metric", default="final.gsm8k_acc")
    pp.add_argument("--min-gain", type=float, default=0.01)
    sp = sub.add_parser("show")
    sp.add_argument("--model", required=True)
    a = p.parse_args()
    print(promote(a.model, a.metric, a.min_gain) if a.cmd == "promote" else show(a.model))


if __name__ == "__main__":
    main()

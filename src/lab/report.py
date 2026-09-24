"""Turn run records into a comparison report.

    python -m lab.report --records "outputs/records/*.json" --out reports/latest

Writes report.md plus PNG charts (if matplotlib is installed). Only the JSON records are read,
so a report can be rebuilt anywhere from the records alone.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Any

from lab.methods import METHOD_PROFILES, complexity_score
from lab.record import load_records

PRIMARY = "metrics.eval.gsm8k_acc"
N_KEY = "metrics.eval.gsm8k_n"


def get(rec: dict[str, Any], path: str, default=None):
    node: Any = rec
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n <= 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def pareto_front(points: list[tuple[str, float, float]]) -> set[str]:
    """Names of points not dominated by another (lower-or-equal cost AND higher-or-equal score, one strict)."""
    front = set()
    for name, c, s in points:
        dominated = any(c2 <= c and s2 >= s and (c2 < c or s2 > s) for n2, c2, s2 in points if n2 != name)
        if not dominated:
            front.add(name)
    return front


def _label(rec: dict[str, Any]) -> str:
    return f"{rec['experiment']}-{rec['run_id']}"


def _fmt(x, spec=".3f", none="n/a") -> str:
    return none if x is None else format(x, spec)


def _baselines(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Latest baseline record per base model name."""
    out: dict[str, dict[str, Any]] = {}
    for r in sorted(records, key=lambda r: r["created_at"]):
        if r["method"] == "baseline":
            out[r["model"]["name"]] = r
    return out


def _cost_axis(records: list[dict[str, Any]]) -> tuple[str, str]:
    """Use dollars only if every trained run has a price, otherwise GPU hours."""
    trained = [r for r in records if r["method"] != "baseline"]
    if trained and all(get(r, "cost.cost_usd") is not None for r in trained):
        return "cost.cost_usd", "Cost (USD)"
    return "cost.gpu_hours", "GPU hours"


def build_report(records: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = _baselines(records)
    cost_key, cost_label = _cost_axis(records)
    trained = [r for r in records if r["method"] != "baseline" and get(r, PRIMARY) is not None]

    rows = []
    for r in trained:
        acc, n = get(r, PRIMARY), int(get(r, N_KEY, 0))
        lo, hi = wilson_ci(acc, n)
        b = base.get(r["model"]["name"])
        b_acc = get(b, PRIMARY) if b else None
        b_lo, b_hi = wilson_ci(b_acc, int(get(b, N_KEY, 0))) if b_acc is not None else (None, None)
        delta = acc - b_acc if b_acc is not None else None
        significant = None if b_acc is None else (lo > b_hi or hi < b_lo)
        cost = get(r, cost_key)
        rows.append({
            "rec": r, "label": _label(r), "acc": acc, "n": n, "ci": (lo, hi), "delta": delta,
            "significant": significant, "cost": cost, "hours": get(r, "cost.gpu_hours"),
            "gain_per_hour": (delta / get(r, "cost.gpu_hours")) if delta is not None and get(r, "cost.gpu_hours") else None,
        })

    front = pareto_front([(x["label"], x["cost"], x["delta"] if x["delta"] is not None else x["acc"])
                          for x in rows if x["cost"] is not None])
    charts = _charts(rows, front, cost_label, out)

    md: list[str] = []
    md.append("# Post-training experiment report\n")
    md.append(f"{len(records)} run records, {len(trained)} trained runs with an evaluation, "
              f"{len(base)} baseline model(s). Primary metric: GSM8K exact-match accuracy.\n")

    md.append("## 1. Key findings\n")
    md += _findings(rows, front, cost_label)

    md.append("\n## 2. Leaderboard\n")
    md.append(f"| Run | Method | Base model | GSM8K acc | 95% CI | vs baseline | Significant | GPU h | {cost_label} | Peak mem GB |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for x in sorted(rows, key=lambda x: -x["acc"]):
        r = x["rec"]
        sig = {None: "no baseline", True: "yes", False: "no (CI overlaps)"}[x["significant"]]
        md.append(f"| {x['label']} | {r['method']} | {r['model']['name']} | {x['acc']:.3f} | "
                  f"{x['ci'][0]:.3f}-{x['ci'][1]:.3f} | {_fmt(x['delta'], '+.3f')} | {sig} | "
                  f"{_fmt(x['hours'], '.2f')} | {_fmt(x['cost'], '.2f')} | {_fmt(get(r, 'cost.peak_gpu_mem_gb'), '.1f')} |")
    md.append("\nCI is a 95% Wilson interval from the test-set size. A gain counts as significant only when the "
              "run's interval does not overlap the baseline's. Every row is a single seed.\n")

    md.append("## 3. Trade-offs: performance, cost, complexity\n")
    if charts.get("tradeoff"):
        md.append(f"![accuracy gain vs cost]({charts['tradeoff']})\n")
        md.append("Points on the Pareto front are marked; nothing else is both cheaper and better.\n")
    md += _method_table(rows)

    md.append("\n## 4. Pros and cons by method\n")
    for m in sorted({x["rec"]["method"] for x in rows}, key=complexity_score):
        p = METHOD_PROFILES.get(m)
        if not p:
            continue
        md.append(f"### {p['title']} (`{m}`, complexity {complexity_score(m)})\n")
        md.append(f"Learning signal: {p['signal']}. Key hyperparameters: {', '.join(p['key_hparams']) or 'none'}.\n")
        md.append("Pros:\n" + "\n".join(f"- {s}" for s in p["pros"]) + "\n")
        md.append("Cons:\n" + "\n".join(f"- {s}" for s in p["cons"]) + "\n")

    md.append("## 5. Reproducibility and governance\n")
    md.append("| Run | Git commit | Dirty | Base model revision | Train data revision | Rows used | Seed | Registry |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in records:
        tr = get(r, "dataset.train") or {}
        md.append(f"| {_label(r)} | {(get(r, 'git.commit') or 'n/a')[:8]} | {get(r, 'git.dirty')} | "
                  f"{(get(r, 'model.revision') or 'n/a')[:8]} | {(tr.get('revision') or 'n/a')[:8]} | "
                  f"{tr.get('rows_used', 'n/a')} | {get(r, 'config.train.seed', 'n/a')} | {get(r, 'artifacts.mlflow_model_uri') or 'n/a'} |")
    md.append("\n## 6. Limitations\n")
    md.append("- Single seed per configuration; differences smaller than the confidence interval are noise.")
    md.append("- One benchmark (GSM8K). Gains on it say little about general chat quality or safety.")
    md.append("- Cost depends on the rental price entered in the config; runs without a price are compared in GPU hours.")
    md.append("- Complexity scores are design-level (components required), not a measure of implementation effort.\n")

    path = out / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def _findings(rows, front, cost_label) -> list[str]:
    if not rows:
        return ["- No evaluated training runs yet."]
    out = []
    best = max(rows, key=lambda x: x["acc"])
    out.append(f"- Best accuracy: **{best['label']}** ({best['rec']['method']}) at {best['acc']:.3f}.")
    priced = [x for x in rows if x["cost"] is not None]
    if priced:
        cheapest = min(priced, key=lambda x: x["cost"])
        out.append(f"- Cheapest run: **{cheapest['label']}** ({cheapest['rec']['method']}), {cost_label} = {cheapest['cost']:.2f}.")
    eff = [x for x in rows if x["gain_per_hour"] is not None]
    if eff:
        e = max(eff, key=lambda x: x["gain_per_hour"])
        out.append(f"- Best accuracy gain per GPU hour: **{e['label']}** ({e['gain_per_hour']:+.3f} per hour).")
    if front:
        out.append(f"- Pareto-optimal runs (cost vs gain): {', '.join(sorted(front))}.")
    ns = [x for x in rows if x["significant"] is False]
    if ns:
        out.append(f"- Not distinguishable from baseline at 95%: {', '.join(x['label'] for x in ns)}.")
    return out


def _method_table(rows) -> list[str]:
    by: dict[str, list] = {}
    for x in rows:
        by.setdefault(x["rec"]["method"], []).append(x)
    out = ["| Method | Runs | Best acc | Mean gain vs baseline | Mean wall min | Mean peak mem GB | Complexity | Extra models / parts |",
           "|---|---|---|---|---|---|---|---|"]
    for m, xs in sorted(by.items(), key=lambda kv: complexity_score(kv[0]) if kv[0] in METHOD_PROFILES else 99):
        p = METHOD_PROFILES.get(m, {})
        parts = [k for k in ("reference", "reward_model", "value_model", "teacher", "teacher_resident", "online_generation") if p.get(k)]
        deltas = [x["delta"] for x in xs if x["delta"] is not None]
        mems = [get(x["rec"], "cost.peak_gpu_mem_gb") for x in xs if get(x["rec"], "cost.peak_gpu_mem_gb") is not None]
        mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
        out.append(f"| {m} | {len(xs)} | {max(x['acc'] for x in xs):.3f} | {_fmt(mean(deltas), '+.3f')} | "
                   f"{_fmt(mean([get(x['rec'], 'cost.wall_seconds', 0) / 60 for x in xs]), '.1f')} | {_fmt(mean(mems), '.1f')} | "
                   f"{complexity_score(m) if p else 'n/a'} | {', '.join(parts) or 'none'} |")
    return out


def _charts(rows, front, cost_label, out: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    pts = [x for x in rows if x["cost"] is not None]
    if not pts:
        return {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for x in pts:
        y = x["delta"] if x["delta"] is not None else x["acc"]
        ax.scatter(x["cost"], y, s=90 if x["label"] in front else 45, marker="D" if x["label"] in front else "o")
        ax.annotate(x["rec"]["method"], (x["cost"], y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(cost_label)
    ax.set_ylabel("GSM8K accuracy gain vs baseline" if any(x["delta"] is not None for x in pts) else "GSM8K accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=150)
    plt.close(fig)
    return {"tradeoff": "tradeoff.png"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="outputs/records/*.json")
    p.add_argument("--out", default="reports/latest")
    a = p.parse_args()
    paths = sorted(glob.glob(a.records))
    if not paths:
        raise SystemExit(f"no records match {a.records}")
    print(build_report(load_records(paths), a.out))


if __name__ == "__main__":
    main()

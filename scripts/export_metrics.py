"""Export the full MLflow metric history of every recorded run to CSV, so curves are versioned without mlflow.db.

Usage: python scripts/export_metrics.py
Writes outputs/metrics/<experiment>-<run_id>.csv with columns metric, step, value, timestamp.
"""
import csv
import json
import os
from pathlib import Path

from mlflow import MlflowClient

from lab.tracking import DEFAULT_TRACKING_URI

out = Path("outputs/metrics")
out.mkdir(parents=True, exist_ok=True)
client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
for path in sorted(Path("outputs/records").glob("*.json")):
    rec = json.loads(path.read_text())
    run_id = rec.get("mlflow_run_id")
    if not run_id:
        print(f"skip {path.name}: no mlflow_run_id")
        continue
    rows = []
    for key in sorted(client.get_run(run_id).data.metrics):
        rows += [(key, m.step, m.value, m.timestamp) for m in client.get_metric_history(run_id, key)]
    dest = out / f"{path.stem}.csv"
    with dest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "step", "value", "timestamp"])
        w.writerows(rows)
    print(f"{dest}: {len(rows)} points, {len({r[0] for r in rows})} metrics")

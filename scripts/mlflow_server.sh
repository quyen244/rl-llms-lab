#!/usr/bin/env bash
# Tracking server (sqlite + artifact folder). Point jobs at it via MLFLOW_TRACKING_URI.
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

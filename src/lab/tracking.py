"""MLflow helpers: dataset lineage and model registry."""
from __future__ import annotations

from typing import Any

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"  # a DB-backed store is required for the model registry


def log_dataset(ds, name: str, split: str, context: str) -> None:
    """Attach the dataset to the current run as an MLflow input (name, digest, schema, profile)."""
    try:
        import mlflow
        import mlflow.data

        mlflow.log_input(mlflow.data.from_huggingface(ds, path=name, name=f"{name}/{split}"), context=context)
    except Exception as e:  # lineage must never kill a training run
        print(f"warning: could not log dataset lineage: {e}")


def register_adapter(run_id: str, artifact_path: str, model_name: str, tags: dict[str, Any]) -> str | None:
    """Register the adapter as a new model version and alias it 'candidate'."""
    try:
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except MlflowException:
            pass  # already exists
        # MLflow 3 register_model needs a logged model; a raw adapter folder is registered by artifact URI.
        source = f"{client.get_run(run_id).info.artifact_uri}/{artifact_path}"
        mv = client.create_model_version(model_name, source, run_id=run_id,
                                         tags={k: str(v) for k, v in tags.items()})
        client.set_registered_model_alias(model_name, "candidate", mv.version)
        return f"models:/{model_name}/{mv.version}"
    except Exception as e:
        print(f"warning: could not register model: {e}")
        return None

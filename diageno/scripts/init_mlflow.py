#!/usr/bin/env python3
"""
scripts/init_mlflow.py — One-time MLflow setup.

Creates the 'diageno' experiment and verifies MLflow / MinIO connectivity.
"""
import os
import sys

import mlflow


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "diageno"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        eid = mlflow.create_experiment(
            experiment_name,
            artifact_location="s3://diageno-artifacts/mlflow",
        )
        print(f"Created MLflow experiment '{experiment_name}' (id={eid})")
    else:
        print(f"MLflow experiment '{experiment_name}' already exists (id={experiment.experiment_id})")

    # Quick connectivity check
    try:
        with mlflow.start_run(experiment_id=experiment.experiment_id if experiment else eid, run_name="connectivity-test"):
            mlflow.log_param("test", "ok")
        print("MLflow connectivity OK ✓")
    except Exception as exc:
        print(f"MLflow connectivity failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Train All — orchestrator that builds/fits all artifacts.

Artifacts:
  A: Disease posterior scorer  (disease_hpo_matrix, calibration)
  B: Next-best-phenotype selector (uses A's matrix; no separate training)
  C: Test recommendation policy (policy.yaml)
  D: Embeddings (sentence-transformer → pgvector)

Optionally logs everything to MLflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from diageno.config import settings

logger = logging.getLogger("diageno.training")


def _dataset_hash() -> str:
    """Compute a hash over silver parquet files for versioning."""
    from diageno.etl.utils import sha256_dir
    return sha256_dir(settings.silver_dir)[:12]


def run(skip_embeddings: bool = False, use_mlflow: bool = True) -> None:
    """Run the full training pipeline."""
    start = time.time()
    settings.ensure_dirs()

    logger.info("=" * 60)
    logger.info("DIAGENO MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    dataset_hash = _dataset_hash()
    logger.info("Dataset hash: %s", dataset_hash)

    # ── MLflow setup ──
    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment("diageno-training")
            mlflow_run = mlflow.start_run(run_name=f"train-{dataset_hash}")
            mlflow.log_param("dataset_hash", dataset_hash)
            logger.info("MLflow run started: %s", mlflow_run.info.run_id)
        except Exception as e:
            logger.warning("MLflow not available (%s) — continuing without tracking", e)
            use_mlflow = False

    # ── Artifact A: Disease Scorer (Enhanced with cosine similarity, HPO expansion, gene integration) ──
    logger.info("── Building Artifact A: Enhanced Disease Scorer ──")
    from diageno.training.enhanced_scorer import run as run_enhanced_scorer
    run_enhanced_scorer()

    # ── Artifact B: verify matrix loads ──
    logger.info("── Verifying Artifact B: Phenotype Selector ──")
    from diageno.training.phenotype_selector import run as run_selector
    run_selector()

    # ── Artifact C: Test Policy ──
    logger.info("── Building Artifact C: Test Policy ──")
    from diageno.training.test_policy import run as run_policy
    run_policy()

    # ── Artifact D: Embeddings (optional) ──
    if not skip_embeddings:
        logger.info("── Building Artifact D: Embeddings ──")
        from diageno.training.embeddings import run as run_embeddings
        run_embeddings()
    else:
        logger.info("── Skipping Artifact D (embeddings) ──")

    elapsed = time.time() - start

    # ── Log to MLflow ──
    if use_mlflow and mlflow_run:
        try:
            import mlflow
            mlflow.log_metric("training_time_sec", elapsed)
            # Log artifact files
            artifacts = settings.artifacts_dir
            for f in artifacts.iterdir():
                if f.is_file():
                    mlflow.log_artifact(str(f))
            mlflow.end_run()
            logger.info("MLflow run logged successfully")
        except Exception as e:
            logger.warning("MLflow logging error: %s", e)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE in %.1f seconds", elapsed)
    logger.info("Artifacts → %s", settings.artifacts_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train Diageno model artifacts")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    args = parser.parse_args()

    run(skip_embeddings=args.skip_embeddings, use_mlflow=not args.no_mlflow)

"""Artifact D — Embedding Model.

Generates case and disease embeddings using a sentence-transformer model,
stores them in pgvector for ANN retrieval ("patients-like-me").

Inputs:  disease table, hpo_term table, case + phenotype data
Outputs: embeddings written to pgvector tables
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from diageno.config import settings

logger = logging.getLogger("diageno.training.embeddings")


def _get_model(model_name: str | None = None):
    """Load sentence-transformer model (lazy import)."""
    from sentence_transformers import SentenceTransformer

    name = model_name or settings.embedding_model_name
    logger.info("Loading embedding model: %s", name)
    return SentenceTransformer(name)


def disease_text(disease_name: str, hpo_names: list[str]) -> str:
    """Create text representation of a disease for embedding."""
    hpo_str = "; ".join(hpo_names[:30])  # limit for context window
    return f"Disease: {disease_name}. Associated phenotypes: {hpo_str}"


def case_text(phenotype_labels: list[str], age: int | None = None, sex: str | None = None) -> str:
    """Create text representation of a case for embedding."""
    parts = []
    if age:
        parts.append(f"Age: {age}")
    if sex:
        parts.append(f"Sex: {sex}")
    parts.append(f"Phenotypes: {'; '.join(phenotype_labels)}")
    return ". ".join(parts)


def embed_diseases(artifacts: Path) -> None:
    """Generate embeddings for all diseases and save to artifacts + DB.

    Reads disease_index.json and HPO dict to create text representations,
    then embeds them.
    """
    di_path = artifacts / "disease_index.json"
    if not di_path.exists():
        logger.warning("disease_index.json not found — run Artifact A first")
        return

    with open(di_path) as f:
        disease_index = json.load(f)

    # Try to load disease names from silver
    silver = settings.silver_dir
    diseases_path = silver / "diseases.parquet"
    hpo_path = silver / "disease_hpo.parquet"
    hpo_terms_path = silver / "hpo_terms.parquet"

    disease_names: dict[str, str] = {}
    disease_hpos: dict[str, list[str]] = {}

    if diseases_path.exists():
        import pandas as pd
        df = pd.read_parquet(diseases_path)
        for _, row in df.iterrows():
            disease_names[row["disease_id"]] = row.get("name", row["disease_id"])

    if hpo_path.exists() and hpo_terms_path.exists():
        import pandas as pd
        df_hpo = pd.read_parquet(hpo_path)
        df_terms = pd.read_parquet(hpo_terms_path)
        hpo_name_map = dict(zip(df_terms["hpo_id"], df_terms["name"]))
        for did, group in df_hpo.groupby("disease_id"):
            disease_hpos[did] = [
                hpo_name_map.get(h, h) for h in group["hpo_id"].tolist()
            ]

    # Generate texts and embed
    model = _get_model()
    texts = []
    ids = []
    for did in disease_index:
        name = disease_names.get(did, did)
        hpos = disease_hpos.get(did, [])
        texts.append(disease_text(name, hpos))
        ids.append(did)

    if not texts:
        logger.warning("No disease texts to embed")
        return

    logger.info("Embedding %d diseases …", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # Save as numpy for portability
    np.save(artifacts / "disease_embeddings.npy", embeddings)
    with open(artifacts / "disease_embedding_ids.json", "w") as f:
        json.dump(ids, f)

    logger.info("Disease embeddings saved → disease_embeddings.npy (%s)", embeddings.shape)

    # Optionally write to pgvector
    _write_embeddings_to_db(ids, embeddings, table="disease_embedding", id_col="disease_id")


def _write_embeddings_to_db(
    ids: list[str],
    embeddings: np.ndarray,
    table: str,
    id_col: str,
) -> None:
    """Batch-write embeddings to pgvector table."""
    try:
        from sqlalchemy import text as sa_text
        from diageno.db.session import engine

        model_name = settings.embedding_model_name
        with engine.begin() as conn:
            # Truncate first
            conn.execute(sa_text(f"TRUNCATE TABLE {table}"))
            for i, (eid, emb) in enumerate(zip(ids, embeddings)):
                emb_list = emb.tolist()
                conn.execute(
                    sa_text(
                        f"INSERT INTO {table} ({id_col}, embedding, model_name) "
                        f"VALUES (:id, :emb, :model)"
                    ),
                    {"id": eid, "emb": str(emb_list), "model": model_name},
                )
            logger.info("Wrote %d embeddings → %s", len(ids), table)
    except Exception as e:
        logger.warning("Could not write embeddings to DB (non-fatal): %s", e)


def run() -> None:
    """Build Artifact D."""
    artifacts = settings.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    embed_diseases(artifacts)

    # Save config
    config = {
        "model_name": settings.embedding_model_name,
        "dim": settings.embedding_dim,
    }
    with open(artifacts / "embedding_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("=== Artifact D (Embeddings) complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()

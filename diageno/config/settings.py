"""Application settings loaded from env / .env file."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # ── Postgres ──────────────────────────────────────
    postgres_dsn: str = os.getenv(
        "POSTGRES_DSN", "postgresql://diageno:diageno@localhost:5432/diageno"
    )

    # ── Redis ─────────────────────────────────────────
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ── MinIO / S3 ────────────────────────────────────
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")

    # ── MLflow ────────────────────────────────────────
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # ── Data directories ──────────────────────────────
    data_dir: Path = Path(os.getenv("DATA_DIR", str(_ROOT / "data")))
    bronze_dir: Path = field(default=None)  # type: ignore[assignment]
    silver_dir: Path = field(default=None)  # type: ignore[assignment]
    gold_dir: Path = field(default=None)  # type: ignore[assignment]
    artifacts_dir: Path = field(default=None)  # type: ignore[assignment]

    # ── Embedding ─────────────────────────────────────
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # ── Cache TTL (seconds) ───────────────────────────
    cache_ttl_hpo_lookup: int = int(os.getenv("CACHE_TTL_HPO_LOOKUP", "3600"))
    cache_ttl_disease_top: int = int(os.getenv("CACHE_TTL_DISEASE_TOP", "300"))
    cache_ttl_recommend: int = int(os.getenv("CACHE_TTL_RECOMMEND", "300"))
    cache_ttl_evidence: int = int(os.getenv("CACHE_TTL_EVIDENCE", "1800"))

    # ── API ───────────────────────────────────────────
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_workers: int = int(os.getenv("API_WORKERS", "2"))
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")

    def __post_init__(self) -> None:
        # Derive sub-dirs from data_dir if not explicitly set
        object.__setattr__(
            self, "bronze_dir", Path(os.getenv("BRONZE_DIR", str(self.data_dir / "bronze")))
        )
        object.__setattr__(
            self, "silver_dir", Path(os.getenv("SILVER_DIR", str(self.data_dir / "silver")))
        )
        object.__setattr__(
            self, "gold_dir", Path(os.getenv("GOLD_DIR", str(self.data_dir / "gold")))
        )
        object.__setattr__(
            self,
            "artifacts_dir",
            Path(os.getenv("ARTIFACTS_DIR", str(self.data_dir / "model_artifacts"))),
        )

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in (self.bronze_dir, self.silver_dir, self.gold_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()

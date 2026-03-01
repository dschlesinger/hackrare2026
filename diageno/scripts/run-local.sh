#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
# run-local.sh — Run the full stack locally via Docker Compose
# ──────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SCRIPT_DIR}"

echo "=== Starting Diageno (Docker Compose) ==="

# 1. Start infra
docker compose up -d postgres redis minio

echo "--- Waiting for Postgres ---"
until docker compose exec -T postgres pg_isready -U diageno; do
  sleep 2
done

# 2. Start MLflow
docker compose up -d mlflow

# 3. Run ETL pipeline
echo "--- Running ETL: download → parse → load → train ---"
docker compose run --rm etl python -m etl.download_bronze
docker compose run --rm etl python -m etl.parse_silver
docker compose run --rm etl python -m etl.load_gold
docker compose run --rm etl python -m training.train_all

# 4. Start API + UI
docker compose up -d api ui

# 5. Monitoring
docker compose up -d prometheus grafana alertmanager n8n

echo ""
echo "=== All services running ==="
echo ""
echo "  Streamlit UI   → http://localhost:8501"
echo "  FastAPI         → http://localhost:8000/docs"
echo "  Prometheus      → http://localhost:9090"
echo "  Grafana         → http://localhost:3000  (admin/admin)"
echo "  MLflow          → http://localhost:5000"
echo "  MinIO Console   → http://localhost:9001  (minioadmin/minioadmin)"
echo "  n8n             → http://localhost:5678  (admin/admin)"
echo ""

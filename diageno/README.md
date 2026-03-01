# Diageno — Rare Disease Diagnostic Recommendation Engine

**End-to-end rare disease diagnostic platform** built on
Streamlit + FastAPI + Postgres (pgvector) + Redis + Kubernetes.

## Architecture

| Service | Port | Purpose |
|---------|------|---------|
| Streamlit UI | 8501 | Case intake, differential, next-best-steps |
| FastAPI API | 8000 | `/recommend`, `/simulate_step`, `/validate_schema`, `/health` |
| Postgres + pgvector | 5432 | Cases, phenotype events, disease knowledge, embeddings |
| Redis | 6379 | HPO lookup cache, top-disease cache, /recommend response cache |
| MinIO | 9000 | Raw datasets, ETL outputs, trained artifacts |
| MLflow | 5000 | Experiment tracking, model registry |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards + alerting |
| n8n | 5678 | Pipeline orchestration |

## Quick Start (Docker Compose)

```bash
# One-liner — starts infra, runs ETL, launches app + monitoring
bash scripts/run-local.sh
```

Or step by step:

```bash
# 1. Build and start all services
docker compose up --build -d

# 2. Run ETL pipeline
docker compose run --rm etl python -m etl.download_bronze
docker compose run --rm etl python -m etl.parse_silver
docker compose run --rm etl python -m etl.load_gold

# 3. Train model artifacts
docker compose run --rm etl python -m training.train_all

# 4. Initialize MLflow experiment
python scripts/init_mlflow.py

# 5. Open UI
open http://localhost:8501
```

## Kubernetes (kind) Deployment

```bash
# 1. Create cluster + build/load images
bash scripts/setup-kind.sh

# 2. Deploy all services (infra → ETL → app → monitoring)
bash scripts/deploy-all.sh

# 3. Port-forward all services for local access
bash scripts/port-forward.sh
```

## n8n Pipeline Orchestration

Import `n8n/pipeline-workflow.json` into n8n (http://localhost:5678).
Trigger the full ETL → Train → Smoke Test pipeline via:

```bash
curl -X POST http://localhost:5678/webhook/run-pipeline
```

## Data Pipeline (Bronze → Silver → Gold)

| Stage | Description |
|-------|-------------|
| **Bronze** | Raw downloads: Zenodo vignettes, HPO obo, ORPHApackets, Orphadata, MONDO |
| **Silver** | Parsed + normalized Parquet files |
| **Gold** | Loaded into Postgres tables with indexes |

## "Trained" Model Artifacts

| Artifact | Description |
|----------|-------------|
| A: Disease Posterior Scorer | P(disease \| observed HPOs), calibrated |
| B: Next-Best-Phenotype Selector | Entropy-based HPO question ranking |
| C: Test Recommendation Policy | Rule + calibration for test ordering |
| D: Embedding Model | Sentence-transformer case/disease embeddings → pgvector |

## Project Structure

```
diageno/
├── config/           # Settings, logging
├── db/               # SQLAlchemy models, migrations
├── etl/              # Bronze → Silver → Gold pipeline
├── training/         # Model artifact builders
├── api/              # FastAPI inference service
├── ui/               # Streamlit application
├── k8s/              # Kubernetes manifests
├── monitoring/       # Prometheus + Grafana configs
├── n8n/              # n8n workflow templates
├── scripts/          # Deployment helper scripts
│   ├── setup-kind.sh
│   ├── deploy-all.sh
│   ├── port-forward.sh
│   ├── run-local.sh
│   └── init_mlflow.py
└── docker-compose.yml
```

## Service URLs (after deployment)

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit UI | http://localhost:8501 | — |
| FastAPI Docs | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |
| MLflow | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| n8n | http://localhost:5678 | admin / admin |

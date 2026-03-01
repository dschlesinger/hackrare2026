# Diageno — Rare Disease Diagnostic Recommendation Engine

**End-to-end rare disease diagnostic platform** built on
Streamlit + FastAPI + Postgres (pgvector) + Redis + Kubernetes.

> **Package manager:** [uv](https://docs.astral.sh/uv/) (fast Python package manager by Astral)

---

## Quick Run (pre-built data included)

If the data artifacts already exist (the `diageno/data/` directory is populated), you can start the app immediately:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start both servers
cd hackrare2026
uv sync
uv run uvicorn diageno.api.main:app --host 0.0.0.0 --port 8099 --workers 1 &
uv run streamlit run diageno/ui/app.py --server.port 8501
```

Then open:
- **UI:** http://localhost:8501
- **API docs:** http://localhost:8099/docs
- **Health check:** http://localhost:8099/health

---

## Full Build + Run (from scratch)

### Step 1: Environment Setup

```bash
cd hackrare2026

# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv + install all dependencies (reads pyproject.toml)
uv sync

# Or with dev tools (pytest, ruff, mypy, black):
uv sync --extra dev
```

### Step 2: Download Raw Data (Bronze)

```bash
# Downloads HPO ontology, phenopacket-store, Orphadata XMLs, MONDO
uv run python -m diageno.etl.download_bronze
```

### Step 3: Parse into Parquet (Silver)

```bash
# Produces: hpo_terms, diseases, disease_hpo, disease_gene, cases, etc.
uv run python -m diageno.etl.parse_silver
```

### Step 4: (Optional) Load into Postgres (Gold)

```bash
# Only needed if running Postgres — the app works fine without it
uv run python -m diageno.etl.load_gold
```

### Step 5: Train Model Artifacts

```bash
# Builds: disease_hpo_matrix, calibration, hpo_ancestors, disease_genes, policy
uv run python -m diageno.training.train_all --skip-embeddings --no-mlflow
```

### Step 6: Start the App

```bash
# Start FastAPI backend (port 8099)
uv run uvicorn diageno.api.main:app --host 0.0.0.0 --port 8099 --workers 1 &

# Start Streamlit frontend (port 8501)
uv run streamlit run diageno/ui/app.py --server.port 8501
```

---

## Docker Compose

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

# 4. Open UI
open http://localhost:8501
```

## Kubernetes (kind) Deployment

```bash
bash scripts/setup-kind.sh      # Create cluster + build/load images
bash scripts/deploy-all.sh      # Deploy all services
bash scripts/port-forward.sh    # Port-forward for local access
```

---

## Architecture

| Service | Port | Purpose |
|---------|------|---------|
| Streamlit UI | 8501 | Case intake, differential, next-best-steps, evaluation |
| FastAPI API | 8099 | `/recommend`, `/evaluate`, `/health`, `/hpo/search` |
| Postgres + pgvector | 5432 | Cases, phenotype events, disease knowledge (optional) |
| Redis | 6379 | HPO lookup cache, response cache (optional) |

## Data Pipeline (Bronze → Silver → Gold)

| Stage | Description |
|-------|-------------|
| **Bronze** | Raw downloads: Zenodo vignettes, HPO obo, ORPHApackets, Orphadata, MONDO |
| **Silver** | Parsed + normalized Parquet files |
| **Gold** | Loaded into Postgres tables with indexes |

## Model Artifacts

| Artifact | Description |
|----------|-------------|
| Disease Posterior Scorer | IC-weighted cosine similarity with HPO expansion, calibrated |
| Next-Best-Phenotype Selector | Entropy-based HPO question ranking |
| Test Recommendation Policy | Rule + calibration for test ordering (`policy.yaml`) |
| HPO Ancestors | Ontology ancestor graph for phenotype expansion |
| Disease-Gene Links | Gene ↔ disease associations for genomic integration |

## UI Pages

| Page | Description |
|------|-------------|
| 1. Home | Overview and system status |
| 2. Recommend | Enter phenotypes/genes → disease differential + uncertainty + VOI + evidence |
| 3. Simulate Step | Step-by-step diagnostic simulation |
| 4. Validation | Run against 5 ground-truth validation cases |
| 5. Research Evaluation | 5 experiments: replay, missingness, calibration, ablation, clinician rubric |
| 6. Clinician Demo | Guided 3-scenario walkthrough |

## Project Structure

```
hackrare2026/
├── pyproject.toml          # Root project config (uv reads this)
├── uv.lock                 # Locked dependencies
├── ValidationCase1–5       # Ground truth validation cases
├── diageno/
│   ├── config/             # Settings, logging
│   ├── core/               # Patient state, uncertainty, VOI, genomic advisor, equity, evidence
│   ├── db/                 # SQLAlchemy models, migrations
│   ├── etl/                # Bronze → Silver → Gold pipeline
│   ├── evaluation/         # Research metrics, replay, experiments
│   ├── training/           # Model artifact builders
│   ├── api/                # FastAPI inference service
│   │   ├── routes/         # recommend, evaluate, hpo, simulate
│   │   └── services/       # inference engine, HPO index
│   ├── ui/                 # Streamlit application
│   │   └── pages/          # 6 UI pages
│   ├── data/
│   │   ├── bronze/         # Raw downloaded files
│   │   ├── silver/         # Parsed parquet files
│   │   ├── gold/           # Postgres-loaded (optional)
│   │   └── model_artifacts/# Trained artifacts (matrix, calibration, etc.)
│   ├── k8s/                # Kubernetes manifests
│   ├── monitoring/         # Prometheus + Grafana configs
│   ├── n8n/                # n8n workflow templates
│   ├── scripts/            # Deployment helper scripts
│   └── docker-compose.yml
```

## Useful Commands

```bash
# Check health
curl http://localhost:8099/health

# Run research evaluation (all 5 experiments)
curl -X POST http://localhost:8099/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"experiments":["replay","missingness","calibration","ablation","rubric"]}'

# Run tests
uv run pytest

# Lint
uv run ruff check diageno/

# Type check
uv run mypy diageno/
```

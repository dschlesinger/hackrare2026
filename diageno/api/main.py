"""FastAPI main application — Diageno Inference API.

Endpoints:
  POST /recommend          — full diagnostic recommendation
  POST /simulate_step      — simulate adding/removing a phenotype
  POST /validate_schema    — validate JSON against case schema
  POST /hpo_lookup         — text → HPO candidates
  GET  /health             — health check
"""

from __future__ import annotations

import logging
import logging.config
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from diageno import __version__
from diageno.config import settings

# ── Logging setup ─────────────────────────────────────
logging_config_path = Path(__file__).resolve().parent.parent / "config" / "logging.yaml"
if logging_config_path.exists():
    with open(logging_config_path) as f:
        log_cfg = yaml.safe_load(f)
    logging.config.dictConfig(log_cfg)
logger = logging.getLogger("diageno.api")

# ── Prometheus metrics ────────────────────────────────
REQUEST_COUNT = Counter(
    "diageno_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "diageno_api_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)


# ── Lifespan (startup / shutdown) ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts and HPO index on startup."""
    from diageno.api.services.inference import engine
    from diageno.api.services.hpo_index import get_hpo_index

    try:
        engine.load()
        logger.info("Inference engine loaded successfully")
    except Exception as e:
        logger.error("Failed to load inference engine: %s", e)
        logger.info("API will start in degraded mode (no model)")

    try:
        hpo_idx = get_hpo_index()
        logger.info("HPO index loaded: %d terms", len(hpo_idx.terms))
    except Exception as e:
        logger.error("Failed to load HPO index: %s", e)

    yield

    logger.info("Shutting down Diageno API")


# ── App creation ──────────────────────────────────────
app = FastAPI(
    title="Diageno Inference API",
    description="Rare disease diagnostic recommendation engine",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: metrics collection ────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=endpoint,
    ).observe(duration)

    return response


# ── Prometheus metrics endpoint ───────────────────────
@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


# ── Root endpoint ─────────────────────────────────────
@app.get("/")
def root():
    """API root — landing info."""
    return {
        "service": "Diageno Inference API",
        "version": __version__,
        "status": "active",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /recommend",
            "POST /simulate_step",
            "POST /validate_schema",
            "POST /hpo_lookup",
            "POST /evaluate",
            "GET  /evaluate/latest",
            "GET  /health",
            "GET  /metrics",
        ],
    }


# ── Register routers ─────────────────────────────────
from diageno.api.routes.health import router as health_router
from diageno.api.routes.recommend import router as recommend_router
from diageno.api.routes.simulate import router as simulate_router
from diageno.api.routes.validate import router as validate_router
from diageno.api.routes.hpo_lookup import router as hpo_lookup_router
from diageno.api.routes.evaluate import router as evaluate_router

app.include_router(health_router, tags=["health"])
app.include_router(recommend_router, tags=["inference"])
app.include_router(simulate_router, tags=["inference"])
app.include_router(validate_router, tags=["validation"])
app.include_router(hpo_lookup_router, tags=["lookup"])
app.include_router(evaluate_router, tags=["evaluation"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "diageno.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
    )

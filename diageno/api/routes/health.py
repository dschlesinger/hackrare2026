"""/health endpoint."""

from fastapi import APIRouter

from diageno.api.schemas import HealthResponse
from diageno.api.services.inference import engine
from diageno import __version__

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint.

    Status logic:
      - "ok"       → model loaded (core inference ready)
      - "degraded" → model not loaded (can't score)

    DB and Redis are optional services; their absence does not
    affect core status since inference works without them.
    """
    # Check DB (optional)
    db_ok = False
    try:
        from diageno.db.session import get_db
        from sqlalchemy import text
        db = next(get_db())
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    # Check Redis (optional)
    cache_ok = False
    try:
        from diageno.api.services.cache import is_connected as redis_connected
        cache_ok = redis_connected()
    except Exception:
        pass

    return HealthResponse(
        status="ok" if engine.is_loaded else "degraded",
        version=__version__,
        model_loaded=engine.is_loaded,
        db_connected=db_ok,
        cache_connected=cache_ok,
    )

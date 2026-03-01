"""HPO lookup route — text → HPO candidates.

Uses in-memory HPO index with fuzzy matching as primary search.
No PostgreSQL dependency required.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from diageno.api.schemas import HPOLookupRequest, HPOLookupResponse
from diageno.api.services.hpo_index import get_hpo_index

logger = logging.getLogger("diageno.api.routes.hpo_lookup")
router = APIRouter()


@router.post("/hpo_lookup", response_model=HPOLookupResponse)
def hpo_lookup(req: HPOLookupRequest) -> HPOLookupResponse:
    """Lookup HPO terms by free text using in-memory index.

    Supports exact, prefix, substring, token, and fuzzy matching.
    Does NOT require PostgreSQL.
    """
    # Cache check (Redis optional — gracefully skip)
    try:
        from diageno.api.services.cache import get_hpo_lookup, cache_hpo_lookup

        cached = get_hpo_lookup(req.text)
        if cached:
            return HPOLookupResponse(results=cached)
    except Exception:
        pass

    # Search using in-memory index
    index = get_hpo_index()
    results = index.search(req.text, max_results=req.max_results)

    # Cache result (optional)
    try:
        from diageno.api.services.cache import cache_hpo_lookup

        cache_hpo_lookup(req.text, results)
    except Exception:
        pass

    return HPOLookupResponse(results=results)


@router.get("/hpo_validate/{hpo_id}")
def hpo_validate(hpo_id: str) -> dict:
    """Validate an HPO ID and return its name."""
    index = get_hpo_index()
    if index.is_valid(hpo_id):
        return {
            "valid": True,
            "hpo_id": hpo_id,
            "name": index.get_name(hpo_id),
        }
    return {"valid": False, "hpo_id": hpo_id, "name": ""}

"""Redis cache service for Diageno.

Cache keys:
  hpo:lookup:<text>                → HPO candidates
  disease:top:<hash(observed)>     → top disease list
  nextsteps:<hash(full_input)>     → full /recommend response
  evidence:<disease_id>:<hpo_id>   → explanation snippets
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

import redis

from diageno.config import settings

logger = logging.getLogger("diageno.api.cache")

_pool: Optional[redis.ConnectionPool] = None
_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Get or create a Redis client (connection-pooled)."""
    global _pool, _client
    if _client is None:
        _pool = redis.ConnectionPool.from_url(settings.redis_url, decode_responses=True)
        _client = redis.Redis(connection_pool=_pool)
    return _client


def is_connected() -> bool:
    """Check Redis connectivity."""
    try:
        return get_redis().ping()
    except Exception:
        return False


def _hash_key(*parts: str) -> str:
    """Create a short hash for cache key construction."""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─── HPO Lookup Cache ────────────────────────────────


def cache_hpo_lookup(text: str, results: list[dict]) -> None:
    key = f"hpo:lookup:{_hash_key(text.lower().strip())}"
    get_redis().setex(key, settings.cache_ttl_hpo_lookup, json.dumps(results))


def get_hpo_lookup(text: str) -> list[dict] | None:
    key = f"hpo:lookup:{_hash_key(text.lower().strip())}"
    val = get_redis().get(key)
    return json.loads(val) if val else None


# ─── Disease Top Cache ───────────────────────────────


def cache_disease_top(observed_hpos: list[str], results: list[dict]) -> None:
    hpo_str = ",".join(sorted(observed_hpos))
    key = f"disease:top:{_hash_key(hpo_str)}"
    get_redis().setex(key, settings.cache_ttl_disease_top, json.dumps(results))


def get_disease_top(observed_hpos: list[str]) -> list[dict] | None:
    hpo_str = ",".join(sorted(observed_hpos))
    key = f"disease:top:{_hash_key(hpo_str)}"
    val = get_redis().get(key)
    return json.loads(val) if val else None


# ─── Full Recommend Response Cache ───────────────────


def cache_recommend(input_hash: str, response: dict) -> None:
    key = f"nextsteps:{input_hash}"
    get_redis().setex(key, settings.cache_ttl_recommend, json.dumps(response))


def get_recommend(input_hash: str) -> dict | None:
    key = f"nextsteps:{input_hash}"
    val = get_redis().get(key)
    return json.loads(val) if val else None


# ─── Evidence Cache ──────────────────────────────────


def cache_evidence(disease_id: str, hpo_id: str, explanation: str) -> None:
    key = f"evidence:{disease_id}:{hpo_id}"
    get_redis().setex(key, settings.cache_ttl_evidence, explanation)


def get_evidence(disease_id: str, hpo_id: str) -> str | None:
    key = f"evidence:{disease_id}:{hpo_id}"
    return get_redis().get(key)


# ─── Generic helpers ─────────────────────────────────


def hash_case_input(case_input: dict) -> str:
    """Deterministic hash of a full case input (for caching)."""
    canonical = json.dumps(case_input, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def invalidate_pattern(pattern: str) -> int:
    """Delete all keys matching a pattern."""
    r = get_redis()
    keys = list(r.scan_iter(match=pattern, count=1000))
    if keys:
        return r.delete(*keys)
    return 0

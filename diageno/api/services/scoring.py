"""HPO lookup scoring service — search HPO terms by text."""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("diageno.api.services.scoring")


def search_hpo_terms(
    query: str,
    db: Session,
    max_results: int = 10,
) -> list[dict[str, str]]:
    """Search HPO terms by name or synonym (case-insensitive ILIKE).

    Returns list of {hpo_id, name, match_type}.
    """
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    pattern = f"%{query}%"

    # Search by term name
    rows = db.execute(
        text(
            "SELECT hpo_id, name FROM hpo_term "
            "WHERE LOWER(name) LIKE LOWER(:pat) AND is_obsolete = 0 "
            "ORDER BY LENGTH(name) LIMIT :lim"
        ),
        {"pat": pattern, "lim": max_results},
    ).fetchall()

    for row in rows:
        if row[0] not in seen:
            results.append({"hpo_id": row[0], "name": row[1], "match_type": "name"})
            seen.add(row[0])

    # Search by synonym if needed
    if len(results) < max_results:
        remaining = max_results - len(results)
        rows = db.execute(
            text(
                "SELECT s.hpo_id, t.name, s.synonym FROM hpo_synonym s "
                "JOIN hpo_term t ON s.hpo_id = t.hpo_id "
                "WHERE LOWER(s.synonym) LIKE LOWER(:pat) AND t.is_obsolete = 0 "
                "ORDER BY LENGTH(s.synonym) LIMIT :lim"
            ),
            {"pat": pattern, "lim": remaining},
        ).fetchall()

        for row in rows:
            if row[0] not in seen:
                results.append({
                    "hpo_id": row[0],
                    "name": row[1],
                    "match_type": f"synonym: {row[2]}",
                })
                seen.add(row[0])

    return results

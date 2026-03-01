"""In-memory HPO term index with fuzzy, token-based, and prefix search.

This replaces the PostgreSQL-dependent HPO search with an in-memory index
built from silver parquet files. It provides:
  - Exact prefix/substring matching
  - Token-based multi-word matching (AND/OR)
  - Fuzzy matching via edit distance
  - HPO ID direct lookup
  - Synonym search
  - Relevance ranking (exact > prefix > token > synonym > fuzzy)

Loads once at startup and is thread-safe for reads.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd

from diageno.config import settings

logger = logging.getLogger("diageno.api.services.hpo_index")

# ── Singleton index ──────────────────────────────────
_index: Optional["HPOIndex"] = None


def get_hpo_index() -> "HPOIndex":
    """Get or build the singleton HPO index."""
    global _index
    if _index is None:
        _index = HPOIndex()
        _index.build()
    return _index


class HPOIndex:
    """In-memory searchable index of HPO terms and synonyms."""

    def __init__(self) -> None:
        # hpo_id → canonical name
        self.terms: dict[str, str] = {}
        # hpo_id → definition
        self.definitions: dict[str, str] = {}
        # hpo_id → list of synonyms
        self.synonyms: dict[str, list[str]] = defaultdict(list)
        # lowered token → set of hpo_ids (inverted index)
        self.token_index: dict[str, set[str]] = defaultdict(set)
        # lowered full name → hpo_id (for exact match)
        self.name_to_id: dict[str, str] = {}
        # All searchable texts: list of (hpo_id, text, source_type, original_text)
        self.search_entries: list[tuple[str, str, str, str]] = []
        self._built = False

    def build(self, silver_dir: Path | None = None) -> None:
        """Build the index from parquet files."""
        silver = silver_dir or settings.silver_dir

        terms_path = silver / "hpo_terms.parquet"
        syns_path = silver / "hpo_synonyms.parquet"

        if not terms_path.exists():
            logger.error("hpo_terms.parquet not found at %s", terms_path)
            return

        # Load terms
        terms_df = pd.read_parquet(terms_path)
        obsolete_ids: set[str] = set()

        for _, row in terms_df.iterrows():
            hpo_id = row["hpo_id"]
            name = row["name"]
            is_obsolete = row.get("is_obsolete", 0)

            if is_obsolete:
                obsolete_ids.add(hpo_id)
                continue

            self.terms[hpo_id] = name
            self.definitions[hpo_id] = row.get("definition") or ""
            name_lower = name.lower()
            self.name_to_id[name_lower] = hpo_id

            # Add to search entries
            self.search_entries.append((hpo_id, name_lower, "name", name))

            # Tokenize and index
            tokens = _tokenize(name_lower)
            for token in tokens:
                self.token_index[token].add(hpo_id)

        # Load synonyms
        if syns_path.exists():
            syns_df = pd.read_parquet(syns_path)
            for _, row in syns_df.iterrows():
                hpo_id = row["hpo_id"]
                synonym = row["synonym"]

                if hpo_id in obsolete_ids or hpo_id not in self.terms:
                    continue

                self.synonyms[hpo_id].append(synonym)
                syn_lower = synonym.lower()
                self.search_entries.append((hpo_id, syn_lower, "synonym", synonym))

                # Index synonym tokens too
                tokens = _tokenize(syn_lower)
                for token in tokens:
                    self.token_index[token].add(hpo_id)

        self._built = True
        logger.info(
            "HPO index built: %d terms, %d synonym entries, %d tokens",
            len(self.terms),
            sum(len(v) for v in self.synonyms.values()),
            len(self.token_index),
        )

    def search(
        self,
        query: str,
        max_results: int = 10,
        fuzzy_threshold: float = 0.6,
    ) -> list[dict[str, str]]:
        """Search for HPO terms matching the query.

        Tries multiple strategies in priority order:
        1. Direct HPO ID lookup (if query looks like HP:NNNNNNN)
        2. Exact name match
        3. Prefix match on names
        4. Substring match on names and synonyms
        5. Token-based multi-word match (all tokens must match)
        6. Fuzzy match via sequence similarity

        Returns list of {hpo_id, name, match_type, score}.
        """
        if not self._built:
            self.build()

        query = query.strip()
        if not query:
            return []

        results: list[dict[str, str]] = []
        seen: set[str] = set()

        # Strategy 1: Direct HPO ID lookup
        if re.match(r"^HP:\d{7}$", query, re.IGNORECASE):
            hpo_id = query.upper()
            if hpo_id in self.terms:
                results.append({
                    "hpo_id": hpo_id,
                    "name": self.terms[hpo_id],
                    "match_type": "exact_id",
                })
                seen.add(hpo_id)
                return results

        query_lower = query.lower()
        query_tokens = _tokenize(query_lower)

        # Strategy 2: Exact name match
        if query_lower in self.name_to_id:
            hpo_id = self.name_to_id[query_lower]
            if hpo_id not in seen:
                results.append({
                    "hpo_id": hpo_id,
                    "name": self.terms[hpo_id],
                    "match_type": "exact_name",
                })
                seen.add(hpo_id)

        if len(results) >= max_results:
            return results[:max_results]

        # Strategy 3: Prefix match (name starts with query)
        prefix_matches: list[tuple[str, str, int]] = []
        for hpo_id, text_lower, source, original in self.search_entries:
            if hpo_id in seen:
                continue
            if text_lower.startswith(query_lower):
                prefix_matches.append((hpo_id, source, len(text_lower)))

        # Sort prefix matches by length (shorter = more specific)
        prefix_matches.sort(key=lambda x: x[2])
        for hpo_id, source, _ in prefix_matches[:max_results]:
            if hpo_id not in seen:
                match_type = f"prefix_{source}"
                results.append({
                    "hpo_id": hpo_id,
                    "name": self.terms[hpo_id],
                    "match_type": match_type,
                })
                seen.add(hpo_id)
                if len(results) >= max_results:
                    return results[:max_results]

        # Strategy 4: Substring match
        substring_matches: list[tuple[str, str, int]] = []
        for hpo_id, text_lower, source, original in self.search_entries:
            if hpo_id in seen:
                continue
            if query_lower in text_lower:
                substring_matches.append((hpo_id, source, len(text_lower)))

        substring_matches.sort(key=lambda x: x[2])
        for hpo_id, source, _ in substring_matches[:max_results]:
            if hpo_id not in seen:
                match_type = f"substring_{source}" if source == "synonym" else "name"
                results.append({
                    "hpo_id": hpo_id,
                    "name": self.terms[hpo_id],
                    "match_type": match_type,
                })
                seen.add(hpo_id)
                if len(results) >= max_results:
                    return results[:max_results]

        # Strategy 5: Token match (all query tokens found)
        if query_tokens and len(results) < max_results:
            # Find HPO IDs that contain ALL query tokens
            candidate_sets = [
                self.token_index.get(token, set()) for token in query_tokens
            ]
            if candidate_sets:
                # Intersection: terms matching all tokens
                all_match = set.intersection(*candidate_sets) if candidate_sets else set()
                # Also try OR matching if AND gives nothing
                any_match = set.union(*candidate_sets) if candidate_sets else set()

                # Score: fraction of query tokens matched
                token_scored: list[tuple[str, float]] = []
                for hpo_id in all_match - seen:
                    token_scored.append((hpo_id, 1.0))

                # Partial matches (at least half the tokens)
                if len(query_tokens) >= 2:
                    for hpo_id in any_match - seen - all_match:
                        matched = sum(
                            1 for t in query_tokens
                            if hpo_id in self.token_index.get(t, set())
                        )
                        ratio = matched / len(query_tokens)
                        if ratio >= 0.5:
                            token_scored.append((hpo_id, ratio))

                token_scored.sort(key=lambda x: (-x[1], len(self.terms.get(x[0], ""))))

                for hpo_id, score in token_scored[:max_results]:
                    if hpo_id not in seen:
                        results.append({
                            "hpo_id": hpo_id,
                            "name": self.terms[hpo_id],
                            "match_type": "token_match",
                        })
                        seen.add(hpo_id)
                        if len(results) >= max_results:
                            return results[:max_results]

        # Strategy 6: Fuzzy matching (edit distance via SequenceMatcher)
        if len(results) < max_results and len(query_lower) >= 3:
            fuzzy_scored: list[tuple[str, str, float]] = []

            for hpo_id, text_lower, source, original in self.search_entries:
                if hpo_id in seen:
                    continue
                # Quick filter: at least one character overlap
                if not any(c in text_lower for c in query_lower[:3]):
                    continue

                ratio = SequenceMatcher(None, query_lower, text_lower).ratio()
                if ratio >= fuzzy_threshold:
                    fuzzy_scored.append((hpo_id, source, ratio))

            fuzzy_scored.sort(key=lambda x: -x[2])

            for hpo_id, source, ratio in fuzzy_scored[:max_results]:
                if hpo_id not in seen:
                    match_type = f"fuzzy_{source}" if source == "synonym" else "fuzzy"
                    results.append({
                        "hpo_id": hpo_id,
                        "name": self.terms[hpo_id],
                        "match_type": match_type,
                    })
                    seen.add(hpo_id)
                    if len(results) >= max_results:
                        break

        return results[:max_results]

    def get_name(self, hpo_id: str) -> str:
        """Get canonical name for an HPO ID."""
        return self.terms.get(hpo_id, "")

    def is_valid(self, hpo_id: str) -> bool:
        """Check if an HPO ID exists and is non-obsolete."""
        return hpo_id in self.terms

    def get_all_ids(self) -> set[str]:
        """Get all valid (non-obsolete) HPO IDs."""
        return set(self.terms.keys())


def _tokenize(text: str) -> list[str]:
    """Tokenize text into searchable tokens.

    Strips common stopwords and normalizes.
    """
    stopwords = {
        "of", "the", "a", "an", "in", "on", "at", "to", "for",
        "and", "or", "with", "by", "is", "are", "was", "were",
        "not", "no", "from", "as", "that", "this", "be",
    }
    # Split on non-alphanumeric
    raw = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in raw if t and len(t) >= 2 and t not in stopwords]

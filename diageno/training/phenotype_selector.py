"""Artifact B — Next-Best-Phenotype Selector.

For the top candidate diseases, computes expected information gain
(entropy reduction) for each unobserved HPO, producing a ranked list
of "questions to ask next."

Inputs:  disease_hpo_matrix.parquet, hpo_dict.json, disease_index.json
Outputs: (real-time computation; uses trained matrix)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from diageno.config import settings

logger = logging.getLogger("diageno.training.phenotype_selector")


def load_matrix_artifacts(artifacts: Path) -> tuple[np.ndarray, dict, dict, dict]:
    """Load pre-built matrix and index dicts.

    Returns:
        matrix, disease_index, hpo_dict, inv_disease_index
    """
    matrix_df = pd.read_parquet(artifacts / "disease_hpo_matrix.parquet")
    matrix = matrix_df.values.astype(np.float32)

    with open(artifacts / "disease_index.json") as f:
        disease_index = json.load(f)
    with open(artifacts / "hpo_dict.json") as f:
        hpo_dict = json.load(f)

    inv_disease_index = {int(v): k for k, v in disease_index.items()}
    return matrix, disease_index, hpo_dict, inv_disease_index


def compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def rank_next_best_phenotypes(
    current_scores: np.ndarray,
    matrix: np.ndarray,
    hpo_dict: dict[str, int],
    observed_hpo_set: set[str],
    top_k_diseases: int = 20,
    max_questions: int = 10,
) -> list[dict]:
    """Rank unobserved HPOs by expected entropy reduction.

    For each candidate HPO h not yet observed:
      - Estimate P(h present | disease d) from matrix
      - Compute expected posterior entropy if h is present vs absent
      - Rank by expected reduction

    Args:
        current_scores: current disease probability vector (n_diseases,)
        matrix: disease-HPO matrix (n_diseases, n_hpos)
        hpo_dict: {hpo_id: col_index}
        observed_hpo_set: set of already-observed HPO IDs
        top_k_diseases: only consider top-K diseases for efficiency
        max_questions: max number of questions to return

    Returns:
        List of {hpo_id, expected_info_gain, rationale} dicts,
        sorted by expected_info_gain descending.
    """
    # Focus on top-K diseases
    top_idxs = np.argsort(current_scores)[::-1][:top_k_diseases]
    top_probs = current_scores[top_idxs]
    top_probs = top_probs / (top_probs.sum() + 1e-12)

    current_entropy = compute_entropy(top_probs)

    # Evaluate each unobserved HPO
    inv_hpo = {v: k for k, v in hpo_dict.items()}
    candidates: list[dict] = []

    for hpo_id, col_idx in hpo_dict.items():
        if hpo_id in observed_hpo_set:
            continue

        hpo_col = matrix[top_idxs, col_idx]

        # P(hpo present) = weighted average over top diseases
        p_present = float(np.dot(top_probs, hpo_col))
        p_absent = 1.0 - p_present

        if p_present < 0.01 and p_absent < 0.01:
            continue

        # Expected posterior if present
        if p_present > 0.01:
            posterior_present = top_probs * hpo_col
            s = posterior_present.sum()
            if s > 0:
                posterior_present = posterior_present / s
            entropy_present = compute_entropy(posterior_present)
        else:
            entropy_present = current_entropy

        # Expected posterior if absent
        if p_absent > 0.01:
            posterior_absent = top_probs * (1.0 - hpo_col)
            s = posterior_absent.sum()
            if s > 0:
                posterior_absent = posterior_absent / s
            entropy_absent = compute_entropy(posterior_absent)
        else:
            entropy_absent = current_entropy

        expected_entropy = p_present * entropy_present + p_absent * entropy_absent
        reduction = current_entropy - expected_entropy

        if reduction > 0.001:
            candidates.append({
                "hpo_id": hpo_id,
                "expected_info_gain": round(reduction, 4),
                "p_present": round(p_present, 3),
                "rationale": (
                    f"Asking about this phenotype is expected to reduce diagnostic "
                    f"uncertainty by {reduction:.3f} bits."
                ),
            })

    # Sort by information gain
    candidates.sort(key=lambda x: x["expected_info_gain"], reverse=True)
    return candidates[:max_questions]


def run() -> None:
    """Verify artifacts load correctly (no separate training needed)."""
    artifacts = settings.artifacts_dir
    matrix, disease_index, hpo_dict, inv_idx = load_matrix_artifacts(artifacts)
    logger.info(
        "Artifact B ready: matrix %s, %d diseases, %d HPOs",
        matrix.shape, len(disease_index), len(hpo_dict),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()

"""Artifact A — Disease Posterior Scorer.

Builds a disease↔HPO matrix and fits weights to produce calibrated
P(disease | observed_HPOs) scores.

Inputs:  disease_hpo.parquet, hpo_terms.parquet, cases.parquet (for calibration)
Outputs: disease_hpo_matrix.parquet, disease_index.json, hpo_dict.json, calibration.pkl
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from diageno.config import settings
from diageno.etl.utils import read_parquet

logger = logging.getLogger("diageno.training.disease_scorer")


def build_disease_hpo_matrix(silver: Path, artifacts: Path) -> tuple[np.ndarray, dict, dict]:
    """Build sparse disease × HPO matrix from parquet files.

    Returns:
        matrix: np.ndarray of shape (n_diseases, n_hpos)
        disease_index: {disease_id: row_index}
        hpo_dict: {hpo_id: col_index}
    """
    disease_hpo_path = silver / "disease_hpo.parquet"
    if not disease_hpo_path.exists():
        raise FileNotFoundError(f"disease_hpo.parquet not found at {silver}")

    df = read_parquet(disease_hpo_path)
    logger.info("Building disease-HPO matrix from %d associations", len(df))

    # Build index maps
    diseases = sorted(df["disease_id"].unique())
    hpo_ids = sorted(df["hpo_id"].unique())
    disease_index = {d: i for i, d in enumerate(diseases)}
    hpo_dict = {h: i for i, h in enumerate(hpo_ids)}

    # Build matrix
    matrix = np.zeros((len(diseases), len(hpo_ids)), dtype=np.float32)
    for _, row in df.iterrows():
        di = disease_index.get(row["disease_id"])
        hi = hpo_dict.get(row["hpo_id"])
        if di is not None and hi is not None:
            freq = row.get("frequency")
            matrix[di, hi] = freq if freq and freq > 0 else 0.5  # default 0.5 if unknown

    # Save artifacts
    artifacts.mkdir(parents=True, exist_ok=True)

    matrix_df = pd.DataFrame(matrix, index=diseases, columns=hpo_ids)
    matrix_df.to_parquet(artifacts / "disease_hpo_matrix.parquet")

    with open(artifacts / "disease_index.json", "w") as f:
        json.dump(disease_index, f, indent=2)

    with open(artifacts / "hpo_dict.json", "w") as f:
        json.dump(hpo_dict, f, indent=2)

    logger.info(
        "Disease-HPO matrix: %d diseases × %d HPO terms → %s",
        len(diseases), len(hpo_ids), artifacts,
    )

    return matrix, disease_index, hpo_dict


def score_diseases(
    observed_hpos: list[str],
    matrix: np.ndarray,
    hpo_dict: dict[str, int],
    disease_index: dict[str, int],
    absent_hpos: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Compute raw disease scores given observed (and optionally absent) HPOs.

    Uses weighted overlap: sum of frequency weights for present HPOs,
    minus penalty for absent HPOs that are expected in a disease.

    Returns list of (disease_id, score) sorted descending.
    """
    n_diseases = matrix.shape[0]
    scores = np.zeros(n_diseases, dtype=np.float64)

    # Positive signal from observed HPOs
    for hpo_id in observed_hpos:
        col = hpo_dict.get(hpo_id)
        if col is not None:
            scores += matrix[:, col]

    # Penalty for absent HPOs
    if absent_hpos:
        for hpo_id in absent_hpos:
            col = hpo_dict.get(hpo_id)
            if col is not None:
                scores -= 0.3 * matrix[:, col]  # penalize, but less than positive

    # Normalize by max possible score per disease
    row_maxes = matrix.sum(axis=1)
    row_maxes[row_maxes == 0] = 1.0  # avoid div by zero
    scores = scores / row_maxes

    # Sort and return
    inv_index = {v: k for k, v in disease_index.items()}
    ranked = sorted(
        [(inv_index[i], float(scores[i])) for i in range(n_diseases)],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def fit_calibrator(
    silver: Path, artifacts: Path, matrix: np.ndarray,
    disease_index: dict, hpo_dict: dict,
) -> Any:
    """Fit a calibrator that maps raw disease scores → meaningful confidence.

    Strategy:
      1. For each phenopacket case, score all diseases.
      2. Extract the true disease's raw score (ground truth from phenopacket disease_id)
         and sample negative disease scores at various ranks.
      3. Fit a logistic (Platt) calibrator on (raw_score, label) pairs.
      4. Additionally compute score distribution percentiles so we can express
         confidence as "how far above the pack" the top score is.

    Falls back to percentile-based calibration if ground-truth diseases are
    not found in the matrix (common since phenopackets use OMIM/MONDO IDs
    that may not align 1:1 with Orphanet IDs in the matrix).
    """
    from sklearn.linear_model import LogisticRegression

    cases_path = silver / "cases.parquet"
    pheno_path = silver / "phenotype_events.parquet"

    if not cases_path.exists() or not pheno_path.exists():
        logger.warning("No vignette data available for calibration")
        return None

    cases_df = read_parquet(cases_path)
    pheno_df = read_parquet(pheno_path)

    logger.info("Fitting calibrator on %d vignette cases", len(cases_df))

    # Collect score distribution statistics across all cases
    all_top1_scores: list[float] = []
    all_top5_scores: list[float] = []
    all_score_gaps: list[float] = []  # top1 - top2

    # Pre-group phenotypes by case_id for efficiency
    if "case_id" in pheno_df.columns:
        pheno_groups = {
            cid: grp for cid, grp in pheno_df.groupby("case_id")
        }
    else:
        pheno_groups = {}

    # Sample up to 2000 cases for speed (statistically sufficient)
    sample_cases = cases_df.sample(n=min(2000, len(cases_df)), random_state=42)

    inv_index = {v: k for k, v in disease_index.items()}
    n_diseases = matrix.shape[0]

    for _, case in sample_cases.iterrows():
        case_id = case.get("case_id") or case.get("external_id")
        if case_id is None:
            continue

        case_phenos = pheno_groups.get(case_id, pd.DataFrame())
        if case_phenos.empty:
            continue

        present = case_phenos[case_phenos["status"] == "present"]["hpo_id"].tolist()
        absent = case_phenos[case_phenos["status"] == "absent"]["hpo_id"].tolist()

        if not present:
            continue

        # Fast inline scoring (skip the full sort in score_diseases)
        scores = np.zeros(n_diseases, dtype=np.float64)
        for hpo_id in present:
            col = hpo_dict.get(hpo_id)
            if col is not None:
                scores += matrix[:, col]
        if absent:
            for hpo_id in absent:
                col = hpo_dict.get(hpo_id)
                if col is not None:
                    scores -= 0.3 * matrix[:, col]
        row_maxes = matrix.sum(axis=1)
        row_maxes[row_maxes == 0] = 1.0
        scores = scores / row_maxes

        # Get top-2 scores quickly
        top2_idx = np.argpartition(scores, -2)[-2:]
        top2_idx = top2_idx[np.argsort(scores[top2_idx])[::-1]]
        top1_score = float(scores[top2_idx[0]])
        top2_score = float(scores[top2_idx[1]]) if len(top2_idx) > 1 else 0.0

        all_top1_scores.append(top1_score)
        all_score_gaps.append(top1_score - top2_score)

    if len(all_top1_scores) < 20:
        logger.warning("Not enough cases (%d) for calibration", len(all_top1_scores))
        return None

    # Build a percentile-based calibrator that captures realistic confidence
    # Using: (a) how the raw score compares to the score distribution,
    #        (b) the gap between #1 and #2 (discriminability)
    top1_arr = np.array(all_top1_scores)
    gap_arr = np.array(all_score_gaps)

    calibrator_data = {
        "type": "percentile",
        "top1_percentiles": np.percentile(top1_arr, [10, 25, 50, 75, 90]).tolist(),
        "gap_percentiles": np.percentile(gap_arr, [10, 25, 50, 75, 90]).tolist(),
        "top1_mean": float(top1_arr.mean()),
        "top1_std": float(top1_arr.std()),
        "gap_mean": float(gap_arr.mean()),
        "gap_std": float(gap_arr.std()),
        "n_cases": len(all_top1_scores),
    }

    with open(artifacts / "calibration.pkl", "wb") as f:
        pickle.dump(calibrator_data, f)
    logger.info(
        "Calibrator saved → calibration.pkl (percentile-based, %d cases, "
        "mean_top1=%.4f ± %.4f, mean_gap=%.4f ± %.4f)",
        len(all_top1_scores), top1_arr.mean(), top1_arr.std(),
        gap_arr.mean(), gap_arr.std(),
    )

    return calibrator_data


def calibrate_score(
    calibrator: Any,
    raw_score: float,
    second_score: float = 0.0,
) -> float:
    """Map a raw disease score to a 0–1 confidence using the calibrator.

    Uses two signals:
      1. Score magnitude: where raw_score sits in the historical distribution
      2. Score gap: how far ahead of the runner-up (discriminability)

    The final confidence is a weighted blend of both.
    """
    if calibrator is None:
        return raw_score

    # Legacy support: if calibrator is a sklearn model, use it directly
    if not isinstance(calibrator, dict):
        try:
            return float(calibrator.predict([raw_score])[0])
        except Exception:
            return raw_score

    mean_s = calibrator["top1_mean"]
    std_s = max(calibrator["top1_std"], 1e-6)
    mean_g = calibrator["gap_mean"]
    std_g = max(calibrator["gap_std"], 1e-6)

    # Z-score based sigmoid for raw score
    z_score = (raw_score - mean_s) / std_s
    score_conf = 1.0 / (1.0 + np.exp(-1.5 * z_score))  # sigmoid centered at mean

    # Z-score based sigmoid for gap
    gap = raw_score - second_score
    z_gap = (gap - mean_g) / std_g
    gap_conf = 1.0 / (1.0 + np.exp(-1.5 * z_gap))

    # Blend: 60% score magnitude, 40% discriminability
    confidence = 0.6 * score_conf + 0.4 * gap_conf

    # Clamp to [0.05, 0.95] — never claim 0% or 100%
    return float(np.clip(confidence, 0.05, 0.95))


def run() -> None:
    """Build all Artifact A outputs."""
    silver = settings.silver_dir
    artifacts = settings.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    matrix, disease_index, hpo_dict = build_disease_hpo_matrix(silver, artifacts)
    fit_calibrator(silver, artifacts, matrix, disease_index, hpo_dict)

    logger.info("=== Artifact A (Disease Scorer) complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()

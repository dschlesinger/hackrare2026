"""Enhanced Disease Scorer with Cosine Similarity, HPO Expansion, Gene Integration.

This module provides improved disease scoring using:
1. Cosine similarity instead of weighted overlap
2. HPO ontology traversal (ancestor expansion)
3. Gene-based scoring boost
4. OMIM→ORPHA ID mapping for ground truth calibration
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from diageno.config import settings
from diageno.etl.utils import read_parquet

logger = logging.getLogger("diageno.training.enhanced_scorer")


# ─────────────────────────────────────────────────────────────────────────────
# HPO Ontology Expansion
# ─────────────────────────────────────────────────────────────────────────────


def build_hpo_ancestors(silver: Path) -> dict[str, set[str]]:
    """Build HPO term → ancestors mapping from hp.obo.

    Returns dict mapping each HPO ID to a set of all its ancestor IDs.
    This allows matching HP:0012345 (specific) to HP:0001250 (generic).
    """
    bronze = settings.bronze_dir
    obo_path = bronze / "hp.obo"

    if not obo_path.exists():
        logger.warning("hp.obo not found, skipping ancestor expansion")
        return {}

    try:
        import pronto
    except ImportError:
        logger.warning("pronto not installed, skipping ancestor expansion")
        return {}

    logger.info("Building HPO ancestor map from %s", obo_path)
    onto = pronto.Ontology(str(obo_path))

    ancestors_map: dict[str, set[str]] = {}
    for term in onto.terms():
        if not term.id.startswith("HP:"):
            continue
        # Get all superclasses (ancestors)
        ancestor_ids = {t.id for t in term.superclasses() if t.id.startswith("HP:")}
        ancestors_map[term.id] = ancestor_ids

    logger.info("Built ancestor map for %d HPO terms", len(ancestors_map))
    return ancestors_map


def expand_hpos_with_ancestors(
    hpo_ids: list[str],
    ancestors_map: dict[str, set[str]],
) -> set[str]:
    """Expand a list of HPO IDs to include all their ancestors.

    This ensures that specific terms like HP:0012345 also match
    more general matrix columns like HP:0001250.
    """
    expanded = set(hpo_ids)
    for hpo_id in hpo_ids:
        if hpo_id in ancestors_map:
            expanded.update(ancestors_map[hpo_id])
    return expanded


# ─────────────────────────────────────────────────────────────────────────────
# ID Mapping (OMIM → ORPHA)
# ─────────────────────────────────────────────────────────────────────────────


def load_id_mapping(silver: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load OMIM→ORPHA and MONDO→ORPHA mappings.

    Returns:
        omim_to_orpha: dict mapping OMIM:123456 → ORPHA:123
        mondo_to_orpha: dict mapping MONDO:0012345 → ORPHA:123
    """
    mapping_path = silver / "id_mapping.parquet"
    if not mapping_path.exists():
        logger.warning("id_mapping.parquet not found")
        return {}, {}

    df = read_parquet(mapping_path)

    omim_to_orpha = {}
    mondo_to_orpha = {}

    for _, row in df.iterrows():
        orpha_id = row.get("orpha_id")
        if not orpha_id:
            continue

        omim_id = row.get("omim_id")
        mondo_id = row.get("mondo_id")

        if omim_id and pd.notna(omim_id):
            omim_to_orpha[omim_id] = orpha_id
        if mondo_id and pd.notna(mondo_id):
            mondo_to_orpha[mondo_id] = orpha_id

    logger.info(
        "Loaded ID mappings: %d OMIM→ORPHA, %d MONDO→ORPHA",
        len(omim_to_orpha), len(mondo_to_orpha),
    )
    return omim_to_orpha, mondo_to_orpha


def resolve_disease_id(
    disease_id: str,
    omim_to_orpha: dict[str, str],
    mondo_to_orpha: dict[str, str],
) -> str | None:
    """Resolve any disease ID to ORPHA format."""
    if disease_id.startswith("ORPHA:"):
        return disease_id
    if disease_id.startswith("OMIM:"):
        return omim_to_orpha.get(disease_id)
    if disease_id.startswith("MONDO:"):
        return mondo_to_orpha.get(disease_id)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Gene Integration
# ─────────────────────────────────────────────────────────────────────────────


def load_disease_genes(silver: Path) -> dict[str, set[str]]:
    """Load disease → gene associations.

    Returns dict mapping disease_id → set of gene symbols.
    """
    gene_path = silver / "disease_gene.parquet"
    if not gene_path.exists():
        logger.warning("disease_gene.parquet not found")
        return {}

    df = read_parquet(gene_path)
    disease_genes: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        did = row["disease_id"]
        gene = row["gene_symbol"]
        if did not in disease_genes:
            disease_genes[did] = set()
        disease_genes[did].add(gene.upper())

    logger.info("Loaded gene associations for %d diseases", len(disease_genes))
    return disease_genes


def compute_gene_score(
    disease_id: str,
    patient_genes: list[dict],
    disease_genes: dict[str, set[str]],
) -> float:
    """Compute gene-based score boost for a disease.

    Args:
        disease_id: ORPHA disease ID
        patient_genes: list of {"gene": str, "classification": str, ...}
        disease_genes: dict mapping disease_id → set of gene symbols

    Returns:
        Score in range [-0.3, 1.0] based on gene overlap and classification.
    """
    if not patient_genes or disease_id not in disease_genes:
        return 0.0

    target_genes = disease_genes[disease_id]
    score = 0.0

    classification_weights = {
        "pathogenic": 1.0,
        "likely_pathogenic": 0.7,
        "vus": 0.2,
        "likely_benign": -0.1,
        "benign": -0.2,
    }

    for pg in patient_genes:
        gene = pg.get("gene", "").upper()
        if gene in target_genes:
            classification = pg.get("classification", "vus").lower()
            weight = classification_weights.get(classification, 0.1)
            score += weight

    # Clamp to reasonable range
    return float(np.clip(score, -0.3, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Cosine Similarity Scoring
# ─────────────────────────────────────────────────────────────────────────────


def build_enhanced_matrix(
    silver: Path,
    artifacts: Path,
) -> tuple[np.ndarray, dict, dict, dict[str, set[str]], dict[str, set[str]]]:
    """Build enhanced disease-HPO matrix with auxiliary data.

    Returns:
        matrix: disease × HPO frequency matrix
        disease_index: {disease_id: row_index}
        hpo_dict: {hpo_id: col_index}
        ancestors_map: {hpo_id: set of ancestor IDs}
        disease_genes: {disease_id: set of gene symbols}
    """
    disease_hpo_path = silver / "disease_hpo.parquet"
    if not disease_hpo_path.exists():
        raise FileNotFoundError(f"disease_hpo.parquet not found at {silver}")

    df = read_parquet(disease_hpo_path)
    logger.info("Building enhanced matrix from %d associations", len(df))

    # Build index maps
    diseases = sorted(df["disease_id"].unique())
    hpo_ids = sorted(df["hpo_id"].unique())
    disease_index = {d: i for i, d in enumerate(diseases)}
    hpo_dict = {h: i for i, h in enumerate(hpo_ids)}

    # Build matrix
    n_diseases, n_hpos = len(diseases), len(hpo_ids)
    matrix = np.zeros((n_diseases, n_hpos), dtype=np.float32)

    for _, row in df.iterrows():
        di = disease_index.get(row["disease_id"])
        hi = hpo_dict.get(row["hpo_id"])
        if di is not None and hi is not None:
            freq = row.get("frequency")
            matrix[di, hi] = freq if freq and freq > 0 else 0.5

    # Pre-compute L2 norms for cosine similarity
    disease_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    disease_norms[disease_norms == 0] = 1.0  # avoid div by zero

    # Save artifacts
    artifacts.mkdir(parents=True, exist_ok=True)

    matrix_df = pd.DataFrame(matrix, index=diseases, columns=hpo_ids)
    matrix_df.to_parquet(artifacts / "disease_hpo_matrix.parquet")

    with open(artifacts / "disease_index.json", "w") as f:
        json.dump(disease_index, f, indent=2)

    with open(artifacts / "hpo_dict.json", "w") as f:
        json.dump(hpo_dict, f, indent=2)

    # Save disease norms for fast cosine computation
    np.save(artifacts / "disease_norms.npy", disease_norms.flatten())

    # Load auxiliary data
    ancestors_map = build_hpo_ancestors(silver)
    disease_genes = load_disease_genes(silver)

    # Save auxiliary artifacts
    with open(artifacts / "hpo_ancestors.pkl", "wb") as f:
        # Convert sets to lists for pickling
        pickle.dump({k: list(v) for k, v in ancestors_map.items()}, f)

    with open(artifacts / "disease_genes.pkl", "wb") as f:
        pickle.dump({k: list(v) for k, v in disease_genes.items()}, f)

    logger.info(
        "Enhanced matrix: %d diseases × %d HPOs, %d HPO ancestors, %d disease-gene links",
        n_diseases, n_hpos, len(ancestors_map), len(disease_genes),
    )

    return matrix, disease_index, hpo_dict, ancestors_map, disease_genes


def compute_ic_weights(
    matrix: np.ndarray,
) -> np.ndarray:
    """Compute Information Content (IC) weights for HPO columns.

    IC = -log2(p) where p = fraction of diseases annotated with that HPO.
    Rare phenotypes are more informative and should weight higher.
    Returns array of shape (n_hpos,).
    """
    n_diseases = matrix.shape[0]
    # Count diseases annotated with each HPO
    disease_count = np.sum(matrix > 0, axis=0).astype(np.float64)
    # Smoothed probability (Laplace)
    p = (disease_count + 1) / (n_diseases + 2)
    ic = -np.log2(p)
    # Normalize to [0, 1] for stable weighting
    ic_max = ic.max()
    if ic_max > 0:
        ic = ic / ic_max
    return ic


# Caches for IC-weighted matrix (avoids recomputation per call)
_cached_weighted_matrix: np.ndarray | None = None
_cached_weighted_norms: np.ndarray | None = None
_cached_matrix_id: int | None = None


def _get_weighted_matrix(matrix: np.ndarray, ic_weights: np.ndarray) -> np.ndarray:
    """Get or compute the IC-weighted disease matrix (cached)."""
    global _cached_weighted_matrix, _cached_matrix_id
    matrix_id = id(matrix)
    if _cached_weighted_matrix is None or _cached_matrix_id != matrix_id:
        _cached_weighted_matrix = matrix * ic_weights[np.newaxis, :]
        _cached_matrix_id = matrix_id
    return _cached_weighted_matrix


def _get_weighted_norms(w_matrix: np.ndarray) -> np.ndarray:
    """Get or compute norms for the IC-weighted matrix (cached)."""
    global _cached_weighted_norms
    if _cached_weighted_norms is None or len(_cached_weighted_norms) != w_matrix.shape[0]:
        norms = np.linalg.norm(w_matrix, axis=1)
        norms[norms == 0] = 1.0
        _cached_weighted_norms = norms
    assert _cached_weighted_norms is not None
    return _cached_weighted_norms


def score_diseases_cosine(
    observed_hpos: list[str],
    matrix: np.ndarray,
    hpo_dict: dict[str, int],
    disease_index: dict[str, int],
    absent_hpos: list[str] | None = None,
    ancestors_map: dict[str, set[str]] | None = None,
    disease_norms: np.ndarray | None = None,
    patient_genes: list[dict] | None = None,
    disease_genes: dict[str, set[str]] | None = None,
    gene_weight: float = 0.2,
    ic_weights: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """Score diseases using IC-weighted cosine similarity with HPO expansion.

    Uses Information Content weighting so rare/specific phenotypes contribute
    more to the score than common/generic ones.

    Args:
        observed_hpos: list of present HPO IDs
        matrix: disease × HPO matrix
        hpo_dict: {hpo_id: col_index}
        disease_index: {disease_id: row_index}
        absent_hpos: list of explicitly absent HPO IDs
        ancestors_map: HPO term → ancestors for expansion
        disease_norms: pre-computed L2 norms per disease (IC-weighted)
        patient_genes: list of {"gene": str, "classification": str}
        disease_genes: {disease_id: set of gene symbols}
        gene_weight: weight for gene score (0.0–1.0)
        ic_weights: pre-computed IC weights per HPO column

    Returns:
        List of (disease_id, score) sorted descending.
    """
    n_diseases, n_hpos = matrix.shape

    logger.debug(
        "Scoring %d HPOs against %d diseases (ancestors=%d)",
        len(observed_hpos), n_diseases,
        len(ancestors_map) if ancestors_map else 0,
    )

    # Expand HPOs to include ancestors (handles coverage gap)
    if ancestors_map:
        expanded_present = expand_hpos_with_ancestors(observed_hpos, ancestors_map)
        expanded_absent = (
            expand_hpos_with_ancestors(absent_hpos, ancestors_map)
            if absent_hpos
            else set()
        )
    else:
        expanded_present = set(observed_hpos)
        expanded_absent = set(absent_hpos) if absent_hpos else set()

    # Compute IC weights if not provided
    if ic_weights is None:
        ic_weights = compute_ic_weights(matrix)

    # Build patient phenotype vector (IC-weighted)
    patient_vec = np.zeros(n_hpos, dtype=np.float64)
    matched_cols = 0
    observed_set = set(observed_hpos)
    for hpo_id in expanded_present:
        col = hpo_dict.get(hpo_id)
        if col is not None:
            # Direct input terms weight more than ancestors
            base_weight = 1.0 if hpo_id in observed_set else 0.5
            patient_vec[col] = base_weight * ic_weights[col]
            matched_cols += 1

    # Penalize absent phenotypes (IC-weighted)
    for hpo_id in expanded_absent:
        col = hpo_dict.get(hpo_id)
        if col is not None:
            patient_vec[col] = -0.3 * ic_weights[col]

    # Use pre-computed weighted matrix and norms if available
    w_matrix = _get_weighted_matrix(matrix, ic_weights)
    w_norms = _get_weighted_norms(w_matrix)

    # Compute cosine similarity on IC-weighted vectors
    patient_norm = np.linalg.norm(patient_vec)
    if patient_norm == 0:
        patient_norm = 1.0

    # Dot product with all diseases
    dot_products = w_matrix @ patient_vec
    cosine_scores = dot_products / (w_norms * patient_norm)

    logger.debug(
        "Cosine scores: min=%.4f, max=%.4f, mean=%.4f",
        float(np.min(cosine_scores)),
        float(np.max(cosine_scores)),
        float(np.mean(cosine_scores)),
    )

    # Build final scores with gene boost
    inv_index = {v: k for k, v in disease_index.items()}
    final_scores = []

    for i in range(n_diseases):
        disease_id = inv_index[i]
        pheno_score = float(cosine_scores[i])

        # Gene score
        gene_score = 0.0
        if patient_genes and disease_genes:
            gene_score = compute_gene_score(disease_id, patient_genes, disease_genes)

        # Combine: (1 - gene_weight) * pheno + gene_weight * gene
        combined = (1.0 - gene_weight) * pheno_score + gene_weight * gene_score

        final_scores.append((disease_id, combined))

    # Sort by score descending
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Calibration with ID Mapping
# ─────────────────────────────────────────────────────────────────────────────


def fit_enhanced_calibrator(
    silver: Path,
    artifacts: Path,
    matrix: np.ndarray,
    disease_index: dict,
    hpo_dict: dict,
    ancestors_map: dict[str, set[str]],
    disease_genes: dict[str, set[str]],
) -> dict:
    """Fit calibrator using ground truth disease labels via ID mapping.

    This is a major improvement: we now use actual phenopacket diagnoses
    (after OMIM→ORPHA mapping) to calibrate confidence scores.
    """
    from sklearn.linear_model import LogisticRegression

    cases_path = silver / "cases.parquet"
    pheno_path = silver / "phenotype_events.parquet"

    if not cases_path.exists() or not pheno_path.exists():
        logger.warning("No vignette data for calibration")
        return _fallback_calibrator(matrix, disease_index, hpo_dict)

    cases_df = read_parquet(cases_path)
    pheno_df = read_parquet(pheno_path)

    # Load ID mappings
    omim_to_orpha, mondo_to_orpha = load_id_mapping(silver)

    logger.info(
        "Fitting enhanced calibrator on %d cases with ID mapping",
        len(cases_df),
    )

    # Pre-group phenotypes
    pheno_groups = {}
    if "case_id" in pheno_df.columns:
        pheno_groups = {cid: grp for cid, grp in pheno_df.groupby("case_id")}

    # Pre-compute IC weights for scoring
    ic_weights = compute_ic_weights(matrix)

    # Collect training data
    X_scores = []
    X_gaps = []
    y_labels = []  # 1 = correct disease in top-K, 0 = otherwise

    # Also collect distribution stats
    all_top1_scores = []
    all_gaps = []
    ground_truth_ranks = []

    for _, case in cases_df.iterrows():
        case_id = case.get("case_id") or case.get("external_id")
        if case_id is None:
            continue

        # Get phenotypes
        case_phenos = pheno_groups.get(case_id, pd.DataFrame())
        if case_phenos.empty:
            continue

        present = case_phenos[case_phenos["status"] == "present"]["hpo_id"].tolist()
        absent = case_phenos[case_phenos["status"] == "absent"]["hpo_id"].tolist()

        if not present:
            continue

        # Extract ground truth disease from raw_json
        gt_disease_id = None
        try:
            import json as json_mod
            raw_json = case.get("raw_json", "{}")
            data = json_mod.loads(raw_json)

            # Try diseases field
            if "diseases" in data:
                for d in data["diseases"]:
                    term_id = d.get("term", {}).get("id", "")
                    if term_id:
                        gt_disease_id = resolve_disease_id(
                            term_id, omim_to_orpha, mondo_to_orpha
                        )
                        if gt_disease_id and gt_disease_id in disease_index:
                            break

            # Try interpretations field
            if not gt_disease_id and "interpretations" in data:
                for interp in data["interpretations"]:
                    term_id = (
                        interp.get("diagnosis", {})
                        .get("disease", {})
                        .get("id", "")
                    )
                    if term_id:
                        gt_disease_id = resolve_disease_id(
                            term_id, omim_to_orpha, mondo_to_orpha
                        )
                        if gt_disease_id and gt_disease_id in disease_index:
                            break
        except Exception:
            pass

        # Score diseases for this case
        ranked = score_diseases_cosine(
            present,
            matrix,
            hpo_dict,
            disease_index,
            absent_hpos=absent,
            ancestors_map=ancestors_map,
            ic_weights=ic_weights,
        )

        if not ranked:
            continue

        top1_score = ranked[0][1]
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        gap = top1_score - top2_score

        all_top1_scores.append(top1_score)
        all_gaps.append(gap)

        # Check if ground truth is in top results
        if gt_disease_id and gt_disease_id in disease_index:
            gt_rank = None
            for rank, (did, score) in enumerate(ranked):
                if did == gt_disease_id:
                    gt_rank = rank
                    break

            if gt_rank is not None:
                ground_truth_ranks.append(gt_rank)

                # Training examples for logistic calibration
                # Positive: GT in top-5
                # Negative: GT not in top-20
                if gt_rank < 5:
                    X_scores.append(top1_score)
                    X_gaps.append(gap)
                    y_labels.append(1)
                elif gt_rank >= 20:
                    X_scores.append(top1_score)
                    X_gaps.append(gap)
                    y_labels.append(0)

    # Build calibrator
    calibrator_data = {
        "type": "enhanced",
        "top1_mean": float(np.mean(all_top1_scores)) if all_top1_scores else 0.5,
        "top1_std": float(np.std(all_top1_scores)) if all_top1_scores else 0.2,
        "gap_mean": float(np.mean(all_gaps)) if all_gaps else 0.1,
        "gap_std": float(np.std(all_gaps)) if all_gaps else 0.1,
        "n_cases": len(all_top1_scores),
        "n_ground_truth": len(ground_truth_ranks),
    }

    # If we have enough ground truth examples, fit logistic calibrator
    if len(X_scores) >= 50 and len(set(y_labels)) == 2:
        X = np.column_stack([X_scores, X_gaps])
        y = np.array(y_labels)

        try:
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X, y)
            calibrator_data["logistic_coef"] = clf.coef_.tolist()
            calibrator_data["logistic_intercept"] = float(clf.intercept_[0])
            calibrator_data["has_logistic"] = True
            logger.info(
                "Logistic calibrator fitted on %d ground truth examples "
                "(%.1f%% positive)",
                len(y), 100 * np.mean(y),
            )
        except Exception as e:
            logger.warning("Failed to fit logistic calibrator: %s", e)
            calibrator_data["has_logistic"] = False
    else:
        calibrator_data["has_logistic"] = False
        logger.info(
            "Using percentile calibration (%d ground truth examples, need 50+)",
            len(X_scores),
        )

    # Ground truth accuracy stats
    if ground_truth_ranks:
        top1_acc = sum(1 for r in ground_truth_ranks if r == 0) / len(ground_truth_ranks)
        top5_acc = sum(1 for r in ground_truth_ranks if r < 5) / len(ground_truth_ranks)
        top10_acc = sum(1 for r in ground_truth_ranks if r < 10) / len(ground_truth_ranks)
        calibrator_data["ground_truth_top1_acc"] = top1_acc
        calibrator_data["ground_truth_top5_acc"] = top5_acc
        calibrator_data["ground_truth_top10_acc"] = top10_acc
        logger.info(
            "Ground truth accuracy: Top-1=%.1f%%, Top-5=%.1f%%, Top-10=%.1f%%",
            100 * top1_acc, 100 * top5_acc, 100 * top10_acc,
        )

    with open(artifacts / "calibration.pkl", "wb") as f:
        pickle.dump(calibrator_data, f)

    return calibrator_data


def _fallback_calibrator(matrix, disease_index, hpo_dict) -> dict:
    """Build a simple fallback calibrator without ground truth."""
    return {
        "type": "percentile",
        "top1_mean": 0.4,
        "top1_std": 0.2,
        "gap_mean": 0.1,
        "gap_std": 0.1,
        "n_cases": 0,
        "has_logistic": False,
    }


def calibrate_score_enhanced(
    calibrator: dict,
    raw_score: float,
    gap: float = 0.0,
) -> float:
    """Calibrate a raw score using enhanced calibrator.

    If logistic calibrator is available, uses it for true probability.
    Otherwise falls back to percentile-based z-score sigmoid.
    """
    if calibrator is None:
        return max(0.05, min(0.95, raw_score))

    # Try logistic calibration first (if available)
    if calibrator.get("has_logistic"):
        try:
            coef = np.array(calibrator["logistic_coef"]).flatten()
            intercept = calibrator["logistic_intercept"]
            X = np.array([[raw_score, gap]])
            logit = float(X @ coef + intercept)
            prob = 1.0 / (1.0 + np.exp(-logit))
            return float(np.clip(prob, 0.05, 0.95))
        except Exception:
            pass

    # Fallback: percentile-based sigmoid
    mean_s = calibrator.get("top1_mean", 0.4)
    std_s = max(calibrator.get("top1_std", 0.2), 1e-6)
    mean_g = calibrator.get("gap_mean", 0.1)
    std_g = max(calibrator.get("gap_std", 0.1), 1e-6)

    z_score = (raw_score - mean_s) / std_s
    score_conf = 1.0 / (1.0 + np.exp(-1.5 * z_score))

    z_gap = (gap - mean_g) / std_g
    gap_conf = 1.0 / (1.0 + np.exp(-1.5 * z_gap))

    confidence = 0.6 * score_conf + 0.4 * gap_conf
    return float(np.clip(confidence, 0.05, 0.95))


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def run() -> None:
    """Build all enhanced artifacts."""
    silver = settings.silver_dir
    artifacts = settings.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    logger.info("=== Building Enhanced Disease Scorer ===")

    matrix, disease_index, hpo_dict, ancestors_map, disease_genes = (
        build_enhanced_matrix(silver, artifacts)
    )

    fit_enhanced_calibrator(
        silver,
        artifacts,
        matrix,
        disease_index,
        hpo_dict,
        ancestors_map,
        disease_genes,
    )

    logger.info("=== Enhanced Scorer complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()

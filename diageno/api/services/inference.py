"""Enhanced Inference service — IC-weighted cosine similarity, HPO expansion, gene integration."""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from diageno.config import settings
from diageno.training.enhanced_scorer import (
    score_diseases_cosine,
    calibrate_score_enhanced,
    expand_hpos_with_ancestors,
    compute_gene_score,
    compute_ic_weights,
)
from diageno.training.phenotype_selector import (
    load_matrix_artifacts,
    rank_next_best_phenotypes,
)
from diageno.training.test_policy import load_policy, match_rules

logger = logging.getLogger("diageno.api.services.inference")

_HPO_RE = re.compile(r"^HP:\d{7}$")


class InferenceEngine:
    """Enhanced inference engine with cosine similarity and gene integration."""

    def __init__(self) -> None:
        self._loaded = False
        self.matrix: Optional[np.ndarray] = None
        self.disease_index: dict[str, int] = {}
        self.hpo_dict: dict[str, int] = {}
        self.inv_disease_index: dict[int, str] = {}
        self.calibrator: Any = None
        self.policy: dict = {}
        self.disease_names: dict[str, str] = {}
        self.hpo_names: dict[str, str] = {}
        self.version: str = "unknown"
        # Enhanced artifacts
        self.hpo_ancestors: dict[str, set[str]] = {}
        self.disease_genes: dict[str, set[str]] = {}
        self.disease_norms: Optional[np.ndarray] = None
        self.ic_weights: Optional[np.ndarray] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, artifacts_dir: Path | None = None) -> None:
        """Load all artifacts from disk including enhanced ones."""
        artifacts = artifacts_dir or settings.artifacts_dir
        logger.info("Loading enhanced model artifacts from %s …", artifacts)

        # Matrix + indexes
        self.matrix, self.disease_index, self.hpo_dict, self.inv_disease_index = (
            load_matrix_artifacts(artifacts)
        )

        # Disease norms for cosine similarity
        norms_path = artifacts / "disease_norms.npy"
        if norms_path.exists():
            self.disease_norms = np.load(norms_path)
            logger.info("Disease norms loaded")
        else:
            # Compute on the fly
            self.disease_norms = np.linalg.norm(self.matrix, axis=1)
            self.disease_norms[self.disease_norms == 0] = 1.0

        # Pre-compute IC weights
        self.ic_weights = compute_ic_weights(self.matrix)
        logger.info("IC weights computed (%d HPO columns)", len(self.ic_weights))

        # HPO ancestors for expansion
        ancestors_path = artifacts / "hpo_ancestors.pkl"
        if ancestors_path.exists():
            with open(ancestors_path, "rb") as f:
                ancestors_data = pickle.load(f)
                self.hpo_ancestors = {k: set(v) for k, v in ancestors_data.items()}
            logger.info("HPO ancestors loaded (%d terms)", len(self.hpo_ancestors))

        # Disease-gene associations
        genes_path = artifacts / "disease_genes.pkl"
        if genes_path.exists():
            with open(genes_path, "rb") as f:
                genes_data = pickle.load(f)
                self.disease_genes = {k: set(v) for k, v in genes_data.items()}
            logger.info("Disease-gene links loaded (%d diseases)", len(self.disease_genes))

        # Calibrator (optional)
        cal_path = artifacts / "calibration.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                self.calibrator = pickle.load(f)
            cal_type = self.calibrator.get("type", "unknown") if isinstance(self.calibrator, dict) else "legacy"
            logger.info("Calibrator loaded (type=%s)", cal_type)

        # Policy
        policy_path = artifacts / "policy.yaml"
        if policy_path.exists():
            self.policy = load_policy(artifacts)
            logger.info("Policy loaded (%d rules)", len(self.policy.get("rules", [])))

        # Disease names (from silver)
        silver = settings.silver_dir
        try:
            import pandas as pd

            diseases_path = silver / "diseases.parquet"
            if diseases_path.exists():
                df = pd.read_parquet(diseases_path)
                self.disease_names = dict(zip(df["disease_id"], df["name"]))

            hpo_path = silver / "hpo_terms.parquet"
            if hpo_path.exists():
                df = pd.read_parquet(hpo_path)
                self.hpo_names = dict(zip(df["hpo_id"], df["name"]))
        except Exception as e:
            logger.warning("Could not load name lookups: %s", e)

        # Version info
        version_info = {
            "n_diseases": len(self.disease_index),
            "n_hpos": len(self.hpo_dict),
            "has_calibrator": self.calibrator is not None,
            "has_hpo_ancestors": len(self.hpo_ancestors) > 0,
            "has_gene_data": len(self.disease_genes) > 0,
            "scoring_method": "cosine_similarity",
        }
        self.version = json.dumps(version_info)

        self._loaded = True
        logger.info(
            "Engine ready: %d diseases, %d HPOs, %d ancestors, %d gene links",
            len(self.disease_index),
            len(self.hpo_dict),
            len(self.hpo_ancestors),
            len(self.disease_genes),
        )

    @staticmethod
    def _validate_hpo_ids(hpo_ids: list[str]) -> list[str]:
        """Filter to well-formed HPO IDs (HP:NNNNNNN)."""
        return [h for h in hpo_ids if _HPO_RE.match(h)]

    def recommend(
        self,
        present_hpos: list[str],
        absent_hpos: list[str] | None = None,
        prior_testing: str = "none",
        test_result: str | None = None,
        inheritance_hint: str | None = None,
        vus_present: bool = False,
        gene_results: list[dict] | None = None,
        top_k: int = 20,
    ) -> dict:
        """Run full recommendation pipeline with enhanced scoring.

        Uses cosine similarity, HPO ancestor expansion, and gene integration.
        Returns dict with diseases, next_best_phenotypes, test_recommendations.
        """
        if not self._loaded:
            self.load()

        # Validate input HPO IDs
        present_hpos = self._validate_hpo_ids(present_hpos)
        absent_hpos = self._validate_hpo_ids(absent_hpos or [])

        if not present_hpos:
            logger.warning("No valid HPO IDs provided")
            return {
                "diseases": [],
                "next_best_phenotypes": [],
                "test_recommendations": [],
                "confidence": 0.0,
                "scoring_method": "cosine_similarity",
                "hpo_expansion": {"input_hpos": 0, "expanded_hpos": 0, "ancestors_added": 0},
                "gene_integration": {"patient_genes": len(gene_results) if gene_results else 0, "diseases_with_gene_data": len(self.disease_genes)},
            }

        logger.debug("recommend: %d present HPOs, %d absent", len(present_hpos), len(absent_hpos))

        # 1. Score diseases using IC-weighted cosine similarity
        ranked = score_diseases_cosine(
            present_hpos,
            self.matrix,
            self.hpo_dict,
            self.disease_index,
            absent_hpos=absent_hpos,
            ancestors_map=self.hpo_ancestors,
            patient_genes=gene_results,
            disease_genes=self.disease_genes,
            gene_weight=0.2 if gene_results else 0.0,
            ic_weights=self.ic_weights,
        )

        top_diseases = ranked[:top_k]

        # Calibrate scores using enhanced calibrator
        calibrated_top = []
        for i, (did, raw_score) in enumerate(top_diseases):
            next_score = ranked[i + 1][1] if i + 1 < len(ranked) else 0.0
            gap = raw_score - next_score
            cal_score = calibrate_score_enhanced(self.calibrator, raw_score, gap)
            calibrated_top.append((did, raw_score, cal_score))

        # Expand HPOs for supporting/contradicting analysis
        if self.hpo_ancestors:
            expanded_present = expand_hpos_with_ancestors(present_hpos, self.hpo_ancestors)
            expanded_absent = expand_hpos_with_ancestors(absent_hpos or [], self.hpo_ancestors)
        else:
            expanded_present = set(present_hpos)
            expanded_absent = set(absent_hpos or [])

        # Build disease candidates with supporting/contradicting info + rationale
        observed_set = expanded_present
        absent_set = expanded_absent
        disease_candidates = []

        for rank_pos, (did, raw, cal) in enumerate(calibrated_top):
            di = self.disease_index.get(did)
            supporting = []
            contradicting = []
            missing_key = []
            total_disease_hpos = 0

            if di is not None:
                for hpo_id, col_idx in self.hpo_dict.items():
                    weight = self.matrix[di, col_idx]
                    if weight > 0:
                        total_disease_hpos += 1
                        if hpo_id in observed_set:
                            supporting.append(hpo_id)
                        elif hpo_id in absent_set:
                            contradicting.append(hpo_id)
                        elif weight >= 0.3:
                            missing_key.append((hpo_id, weight))

            missing_key.sort(key=lambda x: x[1], reverse=True)

            # Build rationale including gene info
            match_pct = (len(supporting) / max(total_disease_hpos, 1)) * 100
            disease_name = self.disease_names.get(did, did)
            rationale_parts = [
                f"{len(supporting)} of {total_disease_hpos} known phenotypes for "
                f"{disease_name} match ({match_pct:.0f}% overlap via cosine similarity)."
            ]

            if contradicting:
                contra_names = [self.hpo_names.get(h, h) for h in contradicting[:3]]
                rationale_parts.append(
                    f"However, {len(contradicting)} phenotype(s) reported absent "
                    f"are typically expected ({', '.join(contra_names)})."
                )

            if missing_key:
                missing_names = [self.hpo_names.get(h, h) for h, w in missing_key[:3]]
                rationale_parts.append(
                    f"Key phenotypes not yet assessed: {', '.join(missing_names)}."
                )

            # Add gene context to rationale
            gene_score = 0.0
            if gene_results and did in self.disease_genes:
                gene_score = compute_gene_score(did, gene_results, self.disease_genes)
                disease_gene_set = self.disease_genes.get(did, set())
                patient_gene_names = [g.get("gene", "").upper() for g in gene_results]
                matching_genes = [g for g in patient_gene_names if g in disease_gene_set]
                if matching_genes:
                    rationale_parts.append(
                        f"Gene(s) {', '.join(matching_genes)} associated with this disease "
                        f"(gene score: {gene_score:+.2f})."
                    )

            disease_candidates.append({
                "disease_id": did,
                "name": disease_name,
                "score": round(raw, 4),
                "calibrated_score": round(cal, 4) if cal is not None else None,
                "supporting_hpos": supporting[:10],
                "contradicting_hpos": contradicting[:10],
                "rationale": " ".join(rationale_parts),
                "phenotype_match": f"{len(supporting)}/{total_disease_hpos}",
                "gene_score": round(gene_score, 3) if gene_results else None,
            })

        # 2. Top disease confidence
        confidence = calibrated_top[0][2] if calibrated_top else 0.0

        # 3. Next-best-phenotypes
        scores_array = np.zeros(len(self.disease_index), dtype=np.float64)
        for did, raw, _ in calibrated_top:
            idx = self.disease_index.get(did)
            if idx is not None:
                scores_array[idx] = raw

        next_phenos = rank_next_best_phenotypes(
            scores_array, self.matrix, self.hpo_dict, observed_set | absent_set,
            top_k_diseases=min(20, len(calibrated_top)),
        )

        # Add labels
        for p in next_phenos:
            p["label"] = self.hpo_names.get(p["hpo_id"], "")

        # 4. Test recommendations
        test_recs = match_rules(
            self.policy,
            prior_testing=prior_testing,
            test_result=test_result,
            inheritance_hint=inheritance_hint,
            confidence=confidence,
            vus_present=vus_present,
        )

        # 5. Enhance test rationale with gene context
        if gene_results:
            gene_summary = ", ".join(
                f"{g['gene']} ({g.get('classification', 'unknown')})"
                for g in gene_results[:5]
            )
            for rec in test_recs:
                rec["rationale"] = (
                    rec.get("rationale", "") +
                    f" (Patient has prior gene findings: {gene_summary})"
                )

        # 6. Report HPO expansion stats
        n_expanded = len(expanded_present) - len(present_hpos)

        return {
            "diseases": disease_candidates,
            "next_best_phenotypes": next_phenos,
            "test_recommendations": test_recs,
            "confidence": round(float(confidence), 4),
            "scoring_method": "cosine_similarity",
            "hpo_expansion": {
                "input_hpos": len(present_hpos),
                "expanded_hpos": len(expanded_present),
                "ancestors_added": n_expanded,
            },
            "gene_integration": {
                "patient_genes": len(gene_results) if gene_results else 0,
                "diseases_with_gene_data": len(self.disease_genes),
            },
        }


# Module-level singleton
engine = InferenceEngine()

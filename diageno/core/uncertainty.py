"""Uncertainty Decomposition Module.

Returns three uncertainty signals:
  1. Phenotype uncertainty — how well do observed phenotypes discriminate among candidates
  2. Genomic uncertainty — how much unresolved genetic information remains
  3. Decision uncertainty — confidence in the recommended next action

Also provides "What would change?" counterfactual panel:
  Top 3 missing signals that would reorder top diagnoses/actions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("diageno.core.uncertainty")


@dataclass
class UncertaintyDecomposition:
    """Three-axis uncertainty decomposition."""
    phenotype_uncertainty: float = 0.0   # 0 = fully resolved, 1 = maximally ambiguous
    genomic_uncertainty: float = 0.0     # 0 = fully resolved, 1 = no genetic data
    decision_uncertainty: float = 0.0    # 0 = clear next step, 1 = no clear action
    overall_uncertainty: float = 0.0     # weighted composite

    # Counterfactuals: what would change the recommendation
    counterfactuals: list[Counterfactual] = field(default_factory=list)

    # Calibration metadata
    entropy_bits: float = 0.0
    effective_candidates: float = 0.0    # 2^entropy — how many diseases are "in play"

    def to_dict(self) -> dict:
        return {
            "phenotype_uncertainty": round(self.phenotype_uncertainty, 4),
            "genomic_uncertainty": round(self.genomic_uncertainty, 4),
            "decision_uncertainty": round(self.decision_uncertainty, 4),
            "overall_uncertainty": round(self.overall_uncertainty, 4),
            "entropy_bits": round(self.entropy_bits, 3),
            "effective_candidates": round(self.effective_candidates, 1),
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
        }


@dataclass
class Counterfactual:
    """A missing signal that would change the recommendation."""
    signal_type: str              # "phenotype" | "genetic_test" | "family_history" | "lab_result"
    description: str              # human-readable
    expected_impact: str          # what would change
    hpo_id: str | None = None    # if phenotype
    gene: str | None = None      # if genetic
    impact_magnitude: float = 0.0  # 0-1, how much it would change things

    def to_dict(self) -> dict:
        d = {
            "signal_type": self.signal_type,
            "description": self.description,
            "expected_impact": self.expected_impact,
            "impact_magnitude": round(self.impact_magnitude, 3),
        }
        if self.hpo_id:
            d["hpo_id"] = self.hpo_id
        if self.gene:
            d["gene"] = self.gene
        return d


def compute_uncertainty(
    disease_scores: list[tuple[str, float]],
    n_present_hpos: int,
    n_absent_hpos: int,
    has_genetic_testing: bool,
    has_vus: bool,
    gene_results: list[dict] | None,
    confidence: float,
    phenotype_questions: list[dict] | None = None,
    disease_genes: dict[str, set[str]] | None = None,
    disease_names: dict[str, str] | None = None,
    hpo_names: dict[str, str] | None = None,
) -> UncertaintyDecomposition:
    """Compute three-axis uncertainty decomposition.

    Args:
        disease_scores: ranked list of (disease_id, score)
        n_present_hpos: number of present phenotypes
        n_absent_hpos: number of absent phenotypes
        has_genetic_testing: whether any genetic testing was done
        has_vus: whether VUS is present
        gene_results: list of gene finding dicts
        confidence: calibrated confidence score
        phenotype_questions: entropy-based next-best-phenotype list
        disease_genes: disease → gene set mapping
        disease_names: disease_id → name mapping
        hpo_names: hpo_id → name mapping
    """
    disease_names = disease_names or {}
    hpo_names = hpo_names or {}
    disease_genes = disease_genes or {}

    # ── 1. Phenotype Uncertainty ─────────────────────
    # Based on: entropy of disease distribution, number of phenotypes, and discriminability
    scores = np.array([s for _, s in disease_scores[:100]])
    if len(scores) == 0:
        return UncertaintyDecomposition(
            phenotype_uncertainty=1.0,
            genomic_uncertainty=1.0,
            decision_uncertainty=1.0,
            overall_uncertainty=1.0,
        )

    # Normalize scores to probability distribution
    scores_pos = np.maximum(scores, 0)
    total = scores_pos.sum()
    if total > 0:
        probs = scores_pos / total
    else:
        probs = np.ones_like(scores_pos) / len(scores_pos)

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = np.log2(len(probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    # Effective number of candidates
    effective_candidates = 2 ** entropy

    # Phenotype coverage factor: more phenotypes → less uncertainty
    phenotype_coverage = 1.0 - min(1.0, (n_present_hpos + n_absent_hpos) / 15.0)

    # Top-1 vs top-2 gap (discriminability)
    if len(scores) >= 2:
        gap_ratio = (scores[0] - scores[1]) / (scores[0] + 1e-8)
        discriminability = min(1.0, gap_ratio * 5)
    else:
        discriminability = 0.0

    phenotype_uncertainty = (
        0.4 * normalized_entropy +
        0.3 * phenotype_coverage +
        0.3 * (1.0 - discriminability)
    )
    phenotype_uncertainty = float(np.clip(phenotype_uncertainty, 0, 1))

    # ── 2. Genomic Uncertainty ───────────────────────
    # Based on: whether genetic testing done, VUS status, gene coverage
    if not has_genetic_testing:
        genomic_uncertainty = 0.9   # almost all genomic info missing
    elif has_vus:
        genomic_uncertainty = 0.6   # VUS = partially resolved
    elif gene_results and any(g.get("classification") in ("pathogenic", "likely_pathogenic") for g in (gene_results or [])):
        genomic_uncertainty = 0.2   # pathogenic finding = mostly resolved
    else:
        genomic_uncertainty = 0.5   # tested but nothing found

    # Reduce if genes match top disease
    if gene_results and disease_scores:
        top_disease = disease_scores[0][0]
        top_genes = disease_genes.get(top_disease, set())
        patient_gene_names = {g.get("gene", "").upper() for g in (gene_results or [])}
        if top_genes & patient_gene_names:
            genomic_uncertainty *= 0.5  # strong gene-disease match

    genomic_uncertainty = float(np.clip(genomic_uncertainty, 0, 1))

    # ── 3. Decision Uncertainty ──────────────────────
    # Based on: confidence, agreement between pheno and genomic signals
    decision_uncertainty = 1.0 - confidence
    # Increase if pheno and genomic signals disagree
    if phenotype_uncertainty > 0.5 and genomic_uncertainty < 0.3:
        decision_uncertainty = min(1.0, decision_uncertainty + 0.1)
    elif phenotype_uncertainty < 0.3 and genomic_uncertainty > 0.7:
        decision_uncertainty = min(1.0, decision_uncertainty + 0.1)

    decision_uncertainty = float(np.clip(decision_uncertainty, 0, 1))

    # ── Composite ────────────────────────────────────
    overall = 0.4 * phenotype_uncertainty + 0.3 * genomic_uncertainty + 0.3 * decision_uncertainty
    overall = float(np.clip(overall, 0, 1))

    # ── Counterfactuals ──────────────────────────────
    counterfactuals = _compute_counterfactuals(
        disease_scores=disease_scores,
        phenotype_questions=phenotype_questions,
        has_genetic_testing=has_genetic_testing,
        has_vus=has_vus,
        gene_results=gene_results,
        disease_genes=disease_genes,
        disease_names=disease_names,
        hpo_names=hpo_names,
        confidence=confidence,
    )

    return UncertaintyDecomposition(
        phenotype_uncertainty=phenotype_uncertainty,
        genomic_uncertainty=genomic_uncertainty,
        decision_uncertainty=decision_uncertainty,
        overall_uncertainty=overall,
        counterfactuals=counterfactuals,
        entropy_bits=float(entropy),
        effective_candidates=float(effective_candidates),
    )


def _compute_counterfactuals(
    disease_scores: list[tuple[str, float]],
    phenotype_questions: list[dict] | None,
    has_genetic_testing: bool,
    has_vus: bool,
    gene_results: list[dict] | None,
    disease_genes: dict[str, set[str]],
    disease_names: dict[str, str],
    hpo_names: dict[str, str],
    confidence: float,
    max_counterfactuals: int = 3,
) -> list[Counterfactual]:
    """Identify top N missing signals that would most change the recommendation."""
    candidates: list[Counterfactual] = []

    # 1. Top phenotype question (highest entropy reduction)
    if phenotype_questions:
        top_pq = phenotype_questions[0]
        hpo_id = top_pq["hpo_id"]
        label = hpo_names.get(hpo_id, top_pq.get("label", hpo_id))
        info_gain = top_pq.get("expected_info_gain", 0)

        if disease_scores:
            top_name = disease_names.get(disease_scores[0][0], disease_scores[0][0])
            second_name = disease_names.get(disease_scores[1][0], disease_scores[1][0]) if len(disease_scores) > 1 else "N/A"
        else:
            top_name = second_name = "N/A"

        candidates.append(Counterfactual(
            signal_type="phenotype",
            description=f"Confirm or deny presence of '{label}'",
            expected_impact=(
                f"If present: strengthens candidates where this phenotype is expected. "
                f"Could shift ranking between {top_name} and {second_name}. "
                f"Info gain: {info_gain:.3f} bits."
            ),
            hpo_id=hpo_id,
            impact_magnitude=min(1.0, info_gain * 5),
        ))

    # 2. Genetic testing if not done
    if not has_genetic_testing:
        candidates.append(Counterfactual(
            signal_type="genetic_test",
            description="Obtain genetic testing (panel or exome)",
            expected_impact=(
                "Genetic testing would resolve ~25-40% of rare disease cases. "
                "A positive finding would dramatically change confidence and next steps."
            ),
            impact_magnitude=0.8,
        ))
    elif has_vus:
        # Segregation for VUS
        vus_genes = [g.get("gene", "?") for g in (gene_results or []) if g.get("classification") == "vus"]
        candidates.append(Counterfactual(
            signal_type="genetic_test",
            description=f"Segregation analysis for VUS in {', '.join(vus_genes[:3])}",
            expected_impact=(
                "Segregation could reclassify VUS → likely pathogenic, which would "
                "substantially increase confidence in the leading diagnosis."
            ),
            gene=vus_genes[0] if vus_genes else None,
            impact_magnitude=0.6,
        ))

    # 3. Gene-based counterfactual: genes of top disease not yet tested
    if disease_scores and len(candidates) < max_counterfactuals:
        top_disease = disease_scores[0][0]
        top_genes = disease_genes.get(top_disease, set())
        tested_genes = {g.get("gene", "").upper() for g in (gene_results or [])}
        untested = top_genes - tested_genes
        if untested:
            gene_list = ", ".join(sorted(untested)[:3])
            top_name = disease_names.get(top_disease, top_disease)
            candidates.append(Counterfactual(
                signal_type="genetic_test",
                description=f"Test gene(s) {gene_list} (associated with {top_name})",
                expected_impact=(
                    f"Finding a pathogenic variant in these genes would confirm the "
                    f"top diagnosis. Negative result would lower {top_name} ranking."
                ),
                gene=sorted(untested)[0] if untested else None,
                impact_magnitude=0.7,
            ))

    # 4. Second-best phenotype question
    if phenotype_questions and len(phenotype_questions) > 1 and len(candidates) < max_counterfactuals:
        pq2 = phenotype_questions[1]
        hpo_id = pq2["hpo_id"]
        label = hpo_names.get(hpo_id, pq2.get("label", hpo_id))
        candidates.append(Counterfactual(
            signal_type="phenotype",
            description=f"Assess for '{label}'",
            expected_impact=f"Would reduce uncertainty by {pq2.get('expected_info_gain', 0):.3f} bits",
            hpo_id=hpo_id,
            impact_magnitude=min(1.0, pq2.get("expected_info_gain", 0) * 5),
        ))

    # Sort by impact and return top N
    candidates.sort(key=lambda c: c.impact_magnitude, reverse=True)
    return candidates[:max_counterfactuals]

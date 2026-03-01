"""Evidence-Grounded Explanations Module.

For each suggested action, attach:
  - Supporting evidence (why do this)
  - Contradictory evidence (why not)
  - Confidence level
  - Expected impact
  - "Why this over alternatives" explanation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("diageno.core.evidence")


@dataclass
class EvidenceItem:
    """A single piece of evidence."""
    statement: str
    source: str = "model"       # model | ontology | literature | clinical_guideline | gene_database
    strength: str = "moderate"   # strong | moderate | weak | uncertain
    direction: str = "supporting"  # supporting | contradicting | neutral

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "source": self.source,
            "strength": self.strength,
            "direction": self.direction,
        }


@dataclass
class ExplainedAction:
    """An action with full evidence grounding."""
    action: str
    action_type: str
    supporting_evidence: list[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    expected_impact: str = ""
    why_this_over_alternatives: str = ""
    alternatives_considered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "action_type": self.action_type,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "contradicting_evidence": [e.to_dict() for e in self.contradicting_evidence],
            "confidence": round(self.confidence, 3),
            "expected_impact": self.expected_impact,
            "why_this_over_alternatives": self.why_this_over_alternatives,
            "alternatives_considered": self.alternatives_considered,
        }


@dataclass
class ExplainedDisease:
    """A disease candidate with full evidence grounding."""
    disease_id: str
    name: str
    rank: int
    score: float
    supporting_evidence: list[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = field(default_factory=list)
    missing_evidence: list[EvidenceItem] = field(default_factory=list)
    phenotype_overlap: str = ""
    gene_evidence: str = ""
    why_ranked_here: str = ""

    def to_dict(self) -> dict:
        return {
            "disease_id": self.disease_id,
            "name": self.name,
            "rank": self.rank,
            "score": round(self.score, 4),
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "contradicting_evidence": [e.to_dict() for e in self.contradicting_evidence],
            "missing_evidence": [e.to_dict() for e in self.missing_evidence],
            "phenotype_overlap": self.phenotype_overlap,
            "gene_evidence": self.gene_evidence,
            "why_ranked_here": self.why_ranked_here,
        }


def build_disease_explanation(
    disease: dict,
    rank: int,
    matrix: Any,
    disease_index: dict[str, int],
    hpo_dict: dict[str, int],
    hpo_names: dict[str, str],
    disease_genes: dict[str, set[str]],
    patient_present_hpos: list[str],
    patient_absent_hpos: list[str],
    patient_genes: list[dict] | None = None,
    expanded_hpos: set[str] | None = None,
) -> ExplainedDisease:
    """Build a fully grounded explanation for a disease ranking.

    Examines:
    - Which patient phenotypes match this disease (supporting)
    - Which are absent but expected (contradicting)
    - Which key phenotypes are missing (could change ranking)
    - Gene evidence
    """
    import numpy as np

    disease_id = disease.get("disease_id", "")
    disease_name = disease.get("name", disease_id)
    score = disease.get("score", 0)

    supporting = []
    contradicting = []
    missing = []

    di = disease_index.get(disease_id)
    if di is not None and matrix is not None:
        expanded = expanded_hpos or set(patient_present_hpos)
        absent_set = set(patient_absent_hpos)

        # Categorize each disease-associated HPO
        n_total = 0
        n_matched = 0
        n_contra = 0
        n_missing_key = 0

        for hpo_id, col_idx in hpo_dict.items():
            weight = float(matrix[di, col_idx])
            if weight <= 0:
                continue
            n_total += 1
            label = hpo_names.get(hpo_id, hpo_id)

            if hpo_id in expanded:
                n_matched += 1
                freq_label = _weight_to_frequency(weight)
                supporting.append(EvidenceItem(
                    statement=f"{label} ({hpo_id}) is present — {freq_label} feature of this disease",
                    source="ontology",
                    strength="strong" if weight >= 0.8 else "moderate",
                    direction="supporting",
                ))
            elif hpo_id in absent_set:
                n_contra += 1
                contradicting.append(EvidenceItem(
                    statement=f"{label} ({hpo_id}) is absent but typically present in this disease",
                    source="ontology",
                    strength="moderate" if weight >= 0.5 else "weak",
                    direction="contradicting",
                ))
            elif weight >= 0.3:
                n_missing_key += 1
                missing.append(EvidenceItem(
                    statement=f"{label} ({hpo_id}) — key feature not yet assessed ({_weight_to_frequency(weight)})",
                    source="ontology",
                    strength="moderate",
                    direction="neutral",
                ))

        phenotype_overlap = f"{n_matched}/{n_total} phenotypes match"
    else:
        phenotype_overlap = "N/A"

    # Gene evidence
    gene_evidence = ""
    if patient_genes and disease_id in disease_genes:
        target_genes = disease_genes[disease_id]
        matched_genes = [
            g.get("gene", "").upper()
            for g in patient_genes
            if g.get("gene", "").upper() in target_genes
        ]
        if matched_genes:
            gene_evidence = f"Gene(s) {', '.join(matched_genes)} match known disease associations"
            supporting.append(EvidenceItem(
                statement=gene_evidence,
                source="gene_database",
                strength="strong",
                direction="supporting",
            ))
        else:
            patient_gene_names = [g.get("gene", "").upper() for g in patient_genes]
            gene_evidence = (
                f"Tested genes ({', '.join(patient_gene_names[:3])}) "
                f"not among known genes for this disease ({', '.join(sorted(target_genes)[:3])})"
            )

    # Build explanation
    why_parts = []
    if len(supporting) > 0:
        why_parts.append(f"{len(supporting)} matching phenotypes")
    if len(contradicting) > 0:
        why_parts.append(f"{len(contradicting)} contradicting features")
    if gene_evidence:
        why_parts.append(gene_evidence.lower())

    why_ranked = f"Ranked #{rank} based on: {'; '.join(why_parts)}." if why_parts else f"Ranked #{rank}"

    return ExplainedDisease(
        disease_id=disease_id,
        name=disease_name,
        rank=rank,
        score=score,
        supporting_evidence=supporting[:5],
        contradicting_evidence=contradicting[:5],
        missing_evidence=sorted(missing, key=lambda e: e.strength, reverse=True)[:3],
        phenotype_overlap=phenotype_overlap,
        gene_evidence=gene_evidence,
        why_ranked_here=why_ranked,
    )


def build_action_explanation(
    action: dict,
    disease_scores: list[tuple[str, float]],
    disease_names: dict[str, str],
    confidence: float,
    prior_testing: str,
) -> ExplainedAction:
    """Build evidence-grounded explanation for a recommended action."""
    action_name = action.get("action", "")
    action_type = action.get("action_type", "test")

    supporting = []
    contradicting = []
    alternatives = []

    # Evidence based on action type
    if action_type == "test":
        supporting.append(EvidenceItem(
            statement=action.get("rationale", ""),
            source="clinical_guideline",
            strength="moderate",
        ))
        if confidence < 0.3:
            supporting.append(EvidenceItem(
                statement=f"Low confidence ({confidence:.0%}) supports broader testing strategy",
                source="model",
                strength="moderate",
            ))
        if "exome" in action_name.lower():
            alternatives = ["Gene panel (lower cost)", "WGS (broader coverage)"]
        elif "panel" in action_name.lower():
            alternatives = ["Clinical exome (broader)", "Targeted single-gene test"]
        elif "wgs" in action_name.lower():
            alternatives = ["Exome (lower cost)", "RNA-seq (functional)"]

    elif action_type == "referral":
        supporting.append(EvidenceItem(
            statement=action.get("rationale", ""),
            source="expert_consensus",
            strength="moderate",
        ))

    elif action_type == "reanalysis":
        supporting.append(EvidenceItem(
            statement="~15% additional diagnoses from periodic reanalysis",
            source="literature",
            strength="strong",
        ))

    # Why this over alternatives
    why = action.get("why_this_over_alternatives", "")
    if not why:
        why = f"Recommended based on current testing level ({prior_testing}) and confidence ({confidence:.0%})"

    return ExplainedAction(
        action=action_name,
        action_type=action_type,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        confidence=confidence,
        expected_impact=action.get("expected_impact", ""),
        why_this_over_alternatives=why,
        alternatives_considered=alternatives,
    )


def _weight_to_frequency(weight: float) -> str:
    """Convert matrix weight to human-readable frequency label."""
    if weight >= 0.9:
        return "obligate"
    elif weight >= 0.75:
        return "very frequent"
    elif weight >= 0.4:
        return "frequent"
    elif weight >= 0.2:
        return "occasional"
    else:
        return "very rare"

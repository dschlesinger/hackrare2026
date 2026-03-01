"""Genomic-First Decision Module.

Explicit genomics action logic:
  - VUS triage and reclassification pathway
  - Segregation analysis recommendation
  - Reanalysis cadence (12-18 month intervals)
  - Escalation to RNA-seq / methylation / long-read WGS based on context
  - Clinically sequenced output: what to do NOW vs NEXT VISIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("diageno.core.genomic_advisor")


@dataclass
class GenomicAction:
    """A recommended genomic action with clinical context."""
    action: str
    action_type: str         # test | interpretation | counseling | monitoring
    rationale: str
    urgency: str = "routine"  # stat | urgent | routine | next_visit | periodic
    evidence_level: str = "expert_consensus"  # guideline | expert_consensus | research
    acmg_criteria: list[str] = field(default_factory=list)  # relevant ACMG criteria
    cost_estimate: str = ""
    turnaround: str = ""
    prerequisites: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "action_type": self.action_type,
            "rationale": self.rationale,
            "urgency": self.urgency,
            "evidence_level": self.evidence_level,
            "acmg_criteria": self.acmg_criteria,
            "cost_estimate": self.cost_estimate,
            "turnaround": self.turnaround,
            "prerequisites": self.prerequisites,
        }


@dataclass
class GenomicAssessment:
    """Full genomic assessment output."""
    vus_triage: list[GenomicAction] = field(default_factory=list)
    escalation_path: list[GenomicAction] = field(default_factory=list)
    reanalysis_plan: list[GenomicAction] = field(default_factory=list)
    counseling: list[GenomicAction] = field(default_factory=list)

    # Timeline
    now_actions: list[GenomicAction] = field(default_factory=list)
    next_visit_actions: list[GenomicAction] = field(default_factory=list)
    periodic_actions: list[GenomicAction] = field(default_factory=list)

    # Summary
    summary: str = ""
    genomic_maturity: str = "early"  # early | intermediate | advanced | exhaustive

    def to_dict(self) -> dict:
        return {
            "vus_triage": [a.to_dict() for a in self.vus_triage],
            "escalation_path": [a.to_dict() for a in self.escalation_path],
            "reanalysis_plan": [a.to_dict() for a in self.reanalysis_plan],
            "counseling": [a.to_dict() for a in self.counseling],
            "now_actions": [a.to_dict() for a in self.now_actions],
            "next_visit_actions": [a.to_dict() for a in self.next_visit_actions],
            "periodic_actions": [a.to_dict() for a in self.periodic_actions],
            "summary": self.summary,
            "genomic_maturity": self.genomic_maturity,
        }


def assess_genomics(
    gene_results: list[dict],
    prior_testing: str = "none",
    top_diseases: list[tuple[str, float]] | None = None,
    disease_genes: dict[str, set[str]] | None = None,
    disease_names: dict[str, str] | None = None,
    inheritance_hint: str | None = None,
    confidence: float = 0.0,
    family_history: dict | None = None,
) -> GenomicAssessment:
    """Run full genomic assessment and produce clinically sequenced recommendations.

    Args:
        gene_results: list of gene finding dicts
        prior_testing: none | panel | exome | wgs
        top_diseases: ranked disease candidates
        disease_genes: disease → gene set mapping
        disease_names: disease_id → name mapping
        inheritance_hint: inheritance pattern
        confidence: diagnostic confidence
        family_history: family history dict
    """
    assessment = GenomicAssessment()
    disease_genes = disease_genes or {}
    disease_names = disease_names or {}
    top_diseases = top_diseases or []

    # Determine genomic maturity level
    if prior_testing == "none":
        assessment.genomic_maturity = "early"
    elif prior_testing == "panel":
        assessment.genomic_maturity = "intermediate"
    elif prior_testing == "exome":
        assessment.genomic_maturity = "advanced"
    elif prior_testing == "wgs":
        assessment.genomic_maturity = "exhaustive"

    # ── VUS Triage ───────────────────────────────────
    vus_results = [g for g in gene_results if g.get("classification") == "vus"]
    for vus in vus_results:
        gene = vus.get("gene", "?")
        _triage_vus(assessment, vus, top_diseases, disease_genes, disease_names, inheritance_hint, family_history)

    # ── Pathogenic / Likely Pathogenic Actions ───────
    pathogenic_results = [
        g for g in gene_results
        if g.get("classification") in ("pathogenic", "likely_pathogenic")
    ]
    for path_result in pathogenic_results:
        _handle_pathogenic(assessment, path_result, top_diseases, disease_genes, disease_names, inheritance_hint)

    # ── Escalation Path ──────────────────────────────
    _build_escalation(assessment, prior_testing, gene_results, confidence, inheritance_hint, top_diseases, disease_genes)

    # ── Reanalysis Plan ──────────────────────────────
    _build_reanalysis_plan(assessment, prior_testing, gene_results)

    # ── Counseling ───────────────────────────────────
    _build_counseling(assessment, gene_results, pathogenic_results, inheritance_hint, confidence)

    # ── Timeline Bucketing ───────────────────────────
    _bucket_by_timeline(assessment)

    # ── Summary ──────────────────────────────────────
    assessment.summary = _build_summary(assessment, gene_results, prior_testing, confidence)

    return assessment


def _triage_vus(
    assessment: GenomicAssessment,
    vus: dict,
    top_diseases: list[tuple[str, float]],
    disease_genes: dict[str, set[str]],
    disease_names: dict[str, str],
    inheritance_hint: str | None,
    family_history: dict | None,
) -> None:
    """Triage a VUS: determine if it's in a relevant gene and what to do next."""
    gene = vus.get("gene", "?").upper()

    # Check if VUS gene is associated with any top candidate disease
    relevant_diseases = []
    for did, score in top_diseases[:10]:
        if gene in disease_genes.get(did, set()):
            relevant_diseases.append((did, disease_names.get(did, did), score))

    if relevant_diseases:
        top_match = relevant_diseases[0]
        assessment.vus_triage.append(GenomicAction(
            action=f"Prioritize VUS in {gene} — gene is associated with top candidate {top_match[1]}",
            action_type="interpretation",
            rationale=(
                f"The VUS in {gene} is in a gene directly associated with the #1 ranked "
                f"diagnosis ({top_match[1]}). This variant warrants urgent follow-up."
            ),
            urgency="urgent",
            evidence_level="guideline",
            acmg_criteria=["PP4 (phenotype highly specific for gene)"],
        ))

        # Segregation
        assessment.vus_triage.append(GenomicAction(
            action=f"Segregation analysis: test parents/affected family members for {gene} variant",
            action_type="test",
            rationale=(
                f"Segregation with disease would add PS2 (de novo) or PP1 (co-segregation) "
                f"evidence, potentially upgrading VUS to likely pathogenic."
            ),
            urgency="urgent",
            evidence_level="guideline",
            acmg_criteria=["PS2 (de novo)", "PP1 (co-segregation)"],
            cost_estimate="$300-500",
            turnaround="2-3 weeks",
        ))

        # Functional assay if available
        assessment.vus_triage.append(GenomicAction(
            action=f"Request functional assay for {gene} variant (if available)",
            action_type="test",
            rationale=(
                f"Functional evidence (PS3) is strong evidence for pathogenicity. "
                f"Check ClinGen for established functional assays for {gene}."
            ),
            urgency="routine",
            evidence_level="guideline",
            acmg_criteria=["PS3 (functional studies)"],
            cost_estimate="$1,000-3,000",
            turnaround="1-3 months",
            prerequisites=[f"Validated functional assay exists for {gene}"],
        ))
    else:
        # VUS in irrelevant gene — deprioritize
        assessment.vus_triage.append(GenomicAction(
            action=f"Monitor VUS in {gene} — gene not directly relevant to current differential",
            action_type="monitoring",
            rationale=(
                f"The VUS in {gene} is not in a gene associated with current top candidate "
                f"diseases. Add to ClinVar watchlist for potential reclassification."
            ),
            urgency="next_visit",
            evidence_level="expert_consensus",
        ))


def _handle_pathogenic(
    assessment: GenomicAssessment,
    result: dict,
    top_diseases: list[tuple[str, float]],
    disease_genes: dict[str, set[str]],
    disease_names: dict[str, str],
    inheritance_hint: str | None,
) -> None:
    """Handle pathogenic/likely pathogenic findings."""
    gene = result.get("gene", "?").upper()
    classification = result.get("classification", "pathogenic")

    # Check gene-disease association
    matching_diseases = []
    for did, score in top_diseases[:10]:
        if gene in disease_genes.get(did, set()):
            matching_diseases.append(disease_names.get(did, did))

    if matching_diseases:
        assessment.escalation_path.append(GenomicAction(
            action=f"Confirm {classification} variant in {gene} as diagnostic for {matching_diseases[0]}",
            action_type="interpretation",
            rationale=(
                f"Pathogenic variant in {gene} aligns with top phenotype-matched disease "
                f"{matching_diseases[0]}. Confirm clinical diagnosis."
            ),
            urgency="stat",
            evidence_level="guideline",
        ))

    # Autosomal recessive: check for second hit
    if inheritance_hint == "autosomal_recessive":
        assessment.escalation_path.append(GenomicAction(
            action=f"Search for second allele in {gene} (compound het / homozygous)",
            action_type="interpretation",
            rationale=(
                f"Autosomal recessive inheritance suspected. A single pathogenic variant "
                f"may require a second hit. Check for compound heterozygosity or homozygosity."
            ),
            urgency="urgent",
            evidence_level="guideline",
            acmg_criteria=["PM3 (trans with pathogenic)"],
        ))


def _build_escalation(
    assessment: GenomicAssessment,
    prior_testing: str,
    gene_results: list[dict],
    confidence: float,
    inheritance_hint: str | None,
    top_diseases: list[tuple[str, float]],
    disease_genes: dict[str, set[str]],
) -> None:
    """Build escalation pathway based on what's been done."""
    has_diagnosis = any(
        g.get("classification") in ("pathogenic", "likely_pathogenic")
        for g in gene_results
    )
    if has_diagnosis:
        return  # No escalation needed if diagnosed

    if prior_testing == "none" or prior_testing == "panel":
        assessment.escalation_path.append(GenomicAction(
            action="Escalate to clinical exome (or exome trio if parents available)",
            action_type="test",
            rationale=(
                f"Current testing level ({prior_testing}) non-diagnostic. "
                f"Exome captures ~85% of coding variants. Trio enables de novo detection."
            ),
            urgency="routine",
            evidence_level="guideline",
            cost_estimate="$3,500-7,000",
            turnaround="6-10 weeks",
        ))

    if prior_testing == "exome":
        # Decide WGS vs RNA-seq based on context
        assessment.escalation_path.append(GenomicAction(
            action="Escalate to whole genome sequencing (WGS)",
            action_type="test",
            rationale=(
                "Exome non-diagnostic. WGS captures structural variants, "
                "non-coding regulatory mutations, repeat expansions, and deep intronic variants."
            ),
            urgency="routine",
            evidence_level="guideline",
            cost_estimate="$5,000",
            turnaround="6-8 weeks",
        ))
        assessment.escalation_path.append(GenomicAction(
            action="Consider RNA-seq if accessible tissue available",
            action_type="test",
            rationale=(
                "RNA-seq detects aberrant splicing, allele-specific expression, "
                "and expression outliers. ~35% additional yield in specific tissues."
            ),
            urgency="next_visit",
            evidence_level="expert_consensus",
            cost_estimate="$4,000",
            turnaround="6-8 weeks",
            prerequisites=["Accessible tissue (blood, skin, muscle)"],
        ))

    if prior_testing == "wgs":
        # Post-WGS: methylation, long-read, research
        if confidence < 0.3:
            assessment.escalation_path.append(GenomicAction(
                action="Methylation array (episignature analysis)",
                action_type="test",
                rationale=(
                    "Methylation signatures can diagnose chromatin remodeling disorders, "
                    "imprinting defects, and some overgrowth syndromes missed by sequencing."
                ),
                urgency="next_visit",
                evidence_level="expert_consensus",
                cost_estimate="$2,500",
                turnaround="4-6 weeks",
            ))
            assessment.escalation_path.append(GenomicAction(
                action="Long-read WGS (ONT/PacBio) for complex SVs and repeat expansions",
                action_type="test",
                rationale=(
                    "Long-read sequencing resolves complex structural variants, "
                    "repeat expansions, and phasing that short-read WGS misses."
                ),
                urgency="next_visit",
                evidence_level="research",
                cost_estimate="$8,000",
                turnaround="8-12 weeks",
            ))


def _build_reanalysis_plan(
    assessment: GenomicAssessment,
    prior_testing: str,
    gene_results: list[dict],
) -> None:
    """Build systematic reanalysis cadence."""
    if prior_testing in ("exome", "wgs"):
        assessment.reanalysis_plan.append(GenomicAction(
            action="Schedule data reanalysis in 12-18 months",
            action_type="monitoring",
            rationale=(
                "~15% additional diagnoses come from periodic reanalysis. "
                "New gene-disease associations are added monthly. "
                "ClinVar reclassifications may upgrade VUS."
            ),
            urgency="periodic",
            evidence_level="guideline",
            cost_estimate="$200-500",
            turnaround="2-4 weeks",
        ))

    if any(g.get("classification") == "vus" for g in gene_results):
        assessment.reanalysis_plan.append(GenomicAction(
            action="ClinVar monitoring for VUS reclassification",
            action_type="monitoring",
            rationale=(
                "VUS variants are periodically reclassified as new evidence accumulates. "
                "Automated alerts via ClinVar or laboratory portals."
            ),
            urgency="periodic",
            evidence_level="guideline",
        ))


def _build_counseling(
    assessment: GenomicAssessment,
    gene_results: list[dict],
    pathogenic_results: list[dict],
    inheritance_hint: str | None,
    confidence: float,
) -> None:
    """Build genetic counseling recommendations."""
    if pathogenic_results:
        assessment.counseling.append(GenomicAction(
            action="Genetic counseling: discuss diagnosis, prognosis, and management",
            action_type="counseling",
            rationale="Positive genetic finding requires counseling for the patient and family.",
            urgency="urgent",
            evidence_level="guideline",
        ))

        if inheritance_hint in ("autosomal_recessive", "autosomal_dominant", "x_linked"):
            assessment.counseling.append(GenomicAction(
                action=f"Discuss {inheritance_hint.replace('_', ' ')} inheritance with family",
                action_type="counseling",
                rationale=(
                    "Family members may be at risk. Carrier testing and "
                    "reproductive counseling should be offered."
                ),
                urgency="routine",
                evidence_level="guideline",
            ))

    if confidence < 0.3 and not pathogenic_results:
        assessment.counseling.append(GenomicAction(
            action="Counseling: manage expectations for diagnostic odyssey",
            action_type="counseling",
            rationale=(
                "Low diagnostic confidence with current data. Discuss the iterative "
                "nature of rare disease diagnosis and the value of ongoing monitoring."
            ),
            urgency="routine",
            evidence_level="expert_consensus",
        ))


def _bucket_by_timeline(assessment: GenomicAssessment) -> None:
    """Sort all actions into now/next_visit/periodic buckets."""
    all_actions = (
        assessment.vus_triage +
        assessment.escalation_path +
        assessment.reanalysis_plan +
        assessment.counseling
    )

    seen = set()
    for action in all_actions:
        key = action.action
        if key in seen:
            continue
        seen.add(key)

        if action.urgency in ("stat", "urgent"):
            assessment.now_actions.append(action)
        elif action.urgency in ("routine", "next_visit"):
            assessment.next_visit_actions.append(action)
        elif action.urgency == "periodic":
            assessment.periodic_actions.append(action)
        else:
            assessment.next_visit_actions.append(action)


def _build_summary(
    assessment: GenomicAssessment,
    gene_results: list[dict],
    prior_testing: str,
    confidence: float,
) -> str:
    """Build a one-paragraph genomic assessment summary."""
    parts = []

    n_vus = len([g for g in gene_results if g.get("classification") == "vus"])
    n_path = len([g for g in gene_results if g.get("classification") in ("pathogenic", "likely_pathogenic")])

    if n_path > 0:
        genes = [g["gene"] for g in gene_results if g.get("classification") in ("pathogenic", "likely_pathogenic")]
        parts.append(f"Pathogenic variant(s) found in {', '.join(genes)}. Diagnosis likely confirmed.")
    elif n_vus > 0:
        genes = [g["gene"] for g in gene_results if g.get("classification") == "vus"]
        parts.append(f"VUS in {', '.join(genes)} requires follow-up (segregation, functional studies).")
    else:
        parts.append(f"No diagnostic genetic findings to date (testing level: {prior_testing}).")

    parts.append(f"Genomic workup maturity: {assessment.genomic_maturity}.")
    parts.append(f"{len(assessment.now_actions)} actions recommended now, "
                 f"{len(assessment.next_visit_actions)} at next visit.")

    return " ".join(parts)

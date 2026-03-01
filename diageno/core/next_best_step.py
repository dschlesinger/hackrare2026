"""Next-Best-Step Optimizer — Value-of-Information Decisioning.

Ranks next actions by expected uncertainty reduction under constraints
(cost, turnaround time, invasiveness).

Output: top phenotype questions, top tests, reanalysis trigger, referral trigger.

Key novelty: replaces static rule selection with VOI-based ranking.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("diageno.core.next_best_step")


# ── Action Catalog with Constraints ──────────────────

@dataclass
class ActionSpec:
    """Specification for a diagnostic action with real-world constraints."""
    action_id: str
    action_type: str           # phenotype_query | lab_test | genetic_test | imaging | referral | reanalysis
    name: str
    description: str = ""
    cost_usd: float = 0.0     # estimated cost
    turnaround_days: float = 0.0  # expected turnaround time
    invasiveness: float = 0.0  # 0 (non-invasive) to 1 (highly invasive)
    availability: str = "common"  # common | specialized | research_only


@dataclass
class ScoredAction:
    """An action scored by the VOI optimizer."""
    action: ActionSpec
    voi_score: float          # value of information (bits of uncertainty reduction)
    cost_adjusted_voi: float  # VOI adjusted for cost/time/invasiveness
    expected_impact: str      # human-readable impact description
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    why_this_over_alternatives: str = ""
    urgency: str = "routine"  # stat | urgent | routine | elective


# Pre-defined test catalog with real-world constraints
TEST_CATALOG: dict[str, ActionSpec] = {
    "gene_panel": ActionSpec(
        "gene_panel", "genetic_test", "Targeted Gene Panel",
        "Panel of genes associated with top candidate diseases",
        cost_usd=1500, turnaround_days=21, invasiveness=0.1,
    ),
    "clinical_exome": ActionSpec(
        "clinical_exome", "genetic_test", "Clinical Exome Sequencing",
        "Whole exome sequencing with clinical interpretation",
        cost_usd=3500, turnaround_days=56, invasiveness=0.1,
    ),
    "exome_trio": ActionSpec(
        "exome_trio", "genetic_test", "Exome Trio (proband + parents)",
        "Exome with parental samples for de novo detection",
        cost_usd=7000, turnaround_days=56, invasiveness=0.1,
    ),
    "wgs": ActionSpec(
        "wgs", "genetic_test", "Whole Genome Sequencing",
        "Short-read WGS for structural variants and non-coding regions",
        cost_usd=5000, turnaround_days=42, invasiveness=0.1,
    ),
    "long_read_wgs": ActionSpec(
        "long_read_wgs", "genetic_test", "Long-Read WGS (PacBio/ONT)",
        "Long-read sequencing for complex SVs, repeat expansions",
        cost_usd=8000, turnaround_days=56, invasiveness=0.1, availability="specialized",
    ),
    "rna_seq": ActionSpec(
        "rna_seq", "genetic_test", "RNA Sequencing",
        "Transcriptome analysis for splice effects and expression",
        cost_usd=4000, turnaround_days=42, invasiveness=0.3, availability="specialized",
    ),
    "methylation_array": ActionSpec(
        "methylation_array", "genetic_test", "Genome-Wide Methylation Array",
        "Episignature analysis for imprinting/chromatin disorders",
        cost_usd=2500, turnaround_days=28, invasiveness=0.1, availability="specialized",
    ),
    "segregation": ActionSpec(
        "segregation", "genetic_test", "Segregation Analysis",
        "Test family members for variant co-segregation",
        cost_usd=500, turnaround_days=14, invasiveness=0.1,
    ),
    "functional_assay": ActionSpec(
        "functional_assay", "lab_test", "Functional Assay",
        "In-vitro functional testing of variant effect",
        cost_usd=3000, turnaround_days=90, invasiveness=0.0, availability="research_only",
    ),
    "reanalysis": ActionSpec(
        "reanalysis", "reanalysis", "Data Reanalysis",
        "Reanalysis of existing sequencing data with updated annotations",
        cost_usd=200, turnaround_days=14, invasiveness=0.0,
    ),
    "mito_sequencing": ActionSpec(
        "mito_sequencing", "genetic_test", "Mitochondrial Genome Sequencing",
        "Full mtDNA sequencing with heteroplasmy quantification",
        cost_usd=1200, turnaround_days=21, invasiveness=0.1,
    ),
    "muscle_biopsy": ActionSpec(
        "muscle_biopsy", "lab_test", "Muscle Biopsy + Respiratory Chain Analysis",
        "Direct functional assessment of mitochondrial function",
        cost_usd=4000, turnaround_days=28, invasiveness=0.8,
    ),
    "carrier_testing": ActionSpec(
        "carrier_testing", "genetic_test", "Carrier Testing (family)",
        "Targeted testing of family members for known variant",
        cost_usd=300, turnaround_days=14, invasiveness=0.1,
    ),
    "genetics_referral": ActionSpec(
        "genetics_referral", "referral", "Clinical Genetics Referral",
        "Re-phenotyping by clinical geneticist",
        cost_usd=500, turnaround_days=60, invasiveness=0.0,
    ),
    "udp_referral": ActionSpec(
        "udp_referral", "referral", "Undiagnosed Disease Program Referral",
        "Multi-omics research pipeline for unsolved cases",
        cost_usd=0, turnaround_days=180, invasiveness=0.0, availability="specialized",
    ),
    "longitudinal_monitoring": ActionSpec(
        "longitudinal_monitoring", "referral", "Longitudinal Phenotyping & Monitoring",
        "Serial clinical assessments to capture evolving phenotype",
        cost_usd=200, turnaround_days=365, invasiveness=0.0,
    ),
}


class NextBestStepOptimizer:
    """Value-of-Information based next-best-step optimizer.

    Ranks diagnostic actions by expected uncertainty reduction,
    adjusted for real-world constraints (cost, time, invasiveness).
    """

    def __init__(
        self,
        disease_scores: list[tuple[str, float]],
        phenotype_questions: list[dict],
        uncertainty: dict[str, float] | None = None,
        prior_testing: str = "none",
        vus_present: bool = False,
        inheritance_hint: str | None = None,
        confidence: float = 0.0,
        gene_results: list[dict] | None = None,
        disease_genes: dict[str, set[str]] | None = None,
    ):
        self.disease_scores = disease_scores
        self.phenotype_questions = phenotype_questions
        self.uncertainty = uncertainty or {}
        self.prior_testing = prior_testing
        self.vus_present = vus_present
        self.inheritance_hint = inheritance_hint
        self.confidence = confidence
        self.gene_results = gene_results or []
        self.disease_genes = disease_genes or {}

    def optimize(self, max_actions: int = 10) -> list[ScoredAction]:
        """Compute value-of-information for all candidate actions.

        Returns scored actions ranked by cost-adjusted VOI.
        """
        scored: list[ScoredAction] = []

        # 1. Score phenotype questions as actions
        for pq in self.phenotype_questions[:10]:
            info_gain = pq.get("expected_info_gain", 0.0)
            action = ActionSpec(
                action_id=f"ask_{pq['hpo_id']}",
                action_type="phenotype_query",
                name=f"Ask about: {pq.get('label', pq['hpo_id'])}",
                description=pq.get("rationale", ""),
                cost_usd=0, turnaround_days=0, invasiveness=0.0,
            )
            scored.append(ScoredAction(
                action=action,
                voi_score=info_gain,
                cost_adjusted_voi=info_gain * 10.0,  # free actions get 10x boost
                expected_impact=f"Reduces diagnostic uncertainty by {info_gain:.3f} bits",
                confidence=pq.get("p_present", 0.5),
                why_this_over_alternatives=(
                    "Phenotype queries are zero-cost, immediate, and can narrow "
                    "the differential without any invasive procedure."
                ),
                urgency="routine",
            ))

        # 2. Score genetic tests based on clinical context
        test_actions = self._score_genetic_tests()
        scored.extend(test_actions)

        # 3. Score reanalysis and referral actions
        reanalysis_actions = self._score_reanalysis_and_referral()
        scored.extend(reanalysis_actions)

        # Sort by cost-adjusted VOI
        scored.sort(key=lambda x: x.cost_adjusted_voi, reverse=True)
        return scored[:max_actions]

    def _score_genetic_tests(self) -> list[ScoredAction]:
        """Score genetic testing actions based on VOI framework."""
        actions = []
        diagnostic_uncertainty = self.uncertainty.get("phenotype", 0.5)
        genomic_uncertainty = self.uncertainty.get("genomic", 0.8)

        # Determine which tests are appropriate based on prior testing
        if self.prior_testing == "none":
            # Gene panel first
            panel_voi = self._compute_test_voi("gene_panel", genomic_uncertainty)
            actions.append(self._build_test_action(
                "gene_panel", panel_voi,
                evidence=[
                    "No prior genetic testing performed",
                    f"Genomic uncertainty is high ({genomic_uncertainty:.0%})",
                    "Gene panels are cost-effective first-line testing",
                ],
                contra=["May miss diagnoses outside panel scope"],
                why="Gene panels offer the highest diagnostic yield per dollar for phenotypically suggestive cases. Starting broader (exome) is warranted only when the differential is very wide.",
            ))

            # Clinical exome as fallback
            exome_voi = self._compute_test_voi("clinical_exome", genomic_uncertainty * 0.8)
            actions.append(self._build_test_action(
                "clinical_exome", exome_voi,
                evidence=[
                    "Broad differential may benefit from exome-level coverage",
                    f"Diagnostic uncertainty: {diagnostic_uncertainty:.0%}",
                ],
                contra=["Higher cost than targeted panel", "Longer turnaround"],
                why="Exome provides broader unbiased coverage but costs ~2x panel. Preferred when phenotype does not clearly map to a single gene panel.",
            ))

        elif self.prior_testing == "panel":
            # Panel was done → recommend exome
            exome_voi = self._compute_test_voi("clinical_exome", genomic_uncertainty)
            actions.append(self._build_test_action(
                "clinical_exome", exome_voi,
                evidence=[
                    "Targeted panel already completed",
                    "~25-30% of rare disease diagnoses missed by panels are found by exome",
                ],
                contra=["If panel was very comprehensive, incremental yield is lower"],
                why="After negative panel, exome is the standard next step. Trio is preferred if parents available, as it enables de novo detection.",
            ))

            # Trio if de novo suspected
            if self.inheritance_hint in ("de_novo", None):
                trio_voi = self._compute_test_voi("exome_trio", genomic_uncertainty * 1.1)
                actions.append(self._build_test_action(
                    "exome_trio", trio_voi,
                    evidence=[
                        "Trio analysis enables de novo variant detection",
                        "De novo variants account for ~50% of developmental disorders",
                    ],
                    contra=["Requires parental samples", "Higher cost"],
                    why="Trio increases diagnostic yield by 10-15% over singleton exome through de novo and compound het detection.",
                ))

        elif self.prior_testing == "exome":
            # Exome was done → WGS, reanalysis, RNA-seq
            wgs_voi = self._compute_test_voi("wgs", genomic_uncertainty)
            actions.append(self._build_test_action(
                "wgs", wgs_voi,
                evidence=[
                    "Exome was non-diagnostic",
                    "WGS detects structural variants, deep intronic, and repeat expansions missed by exome",
                ],
                contra=["Incremental yield over exome is 5-10% for WGS"],
                why="WGS captures non-coding variants and structural variants missed by exome capture. Cost has dropped significantly.",
            ))

            rna_voi = self._compute_test_voi("rna_seq", genomic_uncertainty * 0.7)
            actions.append(self._build_test_action(
                "rna_seq", rna_voi,
                evidence=[
                    "RNA-seq can detect aberrant splicing and expression outliers",
                    "~35% diagnostic uplift in muscle/skin accessible tissues",
                ],
                contra=["Requires accessible tissue", "Not all genes are expressed in available tissues"],
                why="RNA-seq provides functional evidence for VUS and detects splicing defects invisible to DNA-only analysis.",
                availability_note="specialized",
            ))

        elif self.prior_testing == "wgs":
            # WGS done → advanced techniques
            reanalysis_voi = self._compute_test_voi("reanalysis", genomic_uncertainty * 0.5)
            actions.append(self._build_test_action(
                "reanalysis", reanalysis_voi,
                evidence=[
                    "~15% additional diagnoses from reanalysis at 1-3 year intervals",
                    "New gene-disease associations discovered continuously",
                ],
                contra=["Diminishing returns if recently analyzed"],
                why="Reanalysis is the most cost-effective action after comprehensive testing. New gene-disease links are added monthly.",
            ))

        # VUS-specific actions
        if self.vus_present:
            seg_voi = self._compute_test_voi("segregation", 0.6)
            actions.append(self._build_test_action(
                "segregation", seg_voi,
                evidence=[
                    "VUS detected — segregation can provide PS2/PM6 evidence",
                    "Family co-segregation is strong evidence for pathogenicity",
                ],
                contra=["Family members may not be available"],
                why="Segregation analysis is the cheapest, fastest way to upgrade a VUS. A single informative meiosis adds moderate pathogenicity evidence.",
                urgency="urgent",
            ))

        # Mitochondrial-specific
        if self.inheritance_hint == "mitochondrial":
            mito_voi = self._compute_test_voi("mito_sequencing", genomic_uncertainty * 0.9)
            actions.append(self._build_test_action(
                "mito_sequencing", mito_voi,
                evidence=[
                    "Mitochondrial inheritance suspected",
                    "Standard exome/WGS may miss mtDNA heteroplasmy",
                ],
                contra=["Tissue-specific heteroplasmy may require multiple samples"],
                why="Mitochondrial variants require specialized sequencing with heteroplasmy quantification not provided by standard panels.",
            ))

        return actions

    def _score_reanalysis_and_referral(self) -> list[ScoredAction]:
        """Score reanalysis and referral actions."""
        actions = []

        # Low confidence → genetics referral
        if self.confidence < 0.3:
            ref_voi = 0.3 * (1.0 - self.confidence)
            actions.append(self._build_test_action(
                "genetics_referral", ref_voi,
                evidence=[
                    f"Low diagnostic confidence ({self.confidence:.0%})",
                    "Clinical geneticist may identify subtle dysmorphic features",
                ],
                contra=["Wait times for genetics clinics can be 3-6 months"],
                why="Re-phenotyping by a specialist often identifies missed features that dramatically change the differential.",
            ))

        # Very low confidence + extensive testing → UDP
        if self.confidence < 0.2 and self.prior_testing in ("exome", "wgs"):
            udp_voi = 0.2
            actions.append(self._build_test_action(
                "udp_referral", udp_voi,
                evidence=[
                    "Extensive testing non-diagnostic",
                    "UDP programs have ~25-30% diagnostic rate for pre-screened cases",
                ],
                contra=["Long enrollment timelines", "Research setting — not clinical"],
                why="Undiagnosed Disease Programs use multi-omics approaches (RNA-seq, metabolomics, functional studies) not available clinically.",
            ))

        # Longitudinal monitoring for evolving phenotype
        if self.confidence < 0.5:
            mon_voi = 0.1
            actions.append(self._build_test_action(
                "longitudinal_monitoring", mon_voi,
                evidence=[
                    "Phenotype may evolve over time",
                    "Serial assessments catch emerging features",
                ],
                contra=["Delayed diagnosis"],
                why="Some rare diseases have age-dependent phenotypes. Return visits at 6-12 month intervals can reveal diagnostic features.",
            ))

        return actions

    def _compute_test_voi(self, test_id: str, base_uncertainty: float) -> float:
        """Compute value-of-information for a test.

        VOI = expected_uncertainty_reduction * diagnostic_yield ÷ cost_factor
        """
        spec = TEST_CATALOG.get(test_id)
        if not spec:
            return 0.0

        # Base VOI = uncertainty that this test could resolve
        expected_yield = {
            "gene_panel": 0.25,
            "clinical_exome": 0.35,
            "exome_trio": 0.42,
            "wgs": 0.12,
            "long_read_wgs": 0.08,
            "rna_seq": 0.15,
            "methylation_array": 0.10,
            "segregation": 0.30,
            "functional_assay": 0.20,
            "reanalysis": 0.15,
            "mito_sequencing": 0.20,
            "muscle_biopsy": 0.15,
            "carrier_testing": 0.05,
            "genetics_referral": 0.20,
            "udp_referral": 0.25,
            "longitudinal_monitoring": 0.10,
        }.get(test_id, 0.1)

        voi = base_uncertainty * expected_yield
        return voi

    def _build_test_action(
        self,
        test_id: str,
        voi: float,
        evidence: list[str] | None = None,
        contra: list[str] | None = None,
        why: str = "",
        urgency: str = "routine",
        availability_note: str | None = None,
    ) -> ScoredAction:
        """Build a scored action from a test catalog entry."""
        spec = TEST_CATALOG.get(test_id)
        if not spec:
            spec = ActionSpec(test_id, "unknown", test_id)

        # Cost-adjusted VOI: penalize expensive, slow, invasive tests
        cost_factor = 1.0 + (spec.cost_usd / 10000.0)
        time_factor = 1.0 + (spec.turnaround_days / 90.0)
        invasiveness_factor = 1.0 + spec.invasiveness * 2.0
        adjusted_voi = voi / (cost_factor * time_factor * invasiveness_factor)

        # Boost if availability is common
        if spec.availability == "common":
            adjusted_voi *= 1.2

        impact = (
            f"Expected diagnostic yield: ~{voi*100:.0f}% uncertainty reduction. "
            f"Cost: ${spec.cost_usd:,.0f}, turnaround: {spec.turnaround_days:.0f} days."
        )

        return ScoredAction(
            action=spec,
            voi_score=voi,
            cost_adjusted_voi=adjusted_voi,
            expected_impact=impact,
            supporting_evidence=evidence or [],
            contradicting_evidence=contra or [],
            confidence=min(1.0, voi * 2),
            why_this_over_alternatives=why,
            urgency=urgency,
        )

    def get_immediate_vs_next_visit(self) -> dict[str, list[ScoredAction]]:
        """Split actions into now vs next-visit buckets."""
        all_actions = self.optimize(max_actions=15)
        now = []
        next_visit = []

        for sa in all_actions:
            if sa.action.action_type == "phenotype_query":
                now.append(sa)
            elif sa.urgency == "stat" or sa.urgency == "urgent":
                now.append(sa)
            elif sa.action.turnaround_days <= 14 and sa.action.cost_usd <= 1000:
                now.append(sa)
            else:
                next_visit.append(sa)

        return {"now": now[:8], "next_visit": next_visit[:5]}

    def to_dict_list(self) -> list[dict]:
        """Convert optimized actions to serializable dicts."""
        actions = self.optimize()
        result = []
        for i, sa in enumerate(actions):
            result.append({
                "rank": i + 1,
                "action_type": sa.action.action_type,
                "action": sa.action.name,
                "description": sa.action.description,
                "voi_score": round(sa.voi_score, 4),
                "cost_adjusted_voi": round(sa.cost_adjusted_voi, 4),
                "cost_usd": sa.action.cost_usd,
                "turnaround_days": sa.action.turnaround_days,
                "invasiveness": sa.action.invasiveness,
                "expected_impact": sa.expected_impact,
                "supporting_evidence": sa.supporting_evidence,
                "contradicting_evidence": sa.contradicting_evidence,
                "why_this_over_alternatives": sa.why_this_over_alternatives,
                "confidence": round(sa.confidence, 3),
                "urgency": sa.urgency,
            })
        return result

"""Unified Patient State Model.

One canonical case object that every module reads from.
Ingests: HPOs, free-text notes, family history, prior tests,
gene findings, onset/severity, imaging/lab summaries.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PhenotypeStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    PAST_HISTORY = "past_history"
    UNKNOWN = "unknown"
    SUSPECTED = "suspected"

    @classmethod
    def _missing_(cls, value: object):
        """Fall back to PRESENT for any unknown status."""
        return cls.PRESENT


class ACMGClassification(str, Enum):
    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    VUS = "vus"
    LIKELY_BENIGN = "likely_benign"
    BENIGN = "benign"
    UNCERTAIN_SIGNIFICANCE = "uncertain_significance"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object):
        """Fall back to VUS for any unknown classification."""
        return cls.VUS


class TestType(str, Enum):
    NONE = "none"
    PANEL = "panel"
    EXOME = "exome"
    WGS = "wgs"
    TARGETED = "targeted"
    RNA_SEQ = "rna_seq"
    METHYLATION = "methylation"
    LONG_READ = "long_read"

    @classmethod
    def _missing_(cls, value: object):
        return cls.NONE


class InheritancePattern(str, Enum):
    AUTOSOMAL_DOMINANT = "autosomal_dominant"
    AUTOSOMAL_RECESSIVE = "autosomal_recessive"
    X_LINKED = "x_linked"
    MITOCHONDRIAL = "mitochondrial"
    DE_NOVO = "de_novo"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object):
        return cls.UNKNOWN


@dataclass
class Phenotype:
    """A single phenotype observation."""
    hpo_id: str
    label: str = ""
    status: PhenotypeStatus = PhenotypeStatus.PRESENT
    onset: str | None = None          # ISO 8601 duration, e.g. P5Y
    severity: str | None = None       # mild | moderate | severe
    source: str = "clinician"         # clinician | nlp | family_report

    @property
    def is_present(self) -> bool:
        return self.status == PhenotypeStatus.PRESENT

    @property
    def is_absent(self) -> bool:
        return self.status == PhenotypeStatus.ABSENT


@dataclass
class GeneResult:
    """A gene finding from genetic testing."""
    gene: str
    classification: ACMGClassification = ACMGClassification.UNKNOWN
    inheritance: InheritancePattern = InheritancePattern.UNKNOWN
    test_type: TestType = TestType.NONE
    variant: str | None = None        # e.g. c.1234A>G p.(Arg412Gly)
    zygosity: str | None = None       # heterozygous | homozygous | hemizygous
    notes: str | None = None

    @property
    def is_vus(self) -> bool:
        return self.classification == ACMGClassification.VUS

    @property
    def is_pathogenic(self) -> bool:
        return self.classification in (ACMGClassification.PATHOGENIC, ACMGClassification.LIKELY_PATHOGENIC)


@dataclass
class FamilyHistory:
    """Family history information."""
    consanguinity: bool = False
    affected_relatives: int = 0
    inheritance_pattern: InheritancePattern = InheritancePattern.UNKNOWN
    similar_phenotypes_in_family: bool = False
    notes: str | None = None


@dataclass
class ImagingResult:
    """An imaging or lab result."""
    test_name: str
    result_summary: str
    abnormal: bool = False
    date: str | None = None
    details: str | None = None


@dataclass
class PatientState:
    """Unified patient state — the single source of truth for all modules.

    Every diagnostic module reads from this object to ensure consistency.
    """
    # Demographics
    case_id: str | None = None
    age: int | None = None
    sex: str | None = None
    ancestry: str | None = None

    # Phenotypes (core input)
    phenotypes: list[Phenotype] = field(default_factory=list)

    # Genetic testing
    gene_results: list[GeneResult] = field(default_factory=list)
    prior_testing: TestType = TestType.NONE
    test_result: str | None = None   # negative | positive | vus

    # Family history
    family_history: FamilyHistory = field(default_factory=FamilyHistory)

    # Clinical context
    inheritance_hint: InheritancePattern | None = None
    clinical_notes: str | None = None
    imaging_results: list[ImagingResult] = field(default_factory=list)

    # Legacy fields for compatibility
    genes_mentioned: list[str] = field(default_factory=list)

    # ── Derived properties ───────────────────────────

    @property
    def present_hpos(self) -> list[str]:
        """All present HPO IDs."""
        return [p.hpo_id for p in self.phenotypes if p.is_present]

    @property
    def absent_hpos(self) -> list[str]:
        """All explicitly absent HPO IDs."""
        return [p.hpo_id for p in self.phenotypes if p.is_absent]

    @property
    def all_hpo_ids(self) -> list[str]:
        """All HPO IDs regardless of status."""
        return [p.hpo_id for p in self.phenotypes]

    @property
    def has_genetic_testing(self) -> bool:
        return self.prior_testing != TestType.NONE or len(self.gene_results) > 0

    @property
    def has_vus(self) -> bool:
        return any(g.is_vus for g in self.gene_results)

    @property
    def has_pathogenic(self) -> bool:
        return any(g.is_pathogenic for g in self.gene_results)

    @property
    def n_phenotypes(self) -> int:
        return len(self.phenotypes)

    @property
    def n_present(self) -> int:
        return len(self.present_hpos)

    @property
    def n_absent(self) -> int:
        return len(self.absent_hpos)

    @property
    def record_completeness(self) -> float:
        """0–1 score of how complete the patient record is."""
        score = 0.0
        total = 0.0

        # Demographics (weight 1)
        total += 3
        if self.age is not None:
            score += 1
        if self.sex:
            score += 1
        if self.ancestry:
            score += 1

        # Phenotypes (weight 3)
        total += 3
        if self.n_present >= 1:
            score += 1
        if self.n_present >= 3:
            score += 1
        if self.n_present >= 5:
            score += 1

        # Genetic data (weight 2)
        total += 2
        if self.has_genetic_testing:
            score += 1
        if len(self.gene_results) > 0:
            score += 1

        # Family history (weight 1)
        total += 1
        if self.family_history.notes or self.family_history.affected_relatives > 0:
            score += 1

        return score / total if total > 0 else 0.0

    @property
    def deterministic_hash(self) -> str:
        """Deterministic hash for caching and replay reproducibility."""
        key_data = {
            "present_hpos": sorted(self.present_hpos),
            "absent_hpos": sorted(self.absent_hpos),
            "genes": sorted([g.gene for g in self.gene_results]),
            "prior_testing": self.prior_testing.value,
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]

    # ── Factory methods ──────────────────────────────

    @classmethod
    def from_case_input(cls, data: dict) -> "PatientState":
        """Build from CaseInput-style dict (API request format)."""
        phenotypes = []
        for p in data.get("phenotypes", []):
            hpo_id = p.get("hpo_id") or p.get("hpo", "")
            if not hpo_id:
                continue
            phenotypes.append(Phenotype(
                hpo_id=hpo_id,
                label=p.get("label", ""),
                status=PhenotypeStatus(p.get("status", "present")),
                onset=p.get("onset"),
                severity=p.get("severity"),
            ))

        gene_results = []
        for g in data.get("gene_results", []):
            gene_results.append(GeneResult(
                gene=g.get("gene", "").upper(),
                classification=ACMGClassification(g.get("classification", "unknown")),
                inheritance=InheritancePattern(g.get("inheritance", "unknown") or "unknown"),
                test_type=TestType(g.get("test_type", "none") or "none"),
                variant=g.get("variant"),
                zygosity=g.get("zygosity"),
                notes=g.get("notes"),
            ))

        # Legacy genes_mentioned → gene results
        for gene_name in data.get("genes_mentioned", []):
            if gene_name and not any(g.gene == gene_name.upper() for g in gene_results):
                gene_results.append(GeneResult(gene=gene_name.upper()))

        # Family history
        fh_data = data.get("family_history", {})
        family_history = FamilyHistory(
            consanguinity=fh_data.get("consanguinity", False),
            affected_relatives=fh_data.get("affected_relatives", 0),
            notes=fh_data.get("notes"),
        )

        # Imaging
        imaging = []
        for img in data.get("imaging_and_tests", data.get("imaging_results", [])):
            if isinstance(img, dict):
                imaging.append(ImagingResult(
                    test_name=img.get("test_name", img.get("name", "")),
                    result_summary=img.get("result_summary", img.get("result", "")),
                    abnormal=img.get("abnormal", False),
                ))
            elif isinstance(img, str):
                imaging.append(ImagingResult(test_name=img, result_summary=img))

        # Determine prior testing from highest level
        prior_str = data.get("prior_testing", "none") or "none"
        try:
            prior_testing = TestType(prior_str)
        except ValueError:
            prior_testing = TestType.NONE

        # Inheritance hint
        inh_str = data.get("inheritance_hint")
        inheritance_hint = None
        if inh_str:
            try:
                inheritance_hint = InheritancePattern(inh_str)
            except ValueError:
                inheritance_hint = None

        # Patient sub-object
        patient = data.get("patient", {})

        return cls(
            case_id=data.get("case_id"),
            age=data.get("age") or patient.get("age") or patient.get("age_years"),
            sex=data.get("sex") or patient.get("sex"),
            ancestry=data.get("ancestry") or patient.get("ancestry"),
            phenotypes=phenotypes,
            gene_results=gene_results,
            prior_testing=prior_testing,
            test_result=data.get("test_result"),
            family_history=family_history,
            inheritance_hint=inheritance_hint,
            clinical_notes=data.get("clinical_notes"),
            imaging_results=imaging,
            genes_mentioned=data.get("genes_mentioned", []),
        )

    @classmethod
    def from_validation_case(cls, data: dict) -> "PatientState":
        """Build from ValidationCase* JSON format.

        Handles field name inconsistencies across validation cases:
        - 'hpo' vs 'hpo_id' for phenotype IDs
        - 'age' vs 'age_years' for patient age
        - Various gene result formats
        """
        # Normalize phenotypes
        raw_phenos = data.get("phenotypes", [])
        phenotypes = []
        for p in raw_phenos:
            hpo_id = p.get("hpo_id") or p.get("hpo", "")
            if not hpo_id:
                continue
            phenotypes.append(Phenotype(
                hpo_id=hpo_id,
                label=p.get("label", ""),
                status=PhenotypeStatus(p.get("status", "present")),
                onset=p.get("onset"),
            ))

        # Gene results — handle both flat list and structured objects
        gene_results = []
        for g in data.get("gene_results", []):
            if isinstance(g, str):
                gene_results.append(GeneResult(gene=g.upper()))
            elif isinstance(g, dict):
                gene_results.append(GeneResult(
                    gene=g.get("gene", g.get("name", "")).upper(),
                    classification=ACMGClassification(g.get("classification", "unknown")),
                ))

        for gene_name in data.get("genes_mentioned", []):
            if isinstance(gene_name, str) and not any(gr.gene == gene_name.upper() for gr in gene_results):
                gene_results.append(GeneResult(gene=gene_name.upper()))

        # Patient sub-object
        patient = data.get("patient", {})

        # Family history
        fh = FamilyHistory(
            consanguinity=patient.get("consanguinity", False),
            notes=patient.get("family_history_notes"),
        )
        if patient.get("family_history_breast_ovarian_cancer"):
            fh.notes = (fh.notes or "") + " Family history of breast/ovarian cancer."

        # Imaging
        imaging = []
        for img in data.get("imaging_and_tests", []):
            if isinstance(img, dict):
                imaging.append(ImagingResult(
                    test_name=img.get("test", img.get("name", "")),
                    result_summary=img.get("result", img.get("findings", "")),
                    abnormal=img.get("abnormal", True),
                ))

        return cls(
            case_id=data.get("case_id"),
            age=patient.get("age") or patient.get("age_years"),
            sex=patient.get("sex"),
            ancestry=patient.get("ancestry"),
            phenotypes=phenotypes,
            gene_results=gene_results,
            family_history=fh,
            imaging_results=imaging,
            genes_mentioned=data.get("genes_mentioned", []),
        )

    def to_inference_kwargs(self) -> dict:
        """Convert to kwargs for InferenceEngine.recommend()."""
        return {
            "present_hpos": self.present_hpos,
            "absent_hpos": self.absent_hpos,
            "prior_testing": self.prior_testing.value,
            "test_result": self.test_result,
            "inheritance_hint": self.inheritance_hint.value if self.inheritance_hint else None,
            "vus_present": self.has_vus,
            "gene_results": [
                {
                    "gene": g.gene,
                    "classification": g.classification.value,
                    "inheritance": g.inheritance.value,
                    "test_type": g.test_type.value,
                }
                for g in self.gene_results
            ] if self.gene_results else None,
        }

    def drop_phenotypes(self, fraction: float, rng: Any = None) -> "PatientState":
        """Return a new PatientState with a fraction of phenotypes randomly dropped.

        Used for missingness robustness testing.
        """
        import random
        if rng is None:
            rng = random.Random(42)

        n_drop = max(1, int(len(self.phenotypes) * fraction))
        kept = list(self.phenotypes)
        rng.shuffle(kept)
        kept = kept[n_drop:]

        return PatientState(
            case_id=self.case_id,
            age=self.age,
            sex=self.sex,
            ancestry=self.ancestry,
            phenotypes=kept,
            gene_results=self.gene_results,
            prior_testing=self.prior_testing,
            test_result=self.test_result,
            family_history=self.family_history,
            inheritance_hint=self.inheritance_hint,
            clinical_notes=self.clinical_notes,
            imaging_results=self.imaging_results,
            genes_mentioned=self.genes_mentioned,
        )

    def summary(self) -> str:
        """One-line summary for logging."""
        parts = [f"case={self.case_id or '?'}"]
        parts.append(f"{self.n_present}P/{self.n_absent}A HPOs")
        if self.gene_results:
            parts.append(f"{len(self.gene_results)} genes")
        parts.append(f"completeness={self.record_completeness:.0%}")
        return " | ".join(parts)

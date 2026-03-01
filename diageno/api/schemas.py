"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Request Schemas ──────────────────────────────────


class PhenotypeInput(BaseModel):
    hpo_id: str = Field(..., description="HPO identifier, e.g. HP:0001250")
    label: Optional[str] = None
    status: str = Field("present", description="present | absent | past_history")
    onset: Optional[str] = Field(None, description="ISO 8601 onset, e.g. P5Y")


class GeneInput(BaseModel):
    """A gene result with classification and test context."""
    gene: str = Field(..., description="Gene symbol, e.g. SCN9A")
    classification: str = Field(
        "unknown",
        description="pathogenic | likely_pathogenic | vus | likely_benign | benign | unknown",
    )
    inheritance: str = Field(
        "",
        description="autosomal_dominant | autosomal_recessive | x_linked | mitochondrial | de_novo | unknown",
    )
    test_type: str = Field("", description="panel | exome | wgs | targeted")
    notes: Optional[str] = None


class CaseInput(BaseModel):
    case_id: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    ancestry: Optional[str] = None
    phenotypes: list[PhenotypeInput] = Field(default_factory=list)
    prior_testing: str = Field("none", description="none | panel | exome | wgs")
    test_result: Optional[str] = Field(None, description="negative | positive | vus")
    vus_present: bool = False
    inheritance_hint: Optional[str] = None
    genes_mentioned: list[str] = Field(default_factory=list)
    gene_results: list[GeneInput] = Field(
        default_factory=list,
        description="Structured gene findings with classification and inheritance",
    )


class SimulateStepInput(BaseModel):
    """Input for simulating adding/removing a phenotype and seeing ranking changes."""
    case: CaseInput
    new_phenotype: PhenotypeInput
    action: str = Field("add", description="add | remove")


class HPOLookupRequest(BaseModel):
    text: str = Field(..., description="Free-text to lookup HPO terms")
    max_results: int = Field(10, ge=1, le=50)


class ValidateSchemaRequest(BaseModel):
    data: dict = Field(..., description="JSON object to validate against case schema")


# ─── Response Schemas ─────────────────────────────────


class DiseaseCandidate(BaseModel):
    disease_id: str
    name: str = ""
    score: float
    calibrated_score: Optional[float] = None
    supporting_hpos: list[str] = Field(default_factory=list)
    contradicting_hpos: list[str] = Field(default_factory=list)
    rationale: str = Field("", description="Human-readable explanation for this ranking")
    phenotype_match: str = Field("", description="e.g. '3/15' — matched vs total known phenotypes")


class NextBestPhenotype(BaseModel):
    hpo_id: str
    label: Optional[str] = None
    expected_info_gain: float
    p_present: float = 0.0
    rationale: str = ""


class TestRecommendation(BaseModel):
    rank: int
    action_type: str
    action: str
    rationale: str = ""


class HPOExpansionInfo(BaseModel):
    """Info about HPO expansion."""
    input_hpos: int = 0
    expanded_hpos: int = 0
    ancestors_added: int = 0


class GeneIntegrationInfo(BaseModel):
    """Info about gene integration."""
    patient_genes: int = 0
    diseases_with_gene_data: int = 0


# ─── Uncertainty Response ─────────────────────────────


class CounterfactualResponse(BaseModel):
    signal_type: str = ""
    description: str = ""
    expected_impact: str = ""
    impact_magnitude: float = 0.0


class UncertaintyResponse(BaseModel):
    """Three-axis uncertainty decomposition."""
    phenotype_uncertainty: float = 0.0
    genomic_uncertainty: float = 0.0
    decision_uncertainty: float = 0.0
    overall: float = 0.0
    entropy_bits: float = 0.0
    effective_candidates: float = 0.0
    counterfactuals: list[CounterfactualResponse] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "description": "Three-axis uncertainty decomposition: phenotype, genomic, decision."
        }


# ─── VOI-Based Action Response ───────────────────────


class VOIAction(BaseModel):
    """VOI-scored action recommendation."""
    action: str
    action_type: str = ""
    cost_adjusted_voi: float = 0.0
    raw_voi: float = 0.0
    cost_dollars: Optional[float] = None
    turnaround_days: Optional[float] = None
    invasiveness: str = ""
    rationale: str = ""
    timeline_bucket: str = ""


# ─── Genomic Assessment Response ─────────────────────


class GenomicAssessmentResponse(BaseModel):
    """Genomic-first decision module output."""
    genomic_maturity: str = "none"
    escalation_path: list[str] = Field(default_factory=list)
    reanalysis_plan: list[str] = Field(default_factory=list)
    vus_triage: list[str] = Field(default_factory=list)
    counseling: list[str] = Field(default_factory=list)
    now_actions: list[str] = Field(default_factory=list)
    next_visit_actions: list[str] = Field(default_factory=list)
    periodic_actions: list[str] = Field(default_factory=list)


# ─── Evidence Response ────────────────────────────────


class EvidenceItem(BaseModel):
    hpo_id: str = ""
    label: str = ""
    direction: str = ""  # supporting | contradicting | missing
    frequency_label: str = ""


class EvidenceExplanation(BaseModel):
    disease_id: str = ""
    disease_name: str = ""
    rank: int = 0
    score: float = 0.0
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_key_evidence: list[EvidenceItem] = Field(default_factory=list)
    summary: str = ""


class RecommendResponse(BaseModel):
    """Full /recommend response with enhanced modules."""
    case_id: Optional[str] = None
    run_id: Optional[str] = None
    model_version: str = ""
    diseases: list[DiseaseCandidate] = Field(default_factory=list)
    next_best_phenotypes: list[NextBestPhenotype] = Field(default_factory=list)
    test_recommendations: list[TestRecommendation] = Field(default_factory=list)
    confidence: float = 0.0
    inputs_hash: str = ""
    created_at: Optional[datetime] = None
    scoring_method: str = Field("cosine_similarity", description="Scoring algorithm used")
    hpo_expansion: Optional[HPOExpansionInfo] = None
    gene_integration: Optional[GeneIntegrationInfo] = None
    # New: enhanced modules
    uncertainty: Optional[UncertaintyResponse] = None
    voi_actions: list[VOIAction] = Field(default_factory=list)
    genomic_assessment: Optional[GenomicAssessmentResponse] = None
    evidence_explanations: list[EvidenceExplanation] = Field(default_factory=list)
    record_completeness: float = Field(0.0, description="0-1 completeness score of patient record")


class SimulateStepResponse(BaseModel):
    """Response for /simulate_step."""
    before_top5: list[DiseaseCandidate] = Field(default_factory=list)
    after_top5: list[DiseaseCandidate] = Field(default_factory=list)
    rank_changes: list[dict[str, Any]] = Field(default_factory=list)


class HPOLookupResponse(BaseModel):
    results: list[dict[str, str]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    model_loaded: bool = False
    db_connected: bool = False
    cache_connected: bool = False


class ValidationResponse(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


# ─── Evaluation Schemas ──────────────────────────────


class RunEvaluationRequest(BaseModel):
    experiments: list[str] = Field(
        default_factory=lambda: ["replay", "missingness", "calibration", "ablation", "rubric"],
        description="Which experiments to run",
    )


class ExperimentResultResponse(BaseModel):
    experiment: str = ""
    description: str = ""
    metrics: dict = Field(default_factory=dict)
    details: list[dict] = Field(default_factory=list)
    duration_seconds: float = 0.0


class EvaluationSuiteResponse(BaseModel):
    headline_claim: str = ""
    primary_metric: dict = Field(default_factory=dict)
    secondary_metrics: dict = Field(default_factory=dict)
    experiments: list[ExperimentResultResponse] = Field(default_factory=list)

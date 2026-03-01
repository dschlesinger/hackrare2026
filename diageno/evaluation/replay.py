"""Deterministic Replay Pipeline.

Replays ValidationCase* files with reproducible scoring logs.
Evaluates next-step recommendations against ground-truth actions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from diageno.core.patient_state import PatientState
from diageno.evaluation.metrics import (
    steps_to_correct_diagnosis,
    mean_reciprocal_rank,
    hits_at_k,
    compute_all_metrics,
)

logger = logging.getLogger("diageno.evaluation.replay")


@dataclass
class ReplayResult:
    """Result from replaying a single validation case."""
    case_id: str
    filename: str
    n_phenotypes: int
    n_genes: int

    # Disease ranking
    top_diseases: list[dict] = field(default_factory=list)
    confidence: float = 0.0

    # Next-step evaluation
    ground_truth_actions: list[str] = field(default_factory=list)
    recommended_actions: list[dict] = field(default_factory=list)
    steps_to_correct: int | None = None
    action_mrr: float = 0.0

    # Phenotype questions
    phenotype_questions: list[dict] = field(default_factory=list)

    # Test recommendations
    test_recommendations: list[dict] = field(default_factory=list)

    # Uncertainty
    uncertainty: dict = field(default_factory=dict)

    # Genomic assessment
    genomic_assessment: dict = field(default_factory=dict)

    # Timing
    inference_time_ms: float = 0.0

    # Deterministic hash
    input_hash: str = ""

    # Error
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "filename": self.filename,
            "n_phenotypes": self.n_phenotypes,
            "n_genes": self.n_genes,
            "top_diseases": self.top_diseases[:10],
            "confidence": round(self.confidence, 4),
            "ground_truth_actions": self.ground_truth_actions,
            "recommended_actions": self.recommended_actions[:10],
            "steps_to_correct": self.steps_to_correct,
            "action_mrr": round(self.action_mrr, 4),
            "phenotype_questions": self.phenotype_questions[:5],
            "test_recommendations": self.test_recommendations[:5],
            "uncertainty": self.uncertainty,
            "genomic_assessment_summary": self.genomic_assessment.get("summary", ""),
            "inference_time_ms": round(self.inference_time_ms, 1),
            "input_hash": self.input_hash,
            "error": self.error,
        }


@dataclass
class ReplaySummary:
    """Summary of all replayed cases."""
    n_cases: int = 0
    n_errors: int = 0
    mean_confidence: float = 0.0
    mean_steps_to_correct: float = 0.0
    action_mrr: float = 0.0
    action_hits_at_1: float = 0.0
    action_hits_at_3: float = 0.0
    action_hits_at_5: float = 0.0
    mean_inference_time_ms: float = 0.0
    results: list[ReplayResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "n_errors": self.n_errors,
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_steps_to_correct": round(self.mean_steps_to_correct, 2),
            "action_mrr": round(self.action_mrr, 4),
            "action_hits_at_1": round(self.action_hits_at_1, 4),
            "action_hits_at_3": round(self.action_hits_at_3, 4),
            "action_hits_at_5": round(self.action_hits_at_5, 4),
            "mean_inference_time_ms": round(self.mean_inference_time_ms, 1),
            "cases": [r.to_dict() for r in self.results],
        }


def load_validation_cases(workspace_dir: Path) -> list[dict]:
    """Load all ValidationCase* files."""
    cases = []
    for f in sorted(workspace_dir.glob("ValidationCase*")):
        try:
            data = json.loads(f.read_text())
            data["_filename"] = f.name
            cases.append(data)
        except Exception as e:
            logger.warning("Failed to load %s: %s", f.name, e)
    return cases


def extract_ground_truth_actions(case_data: dict) -> list[str]:
    """Extract ground truth next actions from decision points."""
    actions = []
    for dp in case_data.get("decision_points", []):
        # Try different key names used across validation cases
        steps = dp.get("doctor_next_steps_ranked",
                       dp.get("recommended_next_steps_ranked", []))
        for step in steps:
            action_text = step.get("action", "")
            if action_text:
                actions.append(action_text)
    return actions


def replay_single_case(
    case_data: dict,
    engine: Any,
    include_uncertainty: bool = True,
    include_genomics: bool = True,
) -> ReplayResult:
    """Replay a single validation case through the engine.

    Args:
        case_data: raw JSON dict from ValidationCase file
        engine: InferenceEngine instance
        include_uncertainty: whether to compute uncertainty decomposition
        include_genomics: whether to run genomic advisor
    """
    patient = PatientState.from_validation_case(case_data)
    result = ReplayResult(
        case_id=patient.case_id or case_data.get("_filename", "?"),
        filename=case_data.get("_filename", "?"),
        n_phenotypes=patient.n_present,
        n_genes=len(patient.gene_results),
        input_hash=patient.deterministic_hash,
    )

    if patient.n_present == 0:
        result.error = "No present phenotypes"
        return result

    try:
        start = time.time()
        rec = engine.recommend(**patient.to_inference_kwargs())
        result.inference_time_ms = (time.time() - start) * 1000
    except Exception as e:
        result.error = str(e)
        return result

    # Extract results
    result.top_diseases = rec.get("diseases", [])
    result.confidence = rec.get("confidence", 0)
    result.phenotype_questions = rec.get("next_best_phenotypes", [])
    result.test_recommendations = rec.get("test_recommendations", [])
    result.recommended_actions = rec.get("test_recommendations", [])

    # Ground truth next actions
    result.ground_truth_actions = extract_ground_truth_actions(case_data)

    # Steps-to-correct
    if result.ground_truth_actions and result.recommended_actions:
        result.steps_to_correct = steps_to_correct_diagnosis(
            result.ground_truth_actions,
            result.recommended_actions,
        )

    # Action MRR: for each ground truth action, find its rank in recommendations
    if result.ground_truth_actions:
        action_ranks = []
        for gt_action in result.ground_truth_actions:
            found_rank = steps_to_correct_diagnosis([gt_action], result.recommended_actions)
            action_ranks.append(found_rank)
        if action_ranks:
            result.action_mrr = mean_reciprocal_rank(action_ranks)

    # Uncertainty decomposition
    if include_uncertainty:
        try:
            from diageno.core.uncertainty import compute_uncertainty
            unc = compute_uncertainty(
                disease_scores=[(d["disease_id"], d["score"]) for d in result.top_diseases],
                n_present_hpos=patient.n_present,
                n_absent_hpos=patient.n_absent,
                has_genetic_testing=patient.has_genetic_testing,
                has_vus=patient.has_vus,
                gene_results=[g.__dict__ for g in patient.gene_results] if patient.gene_results else None,
                confidence=result.confidence,
                phenotype_questions=result.phenotype_questions,
                disease_genes=engine.disease_genes,
                disease_names=engine.disease_names,
                hpo_names=engine.hpo_names,
            )
            result.uncertainty = unc.to_dict()
        except Exception as e:
            logger.warning("Uncertainty computation failed: %s", e)

    # Genomic assessment
    if include_genomics and patient.gene_results:
        try:
            from diageno.core.genomic_advisor import assess_genomics
            g_assess = assess_genomics(
                gene_results=[
                    {"gene": g.gene, "classification": g.classification.value}
                    for g in patient.gene_results
                ],
                prior_testing=patient.prior_testing.value,
                top_diseases=[(d["disease_id"], d["score"]) for d in result.top_diseases],
                disease_genes=engine.disease_genes,
                disease_names=engine.disease_names,
                inheritance_hint=patient.inheritance_hint.value if patient.inheritance_hint else None,
                confidence=result.confidence,
            )
            result.genomic_assessment = g_assess.to_dict()
        except Exception as e:
            logger.warning("Genomic assessment failed: %s", e)

    return result


def replay_all(
    workspace_dir: Path,
    engine: Any,
    include_uncertainty: bool = True,
    include_genomics: bool = True,
) -> ReplaySummary:
    """Replay all validation cases and compute summary metrics.

    Returns a deterministic, reproducible evaluation.
    """
    cases = load_validation_cases(workspace_dir)
    summary = ReplaySummary(n_cases=len(cases))

    for case_data in cases:
        result = replay_single_case(
            case_data, engine,
            include_uncertainty=include_uncertainty,
            include_genomics=include_genomics,
        )
        summary.results.append(result)
        if result.error:
            summary.n_errors += 1

    # Compute summary metrics
    valid = [r for r in summary.results if not r.error]
    if valid:
        summary.mean_confidence = float(np.mean([r.confidence for r in valid]))
        summary.mean_inference_time_ms = float(np.mean([r.inference_time_ms for r in valid]))

        steps = [r.steps_to_correct for r in valid if r.steps_to_correct is not None]
        if steps:
            summary.mean_steps_to_correct = float(np.mean(steps))
            summary.action_hits_at_1 = hits_at_k(steps, 1)
            summary.action_hits_at_3 = hits_at_k(steps, 3)
            summary.action_hits_at_5 = hits_at_k(steps, 5)

        mrrs = [r.action_mrr for r in valid if r.action_mrr > 0]
        if mrrs:
            summary.action_mrr = float(np.mean(mrrs))

    return summary


def save_replay_log(summary: ReplaySummary, output_path: Path) -> None:
    """Save replay results as JSON for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2, default=str)
    logger.info("Replay log saved to %s", output_path)

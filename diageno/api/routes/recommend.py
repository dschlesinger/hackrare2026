"""/recommend endpoint — full diagnostic recommendation with enhanced modules."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

from diageno.api.schemas import (
    CaseInput,
    RecommendResponse,
    UncertaintyResponse,
    CounterfactualResponse,
    VOIAction,
    GenomicAssessmentResponse,
    EvidenceExplanation,
    EvidenceItem,
)
from diageno.api.services.inference import engine
from diageno.api.services import cache

logger = logging.getLogger("diageno.api.routes.recommend")
router = APIRouter()


def _build_uncertainty(case: CaseInput, result: dict) -> UncertaintyResponse | None:
    """Run uncertainty decomposition on the result."""
    try:
        from diageno.core.uncertainty import compute_uncertainty

        diseases = result.get("diseases", [])
        disease_scores = [(d["disease_id"], d["score"]) for d in diseases]
        present = [p.hpo_id for p in case.phenotypes if p.status == "present"]
        absent = [p.hpo_id for p in case.phenotypes if p.status == "absent"]
        gene_results_raw = [g.model_dump() for g in case.gene_results] if case.gene_results else []
        has_testing = case.prior_testing not in ("none", "", None)
        next_phenos = result.get("next_best_phenotypes", [])

        unc = compute_uncertainty(
            disease_scores=disease_scores,
            n_present_hpos=len(present),
            n_absent_hpos=len(absent),
            has_genetic_testing=has_testing,
            has_vus=case.vus_present,
            gene_results=gene_results_raw,
            confidence=result.get("confidence", 0),
            phenotype_questions=next_phenos,
            disease_genes=engine.disease_genes,
            disease_names=engine.disease_names,
            hpo_names=engine.hpo_names,
        )
        return UncertaintyResponse(
            phenotype_uncertainty=round(unc.phenotype_uncertainty, 4),
            genomic_uncertainty=round(unc.genomic_uncertainty, 4),
            decision_uncertainty=round(unc.decision_uncertainty, 4),
            overall=round(unc.overall_uncertainty, 4),
            entropy_bits=round(unc.entropy_bits, 4),
            effective_candidates=round(unc.effective_candidates, 1),
            counterfactuals=[
                CounterfactualResponse(
                    signal_type=c.signal_type,
                    description=c.description,
                    expected_impact=c.expected_impact,
                    impact_magnitude=round(c.impact_magnitude, 3),
                )
                for c in unc.counterfactuals[:5]
            ],
        )
    except Exception as e:
        logger.warning("Uncertainty computation failed: %s", e)
        return None


def _build_voi_actions(case: CaseInput, result: dict) -> list[dict]:
    """Run VOI-based next-best-step optimizer."""
    try:
        from diageno.core.next_best_step import NextBestStepOptimizer

        diseases = result.get("diseases", [])
        disease_scores = [(d["disease_id"], d["score"]) for d in diseases]
        next_phenos = result.get("next_best_phenotypes", [])
        gene_results_raw = [g.model_dump() for g in case.gene_results] if case.gene_results else []

        optimizer = NextBestStepOptimizer(
            disease_scores=disease_scores,
            phenotype_questions=next_phenos,
            prior_testing=case.prior_testing,
            vus_present=case.vus_present,
            inheritance_hint=case.inheritance_hint,
            confidence=result.get("confidence", 0),
            gene_results=gene_results_raw,
            disease_genes=engine.disease_genes,
        )
        raw = optimizer.to_dict_list()
        # Remap keys to match VOIAction schema
        adapted = []
        for r in raw[:10]:
            adapted.append({
                "action": r.get("action", ""),
                "action_type": r.get("action_type", ""),
                "cost_adjusted_voi": r.get("cost_adjusted_voi", 0),
                "raw_voi": r.get("voi_score", 0),
                "cost_dollars": r.get("cost_usd"),
                "turnaround_days": r.get("turnaround_days"),
                "invasiveness": str(r.get("invasiveness", "")),
                "rationale": r.get("expected_impact", ""),
                "timeline_bucket": r.get("urgency", "routine"),
            })
        return adapted
    except Exception as e:
        logger.warning("VOI action scoring failed: %s", e)
        return []


def _build_genomic_assessment(case: CaseInput, result: dict) -> GenomicAssessmentResponse | None:
    """Run genomic-first decision module."""
    try:
        from diageno.core.genomic_advisor import assess_genomics

        diseases = result.get("diseases", [])
        disease_scores = [(d["disease_id"], d["score"]) for d in diseases]
        gene_results_raw = [g.model_dump() for g in case.gene_results] if case.gene_results else []

        ga = assess_genomics(
            gene_results=gene_results_raw,
            prior_testing=case.prior_testing,
            top_diseases=disease_scores,
            disease_genes=engine.disease_genes,
            disease_names=engine.disease_names,
            inheritance_hint=case.inheritance_hint,
            confidence=result.get("confidence", 0),
        )
        return GenomicAssessmentResponse(
            genomic_maturity=ga.genomic_maturity,
            escalation_path=[a.action for a in ga.escalation_path[:5]],
            reanalysis_plan=[a.action for a in ga.reanalysis_plan[:5]],
            vus_triage=[a.action for a in ga.vus_triage[:5]],
            counseling=[a.action for a in ga.counseling[:5]],
            now_actions=[a.action for a in ga.now_actions[:5]],
            next_visit_actions=[a.action for a in ga.next_visit_actions[:5]],
            periodic_actions=[a.action for a in ga.periodic_actions[:5]],
        )
    except Exception as e:
        logger.warning("Genomic assessment failed: %s", e)
        return None


def _build_evidence(case: CaseInput, result: dict) -> list[EvidenceExplanation]:
    """Build evidence-grounded explanations for top diseases."""
    try:
        from diageno.core.evidence import build_disease_explanation

        present = [p.hpo_id for p in case.phenotypes if p.status == "present"]
        absent = [p.hpo_id for p in case.phenotypes if p.status == "absent"]
        gene_results_raw = [g.model_dump() for g in case.gene_results] if case.gene_results else None

        explanations = []
        for rank_num, d in enumerate(result.get("diseases", [])[:5], 1):
            expl = build_disease_explanation(
                disease=d,
                rank=rank_num,
                matrix=engine.matrix,
                disease_index=engine.disease_index,
                hpo_dict=engine.hpo_dict,
                hpo_names=engine.hpo_names,
                disease_genes=engine.disease_genes,
                patient_present_hpos=present,
                patient_absent_hpos=absent,
                patient_genes=gene_results_raw,
            )
            items_sup = [
                EvidenceItem(
                    hpo_id=getattr(e, "hpo_id", "") if hasattr(e, "hpo_id") else "",
                    label=e.statement[:80] if hasattr(e, "statement") else str(e),
                    direction="supporting",
                    frequency_label=e.strength if hasattr(e, "strength") else "",
                )
                for e in expl.supporting_evidence[:10]
            ]
            items_contra = [
                EvidenceItem(
                    hpo_id="",
                    label=e.statement[:80] if hasattr(e, "statement") else str(e),
                    direction="contradicting",
                    frequency_label=e.strength if hasattr(e, "strength") else "",
                )
                for e in expl.contradicting_evidence[:10]
            ]
            items_missing = [
                EvidenceItem(
                    hpo_id="",
                    label=e.statement[:80] if hasattr(e, "statement") else str(e),
                    direction="missing",
                    frequency_label=e.strength if hasattr(e, "strength") else "",
                )
                for e in expl.missing_evidence[:10]
            ]
            explanations.append(EvidenceExplanation(
                disease_id=d.get("disease_id", ""),
                disease_name=d.get("name", ""),
                rank=rank_num,
                score=d.get("score", 0),
                supporting_evidence=items_sup,
                contradicting_evidence=items_contra,
                missing_key_evidence=items_missing,
                summary=expl.why_ranked_here or expl.phenotype_overlap or "",
            ))
        return explanations
    except Exception as e:
        logger.warning("Evidence explanation failed: %s", e)
        return []


@router.post("/recommend", response_model=RecommendResponse)
def recommend(case: CaseInput) -> RecommendResponse:
    """Generate disease differential + next-best-steps for a case.

    Enhanced with: uncertainty decomposition, VOI-based actions,
    genomic assessment, and evidence-grounded explanations.
    """
    if not case.phenotypes:
        raise HTTPException(
            status_code=422,
            detail="At least one phenotype is required for diagnosis.",
        )
    # Build input hash for caching
    input_dict = case.model_dump()
    input_hash = cache.hash_case_input(input_dict)

    # Check cache
    try:
        cached = cache.get_recommend(input_hash)
        if cached:
            logger.info("Cache HIT for %s", input_hash)
            return RecommendResponse(**cached)
    except Exception as e:
        logger.warning("Redis unavailable: %s", e)

    # Run core inference
    present = [p.hpo_id for p in case.phenotypes if p.status == "present"]
    absent = [p.hpo_id for p in case.phenotypes if p.status == "absent"]

    result = engine.recommend(
        present_hpos=present,
        absent_hpos=absent,
        prior_testing=case.prior_testing,
        test_result=case.test_result,
        inheritance_hint=case.inheritance_hint,
        vus_present=case.vus_present,
        gene_results=[g.model_dump() for g in case.gene_results] if case.gene_results else None,
    )

    # Enhanced modules
    uncertainty_resp = _build_uncertainty(case, result)
    voi_actions = _build_voi_actions(case, result)
    genomic_resp = _build_genomic_assessment(case, result)
    evidence_resp = _build_evidence(case, result)

    # Record completeness
    completeness = 0.0
    try:
        from diageno.core.patient_state import PatientState
        patient = PatientState.from_case_input(case.model_dump())
        completeness = patient.record_completeness
    except Exception:
        pass

    run_id = str(uuid.uuid4())
    response = RecommendResponse(
        case_id=case.case_id,
        run_id=run_id,
        model_version=engine.version,
        diseases=result["diseases"],
        next_best_phenotypes=result["next_best_phenotypes"],
        test_recommendations=result["test_recommendations"],
        confidence=result["confidence"],
        inputs_hash=input_hash,
        created_at=datetime.utcnow(),
        scoring_method=result.get("scoring_method", "cosine_similarity"),
        hpo_expansion=result.get("hpo_expansion"),
        gene_integration=result.get("gene_integration"),
        uncertainty=uncertainty_resp,
        voi_actions=[VOIAction(**a) for a in voi_actions],
        genomic_assessment=genomic_resp,
        evidence_explanations=evidence_resp,
        record_completeness=round(completeness, 3),
    )

    # Cache result
    try:
        cache.cache_recommend(input_hash, response.model_dump(mode="json"))
        cache.cache_disease_top(present, result["diseases"])
    except Exception as e:
        logger.warning("Cache write failed: %s", e)

    return response

"""/evaluate endpoint — run research evaluation suite."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from diageno.api.schemas import (
    RunEvaluationRequest,
    EvaluationSuiteResponse,
    ExperimentResultResponse,
)
from diageno.api.services.inference import engine

logger = logging.getLogger("diageno.api.routes.evaluate")
router = APIRouter()

# Cache the latest evaluation result
_latest_evaluation: dict | None = None


@router.post("/evaluate", response_model=EvaluationSuiteResponse)
def run_evaluation(req: RunEvaluationRequest) -> EvaluationSuiteResponse:
    """Run research-grade evaluation suite.

    Runs selected experiments: replay, missingness, calibration, ablation, rubric.
    """
    global _latest_evaluation

    workspace = Path(__file__).resolve().parent.parent.parent.parent

    from diageno.evaluation.experiments import (
        experiment_retrospective_replay,
        experiment_missingness_robustness,
        experiment_calibration,
        experiment_ablation,
        experiment_clinician_rubric,
        EvaluationSuite,
    )

    suite = EvaluationSuite()
    suite.headline_claim = (
        "Diageno reduces steps-to-correct-diagnosis vs baselines "
        "while maintaining calibrated uncertainty and safe recommendations."
    )

    exp_map = {
        "replay": lambda: experiment_retrospective_replay(workspace, engine),
        "retrospective_replay": lambda: experiment_retrospective_replay(workspace, engine),
        "missingness": lambda: experiment_missingness_robustness(workspace, engine),
        "missingness_robustness": lambda: experiment_missingness_robustness(workspace, engine),
        "calibration": lambda: experiment_calibration(engine),
        "ablation": lambda: experiment_ablation(workspace, engine),
        "rubric": lambda: experiment_clinician_rubric(workspace, engine),
        "clinician_rubric": lambda: experiment_clinician_rubric(workspace, engine),
    }

    for exp_name in req.experiments:
        if exp_name in exp_map:
            logger.info("Running experiment: %s", exp_name)
            try:
                result = exp_map[exp_name]()
                suite.experiments.append(result)
            except Exception as e:
                logger.error("Experiment %s failed: %s", exp_name, e)
                from diageno.evaluation.experiments import ExperimentResult
                suite.experiments.append(ExperimentResult(
                    experiment_name=exp_name,
                    description=f"Failed: {str(e)}",
                ))

    # Build primary/secondary metrics from results
    for exp in suite.experiments:
        if exp.experiment_name == "1_retrospective_replay" and exp.metrics.get("model"):
            suite.primary_metric = {
                "name": "Steps-to-Correct-Diagnosis",
                "model_value": exp.metrics["model"].get("mean_steps"),
            }
        if exp.experiment_name == "3_calibration":
            suite.secondary_metrics["brier_score"] = exp.metrics.get("brier_score")
            suite.secondary_metrics["ece"] = exp.metrics.get("ece")
        if exp.experiment_name == "2_missingness_robustness":
            suite.secondary_metrics["robustness_30pct"] = exp.metrics.get("confidence_at_30pct_drop")
        if exp.experiment_name == "5_clinician_rubric":
            suite.secondary_metrics["clinician_rubric_pct"] = exp.metrics.get("mean_percentage")

    result_dict = suite.to_dict()
    _latest_evaluation = result_dict

    return EvaluationSuiteResponse(
        headline_claim=suite.headline_claim,
        primary_metric=suite.primary_metric,
        secondary_metrics=suite.secondary_metrics,
        experiments=[
            ExperimentResultResponse(**e.to_dict())
            for e in suite.experiments
        ],
    )


@router.get("/evaluate/latest", response_model=EvaluationSuiteResponse)
def get_latest_evaluation() -> EvaluationSuiteResponse:
    """Get the latest cached evaluation result."""
    if _latest_evaluation is None:
        return EvaluationSuiteResponse(
            headline_claim="No evaluation has been run yet.",
        )
    return EvaluationSuiteResponse(**_latest_evaluation)

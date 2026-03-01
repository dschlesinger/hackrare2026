"""/simulate_step endpoint — show how adding/removing a phenotype changes ranking."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from diageno.api.schemas import SimulateStepInput, SimulateStepResponse, DiseaseCandidate
from diageno.api.services.inference import engine

logger = logging.getLogger("diageno.api.routes.simulate")
router = APIRouter()


@router.post("/simulate_step", response_model=SimulateStepResponse)
def simulate_step(req: SimulateStepInput) -> SimulateStepResponse:
    """Simulate adding or removing a phenotype and show rank changes.

    Steps:
      1. Score with current phenotypes → before_top5
      2. Apply the action (add/remove phenotype)
      3. Score again → after_top5
      4. Compute rank changes
    """
    case = req.case
    present_before = [p.hpo_id for p in case.phenotypes if p.status == "present"]
    absent_before = [p.hpo_id for p in case.phenotypes if p.status == "absent"]

    # Before scoring
    before = engine.recommend(present_hpos=present_before, absent_hpos=absent_before)
    before_top5 = before["diseases"][:5]

    # Apply action
    present_after = list(present_before)
    absent_after = list(absent_before)

    if req.action == "add":
        if req.new_phenotype.status == "present":
            present_after.append(req.new_phenotype.hpo_id)
        elif req.new_phenotype.status == "absent":
            absent_after.append(req.new_phenotype.hpo_id)
    elif req.action == "remove":
        present_after = [h for h in present_after if h != req.new_phenotype.hpo_id]
        absent_after = [h for h in absent_after if h != req.new_phenotype.hpo_id]

    # After scoring
    after = engine.recommend(present_hpos=present_after, absent_hpos=absent_after)
    after_top5 = after["diseases"][:5]

    # Compute rank changes
    before_rank = {d["disease_id"]: i + 1 for i, d in enumerate(before["diseases"])}
    after_rank = {d["disease_id"]: i + 1 for i, d in enumerate(after["diseases"])}

    all_ids = set(before_rank.keys()) | set(after_rank.keys())
    rank_changes = []
    for did in all_ids:
        br = before_rank.get(did, len(before_rank) + 1)
        ar = after_rank.get(did, len(after_rank) + 1)
        if br != ar and (br <= 10 or ar <= 10):
            rank_changes.append({
                "disease_id": did,
                "name": engine.disease_names.get(did, did),
                "rank_before": br,
                "rank_after": ar,
                "change": br - ar,  # positive = improved
            })

    rank_changes.sort(key=lambda x: abs(x["change"]), reverse=True)

    return SimulateStepResponse(
        before_top5=[DiseaseCandidate(**d) for d in before_top5],
        after_top5=[DiseaseCandidate(**d) for d in after_top5],
        rank_changes=rank_changes[:20],
    )

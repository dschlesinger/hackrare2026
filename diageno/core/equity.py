"""Equity and Missingness Robustness Layer.

- Simulate low-resource / fragmented records by dropping phenotype/test fields
- Measure degradation gracefully
- Add missingness-aware fallback policy so sparse records still get safe actions
- Track subgroup deltas by age, sex, record completeness, note-language proxy
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("diageno.core.equity")


@dataclass
class SubgroupMetrics:
    """Performance metrics for a subgroup."""
    subgroup: str
    n_cases: int = 0
    mean_confidence: float = 0.0
    mean_top1_score: float = 0.0
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    mean_rank: float = 0.0
    mrr: float = 0.0

    def to_dict(self) -> dict:
        return {
            "subgroup": self.subgroup,
            "n_cases": self.n_cases,
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_top1_score": round(self.mean_top1_score, 4),
            "hit_at_1": round(self.hit_at_1, 4),
            "hit_at_5": round(self.hit_at_5, 4),
            "hit_at_10": round(self.hit_at_10, 4),
            "mean_rank": round(self.mean_rank, 1),
            "mrr": round(self.mrr, 4),
        }


@dataclass
class RobustnessResult:
    """Result of a missingness robustness test."""
    drop_fraction: float
    n_trials: int
    original_confidence: float
    degraded_confidence: float
    confidence_delta: float
    original_top1_rank: int
    degraded_top1_rank_mean: float
    rank_stability: float          # fraction of trials where top-1 stayed the same
    top5_stability: float          # fraction where correct answer stayed in top-5
    safe_actions_maintained: float  # fraction of safe actions that survived degradation

    def to_dict(self) -> dict:
        return {
            "drop_fraction": self.drop_fraction,
            "n_trials": self.n_trials,
            "original_confidence": round(self.original_confidence, 4),
            "degraded_confidence": round(self.degraded_confidence, 4),
            "confidence_delta": round(self.confidence_delta, 4),
            "original_top1_rank": self.original_top1_rank,
            "degraded_top1_rank_mean": round(self.degraded_top1_rank_mean, 1),
            "rank_stability": round(self.rank_stability, 3),
            "top5_stability": round(self.top5_stability, 3),
            "safe_actions_maintained": round(self.safe_actions_maintained, 3),
        }


@dataclass
class FairnessReport:
    """Full equity and fairness analysis."""
    subgroup_metrics: list[SubgroupMetrics] = field(default_factory=list)
    robustness_curve: list[RobustnessResult] = field(default_factory=list)
    max_disparity: float = 0.0    # max difference in hit@5 across subgroups
    fairness_warning: str = ""

    def to_dict(self) -> dict:
        return {
            "subgroup_metrics": [s.to_dict() for s in self.subgroup_metrics],
            "robustness_curve": [r.to_dict() for r in self.robustness_curve],
            "max_disparity": round(self.max_disparity, 4),
            "fairness_warning": self.fairness_warning,
        }


# ── Missingness Simulation ───────────────────────────

def simulate_missingness(
    patient_state: Any,
    recommend_fn: Callable,
    drop_fractions: list[float] | None = None,
    n_trials: int = 5,
    seed: int = 42,
) -> list[RobustnessResult]:
    """Simulate degraded records by dropping phenotypes and measure impact.

    Args:
        patient_state: PatientState object
        recommend_fn: function(PatientState) → dict with 'diseases', 'confidence', 'test_recommendations'
        drop_fractions: list of fractions to drop (default: [0.3, 0.5, 0.7])
        n_trials: number of random trials per fraction
        seed: random seed for reproducibility
    """
    if drop_fractions is None:
        drop_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Get baseline result
    baseline = recommend_fn(patient_state)
    baseline_conf = baseline.get("confidence", 0)
    baseline_diseases = baseline.get("diseases", [])
    baseline_top1 = baseline_diseases[0]["disease_id"] if baseline_diseases else ""
    baseline_actions = {t["action"] for t in baseline.get("test_recommendations", [])}

    results = []
    rng = random.Random(seed)

    for frac in drop_fractions:
        trial_confs = []
        trial_ranks = []
        trial_stable = 0
        trial_top5_stable = 0
        trial_actions_kept = []

        for trial in range(n_trials):
            # Drop phenotypes
            degraded = patient_state.drop_phenotypes(frac, rng=rng)
            if degraded.n_present == 0:
                continue

            try:
                result = recommend_fn(degraded)
            except Exception:
                continue

            trial_confs.append(result.get("confidence", 0))

            # Check rank stability
            diseases = result.get("diseases", [])
            disease_ids = [d["disease_id"] for d in diseases]
            if baseline_top1 in disease_ids:
                rank = disease_ids.index(baseline_top1) + 1
            else:
                rank = len(disease_ids) + 1

            trial_ranks.append(rank)
            if rank == 1:
                trial_stable += 1
            if rank <= 5:
                trial_top5_stable += 1

            # Check action stability
            trial_actions = {t["action"] for t in result.get("test_recommendations", [])}
            if baseline_actions:
                kept = len(baseline_actions & trial_actions) / len(baseline_actions)
            else:
                kept = 1.0
            trial_actions_kept.append(kept)

        if not trial_confs:
            continue

        results.append(RobustnessResult(
            drop_fraction=frac,
            n_trials=len(trial_confs),
            original_confidence=baseline_conf,
            degraded_confidence=float(np.mean(trial_confs)),
            confidence_delta=float(np.mean(trial_confs)) - baseline_conf,
            original_top1_rank=1,
            degraded_top1_rank_mean=float(np.mean(trial_ranks)) if trial_ranks else 0,
            rank_stability=trial_stable / len(trial_confs) if trial_confs else 0,
            top5_stability=trial_top5_stable / len(trial_confs) if trial_confs else 0,
            safe_actions_maintained=float(np.mean(trial_actions_kept)) if trial_actions_kept else 0,
        ))

    return results


# ── Missingness-Aware Fallback Policy ── ──────────────

SAFE_FALLBACK_ACTIONS = [
    {
        "rank": 1,
        "action_type": "referral",
        "action": "Refer to clinical genetics for comprehensive evaluation",
        "rationale": "Sparse clinical data. A clinical geneticist can perform thorough dysmorphology exam and directed testing.",
    },
    {
        "rank": 2,
        "action_type": "test",
        "action": "Comprehensive phenotyping (HPO-based clinical assessment)",
        "rationale": "Record is incomplete. Structured phenotyping will improve diagnostic accuracy before genetic testing.",
    },
    {
        "rank": 3,
        "action_type": "test",
        "action": "Broad gene panel or clinical exome if phenotype remains non-specific",
        "rationale": "When phenotype data is limited, broader unbiased testing may be more productive than targeted panels.",
    },
]


def apply_fallback_policy(
    result: dict,
    patient_state: Any,
    completeness_threshold: float = 0.3,
) -> dict:
    """Apply missingness-aware fallback policy for sparse records.

    If record completeness is below threshold, inject safe fallback
    actions and add missingness warnings.
    """
    completeness = patient_state.record_completeness

    if completeness >= completeness_threshold:
        result["missingness_warning"] = None
        return result

    # Add fallback actions
    existing_actions = {t["action"] for t in result.get("test_recommendations", [])}
    fallback_added = []
    for fb in SAFE_FALLBACK_ACTIONS:
        if fb["action"] not in existing_actions:
            fallback_added.append(fb)

    # Prepend fallback actions
    test_recs = result.get("test_recommendations", [])
    combined = fallback_added + test_recs
    for i, rec in enumerate(combined):
        rec["rank"] = i + 1
    result["test_recommendations"] = combined

    # Add warning
    result["missingness_warning"] = (
        f"Record completeness: {completeness:.0%}. "
        f"Recommendations include safety-net actions due to limited clinical data. "
        f"Please provide additional phenotype information to improve accuracy."
    )

    # Reduce confidence to reflect uncertainty from sparse data
    if completeness < 0.2:
        result["confidence"] = min(result.get("confidence", 0), 0.15)
    elif completeness < 0.3:
        result["confidence"] = min(result.get("confidence", 0), 0.30)

    return result


# ── Subgroup Analysis ────────────────────────────────

def compute_subgroup_metrics(
    cases: list[dict],
    results: list[dict],
    subgroup_key: str = "sex",
) -> list[SubgroupMetrics]:
    """Compute performance metrics stratified by a subgroup key.

    Args:
        cases: list of case dicts with demographics
        results: list of result dicts (parallel with cases)
        subgroup_key: key to stratify by (sex, age_group, completeness_bin)
    """
    # Group by subgroup
    groups: dict[str, list[tuple[dict, dict]]] = {}

    for case, result in zip(cases, results):
        if "error" in result:
            continue

        # Determine subgroup value
        if subgroup_key == "sex":
            val = case.get("patient", {}).get("sex", case.get("sex", "unknown")) or "unknown"
        elif subgroup_key == "age_group":
            age = case.get("patient", {}).get("age", case.get("age"))
            if age is None:
                val = "unknown"
            elif age < 2:
                val = "infant"
            elif age < 12:
                val = "child"
            elif age < 18:
                val = "adolescent"
            elif age < 65:
                val = "adult"
            else:
                val = "elderly"
        elif subgroup_key == "completeness":
            n_pheno = len(case.get("phenotypes", []))
            if n_pheno < 3:
                val = "sparse (<3 HPOs)"
            elif n_pheno < 6:
                val = "moderate (3-5 HPOs)"
            else:
                val = "rich (6+ HPOs)"
        else:
            val = str(case.get(subgroup_key, "unknown"))

        groups.setdefault(val, []).append((case, result))

    # Compute metrics per subgroup
    metrics = []
    for grp_name, items in sorted(groups.items()):
        confs = [r.get("confidence", 0) for _, r in items]
        scores = [r["diseases"][0]["score"] if r.get("diseases") else 0 for _, r in items]

        m = SubgroupMetrics(
            subgroup=grp_name,
            n_cases=len(items),
            mean_confidence=float(np.mean(confs)) if confs else 0,
            mean_top1_score=float(np.mean(scores)) if scores else 0,
        )
        metrics.append(m)

    return metrics

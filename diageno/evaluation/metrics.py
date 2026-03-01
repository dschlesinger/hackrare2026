"""Evaluation Metrics — Steps-to-Diagnosis, MRR, Calibration, Brier.

Primary metric: Steps-to-correct-diagnosis
Secondary: top-k accuracy, MRR, time-to-action, cost-adjusted gain, Brier, ECE
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger("diageno.evaluation.metrics")


def steps_to_correct_diagnosis(
    ground_truth_actions: list[str],
    recommended_actions: list[dict],
    max_steps: int = 20,
) -> int:
    """Compute steps-to-correct-diagnosis.

    Counts how many recommended actions must be taken before
    reaching the correct next clinical action from ground truth.

    Args:
        ground_truth_actions: list of correct next actions (from decision points)
        recommended_actions: list of recommended action dicts with 'action' key
        max_steps: maximum steps to consider

    Returns:
        Number of steps (1-indexed). max_steps+1 if not found.
    """
    gt_lower = {a.lower().strip() for a in ground_truth_actions}
    for i, rec in enumerate(recommended_actions[:max_steps]):
        rec_action = rec.get("action", "").lower().strip()
        # Fuzzy match: check if GT action is a substring or vice versa
        for gt in gt_lower:
            if gt in rec_action or rec_action in gt:
                return i + 1
            # Token overlap: if >50% of words match
            gt_tokens = set(gt.split())
            rec_tokens = set(rec_action.split())
            if gt_tokens and rec_tokens:
                overlap = len(gt_tokens & rec_tokens) / min(len(gt_tokens), len(rec_tokens))
                if overlap >= 0.5:
                    return i + 1
    return max_steps + 1


def mean_reciprocal_rank(ranks: list[int]) -> float:
    """Mean Reciprocal Rank."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks if r > 0]))


def hits_at_k(ranks: list[int], k: int) -> float:
    """Hits@K — fraction of queries where correct answer is in top K."""
    if not ranks:
        return 0.0
    return float(np.mean([1 if r <= k else 0 for r in ranks]))


def brier_score(predicted_probs: list[float], actual_labels: list[int]) -> float:
    """Brier score — mean squared error of probability estimates.

    Lower is better. Perfect calibration → 0.
    """
    if not predicted_probs:
        return 1.0
    n = len(predicted_probs)
    return float(sum((p - y) ** 2 for p, y in zip(predicted_probs, actual_labels)) / n)


def expected_calibration_error(
    predicted_probs: list[float],
    actual_labels: list[int],
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Expected Calibration Error (ECE) with reliability diagram data.

    Groups predictions into bins by confidence, measures gap between
    predicted probability and actual accuracy per bin.

    Returns:
        (ece_score, bin_data) where bin_data is list of
        {bin_center, avg_confidence, avg_accuracy, n_samples}
    """
    if not predicted_probs:
        return 1.0, []

    probs = np.array(predicted_probs)
    labels = np.array(actual_labels)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (probs == bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_conf = float(probs[mask].mean())
        avg_acc = float(labels[mask].mean())
        ece += (n_in_bin / len(probs)) * abs(avg_acc - avg_conf)

        bin_data.append({
            "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
            "n_samples": int(n_in_bin),
            "gap": round(abs(avg_acc - avg_conf), 4),
        })

    return float(ece), bin_data


def cost_adjusted_gain(
    steps_baseline: list[int],
    steps_model: list[int],
    cost_per_step: float = 1000.0,
) -> dict:
    """Compute cost-adjusted gain of model over baseline.

    Args:
        steps_baseline: steps-to-diagnosis for baseline
        steps_model: steps-to-diagnosis for model
        cost_per_step: estimated cost per diagnostic step

    Returns:
        Dict with mean_steps_saved, cost_saved, percent_improvement
    """
    if not steps_baseline or not steps_model:
        return {"mean_steps_saved": 0, "cost_saved": 0, "percent_improvement": 0}

    mean_bl = float(np.mean(steps_baseline))
    mean_mod = float(np.mean(steps_model))
    saved = mean_bl - mean_mod

    return {
        "baseline_mean_steps": round(mean_bl, 2),
        "model_mean_steps": round(mean_mod, 2),
        "mean_steps_saved": round(saved, 2),
        "cost_saved_per_case": round(saved * cost_per_step, 0),
        "percent_improvement": round(100 * saved / mean_bl, 1) if mean_bl > 0 else 0,
    }


def compute_all_metrics(
    ranked_diseases: list[list[tuple[str, float]]],
    ground_truth_ids: list[str | None],
    confidences: list[float],
    gt_actions: list[list[str]] | None = None,
    recommended_actions: list[list[dict]] | None = None,
) -> dict:
    """Compute comprehensive evaluation metrics.

    Args:
        ranked_diseases: list of ranked disease lists [(disease_id, score), ...]
        ground_truth_ids: list of ground truth disease IDs (or None if unknown)
        confidences: list of confidence scores
        gt_actions: optional ground truth next actions per case
        recommended_actions: optional recommended actions per case
    """
    # Disease ranking metrics
    ranks = []
    probs = []
    labels = []

    for i, (diseases, gt_id) in enumerate(zip(ranked_diseases, ground_truth_ids)):
        if gt_id is None:
            continue

        disease_ids = [d[0] for d in diseases]
        if gt_id in disease_ids:
            rank = disease_ids.index(gt_id) + 1
        else:
            rank = len(disease_ids) + 1
        ranks.append(rank)

        conf = confidences[i] if i < len(confidences) else 0.5
        probs.append(conf)
        labels.append(1 if rank <= 5 else 0)

    # Steps-to-diagnosis
    steps_model = []
    if gt_actions and recommended_actions:
        for gt_a, rec_a in zip(gt_actions, recommended_actions):
            if gt_a:
                steps = steps_to_correct_diagnosis(gt_a, rec_a)
                steps_model.append(steps)

    # Calibration
    ece, rel_bins = expected_calibration_error(probs, labels)

    return {
        "n_cases": len(ranked_diseases),
        "n_with_ground_truth": len(ranks),
        "disease_ranking": {
            "mrr": round(mean_reciprocal_rank(ranks), 4),
            "hits_at_1": round(hits_at_k(ranks, 1), 4),
            "hits_at_3": round(hits_at_k(ranks, 3), 4),
            "hits_at_5": round(hits_at_k(ranks, 5), 4),
            "hits_at_10": round(hits_at_k(ranks, 10), 4),
            "mean_rank": round(float(np.mean(ranks)), 2) if ranks else 0,
        },
        "calibration": {
            "brier_score": round(brier_score(probs, labels), 4),
            "ece": round(ece, 4),
            "reliability_bins": rel_bins,
        },
        "steps_to_diagnosis": {
            "mean_steps": round(float(np.mean(steps_model)), 2) if steps_model else None,
            "median_steps": round(float(np.median(steps_model)), 2) if steps_model else None,
            "n_cases_evaluated": len(steps_model),
        },
        "confidence_stats": {
            "mean": round(float(np.mean(confidences)), 4) if confidences else 0,
            "std": round(float(np.std(confidences)), 4) if confidences else 0,
            "min": round(float(np.min(confidences)), 4) if confidences else 0,
            "max": round(float(np.max(confidences)), 4) if confidences else 0,
        },
    }

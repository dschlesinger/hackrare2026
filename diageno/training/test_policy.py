"""Artifact C — Test Recommendation Policy.

Rule-based policy that recommends tests conditional on:
  - Prior testing summary (none / panel / exome / WGS; VUS present)
  - Inheritance hint (AD, AR, XL, mito, de novo, unknown)
  - Confidence level of current top-disease candidate

Optionally calibrates thresholds using replay data.

Outputs: policy.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from diageno.config import settings

logger = logging.getLogger("diageno.training.test_policy")

# ─── Default policy rules ────────────────────────────

DEFAULT_POLICY: dict[str, Any] = {
    "version": "1.0.0",
    "description": "Test recommendation policy for rare disease diagnostic workup",
    "confidence_thresholds": {
        "high": 0.7,
        "medium": 0.3,
        "low": 0.0,
    },
    "rules": [
        # ── No prior genetic testing ──
        {
            "condition": {
                "prior_testing": "none",
                "confidence": "any",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "action_type": "test",
                    "action": "Targeted gene panel based on top phenotype cluster",
                    "rationale": "Gene panel is first-line for phenotypically suggestive presentations with no prior genetic workup.",
                },
                {
                    "rank": 2,
                    "action_type": "test",
                    "action": "Clinical exome sequencing (if panel negative or broad differential)",
                    "rationale": "Exome provides broader coverage when panel is insufficient.",
                },
            ],
        },
        # ── Panel done, negative ──
        {
            "condition": {
                "prior_testing": "panel",
                "result": "negative",
                "confidence": "any",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "action_type": "test",
                    "action": "Clinical exome sequencing (trio if available)",
                    "rationale": "Exome after negative panel; trio increases de novo detection.",
                },
                {
                    "rank": 2,
                    "action_type": "referral",
                    "action": "Refer to clinical genetics for phenotype re-evaluation",
                    "rationale": "Re-phenotyping may reveal features that redirect workup.",
                },
            ],
        },
        # ── Exome done, negative ──
        {
            "condition": {
                "prior_testing": "exome",
                "result": "negative",
                "confidence": "any",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "action_type": "test",
                    "action": "Genome sequencing (WGS — short or long-read)",
                    "rationale": "WGS captures structural variants, non-coding regions missed by exome.",
                },
                {
                    "rank": 2,
                    "action_type": "reanalysis",
                    "action": "Reanalysis of existing exome data with updated annotations",
                    "rationale": "~15% additional diagnoses from exome reanalysis at 1–3 year intervals.",
                },
                {
                    "rank": 3,
                    "action_type": "test",
                    "action": "RNA-seq (if tissue accessible) or methylation array",
                    "rationale": "Functional assays for splice/expression effects.",
                },
            ],
        },
        # ── WGS done, negative ──
        {
            "condition": {
                "prior_testing": "wgs",
                "result": "negative",
                "confidence": "any",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "action_type": "reanalysis",
                    "action": "Periodic reanalysis (12–18 month intervals)",
                    "rationale": "New gene-disease associations discovered continuously.",
                },
                {
                    "rank": 2,
                    "action_type": "referral",
                    "action": "Refer to Undiagnosed Disease Program / research study",
                    "rationale": "Multi-omics and research pipelines for unsolved cases.",
                },
                {
                    "rank": 3,
                    "action_type": "monitoring",
                    "action": "Longitudinal phenotyping + biomarker monitoring",
                    "rationale": "Evolving phenotype may clarify diagnosis over time.",
                },
            ],
        },
        # ── VUS present ──
        {
            "condition": {
                "vus_present": True,
                "confidence": "any",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "action_type": "test",
                    "action": "Segregation analysis (parental / family testing)",
                    "rationale": "Determine inheritance pattern of VUS.",
                },
                {
                    "rank": 2,
                    "action_type": "test",
                    "action": "Functional assay if available for this gene/variant",
                    "rationale": "Functional evidence can reclassify VUS → likely pathogenic.",
                },
                {
                    "rank": 3,
                    "action_type": "monitoring",
                    "action": "ClinVar / literature watch for variant reclassification",
                    "rationale": "VUS may be reclassified as new evidence accrues.",
                },
            ],
        },
        # ── High confidence, specific inheritance ──
        {
            "condition": {
                "confidence": "high",
                "inheritance_hint": "autosomal_recessive",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "action_type": "test",
                    "action": "Carrier testing for parents + at-risk family members",
                    "rationale": "Confirm recessive inheritance and enable family planning.",
                },
            ],
        },
        {
            "condition": {
                "confidence": "high",
                "inheritance_hint": "mitochondrial",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "action_type": "test",
                    "action": "Mitochondrial genome sequencing (+ heteroplasmy quantification)",
                    "rationale": "Mitochondrial variants require specialized sequencing.",
                },
                {
                    "rank": 2,
                    "action_type": "test",
                    "action": "Muscle biopsy for respiratory chain enzyme analysis",
                    "rationale": "Direct functional assessment of mitochondrial function.",
                },
            ],
        },
    ],
}


def build_policy(artifacts: Path) -> None:
    """Write the default policy YAML to artifacts directory."""
    artifacts.mkdir(parents=True, exist_ok=True)
    policy_path = artifacts / "policy.yaml"
    with open(policy_path, "w") as f:
        yaml.dump(DEFAULT_POLICY, f, default_flow_style=False, sort_keys=False, width=120)
    logger.info("Policy written → %s", policy_path)


def load_policy(artifacts: Path) -> dict:
    """Load the policy from YAML."""
    policy_path = artifacts / "policy.yaml"
    with open(policy_path) as f:
        return yaml.safe_load(f)


def match_rules(
    policy: dict,
    prior_testing: str = "none",
    test_result: str | None = None,
    inheritance_hint: str | None = None,
    confidence: float = 0.0,
    vus_present: bool = False,
) -> list[dict]:
    """Match applicable policy rules and return merged recommendations.

    Args:
        policy: loaded policy dict
        prior_testing: "none" | "panel" | "exome" | "wgs"
        test_result: "negative" | "positive" | "vus" | None
        inheritance_hint: e.g. "autosomal_recessive", "autosomal_dominant", etc.
        confidence: 0.0–1.0 top-disease confidence score
        vus_present: whether a VUS was found

    Returns:
        List of recommendation dicts, de-duplicated and ranked.
    """
    thresholds = policy.get("confidence_thresholds", {})
    conf_level = "low"
    if confidence >= thresholds.get("high", 0.7):
        conf_level = "high"
    elif confidence >= thresholds.get("medium", 0.3):
        conf_level = "medium"

    matched: list[dict] = []

    for rule in policy.get("rules", []):
        cond = rule.get("condition", {})

        # Check prior_testing
        if "prior_testing" in cond and cond["prior_testing"] != prior_testing:
            continue

        # Check result
        if "result" in cond and cond["result"] != test_result:
            continue

        # Check VUS
        if "vus_present" in cond and cond["vus_present"] != vus_present:
            continue

        # Check confidence
        cond_conf = cond.get("confidence", "any")
        if cond_conf != "any" and cond_conf != conf_level:
            continue

        # Check inheritance
        if "inheritance_hint" in cond:
            if cond["inheritance_hint"] != inheritance_hint:
                continue

        matched.extend(rule.get("recommendations", []))

    # De-duplicate by action text
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in matched:
        key = rec["action"]
        if key not in seen:
            seen.add(key)
            unique.append(rec)

    # Re-rank
    for i, rec in enumerate(unique, 1):
        rec["rank"] = i

    return unique


def run() -> None:
    """Build policy artifact."""
    artifacts = settings.artifacts_dir
    build_policy(artifacts)
    logger.info("=== Artifact C (Test Policy) complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()

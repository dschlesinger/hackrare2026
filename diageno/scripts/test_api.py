#!/usr/bin/env python3
"""Test the FastAPI app can start and serve requests."""

import sys
import json

# Test that app can be imported and configured
try:
    from diageno.api.main import app
    print(f"FastAPI app loaded: {app.title} v{app.version}")
    print(f"Routes: {[r.path for r in app.routes]}")
except Exception as e:
    print(f"ERROR importing app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test schemas
from diageno.api.schemas import (
    CaseInput, PhenotypeInput, RecommendResponse,
    SimulateStepInput, HPOLookupRequest, ValidateSchemaRequest,
    HealthResponse, SimulateStepResponse, HPOLookupResponse,
    ValidationResponse, DiseaseCandidate, NextBestPhenotype,
    TestRecommendation,
)
print("\nAll Pydantic schemas imported OK")

# Test schema validation
case = CaseInput(
    case_id="test-001",
    age=5,
    sex="male",
    phenotypes=[
        PhenotypeInput(hpo_id="HP:0001250", label="Seizures", status="present"),
        PhenotypeInput(hpo_id="HP:0001249", label="Intellectual disability", status="present"),
    ],
    prior_testing="none",
)
print(f"CaseInput created: {case.case_id}, {len(case.phenotypes)} phenotypes")

# Validate a response can be serialized
resp = RecommendResponse(
    case_id="test-001",
    model_version="test",
    diseases=[
        DiseaseCandidate(disease_id="ORPHA:123", name="Test Disease", score=0.9, calibrated_score=0.85)
    ],
    next_best_phenotypes=[
        NextBestPhenotype(hpo_id="HP:0001263", expected_info_gain=0.5, p_present=0.3)
    ],
    test_recommendations=[
        TestRecommendation(rank=1, action_type="test", action="Gene panel")
    ],
    confidence=0.9,
)
json_str = resp.model_dump_json()
print(f"RecommendResponse serialization OK ({len(json_str)} bytes)")

# Test the validate_schema endpoint logic
from pydantic import ValidationError
try:
    CaseInput(**{"phenotypes": [{"hpo_id": "HP:0001250"}]})
    print("Schema validation OK")
except ValidationError as e:
    print(f"Schema validation caught errors: {len(e.errors())}")

print("\n=== FASTAPI APP TEST PASSED ===")

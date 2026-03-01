#!/usr/bin/env python3
"""End-to-end API endpoint test suite."""

import json
import sys
import time
import httpx

BASE = "http://127.0.0.1:8099"

def test(name, method, path, payload=None, expected_status=200):
    try:
        if method == "GET":
            resp = httpx.get(f"{BASE}{path}", timeout=30)
        else:
            resp = httpx.post(f"{BASE}{path}", json=payload, timeout=30)
        
        if resp.status_code == expected_status:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            print(f"  PASS: {name} [{resp.status_code}]")
            return data
        else:
            print(f"  FAIL: {name} [{resp.status_code}] expected {expected_status}")
            print(f"    Body: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  FAIL: {name} - {e}")
        return None

print("Waiting for server...")
for i in range(15):
    try:
        httpx.get(f"{BASE}/docs", timeout=2)
        break
    except Exception:
        time.sleep(1)

passed = 0
failed = 0

# 1. Health (will be degraded without DB/Redis but should return)
print("\n=== /health ===")
result = test("Health check", "GET", "/health")
if result:
    print(f"    status={result.get('status')}, model_loaded={result.get('model_loaded')}")
    passed += 1
else:
    failed += 1

# 2. Recommend
print("\n=== /recommend ===")
result = test("Recommend", "POST", "/recommend", {
    "phenotypes": [
        {"hpo_id": "HP:0001250", "status": "present"},
        {"hpo_id": "HP:0001249", "status": "present"},
        {"hpo_id": "HP:0000252", "status": "present"},
    ],
    "prior_testing": "panel",
    "test_result": "negative",
})
if result:
    print(f"    diseases: {len(result.get('diseases', []))}")
    print(f"    next_phenos: {len(result.get('next_best_phenotypes', []))}")
    print(f"    test_recs: {len(result.get('test_recommendations', []))}")
    print(f"    confidence: {result.get('confidence')}")
    passed += 1
else:
    failed += 1

# 3. Simulate step
print("\n=== /simulate_step ===")
result = test("Simulate step", "POST", "/simulate_step", {
    "case": {
        "phenotypes": [{"hpo_id": "HP:0001250", "status": "present"}],
        "prior_testing": "none",
    },
    "new_phenotype": {"hpo_id": "HP:0001249", "status": "present"},
    "action": "add",
})
if result:
    print(f"    before_top5: {len(result.get('before_top5', []))}")
    print(f"    after_top5: {len(result.get('after_top5', []))}")
    print(f"    rank_changes: {len(result.get('rank_changes', []))}")
    passed += 1
else:
    failed += 1

# 4. Validate schema (valid)
print("\n=== /validate_schema (valid) ===")
result = test("Validate - valid", "POST", "/validate_schema", {
    "data": {"phenotypes": [{"hpo_id": "HP:0001250"}]}
})
if result and result.get("valid"):
    passed += 1
else:
    failed += 1

# 5. Validate schema (invalid)
print("\n=== /validate_schema (invalid) ===")
result = test("Validate - invalid", "POST", "/validate_schema", {
    "data": {"phenotypes": "not_a_list"}
})
if result and not result.get("valid"):
    print(f"    errors: {result.get('errors', [])}")
    passed += 1
else:
    failed += 1

# 6. Metrics
print("\n=== /metrics ===")
resp = httpx.get(f"{BASE}/metrics", timeout=10)
if resp.status_code == 200 and "diageno_api_requests_total" in resp.text:
    print(f"  PASS: Metrics endpoint [200] - Prometheus metrics present")
    passed += 1
else:
    print(f"  FAIL: Metrics endpoint")
    failed += 1

# 7. OpenAPI schema
print("\n=== /openapi.json ===")
result = test("OpenAPI schema", "GET", "/openapi.json")
if result and "paths" in result:
    print(f"    paths: {list(result['paths'].keys())}")
    passed += 1
else:
    failed += 1

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("=== ALL TESTS PASSED ===")
else:
    print("=== SOME TESTS FAILED ===")
    sys.exit(1)

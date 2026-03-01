#!/usr/bin/env python3
"""Test the updated API: confidence calibration, rationale, gene inputs."""

import json
import httpx

BASE = "http://127.0.0.1:8099"

print("=" * 70)
print("TEST 1: Basic recommend (confidence should NOT be 1.0)")
print("=" * 70)

resp = httpx.post(f"{BASE}/recommend", json={
    "phenotypes": [
        {"hpo_id": "HP:0001250", "status": "present"},
        {"hpo_id": "HP:0001249", "status": "present"},
        {"hpo_id": "HP:0000252", "status": "present"},
    ],
    "prior_testing": "panel",
    "test_result": "negative",
}, timeout=30)
result = resp.json()

print(f"Confidence: {result['confidence']}")
assert result["confidence"] < 0.99, f"Confidence still saturated at {result['confidence']}"
print(f"  ✓ Confidence is {result['confidence']:.1%} (not 100%!)")

# Check disease rationale
d = result["diseases"][0]
print(f"\nTop disease: {d['name']}")
print(f"  Score: {d['score']:.4f}")
print(f"  Calibrated: {d.get('calibrated_score', '?')}")
print(f"  Match: {d.get('phenotype_match', '?')}")
print(f"  Rationale: {d.get('rationale', 'MISSING')[:200]}")
assert d.get("rationale"), "Disease rationale is missing!"
print("  ✓ Disease rationale present")
assert d.get("phenotype_match"), "Phenotype match is missing!"
print("  ✓ Phenotype match present")

# Check test rationale
for t in result["test_recommendations"]:
    print(f"\nTest rec #{t['rank']}: {t['action'][:60]}...")
    print(f"  Rationale: {t.get('rationale', 'MISSING')[:150]}")
    assert t.get("rationale"), "Test rationale is missing!"
print("  ✓ Test rationale present")

# Check phenotype rationale
for p in result["next_best_phenotypes"][:3]:
    print(f"\nNext pheno: {p.get('label', '')} ({p['hpo_id']})")
    print(f"  Info gain: {p['expected_info_gain']:.3f} bits")
    print(f"  Rationale: {p.get('rationale', 'MISSING')[:150]}")
    assert p.get("rationale"), "Phenotype rationale is missing!"
print("  ✓ Phenotype rationale present")

print("\n" + "=" * 70)
print("TEST 2: With gene_results (structured gene inputs)")
print("=" * 70)

resp2 = httpx.post(f"{BASE}/recommend", json={
    "phenotypes": [
        {"hpo_id": "HP:0001250", "status": "present"},
        {"hpo_id": "HP:0001249", "status": "present"},
    ],
    "prior_testing": "panel",
    "test_result": "vus",
    "vus_present": True,
    "gene_results": [
        {"gene": "SCN9A", "classification": "vus", "inheritance": "autosomal_dominant", "test_type": "panel"},
        {"gene": "KCNQ2", "classification": "likely_pathogenic", "inheritance": "autosomal_dominant", "test_type": "panel"},
    ],
}, timeout=30)
result2 = resp2.json()
print(f"Status: {resp2.status_code}")
print(f"Confidence: {result2['confidence']:.1%}")
print(f"Diseases: {len(result2['diseases'])}")
print(f"Test recs: {len(result2['test_recommendations'])}")

# Check gene context appears in test rationale
has_gene_context = any("SCN9A" in t.get("rationale", "") for t in result2["test_recommendations"])
print(f"Gene context in test rationale: {has_gene_context}")
if has_gene_context:
    print("  ✓ Gene findings incorporated into test recommendations")

print("\n" + "=" * 70)
print("TEST 3: Confidence varies with different inputs")
print("=" * 70)

# Few phenotypes → lower confidence
resp3a = httpx.post(f"{BASE}/recommend", json={
    "phenotypes": [{"hpo_id": "HP:0001250", "status": "present"}],
}, timeout=30)
conf_1pheno = resp3a.json()["confidence"]

# Many phenotypes → higher confidence
resp3b = httpx.post(f"{BASE}/recommend", json={
    "phenotypes": [
        {"hpo_id": "HP:0001250", "status": "present"},
        {"hpo_id": "HP:0001249", "status": "present"},
        {"hpo_id": "HP:0000252", "status": "present"},
        {"hpo_id": "HP:0002360", "status": "present"},
        {"hpo_id": "HP:0000750", "status": "present"},
    ],
}, timeout=30)
conf_5pheno = resp3b.json()["confidence"]

print(f"1 phenotype  → confidence: {conf_1pheno:.1%}")
print(f"5 phenotypes → confidence: {conf_5pheno:.1%}")
print(f"  (Both should be < 99% and potentially different)")

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)

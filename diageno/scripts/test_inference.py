#!/usr/bin/env python3
"""Test the inference engine end-to-end with sample HPO terms."""

import sys
sys.path.insert(0, "/Users/bhargav/Desktop/hackrare2026/diageno")

from diageno.api.services.inference import InferenceEngine

eng = InferenceEngine()
eng.load()
print(f"Loaded: {eng.is_loaded}")
print(f"Diseases: {len(eng.disease_index)}")
print(f"HPOs: {len(eng.hpo_dict)}")
print(f"Has calibrator: {eng.calibrator is not None}")
print(f"Policy rules: {len(eng.policy.get('rules', []))}")

# Test recommend with seizures + intellectual disability + microcephaly
result = eng.recommend(
    present_hpos=["HP:0001250", "HP:0001249", "HP:0000252"],
    absent_hpos=["HP:0001263"],
    prior_testing="none",
)
print(f"\nTop 5 diseases:")
for d in result["diseases"][:5]:
    print(f"  {d['disease_id']}: {d['name']} (score={d['score']}, cal={d['calibrated_score']})")

print(f"\nNext best phenotypes: {len(result['next_best_phenotypes'])}")
if result["next_best_phenotypes"]:
    for p in result["next_best_phenotypes"][:3]:
        print(f"  {p['hpo_id']}: gain={p['expected_info_gain']}")

print(f"\nTest recommendations: {len(result['test_recommendations'])}")
for t in result["test_recommendations"]:
    print(f"  [{t['rank']}] {t['action_type']}: {t['action'][:70]}")

print(f"\nConfidence: {result['confidence']}")
print("\n=== INFERENCE ENGINE TEST PASSED ===")

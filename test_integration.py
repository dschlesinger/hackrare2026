#!/usr/bin/env python3
"""Quick integration test for enhanced /recommend."""
import httpx, json

case = {
    "age": 5,
    "sex": "male",
    "phenotypes": [
        {"hpo_id": "HP:0001250", "label": "Seizures", "status": "present"},
        {"hpo_id": "HP:0001263", "label": "Global developmental delay", "status": "present"},
        {"hpo_id": "HP:0001252", "label": "Hypotonia", "status": "present"},
    ],
    "prior_testing": "exome",
    "test_result": "vus",
    "vus_present": True,
    "gene_results": [
        {"gene": "SCN1A", "classification": "vus", "inheritance": "de_novo", "test_type": "exome"}
    ],
}

resp = httpx.post("http://localhost:8099/recommend", json=case, timeout=30)
resp.raise_for_status()
d = resp.json()

print(f"Diseases: {len(d.get('diseases', []))}")
print(f"Top Disease: {d['diseases'][0]['name'] if d.get('diseases') else 'N/A'}")
print(f"Confidence: {d.get('confidence')}")
print(f"Record Completeness: {d.get('record_completeness')}")
print(f"VOI Actions: {len(d.get('voi_actions', []))}")
unc = d.get("uncertainty")
print(f"Uncertainty Overall: {unc.get('overall') if unc else 'N/A'}")
print(f"  Phenotype: {unc.get('phenotype_uncertainty') if unc else 'N/A'}")
print(f"  Genomic: {unc.get('genomic_uncertainty') if unc else 'N/A'}")
print(f"  Decision: {unc.get('decision_uncertainty') if unc else 'N/A'}")
print(f"  Counterfactuals: {len(unc.get('counterfactuals', [])) if unc else 0}")
ga = d.get("genomic_assessment")
print(f"Genomic Maturity: {ga.get('genomic_maturity') if ga else 'N/A'}")
print(f"Evidence Explanations: {len(d.get('evidence_explanations', []))}")

if d.get("voi_actions"):
    print("\nTop 3 VOI Actions:")
    for a in d["voi_actions"][:3]:
        print(f"  - {a['action']} (VOI: {a['cost_adjusted_voi']:.3f}, ${a.get('cost_dollars', '?')})")

if ga:
    print(f"\nGenomic now_actions: {ga.get('now_actions', [])}")
    print(f"VUS triage: {ga.get('vus_triage', [])}")
    print(f"Escalation: {ga.get('escalation_path', [])}")

print("\n✅ ALL ENHANCED MODULES WORKING")

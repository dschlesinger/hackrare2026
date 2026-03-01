#!/usr/bin/env python3
"""Full integration test for all validation cases + evaluate endpoint."""
import requests, json, glob, sys, os

BASE = "http://localhost:8099"

# Test evaluate endpoint
print("=== EVALUATE ENDPOINT ===")
resp = requests.post(f"{BASE}/evaluate", json={"experiments": ["replay"], "n_repeats": 1})
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    d = resp.json()
    exps = d.get("experiments", [])
    if isinstance(exps, list):
        for exp in exps:
            mk = list(exp.get("metrics", {}).keys())[:5]
            print(f"  {exp.get('experiment_name','?')}: metrics={mk}")
    else:
        for name, exp in exps.items():
            print(f"  {name}: metrics={list(exp.get('metrics', {}).keys())[:5]}")
    if not exps:
        print("  (no experiments returned)")
else:
    print(f"  Error: {resp.text[:200]}")

# Test all 5 validation cases
print("\n=== VALIDATION CASES ===")
cases = sorted(glob.glob("ValidationCase[0-9]*"))
all_ok = True
for case_file in cases:
    if not os.path.isfile(case_file):
        continue
    with open(case_file) as f:
        raw = json.load(f)
    phenos = []
    for p in raw.get("phenotypes", []):
        hpo = p.get("hpo_id") or p.get("hpo", "")
        phenos.append({"hpo_id": hpo, "status": p.get("status", "present"), "onset_age": p.get("onset_age")})
    genes = []
    for g in raw.get("gene_results", []):
        genes.append({
            "gene": g.get("gene", ""),
            "variant": g.get("variant", ""),
            "zygosity": g.get("zygosity", "heterozygous"),
            "classification": g.get("classification", "uncertain_significance"),
        })
    patient = raw.get("patient", {})
    body = {
        "phenotypes": phenos,
        "gene_results": genes,
        "prior_testing": raw.get("prior_testing", "none"),
        "age_years": raw.get("age_years") or raw.get("age") or patient.get("age", 5),
        "sex": raw.get("sex") or patient.get("sex", "unknown"),
    }
    r = requests.post(f"{BASE}/recommend", json=body)
    if r.status_code == 200:
        j = r.json()
        top = j["diseases"][0]["name"] if j["diseases"] else "N/A"
        n_voi = len(j.get("voi_actions") or [])
        unc = j.get("uncertainty") or {}
        unc_ov = unc.get("overall", "N/A")
        n_ev = len(j.get("evidence_explanations") or [])
        gen = j.get("genomic_assessment") or {}
        gen_mat = gen.get("genomic_maturity", "N/A")
        gt = raw.get("ground_truth_disease", "N/A")
        ok = n_voi > 0 and unc_ov != "N/A" and n_ev > 0
        status = "OK" if ok else "PARTIAL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {case_file}")
        print(f"    Top1: {top[:50]}")
        print(f"    GT:   {gt[:50]}")
        print(f"    VOI={n_voi}, Unc={unc_ov}, GenMat={gen_mat}, Ev={n_ev}")
    else:
        all_ok = False
        print(f"  [FAIL] {case_file}: HTTP {r.status_code}")

print()
if all_ok:
    print("ALL ENHANCED MODULES WORKING ACROSS ALL CASES")
else:
    print("SOME CASES HAD ISSUES - check above")

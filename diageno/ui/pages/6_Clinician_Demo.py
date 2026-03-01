"""Page 6: Clinician Demo — 4-step guided clinical walkthrough."""

import streamlit as st
import httpx
import plotly.graph_objects as go
import pandas as pd
import json

st.set_page_config(page_title="Clinician Demo", page_icon="🏥", layout="wide")
st.title("🏥 Clinician Demo Flow")

API_URL = st.session_state.get("api_url", "http://localhost:8099")

st.markdown("""
**Four-step guided demonstration** of the diagnostic copilot in action.
Choose a demo scenario or use the current case to walk through the complete pipeline.
""")

# ── Demo Scenarios ────────────────────────────────────
DEMO_SCENARIOS = {
    "sparse_infant": {
        "title": "Sparse Record — Infant with Few Phenotypes",
        "description": "A 6-month-old with global developmental delay and hypotonia — only 2 phenotypes recorded.",
        "case": {
            "age": 0,
            "sex": "female",
            "phenotypes": [
                {"hpo_id": "HP:0001263", "label": "Global developmental delay", "status": "present"},
                {"hpo_id": "HP:0001252", "label": "Hypotonia", "status": "present"},
            ],
            "prior_testing": "none",
            "gene_results": [],
        },
    },
    "genomic_vus": {
        "title": "Genomic-Heavy — Exome with VUS",
        "description": "A 5-year-old with seizures, exome found VUS in SCN1A. Classic genetic uncertainty case.",
        "case": {
            "age": 5,
            "sex": "male",
            "phenotypes": [
                {"hpo_id": "HP:0001250", "label": "Seizures", "status": "present"},
                {"hpo_id": "HP:0001263", "label": "Global developmental delay", "status": "present"},
                {"hpo_id": "HP:0000750", "label": "Delayed speech and language development", "status": "present"},
                {"hpo_id": "HP:0001252", "label": "Hypotonia", "status": "present"},
            ],
            "prior_testing": "exome",
            "test_result": "vus",
            "vus_present": True,
            "gene_results": [
                {"gene": "SCN1A", "classification": "vus", "inheritance": "de_novo", "test_type": "exome"},
            ],
        },
    },
    "negative_exome": {
        "title": "Diagnostic Odyssey — Negative Exome",
        "description": "A 3-year-old with complex phenotype and negative exome. What comes next?",
        "case": {
            "age": 3,
            "sex": "female",
            "phenotypes": [
                {"hpo_id": "HP:0001250", "label": "Seizures", "status": "present"},
                {"hpo_id": "HP:0001263", "label": "Global developmental delay", "status": "present"},
                {"hpo_id": "HP:0001999", "label": "Abnormal facial shape", "status": "present"},
                {"hpo_id": "HP:0002910", "label": "Elevated hepatic transaminase", "status": "present"},
                {"hpo_id": "HP:0001252", "label": "Hypotonia", "status": "present"},
            ],
            "prior_testing": "exome",
            "test_result": "negative",
            "vus_present": False,
            "gene_results": [],
        },
    },
}

# ── Scenario Selection ────────────────────────────────
st.subheader("Step 0: Choose Scenario")
scenario_choice = st.radio(
    "Select a demo scenario:",
    options=["current_case"] + list(DEMO_SCENARIOS.keys()),
    format_func=lambda x: "Current Case (from Case Builder)" if x == "current_case" else DEMO_SCENARIOS[x]["title"],
    horizontal=True,
)

if scenario_choice == "current_case":
    case = st.session_state.get("current_case")
    if not case:
        st.warning("No case loaded. Go to **Case Builder** or choose a demo scenario.")
        st.stop()
    scenario_desc = "Using the case you built in the Case Builder."
else:
    scenario = DEMO_SCENARIOS[scenario_choice]
    case = scenario["case"]
    scenario_desc = scenario["description"]
    st.info(f"**{scenario['title']}**: {scenario_desc}")

# ── Show Case Summary ─────────────────────────────────
with st.expander("📋 Case Details", expanded=False):
    phenos = case.get("phenotypes", [])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Age:** {case.get('age', '?')}")
        st.write(f"**Sex:** {case.get('sex', '?')}")
    with col2:
        st.write(f"**Phenotypes:** {len(phenos)}")
        st.write(f"**Prior Testing:** {case.get('prior_testing', 'none')}")
    with col3:
        st.write(f"**VUS Present:** {'Yes' if case.get('vus_present') else 'No'}")
        genes = case.get("gene_results", [])
        st.write(f"**Gene Findings:** {len(genes)}")

    for p in phenos:
        st.markdown(f"- {'✅' if p.get('status') == 'present' else '❌'} {p.get('label', p.get('hpo_id', ''))}")

# ── Run Full Pipeline ─────────────────────────────────
if st.button("▶️ Run Full Demo Pipeline", type="primary"):
    with st.spinner("Running diagnostic pipeline..."):
        try:
            resp = httpx.post(f"{API_URL}/recommend", json=case, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            st.session_state["demo_result"] = result
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

result = st.session_state.get("demo_result")
if not result:
    st.info("Click 'Run Full Demo Pipeline' to start.")
    st.stop()

# ═════════════════════════════════════════════════════════
# STEP 1: Ingest Messy Case → Disease Differential
# ═════════════════════════════════════════════════════════
st.divider()
st.subheader("Step 1: Disease Differential")
st.markdown("The copilot ingests the clinical phenotype and returns a ranked differential diagnosis.")

diseases = result.get("diseases", [])
confidence = result.get("confidence", 0)

# Confidence gauge
if confidence >= 0.7:
    st.success(f"**Confidence: {confidence:.1%}** — Strong phenotype-disease alignment")
elif confidence >= 0.3:
    st.warning(f"**Confidence: {confidence:.1%}** — Moderate certainty, broad differential")
else:
    st.error(f"**Confidence: {confidence:.1%}** — Low confidence, sparse or ambiguous data")

completeness = result.get("record_completeness", 0)
st.caption(f"Record completeness: {completeness:.0%}")

# Top diseases
if diseases:
    df_d = pd.DataFrame(diseases[:10])
    fig = go.Figure(go.Bar(
        x=df_d["score"],
        y=df_d["name"],
        orientation="h",
        marker_color=["#e74c3c" if i == 0 else "#3498db" for i in range(len(df_d))],
        text=df_d["score"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        title="Top 10 Disease Candidates",
        height=400,
        margin=dict(l=300),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top-1 rationale
    top = diseases[0]
    st.info(f"**#1 {top['name']}**: {top.get('rationale', 'No rationale available.')}")

# ═════════════════════════════════════════════════════════
# STEP 2: Uncertainty Map
# ═════════════════════════════════════════════════════════
st.divider()
st.subheader("Step 2: Uncertainty Map")
st.markdown("Three-axis uncertainty decomposition: where is the system most uncertain?")

unc = result.get("uncertainty")
if unc:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pheno_unc = unc.get("phenotype_uncertainty", 0)
        st.metric("Phenotype Uncertainty", f"{pheno_unc:.2f}",
                  help="How much phenotype information is missing or ambiguous")
    with col2:
        gen_unc = unc.get("genomic_uncertainty", 0)
        st.metric("Genomic Uncertainty", f"{gen_unc:.2f}",
                  help="How much genomic testing is pending or inconclusive")
    with col3:
        dec_unc = unc.get("decision_uncertainty", 0)
        st.metric("Decision Uncertainty", f"{dec_unc:.2f}",
                  help="How difficult the next clinical action choice is")
    with col4:
        overall = unc.get("overall", 0)
        st.metric("Overall Uncertainty", f"{overall:.2f}")

    # Radar chart
    axes = ["Phenotype", "Genomic", "Decision"]
    vals = [pheno_unc, gen_unc, dec_unc]
    fig = go.Figure(data=go.Scatterpolar(
        r=vals + [vals[0]],
        theta=axes + [axes[0]],
        fill="toself",
        marker_color="#e74c3c",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Uncertainty Decomposition",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Counterfactuals
    cfs = unc.get("counterfactuals", [])
    if cfs:
        st.markdown("**What Would Change?** (Counterfactual Signals)")
        for cf in cfs:
            impact = cf.get("impact_magnitude", 0)
            icon = "🔴" if impact > 0.3 else "🟡" if impact > 0.1 else "🟢"
            st.markdown(
                f"{icon} **{cf['signal_type']}**: {cf['description']} → "
                f"*{cf['expected_impact']}* (magnitude: {impact:.2f})"
            )
else:
    st.info("Uncertainty decomposition not available for this run.")

# ═════════════════════════════════════════════════════════
# STEP 3: Next Best Steps (VOI-Ranked)
# ═════════════════════════════════════════════════════════
st.divider()
st.subheader("Step 3: Next Best Steps")
st.markdown("VOI-ranked (Value of Information) actions — what gives the most diagnostic value per dollar/day?")

voi_actions = result.get("voi_actions", [])
test_recs = result.get("test_recommendations", [])
next_phenos = result.get("next_best_phenotypes", [])

tab_voi, tab_genomic, tab_pheno = st.tabs(["🎯 VOI-Ranked Actions", "🧬 Genomic Assessment", "🔬 Phenotype Questions"])

with tab_voi:
    if voi_actions:
        for i, a in enumerate(voi_actions[:8]):
            col_rank, col_action, col_score, col_meta = st.columns([0.5, 3, 1.5, 2])
            with col_rank:
                st.markdown(f"**#{i+1}**")
            with col_action:
                st.markdown(f"**{a['action']}**")
                if a.get("rationale"):
                    st.caption(a["rationale"])
            with col_score:
                voi_val = a.get("cost_adjusted_voi", 0)
                st.metric("VOI Score", f"{voi_val:.3f}")
            with col_meta:
                cost = a.get("cost_dollars")
                time_d = a.get("turnaround_days")
                inv = a.get("invasiveness", "")
                bucket = a.get("timeline_bucket", "")
                meta_parts = []
                if cost:
                    meta_parts.append(f"${cost:,.0f}")
                if time_d:
                    meta_parts.append(f"{time_d:.0f}d")
                if inv:
                    meta_parts.append(inv)
                if bucket:
                    meta_parts.append(f"📅 {bucket}")
                st.caption(" | ".join(meta_parts))
    elif test_recs:
        st.markdown("*VOI scoring not available — showing rule-based recommendations:*")
        for t in test_recs:
            st.markdown(f"**{t['rank']}. {t['action']}** ({t.get('action_type', '')})")
            if t.get("rationale"):
                st.caption(t["rationale"])
    else:
        st.info("No action recommendations available.")

with tab_genomic:
    ga = result.get("genomic_assessment")
    if ga:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Genomic Maturity:** `{ga.get('genomic_maturity', 'none')}`")

            escalation = ga.get("escalation_path", [])
            if escalation:
                st.markdown("**Escalation Path:**")
                for step in escalation:
                    st.markdown(f"- ➡️ {step}")

            vus_triage = ga.get("vus_triage", [])
            if vus_triage:
                st.markdown("**VUS Triage:**")
                for item in vus_triage:
                    st.markdown(f"- 🟡 {item}")

        with col2:
            st.markdown("**Timeline Buckets:**")
            for bucket, label in [("now_actions", "🔴 Now"), ("next_visit_actions", "🟠 Next Visit"), ("periodic_actions", "🟢 Periodic")]:
                items = ga.get(bucket, [])
                if items:
                    st.markdown(f"**{label}:**")
                    for item in items:
                        st.markdown(f"- {item}")

            reanalysis = ga.get("reanalysis_plan", [])
            if reanalysis:
                st.markdown("**Reanalysis Plan:**")
                for item in reanalysis:
                    st.markdown(f"- 🔄 {item}")

            counseling = ga.get("counseling", [])
            if counseling:
                st.markdown("**Counseling:**")
                for item in counseling:
                    st.markdown(f"- 💬 {item}")
    else:
        st.info("No genomic assessment available.")

with tab_pheno:
    if next_phenos:
        for p in next_phenos[:8]:
            col_q, col_ig = st.columns([4, 1])
            with col_q:
                label = p.get("label") or p["hpo_id"]
                st.markdown(f"**Ask about: {label}** (`{p['hpo_id']}`)")
                if p.get("rationale"):
                    st.caption(p["rationale"])
            with col_ig:
                st.metric("Info Gain", f"{p['expected_info_gain']:.3f}")
    else:
        st.info("No phenotype questions to suggest.")


# ═════════════════════════════════════════════════════════
# STEP 4: Evidence Explanations
# ═════════════════════════════════════════════════════════
st.divider()
st.subheader("Step 4: Evidence-Grounded Explanations")
st.markdown("For each top disease, see the supporting, contradicting, and missing evidence.")

evidence = result.get("evidence_explanations", [])
if evidence:
    for expl in evidence[:5]:
        with st.expander(f"#{expl['rank']} {expl['disease_name']} — Score: {expl['score']:.4f}"):
            if expl.get("summary"):
                st.markdown(f"**Summary:** {expl['summary']}")

            col_s, col_c, col_m = st.columns(3)
            with col_s:
                st.markdown("**✅ Supporting Evidence**")
                for e in expl.get("supporting_evidence", []):
                    label = e.get("label") or e.get("hpo_id", "")
                    freq = e.get("frequency_label", "")
                    st.markdown(f"- {label}" + (f" ({freq})" if freq else ""))
                if not expl.get("supporting_evidence"):
                    st.caption("None found")

            with col_c:
                st.markdown("**❌ Contradicting Evidence**")
                for e in expl.get("contradicting_evidence", []):
                    label = e.get("label") or e.get("hpo_id", "")
                    st.markdown(f"- {label}")
                if not expl.get("contradicting_evidence"):
                    st.caption("None found")

            with col_m:
                st.markdown("**❓ Missing Key Evidence**")
                for e in expl.get("missing_key_evidence", []):
                    label = e.get("label") or e.get("hpo_id", "")
                    freq = e.get("frequency_label", "")
                    st.markdown(f"- {label}" + (f" ({freq})" if freq else ""))
                if not expl.get("missing_key_evidence"):
                    st.caption("None found")
else:
    st.info("Evidence explanations not available for this run.")

# ── Raw JSON ──────────────────────────────────────────
st.divider()
with st.expander("🔧 Full JSON Response"):
    st.code(json.dumps(result, indent=2, default=str), language="json")

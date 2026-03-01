"""Page 2: Recommend — differential diagnosis + next-best-steps with enhanced modules."""

import streamlit as st
import httpx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Recommend", page_icon="🔍", layout="wide")
st.title("🔍 Diagnostic Recommendation")

API_URL = st.session_state.get("api_url", "http://localhost:8099")

# ── Check for saved case ─────────────────────────────
case = st.session_state.get("current_case")
if not case:
    st.warning("No case loaded. Go to **Case Builder** first.")
    st.stop()

st.markdown(f"**Patient:** Age {case.get('age', '?')}, {case.get('sex', '?')}")
st.markdown(f"**Phenotypes:** {len(case.get('phenotypes', []))} terms entered")

# ── Run Recommendation ───────────────────────────────
if st.button("🚀 Run Diagnosis", type="primary") or "last_recommend" in st.session_state:
    with st.spinner("Running inference..."):
        try:
            resp = httpx.post(f"{API_URL}/recommend", json=case, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            st.session_state["last_recommend"] = result
        except Exception as e:
            if "last_recommend" not in st.session_state:
                st.error(f"API error: {e}")
                st.stop()
            result = st.session_state["last_recommend"]

    # ── Confidence + Uncertainty Overview ─────────────
    confidence = result.get("confidence", 0)
    completeness = result.get("record_completeness", 0)
    unc = result.get("uncertainty")

    col_conf, col_comp, col_unc = st.columns([1, 1, 1])
    with col_conf:
        if confidence >= 0.7:
            st.success(f"Confidence: {confidence:.1%}")
        elif confidence >= 0.3:
            st.warning(f"Confidence: {confidence:.1%}")
        else:
            st.error(f"Confidence: {confidence:.1%}")
    with col_comp:
        st.metric("Record Completeness", f"{completeness:.0%}")
    with col_unc:
        if unc:
            overall_unc = unc.get("overall", 0)
            st.metric("Overall Uncertainty", f"{overall_unc:.2f}",
                      help="0=certain, 1=maximum uncertainty")

    st.caption(f"Run: {result.get('run_id', '?')[:8]}… | Hash: {result.get('inputs_hash', '')[:12]}")

    # ── Uncertainty Decomposition (collapsed) ─────────
    if unc:
        with st.expander("🎯 Uncertainty Decomposition"):
            col_p, col_g, col_d = st.columns(3)
            with col_p:
                st.metric("Phenotype", f"{unc.get('phenotype_uncertainty', 0):.2f}")
            with col_g:
                st.metric("Genomic", f"{unc.get('genomic_uncertainty', 0):.2f}")
            with col_d:
                st.metric("Decision", f"{unc.get('decision_uncertainty', 0):.2f}")

            cfs = unc.get("counterfactuals", [])
            if cfs:
                st.markdown("**Counterfactuals — What Would Change:**")
                for cf in cfs[:3]:
                    impact = cf.get("impact_magnitude", 0)
                    icon = "🔴" if impact > 0.3 else "🟡" if impact > 0.1 else "🟢"
                    st.markdown(
                        f"{icon} **{cf['signal_type']}**: {cf['description']} "
                        f"→ *{cf['expected_impact']}*"
                    )

    # ── Disease Differential ──────────────────────────
    st.subheader("Differential Diagnosis")
    diseases = result.get("diseases", [])
    if diseases:
        df_diseases = pd.DataFrame(diseases[:20])
        df_diseases["rank"] = range(1, len(df_diseases) + 1)

        # Bar chart
        fig = px.bar(
            df_diseases.head(10),
            x="score",
            y="name",
            orientation="h",
            color="score",
            color_continuous_scale="Reds",
            title="Top 10 Disease Candidates",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Detail table with rationale + evidence
        evidence_map = {}
        for expl in result.get("evidence_explanations", []):
            evidence_map[expl.get("disease_id", "")] = expl

        for d in diseases[:10]:
            rank_num = diseases.index(d) + 1
            cal_str = f" | Calibrated: {d['calibrated_score']:.1%}" if d.get("calibrated_score") is not None else ""
            match_str = f" | Match: {d.get('phenotype_match', '?')}" if d.get("phenotype_match") else ""
            with st.expander(f"#{rank_num} — {d['name']} (score: {d['score']:.4f}{cal_str}{match_str})"):
                # Rationale
                if d.get("rationale"):
                    st.info(f"💡 **Rationale:** {d['rationale']}")

                col_s, col_c = st.columns(2)
                with col_s:
                    st.markdown("**Supporting HPOs:**")
                    for h in d.get("supporting_hpos", []):
                        st.markdown(f"- ✅ {h}")
                    if not d.get("supporting_hpos"):
                        st.caption("None in current phenotype set")
                with col_c:
                    st.markdown("**Contradicting HPOs:**")
                    for h in d.get("contradicting_hpos", []):
                        st.markdown(f"- ❌ {h}")
                    if not d.get("contradicting_hpos"):
                        st.caption("None in current phenotype set")

                # Evidence from evidence module
                expl = evidence_map.get(d.get("disease_id", ""))
                if expl:
                    missing = expl.get("missing_key_evidence", [])
                    if missing:
                        st.markdown("**❓ Missing Key Evidence — Ask About:**")
                        for e in missing[:5]:
                            label = e.get("label") or e.get("hpo_id", "")
                            freq = e.get("frequency_label", "")
                            st.markdown(f"- {label}" + (f" ({freq})" if freq else ""))
    else:
        st.info("No diseases scored. Check that model artifacts are loaded.")

    # ── Next Best Steps (4 Tabs) ──────────────────────
    st.subheader("Next Best Steps")
    tab_voi, tab_pheno, tab_test, tab_other = st.tabs(
        ["🎯 VOI-Ranked Actions", "🔬 Phenotype Questions", "🧪 Tests", "📋 Referral / Monitoring"]
    )

    with tab_voi:
        voi_actions = result.get("voi_actions", [])
        if voi_actions:
            for i, a in enumerate(voi_actions[:8]):
                col_a, col_v, col_m = st.columns([4, 1, 2])
                with col_a:
                    st.markdown(f"**#{i+1}. {a['action']}**")
                    if a.get("rationale"):
                        st.caption(a["rationale"])
                with col_v:
                    st.metric("VOI", f"{a.get('cost_adjusted_voi', 0):.3f}")
                with col_m:
                    meta = []
                    if a.get("cost_dollars"):
                        meta.append(f"${a['cost_dollars']:,.0f}")
                    if a.get("turnaround_days"):
                        meta.append(f"{a['turnaround_days']:.0f}d")
                    if a.get("timeline_bucket"):
                        meta.append(f"📅 {a['timeline_bucket']}")
                    st.caption(" | ".join(meta))
        else:
            st.info("VOI-based action scoring not available for this run.")

    with tab_pheno:
        next_phenos = result.get("next_best_phenotypes", [])
        if next_phenos:
            for p in next_phenos[:10]:
                col_q, col_ig = st.columns([4, 1])
                with col_q:
                    label = p.get("label") or p["hpo_id"]
                    st.markdown(f"**Ask about: {label}** (`{p['hpo_id']}`)")
                    if p.get("rationale"):
                        st.caption(f"💡 {p['rationale']}")
                with col_ig:
                    st.metric("Info Gain", f"{p['expected_info_gain']:.3f} bits")
        else:
            st.info("No phenotype questions to suggest.")

    with tab_test:
        test_recs = [t for t in result.get("test_recommendations", []) if t.get("action_type") == "test"]
        if test_recs:
            for t in test_recs:
                st.markdown(f"**{t['rank']}. {t['action']}**")
                if t.get("rationale"):
                    st.info(f"💡 {t['rationale']}")
        else:
            st.info("No test recommendations at this time.")

    with tab_other:
        others = [
            t for t in result.get("test_recommendations", [])
            if t.get("action_type") in ("referral", "reanalysis", "monitoring")
        ]
        if others:
            for t in others:
                icon = {"referral": "🏥", "reanalysis": "🔄", "monitoring": "📊"}.get(t["action_type"], "📋")
                st.markdown(f"**{icon} {t['action']}**")
                if t.get("rationale"):
                    st.info(f"💡 {t['rationale']}")
        else:
            st.info("No referral/monitoring recommendations at this time.")

    # ── Genomic Assessment ────────────────────────────
    ga = result.get("genomic_assessment")
    if ga:
        st.subheader("🧬 Genomic Assessment")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Maturity:** `{ga.get('genomic_maturity', 'none')}`")
            escalation = ga.get("escalation_path", [])
            if escalation:
                st.markdown("**Escalation:**")
                for s in escalation:
                    st.markdown(f"- ➡️ {s}")
            vus_items = ga.get("vus_triage", [])
            if vus_items:
                st.markdown("**VUS Triage:**")
                for item in vus_items:
                    st.markdown(f"- 🟡 {item}")
        with col2:
            for bucket, label in [("now_actions", "🔴 Now"), ("next_visit_actions", "🟠 Next Visit"), ("periodic_actions", "🟢 Periodic")]:
                items = ga.get(bucket, [])
                if items:
                    st.markdown(f"**{label}:**")
                    for item in items:
                        st.markdown(f"- {item}")

    # ── What Would Change ─────────────────────────────
    st.subheader("What Would Change?")
    st.markdown(
        "Select a suggested phenotype question above and go to the **Simulation** page "
        "to see how adding it changes the disease ranking."
    )

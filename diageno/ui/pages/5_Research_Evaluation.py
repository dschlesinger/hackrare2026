"""Page 5: Research Evaluation — experiments, ablation, calibration, fairness."""

import streamlit as st
import httpx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Research Evaluation", page_icon="🔬", layout="wide")
st.title("🔬 Research Evaluation Suite")

API_URL = st.session_state.get("api_url", "http://localhost:8099")

st.markdown("""
This dashboard runs 5 research-grade experiments evaluating the diagnostic copilot:

1. **Retrospective Replay** — Steps-to-correct-diagnosis vs baselines
2. **Missingness Robustness** — Degradation under 30-70% phenotype dropout
3. **Calibration Analysis** — Brier score, ECE, reliability curves
4. **Ablation Study** — Module-by-module incremental lift
5. **Clinician Rubric** — Automated proxy for clinician quality scoring
""")

# ── Experiment Selection ──────────────────────────────
st.subheader("Select Experiments")
exp_options = {
    "replay": "Retrospective Replay",
    "missingness": "Missingness Robustness",
    "calibration": "Calibration Analysis",
    "ablation": "Ablation Study",
    "rubric": "Clinician Rubric",
}
selected = st.multiselect(
    "Experiments to run",
    options=list(exp_options.keys()),
    default=list(exp_options.keys()),
    format_func=lambda x: exp_options[x],
)

# ── Run Evaluation ────────────────────────────────────
if st.button("🚀 Run Evaluation Suite", type="primary"):
    with st.spinner("Running experiments (this may take a few minutes)..."):
        try:
            resp = httpx.post(
                f"{API_URL}/evaluate",
                json={"experiments": selected},
                timeout=300,
            )
            resp.raise_for_status()
            st.session_state["eval_suite"] = resp.json()
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()
    st.success("Evaluation complete!")

# ── Display Results ───────────────────────────────────
suite = st.session_state.get("eval_suite")
if not suite:
    st.info("Click 'Run Evaluation Suite' to start. Ensure the API server is running.")
    st.stop()

# ── Headline Claim ────────────────────────────────────
st.markdown(f"### 📄 {suite.get('headline_claim', '')}")

# ── Primary Metric ────────────────────────────────────
primary = suite.get("primary_metric", {})
secondary = suite.get("secondary_metrics", {})

if primary:
    st.subheader("Primary Metric")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(primary.get("name", "Steps-to-Dx"), primary.get("model_value", "—"))
    with col2:
        st.metric("Baseline (Random)", primary.get("baseline_random", "—"))
    with col3:
        st.metric("Baseline (Frequency)", primary.get("baseline_frequency", "—"))

if secondary:
    st.subheader("Secondary Metrics")
    cols = st.columns(4)
    metric_labels = {
        "brier_score": "Brier Score",
        "calibration_brier": "Brier Score",
        "ece": "ECE",
        "calibration_ece": "ECE",
        "robustness_30pct": "Conf. @ 30% Drop",
        "clinician_rubric_pct": "Rubric %",
    }
    i = 0
    for key, label in metric_labels.items():
        val = secondary.get(key)
        if val is not None:
            with cols[i % 4]:
                st.metric(label, f"{val:.4f}" if isinstance(val, float) else str(val))
            i += 1

# ── Per-Experiment Details ────────────────────────────
experiments = suite.get("experiments", [])

for exp in experiments:
    exp_name = exp.get("experiment", "")
    desc = exp.get("description", "")
    metrics = exp.get("metrics", {})
    details = exp.get("details", [])
    duration = exp.get("duration_seconds", 0)

    st.divider()

    # ── Experiment 1: Retrospective Replay ────────────
    if "retrospective" in exp_name.lower() or "replay" in exp_name.lower():
        st.subheader("📊 Experiment 1: Retrospective Case Replay")
        st.caption(f"{desc} | Duration: {duration:.1f}s")

        model_m = metrics.get("model", {})
        base_rand = metrics.get("baseline_random", {})
        base_freq = metrics.get("baseline_frequency", {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Mean Steps", model_m.get("mean_steps", "—"))
        with col2:
            st.metric("Model Median Steps", model_m.get("median_steps", "—"))
        with col3:
            st.metric("Random Baseline", base_rand.get("mean_steps", "—"))
        with col4:
            imp_data = metrics.get("improvement_vs_random")
            if isinstance(imp_data, dict):
                imp_pct = imp_data.get("percent_improvement")
                st.metric("Improvement vs Random", f"{imp_pct:.1f}%" if imp_pct is not None else "—")
            elif imp_data is not None:
                st.metric("Improvement vs Random", f"{imp_data:.1%}")
            else:
                st.metric("Improvement vs Random", "—")

        if details:
            df = pd.DataFrame(details)
            fig = px.bar(
                df, x="case", y=["model_steps", "random_steps", "freq_steps"],
                barmode="group",
                title="Steps-to-Correct by Case",
                labels={"value": "Steps", "variable": "Method"},
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Per-case details"):
                st.dataframe(df)

    # ── Experiment 2: Missingness Robustness ──────────
    elif "missingness" in exp_name.lower() or "robustness" in exp_name.lower():
        st.subheader("🔒 Experiment 2: Missingness Robustness")
        st.caption(f"{desc} | Duration: {duration:.1f}s")

        curve_data = metrics.get("robustness_curve", [])
        if curve_data:
            df_curve = pd.DataFrame(curve_data)

            col1, col2, col3 = st.columns(3)
            with col1:
                v30 = metrics.get("confidence_at_30pct_drop")
                st.metric("Conf @ 30% Drop", f"{v30:.4f}" if v30 is not None else "—")
            with col2:
                v60 = metrics.get("confidence_at_60pct_drop")
                st.metric("Conf @ 60% Drop", f"{v60:.4f}" if v60 is not None else "—")
            with col3:
                s30 = metrics.get("stability_at_30pct_drop")
                st.metric("Top-1 Stability @ 30%", f"{s30:.1%}" if s30 is not None else "—")

            # Confidence curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_curve["drop_fraction"],
                y=df_curve["mean_confidence"],
                mode="lines+markers",
                name="Mean Confidence",
                error_y=dict(type="data", array=df_curve["std_confidence"].tolist()),
            ))
            fig.add_trace(go.Scatter(
                x=df_curve["drop_fraction"],
                y=df_curve["top1_stability"],
                mode="lines+markers",
                name="Top-1 Stability",
                yaxis="y2",
            ))
            fig.update_layout(
                title="Robustness Curve: Confidence & Stability vs Phenotype Dropout",
                xaxis_title="Fraction of Phenotypes Dropped",
                yaxis_title="Mean Confidence",
                yaxis2=dict(title="Top-1 Stability", overlaying="y", side="right", range=[0, 1.1]),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Experiment 3: Calibration ─────────────────────
    elif "calibration" in exp_name.lower():
        st.subheader("🎯 Experiment 3: Calibration Analysis")
        st.caption(f"{desc} | Duration: {duration:.1f}s")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Brier Score", f"{metrics.get('brier_score', '—')}")
        with col2:
            st.metric("ECE", f"{metrics.get('ece', '—')}")
        with col3:
            st.metric("N Samples", metrics.get("n_samples", "—"))
        with col4:
            st.metric("Mean Confidence", f"{metrics.get('mean_confidence', '—')}")

        # Reliability diagram
        rel_bins = metrics.get("reliability_bins", [])
        if rel_bins:
            df_bins = pd.DataFrame(rel_bins)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_bins["bin_center"],
                y=df_bins["avg_accuracy"],
                name="Actual Frequency",
                marker_color="steelblue",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line=dict(dash="dash", color="gray"),
            ))
            fig.update_layout(
                title="Reliability Diagram",
                xaxis_title="Predicted Probability",
                yaxis_title="Observed Frequency",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Experiment 4: Ablation Study ──────────────────
    elif "ablation" in exp_name.lower():
        st.subheader("🧪 Experiment 4: Ablation Study")
        st.caption(f"{desc} | Duration: {duration:.1f}s")

        ablation_table = metrics.get("ablation_table", details)
        if ablation_table:
            df_abl = pd.DataFrame(ablation_table)

            # Highlight best
            st.dataframe(
                df_abl.style.format({
                    "top1_accuracy": "{:.1%}",
                    "top5_accuracy": "{:.1%}",
                    "top10_accuracy": "{:.1%}",
                    "mrr": "{:.4f}",
                    "delta_top1": "{:+.1%}",
                    "delta_top5": "{:+.1%}",
                    "delta_top10": "{:+.1%}",
                }),
                use_container_width=True,
            )

            # Delta chart
            df_delta = df_abl[df_abl["condition"] != "full_model"]
            if not df_delta.empty:
                fig = px.bar(
                    df_delta,
                    x="condition",
                    y=["delta_top1", "delta_top5", "delta_top10"],
                    barmode="group",
                    title="Ablation: Delta on Top-K Accuracy (vs Full Model)",
                    labels={"value": "Delta", "variable": "Metric"},
                    color_discrete_sequence=["#e74c3c", "#e67e22", "#f1c40f"],
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.caption(f"Evaluated on {metrics.get('n_cases', '?')} cases.")

    # ── Experiment 5: Clinician Rubric ────────────────
    elif "rubric" in exp_name.lower() or "clinician" in exp_name.lower():
        st.subheader("👨‍⚕️ Experiment 5: Clinician Rubric Scoring")
        st.caption(f"{desc} | Duration: {duration:.1f}s")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Total Score", f"{metrics.get('mean_total', 0)}/15")
            st.metric("Mean Percentage", f"{metrics.get('mean_percentage', 0):.1f}%")
            st.metric("N Cases", metrics.get('n_cases', 0))

        with col2:
            dim_means = metrics.get("dimension_means", {})
            if dim_means:
                dims = list(dim_means.keys())
                vals = list(dim_means.values())
                # Radar chart
                fig = go.Figure(data=go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=[d.replace("_", " ").title() for d in dims] + [dims[0].replace("_", " ").title()],
                    fill="toself",
                    marker_color="steelblue",
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 3])),
                    title="Rubric Dimensions (0-3 scale)",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        if details:
            with st.expander("Per-case rubric breakdown"):
                df_rubric = pd.DataFrame([
                    {
                        "case": d["case"],
                        "total": d["total"],
                        "pct": f"{d['percentage']:.0f}%",
                        **d["rubric_scores"],
                    }
                    for d in details
                ])
                st.dataframe(df_rubric, use_container_width=True)

# ── Publication-Ready Export ──────────────────────────
st.divider()
st.subheader("📋 Publication Export")
st.markdown(
    "Copy the JSON below for inclusion in research papers or supplementary material."
)
if suite:
    import json
    with st.expander("Full Evaluation JSON"):
        st.code(json.dumps(suite, indent=2, default=str), language="json")

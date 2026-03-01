"""Page 3: Simulation — apply a phenotype and see ranking changes."""

import streamlit as st
import httpx
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Simulation", page_icon="🔄", layout="wide")
st.title("🔄 Simulation — Apply a Phenotype Step")

API_URL = st.session_state.get("api_url", "http://localhost:8099")

case = st.session_state.get("current_case")
if not case:
    st.warning("No case loaded. Go to **Case Builder** first.")
    st.stop()

st.markdown("Simulate adding or removing a phenotype and see how disease rankings change.")

# ── Input ─────────────────────────────────────────────
col_hpo, col_label, col_status, col_action = st.columns([2, 3, 2, 1])
with col_hpo:
    sim_hpo = st.text_input("HPO ID to simulate", placeholder="HP:0001250")
with col_label:
    sim_label = st.text_input("Label", placeholder="Seizures")
with col_status:
    sim_status = st.selectbox("Status", ["present", "absent"])
with col_action:
    sim_action = st.selectbox("Action", ["add", "remove"])

if st.button("▶️ Simulate", type="primary") and sim_hpo:
    payload = {
        "case": case,
        "new_phenotype": {
            "hpo_id": sim_hpo,
            "label": sim_label or sim_hpo,
            "status": sim_status,
        },
        "action": sim_action,
    }

    with st.spinner("Simulating..."):
        try:
            resp = httpx.post(f"{API_URL}/simulate_step", json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

    # ── Before vs After ───────────────────────────────
    col_before, col_after = st.columns(2)

    with col_before:
        st.subheader("Before")
        for i, d in enumerate(result.get("before_top5", []), 1):
            st.markdown(f"**{i}.** {d['name']} — {d['score']:.4f}")

    with col_after:
        st.subheader("After")
        for i, d in enumerate(result.get("after_top5", []), 1):
            st.markdown(f"**{i}.** {d['name']} — {d['score']:.4f}")

    # ── Rank Changes ──────────────────────────────────
    st.subheader("Rank Changes")
    changes = result.get("rank_changes", [])
    if changes:
        df = pd.DataFrame(changes)

        # Waterfall-style chart
        fig = go.Figure()
        colors = ["green" if c > 0 else "red" for c in df["change"]]
        fig.add_trace(
            go.Bar(
                x=df["name"],
                y=df["change"],
                marker_color=colors,
                text=df["change"].apply(lambda x: f"+{x}" if x > 0 else str(x)),
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Rank Change (positive = moved up)",
            yaxis_title="Rank Change",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(
            df[["name", "rank_before", "rank_after", "change"]].rename(
                columns={
                    "name": "Disease",
                    "rank_before": "Rank Before",
                    "rank_after": "Rank After",
                    "change": "Change",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No significant rank changes for this simulation.")

    # ── Quick Apply ───────────────────────────────────
    st.divider()
    if st.button("✅ Apply this phenotype to the case"):
        case["phenotypes"].append({
            "hpo_id": sim_hpo,
            "label": sim_label or sim_hpo,
            "status": sim_status,
            "onset": None,
        })
        st.session_state["current_case"] = case
        # Also update the editable phenotype list
        if "phenotypes" in st.session_state:
            st.session_state.phenotypes.append({
                "hpo_id": sim_hpo,
                "label": sim_label or sim_hpo,
                "status": sim_status,
                "onset": None,
            })
        st.success("Phenotype applied! Go to **Recommend** to see updated results.")

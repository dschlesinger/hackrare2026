"""Page 4: Validation Dashboard — enhanced with VOI, uncertainty, and decision-point eval."""

import streamlit as st
import httpx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from pathlib import Path

st.set_page_config(page_title="Validation", page_icon="📊", layout="wide")
st.title("📊 Validation Dashboard")

API_URL = st.session_state.get("api_url", "http://localhost:8099")


# ── Helper: load validation cases from workspace ─────
@st.cache_data
def load_validation_cases() -> list[dict]:
    """Load ValidationCase* files from the workspace root."""
    workspace = Path(__file__).resolve().parent.parent.parent.parent
    cases = []
    for f in sorted(workspace.glob("ValidationCase*")):
        try:
            data = json.loads(f.read_text())
            data["_filename"] = f.name
            cases.append(data)
        except Exception:
            pass
    return cases


def compute_mrr(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks if r > 0]))


def compute_hits_at_k(ranks: list[int], k: int) -> float:
    if not ranks:
        return 0.0
    return float(np.mean([1 if r <= k else 0 for r in ranks]))


def _fuzzy_match(gt: str, pred: str) -> bool:
    """Fuzzy action match: token overlap >= 50%."""
    gt_tokens = set(gt.lower().split())
    pred_tokens = set(pred.lower().split())
    if not gt_tokens:
        return False
    overlap = len(gt_tokens & pred_tokens)
    return overlap / len(gt_tokens) >= 0.5


# ── Validation cases ──────────────────────────────────
st.subheader("Validation Cases")
val_cases = load_validation_cases()

if not val_cases:
    st.info("No ValidationCase* files found in the workspace root.")
    st.stop()

st.markdown(f"Found **{len(val_cases)}** validation cases.")

# ── Run Evaluation ────────────────────────────────────
if st.button("🔬 Run Full Evaluation", type="primary"):
    progress = st.progress(0)
    results = []

    for i, vc in enumerate(val_cases):
        progress.progress((i + 1) / len(val_cases))

        phenotypes = vc.get("phenotypes", [])
        case_input = {
            "age": vc.get("patient", {}).get("age") or vc.get("age"),
            "sex": vc.get("patient", {}).get("sex") or vc.get("sex"),
            "phenotypes": [
                {"hpo_id": p.get("hpo_id") or p.get("hpo", ""), "label": p.get("label", ""), "status": p.get("status", "present")}
                for p in phenotypes
            ],
            "prior_testing": vc.get("prior_testing", "none"),
            "vus_present": vc.get("vus_present", False),
            "genes_mentioned": vc.get("genes_mentioned", []),
            "gene_results": vc.get("gene_results", []),
        }

        try:
            resp = httpx.post(f"{API_URL}/recommend", json=case_input, timeout=30)
            resp.raise_for_status()
            rec = resp.json()

            results.append({
                "case": vc["_filename"],
                "case_id": vc.get("case_id", ""),
                "n_phenotypes": len(phenotypes),
                "top1_disease": rec["diseases"][0]["name"] if rec["diseases"] else "",
                "top1_score": rec["diseases"][0]["score"] if rec["diseases"] else 0,
                "confidence": rec.get("confidence", 0),
                "record_completeness": rec.get("record_completeness", 0),
                "n_suggestions": len(rec.get("next_best_phenotypes", [])),
                "n_test_recs": len(rec.get("test_recommendations", [])),
                "n_voi_actions": len(rec.get("voi_actions", [])),
                "overall_uncertainty": rec.get("uncertainty", {}).get("overall", 0) if rec.get("uncertainty") else 0,
                "diseases": rec["diseases"],
                "voi_actions": rec.get("voi_actions", []),
                "genomic_assessment": rec.get("genomic_assessment"),
                "decision_points": vc.get("decision_points", []),
            })
        except Exception as e:
            results.append({
                "case": vc["_filename"],
                "case_id": vc.get("case_id", ""),
                "error": str(e),
            })

    st.session_state["eval_results"] = results
    st.success("Evaluation complete!")

# ── Display Results ───────────────────────────────────
eval_results = st.session_state.get("eval_results", [])

if eval_results:
    valid = [r for r in eval_results if "error" not in r]
    errors = [r for r in eval_results if "error" in r]

    if errors:
        st.warning(f"{len(errors)} cases failed: {[e['case'] for e in errors]}")

    if valid:
        df = pd.DataFrame(valid)

        # ── Summary Metrics ───────────────────────────
        st.subheader("Summary Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mean Confidence", f"{df['confidence'].mean():.3f}")
        with col2:
            st.metric("Mean Top-1 Score", f"{df['top1_score'].mean():.3f}")
        with col3:
            st.metric("Mean Uncertainty", f"{df['overall_uncertainty'].mean():.3f}")
        with col4:
            st.metric("Avg VOI Actions", f"{df['n_voi_actions'].mean():.1f}")
        with col5:
            st.metric("Mean Completeness", f"{df['record_completeness'].mean():.1%}")

        # ── Confidence vs Uncertainty ─────────────────
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Confidence", x=df["case"], y=df["confidence"], marker_color="steelblue"))
        fig.add_trace(go.Bar(name="Uncertainty", x=df["case"], y=df["overall_uncertainty"], marker_color="#e74c3c"))
        fig.update_layout(
            barmode="group",
            title="Confidence vs Uncertainty per Case",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Decision Point Evaluation ─────────────────
        st.subheader("Decision Point Evaluation")
        st.markdown(
            "Evaluates VOI-ranked actions and test recommendations against ground-truth "
            "decision points from validation cases."
        )

        action_mrr_scores = []
        action_details = []

        for r in valid:
            dp_list = r.get("decision_points", [])
            if not dp_list:
                continue

            # Collect all model actions
            voi_actions = r.get("voi_actions", [])
            model_actions = [a.get("action", "") for a in voi_actions]

            for stage in dp_list:
                field_key = "doctor_next_steps_ranked"
                gt_steps = stage.get(field_key, stage.get("recommended_next_steps_ranked", []))
                gt_action_strs = [
                    (s.get("action", "") if isinstance(s, dict) else str(s))
                    for s in gt_steps
                ]

                if not gt_action_strs:
                    continue

                # Find rank of first GT action match
                best_rank = len(model_actions) + 1
                matched_gt = ""
                matched_pred = ""
                for gt in gt_action_strs:
                    for k, pred in enumerate(model_actions):
                        if _fuzzy_match(gt, pred):
                            if k + 1 < best_rank:
                                best_rank = k + 1
                                matched_gt = gt
                                matched_pred = pred
                            break

                action_mrr_scores.append(best_rank)
                action_details.append({
                    "case": r["case"],
                    "stage": stage.get("stage", ""),
                    "gt_top_action": gt_action_strs[0] if gt_action_strs else "",
                    "best_match_rank": best_rank,
                    "matched_gt": matched_gt,
                    "matched_pred": matched_pred,
                })

        if action_mrr_scores:
            mrr = compute_mrr(action_mrr_scores)
            hits1 = compute_hits_at_k(action_mrr_scores, 1)
            hits3 = compute_hits_at_k(action_mrr_scores, 3)
            hits5 = compute_hits_at_k(action_mrr_scores, 5)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Action MRR", f"{mrr:.3f}")
            with col_m2:
                st.metric("Hits@1", f"{hits1:.1%}")
            with col_m3:
                st.metric("Hits@3", f"{hits3:.1%}")
            with col_m4:
                st.metric("Hits@5", f"{hits5:.1%}")

            if action_details:
                with st.expander("Per-stage action matching details"):
                    df_actions = pd.DataFrame(action_details)
                    st.dataframe(df_actions, use_container_width=True)

        # ── Robustness Curve ──────────────────────────
        st.subheader("Robustness: Confidence vs Record Quality")
        fig = px.scatter(
            df, x="n_phenotypes", y="confidence",
            size="top1_score",
            color="overall_uncertainty",
            color_continuous_scale="RdYlGn_r",
            hover_data=["case", "top1_disease"],
            title="Robustness — more data → higher confidence, lower uncertainty?",
            trendline="lowess",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Raw Data ──────────────────────────────────
        with st.expander("Raw Evaluation Data"):
            display_cols = ["case", "case_id", "n_phenotypes", "top1_disease",
                            "top1_score", "confidence", "overall_uncertainty",
                            "record_completeness", "n_voi_actions"]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols])

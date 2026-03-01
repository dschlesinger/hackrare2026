"""Diageno Streamlit Application — main entry point.

Pages:
  1. Case Builder
  2. Recommend (differential + next-best-steps)
  3. Simulation
  4. Validation Dashboard
"""

import streamlit as st

st.set_page_config(
    page_title="Diageno — Rare Disease Dx",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 Diageno — Rare Disease Diagnostic Engine")
st.markdown(
    """
    **Diageno** is an end-to-end rare disease diagnostic recommendation platform
    that combines phenotype-driven disease scoring, entropy-based next-best-step
    selection, and evidence-aware test ordering.

    ### Getting Started
    1. **Case Builder** — Enter patient phenotypes (HPO terms) or paste free text
    2. **Recommend** — View differential diagnosis + next best steps
    3. **Simulation** — Simulate adding a phenotype and see how rankings change
    4. **Validation Dashboard** — Evaluate model metrics (MRR, Hits@K, robustness)

    ---
    Use the sidebar to navigate between pages.
    """,
)

# Sidebar info
with st.sidebar:
    st.markdown("### System Status")

    # API URL management — sync input → session state
    default_url = "http://localhost:8099"
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = default_url
    new_url = st.text_input("API Base URL", value=st.session_state["api_url"], key="_api_url_widget")
    if new_url != st.session_state["api_url"]:
        st.session_state["api_url"] = new_url

    api_url = st.session_state["api_url"]

    # Auto-check API health on every page load
    import httpx
    try:
        resp = httpx.get(f"{api_url}/health", timeout=5)
        data = resp.json()
        if data.get("status") == "ok":
            st.success(f"🟢 API Active — Model loaded")
        elif data.get("model_loaded"):
            st.warning(f"🟡 API Running — Model loaded, optional services offline")
        else:
            st.error(f"🔴 API Degraded — Model not loaded")
        with st.expander("Details"):
            st.json(data)
    except Exception as e:
        st.error(f"🔴 Cannot reach API at {api_url}")
        st.caption(str(e))

    if st.button("🔄 Refresh Status"):
        st.rerun()

"""Page 1: Case Builder — input phenotypes, build a case for recommendation."""

import streamlit as st
import httpx
import json

st.set_page_config(page_title="Case Builder", page_icon="📝", layout="wide")
st.title("📝 Case Builder")

API_URL = st.session_state.get("api_url", "http://localhost:8099")

# ── Initialize session state ─────────────────────────
if "phenotypes" not in st.session_state:
    st.session_state.phenotypes = []
if "case_meta" not in st.session_state:
    st.session_state.case_meta = {"age": None, "sex": None, "ancestry": None}

# ── Patient metadata ─────────────────────────────────
st.subheader("Patient Information")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
with col2:
    sex = st.selectbox("Sex", ["", "male", "female", "other", "unknown"])
with col3:
    ancestry = st.text_input("Ancestry (optional)")

st.session_state.case_meta = {"age": age if age > 0 else None, "sex": sex or None, "ancestry": ancestry or None}

# ── HPO Input Methods ────────────────────────────────
st.subheader("Phenotype Entry")
tab_manual, tab_text, tab_paste = st.tabs(["Manual HPO Entry", "Free Text → HPO", "Paste HPO List"])

with tab_manual:
    st.markdown("Search for HPO terms by keyword:")
    search_text = st.text_input("Search HPO terms", placeholder="e.g. seizure, hypotonia")
    if search_text:
        try:
            resp = httpx.post(
                f"{API_URL}/hpo_lookup",
                json={"text": search_text, "max_results": 10},
                timeout=10,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    for r in results:
                        col_a, col_b, col_c = st.columns([3, 2, 1])
                        with col_a:
                            st.write(f"**{r['name']}** ({r['hpo_id']})")
                        with col_b:
                            status = st.selectbox(
                                "Status",
                                ["present", "absent", "past_history"],
                                key=f"status_{r['hpo_id']}",
                            )
                        with col_c:
                            if st.button("Add", key=f"add_{r['hpo_id']}"):
                                st.session_state.phenotypes.append({
                                    "hpo_id": r["hpo_id"],
                                    "label": r["name"],
                                    "status": status,
                                    "onset": None,
                                })
                                st.rerun()
                else:
                    st.info("No results found.")
            else:
                st.warning("API lookup failed — enter HPO IDs manually below.")
        except Exception:
            st.warning("API unavailable — enter HPO IDs manually below.")

    st.markdown("**Or add manually:**")
    col_hpo, col_label, col_status, col_add = st.columns([2, 3, 2, 1])
    with col_hpo:
        manual_hpo = st.text_input("HPO ID", placeholder="HP:0001250")
    with col_label:
        manual_label = st.text_input("Label", placeholder="Seizures")
    with col_status:
        manual_status = st.selectbox("Status", ["present", "absent", "past_history"], key="manual_status")
    with col_add:
        st.write("")
        st.write("")
        if st.button("➕ Add"):
            if manual_hpo:
                st.session_state.phenotypes.append({
                    "hpo_id": manual_hpo,
                    "label": manual_label or manual_hpo,
                    "status": manual_status,
                    "onset": None,
                })
                st.rerun()

with tab_text:
    st.markdown("Paste clinical text and extract HPO terms (requires API):")
    free_text = st.text_area("Clinical description", height=150, placeholder="Paste clinical notes here...")
    if st.button("Extract HPO Terms") and free_text:
        # Simple keyword extraction using HPO lookup
        words = free_text.replace(",", " ").replace(".", " ").split()
        # Try multi-word chunks
        chunks = []
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                chunks.append(" ".join(words[i:j]))

        found_hpos = set()
        for chunk in chunks:
            if len(chunk) < 4:
                continue
            try:
                resp = httpx.post(
                    f"{API_URL}/hpo_lookup",
                    json={"text": chunk, "max_results": 3},
                    timeout=5,
                )
                if resp.status_code == 200:
                    for r in resp.json().get("results", []):
                        if r["hpo_id"] not in found_hpos:
                            st.session_state.phenotypes.append({
                                "hpo_id": r["hpo_id"],
                                "label": r["name"],
                                "status": "present",
                                "onset": None,
                            })
                            found_hpos.add(r["hpo_id"])
            except Exception:
                pass
        if found_hpos:
            st.success(f"Extracted {len(found_hpos)} HPO terms")
            st.rerun()
        else:
            st.warning("No HPO terms extracted. Try entering terms manually.")

with tab_paste:
    st.markdown("Paste a list of HPO IDs (one per line or comma-separated):")
    hpo_list_text = st.text_area("HPO IDs", height=100, placeholder="HP:0001250\nHP:0001252\nHP:0000252")
    if st.button("Import HPO List") and hpo_list_text:
        ids = [x.strip() for x in hpo_list_text.replace(",", "\n").split("\n") if x.strip().startswith("HP:")]
        for hpo_id in ids:
            if not any(p["hpo_id"] == hpo_id for p in st.session_state.phenotypes):
                st.session_state.phenotypes.append({
                    "hpo_id": hpo_id,
                    "label": hpo_id,
                    "status": "present",
                    "onset": None,
                })
        st.success(f"Imported {len(ids)} HPO terms")
        st.rerun()

# ── Current Phenotype Table ──────────────────────────
st.subheader("Current Phenotypes")
if st.session_state.phenotypes:
    # Editable table
    for i, pheno in enumerate(st.session_state.phenotypes):
        col_id, col_name, col_st, col_onset, col_del = st.columns([2, 3, 2, 2, 1])
        with col_id:
            st.text(pheno["hpo_id"])
        with col_name:
            st.text(pheno["label"])
        with col_st:
            new_status = st.selectbox(
                "Status",
                ["present", "absent", "past_history"],
                index=["present", "absent", "past_history"].index(pheno["status"]),
                key=f"edit_status_{i}",
            )
            st.session_state.phenotypes[i]["status"] = new_status
        with col_onset:
            onset = st.text_input("Onset", value=pheno.get("onset") or "", key=f"edit_onset_{i}", placeholder="P5Y")
            st.session_state.phenotypes[i]["onset"] = onset or None
        with col_del:
            st.write("")
            if st.button("🗑️", key=f"del_{i}"):
                st.session_state.phenotypes.pop(i)
                st.rerun()

    if st.button("Clear All Phenotypes"):
        st.session_state.phenotypes = []
        st.rerun()
else:
    st.info("No phenotypes added yet. Use the tabs above to add HPO terms.")

# ── Additional Clinical Info ─────────────────────────
st.subheader("Additional Information")
col_test, col_result, col_vus = st.columns(3)
with col_test:
    prior_testing = st.selectbox("Prior Genetic Testing", ["none", "panel", "exome", "wgs"])
with col_result:
    test_result = st.selectbox("Test Result", ["", "negative", "positive", "vus"])
with col_vus:
    vus_present = st.checkbox("VUS Present?")

inheritance = st.selectbox(
    "Inheritance Hint (if known)",
    ["", "autosomal_dominant", "autosomal_recessive", "x_linked", "mitochondrial", "de_novo", "unknown"],
)

# ── Gene Results Section ─────────────────────────────
st.subheader("Gene Findings")
st.markdown("Enter genes identified through prior genetic testing, with their classification and inheritance pattern.")

if "gene_results" not in st.session_state:
    st.session_state.gene_results = []

# Add new gene form
with st.expander("➕ Add Gene Finding", expanded=len(st.session_state.gene_results) == 0):
    gcol1, gcol2, gcol3, gcol4 = st.columns([2, 2, 2, 2])
    with gcol1:
        new_gene = st.text_input("Gene Symbol", placeholder="e.g. SCN9A, BRCA1")
    with gcol2:
        new_classification = st.selectbox(
            "Classification",
            ["unknown", "pathogenic", "likely_pathogenic", "vus", "likely_benign", "benign"],
            key="gene_class",
        )
    with gcol3:
        new_inheritance = st.selectbox(
            "Inheritance Pattern",
            ["", "autosomal_dominant", "autosomal_recessive", "x_linked", "mitochondrial", "de_novo", "unknown"],
            key="gene_inh",
        )
    with gcol4:
        new_test_type = st.selectbox(
            "Found via",
            ["", "panel", "exome", "wgs", "targeted"],
            key="gene_test",
        )
    gene_notes = st.text_input("Notes (optional)", placeholder="e.g. compound het, homozygous", key="gene_notes")

    if st.button("Add Gene"):
        if new_gene.strip():
            st.session_state.gene_results.append({
                "gene": new_gene.strip().upper(),
                "classification": new_classification,
                "inheritance": new_inheritance or "",
                "test_type": new_test_type or "",
                "notes": gene_notes or None,
            })
            st.rerun()
        else:
            st.warning("Please enter a gene symbol.")

# Display existing gene findings
if st.session_state.gene_results:
    st.markdown("**Current Gene Findings:**")
    for i, g in enumerate(st.session_state.gene_results):
        gcol_a, gcol_b, gcol_c, gcol_d, gcol_e = st.columns([2, 2, 2, 2, 1])
        with gcol_a:
            st.markdown(f"**{g['gene']}**")
        with gcol_b:
            badge = {
                "pathogenic": "🔴", "likely_pathogenic": "🟠", "vus": "🟡",
                "likely_benign": "🟢", "benign": "🟢", "unknown": "⚪",
            }.get(g["classification"], "⚪")
            st.markdown(f"{badge} {g['classification'].replace('_', ' ').title()}")
        with gcol_c:
            st.text(g.get("inheritance", "").replace("_", " ").title() or "—")
        with gcol_d:
            st.text(g.get("test_type", "").upper() or "—")
        with gcol_e:
            if st.button("🗑️", key=f"del_gene_{i}"):
                st.session_state.gene_results.pop(i)
                st.rerun()
else:
    st.info("No gene findings added. This is optional — add genes if prior testing was performed.")

genes = st.text_input("Legacy: Genes mentioned (comma-separated)", placeholder="BRCA1, TP53")

# ── Build & Store Case ───────────────────────────────
if st.button("💾 Save Case & Get Recommendations", type="primary"):
    case = {
        "age": st.session_state.case_meta["age"],
        "sex": st.session_state.case_meta["sex"],
        "ancestry": st.session_state.case_meta["ancestry"],
        "phenotypes": st.session_state.phenotypes,
        "prior_testing": prior_testing,
        "test_result": test_result or None,
        "vus_present": vus_present,
        "inheritance_hint": inheritance or None,
        "genes_mentioned": [g.strip() for g in genes.split(",") if g.strip()] if genes else [],
        "gene_results": st.session_state.gene_results,
    }
    st.session_state["current_case"] = case
    st.success("Case saved! Navigate to **Recommend** page.")

# ── Case JSON preview ────────────────────────────────
with st.expander("View Case JSON"):
    case_dict = {
        **st.session_state.case_meta,
        "phenotypes": st.session_state.phenotypes,
        "prior_testing": prior_testing,
        "test_result": test_result or None,
        "vus_present": vus_present,
        "inheritance_hint": inheritance or None,
        "gene_results": st.session_state.gene_results,
    }
    st.json(case_dict)

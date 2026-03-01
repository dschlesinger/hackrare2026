"""Silver stage — parse raw bronze data into normalized Parquet files.

Produces:
  - cases.parquet + phenotype_events.parquet   (from phenopackets)
  - diseases.parquet + disease_hpo.parquet     (from ORPHApackets product4)
  - disease_gene.parquet                       (from ORPHApackets product6)
  - id_mapping.parquet                         (from Orphadata alignments + MONDO)
  - hpo_terms.parquet + hpo_synonyms.parquet   (from hp.obo)
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from diageno.config import settings
from diageno.etl.utils import save_parquet

logger = logging.getLogger("diageno.etl.silver")


# ─────────────────────────────────────────────────────
# Phenopacket parsing
# ─────────────────────────────────────────────────────


def _parse_single_phenopacket(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Parse one phenopacket JSON → (case_row, [phenotype_rows])."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, []

    # Extract subject
    subject = data.get("subject", {})
    case_id = data.get("id", path.stem)

    case_row = {
        "case_id": case_id,
        "age": subject.get("ageAtCollection", {}).get("age"),
        "sex": subject.get("sex", "").lower() if subject.get("sex") else None,
        "raw_json": json.dumps(data),
    }

    pheno_rows: list[dict[str, Any]] = []
    for pf in data.get("phenotypicFeatures", []):
        hpo_type = pf.get("type", {})
        hpo_id = hpo_type.get("id", "")
        label = hpo_type.get("label", "")
        excluded = pf.get("excluded", False)
        onset = None
        if "onset" in pf:
            onset_data = pf["onset"]
            if isinstance(onset_data, dict):
                onset = onset_data.get("age", {}).get("iso8601duration")
            elif isinstance(onset_data, str):
                onset = onset_data

        pheno_rows.append({
            "case_id": case_id,
            "hpo_id": hpo_id,
            "label": label,
            "status": "absent" if excluded else "present",
            "onset_iso8601": onset,
            "source": "phenopacket",
        })

    return case_row, pheno_rows


def parse_phenopackets(bronze: Path, silver: Path) -> None:
    """Parse all phenopacket JSONs into cases + phenotype_events parquet."""
    pp_dir = bronze / "phenopacket-store"
    if not pp_dir.exists():
        logger.warning("Phenopacket-store dir not found at %s — skipping", pp_dir)
        return

    json_files = list(pp_dir.rglob("*.json"))
    logger.info("Found %d phenopacket JSON files", len(json_files))

    cases: list[dict[str, Any]] = []
    phenos: list[dict[str, Any]] = []

    for jf in json_files:
        case_row, pheno_rows = _parse_single_phenopacket(jf)
        if case_row:
            cases.append(case_row)
        phenos.extend(pheno_rows)

    if cases:
        save_parquet(pd.DataFrame(cases), silver / "cases.parquet")
    if phenos:
        save_parquet(pd.DataFrame(phenos), silver / "phenotype_events.parquet")


# ─────────────────────────────────────────────────────
# HPO OBO parsing
# ─────────────────────────────────────────────────────


def parse_hpo_obo(bronze: Path, silver: Path) -> None:
    """Parse hp.obo → hpo_terms.parquet + hpo_synonyms.parquet."""
    obo_path = bronze / "hp.obo"
    if not obo_path.exists():
        logger.warning("hp.obo not found at %s — skipping", obo_path)
        return

    try:
        import pronto
    except ImportError:
        logger.error("pronto not installed — cannot parse hp.obo")
        return

    logger.info("Parsing HPO OBO from %s …", obo_path)
    onto = pronto.Ontology(str(obo_path))

    terms: list[dict[str, Any]] = []
    synonyms: list[dict[str, Any]] = []

    for term in onto.terms():
        tid = term.id
        if not tid.startswith("HP:"):
            continue
        terms.append({
            "hpo_id": tid,
            "name": term.name,
            "definition": str(term.definition) if term.definition else None,
            "is_obsolete": 1 if term.obsolete else 0,
        })
        for syn in term.synonyms:
            synonyms.append({
                "hpo_id": tid,
                "synonym": syn.description,
                "synonym_type": syn.scope if hasattr(syn, "scope") else None,
            })

    save_parquet(pd.DataFrame(terms), silver / "hpo_terms.parquet")
    save_parquet(pd.DataFrame(synonyms), silver / "hpo_synonyms.parquet")


# ─────────────────────────────────────────────────────
# Orphadata XML parsing
# ─────────────────────────────────────────────────────


def _safe_xml_parse(path: Path) -> ET.Element | None:
    """Safely parse XML, return root or None."""
    if not path.exists():
        logger.warning("XML file not found: %s", path)
        return None
    try:
        tree = ET.parse(str(path))
        return tree.getroot()
    except ET.ParseError as e:
        logger.error("Failed to parse XML %s: %s", path, e)
        return None


def parse_orphadata_disease_hpo(bronze: Path, silver: Path) -> None:
    """Parse en_product4.xml → diseases.parquet + disease_hpo.parquet."""
    root = _safe_xml_parse(bronze / "orphadata_disease_hpo.xml")
    if root is None:
        return

    diseases: list[dict[str, Any]] = []
    disease_hpo_rows: list[dict[str, Any]] = []

    # Different Orphadata XML formats — try common structures
    for disorder in root.iter("Disorder"):
        orpha_code = ""
        name = ""

        orpha_el = disorder.find("OrphaCode")
        if orpha_el is not None and orpha_el.text:
            orpha_code = f"ORPHA:{orpha_el.text}"

        name_el = disorder.find("Name")
        if name_el is not None and name_el.text:
            name = name_el.text

        if not orpha_code:
            continue

        diseases.append({
            "disease_id": orpha_code,
            "orpha_id": orpha_code,
            "mondo_id": None,
            "name": name,
        })

        # HPO associations
        for assoc in disorder.iter("HPODisorderAssociation"):
            hpo_el = assoc.find(".//HPOId")
            freq_el = assoc.find(".//HPOFrequency/Name")
            hpo_id = hpo_el.text if hpo_el is not None and hpo_el.text else ""
            if not hpo_id:
                continue

            freq_text = freq_el.text if freq_el is not None else None
            freq_val = _freq_text_to_float(freq_text) if freq_text else None

            disease_hpo_rows.append({
                "disease_id": orpha_code,
                "hpo_id": hpo_id,
                "frequency": freq_val,
                "evidence_source": "orphadata_product4",
            })

    if diseases:
        save_parquet(pd.DataFrame(diseases).drop_duplicates("disease_id"), silver / "diseases.parquet")
    if disease_hpo_rows:
        save_parquet(pd.DataFrame(disease_hpo_rows), silver / "disease_hpo.parquet")


def _freq_text_to_float(text: str) -> float | None:
    """Convert Orphadata frequency text to a float."""
    text_lower = text.lower()
    freq_map = {
        "obligate": 1.0,
        "very frequent": 0.8,
        "frequent": 0.5,
        "occasional": 0.25,
        "very rare": 0.05,
        "excluded": 0.0,
    }
    for key, val in freq_map.items():
        if key in text_lower:
            return val
    return None


def parse_orphadata_disease_gene(bronze: Path, silver: Path) -> None:
    """Parse en_product6.xml → disease_gene.parquet."""
    root = _safe_xml_parse(bronze / "orphadata_disease_gene.xml")
    if root is None:
        return

    rows: list[dict[str, Any]] = []
    for disorder in root.iter("Disorder"):
        orpha_el = disorder.find("OrphaCode")
        if orpha_el is None or not orpha_el.text:
            continue
        orpha_code = f"ORPHA:{orpha_el.text}"

        for gene_assoc in disorder.iter("DisorderGeneAssociation"):
            gene_el = gene_assoc.find(".//Gene/Symbol")
            if gene_el is not None and gene_el.text:
                rows.append({
                    "disease_id": orpha_code,
                    "gene_symbol": gene_el.text,
                    "evidence_source": "orphadata_product6",
                })

    if rows:
        save_parquet(pd.DataFrame(rows), silver / "disease_gene.parquet")


def parse_orphadata_alignments(bronze: Path, silver: Path) -> None:
    """Parse alignment XMLs → id_mapping.parquet."""
    rows: list[dict[str, Any]] = []

    # OMIM alignment (product1)
    root = _safe_xml_parse(bronze / "orphadata_orpha_omim.xml")
    if root is not None:
        for disorder in root.iter("Disorder"):
            orpha_el = disorder.find("OrphaCode")
            if orpha_el is None or not orpha_el.text:
                continue
            orpha_code = f"ORPHA:{orpha_el.text}"
            for ref in disorder.iter("ExternalReference"):
                source_el = ref.find("Source")
                ref_el = ref.find("Reference")
                if source_el is not None and ref_el is not None:
                    src = source_el.text or ""
                    ref_val = ref_el.text or ""
                    if "OMIM" in src.upper():
                        rows.append({
                            "orpha_id": orpha_code,
                            "mondo_id": None,
                            "omim_id": f"OMIM:{ref_val}",
                            "icd10": None,
                            "icd11": None,
                            "source": "orphadata_product1",
                        })

    # ICD-10 alignment (product3)
    root = _safe_xml_parse(bronze / "orphadata_orpha_icd10.xml")
    if root is not None:
        for disorder in root.iter("Disorder"):
            orpha_el = disorder.find("OrphaCode")
            if orpha_el is None or not orpha_el.text:
                continue
            orpha_code = f"ORPHA:{orpha_el.text}"
            for ref in disorder.iter("ICD10"):
                code_el = ref.find("Code") if ref.find("Code") is not None else ref
                code_text = code_el.text if code_el is not None and code_el.text else None
                if code_text:
                    rows.append({
                        "orpha_id": orpha_code,
                        "mondo_id": None,
                        "omim_id": None,
                        "icd10": code_text,
                        "icd11": None,
                        "source": "orphadata_product3",
                    })

    if rows:
        save_parquet(pd.DataFrame(rows), silver / "id_mapping.parquet")


# ─────────────────────────────────────────────────────
# MONDO OBO parsing
# ─────────────────────────────────────────────────────


def parse_mondo_obo(bronze: Path, silver: Path) -> None:
    """Parse mondo.obo for MONDO→ORPHA/OMIM xrefs, merge into id_mapping."""
    obo_path = bronze / "mondo.obo"
    if not obo_path.exists():
        logger.warning("mondo.obo not found — skipping")
        return

    try:
        import pronto
    except ImportError:
        logger.error("pronto not installed — cannot parse mondo.obo")
        return

    logger.info("Parsing MONDO OBO …")
    onto = pronto.Ontology(str(obo_path))

    rows: list[dict[str, Any]] = []
    for term in onto.terms():
        if not term.id.startswith("MONDO:"):
            continue
        mondo_id = term.id
        orpha_id = None
        omim_id = None
        for xref in term.xrefs:
            xid = str(xref.id) if hasattr(xref, "id") else str(xref)
            if "Orphanet" in xid or "ORPHA" in xid:
                orpha_id = xid.replace("Orphanet:", "ORPHA:")
            elif "OMIM" in xid:
                omim_id = xid
        if orpha_id or omim_id:
            rows.append({
                "orpha_id": orpha_id,
                "mondo_id": mondo_id,
                "omim_id": omim_id,
                "icd10": None,
                "icd11": None,
                "source": "mondo_obo",
            })

    if rows:
        # Merge with existing id_mapping if present
        existing_path = silver / "id_mapping.parquet"
        df_new = pd.DataFrame(rows)
        if existing_path.exists():
            df_existing = pd.read_parquet(existing_path)
            df_merged = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(
                subset=["orpha_id", "mondo_id", "omim_id"], keep="first"
            )
            save_parquet(df_merged, existing_path)
        else:
            save_parquet(df_new, existing_path)


# ─────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────


def run() -> None:
    """Execute the full silver parsing pipeline."""
    bronze = settings.bronze_dir
    silver = settings.silver_dir
    silver.mkdir(parents=True, exist_ok=True)

    logger.info("=== Silver: parsing bronze → %s ===", silver)

    parse_phenopackets(bronze, silver)
    parse_hpo_obo(bronze, silver)
    parse_orphadata_disease_hpo(bronze, silver)
    parse_orphadata_disease_gene(bronze, silver)
    parse_orphadata_alignments(bronze, silver)
    parse_mondo_obo(bronze, silver)

    logger.info("=== Silver parsing complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()

"""Bronze stage — download raw datasets.

Downloads:
  1. Zenodo phenopacket vignettes (5K+)
  2. HPO ontology (hp.obo)
  3. ORPHApackets (disease ↔ HPO, genes)
  4. Orphadata alignment files (ORPHA ↔ MONDO/OMIM/ICD)
  5. MONDO ontology (mondo.obo)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import httpx

from diageno.config import settings

logger = logging.getLogger("diageno.etl.bronze")

# ── Download URLs ─────────────────────────────────────────────
# Zenodo phenopacket-store (latest release archive)
ZENODO_PHENOPACKET_URL = (
    "https://github.com/monarch-initiative/phenopacket-store/archive/refs/heads/main.zip"
)

# HPO OBO file
HPO_OBO_URL = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"

# ORPHApackets (XML product files from Orphadata)
ORPHADATA_DISEASE_HPO_URL = (
    "https://www.orphadata.com/data/xml/en_product4.xml"
)
ORPHADATA_DISEASE_GENE_URL = (
    "https://www.orphadata.com/data/xml/en_product6.xml"
)

# Orphadata Alignments
ORPHADATA_OMIM_URL = "https://www.orphadata.com/data/xml/en_product1.xml"
ORPHADATA_ICD10_URL = "https://www.orphadata.com/data/xml/en_product3.xml"

# MONDO OBO
MONDO_OBO_URL = (
    "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo"
)


def _download(url: str, dest: Path, *, timeout: float = 300) -> None:
    """Download a URL to a local file (streaming)."""
    if dest.exists():
        logger.info("Already exists, skipping: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    logger.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def download_phenopackets(bronze: Path) -> Path:
    """Download phenopacket-store archive."""
    dest = bronze / "phenopacket-store-main.zip"
    _download(ZENODO_PHENOPACKET_URL, dest)
    # Unzip
    unpack_dir = bronze / "phenopacket-store"
    if not unpack_dir.exists():
        logger.info("Unzipping phenopacket-store …")
        shutil.unpack_archive(str(dest), str(bronze))
        # Rename extracted folder
        extracted = bronze / "phenopacket-store-main"
        if extracted.exists():
            extracted.rename(unpack_dir)
    return unpack_dir


def download_hpo(bronze: Path) -> Path:
    """Download HPO OBO file."""
    dest = bronze / "hp.obo"
    _download(HPO_OBO_URL, dest)
    return dest


def download_orphadata(bronze: Path) -> dict[str, Path]:
    """Download ORPHApacket & alignment XMLs."""
    files: dict[str, Path] = {}
    for name, url in [
        ("disease_hpo", ORPHADATA_DISEASE_HPO_URL),
        ("disease_gene", ORPHADATA_DISEASE_GENE_URL),
        ("orpha_omim", ORPHADATA_OMIM_URL),
        ("orpha_icd10", ORPHADATA_ICD10_URL),
    ]:
        dest = bronze / f"orphadata_{name}.xml"
        try:
            _download(url, dest, timeout=600)
        except Exception as e:
            logger.warning("Failed to download %s (%s) — skipping: %s", name, url, e)
        if dest.exists():
            files[name] = dest
    return files


def download_mondo(bronze: Path) -> Path:
    """Download MONDO OBO file."""
    dest = bronze / "mondo.obo"
    _download(MONDO_OBO_URL, dest)
    return dest


def run(force: bool = False) -> None:
    """Execute the full bronze download pipeline."""
    bronze = settings.bronze_dir
    bronze.mkdir(parents=True, exist_ok=True)

    if force:
        logger.info("Force mode — will re-download all datasets")
        # Clear bronze dir
        for p in bronze.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    logger.info("=== Bronze: downloading raw datasets → %s ===", bronze)

    download_phenopackets(bronze)
    download_hpo(bronze)
    download_orphadata(bronze)
    download_mondo(bronze)

    logger.info("=== Bronze download complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()

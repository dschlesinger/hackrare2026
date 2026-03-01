"""SQLAlchemy ORM models — full data model for Diageno.

Tables:
  A) Patient / timeline (Phenopacket-aligned)
  B) Disease knowledge (MVP KG)
  C) Embeddings (pgvector)
  D) Recommendations + validation
  E) HPO vocabulary
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Shared declarative base for all models."""
    pass


# ────────────────────────────────────────────────────────────
# A) Patient / Timeline
# ────────────────────────────────────────────────────────────


class Case(Base):
    __tablename__ = "case"

    case_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String(256), nullable=True, index=True)
    age = Column(Integer, nullable=True)
    sex = Column(String(20), nullable=True)
    ancestry = Column(String(128), nullable=True)
    raw_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # relationships
    phenotype_events = relationship("PhenotypeEvent", back_populates="case", cascade="all,delete")
    test_events = relationship("TestEvent", back_populates="case", cascade="all,delete")
    variant_events = relationship("VariantEvent", back_populates="case", cascade="all,delete")
    embedding = relationship("CaseEmbedding", back_populates="case", uselist=False)
    recommendation_runs = relationship("RecommendationRun", back_populates="case")


class PhenotypeEvent(Base):
    __tablename__ = "phenotype_event"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("case.case_id", ondelete="CASCADE"), nullable=False)
    hpo_id = Column(String(20), nullable=False, index=True)
    label = Column(String(512), nullable=True)
    status = Column(String(20), nullable=False, default="present")  # present / absent / past_history
    onset_iso8601 = Column(String(64), nullable=True)
    source = Column(String(128), nullable=True)  # e.g. "clinician", "nlp", "vignette"

    case = relationship("Case", back_populates="phenotype_events")

    __table_args__ = (
        Index("ix_phenotype_event_case_hpo", "case_id", "hpo_id"),
    )


class TestEvent(Base):
    __tablename__ = "test_event"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("case.case_id", ondelete="CASCADE"), nullable=False)
    test_type = Column(String(64), nullable=True)   # "genetic_panel", "exome", "imaging", ...
    test_name = Column(String(256), nullable=True)
    result_text = Column(Text, nullable=True)
    structured_result = Column(JSONB, nullable=True)

    case = relationship("Case", back_populates="test_events")


class VariantEvent(Base):
    """Phase-2: variant-level events (ClinVar / VUS workflow)."""
    __tablename__ = "variant_event"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("case.case_id", ondelete="CASCADE"), nullable=False)
    gene = Column(String(64), nullable=True, index=True)
    hgvs = Column(String(256), nullable=True)
    zygosity = Column(String(32), nullable=True)
    clinvar_sig = Column(String(64), nullable=True)
    condition_ids = Column(ARRAY(Text), nullable=True)

    case = relationship("Case", back_populates="variant_events")


# ────────────────────────────────────────────────────────────
# B) Disease Knowledge (MVP KG)
# ────────────────────────────────────────────────────────────


class Disease(Base):
    __tablename__ = "disease"

    disease_id = Column(String(64), primary_key=True)   # internal or MONDO
    mondo_id = Column(String(64), nullable=True, index=True)
    orpha_id = Column(String(64), nullable=True, index=True)
    name = Column(String(512), nullable=False)

    hpo_links = relationship("DiseaseHPO", back_populates="disease", cascade="all,delete")
    gene_links = relationship("DiseaseGene", back_populates="disease", cascade="all,delete")
    embedding = relationship("DiseaseEmbedding", back_populates="disease", uselist=False)


class DiseaseHPO(Base):
    __tablename__ = "disease_hpo"

    disease_id = Column(String(64), ForeignKey("disease.disease_id", ondelete="CASCADE"), primary_key=True)
    hpo_id = Column(String(20), primary_key=True)
    frequency = Column(Float, nullable=True)         # 0.0–1.0 if known
    evidence_source = Column(String(128), nullable=True)

    disease = relationship("Disease", back_populates="hpo_links")

    __table_args__ = (
        Index("ix_disease_hpo_hpo", "hpo_id"),
    )


class DiseaseGene(Base):
    __tablename__ = "disease_gene"

    disease_id = Column(String(64), ForeignKey("disease.disease_id", ondelete="CASCADE"), primary_key=True)
    gene_symbol = Column(String(64), primary_key=True)
    evidence_source = Column(String(128), nullable=True)

    disease = relationship("Disease", back_populates="gene_links")

    __table_args__ = (
        Index("ix_disease_gene_symbol", "gene_symbol"),
    )


class IDMapping(Base):
    __tablename__ = "id_mapping"

    id = Column(Integer, primary_key=True, autoincrement=True)
    orpha_id = Column(String(64), nullable=True, index=True)
    mondo_id = Column(String(64), nullable=True, index=True)
    omim_id = Column(String(64), nullable=True, index=True)
    icd10 = Column(String(32), nullable=True)
    icd11 = Column(String(32), nullable=True)
    source = Column(String(128), nullable=True)

    __table_args__ = (
        UniqueConstraint("orpha_id", "mondo_id", "omim_id", name="uq_id_mapping_triple"),
    )


# ────────────────────────────────────────────────────────────
# C) Embeddings (pgvector)
# ────────────────────────────────────────────────────────────


class CaseEmbedding(Base):
    __tablename__ = "case_embedding"

    case_id = Column(
        UUID(as_uuid=True),
        ForeignKey("case.case_id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding = Column(Vector(384), nullable=False)
    model_name = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="embedding")


class DiseaseEmbedding(Base):
    __tablename__ = "disease_embedding"

    disease_id = Column(
        String(64),
        ForeignKey("disease.disease_id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding = Column(Vector(384), nullable=False)
    model_name = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    disease = relationship("Disease", back_populates="embedding")


# ────────────────────────────────────────────────────────────
# D) Recommendations + Validation
# ────────────────────────────────────────────────────────────


class RecommendationRun(Base):
    __tablename__ = "recommendation_run"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("case.case_id"), nullable=False)
    model_version = Column(String(128), nullable=True)
    inputs_hash = Column(String(128), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="recommendation_runs")
    actions = relationship("RecommendationAction", back_populates="run", cascade="all,delete")


class RecommendationAction(Base):
    __tablename__ = "recommendation_action"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("recommendation_run.run_id", ondelete="CASCADE"), nullable=False)
    stage = Column(String(64), nullable=False)
    rank = Column(Integer, nullable=False)
    action_type = Column(String(64), nullable=False)  # phenotype_question / test / referral / monitoring
    action = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    mrr_weight = Column(Float, nullable=True)
    evidence_tags = Column(ARRAY(Text), nullable=True)

    run = relationship("RecommendationRun", back_populates="actions")


# ────────────────────────────────────────────────────────────
# E) HPO Vocabulary
# ────────────────────────────────────────────────────────────


class HPOTerm(Base):
    __tablename__ = "hpo_term"

    hpo_id = Column(String(20), primary_key=True)
    name = Column(String(512), nullable=False)
    definition = Column(Text, nullable=True)
    is_obsolete = Column(Integer, default=0)

    synonyms = relationship("HPOSynonym", back_populates="term", cascade="all,delete")


class HPOSynonym(Base):
    __tablename__ = "hpo_synonym"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hpo_id = Column(String(20), ForeignKey("hpo_term.hpo_id", ondelete="CASCADE"), nullable=False)
    synonym = Column(String(512), nullable=False)
    synonym_type = Column(String(32), nullable=True)  # EXACT / BROAD / NARROW / RELATED

    term = relationship("HPOTerm", back_populates="synonyms")

    __table_args__ = (
        Index("ix_hpo_synonym_text", "synonym"),
    )

-- =============================================================
-- Diageno — initial schema migration
-- Run inside Postgres after CREATE DATABASE diageno;
-- Requires pgvector extension.
-- =============================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ─── A) Patient / Timeline ──────────────────────────────────

CREATE TABLE IF NOT EXISTS "case" (
    case_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id   VARCHAR(256),
    age           INTEGER,
    sex           VARCHAR(20),
    ancestry      VARCHAR(128),
    raw_json      JSONB,
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_case_external_id ON "case"(external_id);
CREATE INDEX IF NOT EXISTS ix_case_created_at  ON "case"(created_at);

CREATE TABLE IF NOT EXISTS phenotype_event (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id       UUID NOT NULL REFERENCES "case"(case_id) ON DELETE CASCADE,
    hpo_id        VARCHAR(20) NOT NULL,
    label         VARCHAR(512),
    status        VARCHAR(20) NOT NULL DEFAULT 'present',
    onset_iso8601 VARCHAR(64),
    source        VARCHAR(128)
);
CREATE INDEX IF NOT EXISTS ix_pe_hpo       ON phenotype_event(hpo_id);
CREATE INDEX IF NOT EXISTS ix_pe_case_hpo  ON phenotype_event(case_id, hpo_id);

CREATE TABLE IF NOT EXISTS test_event (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id           UUID NOT NULL REFERENCES "case"(case_id) ON DELETE CASCADE,
    test_type         VARCHAR(64),
    test_name         VARCHAR(256),
    result_text       TEXT,
    structured_result JSONB
);

CREATE TABLE IF NOT EXISTS variant_event (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id       UUID NOT NULL REFERENCES "case"(case_id) ON DELETE CASCADE,
    gene          VARCHAR(64),
    hgvs          VARCHAR(256),
    zygosity      VARCHAR(32),
    clinvar_sig   VARCHAR(64),
    condition_ids TEXT[]
);
CREATE INDEX IF NOT EXISTS ix_ve_gene ON variant_event(gene);

-- ─── B) Disease Knowledge ───────────────────────────────────

CREATE TABLE IF NOT EXISTS disease (
    disease_id VARCHAR(64) PRIMARY KEY,
    mondo_id   VARCHAR(64),
    orpha_id   VARCHAR(64),
    name       VARCHAR(512) NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_disease_mondo ON disease(mondo_id);
CREATE INDEX IF NOT EXISTS ix_disease_orpha ON disease(orpha_id);

CREATE TABLE IF NOT EXISTS disease_hpo (
    disease_id      VARCHAR(64) NOT NULL REFERENCES disease(disease_id) ON DELETE CASCADE,
    hpo_id          VARCHAR(20) NOT NULL,
    frequency       DOUBLE PRECISION,
    evidence_source VARCHAR(128),
    PRIMARY KEY (disease_id, hpo_id)
);
CREATE INDEX IF NOT EXISTS ix_dh_hpo ON disease_hpo(hpo_id);

CREATE TABLE IF NOT EXISTS disease_gene (
    disease_id      VARCHAR(64) NOT NULL REFERENCES disease(disease_id) ON DELETE CASCADE,
    gene_symbol     VARCHAR(64) NOT NULL,
    evidence_source VARCHAR(128),
    PRIMARY KEY (disease_id, gene_symbol)
);
CREATE INDEX IF NOT EXISTS ix_dg_gene ON disease_gene(gene_symbol);

CREATE TABLE IF NOT EXISTS id_mapping (
    id        SERIAL PRIMARY KEY,
    orpha_id  VARCHAR(64),
    mondo_id  VARCHAR(64),
    omim_id   VARCHAR(64),
    icd10     VARCHAR(32),
    icd11     VARCHAR(32),
    source    VARCHAR(128),
    UNIQUE (orpha_id, mondo_id, omim_id)
);
CREATE INDEX IF NOT EXISTS ix_idm_orpha ON id_mapping(orpha_id);
CREATE INDEX IF NOT EXISTS ix_idm_mondo ON id_mapping(mondo_id);
CREATE INDEX IF NOT EXISTS ix_idm_omim  ON id_mapping(omim_id);

-- ─── C) Embeddings (pgvector) ───────────────────────────────

CREATE TABLE IF NOT EXISTS case_embedding (
    case_id    UUID PRIMARY KEY REFERENCES "case"(case_id) ON DELETE CASCADE,
    embedding  vector(384) NOT NULL,
    model_name VARCHAR(256),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS disease_embedding (
    disease_id VARCHAR(64) PRIMARY KEY REFERENCES disease(disease_id) ON DELETE CASCADE,
    embedding  vector(384) NOT NULL,
    model_name VARCHAR(256),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- HNSW indexes for ANN search
CREATE INDEX IF NOT EXISTS ix_case_emb_hnsw    ON case_embedding    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS ix_disease_emb_hnsw ON disease_embedding USING hnsw (embedding vector_cosine_ops);

-- ─── D) Recommendations ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS recommendation_run (
    run_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id       UUID NOT NULL REFERENCES "case"(case_id),
    model_version VARCHAR(128),
    inputs_hash   VARCHAR(128),
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_rr_hash ON recommendation_run(inputs_hash);

CREATE TABLE IF NOT EXISTS recommendation_action (
    id            SERIAL PRIMARY KEY,
    run_id        UUID NOT NULL REFERENCES recommendation_run(run_id) ON DELETE CASCADE,
    stage         VARCHAR(64) NOT NULL,
    rank          INTEGER NOT NULL,
    action_type   VARCHAR(64) NOT NULL,
    action        TEXT NOT NULL,
    rationale     TEXT,
    mrr_weight    DOUBLE PRECISION,
    evidence_tags TEXT[]
);

-- ─── E) HPO Vocabulary ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS hpo_term (
    hpo_id      VARCHAR(20) PRIMARY KEY,
    name        VARCHAR(512) NOT NULL,
    definition  TEXT,
    is_obsolete INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS hpo_synonym (
    id           SERIAL PRIMARY KEY,
    hpo_id       VARCHAR(20) NOT NULL REFERENCES hpo_term(hpo_id) ON DELETE CASCADE,
    synonym      VARCHAR(512) NOT NULL,
    synonym_type VARCHAR(32)
);
CREATE INDEX IF NOT EXISTS ix_hpo_syn_text ON hpo_synonym(synonym);

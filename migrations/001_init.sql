-- Schema for grounded QA over PUCT filings.
-- Applied automatically by docker-compose on first start.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE dockets (
    control_number    TEXT PRIMARY KEY,
    case_style        TEXT NOT NULL,
    utility_type      CHAR(1),                 -- E electric, W water, T telephone, O other
    first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE filings (
    id                BIGSERIAL PRIMARY KEY,
    control_number    TEXT NOT NULL REFERENCES dockets(control_number),
    item_number       INT NOT NULL,
    filing_date       DATE,
    filing_party      TEXT,
    item_type         TEXT,                    -- PL, CONF, RFI, ...
    UNIQUE (control_number, item_number)
);
CREATE INDEX filings_party_date_idx ON filings (filing_party, filing_date);

CREATE TABLE documents (
    id                BIGSERIAL PRIMARY KEY,
    filing_id         BIGINT NOT NULL REFERENCES filings(id),
    document_id       TEXT NOT NULL,
    source_url        TEXT,
    filename          TEXT,
    description       TEXT,                    -- e.g. "Pages 101 to 200"
    page_offset       INT NOT NULL DEFAULT 0,  -- added to in-PDF pages for true filing pages
    sha256            TEXT NOT NULL UNIQUE,    -- content identity; the idempotency anchor
    page_count        INT,
    chars_per_page    NUMERIC(10,2),
    has_text_layer    BOOLEAN NOT NULL DEFAULT false,
    extraction_status TEXT NOT NULL,           -- ok | no_text_layer | failed
                                               -- | excluded_pii | excluded_confidential
    extraction_error  TEXT,
    fetched_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (filing_id, document_id)
);

-- Full extracted text, kept so citation offsets can be resolved and audited
-- after the fact without re-parsing the PDF.
CREATE TABLE document_text (
    document_id       BIGINT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    text              TEXT NOT NULL,
    page_spans        JSONB NOT NULL           -- [{page_number, char_start, char_end}, ...]
);

CREATE TABLE chunks (
    id                BIGSERIAL PRIMARY KEY,
    document_id       BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal           INT NOT NULL,
    page_start        INT NOT NULL,
    page_end          INT NOT NULL,
    char_start        INT NOT NULL,            -- offset into document_text.text
    char_end          INT NOT NULL,
    text              TEXT NOT NULL,
    embedding         VECTOR(1024),
    UNIQUE (document_id, ordinal),
    CONSTRAINT chunk_span_valid CHECK (char_end > char_start)
);
CREATE INDEX chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE queries (
    id                BIGSERIAL PRIMARY KEY,
    question          TEXT NOT NULL,
    answer            TEXT,
    refused           BOOLEAN NOT NULL DEFAULT false,
    refusal_reason    TEXT,                    -- no_relevant_context | model_declined
                                               -- | verification_failed | untraceable_numeric
    config_id         TEXT NOT NULL,           -- retrieval/prompt version tag
    top_similarity    REAL,
    latency_ms        INT,
    input_tokens      INT,
    output_tokens     INT,
    cost_usd          NUMERIC(10,6),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX queries_created_idx ON queries (created_at DESC);

CREATE TABLE claims (
    id                BIGSERIAL PRIMARY KEY,
    query_id          BIGINT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    chunk_id          BIGINT REFERENCES chunks(id),
    claim_text        TEXT NOT NULL,
    quoted_span       TEXT NOT NULL,
    span_verified     BOOLEAN NOT NULL,
    numbers_verified  BOOLEAN NOT NULL,
    failure_detail    TEXT
);

CREATE TABLE eval_items (
    id                BIGSERIAL PRIMARY KEY,
    question          TEXT NOT NULL,
    bucket            TEXT NOT NULL,           -- factual | numeric | unanswerable
    answerable        BOOLEAN NOT NULL,
    gold_answer       TEXT,
    gold_document_id  BIGINT REFERENCES documents(id),
    gold_page         INT,
    gold_number       NUMERIC,
    notes             TEXT
);

CREATE TABLE eval_runs (
    id                BIGSERIAL PRIMARY KEY,
    config_id         TEXT NOT NULL,
    git_sha           TEXT,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    metrics           JSONB
);

CREATE TABLE eval_results (
    id                BIGSERIAL PRIMARY KEY,
    run_id            BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    eval_item_id      BIGINT NOT NULL REFERENCES eval_items(id),
    query_id          BIGINT REFERENCES queries(id),
    retrieval_hit     BOOLEAN,
    rank_of_gold      INT,
    numeric_correct   BOOLEAN,
    refused           BOOLEAN,
    correct           BOOLEAN
);
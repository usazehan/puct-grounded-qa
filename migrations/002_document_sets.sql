-- Document sets, extraction verdicts, and citation anchors.
--
-- 001 assumed one served PDF per filing. Item 795 is a 371-page tariff served as
-- four ~100-page PDFs, twice over -- two sets, eight documents, one filing --
-- and item 593 is the same shape at smaller scale. Three consequences:
--
--   1. A set layer between filings and documents. The set is the filing as
--      served; the document is a file.
--   2. Extraction verdicts belong to the SET, not the document. They were
--      measured across the concatenated parts, because comparing one part
--      against the whole native caps accuracy at that part's share of the
--      filing (795 measured 26% that way, 99.4% correctly).
--   3. Presence and retrievability are separate. Both 795 sets are real records
--      and belong in the corpus for provenance, but grounding an answer in the
--      superseded one cites the wrong version of the tariff -- a failure no
--      amount of span or numeric verification catches, because the text is
--      quoted correctly from a document that no longer controls.

CREATE TABLE document_sets (
    id                BIGSERIAL PRIMARY KEY,
    filing_id         BIGINT NOT NULL REFERENCES filings(id),
    set_id            TEXT NOT NULL UNIQUE,    -- 49421_795_a; matches manifest
    -- Which of two refiled sets controls is a reading of the record, not a
    -- property of the bytes. Nothing derives it; a human asserts it.
    status            TEXT NOT NULL DEFAULT 'undetermined',
    -- Defaults to false so an undetermined set cannot ground an answer by
    -- omission. Making it retrievable is an explicit act.
    retrieval_eligible BOOLEAN NOT NULL DEFAULT false,
    selection_note    TEXT,

    -- --- Extraction verdict, from scripts/ocr_accuracy.py --json ---
    -- exact_ok | exact_structure_reported | exact_unverified_structure
    -- | refuse_numerics | partial_pairing | likely_mispaired
    verdict           TEXT,
    -- False means numeric claims citing this set are refused by policy, before
    -- the guard ever runs. True means the guard's per-claim check governs.
    numeric_verifiable BOOLEAN,
    word_accuracy     NUMERIC(6,3),
    numeric_accuracy  NUMERIC(6,3),            -- per occurrence, not per unique token
    -- A LOWER BOUND. The residual mixes real misattribution with naming variance
    -- between the native and the served rendering, and the measurement cannot
    -- separate them. Recorded, never used as a gate.
    association_rate  NUMERIC(6,3),
    -- The native file(s) the verdict was measured against. Null means no ground
    -- truth exists for this set -- 795's refiled batch has no native bundle, so
    -- its numbers are unverified rather than verified-and-clean.
    measured_against  JSONB,
    measured_at       TIMESTAMPTZ,

    CONSTRAINT set_status_known
        CHECK (status IN ('operative', 'superseded', 'undetermined')),
    -- An undetermined set must not be retrievable. Deciding which version of a
    -- tariff controls is exactly the judgement a reader is relying on.
    CONSTRAINT undetermined_not_retrievable
        CHECK (status <> 'undetermined' OR NOT retrieval_eligible)
);
CREATE INDEX document_sets_filing_idx ON document_sets (filing_id);
CREATE INDEX document_sets_eligible_idx ON document_sets (retrieval_eligible)
    WHERE retrieval_eligible;

-- Documents hang off a set. Single-document filings get a one-document set
-- rather than a special case, so retrieval never has two code paths.
ALTER TABLE documents
    ADD COLUMN set_id BIGINT REFERENCES document_sets(id),
    -- Order within the set. Parts concatenate by document ID, which is the
    -- Interchange's own upload order; page-range descriptions are NOT used --
    -- 795 serves two documents both described "Pages 101 to 200" that begin at
    -- different content.
    ADD COLUMN part_ordinal INT NOT NULL DEFAULT 0;
CREATE INDEX documents_set_idx ON documents (set_id, part_ordinal);

-- 001 documented page_offset as "added to in-PDF pages for true filing pages".
-- It is now only ever set by explicit human assertion. Deriving it from the
-- Interchange page-range description silently displaces every citation from a
-- document whose description is wrong, and verify_offset_integrity() cannot
-- detect that: it checks that page spans tile the extracted text, which is
-- internal consistency, not agreement with the record.
COMMENT ON COLUMN documents.page_offset IS
    'Explicitly asserted only; never derived from the filing description. '
    '0 means citations resolve to PDF pages, not record pages.';

-- Per-page citation anchors. A page number is only citable if it can say which
-- scheme produced it: a Bates stamp is a position in the record, a PDF page is
-- a position in a file, and a citation that cannot distinguish them has no
-- business claiming to be verifiable. Item 773 resolves 23 of 26 pages by PDF
-- page alone, so this is the common case for attachments, not an edge case.
CREATE TABLE page_anchors (
    document_id       BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    pdf_page          INT NOT NULL,            -- 1-based page within the file
    anchor_scheme     TEXT NOT NULL,           -- bates | page_label | pdf_page
    anchor_value      TEXT NOT NULL,
    PRIMARY KEY (document_id, pdf_page),
    CONSTRAINT anchor_scheme_known
        CHECK (anchor_scheme IN ('bates', 'page_label', 'pdf_page'))
);
CREATE INDEX page_anchors_weak_idx ON page_anchors (document_id)
    WHERE anchor_scheme = 'pdf_page';

-- Citations carry the scheme so a reader can tell a record position from a file
-- position without going back to the source.
ALTER TABLE claims
    ADD COLUMN anchor_scheme TEXT,
    ADD COLUMN anchor_value TEXT,
    -- Set when a numeric claim was refused because its set is not
    -- numeric_verifiable, as opposed to failing per-claim verification. The
    -- results table needs to separate "the corpus could not support this" from
    -- "the model got it wrong".
    ADD COLUMN refused_by_policy BOOLEAN NOT NULL DEFAULT false;

-- Chunks must not straddle documents. A 371-page filing cut at pages 100/101
-- can put a table's label in one file and its values in the next, and a claim
-- whose support spans both has no expressible citation -- one citation is one
-- document and one span. Such a claim should be refused deliberately rather
-- than half-answered from whichever part was retrieved.
COMMENT ON TABLE chunks IS
    'One chunk belongs to exactly one document. Support that spans a part '
    'boundary within a set is not citable and must be refused.';
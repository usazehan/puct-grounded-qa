-- Chunk layout: what kind of page a chunk came from, and where its header lives.
--
-- 001 modelled a chunk as one contiguous span: text, page_start/page_end,
-- char_start/char_end. Layout-aware chunking breaks that in one specific way.
--
-- A table page opens with a title block -- schedule number, company, test year,
-- effective date, and crucially "(amounts in thousands)". Every chunk cut from
-- that page carries it, because a chunk holding "Operations & Maintenance /
-- 4,231" without it is not merely vague: the verified figure is wrong by a
-- factor of a thousand and the guard cannot catch that, since the digits match
-- exactly.
--
-- So a chunk's text is header + body and is NOT equal to
-- document_text[char_start:char_end]. Both pieces are real spans of the
-- document, and both must be recorded, or a citation cannot resolve the half of
-- the chunk a claim actually rests on.
--
-- The guard must verify a claim against ONE span or the other, never against
-- the concatenation: a claim allowed to draw half its support from the header
-- and half from a row that does not sit under it is the misattribution this
-- whole design exists to prevent.

ALTER TABLE chunks
    -- prose | table. Recorded because the two are chunked by different rules --
    -- prose by character budget, tables on row boundaries -- and a retrieval
    -- failure on one says nothing about the other.
    ADD COLUMN kind TEXT NOT NULL DEFAULT 'prose',
    -- The page header prepended to this chunk, as a span of the document. NULL
    -- for prose chunks, which carry no context.
    ADD COLUMN context_char_start INT,
    ADD COLUMN context_char_end INT,
    -- Anchor of the page this chunk came from, denormalised from page_anchors so
    -- retrieval can return a citation without a join. Weakly-anchored chunks
    -- (scheme 'pdf_page') name a position in a file, not in the record.
    ADD COLUMN anchor_scheme TEXT,
    ADD COLUMN anchor_value TEXT,

    ADD CONSTRAINT chunk_kind_known
        CHECK (kind IN ('prose', 'table')),
    -- Half a header span is worse than none: it would resolve to a slice of the
    -- document that is not the header a reader was shown.
    ADD CONSTRAINT context_span_complete
        CHECK ((context_char_start IS NULL) = (context_char_end IS NULL)),
    ADD CONSTRAINT context_span_ordered
        CHECK (context_char_start IS NULL OR context_char_start < context_char_end);

CREATE INDEX chunks_kind_idx ON chunks (kind);

-- Retrieval runs only over retrieval-eligible sets. Without this path the
-- filter is a join through documents and document_sets on every query, and a
-- filter that is expensive is a filter someone eventually drops.
CREATE INDEX chunks_document_idx ON chunks (document_id);

COMMENT ON COLUMN chunks.context_char_start IS
    'Start of the page header prepended to this chunk. Chunk text is header + '
    'body, so it does not equal document_text[char_start:char_end]. Verify a '
    'claim against one span or the other, never the concatenation.';
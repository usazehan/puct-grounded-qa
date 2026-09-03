-- Track which model produced each embedding.
--
-- 001 already created chunks.embedding as vector(1024), a generated
-- search_vector over the 'simple' text-search configuration, and the GIN
-- indexes for both. Only the provenance columns were missing.
--
-- Without embedding_model, re-embedding under a different model leaves a column
-- holding vectors from two spaces. Cosine distance between them is meaningless
-- rather than merely inaccurate, and nothing downstream would notice: the query
-- returns neighbours, they are simply the wrong ones.

ALTER TABLE chunks
    ADD COLUMN embedding_model TEXT,
    ADD COLUMN embedded_at TIMESTAMPTZ;

COMMENT ON COLUMN chunks.embedding IS
    'Dense vector for semantic retrieval, 1024d. Null until '
    'scripts/embed_chunks.py runs. Always filter on embedding_model before '
    'comparing vectors -- two model spaces in one column produce distances '
    'that are meaningless, not merely inaccurate.';

-- Vector search stays exact. At ~800 chunks an ANN index adds approximation
-- error for no measurable latency gain, and exact search gives an unambiguous
-- reference point when comparing retrieval configurations against each other.
--
--   CREATE INDEX chunks_embedding_idx ON chunks
--       USING hnsw (embedding vector_cosine_ops);
"""Retrieval over the eligible corpus: dense, lexical, and their fusion.

Two regimes with complementary failure modes. A question about what Staff
recommended shares almost no wording with the passage answering it; a question
naming PBRAF or Sheet 6.7.2 needs a literal match, and dense embeddings smooth
exactly the distinctions that matter -- PBRAF against PCRF, 9.4% against 9.45%.

Ranks are fused rather than scores: cosine distance and ts_rank are not on a
common scale, and any weighting between them would be a constant nobody could
defend.

Every query is filtered to retrieval-eligible sets. A set is ineligible until a
human decided which version of the record controls, and grounding an answer in
a superseded tariff is the one failure neither span nor numeric verification
catches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Standard RRF constant. Damps the top of each list so a single arm's first
# result cannot dominate a chunk that both arms rank well.
RRF_K = 60

DEFAULT_ARM_LIMIT = 30
DEFAULT_LIMIT = 10


@dataclass
class Hit:
    chunk_id: int
    text: str
    document: str
    set_id: str
    page_start: int
    kind: str
    anchor_scheme: str
    anchor_value: str
    score: float = 0.0
    # Which arms found this, and where each ranked it. Kept because "the dense
    # arm alone found this at rank 1" and "both arms agreed at rank 3" are
    # different evidence, and the eval needs to tell them apart.
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.document} {self.anchor_value}"

    @property
    def weakly_anchored(self) -> bool:
        # True when the citation names a position in a file, not the record
        return self.anchor_scheme == "pdf_page"


SELECT_COLUMNS = """
    c.id, c.text, d.filename, s.set_id, c.page_start, c.kind,
    c.anchor_scheme, c.anchor_value
"""

ELIGIBLE_JOIN = """
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    JOIN document_sets s ON s.id = d.set_id
    WHERE s.retrieval_eligible
"""


def _hit(row: tuple) -> Hit:
    return Hit(
        chunk_id=row[0],
        text=row[1],
        document=row[2],
        set_id=row[3],
        page_start=row[4],
        kind=row[5],
        anchor_scheme=row[6],
        anchor_value=row[7],
    )


def dense_search(cur, embedding: list[float], limit: int = DEFAULT_ARM_LIMIT) -> list[Hit]:
    # Nearest neighbours by cosine distance

    cur.execute(
        f"""
        SELECT {SELECT_COLUMNS}
        {ELIGIBLE_JOIN}
          AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (str(embedding), limit),
    )
    return [_hit(r) for r in cur.fetchall()]


QUESTION_WORDS = frozenset("""
    a an and are as at be by did do does for from how in is it of on or that
    the to was were what when where which who why will with
""".split())


def lexical_terms(query: str) -> str:
    # Content words of a question, ORed

    words = [w.strip(".,?;:()'\"") for w in query.lower().split()]
    content = [w for w in words if w and w not in QUESTION_WORDS]
    return " | ".join(content) or query


def lexical_search(cur, query: str, limit: int = DEFAULT_ARM_LIMIT) -> list[Hit]:
    # Full-text search over the 'simple' configuration
    terms = lexical_terms(query)
    cur.execute(
        f"""
        SELECT {SELECT_COLUMNS}
        {ELIGIBLE_JOIN}
          AND c.search_vector @@ to_tsquery('simple', %s)
        ORDER BY ts_rank(c.search_vector, to_tsquery('simple', %s)) DESC
        LIMIT %s
        """,
        (terms, terms, limit),
    )
    return [_hit(r) for r in cur.fetchall()]


def trigram_search(cur, term: str, limit: int = 10) -> list[Hit]:
    # Fuzzy character-level match, for identifiers the exact index will miss

    cur.execute(
        f"""
        SELECT {SELECT_COLUMNS}
        {ELIGIBLE_JOIN}
          AND c.text %% %s
        ORDER BY similarity(c.text, %s) DESC
        LIMIT %s
        """,
        (term, term, limit),
    )
    return [_hit(r) for r in cur.fetchall()]


def reciprocal_rank_fusion(
    arms: dict[str, list[Hit]], k: int = RRF_K, limit: int = DEFAULT_LIMIT
) -> list[Hit]:
    # Fuse ranked lists by rank, not by score
    merged: dict[int, Hit] = {}
    for arm, hits in arms.items():
        for rank, hit in enumerate(hits, start=1):
            current = merged.setdefault(hit.chunk_id, hit)
            current.score += 1.0 / (k + rank)
            current.ranks[arm] = rank
    return sorted(merged.values(), key=lambda h: -h.score)[:limit]


def search(
    cur,
    query: str,
    embedding: list[float] | None = None,
    limit: int = DEFAULT_LIMIT,
    arm_limit: int = DEFAULT_ARM_LIMIT,
    use_dense: bool = True,
    use_lexical: bool = True,
    use_trigram: bool = False,
) -> list[Hit]:
    # Run the enabled arms and fuse them

    arms: dict[str, list[Hit]] = {}
    if use_dense and embedding is not None:
        arms["dense"] = dense_search(cur, embedding, arm_limit)
    if use_lexical:
        arms["lexical"] = lexical_search(cur, query, arm_limit)
    if use_trigram:
        arms["trigram"] = trigram_search(cur, query, arm_limit // 3)

    if len(arms) == 1:
        # A single arm needs no fusion, and fusing would only reorder by a constant
        return next(iter(arms.values()))[:limit]
    return reciprocal_rank_fusion(arms, limit=limit)


__all__ = [
    "Hit",
    "dense_search",
    "lexical_search",
    "reciprocal_rank_fusion",
    "search",
    "trigram_search",
]
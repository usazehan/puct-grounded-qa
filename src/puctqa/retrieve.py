"""Retrieval over the eligible corpus: dense, lexical, and their fusion.

Two regimes, complementary failure modes.

"What did Staff recommend on return on equity?" is where a dense embedding
earns its place -- the answering passage says "the signatories agreed to a
Return on Equity of 9.4%" and shares almost no wording with the question.

"What is the residential PBRAF?" is the opposite. PBRAF, LGS, Sheet 6.7.2 and
40.4859% are high-information tokens that have to match literally, and dense
embeddings smooth exactly the distinctions that matter: PBRAF against PCRF,
9.4% against 9.45%. No general embedding model preserves an exact decimal.

So both arms run on every query and their RANKS are fused, not their scores.
Cosine distance and ts_rank are not on a common scale, and any weighting that
mixed them would be a constant nobody could justify. Reciprocal rank fusion
needs no calibration: a chunk that both arms rank highly wins, and a chunk only
one arm finds still surfaces.

EVERY QUERY IS FILTERED TO RETRIEVAL-ELIGIBLE SETS

Not as an optimisation. A set is ineligible until a human decided which version
of the record controls, and grounding an answer in a superseded tariff is the
one failure neither span nor numeric verification can catch: the text would be
quoted correctly from a document that no longer governs.
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
        """True when the citation names a position in a file, not the record."""
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
    """Nearest neighbours by cosine distance.

    Restricted to chunks embedded by the same model as the query. Two model
    spaces in one column produce distances that are meaningless rather than
    merely inaccurate -- the query still returns neighbours, they are simply
    the wrong ones.
    """
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
    """Content words of a question, OR-ed.

    OR rather than AND: a chunk holding "Return on Equity of 9.4%" should rank
    for "what return on equity was approved" without also containing
    'approved'. ts_rank orders by how many terms matched and how often, so the
    chunk matching more of them still wins.
    """
    words = [w.strip(".,?;:()'\"") for w in query.lower().split()]
    content = [w for w in words if w and w not in QUESTION_WORDS]
    return " | ".join(content) or query


def lexical_search(cur, query: str, limit: int = DEFAULT_ARM_LIMIT) -> list[Hit]:
    """Full-text search over the 'simple' configuration.

    'simple' rather than 'english' for the index: stemming and stopword removal
    mangle the tokens these documents turn on. LGS and PBRAF are not English
    words, and a rate class named Standby should not collapse with standing.
    """
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
    """Fuzzy character-level match, for identifiers the exact index will miss.

    The corpus contains "PERIODIC BILLING RE UIREMENT" -- a dropped Q -- and
    "ATIACHMENT D", in documents that otherwise measured 99.8% word accuracy.
    An exact lexical match will sometimes miss a term that is visibly on the
    page, and a docket or sheet number is exactly the kind of token a reader
    would type verbatim.
    """
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
    """Fuse ranked lists by rank, not by score.

    Cosine distance, ts_rank, and trigram similarity are not comparable
    quantities. Any weighted sum of them would need a constant nobody could
    defend, and it would drift the moment a model or a text configuration
    changed. Ranks are comparable by construction.
    """
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
    """Run the enabled arms and fuse them.

    The arms are switchable so the eval can measure each configuration
    separately. The interesting comparison is not which embedding model wins but
    whether hybrid beats either arm alone -- on a corpus where half the queries
    name a rate class and half describe a concept, the prediction is that it
    does, and that is a claim worth measuring rather than assuming.
    """
    arms: dict[str, list[Hit]] = {}
    if use_dense and embedding is not None:
        arms["dense"] = dense_search(cur, embedding, arm_limit)
    if use_lexical:
        arms["lexical"] = lexical_search(cur, query, arm_limit)
    if use_trigram:
        arms["trigram"] = trigram_search(cur, query, arm_limit // 3)

    if len(arms) == 1:
        # A single arm needs no fusion, and fusing would only reorder by a
        # constant.
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
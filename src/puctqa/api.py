"""HTTP service over the answer path.

The same pipeline `scripts/answer.py` runs, behind an endpoint. Nothing here
decides anything: retrieval, verification and refusal all happen in the modules,
and this layer's job is to expose them without softening what they returned.

REFUSAL IS A 200, NOT AN ERROR

A refusal is the system working. It carries a reason, it is recorded, and a
client should read it the same way it reads an answer -- as the considered
output of a question. Returning 4xx would put refusals in the same bucket as
malformed requests, and would tempt a client to retry them.

WHAT THE RESPONSE CARRIES

Every claim goes out with the span it rests on, the document and anchor that
span resolves to, and whether that anchor names a position in the record or
merely in a file. A citation a reader cannot check is not a citation, and 14% of
this corpus resolves only to a PDF page.

The model and configuration are echoed back on every response. Two runs of the
same question can differ -- the eval measures roughly two questions in twenty
flipping at temperature 0 -- so a client holding an answer should be able to say
what produced it.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import psycopg  # noqa: E402

from puctqa.generate import BACKENDS  # noqa: E402

DEFAULT_DSN = os.environ.get(
    "PUCTQA_DSN", "postgresql://puctqa:puctqa@db:5432/puctqa"
)
DEFAULT_BACKEND = os.environ.get("PUCTQA_BACKEND", "echo")
DEFAULT_EMBED_MODEL = os.environ.get(
    "PUCTQA_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)
DEFAULT_TOP_K = int(os.environ.get("PUCTQA_TOP_K", "5"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model once, not per request.

    It is ~1.2GB and takes seconds to load. Doing it per request would make the
    first query of every connection pay for it.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DEFAULT_EMBED_MODEL, trust_remote_code=True)
    state["embed"] = lambda text: model.encode(
        text, normalize_embeddings=True
    ).tolist()
    state["backend"] = BACKENDS[DEFAULT_BACKEND]()
    yield
    state.clear()


app = FastAPI(
    title="Grounded QA over Texas PUC rate filings",
    description=(
        "Every claim is verified against the span it cites before it is "
        "returned. When verification fails, the service refuses."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    review: bool = Field(
        default=False,
        description=(
            "Run the responsiveness and completeness pass. Roughly doubles "
            "cost and adds ~18s; off by default."
        ),
    )


class Citation(BaseModel):
    document: str
    page: int
    anchor_scheme: str
    anchor: str
    weakly_anchored: bool = Field(
        description=(
            "True when the anchor names a position in a file rather than in "
            "the record -- a PDF page rather than a Bates stamp or a printed "
            "page label."
        )
    )


class ClaimOut(BaseModel):
    assertion: str
    quoted_span: str
    citation: Citation


class Usage(BaseModel):
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = Field(
        default=None,
        description=(
            "Null rather than zero when the price is unknown: a local model "
            "has no per-token rate, and 0.0 would read as free rather than "
            "unmeasured."
        ),
    )


class AskResponse(BaseModel):
    question: str
    refused: bool
    refusal_reason: str | None = None
    claims: list[ClaimOut] = []
    caveat: str | None = Field(
        default=None,
        description=(
            "Set when the review pass judged the answer unresponsive or "
            "incomplete. The claims are still verified; the caveat says the "
            "answer may not be what was asked for, or may not be all of it."
        ),
    )
    chunks_retrieved: int
    evidence_units: int
    config: str
    usage: Usage


@app.get("/health")
def health() -> dict:
    """Liveness plus whether the corpus is actually queryable.

    A service that is up over an empty index answers nothing, and reporting
    that as healthy hides the more common failure: migrations applied, chunks
    never built.
    """
    try:
        with psycopg.connect(DEFAULT_DSN, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FILTER (WHERE c.embedding IS NOT NULL),
                       count(DISTINCT s.set_id)
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN document_sets s ON s.id = d.set_id
                WHERE s.retrieval_eligible
                """
            )
            embedded, sets = cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"database unreachable: {exc}")

    return {
        "status": "ok" if embedded else "no_index",
        "embedded_chunks": embedded,
        "retrievable_sets": sets,
        "backend": DEFAULT_BACKEND,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question, or refuse.

    A refusal returns 200. It is the system working, it carries a reason, and a
    client should read it as considered output rather than as an error to retry.
    """
    from answer import answer, config_label, persist

    with psycopg.connect(DEFAULT_DSN) as conn, conn.cursor() as cur:
        result = answer(
            cur,
            request.question,
            state["backend"],
            state["embed"],
            request.top_k,
            do_review=request.review,
        )
        config = config_label(DEFAULT_BACKEND, None, request.top_k)
        persist(cur, request.question, result, config)
        conn.commit()

    return AskResponse(
        question=request.question,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        claims=[
            ClaimOut(
                assertion=record.claim.assertion,
                quoted_span=record.claim.quoted_span,
                citation=Citation(
                    document=record.hit.document,
                    page=record.hit.page_start,
                    anchor_scheme=record.hit.anchor_scheme,
                    anchor=record.hit.anchor_value,
                    weakly_anchored=record.hit.anchor_scheme == "pdf_page",
                ),
            )
            for record in result.verified
        ],
        caveat=result.review.caveat,
        chunks_retrieved=len(result.hits),
        evidence_units=result.units_offered,
        config=config,
        usage=Usage(
            latency_ms=result.latency_ms,
            input_tokens=result.usage.input_tokens or None,
            output_tokens=result.usage.output_tokens or None,
            cost_usd=result.usage.cost_usd,
        ),
    )
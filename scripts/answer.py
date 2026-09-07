#!/usr/bin/env python3
"""Answer a question, or refuse.

The whole path in one place: retrieve chunks from retrieval-eligible sets,
segment them into evidence units, ask a backend to select the units supporting
a claim, verify each claim against the units it selected, and keep only what
survives.

REFUSAL IS THE DEFAULT OUTCOME, NOT AN ERROR PATH

A question is answered when at least one claim passes all three checks. Every
other outcome is a refusal, and each one is recorded with its reason:

    the backend selected no evidence          the corpus is silent, or the
                                              passages disagree with each other
    a claim asserted a figure not in its      the model wrote a number the
      evidence                                evidence does not contain
    a claim's predicate did not match         the evidence says "agreed" and the
      its evidence                            claim says "requested"

The middle two are the model being caught. The first is the model declining,
which is the behaviour the instructions ask for and the thing worth measuring.

WHAT IS PERSISTED

Every claim, verified or not, with the span it rested on and which check failed.
A refusal with no record is indistinguishable from a question nobody asked, and
the results table this project is building toward needs to say how often the
system refused and why.

Usage:
    python scripts/answer.py "What return on equity was approved?"
    python scripts/answer.py --backend anthropic "What is the T&D charge for a 175w metal halide?"
    python scripts/answer.py --backend echo --show-evidence "..."
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg  # noqa: E402

from puctqa.evidence import EvidenceUnit, segment_chunk  # noqa: E402
from puctqa.generate import BACKENDS, ProposedClaim, Usage, propose  # noqa: E402
from puctqa.guard import verify_claim_over_units  # noqa: E402
from puctqa.retrieve import Hit, search  # noqa: E402

DEFAULT_DSN = "postgresql://puctqa:puctqa@localhost:5432/puctqa"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TOP_K = 5


@dataclass
class VerifiedClaim:
    claim: ProposedClaim
    verification: Verification
    hit: Hit

    @property
    def citation(self) -> str:
        return f"{self.hit.document} {self.hit.anchor_value}"

    @property
    def weakly_anchored(self) -> bool:
        """A citation naming a position in a file, not in the record."""
        return self.hit.anchor_scheme == "pdf_page"


@dataclass
class Answer:
    question: str
    verified: list[VerifiedClaim] = field(default_factory=list)
    rejected: list[VerifiedClaim] = field(default_factory=list)
    refusal_reason: str | None = None
    units_offered: int = 0
    hits: list[Hit] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # Wall clock for the whole path -- retrieval, segmentation, generation and
    # verification -- not just the model call. What a reader waits for.
    latency_ms: int = 0
    @property
    def refused(self) -> bool:
        return not self.verified


def units_for(
    cur, hits: list[Hit]
) -> tuple[list[EvidenceUnit], dict[int, Hit], dict[int, str]]:
    """Segment each retrieved chunk, keeping the hit each unit came from.

    Segmentation runs against the document text rather than the chunk's stored
    text, because a unit's offsets must be document offsets -- a citation that
    resolves into a chunk rather than into the filing is not checkable by a
    reader holding the PDF.
    """
    units: list[EvidenceUnit] = []
    by_chunk: dict[int, Hit] = {}
    context: dict[int, str] = {}
    for hit in hits:
        cur.execute(
            """
            SELECT t.text, c.char_start, c.char_end, c.kind,
                   c.context_char_start, c.context_char_end
            FROM chunks c
            JOIN document_text t ON t.document_id = c.document_id
            WHERE c.id = %s
            """,
            (hit.chunk_id,),
        )
        row = cur.fetchone()
        if not row:
            continue
        document_text, char_start, char_end, kind, ctx_start, ctx_end = row
        units.extend(segment_chunk(hit.chunk_id, document_text, char_start, char_end, kind))
        by_chunk[hit.chunk_id] = hit
        if ctx_start is not None:
            # The page header: column labels a reader needs to tell which
            # figure in a row is the charge and which is the lumen rating.
            # Shown to the model, never citable.
            context[hit.chunk_id] = document_text[ctx_start:ctx_end]
    return units, by_chunk, context


def answer(cur, question: str, backend, embed, top_k: int = DEFAULT_TOP_K) -> Answer:
    started = time.perf_counter()
    hits = search(cur, question, embedding=embed(question), limit=top_k)
    if not hits:
        return Answer(
            question,
            refusal_reason="nothing retrieved",
            hits=[],
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    units, by_chunk, context = units_for(cur, hits)
    result = Answer(question, units_offered=len(units), hits=hits)
    if not units:
        result.refusal_reason = "retrieved chunks produced no evidence units"
        return result

    proposal = propose(question, units, backend, context)
    result.usage = proposal.usage
    if proposal.refused:
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        result.refusal_reason = (
            "the evidence does not support an answer"
            if not proposal.invalid_ids
            else f"backend cited ids that were not offered: {proposal.invalid_ids}"
        )
        return result

    for claim in proposal.claims:
        sources = {
            u.chunk_id: by_chunk[u.chunk_id]
            for u in claim.units
            if u.chunk_id in by_chunk
        }
        if not sources:
            continue
        # Each unit paired with the chunk it came from. The composition lives in
        # guard.verify_claim_over_units, not here: joining the units and
        # checking the join was written wrong three times in this file, and the
        # guard's tests could not reach it while it lived in a script.
        evidence = [
            (u.text, by_chunk[u.chunk_id].text)
            for u in claim.units
            if u.chunk_id in by_chunk
        ]
        verification = verify_claim_over_units(claim.assertion, evidence)
        record = VerifiedClaim(claim, verification, next(iter(sources.values())))
        (result.verified if verification.verified else result.rejected).append(record)

    if not result.verified:
        reasons = {r.verification.refusal_reason for r in result.rejected}
        result.refusal_reason = "; ".join(sorted(r for r in reasons if r))
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


def config_label(backend: str, model: str | None, top_k: int) -> str:
    """Which configuration produced this run.

    A free-form label rather than a foreign key, so comparing an ollama run
    against an anthropic one is a GROUP BY rather than a schema change.
    """
    return f"{backend}:{model or 'default'}/k{top_k}"


def persist(cur, question: str, result: Answer, config_id: str) -> None:
    """Record the question and every claim, verified or not.

    A refusal with no record is indistinguishable from a question nobody asked,
    and the results table needs to say how often the system refused and why.
    """
    answer_text = " ".join(r.claim.assertion for r in result.verified) or None
    top = max((h.score for h in result.hits), default=None)
    cur.execute(
        """
        INSERT INTO queries (question, answer, refused, refusal_reason,
                             config_id, top_similarity, latency_ms,
                             input_tokens, output_tokens, cost_usd)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            question, answer_text, result.refused, result.refusal_reason,
            config_id, top, result.latency_ms,
            result.usage.input_tokens or None,
            result.usage.output_tokens or None,
            # None rather than zero when the price is unknown: a local model has
            # no per-token rate, and $0.00 in a results table reads as "free"
            # rather than "not measured".
            result.usage.cost_usd,
        ),
    )
    query_id = cur.fetchone()[0]

    for record in result.verified + result.rejected:
        v = record.verification
        cur.execute(
            """
            INSERT INTO claims (
                query_id, chunk_id, claim_text, quoted_span,
                span_verified, numbers_verified, failure_detail,
                anchor_scheme, anchor_value
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                query_id,
                record.claim.chunk_id,
                record.claim.assertion,
                record.claim.quoted_span,
                v.span_verified,
                v.numbers_verified,
                v.failure_detail,
                record.hit.anchor_scheme,
                record.hit.anchor_value,
            ),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--dsn", default=os.environ.get("PUCTQA_DSN", DEFAULT_DSN))
    ap.add_argument("--backend", choices=list(BACKENDS), default="echo")
    ap.add_argument("--model", default=None, help="backend-specific model name")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--show-evidence", action="store_true")
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args()

    factory = BACKENDS[args.backend]
    backend = factory(model=args.model) if args.model else factory()

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(args.embed_model, trust_remote_code=True)
    embed = lambda text: embedder.encode(  # noqa: E731
        text, normalize_embeddings=True
    ).tolist()

    with psycopg.connect(args.dsn) as conn, conn.cursor() as cur:
        result = answer(cur, args.question, backend, embed, args.top_k)
        if not args.no_persist:
            persist(cur, args.question, result,
                    config_label(args.backend, args.model, args.top_k))
            conn.commit()

    cost = result.usage.cost_usd
    print(f"Q: {args.question}")
    print(f"   {len(result.hits)} chunks retrieved, {result.units_offered} evidence units")
    print(
        f"   {result.latency_ms} ms"
        + (
            f", {result.usage.input_tokens} in / {result.usage.output_tokens} out"
            if result.usage.input_tokens
            else ""
        )
        + (f", ${cost:.5f}" if cost is not None else "")
    )
    print()

    if result.refused:
        print("REFUSED")
        print(f"  {result.refusal_reason}")
        if result.rejected:
            print()
            print("  Claims the backend proposed and the guard rejected:")
            for record in result.rejected:
                print(f"    {record.claim.assertion}")
                print(f"      {record.verification.refusal_reason}")
        return 0

    for record in result.verified:
        print(record.claim.assertion)
        weak = "  [weakly anchored]" if record.weakly_anchored else ""
        print(f"  — {record.citation}{weak}")
        if args.show_evidence:
            for unit in record.claim.units:
                print(f"    [{unit.unit_id}] {unit.text[:160]}")
        print()

    if result.rejected:
        print(f"({len(result.rejected)} further claim(s) failed verification)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
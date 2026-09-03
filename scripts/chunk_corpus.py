#!/usr/bin/env python3
"""Chunk the retrieval-eligible corpus and persist the result.

Separate from ingest.py on purpose. Ingestion is about provenance -- every
served PDF, its text, its page spans, its anchors -- and runs over the whole
corpus. Chunking is about what can be retrieved, runs only over sets a human
marked eligible, and gets re-run whenever a chunking rule changes. Item 795's
tariff alone re-chunks four times faster than the 109-document ingest, and
coupling them would mean re-extracting the corpus to test a threshold.

Chunks are replaced per document, not merged: a chunking change invalidates
every chunk from that document, and leaving stale ones behind would let
retrieval return a citation computed under rules that no longer hold.

Usage:
    python scripts/chunk_corpus.py
    python scripts/chunk_corpus.py --dry-run
    python scripts/chunk_corpus.py --set 49421_795_a
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg  # noqa: E402

from puctqa.chunk import chunk_document, summarize, verify_chunk_spans  # noqa: E402
from puctqa.extract import extract_document  # noqa: E402

DEFAULT_DSN = "postgresql://puctqa:puctqa@localhost:5432/puctqa"


def eligible_documents(cur, set_id: str | None) -> list[tuple[int, str, str]]:
    """(document id, filename, set_id) for every retrievable document.

    The eligibility filter lives here rather than in a later retrieval query.
    A set is not retrievable until someone decided which version of the record
    controls, and embedding an ineligible chunk makes it findable no matter what
    the query filter says afterwards.
    """
    cur.execute(
        """
        SELECT d.id, d.filename, s.set_id
        FROM documents d
        JOIN document_sets s ON s.id = d.set_id
        WHERE s.retrieval_eligible
          AND (%s::text IS NULL OR s.set_id = %s::text)
        ORDER BY s.set_id, d.part_ordinal
        """,
        (set_id, set_id),
    )
    return cur.fetchall()


def persist(cur, document_db_id: int, chunks: list) -> None:
    # Replace rather than merge. A chunking change invalidates every chunk from
    # this document, and a stale chunk is a citation computed under rules that
    # no longer apply.
    cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_db_id,))
    if not chunks:
        return
    cur.executemany(
        """
        INSERT INTO chunks (
            document_id, ordinal, text, char_start, char_end,
            page_start, page_end, kind,
            context_char_start, context_char_end, anchor_scheme, anchor_value
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        [
            (
                document_db_id,
                c.ordinal,
                c.text,
                c.char_start,
                c.char_end,
                c.page_start,
                c.page_end,
                c.kind.value,
                c.context_char_start,
                c.context_char_end,
                c.anchor_scheme,
                c.anchor_value,
            )
            for c in chunks
        ],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--dsn", default=os.environ.get("PUCTQA_DSN", DEFAULT_DSN))
    ap.add_argument("--set", dest="set_id", default=None, help="one set_id only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    totals: Counter = Counter()
    all_chunks: list = []

    with psycopg.connect(args.dsn) as conn, conn.cursor() as cur:
        documents = eligible_documents(cur, args.set_id)
        if not documents:
            print("No retrieval-eligible documents.")
            print("A set is not retrievable until someone records which version of")
            print("the record controls -- set status and retrieval_eligible in the")
            print("manifest, then re-run ingest.")
            return 1

        for db_id, filename, set_id in documents:
            path = args.root / filename
            extracted = extract_document(path.read_bytes())
            chunks = chunk_document(extracted, filename)
            # The same role verify_offset_integrity plays for pages. A chunk
            # whose offsets do not resolve cannot produce a citation a human can
            # check; one that resolves to the WRONG span produces a citation that
            # looks checkable and is not.
            verify_chunk_spans(extracted, chunks)
            persist(cur, db_id, chunks)

            all_chunks.extend(chunks)
            totals[set_id] += len(chunks)
            print(f"  {filename:<26} {len(chunks):>4} chunks  ({set_id})")

        if args.dry_run:
            conn.rollback()
            print("\nDry run: rolled back.")
        else:
            conn.commit()

    stats = summarize(all_chunks)
    print()
    for set_id, n in sorted(totals.items()):
        print(f"  {set_id:<16} {n:>5} chunks")
    print()
    print(f"Chunks:           {stats['chunks']}")
    print(f"  prose:          {stats['prose']}")
    print(f"  table:          {stats['table']}")
    print(f"Median chars:     {stats['median_chars']}")
    print(f"Max chars:        {stats['max_chars']}")
    print(
        f"Weakly anchored:  {stats['weakly_anchored']} "
        f"({stats['weakly_anchored'] / max(stats['chunks'], 1) * 100:.0f}%)"
    )
    print("  Those cite a position in a file, not in the record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
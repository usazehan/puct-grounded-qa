#!/usr/bin/env python3
"""Embed persisted chunks for dense retrieval.

Runs over chunks already in the database, so chunking rules and embedding
choices can change independently -- re-chunking is an extraction pass over ten
PDFs, re-embedding is a forward pass over ~800 short texts, and coupling them
means paying for both whenever either changes.

MODEL

Qwen3-Embedding-0.6B at 1024 dimensions, which is the width 001_init.sql chose
for the column. The model is 600M parameters -- larger than strictly necessary
for ~800 chunks, but still something a laptop runs on CPU, which the 4B and 8B
variants are not. The README's quickstart has to stay true.

Dimension is not tuned down from 1024. The corpus is small enough that shaving
it would be an aesthetic choice rather than an empirical one, and Matryoshka
truncation should be justified by a measurement, not assumed.

The corpus is ~800 chunks. Nothing here is at a scale where the model choice
dominates; the ablation in evals/ is what should settle it, and this script
records embedding_model on every row so two configurations can be compared
without guessing which vectors came from where.

Usage:
    python scripts/embed_chunks.py
    python scripts/embed_chunks.py --dry-run
    python scripts/embed_chunks.py --model BAAI/bge-base-en-v1.5 --force
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg  # noqa: E402

DEFAULT_DSN = "postgresql://puctqa:puctqa@localhost:5432/puctqa"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EXPECTED_DIM = 1024
BATCH = 32


def pending(cur, model: str, force: bool) -> list[tuple[int, str]]:
    """Chunks needing an embedding under this model.

    A chunk embedded by a different model counts as pending: mixing vector
    spaces in one column produces distances that are meaningless rather than
    merely inaccurate, and nothing downstream would notice.
    """
    if force:
        cur.execute("SELECT id, text FROM chunks ORDER BY id")
    else:
        cur.execute(
            """
            SELECT id, text FROM chunks
            WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %s
            ORDER BY id
            """,
            (model,),
        )
    return cur.fetchall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("PUCTQA_DSN", DEFAULT_DSN))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--force", action="store_true", help="re-embed everything")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "pip install sentence-transformers\n"
            "The model downloads on first use (~600MB) and runs on CPU."
        )

    with psycopg.connect(args.dsn) as conn, conn.cursor() as cur:
        rows = pending(cur, args.model, args.force)
        if not rows:
            print(f"Every chunk already embedded with {args.model}.")
            return 0
        print(f"{len(rows)} chunks to embed with {args.model}")

        model = SentenceTransformer(args.model, trust_remote_code=True)
        dim = model.get_sentence_embedding_dimension()
        if dim != EXPECTED_DIM:
            # The column is vector(768). A model of another width would fail on
            # insert anyway; failing here says why.
            raise SystemExit(
                f"{args.model} produces {dim}-dimensional vectors but the "
                f"embedding column is vector({EXPECTED_DIM}). Change the column "
                f"in a migration and re-embed the whole table -- vectors of "
                f"different widths cannot coexist."
            )

        for start in range(0, len(rows), args.batch):
            batch = rows[start : start + args.batch]
            vectors = model.encode(
                [text for _, text in batch],
                normalize_embeddings=True,  # cosine distance via inner product
                show_progress_bar=False,
            )
            cur.executemany(
                """
                UPDATE chunks
                SET embedding = %s, embedding_model = %s, embedded_at = now()
                WHERE id = %s
                """,
                [
                    (str(vector.tolist()), args.model, chunk_id)
                    for (chunk_id, _), vector in zip(batch, vectors)
                ],
            )
            print(f"  {min(start + args.batch, len(rows)):>5} / {len(rows)}")

        if args.dry_run:
            conn.rollback()
            print("\nDry run: rolled back.")
        else:
            conn.commit()

    print(f"\nEmbedded {len(rows)} chunks with {args.model} ({dim}d).")
    print("Vector search is exact -- no ANN index at this corpus size, so")
    print("retrieval results are a reference point rather than an approximation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
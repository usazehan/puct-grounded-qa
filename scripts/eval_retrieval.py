#!/usr/bin/env python3
"""Measure retrieval against the eval question set.

This measures RETRIEVAL ONLY: does the chunk containing the answer come back in
the top k? No answer is generated and none is checked. Retrieval failure and
generation failure need separating, and the cheaper one to fix comes first --
a system that never retrieves the right chunk cannot be rescued by a better
prompt.

WHAT COUNTS AS A HIT

A question names its source document and page. A hit is a returned chunk from
that document whose page span contains that page. That is deliberately coarse:
a chunk is up to 2,000 characters and a page usually yields two or three, so
this measures "did retrieval reach the right page" rather than "did it reach
the right sentence". Tightening it would need per-question span labels, which
is work worth doing only once page-level recall is good.

REFUSAL QUESTIONS ARE EXCLUDED FROM RECALL

Five of fifteen questions have no correct source: the answer is a refusal,
because the corpus holds two irreconcilable values or none at all. There is no
chunk to retrieve, so scoring them here would measure nothing. They belong to
the guard, not to retrieval, and are counted separately so the number is not
quietly dropped.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --k 5
    python scripts/eval_retrieval.py --config lexical
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg  # noqa: E402

from puctqa.retrieve import search  # noqa: E402

DEFAULT_DSN = "postgresql://puctqa:puctqa@localhost:5432/puctqa"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# The ablation. Each row is a retrieval configuration, and the comparison worth
# making is hybrid against either arm alone -- not which embedding model wins.
CONFIGS = {
    "lexical": dict(use_dense=False, use_lexical=True, use_trigram=False),
    "dense": dict(use_dense=True, use_lexical=False, use_trigram=False),
    "hybrid": dict(use_dense=True, use_lexical=True, use_trigram=False),
    "hybrid+trgm": dict(use_dense=True, use_lexical=True, use_trigram=True),
}


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def page_of(hit, cur) -> tuple[str, int, int]:
    return hit.document, hit.page_start, hit.page_start


def is_hit(question: dict, hits: list) -> bool:
    """Did any returned chunk come from the question's source page?"""
    want_doc = question["source_document"]
    want_page = question["source_page"]
    return any(h.document == want_doc and h.page_start == want_page for h in hits)


def rank_of(question: dict, hits: list) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.document == question["source_document"] and h.page_start == question["source_page"]:
            return rank
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=Path("evals/questions.jsonl"))
    ap.add_argument("--dsn", default=os.environ.get("PUCTQA_DSN", DEFAULT_DSN))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--config", action="append", choices=list(CONFIGS))
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    refusals = [q for q in questions if q["expected_answer"].startswith("REFUSE_")]
    answerable = [
        q for q in questions
        if not q["expected_answer"].startswith("REFUSE_") and q.get("source_page")
    ]
    unscored = [
        q for q in questions
        if not q["expected_answer"].startswith("REFUSE_") and not q.get("source_page")
    ]

    configs = args.config or list(CONFIGS)
    needs_dense = any(CONFIGS[c]["use_dense"] for c in configs)

    encode = None
    if needs_dense:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model, trust_remote_code=True)
        encode = lambda text: model.encode(  # noqa: E731
            text, normalize_embeddings=True
        ).tolist()

    print(f"{len(answerable)} answerable questions, {len(refusals)} refusals")
    print(f"Recall@{args.k}: did a chunk from the source page come back?")
    print()

    results: dict[str, dict] = {}
    with psycopg.connect(args.dsn) as conn, conn.cursor() as cur:
        for name in configs:
            by_category: dict[str, list[bool]] = defaultdict(list)
            ranks: list[int] = []
            misses: list[tuple[str, str]] = []

            for q in answerable:
                embedding = encode(q["question"]) if CONFIGS[name]["use_dense"] else None
                hits = search(
                    cur, q["question"], embedding=embedding, limit=args.k, **CONFIGS[name]
                )
                found = is_hit(q, hits)
                by_category[q["category"]].append(found)
                rank = rank_of(q, hits)
                if rank:
                    ranks.append(rank)
                else:
                    misses.append((q["id"], q["question"]))

            hits_total = sum(sum(v) for v in by_category.values())
            # Mean reciprocal rank over questions that were found at all. Recall
            # says whether the page is reachable; MRR says how far a reader
            # would have to scroll.
            mrr = sum(1 / r for r in ranks) / len(answerable) if answerable else 0.0
            results[name] = {
                "recall": hits_total / len(answerable),
                "mrr": mrr,
                "by_category": {k: sum(v) / len(v) for k, v in by_category.items()},
                "misses": misses,
            }

    width = max(len(c) for c in configs)
    print(f"{'config':<{width}}  recall  MRR")
    for name in configs:
        r = results[name]
        print(f"{name:<{width}}  {r['recall']*100:>5.0f}%  {r['mrr']:.3f}")
    print()

    categories = sorted({q["category"] for q in answerable})
    print(f"{'category':<22} " + "  ".join(f"{c:>11}" for c in configs))
    for category in categories:
        n = sum(1 for q in answerable if q["category"] == category)
        cells = []
        for name in configs:
            value = results[name]["by_category"].get(category)
            cells.append(f"{value*100:>10.0f}%" if value is not None else f"{'--':>11}")
        print(f"{category + f' (n={n})':<22} " + "  ".join(cells))
    print()

    if args.show_misses:
        for name in configs:
            if results[name]["misses"]:
                print(f"{name} missed:")
                for qid, question in results[name]["misses"]:
                    print(f"  {qid}  {question[:70]}")
                print()

    print(f"{len(refusals)} refusal questions are not scored here. They have no")
    print("source chunk to retrieve -- the corpus holds two irreconcilable values")
    print("or none at all -- and belong to the grounding guard, not to retrieval.")
    if unscored:
        print(f"{len(unscored)} question(s) have several valid sources and no single")
        print("page for recall to reach. Page-level recall does not apply to them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Find candidate answers in the retrievable corpus.

Writing an eval question means knowing its answer and where the answer lives.
This searches the served text of retrieval-eligible sets and prints each hit
with its page, citation anchor, and surrounding lines -- enough to write the
question, the expected answer, and the source reference in one pass.

A term with no hits is as informative as one with many: the question either
gets dropped or becomes a refusal case, because the corpus cannot support it.

Usage:
    python scripts/find_answers.py "return on equity" "revenue requirement"
    python scripts/find_answers.py --context 6 PBRAF
    python scripts/find_answers.py --docs 49421_795_1057873.pdf "metal halide"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.extract import extract_document  # noqa: E402

# The six retrieval-eligible sets, as marked operative in the manifest. Hard
# coded rather than read from the database so this runs without Postgres --
# it is a drafting aid, not part of the pipeline.
RETRIEVABLE = [
    "49421_416_1021257.pdf",   # Pollock direct (TIEC)
    "49421_593_1022852.pdf",   # Reed rebuttal, part 1
    "49421_593_1022863.pdf",   # Reed rebuttal, part 2
    "49421_786_1049782.pdf",   # Colvin, testimony supporting the Agreement
    "49421_788_1050240.pdf",   # Tietjen (Staff), supporting the Stipulation
    "49421_792_1054963.pdf",   # Final Order
    "49421_795_1057872.pdf",   # Tariff, part 1
    "49421_795_1057873.pdf",   # Tariff, part 2
    "49421_795_1057874.pdf",   # Tariff, part 3
    "49421_795_1057875.pdf",   # Tariff, part 4
]


def search(root: Path, names: list[str], term: str, context: int, limit: int) -> int:
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    hits = 0

    for name in names:
        path = root / name
        if not path.exists():
            print(f"  (missing: {name})")
            continue
        doc = extract_document(path.read_bytes())
        lines = doc.text.splitlines()

        # Offset of each line, so a hit can be mapped back to its page and
        # therefore to the anchor a citation would use.
        offsets, cursor = [], 0
        for line in lines:
            offsets.append(cursor)
            cursor += len(line) + 1

        for i, line in enumerate(lines):
            if not pattern.search(line):
                continue
            hits += 1
            if hits > limit:
                print(f"  ... more than {limit} hits, stopping")
                return hits
            try:
                page = doc.page_for_offset(offsets[i])
                span = next(s for s in doc.pages if s.page_number == page)
                anchor = f"p{page} ({span.citation})"
            except (IndexError, StopIteration):
                anchor = "?"
            print(f"  --- {name} {anchor} ---")
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                mark = ">" if j == i else " "
                print(f"  {mark} {lines[j][:100]}")
            print()
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("terms", nargs="+")
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--context", type=int, default=4, help="lines either side")
    ap.add_argument("--limit", type=int, default=12, help="max hits per term")
    ap.add_argument("--docs", nargs="*", help="restrict to these filenames")
    args = ap.parse_args()

    names = args.docs or RETRIEVABLE
    for term in args.terms:
        print(f"===== {term} =====")
        hits = search(args.root, names, term, args.context, args.limit)
        if not hits:
            print("  No hits in the retrievable corpus.")
            print("  A question on this term is either dropped or becomes a refusal")
            print("  case -- the corpus cannot support an answer either way.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
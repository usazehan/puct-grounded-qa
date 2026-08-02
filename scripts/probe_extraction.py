#!/usr/bin/env python3
"""Probe a folder of PDFs before building anything on top of them.

Run this on your first hand-downloaded documents. It answers the two questions
that determine whether the project's core assumption holds:

  1. What fraction of pages are born-digital?
  2. Where does the chars-per-page threshold actually separate real text from
     scanned imagery in THIS corpus?

Do not hardcode a threshold until you have seen this output. The default of 200
is a starting guess, not a finding.

Usage:
    python scripts/probe_extraction.py data/raw
    python scripts/probe_extraction.py data/raw --threshold 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.extract import (  # noqa: E402
    DEFAULT_SCAN_THRESHOLD_CHARS_PER_PAGE,
    extract_document,
    verify_offset_integrity,
)
from puctqa.sources import DocumentRef  # noqa: E402


def histogram(values: list[float], buckets: tuple[float, ...]) -> str:
    lines = []
    prev = 0.0
    for edge in buckets:
        count = sum(1 for v in values if prev <= v < edge)
        bar = "#" * min(count, 50)
        lines.append(f"  {prev:>7.0f}-{edge:<7.0f} {count:>4}  {bar}")
        prev = edge
    count = sum(1 for v in values if v >= prev)
    lines.append(f"  {prev:>7.0f}+        {count:>4}  {'#' * min(count, 50)}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--threshold", type=int, default=DEFAULT_SCAN_THRESHOLD_CHARS_PER_PAGE)
    args = ap.parse_args()

    paths = sorted(p for p in args.folder.glob("*") if p.suffix.lower() == ".pdf")
    if not paths:
        print(f"No PDFs found in {args.folder}", file=sys.stderr)
        return 1

    print(f"Probing {len(paths)} documents in {args.folder}\n")
    print(f"{'file':<34} {'pages':>6} {'chars/pg':>9} {'status':<14} {'parsed':<6}")
    print("-" * 76)

    total_pages = 0
    digital_pages = 0
    cpp_values: list[float] = []
    integrity_failures: list[str] = []
    unparsed_names: list[str] = []

    for path in paths:
        extracted = extract_document(path.read_bytes(), scan_threshold=args.threshold)

        try:
            DocumentRef.from_filename(path.name)
            parsed = "yes"
        except ValueError:
            parsed = "NO"
            unparsed_names.append(path.name)

        try:
            verify_offset_integrity(extracted)
        except AssertionError as exc:
            integrity_failures.append(f"{path.name}: {exc}")

        total_pages += extracted.page_count
        if extracted.has_text_layer:
            digital_pages += extracted.page_count
        if extracted.page_count:
            cpp_values.append(extracted.chars_per_page)

        print(
            f"{path.name[:33]:<34} {extracted.page_count:>6} "
            f"{extracted.chars_per_page:>9.1f} {extracted.status.value:<14} {parsed:<6}"
        )

    coverage = (digital_pages / total_pages * 100) if total_pages else 0.0

    print("\n" + "=" * 76)
    print(f"Documents:            {len(paths)}")
    print(f"Total pages:          {total_pages}")
    print(f"Born-digital pages:   {digital_pages}  ({coverage:.1f}% coverage)")
    print(f"Threshold used:       {args.threshold} chars/page")

    if cpp_values:
        print("\nchars-per-page distribution:")
        print(histogram(cpp_values, (50, 100, 200, 500, 1000, 2000, 4000)))
        print(
            "\nLook for a gap between the scanned cluster (near zero) and the\n"
            "text cluster. Put the threshold in the gap. If there is no gap,\n"
            "your corpus is mixed and needs per-page rather than per-document\n"
            "handling."
        )

    if unparsed_names:
        print(f"\nWARNING: {len(unparsed_names)} filenames did not match the")
        print("Interchange pattern {control}_{item}_{docid}.PDF:")
        for name in unparsed_names[:10]:
            print(f"  - {name}")
        print("Rename them or add manifest.json entries before ingesting.")

    if integrity_failures:
        print(f"\nOFFSET INTEGRITY FAILURES ({len(integrity_failures)}) -- fix before proceeding:")
        for failure in integrity_failures:
            print(f"  - {failure}")
        return 1

    print("\nOffset integrity: OK on all documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

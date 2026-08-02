#!/usr/bin/env python3
"""Measure OCR accuracy against native source files.

The Interchange serves scanned-and-OCR'd PDFs, and separately offers "Native
Files (Zip)" containing the original .docx for many filings. Where both exist,
the native file is ground truth and the PDF text is the thing being measured.

This produces the number that justifies the verification design:

  - Numeric tokens are what the grounding guard checks exactly. If OCR
    corrupted digits, exact numeric verification would be unsound.
  - Prose is what span verification checks. OCR ligature errors (rn -> m) mean
    span matching must be fuzzy rather than exact.

Measuring beats asserting. Run this over every PDF/native pair you have and put
the table in the README.

Layout expected:
    data/raw/49421_788_1050240.PDF
    data/native/49421_788_*.docx        (any name; matched by control_item prefix)

Usage:
    python scripts/ocr_accuracy.py data/raw data/native
    python scripts/ocr_accuracy.py data/raw data/native --show-errors
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.extract import extract_document  # noqa: E402
from puctqa.sources import DocumentRef  # noqa: E402

WORD_RE = re.compile(r"[A-Za-z]{3,}")
# Dollar amounts, percentages, and bare figures -- the tokens the numeric
# verifier will compare exactly.
NUM_RE = re.compile(r"\$?\d[\d,]*\.?\d*%?")


def read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        raise SystemExit("pip install python-docx to run this script")
    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def word_types(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text)}


def numeric_tokens(text: str, min_len: int = 4) -> set[str]:
    return {t for t in NUM_RE.findall(text) if len(t) >= min_len}


def find_native(native_dir: Path, ref: DocumentRef) -> Path | None:
    prefix = f"{ref.control_number}_{ref.item_number}_"
    for path in native_dir.rglob("*.docx"):
        if path.name.startswith("~$"):  # Word lock file
            continue
        if prefix in path.name or ref.control_number in path.name:
            return path
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_dir", type=Path)
    ap.add_argument("native_dir", type=Path)
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args()

    pairs: list[tuple[Path, Path]] = []
    for pdf_path in sorted(args.pdf_dir.glob("*.[pP][dD][fF]")):
        try:
            ref = DocumentRef.from_filename(pdf_path.name)
        except ValueError:
            continue
        native = find_native(args.native_dir, ref)
        if native:
            pairs.append((pdf_path, native))

    if not pairs:
        print(f"No PDF/native pairs found between {args.pdf_dir} and {args.native_dir}")
        print("Download the 'Native Files (Zip)' for a few items and extract the .docx.")
        return 1

    print(f"{'document':<30} {'words':>7} {'word acc':>9} {'nums':>6} {'num acc':>8}")
    print("-" * 64)

    total_words = total_words_ok = 0
    total_nums = total_nums_ok = 0
    all_errors: list[tuple[str, str]] = []

    for pdf_path, native_path in pairs:
        ocr_text = extract_document(pdf_path.read_bytes()).text
        native_text = read_docx(native_path)

        nw, ow = word_types(native_text), word_types(ocr_text)
        missing_words = nw - ow
        words_ok = len(nw) - len(missing_words)

        nn, on = numeric_tokens(native_text), numeric_tokens(ocr_text)
        missing_nums = nn - on
        nums_ok = len(nn) - len(missing_nums)

        total_words += len(nw)
        total_words_ok += words_ok
        total_nums += len(nn)
        total_nums_ok += nums_ok
        all_errors.extend((pdf_path.name, w) for w in sorted(missing_words))

        w_acc = words_ok / len(nw) * 100 if nw else 100.0
        n_acc = nums_ok / len(nn) * 100 if nn else 100.0
        print(
            f"{pdf_path.name[:29]:<30} {len(nw):>7} {w_acc:>8.1f}% "
            f"{len(nn):>6} {n_acc:>7.1f}%"
        )

    w_overall = total_words_ok / total_words * 100 if total_words else 0.0
    n_overall = total_nums_ok / total_nums * 100 if total_nums else 0.0

    print("=" * 64)
    print(f"Documents compared:     {len(pairs)}")
    print(f"Word-type accuracy:     {w_overall:.2f}%  ({total_words_ok}/{total_words})")
    print(f"Numeric-token accuracy: {n_overall:.2f}%  ({total_nums_ok}/{total_nums})")
    print()

    if n_overall == 100.0:
        print("Numeric tokens survive OCR intact -> exact numeric verification is sound.")
    else:
        print("WARNING: OCR corrupted numeric tokens. Exact numeric verification is")
        print("NOT sound on this corpus. Either restrict the corpus to documents with")
        print("native files, or treat numeric claims as unverifiable and refuse them.")

    if w_overall < 100.0:
        print("Prose shows OCR errors -> span verification must be fuzzy, not exact.")

    if args.show_errors and all_errors:
        print(f"\nWords present in native text but absent from OCR ({len(all_errors)}):")
        for doc, word in all_errors[:40]:
            print(f"  {doc:<30} {word}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""PDF text extraction with page mapping and scanned-document detection.

Two invariants this module exists to uphold:

1. Every character in the extracted document text can be mapped back to a page.
   Citations are worthless if they resolve to "somewhere in this PDF".

2. Extraction is honest about failure. A scanned page yields no text layer, and
   the pipeline must record that rather than silently indexing an empty string.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

import fitz  # PyMuPDF

# Below this many characters per page on average, assume the document is scanned
# imagery with no text layer. Calibrate against your own corpus before trusting
# it -- see scripts/probe_extraction.py, which prints the distribution.
DEFAULT_SCAN_THRESHOLD_CHARS_PER_PAGE = 200


class ExtractionStatus(str, Enum):
    OK = "ok"
    NO_TEXT_LAYER = "no_text_layer"
    FAILED = "failed"
    EXCLUDED_PII = "excluded_pii"


# Every page of a filed document carries a Bates stamp (0000001, 0000002, ...).
# It is unique, monotonic across the whole filing, and it is what a lawyer
# actually cites -- "at 0000004", not "page 4 of the PDF". It is therefore the
# citation anchor, not the PDF page index.
#
# The PDF page index is wrong twice over: it counts the barcode cover sheet and
# the file-stamped title page, and the document's own internal labels restart at
# attachments ("Page 1 of 9" ... then "Page 1 of 3").
BATES_RE = re.compile(r"\b(0{4,6}\d{1,4})\b")

# The Interchange prepends a machine-readable cover sheet to filings. It OCRs to
# barcode noise and carries no citable content.
COVER_SHEET_MARKERS = ("Control Number:", "Item Number:", "Addendum StartPage")


@dataclass
class PageSpan:
    """Where one page's text lives inside the full document text."""

    page_number: int  # 1-indexed position within this PDF
    char_start: int
    char_end: int
    bates: str | None = None  # citation anchor, when stamped
    is_cover_sheet: bool = False


@dataclass
class ExtractedDocument:
    text: str
    pages: list[PageSpan]
    page_count: int
    chars_per_page: float
    has_text_layer: bool
    status: ExtractionStatus
    error: str | None = None

    page_offset: int = 0  # added to in-PDF page numbers to get true filing pages

    def page_for_offset(self, offset: int) -> int:
        """In-PDF page number (1-indexed) for a character offset."""
        for span in self.pages:
            if span.char_start <= offset < span.char_end:
                return span.page_number
        raise IndexError(f"Offset {offset} outside document of length {len(self.text)}")

    def bates_for_offset(self, offset: int) -> str | None:
        """Bates stamp for a character offset -- the citation anchor.

        Returns None for unstamped pages (cover sheets, occasional OCR misses).
        A citation with no Bates should fall back to the filing page number and
        be flagged, not silently emitted.
        """
        page = self.page_for_offset(offset)
        return self.pages[page - 1].bates

    def citable_pages(self) -> list[PageSpan]:
        """Pages worth indexing: excludes the Interchange barcode cover sheet."""
        return [p for p in self.pages if not p.is_cover_sheet]

    def filing_page_for_offset(self, offset: int) -> int:
        """True page number within the filing, accounting for split documents.

        Large filings are served as ~100-page PDFs. A citation must report the
        page a human would find in the filing, not the page within the fragment.
        This is the number that belongs in a citation.
        """
        return self.page_for_offset(offset) + self.page_offset

    def slice(self, char_start: int, char_end: int) -> str:
        return self.text[char_start:char_end]


def normalize_text(raw: str) -> str:
    """Normalize extraction artifacts without changing character counts.

    Applied at extraction time so that offsets recorded downstream refer to
    normalized text. NFKC folds ligatures (fi -> fi) which CAN change length,
    so it is applied here, once, before any offset is computed -- never after.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def extract_document(
    payload: bytes,
    scan_threshold: int = DEFAULT_SCAN_THRESHOLD_CHARS_PER_PAGE,
    page_offset: int = 0,
) -> ExtractedDocument:
    """Extract text from PDF bytes, tracking page boundaries as character offsets.

    page_offset shifts reported page numbers for documents that are one fragment
    of a larger split filing (see sources.parse_page_offset).
    """
    try:
        doc = fitz.open(stream=payload, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        return ExtractedDocument(
            text="",
            pages=[],
            page_count=0,
            chars_per_page=0.0,
            has_text_layer=False,
            status=ExtractionStatus.FAILED,
            error=str(exc),
            page_offset=page_offset,
        )

    parts: list[str] = []
    pages: list[PageSpan] = []
    cursor = 0

    try:
        for index, page in enumerate(doc, start=1):
            page_text = normalize_text(page.get_text())
            # Guarantee a separator so page boundaries never fuse two words.
            if not page_text.endswith("\n"):
                page_text += "\n"
            parts.append(page_text)
            bates_matches = BATES_RE.findall(page_text)
            pages.append(
                PageSpan(
                    page_number=index,
                    char_start=cursor,
                    char_end=cursor + len(page_text),
                    # The stamp sits in the page footer; take the last match so a
                    # docket or exhibit number earlier in the body cannot win.
                    bates=bates_matches[-1] if bates_matches else None,
                    is_cover_sheet=sum(m in page_text for m in COVER_SHEET_MARKERS) >= 2,
                )
            )
            cursor += len(page_text)
    except Exception as exc:  # noqa: BLE001
        return ExtractedDocument(
            text="",
            pages=[],
            page_count=doc.page_count,
            chars_per_page=0.0,
            has_text_layer=False,
            status=ExtractionStatus.FAILED,
            error=str(exc),
            page_offset=page_offset,
        )
    finally:
        page_count = doc.page_count
        doc.close()

    text = "".join(parts)
    chars_per_page = len(text) / page_count if page_count else 0.0
    has_text_layer = chars_per_page >= scan_threshold

    return ExtractedDocument(
        text=text,
        pages=pages,
        page_count=page_count,
        chars_per_page=chars_per_page,
        has_text_layer=has_text_layer,
        status=ExtractionStatus.OK if has_text_layer else ExtractionStatus.NO_TEXT_LAYER,
        page_offset=page_offset,
    )


def verify_offset_integrity(extracted: ExtractedDocument) -> None:
    """Assert that page spans tile the document text exactly, with no gaps.

    This is the test that protects every citation the system will ever emit.
    Called in tests and by the probe script; cheap enough to call in ingestion.
    """
    if not extracted.pages:
        if extracted.text:
            raise AssertionError("Document has text but no page spans")
        return

    expected_start = 0
    for span in extracted.pages:
        if span.char_start != expected_start:
            raise AssertionError(
                f"Page {span.page_number} starts at {span.char_start}, expected {expected_start}"
            )
        if span.char_end < span.char_start:
            raise AssertionError(f"Page {span.page_number} has inverted span")
        expected_start = span.char_end

    if expected_start != len(extracted.text):
        raise AssertionError(
            f"Page spans cover {expected_start} chars, document text is {len(extracted.text)}"
        )


__all__ = [
    "ExtractedDocument",
    "ExtractionStatus",
    "PageSpan",
    "extract_document",
    "normalize_text",
    "verify_offset_integrity",
    "DEFAULT_SCAN_THRESHOLD_CHARS_PER_PAGE",
]
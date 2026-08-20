"""Tests for layout-aware chunking.

The fixtures mirror item 773: a memo page of prose and a schedule page that
extracts one cell per line.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.chunk import (
    CHARS_PER_LINE_THRESHOLD,
    Chunk,
    PageKind,
    chunk_document,
    classify_page,
    find_header_end,
    group_rows,
    split_lines,
    summarize,
    verify_chunk_spans,
)
from puctqa.extract import ExtractedDocument, ExtractionStatus, PageSpan


HEADER = [
    "Attachment A",
    "SOAH DOCKET NO",
    "473-19-3864",
    "Commission Number Run Schedule II",
    "PUC DOCKET NO.",
    "49421",
    "O&M Expense",
    "COMPANY NAME",
    "CenterPoint Energy Houston Electric",
    "(amounts in thousands)",
    "TEST YEAR END",
    "31-Dec-18",
    "OPERATIONS AND MAINTENANCE EXPENSE",
]
ROWS = [
    "Transmission Ops Supr & Engr", "560", "$", "13,074", "$", "222",
    "Load Dispatch - Reliability", "561", "$", "5,073", "$", "119",
    "Misc. Transmission Expenses", "566", "$", "3,548", "$", "146",
]
TABLE_PAGE = "\n".join(HEADER + ROWS)

PROSE_PAGE = (
    "Please find attached to this memo the schedules for CenterPoint Houston "
    "Electric, LLC (CEHE)\nbased on the number-running instructions contained in "
    "your memos dated November 15 and 26,\n2019. Also attached are two Staff memos "
    "from Mark Filarowicz and Brian Murphy describing the\nadjustments that Staff "
    "made to carry out the instructions.\n"
)


def build(prose: str = PROSE_PAGE, table: str = TABLE_PAGE) -> ExtractedDocument:
    text = prose + table
    return ExtractedDocument(
        text=text,
        pages=[
            PageSpan(1, 0, len(prose), page_label="1 of 2"),
            PageSpan(2, len(prose), len(text)),
        ],
        page_count=2,
        chars_per_page=len(text) / 2,
        has_text_layer=True,
        status=ExtractionStatus.OK,
    )


def lines_of(text: str) -> list:
    return split_lines(text, 0, len(text))


# --- Page classification ---


def test_prose_and_table_pages_are_distinguished():
    assert classify_page(lines_of(PROSE_PAGE)) is PageKind.PROSE
    assert classify_page(lines_of(TABLE_PAGE)) is PageKind.TABLE


def test_a_table_with_wordy_labels_is_still_a_table():
    """Item 773 p21 is 39% numeric at 17.3 chars/line -- numeric density would
    call it prose, line length calls it a table, and it is a table."""
    wide = "\n".join(
        ["III-A-1 SUMMARY OF WHOLESALE TCOS", "ATTACHMENT B.2"]
        + ["Wholesale Transmission Base Revenue", "1,234", "5,678", "9,012"] * 4
    )
    mean = sum(len(l.text) for l in lines_of(wide) if l.text.strip()) / len(
        [l for l in lines_of(wide) if l.text.strip()]
    )
    assert mean < CHARS_PER_LINE_THRESHOLD
    assert classify_page(lines_of(wide)) is PageKind.TABLE


# --- Header detection ---


def test_header_ends_at_the_first_data_row():
    lines = lines_of(TABLE_PAGE)
    assert lines[find_header_end(lines)].text == "Transmission Ops Supr & Engr"


def test_dates_in_the_header_are_not_mistaken_for_cells():
    """'31-Dec-18' reads as numeric. A window-count rule cut the header there,
    losing '(amounts in thousands)' from every chunk on the page."""
    lines = lines_of(TABLE_PAGE)
    header = "\n".join(l.text for l in lines[: find_header_end(lines)])
    assert "31-Dec-18" in header
    assert "(amounts in thousands)" in header


# --- Rows ---


def test_a_row_owns_the_numeric_lines_that_follow_it():
    rows = group_rows(lines_of("\n".join(ROWS)))
    assert len(rows) == 3
    assert [l.text for l in rows[0]] == ["Transmission Ops Supr & Engr", "560", "$", "13,074", "$", "222"]


def test_rows_are_never_split_across_chunks(monkeypatch):
    """A chunk holding half a row binds values to no label -- the misattribution
    a span-verifying guard cannot catch."""
    import puctqa.chunk as chunk_module

    monkeypatch.setattr(chunk_module, "TARGET_CHARS", 40)
    chunks = chunk_document(build(), "doc.pdf")

    table = [c for c in chunks if c.kind is PageKind.TABLE]
    assert len(table) > 1
    for c in table:
        assert "Transmission Ops Supr & Engr" not in c.body or "222" in c.body


# --- Context ---


def test_every_table_chunk_carries_its_units(monkeypatch):
    """A verified 4,231 without '(amounts in thousands)' is wrong by a factor of
    a thousand, and the digits match exactly, so the guard cannot catch it."""
    import puctqa.chunk as chunk_module

    monkeypatch.setattr(chunk_module, "TARGET_CHARS", 40)
    for c in chunk_document(build(), "doc.pdf"):
        if c.kind is PageKind.TABLE:
            assert "(amounts in thousands)" in c.context
            assert "Schedule II" in c.context


def test_prose_chunks_carry_no_context():
    prose = [c for c in chunk_document(build(), "doc.pdf") if c.kind is PageKind.PROSE]
    assert prose and all(c.context == "" for c in prose)


# --- Spans ---


def test_chunk_text_is_not_a_contiguous_slice_but_every_part_resolves():
    doc = build()
    table = [c for c in chunk_document(doc, "doc.pdf") if c.kind is PageKind.TABLE][0]

    assert table.text != doc.text[table.char_start : table.char_end]
    for start, end in table.spans():
        assert doc.text[start:end]
    assert len(table.spans()) == 2  # header span and body span


def test_span_verification_passes_on_real_chunks():
    doc = build()
    verify_chunk_spans(doc, chunk_document(doc, "doc.pdf"))


def test_span_verification_catches_a_wrong_offset():
    """A chunk that resolves to the wrong span produces a citation that looks
    checkable and is not -- worse than one that fails outright."""
    doc = build()
    chunks = chunk_document(doc, "doc.pdf")
    chunks[0].char_start += 25

    with pytest.raises(AssertionError, match="does not match its recorded span"):
        verify_chunk_spans(doc, chunks)


# --- Document level ---


def test_chunks_never_cross_a_page_boundary():
    for c in chunk_document(build(), "doc.pdf"):
        assert c.page_start == c.page_end


def test_cover_sheets_are_not_chunked():
    """The Interchange barcode page is not part of the record."""
    text = "49421\n795\nAddendum StartPage: 0\n" + PROSE_PAGE
    cover_len = len("49421\n795\nAddendum StartPage: 0\n")
    doc = ExtractedDocument(
        text=text,
        pages=[
            PageSpan(1, 0, cover_len, is_cover_sheet=True),
            PageSpan(2, cover_len, len(text)),
        ],
        page_count=2,
        chars_per_page=len(text) / 2,
        has_text_layer=True,
        status=ExtractionStatus.OK,
    )
    chunks = chunk_document(doc, "doc.pdf")
    assert all(c.page_start == 2 for c in chunks)


def test_anchor_scheme_travels_with_the_chunk():
    """A citation that cannot say whether it names a record page or a file page
    has no business claiming to be verifiable."""
    chunks = chunk_document(build(), "doc.pdf")
    by_page = {c.page_start: c for c in chunks}

    assert by_page[1].anchor_scheme == "page_label"
    assert by_page[2].anchor_scheme == "pdf_page"
    assert summarize(chunks)["weakly_anchored"] >= 1
"""Split extracted documents into citable chunks.

Layout-dependent, not a single recursive splitter: prose splits on a character
budget, table pages on row boundaries. A table page serializes one cell per
line, so a character splitter cuts rows in half and produces chunks of bare
numerals -- retrievable by nothing and citable to nothing.

Every table chunk carries its page header, including "(amounts in thousands)".
Without it a verified 4,231 is wrong by a factor of a thousand and the guard
cannot catch that, because the digits match exactly.

Which means chunk text is header + body and is NOT equal to
document_text[char_start:char_end]. Both are recorded as real spans; a claim
verifies against one or the other, never the concatenation.

Pages classify by MEDIAN line length -- see classify_page. DESIGN.md §17 has
the measurements behind the thresholds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .extract import ExtractedDocument, PageSpan

# Pure numerals, currency, accounting punctuation. Table cells; never start a row.
NUMERIC_LINE_RE = re.compile(r"^[\s$()\-\d,.%]*$")

# Pleading line numbers, the 1-25 down a testimony margin. Half the lines on
# item 788's pages, which drags the median to 3 and makes prose classify as a
# table. Excluded from classification and from chunk text -- an integer on every
# line satisfies exact numeric verification anywhere.
PLEADING_LINE_RE = re.compile(r"^\s*\d{1,2}\s*$")

# Against MEDIAN line length; calibrate per corpus. Item 773's schedules sit at
# 6.8-12.7 and its memo pages at 25.8-72.5.
CHARS_PER_LINE_THRESHOLD = 15.0

# A data row is a label plus this many CONSECUTIVE numeric lines. Counting
# numerics in a window cut item 773's header early, because "31-Dec-18" in the
# title block reads as numeric.
ROW_NUMERIC_RUN = 3

MAX_HEADER_LINES = 30

# Beyond this the "header" is the page: prepending it to every chunk buries the
# rows. Chunked as prose instead.
MAX_HEADER_CHARS = 700

TARGET_CHARS = 1200
MAX_CHARS = 2000


# A DOTTED section number alone on a line: "4.1.1", "3.2". Bare integers are
# excluded -- a schedule's account numbers look identical, and counting them
# flagged item 773's O&M schedule as navigation.
SECTION_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s*$")

# Item 795's table of contents runs seven sheets of "3.1 / APPLICABILITY / 22",
# which the median classifier reads as a table. A count rather than a ratio: a
# schedule page has zero dotted section numbers in its body, a contents page has
# dozens.
MIN_SECTION_NUMBERS = 5


class PageKind(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    NAVIGATION = "navigation"


@dataclass
class Chunk:
    # One citable unit.

    document_id: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    kind: PageKind
    context: str = ""
    context_char_start: int | None = None
    context_char_end: int | None = None
    anchor_scheme: str = "pdf_page"
    anchor_value: str = ""

    @property
    def body(self) -> str:
        # The part of this chunk that is a contiguous document slice
        return self.text[len(self.context):] if self.context else self.text

    def spans(self) -> list[tuple[int, int]]:
        # Every document range this chunk's text came from.
        
        ranges = [(self.char_start, self.char_end)]
        if self.context_char_start is not None:
            ranges.insert(0, (self.context_char_start, self.context_char_end))
        return ranges


@dataclass
class Line:
    text: str
    start: int
    end: int

    @property
    def is_numeric(self) -> bool:
        return bool(self.text.strip()) and bool(NUMERIC_LINE_RE.match(self.text))

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()

    @property
    def is_pleading_number(self) -> bool:
        """A bare 1-25 from a testimony margin. Page furniture, not content."""
        return bool(PLEADING_LINE_RE.match(self.text))


def split_lines(text: str, start: int, end: int) -> list[Line]:
    # Lines of a region, each carrying its absolute offsets in the document
    lines: list[Line] = []
    cursor = start
    for raw in text[start:end].split("\n"):
        lines.append(Line(raw, cursor, cursor + len(raw)))
        cursor += len(raw) + 1
    return lines


def content_lines(lines: list[Line], drop_pleading: bool = True) -> list[Line]:
    # Lines carrying content: neither blank nor margin numbering.
    if drop_pleading:
        return [l for l in lines if not l.is_blank and not l.is_pleading_number]
    return [l for l in lines if not l.is_blank]


def is_navigation(lines: list[Line]) -> bool:
    # A table of contents or index, which is neither prose nor data.
    
    content = [l for l in lines if not l.is_blank]
    if len(content) < 8:
        return False
    return sum(1 for l in content if SECTION_NUMBER_RE.match(l.text)) >= MIN_SECTION_NUMBERS


def classify_page(lines: list[Line]) -> PageKind:
    # Prose or table, by MEDIAN characters per non-blank line.
    content = content_lines(lines)
    if not content:
        return PageKind.PROSE
    if is_navigation(lines):
        return PageKind.NAVIGATION
    lengths = sorted(len(l.text) for l in content)
    median = lengths[len(lengths) // 2]
    return PageKind.PROSE if median >= CHARS_PER_LINE_THRESHOLD else PageKind.TABLE


def find_header_end(lines: list[Line]) -> int:
    # Index where the title block ends and the first data row begins.

    content = [i for i, l in enumerate(lines) if not l.is_blank]
    for position, index in enumerate(content):
        if lines[index].is_numeric:
            continue
        following = content[position + 1 : position + 1 + ROW_NUMERIC_RUN]
        if len(following) < ROW_NUMERIC_RUN:
            break
        if all(lines[i].is_numeric for i in following):
            return index
    return min(len(lines), MAX_HEADER_LINES)


def group_rows(lines: list[Line]) -> list[list[Line]]:
    # Group a table region into rows.
    rows: list[list[Line]] = []
    for line in lines:
        if line.is_blank:
            continue
        if line.is_numeric and rows:
            rows[-1].append(line)
        else:
            rows.append([line])
    return rows


def _render(lines: list[Line], drop_pleading: bool = True) -> str:
    # Chunk text, with margin numbering dropped on prose pages.
    return "\n".join(l.text.strip() for l in content_lines(lines, drop_pleading))


def chunk_table_page(
    lines: list[Line], span: PageSpan, document_id: str, start_ordinal: int
) -> list[Chunk]:
    header_end = find_header_end(lines)
    header_lines = content_lines(lines[:header_end], drop_pleading=False)
    header = _render(header_lines, drop_pleading=False)
    if len(header) > MAX_HEADER_CHARS:
        # No title block was found, so this page is not a table after all.
        return chunk_prose_page(lines, span, document_id, start_ordinal)
    header_range = (
        (header_lines[0].start, header_lines[-1].end) if header_lines else (None, None)
    )

    rows = group_rows(content_lines(lines[header_end:], drop_pleading=False))
    chunks: list[Chunk] = []
    batch: list[list[Line]] = []
    size = 0

    def flush() -> None:
        nonlocal batch, size
        if not batch:
            return
        flat = [line for row in batch for line in row]
        chunks.append(
            Chunk(
                document_id=document_id,
                ordinal=start_ordinal + len(chunks),
                text=(
                    header + "\n\n" + _render(flat, drop_pleading=False)
                    if header
                    else _render(flat, drop_pleading=False)
                ),
                char_start=flat[0].start,
                char_end=flat[-1].end,
                page_start=span.page_number,
                page_end=span.page_number,
                kind=PageKind.TABLE,
                context=header + "\n\n" if header else "",
                context_char_start=header_range[0],
                context_char_end=header_range[1],
                anchor_scheme=span.anchor_scheme.value,
                anchor_value=span.citation,
            )
        )
        batch, size = [], 0

    for row in rows:
        row_chars = sum(len(l.text) for l in row)
        # A row is never split. A chunk holding half a row binds values to no
        # label, which is the failure this whole module exists to avoid.
        if batch and size + row_chars > TARGET_CHARS:
            flush()
        batch.append(row)
        size += row_chars
    flush()
    return chunks


def chunk_prose_page(
    lines: list[Line], span: PageSpan, document_id: str, start_ordinal: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    batch: list[Line] = []
    size = 0

    def flush() -> None:
        nonlocal batch, size
        content = [l for l in batch if not l.is_blank]
        if not content:
            batch, size = [], 0
            return
        chunks.append(
            Chunk(
                document_id=document_id,
                ordinal=start_ordinal + len(chunks),
                text=_render(content),
                char_start=content[0].start,
                char_end=content[-1].end,
                page_start=span.page_number,
                page_end=span.page_number,
                kind=PageKind.PROSE,
                anchor_scheme=span.anchor_scheme.value,
                anchor_value=span.citation,
            )
        )
        batch, size = [], 0

    for line in lines:
        if size + len(line.text) > MAX_CHARS and batch:
            flush()
        batch.append(line)
        size += len(line.text)
        if size >= TARGET_CHARS and line.is_blank:
            flush()
    flush()
    return chunks


def chunk_document(extracted: ExtractedDocument, document_id: str) -> list[Chunk]:
    # Chunk one document, page by page.
    
    chunks: list[Chunk] = []
    for span in extracted.citable_pages():
        lines = split_lines(extracted.text, span.char_start, span.char_end)
        kind = classify_page(lines)
        if kind is PageKind.NAVIGATION:
            continue
        if kind is PageKind.TABLE:
            chunks.extend(chunk_table_page(lines, span, document_id, len(chunks)))
        else:
            chunks.extend(chunk_prose_page(lines, span, document_id, len(chunks)))
    return chunks


def verify_chunk_spans(extracted: ExtractedDocument, chunks: list[Chunk]) -> None:
    # Assert every chunk's recorded spans really are slices of the document.
    for chunk in chunks:
        for start, end in chunk.spans():
            if not (0 <= start < end <= len(extracted.text)):
                raise AssertionError(
                    f"chunk {chunk.ordinal} span ({start}, {end}) outside document"
                )
        body = extracted.text[chunk.char_start : chunk.char_end]

        rendered = _render(
            split_lines(body, 0, len(body)),
            drop_pleading=chunk.kind is not PageKind.TABLE,
        )
        if rendered != chunk.body.strip():
            raise AssertionError(
                f"chunk {chunk.ordinal} body does not match its recorded span"
            )


def summarize(chunks: list[Chunk]) -> dict:
    table = [c for c in chunks if c.kind is PageKind.TABLE]
    prose = [c for c in chunks if c.kind is PageKind.PROSE]
    sizes = sorted(len(c.text) for c in chunks) or [0]
    weak = [c for c in chunks if c.anchor_scheme == "pdf_page"]
    return {
        "chunks": len(chunks),
        "table": len(table),
        "prose": len(prose),
        "median_chars": sizes[len(sizes) // 2],
        "max_chars": sizes[-1],
        "weakly_anchored": len(weak),
    }


__all__ = [
    "Chunk",
    "PageKind",
    "chunk_document",
    "classify_page",
    "group_rows",
    "summarize",
    "verify_chunk_spans",
]
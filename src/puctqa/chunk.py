"""Split extracted documents into citable chunks.

Chunking is layout-dependent here, not a single recursive splitter, because the
corpus contains two kinds of page and a splitter that suits one destroys the
other.

Measured on item 773 (Commission number run, 26 pages), chars per line:

    p2   28.2   memo prose
    p4   72.5   prose
    p10   6.8   O&M schedule -- one cell per line
    p21  17.3   wholesale TCOS schedule, wordier row labels
    p26   9.8   rate design summary

Sorted, those cluster below 13 and then jump to 17.3, 25.8, 28.2, 37.6 -- a gap
at roughly 15. Numeric density does NOT separate as cleanly: page 3 is 30%
numeric at 37.6 chars/line (prose with figures) while page 21 is 39% numeric at
17.3 (a table with long headers). So pages are classified by line length, and
CHARS_PER_LINE_THRESHOLD is a calibration knob with the same status as the
chars-per-page threshold in probe_extraction.py: look at the histogram for your
corpus and put it in the gap.

TABLE PAGES SERIALIZE ONE CELL PER LINE

    Transmission Ops Supr & Engr
    560
    $
    13,074
    $
    222

A row occupies a dozen lines, currency symbols included. Splitting on character
count cuts rows in half and produces chunks of bare numerals -- retrievable by
nothing, citable to nothing, and separating a value from its label is exactly
the misattribution a span-verifying guard cannot catch. So table pages are split
on ROW BOUNDARIES: a non-numeric line starts a row and owns the numeric lines
that follow it, which is the same reading-order rule the extraction-fidelity
measurement settled on.

EVERY TABLE CHUNK CARRIES ITS PAGE HEADER

A table page opens with a title block -- schedule number, company, test year,
and crucially "(amounts in thousands)". A chunk holding "Operations &
Maintenance / 4,231" without it is not merely vague: a verified 4,231 is wrong
by a factor of a thousand, and the guard cannot catch that, because the digits
match exactly. The header is prepended to every chunk cut from that page.

WHICH MEANS CHUNK TEXT IS NOT A CONTIGUOUS DOCUMENT SLICE

`text` is header + body, so it does not equal document[char_start:char_end].
Both pieces are real spans of the document and both are recorded -- body as
char_start/char_end, header as context_char_start/context_char_end -- so every
character a chunk shows can still be located in the source. The guard must
verify a claim against one span or the other, never against the concatenation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .extract import ExtractedDocument, PageSpan

# Lines that are pure numerals, currency, or accounting punctuation. These are
# table cells, not content, and they never start a row.
NUMERIC_LINE_RE = re.compile(r"^[\s$()\-\d,.%]*$")

# Pleading line numbers -- the 1-25 running down a testimony margin -- extract as
# standalone one- and two-character lines. On item 788 they are roughly half of
# every page, which drags the median line length to 3 and makes prose testimony
# classify as a table. find_header_end then finds no data row, falls back to
# MAX_HEADER_LINES, and produces a chunk of 2,854 characters of "header" around
# 100 characters of body.
#
# They are page furniture, not content, and are excluded from both the
# classification statistic and the chunk text. The same filter exists in
# ocr_accuracy.py for the same reason: an integer appearing on every line of
# every page satisfies exact numeric verification anywhere.
PLEADING_LINE_RE = re.compile(r"^\s*\d{1,2}\s*$")

# Calibrate per corpus, against MEDIAN line length -- see classify_page. Item
# 773's schedules sit at 6.8-12.7 and its memo pages at 25.8-72.5; item 795's
# tariff runs to a median of 55.9 overall with rate sheets far below it.
CHARS_PER_LINE_THRESHOLD = 15.0

# A data row is a label followed by at least this many numeric lines. A window
# count was tried first and cut the header a line early on item 773, because
# "31-Dec-18" in the title block reads as numeric and tripped the burst before
# the first real row. Requiring a CONSECUTIVE run after a label ignores stray
# numerals among header text, which is what dates and docket numbers are.
ROW_NUMERIC_RUN = 3

# Headers longer than this are not title blocks -- the page was misclassified,
# or the whole page is header. Cap rather than prepend hundreds of lines to
# every chunk on the page.
MAX_HEADER_LINES = 30

# A title block names the schedule, company, and units in a handful of lines.
# Beyond this the "header" is the page, and prepending it to every chunk buries
# the rows it was meant to caption. The page is chunked as prose instead, which
# is what it was.
MAX_HEADER_CHARS = 700

TARGET_CHARS = 1200
MAX_CHARS = 2000


# A DOTTED section number alone on a line: "4.1.1", "3.2". Bare integers are
# deliberately excluded -- a rate schedule's account numbers and a table of
# contents' page numbers look identical, and counting them flagged item 773's
# O&M schedule (560, 561, 566, and bare currency symbols) as navigation.
SECTION_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s*$")

# Item 795's table of contents runs seven sheets of "3.1 / APPLICABILITY / 22",
# which the median classifier reads as a table: short lines, numeric cells, a
# header block. It is neither prose nor a rate schedule -- it answers no
# question, and its figures are page numbers. A count rather than a ratio,
# because a schedule page has ZERO dotted section numbers in its body while a
# contents page has dozens; the separation is not close enough to need tuning.
MIN_SECTION_NUMBERS = 5


class PageKind(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    NAVIGATION = "navigation"


@dataclass
class Chunk:
    """One citable unit.

    `text` is what gets embedded and shown. `char_start`/`char_end` locate the
    body in the document; `context_*` locate the header. text is the two joined,
    so it is deliberately NOT equal to document[char_start:char_end].
    """

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
        """The part of this chunk that is a contiguous document slice."""
        return self.text[len(self.context):] if self.context else self.text

    def spans(self) -> list[tuple[int, int]]:
        """Every document range this chunk's text came from.

        A claim must verify against one of these. Verifying against the
        concatenation would let a claim draw half its support from the header
        and half from a row that does not sit under it.
        """
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
    """Lines of a region, each carrying its absolute offsets in the document."""
    lines: list[Line] = []
    cursor = start
    for raw in text[start:end].split("\n"):
        lines.append(Line(raw, cursor, cursor + len(raw)))
        cursor += len(raw) + 1
    return lines


def content_lines(lines: list[Line]) -> list[Line]:
    """Lines carrying content: neither blank nor margin numbering."""
    return [l for l in lines if not l.is_blank and not l.is_pleading_number]


def is_navigation(lines: list[Line]) -> bool:
    """A table of contents or index, which is neither prose nor data.

    Indexing it means retrieval surfaces "3.4 CHARGES ASSOCIATED WITH DELIVERY
    SERVICE 23" for a query about delivery charges -- a chunk that names the
    section a reader wants and contains none of it.
    """
    content = [l for l in lines if not l.is_blank]
    if len(content) < 8:
        return False
    return sum(1 for l in content if SECTION_NUMBER_RE.match(l.text)) >= MIN_SECTION_NUMBERS


def classify_page(lines: list[Line]) -> PageKind:
    """Prose or table, by MEDIAN characters per non-blank line.

    The mean was tried first and misclassified item 795's rate schedules. Every
    tariff sheet opens with a fixed header block -- chapter, sheet number,
    company, service area, applicability -- and those long lines pull the mean
    of a page whose body is short numeric cells above the threshold. Page 43 of
    one part is a class allocation table (CLASS / PBRAF / Residential /
    40.4859 %) and measured 15.4 chars per line; page 12 is a fee schedule at
    16.3. Both chunked as prose, which loses the header block those figures
    depend on -- a chunk reading "Residential 40.4859 %" with no sheet number
    and no effective date is a rate with no idea which tariff version it is.

    A median ignores a fixed number of outlying lines by construction, which is
    exactly what a page header is.
    """
    content = content_lines(lines)
    if not content:
        return PageKind.PROSE
    if is_navigation(lines):
        return PageKind.NAVIGATION
    lengths = sorted(len(l.text) for l in content)
    median = lengths[len(lengths) // 2]
    return PageKind.PROSE if median >= CHARS_PER_LINE_THRESHOLD else PageKind.TABLE


def find_header_end(lines: list[Line]) -> int:
    """Index where the title block ends and the first data row begins.

    The first data row is a non-numeric line followed by ROW_NUMERIC_RUN
    consecutive numeric lines. The label belongs to the data, not the header, so
    the header ends AT that line.

    Header text is full of stray numerals -- docket numbers, test-year dates --
    and a rule that counts numerics in a window mistakes them for cells.
    """
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
    """Group a table region into rows.

    A non-numeric line starts a row and owns every numeric line that follows.
    Same reading-order rule the fidelity measurement uses: whatever the column
    count, the label comes first.
    """
    rows: list[list[Line]] = []
    for line in lines:
        if line.is_blank:
            continue
        if line.is_numeric and rows:
            rows[-1].append(line)
        else:
            rows.append([line])
    return rows


def _render(lines: list[Line]) -> str:
    """Chunk text, with margin numbering dropped.

    A chunk reading "1 / Q. / 2 / A. / 3" is mostly furniture, and those digits
    would satisfy a numeric claim about anything.
    """
    return "\n".join(l.text.strip() for l in content_lines(lines))


def chunk_table_page(
    lines: list[Line], span: PageSpan, document_id: str, start_ordinal: int
) -> list[Chunk]:
    header_end = find_header_end(lines)
    header_lines = content_lines(lines[:header_end])
    header = _render(header_lines)
    if len(header) > MAX_HEADER_CHARS:
        # No title block was found, so this page is not a table after all.
        return chunk_prose_page(lines, span, document_id, start_ordinal)
    header_range = (
        (header_lines[0].start, header_lines[-1].end) if header_lines else (None, None)
    )

    rows = group_rows(lines[header_end:])
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
                text=(header + "\n\n" + _render(flat)) if header else _render(flat),
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
    """Chunk one document, page by page.

    Chunks never cross a page boundary, and never cross a document boundary --
    a filing served in four parts is four documents, and support spanning a part
    boundary has no expressible citation, because one citation is one document
    and one span.

    Cover sheets are skipped: the Interchange barcode page is not part of the
    record and nothing should cite it.
    """
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
    """Assert every chunk's recorded spans really are slices of the document.

    The same role verify_offset_integrity() plays for pages. A chunk whose
    offsets do not resolve cannot produce a citation a human can check, and a
    chunk that silently resolves to the WRONG span produces a citation that
    looks checkable and is not -- the worse failure.
    """
    for chunk in chunks:
        for start, end in chunk.spans():
            if not (0 <= start < end <= len(extracted.text)):
                raise AssertionError(
                    f"chunk {chunk.ordinal} span ({start}, {end}) outside document"
                )
        body = extracted.text[chunk.char_start : chunk.char_end]
        rendered = _render(split_lines(body, 0, len(body)))
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
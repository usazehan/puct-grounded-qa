"""Split chunks into addressable evidence units.

A claim must rest on an exact span of the source. The obvious way to get one is
to ask a model to quote the passage, but a model that paraphrases fails span
verification even when its claim is correct -- so verbatim quotation becomes a
capability requirement, and the guard's fuzzy span check ends up compensating
for model behaviour rather than for OCR damage.

Segmenting first removes the problem. Each unit gets an id, its text, and its
character offsets in the document. A model selects ids from a closed list; code
resolves them to text and offsets. Quotation is verbatim by construction, and
the model cannot cite a passage that was never given to it.

That shifts what the guard is for. verify_span stops being load-bearing -- the
span is exact by construction -- while the other two checks matter more:
verify_numbers catches a claim asserting a figure the selected evidence does not
contain, and verify_predicate catches a claim that selects a correct span and
says the wrong thing about it, which is the failure retrieval already produces
on its own.

UNIT BOUNDARIES FOLLOW PAGE LAYOUT, AS CHUNKING DOES

Prose splits on sentences. Tables split on rows, using the same reading-order
rule as chunking: a label owns the numeric lines that follow it. A rate schedule
row is one fact -- "Metal Halide (175w) / $9.24 / 12,900 / 210 / N/A / 70" --
and splitting it further would hand a model a bare figure with no label, which
is the misattribution the row-boundary rule exists to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunk import Line, PageKind, content_lines, split_lines

# Sentence boundary: terminator, closing quote or bracket, then whitespace and a
# capital or a digit. Abbreviations common in these filings are excluded below
# rather than by a general abbreviation list, because the failure mode here is
# splitting a citation in half.
SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9])")

# Splitting after these would cut a reference in two. "No." and "Sec." are
# everywhere in a docket; "U.S.C." and "Tex." appear in every legal citation.
# A single capital letter followed by a period is an initial or a rule
# reference -- "P.U.C. Subst. R. 25.243" splits at "R." otherwise, because the
# next token starts with a digit.
INITIAL_RE = re.compile(r"(?:^|\s)[A-Z]\.$")

ABBREVIATIONS = (
    "No.", "Nos.", "Sec.", "Secs.", "Art.", "Ch.", "Chs.", "Ex.", "Exh.",
    "Tex.", "U.S.", "U.S.C.", "P.U.C.", "Subst.", "Rule.", "Inc.", "LLC.",
    "Co.", "Corp.", "Dkt.", "Docket.", "Mr.", "Ms.", "Dr.", "Jr.", "Sr.",
    "approx.", "est.", "et al.", "v.", "vs.", "cf.",
)

# Below this a "sentence" is a fragment -- a heading, a stray label, an
# extraction artifact -- and a claim resting on it would cite almost nothing.
MIN_UNIT_CHARS = 25

# A unit longer than this is a paragraph the sentence splitter could not break,
# usually because extraction lost the terminators. Cited, it would be a wall of
# text standing in for a specific statement.
MAX_UNIT_CHARS = 600


@dataclass
class EvidenceUnit:
    """One addressable piece of a chunk.

    `unit_id` is what a model selects. Everything else is resolved in code, so
    the text a claim cites is the text that is actually in the document.
    """

    unit_id: str
    chunk_id: int
    text: str
    char_start: int
    char_end: int
    kind: str  # prose | table_row

    def resolves_in(self, document_text: str) -> bool:
        """Does this unit's recorded span still hold the text it claims?

        The same role verify_chunk_spans plays for chunks. A unit whose offsets
        drift produces a citation that looks checkable and is not.
        """
        if not (0 <= self.char_start < self.char_end <= len(document_text)):
            return False
        actual = " ".join(document_text[self.char_start : self.char_end].split())
        return " ".join(self.text.split()) in actual


def _ends_with_abbreviation(text: str) -> bool:
    stripped = text.rstrip()
    if INITIAL_RE.search(stripped):
        return True
    return any(stripped.endswith(abbrev) for abbrev in ABBREVIATIONS)


# Placeholder cells. A rate schedule marks an inapplicable column "N/A", and
# because that is not a numeral it reads as a new row label -- splitting
# "Metal Halide (175w) / $9.24 / 12,900 / 210 / N/A / 70" into three pieces, one
# of which is the bare word N/A and another the orphaned 70.
#
# The same flaw is in chunk.group_rows, which means table CHUNKS are being cut
# at N/A too. Fixed here because evidence units are what a claim cites; the
# chunking case is cosmetic by comparison, but it is the same bug.
PLACEHOLDER_CELLS = frozenset({"n/a", "n.a.", "na", "--", "-", "—", "n/a.", "tbd"})


def _is_cell(line: Line) -> bool:
    """A continuation of the current row rather than the start of a new one."""
    return line.is_numeric or line.text.strip().lower() in PLACEHOLDER_CELLS


def split_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of sentences within `text`.

    Offsets rather than strings, so a caller can map them back onto document
    positions. A split that would cut after an abbreviation is rejoined --
    "PUC Docket No. 49421" is one reference, not two sentences.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        candidate = text[start : match.start()]
        if _ends_with_abbreviation(candidate):
            continue
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _merge_short(units: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
    """Fold fragments into the neighbour they belong to.

    Extraction leaves headings and stray labels on their own. Cited alone they
    carry no statement, and dropping them would lose text a longer unit needs.
    """
    merged: list[tuple[int, int]] = []
    for start, end in units:
        if merged and len(text[start:end].strip()) < MIN_UNIT_CHARS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def units_from_prose(
    chunk_id: int, lines: list[Line], ordinal_start: int = 0
) -> list[EvidenceUnit]:
    """Sentence-level units, with offsets mapped back onto the document.

    The rendered text of a prose chunk drops blank lines and margin numbering,
    so a sentence's position in that rendering is not its position in the
    document. Offsets are taken from the Line objects the sentence spans, which
    carry real document positions.
    """
    content = content_lines(lines)
    if not content:
        return []

    # Rendered text, plus the document offset each rendered character came from.
    rendered_parts: list[str] = []
    offsets: list[int] = []
    for line in content:
        stripped = line.text.strip()
        if not stripped:
            continue
        lead = len(line.text) - len(line.text.lstrip())
        if rendered_parts:
            rendered_parts.append(" ")
            offsets.append(line.start + lead)
        rendered_parts.append(stripped)
        offsets.extend(range(line.start + lead, line.start + lead + len(stripped)))
    rendered = "".join(rendered_parts)
    if len(offsets) != len(rendered):
        # Should not happen; a mismatch means offsets would be silently wrong.
        raise AssertionError("rendered text and offset map disagree")

    spans = _merge_short(split_sentences(rendered), rendered)
    units: list[EvidenceUnit] = []
    for index, (start, end) in enumerate(spans):
        body = rendered[start:end].strip()
        if len(body) < MIN_UNIT_CHARS:
            continue
        # Trailing whitespace was stripped, so walk the end back to a real char.
        last = end - 1
        while last > start and rendered[last].isspace():
            last -= 1
        units.append(
            EvidenceUnit(
                unit_id=f"c{chunk_id}:e{ordinal_start + index:02d}",
                chunk_id=chunk_id,
                text=body,
                char_start=offsets[start],
                char_end=offsets[last] + 1,
                kind="prose",
            )
        )
    return units


def units_from_table(
    chunk_id: int, lines: list[Line], ordinal_start: int = 0
) -> list[EvidenceUnit]:
    """One unit per table row.

    A row is one fact. Splitting further hands a model a bare figure with no
    label -- the misattribution the row-boundary rule exists to prevent -- and
    merging rows would let a claim cite a schedule rather than a rate.
    """
    # drop_pleading=False: on a table page a bare one- or two-digit line is a
    # cell value, not margin numbering. The monthly kWh column of item 795-A's
    # street lighting schedule reads 70, 98, 159, 367.
    rows: list[list[Line]] = []
    for line in content_lines(lines, drop_pleading=False):
        if _is_cell(line) and rows:
            rows[-1].append(line)
        else:
            rows.append([line])

    units: list[EvidenceUnit] = []
    for index, row in enumerate(rows):
        if not row:
            continue
        text = " ".join(line.text.strip() for line in row if line.text.strip())
        if not text:
            continue
        units.append(
            EvidenceUnit(
                unit_id=f"c{chunk_id}:e{ordinal_start + index:02d}",
                chunk_id=chunk_id,
                text=text,
                char_start=row[0].start,
                char_end=row[-1].end,
                kind="table_row",
            )
        )
    return units


def segment_chunk(
    chunk_id: int,
    document_text: str,
    char_start: int,
    char_end: int,
    kind: str,
) -> list[EvidenceUnit]:
    """Split one chunk's body into evidence units.

    Only the body. A chunk's header is context prepended to every unit on the
    page, not a statement any claim should rest on: "(amounts in thousands)"
    tells a reader how to read a figure, and a claim citing it alone asserts
    nothing.
    """
    lines = split_lines(document_text, char_start, char_end)
    if kind == PageKind.TABLE.value or kind == PageKind.TABLE:
        return units_from_table(chunk_id, lines)
    return units_from_prose(chunk_id, lines)


def verify_units(units: list[EvidenceUnit], document_text: str) -> None:
    """Assert every unit's offsets still hold its text.

    Raises rather than returns, for the same reason verify_chunk_spans does: a
    unit with drifting offsets produces a citation that resolves to the wrong
    passage, which is worse than one that fails outright.
    """
    for unit in units:
        if not unit.resolves_in(document_text):
            raise AssertionError(
                f"evidence unit {unit.unit_id} does not resolve to its recorded span"
            )


def summarize(units: list[EvidenceUnit]) -> dict:
    sizes = sorted(len(u.text) for u in units) or [0]
    return {
        "units": len(units),
        "prose": sum(1 for u in units if u.kind == "prose"),
        "table_row": sum(1 for u in units if u.kind == "table_row"),
        "median_chars": sizes[len(sizes) // 2],
        "max_chars": sizes[-1],
        "over_max": sum(1 for u in units if len(u.text) > MAX_UNIT_CHARS),
    }


__all__ = [
    "EvidenceUnit",
    "segment_chunk",
    "split_sentences",
    "summarize",
    "units_from_prose",
    "units_from_table",
    "verify_units",
]
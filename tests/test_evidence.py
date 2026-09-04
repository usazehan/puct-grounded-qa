"""Tests for evidence segmentation.

Units are what a model selects from, so their boundaries decide what a claim can
cite. A unit that is half a table row is a figure with no label; one that is a
whole page cites nothing in particular.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.chunk import split_lines  # noqa: E402
from puctqa.evidence import (  # noqa: E402
    MIN_UNIT_CHARS,
    EvidenceUnit,
    segment_chunk,
    split_sentences,
    summarize,
    units_from_prose,
    units_from_table,
    verify_units,
)


# Item 792, Bates 000000010.
ORDER_TEXT = (
    "The signatories agreed that CenterPoint Houston's weighted average cost of "
    "capital will be 6.51%, based on a cost of debt of 4.38%, a return on equity "
    "of 9.4%, and a capital structure of 57.5% long-term debt. It is appropriate "
    "for CenterPoint Houston to have an overall rate of return of 6.51%."
)

# Item 795-A, Sheet 6.6, one cell per line.
SCHEDULE_TEXT = "\n".join([
    "Metal Halide (175w) (no new installations)", "$9.24", "12,900", "210", "N/A", "70",
    "Metal Halide (250w) (no new installations)", "$17.08", "19,475", "294", "N/A", "98",
])


def lines_of(text: str) -> list:
    return split_lines(text, 0, len(text))


# --- Sentence splitting ---


def test_sentences_split_on_terminators():
    spans = split_sentences(ORDER_TEXT)
    assert len(spans) == 2
    assert ORDER_TEXT[spans[0][0] : spans[0][1]].startswith("The signatories agreed")
    assert ORDER_TEXT[spans[1][0] : spans[1][1]].startswith("It is appropriate")


def test_an_abbreviation_does_not_end_a_sentence():
    """'PUC Docket No. 49421' is one reference. Splitting it hands a model half
    a citation."""
    text = "The Commission considered PUC Docket No. 49421 at its open meeting."
    assert len(split_sentences(text)) == 1


def test_a_legal_citation_stays_whole():
    text = "This is in accord with P.U.C. Subst. R. 25.243 and applies to all filings."
    assert len(split_sentences(text)) == 1


# --- Prose units ---


def test_prose_units_resolve_to_their_offsets():
    lines = lines_of(ORDER_TEXT)
    units = units_from_prose(17, lines)

    assert len(units) == 2
    verify_units(units, ORDER_TEXT)


def test_prose_offsets_survive_margin_numbering():
    """Rendered text drops blank lines and pleading numbers, so a sentence's
    position in the rendering is not its position in the document."""
    document = "\n".join([
        "1", "The signatories agreed to a return on equity of 9.4% in this docket.",
        "2", "It is appropriate for CenterPoint Houston to have that rate of return.",
    ])
    units = units_from_prose(17, lines_of(document))

    assert units
    verify_units(units, document)
    assert all("\n1\n" not in u.text for u in units)


def test_a_fragment_is_merged_not_emitted():
    """A heading cited alone asserts nothing."""
    document = "SECTION 5\nThe signatories agreed to a return on equity of 9.4% here."
    units = units_from_prose(17, lines_of(document))

    assert len(units) == 1
    assert all(len(u.text) >= MIN_UNIT_CHARS for u in units)


# --- Table units ---


def test_a_table_row_is_one_unit():
    """Splitting further hands a model a bare figure with no label."""
    units = units_from_table(42, lines_of(SCHEDULE_TEXT))

    assert len(units) == 2
    assert "Metal Halide (175w)" in units[0].text
    assert "$9.24" in units[0].text
    assert "70" in units[0].text
    assert "17.08" not in units[0].text  # the next row is a different fact


def test_table_units_resolve_to_their_offsets():
    units = units_from_table(42, lines_of(SCHEDULE_TEXT))
    verify_units(units, SCHEDULE_TEXT)


def test_a_row_keeps_its_label_with_its_figures():
    units = units_from_table(42, lines_of(SCHEDULE_TEXT))
    for unit in units:
        assert "Metal Halide" in unit.text


# --- Ids and dispatch ---


def test_unit_ids_are_unique_and_addressable():
    units = units_from_table(42, lines_of(SCHEDULE_TEXT))
    ids = [u.unit_id for u in units]

    assert ids == ["c42:e00", "c42:e01"]
    assert len(set(ids)) == len(ids)


def test_segment_dispatches_on_page_kind():
    prose = segment_chunk(1, ORDER_TEXT, 0, len(ORDER_TEXT), "prose")
    table = segment_chunk(2, SCHEDULE_TEXT, 0, len(SCHEDULE_TEXT), "table")

    assert all(u.kind == "prose" for u in prose)
    assert all(u.kind == "table_row" for u in table)


# --- Verification ---


def test_a_drifting_offset_is_caught():
    """A unit that resolves to the wrong passage produces a citation that looks
    checkable and is not."""
    units = units_from_prose(17, lines_of(ORDER_TEXT))
    units[0].char_start += 40

    with pytest.raises(AssertionError, match="does not resolve"):
        verify_units(units, ORDER_TEXT)


def test_summary_reports_oversized_units():
    units = units_from_prose(17, lines_of(ORDER_TEXT))
    stats = summarize(units)

    assert stats["units"] == 2
    assert stats["prose"] == 2
    assert stats["over_max"] == 0
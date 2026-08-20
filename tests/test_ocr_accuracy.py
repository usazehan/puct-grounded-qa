"""Tests for the OCR accuracy measurement.

Per DESIGN §8 the numeric normalizer gets the most cases. The association check
gets the two that motivated it: a table that survived extraction as rows, and
one that did not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ocr_accuracy import (  # noqa: E402
    AssociationResult,
    NumericResult,
    check_association,
    compare_numerics,
    find_natives,
    numeric_counter,
    verdict,
    _pairs_from_rows,
)
from puctqa.sources import DocumentRef  # noqa: E402


# --------------------------------------------------------------------------
# Canonicalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1,234", {"1234": 1}),
        ("$1,234.50", {"1234.5": 1}),
        ("10.40%", {"10.4%": 1}),
        # The guard's normalizer treats these as signed; the measurement must agree.
        ("(1,234)", {"-1234": 1}),
        ("-45", {"-45": 1}),
        ("see line 5)", {"5": 1}),
        ("10.4", {"10.4": 1}),
    ],
)
def test_canonical_forms(text, expected):
    assert dict(numeric_counter(text)) == expected


def test_paren_and_bare_are_distinct_tokens():
    """(1,234) and 1,234 must not compare equal -- that is a sign flip."""
    counts = numeric_counter("(1,234) and 1,234")
    assert counts["-1234"] == 1
    assert counts["1234"] == 1


# --------------------------------------------------------------------------
# Occurrence vs set counting
# --------------------------------------------------------------------------


def test_set_recall_hides_a_dropped_occurrence():
    """Native has 4,231 three times, OCR twice. Set membership says it survived."""
    native = "4,231 line item\n4,231 subtotal\n4,231 total"
    ocr = "4,231 line item\n4,231 subtotal\n4,Z31 total"

    result = compare_numerics(native, ocr)
    assert result.type_accuracy == 100.0
    assert result.occurrence_accuracy == pytest.approx(66.667, abs=0.01)


def test_short_tokens_are_measured():
    """9.5 and 847 were excluded outright by the old min_len=4 filter."""
    result = compare_numerics("9.5 847 1.2", "9.5 847 1.2")
    assert result.short_expected == 3
    assert result.short_accuracy == 100.0


def test_sign_loss_is_counted_separately():
    """A dropped parenthesis reads as a positive and must not pass quietly."""
    result = compare_numerics("O&M expense  (45)", "O&M expense  45")
    assert result.occurrence_accuracy == 0.0
    assert result.sign_losses == 1


def test_known_limit_a_lone_small_negative_reads_as_enumeration():
    """The one case strip_furniture gets wrong, recorded rather than hidden.

    A negative alone on its own line is indistinguishable from a subsection
    marker by position, and item 795 shows markers are the common case. A cell
    serialized one-per-line would be misread. The guard must not lean on this.
    """
    assert numeric_counter("(45)") == {}


# --------------------------------------------------------------------------
# Row association
# --------------------------------------------------------------------------


ROWS = [
    ["Line", "Test Year", "Adjustment"],
    ["O&M expense", "4,231", "(45)"],
    ["Depreciation expense", "1,208", "9.5"],
    ["Federal income tax", "847", "(12)"],
]

ROW_MAJOR = "\n".join("  ".join(r) for r in ROWS)
COLUMN_MAJOR = "\n".join("  ".join(c) for c in zip(*ROWS))


def test_pairs_recovered_from_table_rows():
    pairs = _pairs_from_rows(ROWS)
    assert ("O&M expense", "4231") in pairs
    assert ("O&M expense", "-45") in pairs
    assert ("Depreciation expense", "9.5") in pairs


def test_row_major_extraction_keeps_association():
    result = check_association(_pairs_from_rows(ROWS), ROW_MAJOR)
    assert result.rate == 100.0


def test_column_major_extraction_breaks_association():
    """Digits all survive; every value binds to the wrong row. Token accuracy
    scores this clean, which is why it cannot license exact verification alone."""
    numeric = compare_numerics(ROW_MAJOR, COLUMN_MAJOR)
    assert numeric.occurrence_accuracy == 100.0

    result = check_association(_pairs_from_rows(ROWS), COLUMN_MAJOR, tolerance=0)
    assert result.rate < 100.0
    assert result.broken


def test_prose_ocr_errors_do_not_count_as_broken_association():
    """rn -> m in a label is a span-verification problem, not an association one."""
    degraded = ROW_MAJOR.replace("Federal", "Federai")
    result = check_association(_pairs_from_rows(ROWS), degraded)
    assert result.rate == 100.0


def test_unlocatable_pairs_are_excluded_not_failed():
    """A value absent from OCR is a numeric miss, already counted there."""
    result = check_association([("O&M expense", "99999")], ROW_MAJOR)
    assert result.checked == 1
    assert result.unlocatable == 1
    assert result.rate == 100.0


# --- Native bundle pairing ---


def _touch(root: Path, *names: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"")


def test_primary_prefers_text_format_over_workpapers(tmp_path):
    """Item 786 ships four .xlsx workpapers next to the born-digital PDF.

    Scoring a spreadsheet's cell dump against a testimony PDF yields a garbage
    word accuracy under a confident-looking verdict, so the pick is explicit
    rather than whatever rglob happens to yield.
    """
    _touch(
        tmp_path / "49421_786",
        "aaa workpaper.xlsx",
        "Colvin Direct Testimony.pdf",
        "bbb workpaper.xlsx",
    )
    ref = DocumentRef.from_filename("49421_786_1049723.pdf")
    match = find_natives(tmp_path, ref)

    assert match.primary.suffix == ".pdf"
    assert [w.name for w in match.structured] == ["aaa workpaper.xlsx", "bbb workpaper.xlsx"]


def test_pairing_is_deterministic_within_a_format(tmp_path):
    _touch(tmp_path / "49421_788", "zzz.docx", "aaa.docx")
    ref = DocumentRef.from_filename("49421_788_1050240.pdf")
    assert find_natives(tmp_path, ref).primary.name == "aaa.docx"


def test_workpapers_are_not_discarded(tmp_path):
    """A single-file pick would drop the only files carrying table structure."""
    _touch(tmp_path / "49421_773", "Number Run.xlsx", "Cover.docx")
    ref = DocumentRef.from_filename("49421_773_1049999.pdf")
    assert find_natives(tmp_path, ref).structured


def test_mispairing_outranks_every_other_verdict():
    """A confident verdict on the wrong pair of files is the failure to avoid."""
    clean = NumericResult(occurrences_expected=8000, occurrences_found=8000)
    assert verdict(
        clean, AssociationResult(), structured=False, word_accuracy=12.0,
        containment=8.0, native_words=7000
    ) == "likely_mispaired"
    assert verdict(
        clean, AssociationResult(), structured=False, word_accuracy=99.9,
        containment=99.0, native_words=7000
    ) != "likely_mispaired"


def test_a_contained_fragment_is_partial_not_mispaired():
    """795 served a 371-page tariff as four ~100-page parts. Each part is really
    in the native; it just cannot fill it, and 26% measured the split."""
    clean = NumericResult(occurrences_expected=8000, occurrences_found=8000)
    assert verdict(
        clean, AssociationResult(), structured=False, word_accuracy=26.0,
        containment=98.0, native_words=7000
    ) == "partial_pairing"


def test_sets_group_by_document_id_run(tmp_path):
    """Two refiled batches of the same filing, not eight loose documents."""
    from ocr_accuracy import group_served_sets

    names = [
        "49421_795_1057872.pdf", "49421_795_1057873.pdf",
        "49421_795_1057874.pdf", "49421_795_1057875.pdf",
        "49421_795_1119824.pdf", "49421_795_1119825.pdf",
        "49421_795_1119826.pdf", "49421_795_1119827.pdf",
        "49421_773_1043164.pdf",
    ]
    for n in names:
        (tmp_path / n).touch()

    sets = group_served_sets(sorted(tmp_path.glob("*.pdf")))
    sizes = sorted(len(s.parts) for s in sets)
    assert sizes == [1, 4, 4]
    assert all(len(s.parts) == 1 for s in sets if s.item_number == 773)


def test_pleading_line_numbers_do_not_break_row_association():
    """Item 786 interleaves 1-25 margin numbers between every content line."""
    interleaved = "\n".join(
        part
        for i, row in enumerate(ROW_MAJOR.split("\n"))
        for part in (row, str(i + 1))
    )
    assert check_association(_pairs_from_rows(ROWS), interleaved).rate == 100.0


def test_over_common_values_are_not_scored():
    """A value on every line cannot be attributed to one row by proximity."""
    noisy = ROW_MAJOR + "\n" + "\n".join("filler 4231" for _ in range(10))
    result = check_association([("O&M expense", "4231")], noisy)
    assert result.ambiguous == 1
    assert result.measurable == 0


# (superseded: selection is by numeric agreement, not word coverage --
#  see test_the_native_whose_figures_round_trip_wins)


# --- Page apparatus ---


def test_enumeration_markers_are_not_negatives():
    """Tariff prose numbers subsections "(1) The Competitive Retailer...".

    Reading those as negative one manufactured 24 phantom tokens in item 795.
    """
    assert numeric_counter("(1) \nThe Competitive Retailer and Company have agreed") == {}


def test_accounting_negatives_still_sign():
    """A real (1,234) adjustment sits inside a row, never at the head of a line."""
    assert numeric_counter("O&M expense  4,231  (1,234)")["-1234"] == 1
    assert numeric_counter("Depreciation  1,208  (45)")["-45"] == 1


def test_table_of_contents_lines_are_excluded():
    toc = "4.1.1 \tAPPLICABILITY OF CHAPTER ................................ 28"
    assert numeric_counter(toc) == {}


def test_real_table_rows_survive_both_filters():
    row = "Roadway (1,000w) \n$13.44 \n104,500 \n1,100\n N/A \n367"
    counts = numeric_counter(row)
    assert counts["367"] == 1
    assert counts["104500"] == 1
    assert counts["13.44"] == 1


# --- Verdict thresholds ---


def test_a_handful_of_misses_does_not_condemn_a_document():
    """795 sits at 99.4%: six citable values out of eight thousand.

    Those fail per-claim at query time. Refusing every numeric claim citing a
    371-page tariff throws the document away to avoid six errors.
    """
    num = NumericResult(occurrences_expected=8000, occurrences_found=7952)
    assert verdict(num, AssociationResult(), structured=False, word_accuracy=99.8) == (
        "exact_unverified_structure"
    )


def test_systemic_corruption_still_refuses():
    """A genuinely degraded scan sits far below the floor, not just under 100%."""
    num = NumericResult(occurrences_expected=1000, occurrences_found=880)
    assert verdict(num, AssociationResult(), structured=False, word_accuracy=97.0) == (
        "refuse_numerics"
    )


def test_association_reports_rather_than_gates():
    """The rate is a lower bound: item 773's residual is naming variance between
    the spreadsheet and the printed attachment, not extraction failure. A number
    that cannot tell a plural from a broken row must not exclude a document at
    100% numeric fidelity."""
    num = NumericResult(occurrences_expected=1000, occurrences_found=1000)
    assoc = AssociationResult(checked=1000, intact=774)
    verd = verdict(num, assoc, structured=True, word_accuracy=100.0)
    assert verd == "exact_structure_reported"
    assert verd.startswith("exact_")


def test_cell_per_line_extraction_keeps_association():
    """Item 773 extracts one cell per line, currency symbols included.

    A row spans a dozen lines, so distance-to-label measures column position and
    nothing else. Reading order survives what distance could not.
    """
    shredded = "\n".join([
        "Transmission Ops Supr & Engr", "560", "$", "13,074", "$", "222",
        "Load Dispatch - Reliability", "561", "$", "5,073", "$", "119",
    ])
    pairs = [
        ("Transmission Ops Supr & Engr", "13074"),
        ("Load Dispatch - Reliability", "5073"),
    ]
    assert check_association(pairs, shredded).rate == 100.0


def test_a_value_under_the_wrong_heading_is_still_caught():
    """Reading order must not become a rubber stamp: a value that falls under
    another row's label is broken however close it sits."""
    shredded = "\n".join([
        "Transmission Ops Supr & Engr", "560", "13,074",
        "Load Dispatch - Reliability", "561", "5,073",
    ])
    # The label vocabulary comes from the pairs themselves, so both rows must be
    # present for either to be checkable -- which is how _pairs_from_rows builds
    # them in real use.
    pairs = [
        ("Transmission Ops Supr & Engr", "5073"),  # wrong: 5073 is the next row
        ("Load Dispatch - Reliability", "5073"),
    ]
    result = check_association(pairs, shredded)
    assert result.rate == 50.0


def test_synonym_labels_make_reading_order_meaningless():
    """Two workbooks naming the same row differently is why only one folds.

    "Meter Exp" wins the line, so the pair carrying "Meter Expenses" is scored
    broken while the row itself is perfectly intact.
    """
    shredded = "\n".join(["Meter Exp", "586", "27,262"])
    both = [("Meter Expenses", "27262"), ("Meter Exp", "27262")]
    assert check_association(both, shredded).rate == 50.0

    alone = [("Meter Expenses", "27262")]
    assert check_association(alone, shredded).rate == 100.0


# --- Ground truth must exist ---


def test_an_empty_native_is_unmeasured_not_perfect():
    """Item 773's native bundle contains a PDF of the same scan, with no text
    layer. It extracted to nothing, 0 of 0 numeric tokens matched, and the
    verdict read 100% word and 100% numeric. Absence of ground truth must never
    present as perfect agreement."""
    empty = compare_numerics("", "1,234 rows of served text")

    assert empty.occurrence_accuracy == 100.0  # the arithmetic is unavoidable
    assert verdict(
        empty, AssociationResult(), structured=False, word_accuracy=100.0, native_words=0
    ) == "no_ground_truth"


def test_a_thin_native_is_also_rejected():
    """A cover page or a one-line stub is not ground truth either."""
    thin = NumericResult(occurrences_expected=3, occurrences_found=3)

    assert verdict(
        thin, AssociationResult(), structured=False, word_accuracy=100.0, native_words=12
    ) == "no_ground_truth"


def test_a_real_native_still_scores_normally():
    num = NumericResult(occurrences_expected=8000, occurrences_found=8000)

    assert verdict(
        num, AssociationResult(), structured=False, word_accuracy=99.8, native_words=7000
    ).startswith("exact_")


def test_a_native_that_covers_little_of_the_document_is_not_ground_truth():
    """Item 773's best native is a two-page memo: 191 words, 24 numeric tokens,
    23.6% coverage of a 26-page filing. It clears every absolute floor and scores
    100%, because every figure it contains really is in the served text. The
    twenty pages of schedules it says nothing about are never examined."""
    num = NumericResult(occurrences_expected=24, occurrences_found=24)

    assert verdict(
        num, AssociationResult(), structured=True, word_accuracy=100.0,
        native_words=191, coverage=23.6,
    ) == "no_ground_truth"


def test_a_native_covering_most_of_the_document_still_scores():
    num = NumericResult(occurrences_expected=8000, occurrences_found=8000)

    assert verdict(
        num, AssociationResult(), structured=False, word_accuracy=99.8,
        native_words=7000, coverage=94.0,
    ).startswith("exact_")




# --- Ground truth is reported, not selected ---


def test_every_candidate_is_reported(tmp_path, monkeypatch):
    """Item 785's tariff exhibit round-trips 4,332 of 4,337 figures; its cover
    letter round-trips 270 of 270. Six rules tried to name one of them as THE
    native and each picked wrong somewhere. Both facts are recorded instead."""
    import ocr_accuracy as oa

    cover = tmp_path / "cover letter.docx"
    exhibit = tmp_path / "Exhibit C - Tariff.docx"
    cover.touch()
    exhibit.touch()

    prose = " ".join(f"word{a}{b}" for a in "abcdefgh" for b in "ijklmnop")
    many = " ".join(f"{n},{n:03d}" for n in range(100, 200))
    served = prose + " " + many + " 7,777"
    texts = {
        cover.name: prose + " " + " ".join(f"{n},{n:03d}" for n in range(100, 130)),
        exhibit.name: prose + " " + many + " 9,999",
    }
    monkeypatch.setattr(oa, "read_native", lambda p: oa.NativeDoc(text=texts[p.name]))

    match = oa.NativeMatch(primary=cover, candidates=[cover, exhibit])
    scores = oa.report_natives(match, served)

    assert len(scores) == 2
    # Most figures first: the verdict rests on the best-evidenced candidate,
    # not the highest rate.
    assert scores[0].path == exhibit
    assert scores[0].numeric_expected > scores[1].numeric_expected
    assert scores[1].numeric_accuracy > scores[0].numeric_accuracy


def test_natives_below_the_word_floor_are_dropped(tmp_path, monkeypatch):
    """A PDF of the same scan carries no text and is not ground truth."""
    import ocr_accuracy as oa

    empty = tmp_path / "scan.pdf"
    real = tmp_path / "real.docx"
    empty.touch()
    real.touch()

    prose = " ".join(f"word{a}{b}" for a in "abcdefgh" for b in "ijklmnop")
    texts = {empty.name: "", real.name: prose + " 1,234"}
    monkeypatch.setattr(oa, "read_native", lambda p: oa.NativeDoc(text=texts[p.name]))

    match = oa.NativeMatch(primary=empty, candidates=[empty, real])
    scores = oa.report_natives(match, prose + " 1,234")

    assert [s.path for s in scores] == [real]


def test_several_plausible_natives_defers_the_verdict():
    """Item 785's Settlement Agreement carries 12,049 figures at 95.88% because
    the native PDF holds the agreement AND its exhibits inline; Exhibit C holds
    the 4,337 that were served, at 99.89%. No single number describes the set."""
    num = NumericResult(occurrences_expected=12049, occurrences_found=11553)

    assert verdict(
        num, AssociationResult(), structured=False, word_accuracy=99.7,
        native_words=8000, coverage=94.7, candidates=5,
    ) == "multiple_candidates"


def test_a_single_native_still_gets_a_verdict():
    num = NumericResult(occurrences_expected=8081, occurrences_found=8031)

    assert verdict(
        num, AssociationResult(), structured=False, word_accuracy=99.8,
        native_words=7000, coverage=95.4, candidates=1,
    ).startswith("exact_")


def test_deferral_does_not_mask_a_missing_native():
    """A bundle of fragments is still no_ground_truth, not merely ambiguous."""
    num = NumericResult(occurrences_expected=42, occurrences_found=42)

    assert verdict(
        num, AssociationResult(), structured=False, word_accuracy=100.0,
        native_words=200, coverage=23.6, candidates=5,
    ) == "no_ground_truth"
"""Tests for the grounding guard.

Every claim here is hand-written against real text from the eligible corpus,
half of them correct and half wrong in a way that matters. A verifier is only
worth having if it separates the two, and hand-written claims test that without
a model in the loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.guard import (  # noqa: E402
    canonical_numbers,
    verify_claim,
    verify_numbers,
    verify_predicate,
    verify_span,
)


# Item 792, Bates 000000010, findings of fact 60-61.
APPROVED_SPAN = (
    "The signatories agreed that, beginning on the effective date of the rates "
    "approved by this Order, CenterPoint Houston's weighted average cost of capital "
    "will be 6.51%, based on a cost of debt of 4.38%, a return on equity of 9.4%, "
    "and a capital structure of 57.5% long-term debt and 42.5% equity."
)

# Item 792, Bates 00000001, background.
REQUESTED_SPAN = (
    "CenterPoint Houston requested an overall rate of return of 7.39%, based on a "
    "cost of debt of 4.38%, a return on equity of 10.4%, and a capital structure of "
    "50% long-term debt and 50% equity."
)

CHUNK = "PUC Docket No. 49421 Order Page 10 of 25\n" + APPROVED_SPAN + "\n61. It is appropriate."


# --- Numeric canonicalisation ---


def test_sign_and_unit_are_part_of_identity():
    """(1,234) is not 1,234 and 10.4% is not 10.4. A rate case turns on both."""
    assert canonical_numbers("1,234") == ["1234"]
    assert canonical_numbers("(1,234)") == ["-1234"]
    assert canonical_numbers("$1,234.50") == ["1234.5"]
    assert canonical_numbers("10.4%") == ["10.4%"]
    assert canonical_numbers("10.4") == ["10.4"]


def test_a_stray_closing_paren_is_not_a_negative():
    assert canonical_numbers("see line 5)") == ["5"]


# --- Span verification ---


def test_an_exact_quotation_verifies():
    ok, similarity = verify_span("a return on equity of 9.4%", CHUNK)
    assert ok and similarity == 1.0


def test_ocr_damage_does_not_refuse_a_real_quotation():
    """795-A contains 'PERIODIC BILLING RE UIREMENT' -- a dropped Q -- in a
    document measuring 99.8% word accuracy. An exact match would refuse a
    quotation that is visibly on the page."""
    chunk = "Sheet No. 6.7.2\nPERIODIC BILLING RE UIREMENT ALLOCATION FACTORS\nResidential"
    ok, similarity = verify_span("PERIODIC BILLING REQUIREMENT ALLOCATION FACTORS", chunk)
    assert ok
    assert similarity < 1.0


def test_a_fabricated_quotation_is_refused():
    ok, _ = verify_span(
        "CenterPoint Houston agreed to a return on equity of 11.2%", CHUNK
    )
    assert not ok


def test_a_short_quotation_is_not_penalised_by_chunk_length():
    """A 6-word quotation from a 2,000-character chunk would score near zero
    against the whole chunk however accurate it is."""
    long_chunk = CHUNK + " " + ("filler sentence about rate design. " * 60)
    ok, similarity = verify_span("a return on equity of 9.4%", long_chunk)
    assert ok and similarity == 1.0


# --- Numeric verification ---


def test_a_figure_the_span_does_not_contain_is_refused():
    ok, missing = verify_numbers(
        "The approved return on equity was 9.45%.", APPROVED_SPAN
    )
    assert not ok and missing == ["9.45%"]


def test_a_span_may_hold_more_figures_than_the_claim_uses():
    ok, missing = verify_numbers("The approved return on equity was 9.4%.", APPROVED_SPAN)
    assert ok and not missing


def test_9_4_and_9_45_are_different_figures():
    """The corpus holds 10.4% requested, 9.45% recommended, and 9.4% approved.
    Nothing about them is close enough to treat as a rendering difference."""
    ok, _ = verify_numbers("a return on equity of 9.45%", APPROVED_SPAN)
    assert not ok


# --- Predicate verification ---


def test_the_retrieval_trap_is_caught():
    """Retrieval's rank-1 chunk for 'what ROE did CenterPoint request?' is the
    findings-of-fact page stating the AGREED 9.4%. A claim built on it quotes
    accurately and its figure is really in the span -- span and numeric
    verification both pass, and the answer is still wrong."""
    result = verify_claim(
        "CenterPoint Houston requested a return on equity of 9.4%.",
        "a return on equity of 9.4%, and a capital structure of 57.5% long-term debt",
        CHUNK,
    )
    assert result.span_verified
    assert result.numbers_verified
    assert not result.predicate_supported
    assert not result.verified
    assert "requested" in result.failure_detail


def test_the_matching_predicate_passes():
    result = verify_claim(
        "The signatories agreed to a return on equity of 9.4%.",
        "The signatories agreed that, beginning on the effective date of the rates "
        "approved by this Order, CenterPoint Houston's weighted average cost of "
        "capital will be 6.51%, based on a cost of debt of 4.38%, a return on equity "
        "of 9.4%",
        CHUNK,
    )
    assert result.verified


def test_a_claim_asserting_no_predicate_is_not_refused():
    """Most claims assert nothing contested. Refusing them would make the guard
    useless, so the check fires only when both sides name a predicate."""
    ok, detail = verify_predicate(
        "The capital structure is 57.5% long-term debt.", APPROVED_SPAN
    )
    assert ok and detail is None


def test_requested_against_a_requested_span_passes():
    result = verify_claim(
        "CenterPoint Houston requested a return on equity of 10.4%.",
        "CenterPoint Houston requested an overall rate of return of 7.39%, based on a "
        "cost of debt of 4.38%, a return on equity of 10.4%",
        "PUC Docket No. 49421\n" + REQUESTED_SPAN,
    )
    assert result.verified


# --- Composition ---


def test_a_missing_span_short_circuits():
    """The other checks would verify a fabricated quotation against itself."""
    result = verify_claim(
        "The approved return on equity was 9.4%.",
        "The Commission approved a return on equity of 9.4% after full hearing.",
        "Unrelated text about street lighting charges and lamp types.",
    )
    assert not result.span_verified
    assert not result.numbers_verified
    assert not result.verified


def test_refusal_reason_names_which_check_failed():
    wrong_figure = verify_claim(
        "The approved return on equity was 9.45%.",
        "a return on equity of 9.4%, and a capital structure of 57.5%",
        CHUNK,
    )
    assert "9.45%" in wrong_figure.refusal_reason

    fabricated = verify_claim("x", "not in the chunk at all whatsoever", CHUNK)
    assert "not found" in fabricated.refusal_reason


def test_a_narrow_quotation_cannot_dodge_the_predicate_check():
    """The span "a return on equity of 9.4%" names no predicate at all. Checking
    only the span would let a claim quote tightly around the figure and pass."""
    ok, _ = verify_predicate(
        "CenterPoint requested a return on equity of 9.4%.",
        "a return on equity of 9.4%",
    )
    assert ok  # span alone: nothing to compare against

    ok, detail = verify_predicate(
        "CenterPoint requested a return on equity of 9.4%.",
        "a return on equity of 9.4%",
        CHUNK,
    )
    assert not ok and "requested" in detail
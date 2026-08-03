"""Tests for document-type exclusion.

Exclusions come from filing metadata, not content heuristics. The point of these
tests is that substantive filings survive: an over-eager filter that drops the
final order is worse than no filter at all.
"""

from __future__ import annotations

import pytest

from puctqa.filters import ExclusionReason, evaluate, summarize


# Real filing descriptions observed in docket 49421.
SUBSTANTIVE = [
    "APPLICATION OF CENTERPOINT ENERGY HOUSTON ELECTRIC, LLC FOR AUTHORITY TO CHANGE RATES",
    "TESTIMONY OF DARRYL TIETJEN IN SUPPORT OF STIPULATION",
    "TESTIMONY IN SUPPORT OF AGREEMENT OF KRISTIE L. COLVIN",
    "ORDER 1 SUSPENDING EFFECTIVE DATE AND ENTERING PROTECTIVE ORDER",
    "PROPOSAL FOR DECISION",
    "FINAL ORDER",
]

PROCEDURAL = [
    "CENTERPOINT ENERGY HOUSTON ELECTRIC, LLC'S MOTION TO ADMIT AGREEMENT AND SUPPORTING TESTIMONY",
    "MOTION TO INTERVENE",
    "CERTIFICATE OF SERVICE",
    "NOTICE OF APPEARANCE",
]


@pytest.mark.parametrize("description", SUBSTANTIVE)
def test_substantive_filings_are_kept(description):
    verdict = evaluate(item_type="PL", filing_description=description)
    assert verdict.include, f"wrongly excluded by {verdict.matched_rule}"


@pytest.mark.parametrize("description", PROCEDURAL)
def test_procedural_filings_are_dropped(description):
    verdict = evaluate(item_type="PL", filing_description=description)
    assert not verdict.include
    assert verdict.reason is ExclusionReason.PROCEDURAL


def test_confidential_item_type_excluded():
    """Item 2 of docket 49421: CONFIDENTIAL rate filing package."""
    verdict = evaluate(
        item_type="CONF",
        filing_description="CONFIDENTIAL- CEHE 2019 RATE FILING PACKAGE; BATE STAMP FLASH DRIVE",
    )
    assert not verdict.include
    assert verdict.reason is ExclusionReason.CONFIDENTIAL
    assert "item_type" in verdict.matched_rule


def test_confidential_caught_by_description_even_if_type_missing():
    verdict = evaluate(filing_description="CONFIDENTIAL WORKPAPERS OF WITNESS")
    assert verdict.reason is ExclusionReason.CONFIDENTIAL


def test_discovery_is_out_of_scope_not_excluded_for_risk():
    """RFI responses are legitimate record; 200 of them would swamp a 50-doc corpus."""
    verdict = evaluate(
        filing_description="Commission Staff's First RFI to Centerpoint Energy Question Nos. 1-1"
    )
    assert not verdict.include
    assert verdict.reason is ExclusionReason.OUT_OF_SCOPE


def test_pii_bearing_filings_excluded():
    for description in [
        "COMMENTS FROM JANE DOE",
        "RATEPAYER COMMENT FORM",
        "PROTEST OF PROPOSED RATE INCREASE",
    ]:
        verdict = evaluate(filing_description=description)
        assert not verdict.include
        assert verdict.reason is ExclusionReason.PII_RISK


def test_native_file_bundles_excluded():
    verdict = evaluate(filing_description="Native Files (Zip)")
    assert verdict.reason is ExclusionReason.NON_TEXT


def test_protective_order_certification_excluded():
    verdict = evaluate(filing_description="PROTECTIVE ORDER CERTIFICATIONS")
    assert verdict.reason is ExclusionReason.CONFIDENTIAL


def test_default_is_inclusion_when_metadata_absent():
    """Unknown metadata must not silently drop a document."""
    assert evaluate().include
    assert evaluate(filing_description="").include


def test_verdict_is_truthy_for_use_in_conditionals():
    assert bool(evaluate(filing_description="FINAL ORDER")) is True
    assert bool(evaluate(filing_description="MOTION TO INTERVENE")) is False


def test_summary_counts_by_reason():
    verdicts = [evaluate(filing_description=d) for d in SUBSTANTIVE + PROCEDURAL]
    counts = summarize(verdicts)
    assert counts["included"] == len(SUBSTANTIVE)
    assert counts["procedural"] == len(PROCEDURAL)


# --------------------------------------------------------------------------
# Person detection
#
# Individuals writing to the Commission are the sharpest PII case, and their
# filing descriptions are indistinguishable from the utility's. Measured in
# docket 49421: item 760 is "LETTER TO THE COMMISSIONER" from an individual,
# item 782 is "LETTER TO COMISSIONERS" from CenterPoint. Only the party differs.
# --------------------------------------------------------------------------

from puctqa.filters import looks_like_person, normalize_description  # noqa: E402


@pytest.mark.parametrize(
    "party", ["JOHN KAJANDER", "MARY ANDERSON", "M. LINDEE", "JONATHAN HJELTE", "ARLENE YAX"]
)
def test_real_individual_filers_detected(party):
    assert looks_like_person(party)


@pytest.mark.parametrize(
    "party",
    [
        "CENTERPOINT ENERGY HOUSTON ELECTRIC, LLC",
        "CENTERPOINT ENERGY",
        "PUC LEGAL",
        "PUC OPDM",
        "SOAH",
        "OPUC",
        "TIEC",
        "WALMART INC",
        "H-E-B, LP",
        "REIGNING GLORY CHURCH",
        "GULF COAST COALITION OF CITIES",
        "KENNEDY REPORTING SERVICE",
        "CENTRAL RECORDS",
        "TEXAS FOR LAWSUIT REFORM",
        "MINNEOTA UTILITY INVESTORS",
        "GENERATION PARK MANAGEMENT DISTRICT",
    ],
)
def test_organizations_not_flagged_as_people(party):
    assert not looks_like_person(party)


def test_individual_excluded_despite_innocuous_description():
    """Item 760: description alone is identical to the utility's own letters."""
    individual = evaluate(filing_description="LETTER TO THE COMMISSIONER",
                          filing_party="JONATHAN HJELTE")
    assert not individual.include
    assert individual.reason is ExclusionReason.PII_RISK
    assert individual.matched_rule == "filing_party=individual"


def test_person_detection_handles_missing_party():
    assert not looks_like_person(None)
    assert not looks_like_person("")


# --------------------------------------------------------------------------
# Description normalization
#
# Filing descriptions are hand-typed and contain typos at a rate that breaks
# exact keyword rules -- the same lesson as OCR on document text.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "typo,expected_reason",
    [
        ("CONFIDENTILITY STATEMENT UNDER SECTION 4", ExclusionReason.CONFIDENTIAL),
        ("PROTECTIVE ORDER CERTITFICATIONS", ExclusionReason.CONFIDENTIAL),
        ("RESPONSE TO CITY OF HOUSTON'S THIRD RIF QUESTION", ExclusionReason.OUT_OF_SCOPE),
        ("MOTION TO  INTERVENE OF TEXAS COAST UTILITIES", ExclusionReason.PROCEDURAL),
        ("ERATTA TO STAFF'S REPLIES", ExclusionReason.OUT_OF_SCOPE),
    ],
)
def test_real_typos_still_match(typo, expected_reason):
    verdict = evaluate(filing_description=typo)
    assert not verdict.include
    assert verdict.reason is expected_reason


def test_normalization_collapses_whitespace():
    assert normalize_description("MOTION TO   INTERVENE") == "MOTION TO INTERVENE"


def test_normalization_leaves_clean_text_alone():
    clean = "TESTIMONY OF DARRYL TIETJEN IN SUPPORT OF STIPULATION"
    assert normalize_description(clean) == clean


def test_dead_records_dropped():
    for description in ["DUPLICATE FILING- SEE ITEM 443", "VOID SEE ITEM #485",
                        "WRONG DOCKET. MOVED TO DOCKET 49628"]:
        assert not evaluate(filing_description=description).include
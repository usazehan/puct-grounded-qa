"""Tests for claim proposal.

No network and no model: backends are stubs returning fixed strings, so what is
tested is the part that must hold whatever the model does -- ids resolve against
the supplied units, invented ids are dropped, and a malformed response reads as
a refusal rather than as support.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.evidence import EvidenceUnit  # noqa: E402
from puctqa.generate import (  # noqa: E402
    build_prompt,
    echo_backend,
    parse_proposal,
    propose,
)


UNITS = [
    EvidenceUnit(
        unit_id="c17:e00",
        chunk_id=17,
        text=(
            "The signatories agreed that CenterPoint Houston's weighted average cost "
            "of capital will be 6.51%, based on a cost of debt of 4.38%, a return on "
            "equity of 9.4%, and a capital structure of 57.5% long-term debt."
        ),
        char_start=100,
        char_end=280,
        kind="prose",
    ),
    EvidenceUnit(
        unit_id="c17:e01",
        chunk_id=17,
        text="It is appropriate for CenterPoint Houston to have an overall rate of return of 6.51%.",
        char_start=281,
        char_end=366,
        kind="prose",
    ),
]


def response(status: str, claims: list[dict]) -> str:
    return json.dumps({"status": status, "claims": claims})


# --- Prompt ---


def test_the_prompt_lists_every_id():
    prompt = build_prompt("What ROE was approved?", UNITS)
    assert "[c17:e00]" in prompt
    assert "[c17:e01]" in prompt
    assert "Valid ids: c17:e00, c17:e01" in prompt


# --- Id resolution ---


def test_ids_resolve_to_segmented_text_not_model_text():
    """The span a claim carries comes from segmentation. A model that
    paraphrased its source would still cite the passage verbatim."""
    raw = response(
        "supported",
        [{"assertion": "The approved ROE was 9.4%.", "evidence_ids": ["c17:e00"]}],
    )
    proposal = parse_proposal(raw, UNITS)

    assert len(proposal.claims) == 1
    claim = proposal.claims[0]
    assert claim.quoted_span == UNITS[0].text
    assert claim.units[0].char_start == 100
    assert claim.chunk_id == 17


def test_an_invented_id_is_dropped_and_recorded():
    """A backend that invents ids is a backend to stop using, so the fact is
    kept rather than silently repaired."""
    raw = response(
        "supported",
        [{"assertion": "The approved ROE was 9.4%.", "evidence_ids": ["c99:e42"]}],
    )
    proposal = parse_proposal(raw, UNITS)

    assert proposal.refused
    assert proposal.invalid_ids == ["c99:e42"]


def test_a_claim_keeps_its_valid_ids_when_one_is_invented():
    raw = response(
        "supported",
        [{"assertion": "The approved ROE was 9.4%.", "evidence_ids": ["c17:e00", "c99:e42"]}],
    )
    proposal = parse_proposal(raw, UNITS)

    assert len(proposal.claims) == 1
    assert [u.unit_id for u in proposal.claims[0].units] == ["c17:e00"]
    assert proposal.invalid_ids == ["c99:e42"]


# --- Refusal ---


def test_no_support_is_a_refusal():
    proposal = parse_proposal(response("no_support", []), UNITS)
    assert proposal.refused and not proposal.claims


def test_a_malformed_response_is_a_refusal_not_support():
    """Guessing at an unparseable response would manufacture a claim nobody
    made."""
    proposal = parse_proposal("I'm not sure how to answer that.", UNITS)
    assert proposal.refused


def test_supported_with_no_usable_claims_is_a_refusal():
    raw = response("supported", [{"assertion": "", "evidence_ids": ["c17:e00"]}])
    assert parse_proposal(raw, UNITS).refused


def test_a_backend_failure_does_not_read_as_support():
    def broken(prompt: str, unit_ids: list[str]) -> str:
        raise ConnectionError("ollama is not running")

    proposal = propose("What ROE was approved?", UNITS, broken)
    assert proposal.refused
    assert "backend error" in proposal.raw


def test_no_units_is_a_refusal_without_calling_the_backend():
    def explode(prompt: str, unit_ids: list[str]) -> str:
        raise AssertionError("backend should not be called")

    assert propose("What ROE was approved?", [], explode).refused


# --- Fenced and noisy output ---


def test_json_inside_a_fenced_block_parses():
    raw = "Here is my answer:\n```json\n" + response(
        "supported",
        [{"assertion": "The approved ROE was 9.4%.", "evidence_ids": ["c17:e00"]}],
    ) + "\n```"
    assert not parse_proposal(raw, UNITS).refused


# --- The echo backend ---


def test_echo_selects_a_unit_sharing_a_term_with_the_question():
    proposal = propose("What was the agreed capital structure?", UNITS, echo_backend)
    assert not proposal.refused
    assert proposal.claims[0].units[0].unit_id == "c17:e00"


def test_echo_refuses_when_nothing_matches():
    proposal = propose("How many outage events were reported?", UNITS, echo_backend)
    assert proposal.refused


def test_echo_is_deterministic():
    question = "What was the agreed capital structure?"
    first = propose(question, UNITS, echo_backend)
    second = propose(question, UNITS, echo_backend)
    assert first.raw == second.raw
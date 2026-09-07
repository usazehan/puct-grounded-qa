"""Tests for the review pass.

The cases are the two failures the eval found: an answer that addressed a
different question than the one asked, and one that gave two of three reasons
and stopped. Backends are stubs returning fixed verdicts, so what is tested is
the part that must hold whatever the reviewer says.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.evidence import EvidenceUnit  # noqa: E402
from puctqa.review import (  # noqa: E402
    Review,
    build_review_prompt,
    parse_review,
    review,
)


def unit(uid: str, text: str) -> EvidenceUnit:
    return EvidenceUnit(
        unit_id=uid, chunk_id=int(uid.split(":")[0][1:]), text=text,
        char_start=0, char_end=len(text), kind="prose",
    )


# Item 786, the three reasons approval is said to be in the public interest.
CITED = [
    unit("c17:e01", "The Agreement resolves the uncertainty associated with the "
                    "outcome of a general base rate proceeding."),
    unit("c17:e02", "It provides the Company with the direction and regulatory "
                    "parameters it needs in order to operate."),
]
UNCITED = [
    unit("c17:e03", "It avoids further professional fees and expenses associated "
                    "with rate case expense recovery and appeals."),
]


def verdict(**kwargs) -> str:
    payload = {"responsive": True, "complete": True, "missing": ""}
    payload.update(kwargs)
    return json.dumps(payload)


# --- The prompt ---


def test_uncited_evidence_is_shown_to_the_reviewer():
    """A partial answer's omission sits in the units the claim did not select.
    A reviewer shown only the citations has no way to notice."""
    prompt = build_review_prompt(
        "Why is approval in the public interest?",
        ["The Agreement resolves uncertainty and provides regulatory parameters."],
        CITED,
        CITED + UNCITED,
    )

    assert "NOT cited" in prompt
    assert "c17:e03" in prompt
    assert "professional fees" in prompt


def test_a_fully_cited_answer_shows_no_uncited_section():
    prompt = build_review_prompt("Why?", ["Because."], CITED, CITED)
    assert "NOT cited" not in prompt


# --- Verdicts ---


def test_an_incomplete_answer_is_qualified_not_refused():
    """The review downgrades; it never rescues and never rejects outright."""
    result = parse_review(
        verdict(complete=False, missing="the third reason: further fees and appeals")
    )

    assert result.reviewed
    assert result.qualified
    assert "further fees" in result.caveat


def test_an_unresponsive_answer_says_so():
    result = parse_review(verdict(responsive=False))
    assert result.qualified
    assert "answer the question" in result.caveat


def test_a_clean_answer_carries_no_caveat():
    result = parse_review(verdict())
    assert result.reviewed
    assert not result.qualified
    assert result.caveat is None


# --- Failure is not a pass ---


def test_an_unparseable_review_is_unreviewed_not_clean():
    """Saying nothing about an answer is not the same as clearing it."""
    result = parse_review("I think the answer looks fine to me.")

    assert not result.reviewed
    assert not result.qualified  # nothing to qualify it with
    assert result.caveat is None


def test_a_backend_failure_is_unreviewed():
    def broken(prompt: str, unit_ids: list[str]) -> str:
        raise ConnectionError("no backend")

    result = review("Why?", ["Because."], CITED, CITED + UNCITED, broken)

    assert not result.reviewed
    assert "review backend error" in result.raw


def test_no_claims_means_nothing_to_review():
    def explode(prompt: str, unit_ids: list[str]) -> str:
        raise AssertionError("backend should not be called")

    assert not review("Why?", [], CITED, CITED, explode).reviewed


# --- Composition ---


def test_review_resolves_a_fenced_verdict():
    raw = "Here is my review:\n```json\n" + verdict(complete=False, missing="x") + "\n```"
    assert parse_review(raw).qualified


def test_the_review_runs_over_retrieved_evidence_not_only_citations():
    seen: dict = {}

    def capture(prompt: str, unit_ids: list[str]) -> str:
        seen["ids"] = unit_ids
        return verdict()

    review("Why?", ["Because."], CITED, CITED + UNCITED, capture)

    assert "c17:e03" in seen["ids"]
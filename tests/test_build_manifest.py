"""Regenerating a manifest must not discard human assertions.

`status`, `retrieval_eligible`, and `page_offset` are decisions someone made by
reading the record; a `selection_note` is the record of why. Losing those to a
routine --scan would be worse than not regenerating at all, because the file
would still look complete.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_manifest import (  # noqa: E402
    PRESERVED_DOCUMENT_FIELDS,
    PRESERVED_SET_FIELDS,
    carry_over,
    load_prior,
)


PRIOR = {
    "control_number": "49421",
    "items": [
        {
            "item_number": 795,
            "sets": [
                {
                    "set_id": "49421_773",
                    "status": "operative",
                    "retrieval_eligible": True,
                    "selection_note": "Only served set for this item.",
                    "documents": [
                        {
                            "filename": "49421_773_1043164.pdf",
                            "page_offset": 0,
                            "description": None,
                        }
                    ],
                },
                {
                    "set_id": "49421_795_a",
                    "status": "undetermined",
                    "retrieval_eligible": None,
                    "selection_note": "Left undetermined rather than guessed.",
                    "documents": [
                        {
                            "filename": "49421_795_1057873.pdf",
                            "page_offset": 100,
                            "description": "Pages 101 to 200",
                        }
                    ],
                },
            ],
        }
    ],
}


@pytest.fixture
def prior(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(PRIOR))
    return path


def _fresh_set(set_id: str) -> dict:
    return {
        "set_id": set_id,
        "status": "undetermined",
        "retrieval_eligible": None,
        "selection_note": None,
    }


def test_operative_set_survives_regeneration(prior: Path):
    sets, _ = load_prior(prior)
    merged = carry_over(_fresh_set("49421_773"), sets["49421_773"], PRESERVED_SET_FIELDS)

    assert merged["status"] == "operative"
    assert merged["retrieval_eligible"] is True
    assert merged["selection_note"] == "Only served set for this item."


def test_a_note_survives_even_when_the_decision_is_still_open(prior: Path):
    """The reasoning for leaving 795 undetermined is the valuable part."""
    sets, _ = load_prior(prior)
    merged = carry_over(
        _fresh_set("49421_795_a"), sets["49421_795_a"], PRESERVED_SET_FIELDS
    )

    assert merged["status"] == "undetermined"
    assert merged["retrieval_eligible"] is None
    assert merged["selection_note"].startswith("Left undetermined")


def test_asserted_page_offset_survives(prior: Path):
    _, docs = load_prior(prior)
    fresh = {"filename": "49421_795_1057873.pdf", "page_offset": None, "description": None}
    merged = carry_over(fresh, docs[fresh["filename"]], PRESERVED_DOCUMENT_FIELDS)

    assert merged["page_offset"] == 100
    assert merged["description"] == "Pages 101 to 200"


def test_a_new_set_gets_the_defaults(prior: Path):
    """Nothing carried means undetermined and not retrievable, as it should."""
    sets, _ = load_prior(prior)
    merged = carry_over(_fresh_set("49421_795_b"), sets.get("49421_795_b"), PRESERVED_SET_FIELDS)

    assert merged["status"] == "undetermined"
    assert merged["retrieval_eligible"] is None


def test_missing_manifest_is_not_an_error(tmp_path: Path):
    assert load_prior(tmp_path / "absent.json") == ({}, {})


def test_corrupt_manifest_refuses_rather_than_overwrites(tmp_path: Path):
    """Overwriting an unreadable manifest would destroy the assertions in it."""
    path = tmp_path / "manifest.json"
    path.write_text("{ not json")

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        load_prior(path)
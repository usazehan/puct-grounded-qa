"""Contract tests for the HTTP layer.

The pipeline is stubbed. What is tested is the boundary: that a refusal is a 200
carrying a reason, that every claim goes out with an anchor and a weakly-anchored
flag, and that a cost of nothing is null rather than zero. The pipeline itself is
tested in test_guard, test_generate and test_review.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from puctqa import api as api_module  # noqa: E402
from puctqa.generate import ProposedClaim, Usage  # noqa: E402
from puctqa.review import Review  # noqa: E402


@dataclass
class FakeHit:
    document: str = "49421_792_1054963.pdf"
    page_start: int = 11
    anchor_scheme: str = "bates"
    anchor_value: str = "Bates 000000010"
    text: str = ""


@dataclass
class FakeUnit:
    text: str = "a return on equity of 9.4%"
    chunk_id: int = 17


@dataclass
class FakeRecord:
    claim: ProposedClaim
    hit: FakeHit = field(default_factory=FakeHit)


@dataclass
class FakeAnswer:
    question: str
    verified: list = field(default_factory=list)
    refusal_reason: str | None = None
    units_offered: int = 56
    hits: list = field(default_factory=lambda: [FakeHit()] * 5)
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 2900
    review: Review = field(default_factory=Review)

    @property
    def refused(self) -> bool:
        return not self.verified


def install(monkeypatch, result: FakeAnswer) -> None:
    """Stub the pipeline and the database out of the request path."""
    import answer as answer_module

    monkeypatch.setattr(answer_module, "answer", lambda *a, **k: result)
    monkeypatch.setattr(answer_module, "persist", lambda *a, **k: None)
    monkeypatch.setattr(api_module, "psycopg", _FakePsycopg())
    api_module.state["backend"] = lambda p, ids: ""
    api_module.state["embed"] = lambda text: [0.0]


class _FakeCursor:
    def execute(self, *a, **k):
        return None

    def fetchone(self):
        return (817, 6)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def cursor(self):
        return _FakeCursor()

    def commit(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakePsycopg:
    def connect(self, *a, **k):
        return _FakeConn()


@pytest.fixture
def client():
    # lifespan is skipped: it would load a 1.2GB embedding model.
    return TestClient(api_module.app)


def verified_answer() -> FakeAnswer:
    claim = ProposedClaim(
        assertion="The Final Order approved a return on equity of 9.4%.",
        evidence_ids=["c17:e08"],
        units=[FakeUnit()],
    )
    return FakeAnswer(
        question="What ROE was approved?",
        verified=[FakeRecord(claim=claim)],
        usage=Usage(input_tokens=3683, output_tokens=83, model="claude-sonnet-4-6"),
    )


# --- Refusal is not an error ---


def test_a_refusal_is_a_200_carrying_a_reason(client, monkeypatch):
    """A refusal is the system working. A 4xx would put it in the same bucket
    as a malformed request and tempt a client to retry it."""
    install(monkeypatch, FakeAnswer(
        question="How many outage events?",
        refusal_reason="the evidence does not support an answer",
    ))

    response = client.post("/ask", json={"question": "How many outage events?"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["refusal_reason"]
    assert body["claims"] == []


# --- Citations ---


def test_every_claim_carries_a_resolvable_citation(client, monkeypatch):
    install(monkeypatch, verified_answer())

    body = client.post("/ask", json={"question": "What ROE was approved?"}).json()

    assert body["refused"] is False
    citation = body["claims"][0]["citation"]
    assert citation["document"] == "49421_792_1054963.pdf"
    assert citation["anchor"] == "Bates 000000010"
    assert citation["anchor_scheme"] == "bates"


def test_a_weak_anchor_is_flagged(client, monkeypatch):
    """14% of this corpus resolves only to a PDF page, which names a position in
    a file rather than in the record. A client cannot tell without being told."""
    result = verified_answer()
    result.verified[0].hit = FakeHit(
        anchor_scheme="pdf_page", anchor_value="PDF page 9"
    )
    install(monkeypatch, result)

    body = client.post("/ask", json={"question": "What ROE was approved?"}).json()

    assert body["claims"][0]["citation"]["weakly_anchored"] is True


def test_a_bates_anchor_is_not_flagged(client, monkeypatch):
    install(monkeypatch, verified_answer())

    body = client.post("/ask", json={"question": "What ROE was approved?"}).json()

    assert body["claims"][0]["citation"]["weakly_anchored"] is False


# --- Usage ---


def test_an_unknown_price_is_null_not_zero(client, monkeypatch):
    """0.0 would read as free rather than unmeasured."""
    result = verified_answer()
    result.usage = Usage(input_tokens=0, output_tokens=0, model=None)
    install(monkeypatch, result)

    body = client.post("/ask", json={"question": "What ROE was approved?"}).json()

    assert body["usage"]["cost_usd"] is None
    assert body["usage"]["latency_ms"] == 2900


def test_cost_is_reported_when_the_price_is_known(client, monkeypatch):
    install(monkeypatch, verified_answer())

    body = client.post("/ask", json={"question": "What ROE was approved?"}).json()

    assert body["usage"]["cost_usd"] == pytest.approx(0.012294, abs=1e-6)


# --- The review caveat ---


def test_a_review_caveat_travels_with_a_verified_answer(client, monkeypatch):
    """The claims are still verified. The caveat says the answer may not be
    what was asked for, or may not be all of it."""
    result = verified_answer()
    result.review = Review(
        complete=False, missing="the third reason", reviewed=True
    )
    install(monkeypatch, result)

    body = client.post("/ask", json={"question": "Why?", "review": True}).json()

    assert body["refused"] is False
    assert "third reason" in body["caveat"]


def test_no_caveat_when_the_review_did_not_run(client, monkeypatch):
    install(monkeypatch, verified_answer())

    body = client.post("/ask", json={"question": "What ROE?"}).json()

    assert body["caveat"] is None


# --- Validation and health ---


def test_an_empty_question_is_rejected(client):
    assert client.post("/ask", json={"question": "  "}).status_code == 422


def test_top_k_is_bounded(client):
    assert client.post(
        "/ask", json={"question": "What ROE was approved?", "top_k": 500}
    ).status_code == 422


def test_health_reports_whether_the_corpus_is_queryable(client, monkeypatch):
    """A service up over an empty index answers nothing, and reporting that as
    healthy hides the common failure: migrations applied, chunks never built."""
    monkeypatch.setattr(api_module, "psycopg", _FakePsycopg())

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["embedded_chunks"] == 817
    assert body["retrievable_sets"] == 6
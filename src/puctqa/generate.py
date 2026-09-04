"""Propose claims from retrieved evidence.

The model's job is deliberately small: read a question and a numbered list of
evidence units, and either select the ids that support an answer or say the
passages do not support one. It never writes a span, a citation, or a final
answer.

WHY IDS AND NOT QUOTATIONS

Asking a model to quote its source makes verbatim copying a capability
requirement -- a model that paraphrases fails span verification even when its
claim is correct, and the guard's fuzzy span check ends up compensating for
model behaviour rather than for OCR damage. Selecting from a closed list of ids
makes exact quotation an invariant: code resolves each id to the text and the
document offsets recorded during segmentation, so a claim cannot cite a passage
that was never given to it, and cannot cite it inaccurately.

That does not make the guard redundant. It retires one failure mode and leaves
two: a claim can assert a figure the selected evidence does not contain, and a
claim can select correct evidence and say the wrong thing about it -- which is
exactly what retrieval already produces for "what return on equity did
CenterPoint request?", where the top chunk states the agreed 9.4% rather than
the requested 10.4%.

BACKENDS

Three, behind one callable, because they answer different questions.

    anthropic  needs a key, runs anywhere, fast. The demo default.
    ollama     needs Ollama and a pulled model, no key, runs locally.
    echo       needs nothing. Deterministic, not intelligent -- it exercises
               the whole path so the test suite covers it and a reviewer who
               has cloned the repository can watch it run.

The point of the seam is that backends are comparable. How small a model can
abstain when the passages do not support an answer is a measurement, not a
guess, and the eval can make it the same way it compared retrieval arms.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .evidence import EvidenceUnit

# The model returns exactly this, and nothing else. status is an enum of two
# values and evidence_ids is constrained to the ids supplied with the request,
# so a well-formed response cannot cite something absent.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["supported", "no_support"]},
        "claims": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "assertion": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                },
                "required": ["assertion", "evidence_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "claims"],
    "additionalProperties": False,
}

INSTRUCTIONS = """\
You answer questions about Texas utility rate-case filings by selecting evidence,
not by writing prose.

You are given a question and numbered evidence units. Each unit is a verbatim
passage from a filing. Return JSON:

  {"status": "supported", "claims": [
      {"assertion": "<one sentence>", "evidence_ids": ["c17:e04"]}
  ]}

or, when the units do not support an answer:

  {"status": "no_support", "claims": []}

Rules:
- Use only the ids listed below. Never invent one.
- Each assertion must be supported by the units you cite, on its own, without
  anything you know from elsewhere.
- Every figure in an assertion must appear in the cited units. Copy figures
  exactly, including signs, decimals and percent marks.
- Say what the units say. If a unit says a figure was AGREED, do not write that
  it was REQUESTED; those are different figures in these dockets.
- Return no_support when the units are silent on the question, when they give
  two irreconcilable answers, or when answering would need a figure that is not
  there. Refusing is a correct answer.
"""


@dataclass
class ProposedClaim:
    assertion: str
    evidence_ids: list[str]
    # Resolved in code, never taken from the model.
    units: list[EvidenceUnit] = field(default_factory=list)

    @property
    def quoted_span(self) -> str:
        return " ".join(u.text for u in self.units)

    @property
    def chunk_id(self) -> int | None:
        return self.units[0].chunk_id if self.units else None


@dataclass
class Proposal:
    status: str  # supported | no_support
    claims: list[ProposedClaim] = field(default_factory=list)
    raw: str = ""
    # Ids the model returned that were not in the supplied list. A well-formed
    # response has none; recorded rather than discarded, because a backend that
    # invents ids is a backend to stop using.
    invalid_ids: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.status == "no_support" or not self.claims


class Backend(Protocol):
    def __call__(self, prompt: str, unit_ids: list[str]) -> str: ...


def render_units(units: list[EvidenceUnit]) -> str:
    return "\n".join(f"[{u.unit_id}] {u.text}" for u in units)


def build_prompt(question: str, units: list[EvidenceUnit]) -> str:
    return (
        f"{INSTRUCTIONS}\n"
        f"Question: {question}\n\n"
        f"Evidence units:\n{render_units(units)}\n\n"
        f"Valid ids: {', '.join(u.unit_id for u in units)}\n"
    )


def _extract_json(text: str) -> dict:
    """Parse the response, tolerating a fenced block or surrounding prose.

    A backend that wraps JSON in explanation is not malformed enough to discard
    -- but anything that is not parseable at all is treated as a refusal by the
    caller rather than guessed at.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def parse_proposal(raw: str, units: list[EvidenceUnit]) -> Proposal:
    """Turn a response into claims with resolved evidence.

    Ids are resolved against the supplied units, so the text and offsets a claim
    carries come from segmentation rather than from the model. An id that was
    not supplied is dropped and recorded; a claim left with no valid evidence is
    dropped entirely, because a claim with nothing behind it is exactly what the
    guard exists to refuse.
    """
    by_id = {u.unit_id: u for u in units}
    try:
        payload = _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        return Proposal(status="no_support", raw=raw)

    if payload.get("status") != "supported":
        return Proposal(status="no_support", raw=raw)

    claims: list[ProposedClaim] = []
    invalid: list[str] = []
    for item in payload.get("claims", []):
        assertion = (item.get("assertion") or "").strip()
        ids = item.get("evidence_ids") or []
        resolved = [by_id[i] for i in ids if i in by_id]
        invalid.extend(i for i in ids if i not in by_id)
        if assertion and resolved:
            claims.append(
                ProposedClaim(assertion=assertion, evidence_ids=ids, units=resolved)
            )

    return Proposal(
        status="supported" if claims else "no_support",
        claims=claims,
        raw=raw,
        invalid_ids=invalid,
    )


def propose(question: str, units: list[EvidenceUnit], backend: Backend) -> Proposal:
    if not units:
        return Proposal(status="no_support", raw="")
    prompt = build_prompt(question, units)
    try:
        raw = backend(prompt, [u.unit_id for u in units])
    except Exception as exc:  # noqa: BLE001
        # A backend failure is not a refusal by the system, but it must not read
        # as support either. Recorded so a run with a broken backend is
        # distinguishable from a run where the corpus was silent.
        return Proposal(status="no_support", raw=f"backend error: {exc}")
    return parse_proposal(raw, units)


# --- Backends ---


def echo_backend(prompt: str, unit_ids: list[str]) -> str:
    """Deterministic stand-in. Not intelligent; it exercises the path.

    Selects the first unit whose text shares a distinctive token with the
    question, and refuses otherwise. That is enough to prove the pipeline runs
    end to end with no key, no download, and no network -- which is what lets
    the test suite cover the generator at all.
    """
    question = prompt.split("Question:", 1)[-1].split("\n", 1)[0].lower()
    words = {w.strip("?,.'\"") for w in question.split() if len(w) > 4}

    for line in prompt.splitlines():
        match = re.match(r"\[([^\]]+)\]\s*(.+)", line)
        if not match:
            continue
        unit_id, text = match.group(1), match.group(2)
        if unit_id not in unit_ids:
            continue
        if words & {w.strip("?,.'\"").lower() for w in text.split() if len(w) > 4}:
            return json.dumps(
                {
                    "status": "supported",
                    "claims": [{"assertion": text[:200], "evidence_ids": [unit_id]}],
                }
            )
    return json.dumps({"status": "no_support", "claims": []})


def ollama_backend(
    model: str = "qwen3.5:9b", host: str = "http://localhost:11434"
) -> Backend:
    """Local generation through Ollama. No key; needs `ollama pull <model>`.

    Uses Ollama's structured-output support so the response conforms to the
    schema. That guarantees valid JSON and the two status values -- it does not
    guarantee the ids are real, which is why parse_proposal resolves them
    against the supplied units rather than trusting them.
    """

    def call(prompt: str, unit_ids: list[str]) -> str:
        schema = json.loads(json.dumps(RESPONSE_SCHEMA))
        # Constrain ids to exactly what this request supplied.
        schema["properties"]["claims"]["items"]["properties"]["evidence_ids"][
            "items"
        ] = {"type": "string", "enum": unit_ids}

        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": schema,
                # Greedy. A claim that changes between runs is not a claim about
                # the record.
                "options": {"temperature": 0},
            }
        ).encode()
        request = urllib.request.Request(
            f"{host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())["response"]

    return call


def anthropic_backend(
    model: str = "claude-sonnet-4-6", api_key: str | None = None
) -> Backend:
    """Hosted generation. Needs ANTHROPIC_API_KEY.

    The default for a demo: one dependency, works on any machine, and the
    volume here is a fraction of a cent per eval run.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Use --backend ollama for a local "
            "model, or --backend echo to exercise the pipeline without one."
        )

    def call(prompt: str, unit_ids: list[str]) -> str:
        body = json.dumps(
            {
                "model": model,
                "max_tokens": 1024,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
        return "".join(
            block.get("text", "") for block in payload.get("content", [])
        )

    return call


BACKENDS: dict[str, Callable[..., Backend]] = {
    "echo": lambda **_: echo_backend,
    "ollama": ollama_backend,
    "anthropic": anthropic_backend,
}


__all__ = [
    "BACKENDS",
    "ProposedClaim",
    "Proposal",
    "anthropic_backend",
    "build_prompt",
    "echo_backend",
    "ollama_backend",
    "parse_proposal",
    "propose",
]
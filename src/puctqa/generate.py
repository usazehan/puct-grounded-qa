"""Propose claims from retrieved evidence.

The model reads a question and a numbered list of evidence units, and either
selects the ids supporting an answer or says the passages do not support one.
It never writes a span, a citation, or a final answer.

Ids rather than quotations: asking a model to quote its source makes verbatim
copying a capability requirement, and a model that paraphrases fails span
verification even when its claim is right. Selecting from a closed list makes
exact quotation an invariant, since code resolves each id to text and offsets
recorded during segmentation.

That retires one failure mode and leaves two -- a claim can assert a figure its
evidence lacks, or select correct evidence and say the wrong thing about it.

Three backends: anthropic (a key), ollama (local), echo (neither; not
intelligent, but it makes the pipeline testable).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .evidence import EvidenceUnit

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
- A table row is a label followed by its columns, in the order the page context
  gives them. Use the context to tell which column a figure belongs to, and
  cite the row.
"""


# Per-million-token prices, input and output. Hard coded because a cost figure
# whose rate nobody can see is not a cost figure, and because the alternative --
# reading prices at runtime -- makes a recorded cost depend on when the report
# was generated rather than on what the run actually did.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


@dataclass
class Usage:
    """What one call cost, in tokens and seconds.

    A backend that cannot report tokens leaves them at zero; the distinction
    between "free" and "unmeasured" is carried by `model`, which is None for the
    echo backend. Latency is measured for every backend, because a local model
    that answers in forty seconds is a different product from one that answers
    in two, whatever the token count says.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model: str | None = None

    @property
    def cost_usd(self) -> float | None:
        # None when the price is unknown, never zero
        
        rates = PRICES_PER_MTOK.get(self.model or "")
        if rates is None:
            return None
        return (
            self.input_tokens * rates[0] + self.output_tokens * rates[1]
        ) / 1_000_000


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
    usage: Usage = field(default_factory=Usage)

    invalid_ids: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.status == "no_support" or not self.claims


class Backend(Protocol):
    def __call__(self, prompt: str, unit_ids: list[str]) -> str: ...

    # Set by the backend after each call. An attribute rather than a return
    # value so the seam stays `str -> str` and a caller that does not care about
    # cost is unaffected.
    last_usage: Usage


def render_units(units: list[EvidenceUnit]) -> str:
    return "\n".join(f"[{u.unit_id}] {u.text}" for u in units)


def build_prompt(
    question: str,
    units: list[EvidenceUnit],
    context: dict[int, str] | None = None,
) -> str:
    # Assemble the request
    # `context` maps a chunk id to its page header
    
    parts = [INSTRUCTIONS, f"Question: {question}", ""]
    if context:
        parts.append(
            "Page context (read this to interpret the units; it is NOT "
            "citable and has no ids):"
        )
        for chunk_id, header in context.items():
            if header:
                parts.append(f"  chunk c{chunk_id}: {' | '.join(header.split(chr(10)))}")
        parts.append("")
    parts.append(f"Evidence units:\n{render_units(units)}")
    parts.append("")
    parts.append(f"Valid ids: {', '.join(u.unit_id for u in units)}")
    return "\n".join(parts) + "\n"


def _extract_json(text: str) -> dict:
    # Parse the response, tolerating a fenced block or surrounding prose
    
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def parse_proposal(raw: str, units: list[EvidenceUnit]) -> Proposal:
    # Turn a response into claims with resolved evidence.
    
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


def propose(
    question: str,
    units: list[EvidenceUnit],
    backend: Backend,
    context: dict[int, str] | None = None,
) -> Proposal:
    if not units:
        return Proposal(status="no_support", raw="")
    prompt = build_prompt(question, units, context)
    try:
        raw = backend(prompt, [u.unit_id for u in units])
    except Exception as exc:  
        return Proposal(status="no_support", raw=f"backend error: {exc}")
    proposal = parse_proposal(raw, units)
    proposal.usage = getattr(backend, "last_usage", Usage())
    return proposal




def _timed(fn):
    # Record latency around a backend call

    def wrapper(prompt: str, unit_ids: list[str]) -> str:
        start = time.perf_counter()
        try:
            return fn(prompt, unit_ids)
        finally:
            inner = getattr(fn, "last_usage", None)
            if inner is not None:
                wrapper.last_usage = inner
            wrapper.last_usage.latency_ms = int((time.perf_counter() - start) * 1000)

    wrapper.last_usage = Usage()
    fn.last_usage = Usage()
    return wrapper


def _echo(prompt: str, unit_ids: list[str]) -> str:
    """Deterministic stand-in. Not intelligent; it exercises the path.
    Selects the first unit whose text shares a distinctive token with the
    question, and refuses otherwise. 
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


echo_backend = _timed(_echo)


def ollama_backend(
    model: str = "qwen3.5:9b", host: str = "http://localhost:11434"
) -> Backend:
    # Local generation through Ollama. No key; needs `ollama pull <model>`.

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
            payload = json.loads(response.read())
        # Ollama reports token counts; it has no price, so cost stays None.
        call.last_usage = Usage(
            input_tokens=payload.get("prompt_eval_count", 0),
            output_tokens=payload.get("eval_count", 0),
            latency_ms=call.last_usage.latency_ms,
            model=None,
        )
        return payload["response"]

    return _timed(call)


def anthropic_backend(
    model: str = "claude-sonnet-4-6", api_key: str | None = None
) -> Backend:
    # Hosted generation. Needs ANTHROPIC_API_KEY.
    
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
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # The API explains what it rejected; a bare status code does not.
            # "HTTP Error 400" hid a message that said the credit balance was
            # too low.
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"anthropic {exc.code}: {detail}") from exc

        usage = payload.get("usage", {})
        call.last_usage = Usage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=call.last_usage.latency_ms,
            model=model,
        )
        return "".join(
            block.get("text", "") for block in payload.get("content", [])
        )

    return _timed(call)


BACKENDS: dict[str, Callable[..., Backend]] = {
    "echo": lambda **_: echo_backend,
    "ollama": ollama_backend,
    "anthropic": anthropic_backend,
}


__all__ = [
    "BACKENDS",
    "PRICES_PER_MTOK",
    "Usage",
    "ProposedClaim",
    "Proposal",
    "anthropic_backend",
    "build_prompt",
    "echo_backend",
    "ollama_backend",
    "parse_proposal",
    "propose",
]
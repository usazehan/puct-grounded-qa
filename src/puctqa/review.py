"""Review a verified answer for responsiveness and completeness.

The guard asks three questions, all variants of *is this claim supported by the
evidence it cites*: does the quoted span appear in the chunk, does every figure
appear in the span, does the predicate match. A claim can pass all three and
still fail a reader in two ways neither check can see.

    RESPONSIVE   does the claim answer the question that was asked?
                 Asked what an adjustment CONSISTED OF, the system explained the
                 witness's rationale instead. Every word true, every word cited,
                 and not the question.

    COMPLETE     does the claim say what the retrieved evidence says?
                 Asked why approval is in the public interest, it gave two of
                 the three reasons the document lists and stopped. Asked for the
                 residential PBRAF, it could have named one schedule out of
                 three and passed every check.

THIS CHECK IS DIFFERENT IN KIND FROM THE OTHER THREE, AND THAT MATTERS

The guard is deterministic. Given a claim and a chunk, anyone can rerun it and
get the same answer, and a reader who disagrees can look at the span themselves.
"Does this answer the question" is not mechanically decidable, so this pass asks
a model -- which means a model is grading a model, the thing the rest of this
design avoids.

Three consequences, all deliberate:

  - It runs AFTER verification, never instead of it. An unsupported claim is
    rejected by the guard and never reaches here.
  - It downgrades, never upgrades. It can turn a verified answer into a
    qualified or refused one; it cannot rescue a claim the guard rejected.
  - Its verdict is recorded separately, so a results table can report the
    deterministic checks and this one apart. Reporting them together would
    lend this the guarantee the others have.

The evidence it reviews is what was RETRIEVED, not only what was cited. That is
the point: the omission in a partial answer is sitting in the units the claim
did not select.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .evidence import EvidenceUnit

REVIEW_INSTRUCTIONS = """\
You are reviewing an answer that has already been verified: every claim quotes
its source accurately and every figure is present in the quoted evidence. Do not
re-check any of that.

Judge two things, and nothing else.

RESPONSIVE — does the answer address the question that was asked? An answer can
be entirely true and cited and still answer a different question. "What did the
adjustment consist of" asks for its components; explaining why the witness
proposed it is not responsive.

COMPLETE — does the answer say what the evidence says? The evidence below
includes units the answer did not cite. If one of them carries part of the
answer — another reason, another schedule, another component — the answer is
incomplete. Name what is missing.

Return JSON:

  {"responsive": true, "complete": false,
   "missing": "the third reason: avoids further professional fees and appeals"}

`missing` is a short phrase, empty when nothing is missing. Do not restate the
answer. Do not add claims of your own. If the answer is responsive and complete,
say so plainly rather than looking for something to criticise.
"""


@dataclass
class Review:
    responsive: bool = True
    complete: bool = True
    missing: str = ""
    raw: str = ""
    # A review that could not be obtained is not a pass. Recorded so a run with
    # a broken backend is distinguishable from one that was reviewed and cleared.
    reviewed: bool = False

    @property
    def qualified(self) -> bool:
        """Should the answer be returned with a caveat rather than plainly?"""
        return self.reviewed and (not self.responsive or not self.complete)

    @property
    def caveat(self) -> str | None:
        if not self.qualified:
            return None
        if not self.responsive:
            return "this may not answer the question as asked"
        return f"this may be incomplete — {self.missing}" if self.missing else (
            "this may be incomplete"
        )


def build_review_prompt(
    question: str,
    claims: list[str],
    cited: list[EvidenceUnit],
    retrieved: list[EvidenceUnit],
) -> str:
    """Show the answer, the evidence it used, and the evidence it did not.

    Uncited units are what makes the completeness judgement possible: a partial
    answer's omission is sitting in the units the claim did not select, and a
    reviewer shown only the citations has no way to notice.
    """
    cited_ids = {u.unit_id for u in cited}
    uncited = [u for u in retrieved if u.unit_id not in cited_ids]

    parts = [
        REVIEW_INSTRUCTIONS,
        f"Question: {question}",
        "",
        "Answer:",
        *(f"  {c}" for c in claims),
        "",
        "Evidence the answer cited:",
        *(f"  [{u.unit_id}] {u.text}" for u in cited),
    ]
    if uncited:
        parts += [
            "",
            "Evidence that was retrieved and NOT cited:",
            *(f"  [{u.unit_id}] {u.text}" for u in uncited[:40]),
        ]
    return "\n".join(parts) + "\n"


def parse_review(raw: str) -> Review:
    """Read the verdict, treating anything unparseable as unreviewed.

    Not as a failure and not as a pass: an unparseable review says nothing about
    the answer, and recording it as either would be a claim this function cannot
    support.
    """
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return Review(raw=raw, reviewed=False)
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return Review(raw=raw, reviewed=False)

    return Review(
        responsive=bool(payload.get("responsive", True)),
        complete=bool(payload.get("complete", True)),
        missing=(payload.get("missing") or "").strip(),
        raw=raw,
        reviewed=True,
    )


def review(
    question: str,
    claims: list[str],
    cited: list[EvidenceUnit],
    retrieved: list[EvidenceUnit],
    backend,
) -> Review:
    """Run the review pass. Never raises; a failed review is an unreviewed one."""
    if not claims:
        return Review(reviewed=False)
    prompt = build_review_prompt(question, claims, cited, retrieved)
    try:
        raw = backend(prompt, [u.unit_id for u in retrieved])
    except Exception as exc:  # noqa: BLE001
        return Review(raw=f"review backend error: {exc}", reviewed=False)
    return parse_review(raw)


__all__ = ["Review", "build_review_prompt", "parse_review", "review"]
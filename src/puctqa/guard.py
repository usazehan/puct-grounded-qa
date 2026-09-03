"""Verify a claim against the span it cites.

Two checks, deliberately separate, because they fail for different reasons and
a system that collapsed them would lose which one failed.

SPAN VERIFICATION IS FUZZY

The quoted span must actually appear in the cited chunk, but not byte for byte.
The corpus contains "PERIODIC BILLING RE UIREMENT" -- a dropped Q -- and
"ATIACHMENT D" in item 795-A, which measured 99.8% word accuracy overall. An
exact match would refuse a quotation that is visibly on the page.

NUMERIC VERIFICATION IS EXACT

Item 795-A round-trips 8,031 of 8,081 figures against a native covering 95.4%
of its text. Digits survive extraction, so a figure that does not match is
wrong rather than merely rendered differently. Sign and unit are part of
identity: (1,234) is not 1,234, and 10.4% is not 10.4.

WHAT NEITHER CHECK CATCHES

Retrieval's rank-1 chunk for "what return on equity did CenterPoint request?"
is the findings-of-fact page stating the *agreed* 9.4%. A claim quoting it
passes span verification -- the text is really there -- and passes numeric
verification -- 9.4 is really in it. It is still the wrong answer, because the
question asked what was requested and the span says what was agreed.

So there is a third check: the claim's predicate must be supported by the span.
"CenterPoint requested" against a span saying "the signatories agreed" is a
mismatch that no amount of quoting accuracy repairs. This is the weakest of the
three and the one most likely to need revision -- it is a word-level test
standing in for a semantic one -- but leaving it out means the guard verifies
two things carefully and misses the failure that actually occurs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher

WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Same pattern the extraction-fidelity measurement uses. Parenthesised
# negatives, currency, and percent are captured rather than discarded, because
# each changes what a figure means.
NUM_RE = re.compile(
    r"""
    (?P<open>\()?
    \s*\$?\s*
    (?P<sign>-)?
    (?P<num>\d[\d,]*(?:\.\d+)?)
    \s*(?P<close>\))?
    \s*(?P<pct>%)?
    """,
    re.VERBOSE,
)

# Below this similarity the quoted span is not a rendering difference, it is a
# different passage. Measured word accuracy on the eligible corpus is 99.6-100%,
# so a genuine quotation lands far above this and a fabricated one far below.
SPAN_SIMILARITY_FLOOR = 0.85

# Predicates that assert different things about the same figure. A rate case
# turns on these distinctions: requested, recommended, and approved are three
# different numbers in this docket (10.4%, 9.45%, 9.4%), and a claim that swaps
# them is wrong while quoting perfectly.
CONTESTED_PREDICATES = {
    "requested": {"request", "requested", "sought", "proposed", "initially"},
    "recommended": {"recommend", "recommended", "recommendation", "proposal"},
    "agreed": {"agreed", "agreement", "stipulated", "signatories", "settlement"},
    "approved": {"approved", "approves", "ordered", "orders", "must", "adopted"},
    "current": {"current", "currently", "existing", "present"},
}


@dataclass
class Verification:
    span_verified: bool
    numbers_verified: bool
    predicate_supported: bool
    failure_detail: str | None = None
    missing_numbers: list[str] = field(default_factory=list)
    span_similarity: float = 0.0

    @property
    def verified(self) -> bool:
        return self.span_verified and self.numbers_verified and self.predicate_supported

    @property
    def refusal_reason(self) -> str | None:
        if self.verified:
            return None
        if not self.span_verified:
            return "quoted span not found in the cited chunk"
        if not self.numbers_verified:
            return f"figures not in the span: {', '.join(self.missing_numbers)}"
        return "the span does not support what the claim asserts"


def canonical_numbers(text: str) -> list[str]:
    """Canonical numeric tokens: signed value, percent preserved.

    1,234 -> "1234"   (1,234) -> "-1234"   $1,234.50 -> "1234.5"   10.4% -> "10.4%"

    A closing paren signs the value only if an opening one was captured too, so
    "see line 5)" is not negative five.
    """
    tokens: list[str] = []
    for match in NUM_RE.finditer(text):
        raw = match["num"].replace(",", "")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        if match["sign"] or (match["open"] and match["close"]):
            value = -value
        tokens.append(format(value.normalize(), "f") + ("%" if match["pct"] else ""))
    return tokens


def verify_span(quoted: str, chunk_text: str) -> tuple[bool, float]:
    """Does the quoted span appear in the chunk, allowing for OCR damage?

    Compared on normalised whitespace and case, then by best-matching window
    rather than whole-text similarity: a 40-word quotation from a 2,000
    character chunk would score low against the whole chunk however accurate it
    is.
    """
    needle = " ".join(quoted.split()).lower()
    haystack = " ".join(chunk_text.split()).lower()
    if not needle:
        return False, 0.0
    if needle in haystack:
        return True, 1.0

    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    match = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    if not match.size:
        return False, 0.0
    window = haystack[match.b - match.a : match.b - match.a + len(needle)]
    similarity = SequenceMatcher(None, needle, window, autojunk=False).ratio()
    return similarity >= SPAN_SIMILARITY_FLOOR, similarity


def verify_numbers(claim_text: str, quoted: str) -> tuple[bool, list[str]]:
    """Is every figure the claim asserts present in the span it quotes?

    Exact, because digits survive extraction here. Direction matters: a claim
    may quote a span holding more figures than it uses, but it may not assert a
    figure the span does not contain.
    """
    claimed = canonical_numbers(claim_text)
    available = canonical_numbers(quoted)
    missing = [n for n in claimed if n not in available]
    return not missing, missing


def verify_predicate(
    claim_text: str, quoted: str, chunk_text: str | None = None
) -> tuple[bool, str | None]:
    """Does the span support what the claim asserts about its figure?

    The check that catches the failure the other two cannot. In this docket
    "requested", "recommended" and "approved" name three different return-on-
    equity figures, and retrieval's top result for a question about one of them
    is often the page stating another. A claim asserting one predicate while
    quoting a span that states a different one is wrong however exactly it
    quotes.

    Checked against the span first, then the CHUNK the span came from. A narrow
    quotation often contains no predicate at all -- "a return on equity of 9.4%"
    names none -- and checking only the span lets a claim dodge the check by
    quoting tightly around the figure. The predicate is a property of the
    passage, and the passage is the chunk; the figure is what must be in the
    span.

    Deliberately narrow otherwise: it fires only when the claim names a
    contested predicate AND the source names a different one. An unmarked claim
    passes, because most claims assert nothing of the kind and refusing them
    would make the guard useless.
    """
    claim_words = {w.lower() for w in WORD_RE.findall(claim_text)}
    span_words = {w.lower() for w in WORD_RE.findall(quoted)}
    if chunk_text and not any(
        span_words & markers for markers in CONTESTED_PREDICATES.values()
    ):
        span_words = {w.lower() for w in WORD_RE.findall(chunk_text)}

    claim_predicates = {
        name for name, markers in CONTESTED_PREDICATES.items() if claim_words & markers
    }
    span_predicates = {
        name for name, markers in CONTESTED_PREDICATES.items() if span_words & markers
    }

    if not claim_predicates or not span_predicates:
        return True, None
    if claim_predicates & span_predicates:
        return True, None
    return False, (
        f"claim asserts {'/'.join(sorted(claim_predicates))} "
        f"but the source states {'/'.join(sorted(span_predicates))}"
    )


def verify_claim(claim_text: str, quoted: str, chunk_text: str) -> Verification:
    """Run all three checks. A claim is verified only if all three pass."""
    span_ok, similarity = verify_span(quoted, chunk_text)
    if not span_ok:
        # The other checks are meaningless against a span that is not in the
        # chunk: they would verify a quotation against itself.
        return Verification(
            span_verified=False,
            numbers_verified=False,
            predicate_supported=False,
            span_similarity=similarity,
            failure_detail=(
                f"quoted span not found in chunk (best similarity {similarity:.2f})"
            ),
        )

    numbers_ok, missing = verify_numbers(claim_text, quoted)
    predicate_ok, predicate_detail = verify_predicate(claim_text, quoted, chunk_text)

    detail = None
    if not numbers_ok:
        detail = f"figures asserted but absent from the span: {', '.join(missing)}"
    elif not predicate_ok:
        detail = predicate_detail

    return Verification(
        span_verified=True,
        numbers_verified=numbers_ok,
        predicate_supported=predicate_ok,
        failure_detail=detail,
        missing_numbers=missing,
        span_similarity=similarity,
    )


__all__ = [
    "Verification",
    "canonical_numbers",
    "verify_claim",
    "verify_numbers",
    "verify_predicate",
    "verify_span",
]
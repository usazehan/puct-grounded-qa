"""Verify a claim against the span it cites.

Three checks, separate because they fail for different reasons and collapsing
them would lose which one failed.

Span verification is fuzzy: the corpus contains "PERIODIC BILLING RE UIREMENT"
and "ATIACHMENT D" in a document measuring 99.8% word accuracy, and an exact
match would refuse a quotation that is visibly on the page.

Numeric verification is exact, and sign and unit are part of identity:
(1,234) is not 1,234, and 10.4% is not 10.4.

Predicate verification catches what neither can. Retrieval's top chunk for
"what return on equity did CenterPoint request?" states the AGREED 9.4% rather
than the requested 10.4%; a claim built on it quotes accurately and its figure
is present. Three figures in this docket are one word apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher

WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Parenthesised negatives, currency and percent are captured rather than
# discarded: each changes what a figure means.
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

# Below this the quoted span is a different passage, not a rendering difference.
# Word accuracy on the eligible corpus is 99.6-100%.
SPAN_SIMILARITY_FLOOR = 0.85

# Requested, recommended and settled are three different numbers in this docket
# -- 10.4%, 9.45%, 9.4% -- and a claim that swaps them is wrong while quoting
# perfectly. "agreed" and "approved" are one group: the signatories agreed and
# the Commission approved the same figure, and separating them refused the
# clearest question in the eval set.
CONTESTED_PREDICATES = {
    "requested": {"request", "requested", "sought", "proposed", "initially"},
    "recommended": {"recommend", "recommended", "recommendation", "proposal"},
    "settled": {
        "agreed", "agreement", "stipulated", "signatories", "settlement",
        "approved", "approves", "adopted", "ordered", "orders", "must",
    },
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
    # Canonical numeric tokens: signed value, percent preserved.
    # 1,234 -> "1234"   (1,234) -> "-1234"   $1,234.50 -> "1234.5"   10.4% -> "10.4%"
    
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
    # Does the quoted span appear in the chunk, allowing for OCR damage?
    
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
    # Is every figure the claim asserts present in the span it quotes?
    
    claimed = canonical_numbers(claim_text)
    available = canonical_numbers(quoted)
    missing = [n for n in claimed if n not in available]
    return not missing, missing


def verify_predicate(
    claim_text: str, quoted: str, chunk_text: str | None = None
) -> tuple[bool, str | None]:
    # Does the span support what the claim asserts about its figure?
    
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
    # Run all three checks. A claim is verified only if all three pass
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


def verify_claim_over_units(
    claim_text: str,
    units: list[tuple[str, str]],
) -> Verification:
    # Verify a claim whose evidence spans several units, possibly several chunks.
    
    if not units:
        return Verification(
            span_verified=False,
            numbers_verified=False,
            predicate_supported=False,
            failure_detail="the claim cites no evidence",
        )

    unverified = [
        text for text, chunk in units if not verify_span(text, chunk)[0]
    ]
    joined = " ".join(text for text, _ in units)
    context = "\n\n".join(dict.fromkeys(chunk for _, chunk in units))

    numbers_ok, missing = verify_numbers(claim_text, joined)
    predicate_ok, predicate_detail = verify_predicate(claim_text, joined, context)

    detail = None
    if unverified:
        detail = (
            f"{len(unverified)} cited unit(s) not found in the chunk they came from"
        )
    elif not numbers_ok:
        detail = f"figures asserted but absent from the evidence: {', '.join(missing)}"
    elif not predicate_ok:
        detail = predicate_detail

    return Verification(
        span_verified=not unverified,
        numbers_verified=numbers_ok,
        predicate_supported=predicate_ok,
        missing_numbers=missing,
        failure_detail=detail,
    )


__all__ = [
    "Verification",
    "canonical_numbers",
    "verify_claim",
    "verify_claim_over_units",
    "verify_numbers",
    "verify_predicate",
    "verify_span",
]
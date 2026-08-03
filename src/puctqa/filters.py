"""Which filings enter the index.

A docket is mostly procedure. Docket 49421 has 797 filings; the substantive
record — application, testimony, staff recommendation, order — is a small
fraction. The rest is motions, notices, certificates of service, and protective
order certifications, none of which anyone asks questions about.

Excluding them is not just tidiness. Measured on item 787 (Motion to Admit
Agreement), a two-page procedural filing:

    chars/page          663      (vs 1454-1991 for testimony)
    dollar figures      0
    citation anchors    none     (no Bates, no "Page N of M")
    contact details     4 emails, 6 phone numbers

Indexing that adds retrieval noise and pulls personal contact details into the
corpus, in exchange for nothing.

DESIGN NOTE — metadata, not content heuristics.

The filter reads the Interchange's own `item_type` and `filing_description`.
It would be easy to instead infer "this looks procedural" from low chars/page
or absent dollar figures, and it would be a mistake: a short substantive order
would be silently dropped, and there would be no way to audit why a document is
missing. Metadata rules are transparent, testable, and reviewable by someone who
knows the domain but not the code.

EXCLUSION IS ALSO THE PII CONTROL. Removing a document type is a stronger
guarantee than scrubbing text inside it: there is no regex to get wrong and no
partial match to leak. Ratepayer comment forms carry home addresses; procedural
filings carry counsel contact details. Both are excluded by type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ExclusionReason(str, Enum):
    CONFIDENTIAL = "confidential"
    PII_RISK = "pii_risk"
    PROCEDURAL = "procedural"
    NON_TEXT = "non_text"
    OUT_OF_SCOPE = "out_of_scope"


# Tokens that mark a filing party as an organization rather than a person.
# Built from the 797-row export for docket 49421; extend as new dockets arrive.
ORG_TOKENS = {
    "LLC", "LP", "INC", "INC.", "LTD", "CORP", "CORPORATION", "COMPANY", "CO",
    "COALITION", "ASSOCIATION", "ASSOCIATIONS", "ALLIANCE", "ADVOCATES",
    "CONSUMERS", "MARKETS", "INVESTORS", "DISTRICT", "CHURCH", "CITY", "CITIES",
    "COUNTY", "STATE", "COMMISSION", "PUC", "SOAH", "OPUC", "TIEC", "TCUC",
    "ENERGY", "ELECTRIC", "POWER", "UTILITY", "UTILITIES", "SOLAR", "TEXAS",
    "SERVICE", "SERVICES", "REPORTING", "RECORDS", "LEGAL", "OPDM", "STAFF",
    "REFORM", "MANAGEMENT", "MGMT", "DEVELOPMENT", "AMERICA", "NORTH", "GROUP",
    "PARTNERS", "HOLDINGS", "WALMART", "AEP", "TEAM", "HEB", "H-E-B",
}


def looks_like_person(party: str | None) -> bool:
    """Heuristic: is this filing party a private individual?

    Private individuals writing to the Commission about their electricity bill
    are the sharpest PII case in the corpus. Their filings are also textually
    indistinguishable from the utility's: measured in docket 49421, item 760 is
    "LETTER TO THE COMMISSIONER" from an individual while item 782 is "LETTER TO
    COMISSIONERS" from CenterPoint. The description cannot separate them; only
    the party can.

    Deliberately conservative in the safe direction -- an organization
    misclassified as a person costs one excluded document; a person
    misclassified as an organization puts their name in a public index.
    """
    if not party:
        return False
    cleaned = party.replace(",", " ").replace(".", ". ").strip()
    tokens = [t for t in cleaned.split() if t]
    if not 2 <= len(tokens) <= 3:
        return False  # acronyms and long org names are not people
    upper = {t.upper().rstrip(".") for t in tokens}
    if upper & {t.rstrip(".") for t in ORG_TOKENS}:
        return False
    # Each token should be a name-shaped word or an initial ("M.")
    return all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*\.?", t) for t in tokens)


# Interchange item types. CONF is sealed material -- the Commission restricted
# it, and indexing it would be indefensible regardless of technical access.
EXCLUDED_ITEM_TYPES = {"CONF"}

# Matched case-insensitively against the filing description as whole phrases.
# Counts below are from the 797-filing export for docket 49421.
DESCRIPTION_RULES: list[tuple[str, ExclusionReason]] = [
    # Sealed or protected -- 207 filings, 26% of the docket
    (r"\bconfidential", ExclusionReason.CONFIDENTIAL),
    (r"\bhighly sensitive\b", ExclusionReason.CONFIDENTIAL),
    (r"\bprotective order certification", ExclusionReason.CONFIDENTIAL),
    (r"\blist of issu", ExclusionReason.PROCEDURAL),
    (r"\bprocedural schedule\b", ExclusionReason.PROCEDURAL),
    (r"\bproposed notice\b|\brecommendation on notice\b", ExclusionReason.PROCEDURAL),
    (r"\bmoved to correct\b", ExclusionReason.PROCEDURAL),
    (r"\baffidavit of\b", ExclusionReason.PROCEDURAL),
    (r"\bprehearing conference\b|\bhearing on the merits\b", ExclusionReason.OUT_OF_SCOPE),
    (r"\bpages \d+ ?- ?\d+", ExclusionReason.OUT_OF_SCOPE),
    (r"\btable of contents\b|\boffer of proof\b", ExclusionReason.OUT_OF_SCOPE),
    (r"\bconfidentiality statement\b", ExclusionReason.CONFIDENTIAL),
    # Personal information
    (r"\bratepayer\b", ExclusionReason.PII_RISK),
    (r"\bletter of (protest|complaint)\b", ExclusionReason.PII_RISK),
    (r"\bprotest\b", ExclusionReason.PII_RISK),
    (r"\bcomments? (from|of|by)\b", ExclusionReason.PII_RISK),
    (r"\bcustomer (comment|complaint)", ExclusionReason.PII_RISK),
    # Correspondence and procedure -- no substantive record content
    (r"^comments?$", ExclusionReason.PROCEDURAL),
    (r"\bletter to\s+(the\s+)?[\w\s]*?(commissioner|chairman|judge|mr\.|ms\.)", ExclusionReason.PROCEDURAL),
    (r"\bcorrespondence\b", ExclusionReason.PROCEDURAL),
    (r"\b(motion|petition) to\s+(admit|intervene|withdraw|extend|compel|strike|consolidate|sever)\b",
     ExclusionReason.PROCEDURAL),
    (r"\bcertificate of service\b", ExclusionReason.PROCEDURAL),
    (r"\bnotice of (appearance|filing|change|modification|witness)\b", ExclusionReason.PROCEDURAL),
    (r"\bproof of publication\b", ExclusionReason.PROCEDURAL),
    (r"\bagreed (motion|notice)\b", ExclusionReason.PROCEDURAL),
    (r"\bentry of appearance\b", ExclusionReason.PROCEDURAL),
    (r"\bmail log\b", ExclusionReason.PROCEDURAL),
    # Filer-flagged dead records. These are not documents, they are notes.
    (r"\bduplicate filing\b", ExclusionReason.PROCEDURAL),
    (r"^void\b|\bvoid see item\b", ExclusionReason.PROCEDURAL),
    (r"\bwrong docket\b|\bfiled in the wrong docket\b", ExclusionReason.PROCEDURAL),
    (r"\bnotice of (deposition|desposition)\b|\bsubpoena\b", ExclusionReason.PROCEDURAL),
    (r"\bstatement of position\b", ExclusionReason.PROCEDURAL),
    (r"\bsent offsite\b", ExclusionReason.PROCEDURAL),
    (r"\bplaced on the agenda\b", ExclusionReason.PROCEDURAL),
    # Legitimate record, but outside a 40-60 document corpus. 200 RFI filings
    # alone would swamp the index; discovery is broad and mostly narrow-issue.
    (r"\brfis?\b", ExclusionReason.OUT_OF_SCOPE),
    (r"\brequests? for information\b", ExclusionReason.OUT_OF_SCOPE),
    (r"\berrata\b", ExclusionReason.OUT_OF_SCOPE),
    (r"\bworkpaper", ExclusionReason.OUT_OF_SCOPE),
    # Not text
    (r"\bnative files\b", ExclusionReason.NON_TEXT),
    (r"\bflash drive\b", ExclusionReason.NON_TEXT),
    (r"\bbate stamp\b", ExclusionReason.NON_TEXT),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in DESCRIPTION_RULES]


def normalize_description(text: str) -> str:
    """Collapse the noise in hand-typed filing descriptions.

    Measured in docket 49421, filers' own descriptions contain typos at a rate
    that breaks exact keyword rules: CONFIDENTILITY, CERTITFICATIONS, RIF for
    RFI, CONFERNECE, ADPOTING, REGUARDING, RETIAL, ISSUS, plus stray double
    spaces. This mirrors the OCR finding on document text -- human input is
    noisy at both layers, so matching is tolerant at both layers.
    """
    text = re.sub(r"\s+", " ", text.strip())
    # Repair the specific misspellings observed in this docket rather than
    # attempting general fuzzy matching, which would create false positives.
    repairs = {
        r"\bconfidentility\b": "confidentiality",
        r"\bcertitfications?\b": "certifications",
        r"\bcertifcations?\b": "certifications",
        r"\brif\b": "rfi",
        r"\brifs\b": "rfis",
        r"\beratta\b": "errata",
        r"\bintitial\b": "initial",
        r"\bcompetative\b": "competitive",
        r"\bcomissioners?\b": "commissioners",
        r"\bconfernece\b": "conference",
        r"\bmertis\b": "merits",
        r"\bbreif\b": "brief",
        r"\bnovemebr\b|\bnovemeber\b": "november",
    }
    for wrong, right in repairs.items():
        text = re.sub(wrong, right, text, flags=re.IGNORECASE)
    return text


@dataclass(frozen=True)
class FilterVerdict:
    include: bool
    reason: ExclusionReason | None = None
    matched_rule: str | None = None

    def __bool__(self) -> bool:
        return self.include


def evaluate(
    item_type: str | None = None,
    filing_description: str | None = None,
    filing_party: str | None = None,
) -> FilterVerdict:
    """Decide whether a filing enters the index.

    Default is to INCLUDE. A document is only dropped when a rule fires, and the
    rule that fired is recorded so every exclusion can be explained.

    NOTE on item_type: the Interchange web table shows a type column (TEST, BR,
    CONF, RFI, TARF, ADMN), but the Excel export does NOT include it. Rules
    therefore cannot depend on it. It is accepted here as an optional extra
    signal for callers who scraped the page.
    """
    if item_type and item_type.strip().upper() in EXCLUDED_ITEM_TYPES:
        return FilterVerdict(False, ExclusionReason.CONFIDENTIAL, f"item_type={item_type}")

    # Party check first: an individual's filing is excluded regardless of how
    # innocuous its description looks.
    if looks_like_person(filing_party):
        return FilterVerdict(False, ExclusionReason.PII_RISK, "filing_party=individual")

    if filing_description:
        normalized = normalize_description(filing_description)
        for pattern, reason in _COMPILED:
            if pattern.search(normalized):
                return FilterVerdict(False, reason, pattern.pattern)

    return FilterVerdict(True)


def summarize(verdicts: list[FilterVerdict]) -> dict[str, int]:
    """Counts by outcome, for the ingestion report and the README."""
    counts = {"included": 0}
    counts.update({r.value: 0 for r in ExclusionReason})
    for verdict in verdicts:
        if verdict.include:
            counts["included"] += 1
        elif verdict.reason:
            counts[verdict.reason.value] += 1
    return counts


__all__ = ["ExclusionReason", "FilterVerdict", "evaluate", "summarize"]
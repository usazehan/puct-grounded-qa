#!/usr/bin/env python3
"""Score the whole pipeline against the question set.

eval_retrieval.py answers "can the right page be found". This answers "does the
system say the right thing, or refuse when it should" -- which is the number the
project exists to produce, and the first one that describes the system rather
than a component.

FOUR OUTCOMES, AND ONLY ONE OF THEM IS SILENT

    answered correctly   a verified claim carries the expected answer
    refused correctly    a refusal question got a refusal
    wrongly refused      the answer is in the corpus and the system declined
    WRONGLY ANSWERED     a verified claim carries the wrong answer

The last is the failure this design exists to prevent, and it is the only one
that reaches a reader as a mistake rather than as an absence. A wrongly refused
question wastes someone's time; a wrongly answered one gives them a plausible
number they cannot distinguish from a correct one without doing the research
themselves. They are counted separately for that reason, and never summed into
a single accuracy figure that would hide the difference.

MATCHING IS DETERMINISTIC, AND DEPENDS ON THE ANSWER MODE

    ANSWER_SINGLE          one figure, checked by containment in the verified
                           claims, after the guard's numeric normalisation --
                           "$450" and "450" agree, "(450)" does not
    ANSWER_QUALIFIED_SET   several figures, each of which must appear in the
                           SAME claim as its qualifier
    REFUSE_NO_SUPPORT      the corpus is silent
    REFUSE_UNRESOLVABLE    the corpus holds values that conflict under the same
                           qualifier, with nothing to choose between them

The qualified-set mode exists because two questions were mislabelled as
refusals. "What is the residential PBRAF?" has three answers across three
schedules, and enumerating them with attribution resolves the ambiguity rather
than guessing at it -- a better answer than declining. But a bare "64.9176%"
with no schedule named is a bad answer to the same question, and joined-text
containment scores both identically.

So a qualified set is scored per CLAIM, not over the joined text. A claim
pairing the right figures with the wrong schedules would satisfy every required
substring while being exactly the misattribution this project exists to prevent.

RESPONSES ARE CACHED

Keyed on question, backend, and the ids offered. Re-running while tuning the
guard should not pay for identical calls, and two runs that differ only in guard
behaviour should differ only in guard behaviour -- an uncached run varies with
the model's sampling even at temperature zero.

Usage:
    python scripts/eval_answers.py --backend echo
    python scripts/eval_answers.py --backend anthropic --show-failures
    python scripts/eval_answers.py --backend anthropic --no-cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg  # noqa: E402

from puctqa.generate import BACKENDS  # noqa: E402
from puctqa.guard import canonical_numbers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import answer as run_answer  # noqa: E402

DEFAULT_DSN = "postgresql://puctqa:puctqa@localhost:5432/puctqa"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CACHE_DIR = Path("data/eval_cache")


@dataclass
class Outcome:
    question_id: str
    category: str
    expected: str
    outcome: str  # answered_correctly | refused_correctly | wrongly_refused | wrongly_answered
    detail: str = ""


def cached_backend(backend, name: str, enabled: bool):
    """Wrap a backend so identical requests are answered from disk.

    Keyed on the prompt and the ids offered, which together determine what the
    model was asked. A guard change does not alter either, so two runs that
    differ only in the guard differ only in the guard.
    """
    if not enabled:
        return backend
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def call(prompt: str, unit_ids: list[str]) -> str:
        key = hashlib.sha256(
            (name + "\0" + prompt + "\0" + ",".join(unit_ids)).encode()
        ).hexdigest()[:32]
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        raw = backend(prompt, unit_ids)
        path.write_text(json.dumps({"prompt": prompt[:2000], "response": raw}))
        return raw

    return call


def qualified_matches(required: list[dict], claims: list[str]) -> tuple[int, int]:
    """(matched, total) required (value, qualifier) pairs.

    Each pair must be satisfied by ONE claim carrying both. Checking the joined
    text would accept "Schedules SRC and 6.7.2 report 40.4859% and 64.9176%",
    which contains every required substring with the values swapped.
    """
    matched = 0
    for item in required:
        wanted = canonical_numbers(item["value"])
        qualifier = item["qualifier"].lower()
        for claim in claims:
            if qualifier not in claim.lower():
                continue
            available = canonical_numbers(claim)
            if all(w in available for w in wanted):
                matched += 1
                break
    return matched, len(required)


def answer_matches(expected: str, claims: list[str]) -> bool:
    """Does any verified claim carry the expected answer?

    Figures are compared after the guard's normalisation, so "$450" and "450"
    agree while "(450)" does not -- a parenthesised negative is a different
    number, and this docket is full of them. Non-numeric expectations fall back
    to case-insensitive containment.
    """
    text = " ".join(claims)
    wanted = canonical_numbers(expected)
    if wanted:
        available = canonical_numbers(text)
        return all(w in available for w in wanted)
    return expected.lower() in text.lower()


def classify(question: dict, result) -> Outcome:
    expects_refusal = question["expected_answer"].startswith("REFUSE_")
    claims = [r.claim.assertion for r in result.verified]

    if question.get("answer_mode") == "ANSWER_QUALIFIED_SET":
        if result.refused:
            return Outcome(
                question["id"], question["category"], question["expected_answer"],
                "wrongly_refused", detail=result.refusal_reason or "",
            )
        matched, total = qualified_matches(question["required_claims"], claims)
        # "any" accepts a partial enumeration as long as what IS said is
        # correctly attributed. Requiring all three would fail an answer that
        # names two schedules accurately, which is not a wrong answer -- it is
        # an incomplete one, and the difference matters here.
        needed = total if question.get("coverage") == "all" else 1
        if matched >= needed:
            return Outcome(
                question["id"], question["category"], question["expected_answer"],
                "answered_correctly", detail=f"{matched}/{total} qualified pairs",
            )
        return Outcome(
            question["id"], question["category"], question["expected_answer"],
            "wrongly_answered",
            detail=f"{matched}/{total} pairs correctly attributed | "
                   + " | ".join(claims)[:200],
        )

    if expects_refusal:
        return Outcome(
            question["id"],
            question["category"],
            question["expected_answer"],
            "refused_correctly" if result.refused else "wrongly_answered",
            detail="" if result.refused else " | ".join(claims),
        )

    if result.refused:
        return Outcome(
            question["id"],
            question["category"],
            question["expected_answer"],
            "wrongly_refused",
            detail=result.refusal_reason or "",
        )

    if answer_matches(question["expected_answer"], claims):
        return Outcome(
            question["id"], question["category"], question["expected_answer"],
            "answered_correctly",
        )
    return Outcome(
        question["id"],
        question["category"],
        question["expected_answer"],
        "wrongly_answered",
        detail=" | ".join(claims),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=Path("evals/questions.jsonl"))
    ap.add_argument("--dsn", default=os.environ.get("PUCTQA_DSN", DEFAULT_DSN))
    ap.add_argument("--backend", choices=list(BACKENDS), default="echo")
    ap.add_argument("--model", default=None)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--show-failures", action="store_true")
    args = ap.parse_args()

    questions = [
        json.loads(line)
        for line in args.questions.read_text().splitlines()
        if line.strip()
    ]

    factory = BACKENDS[args.backend]
    backend = factory(model=args.model) if args.model else factory()
    label = f"{args.backend}:{args.model or 'default'}/k{args.top_k}"
    backend = cached_backend(backend, label, enabled=not args.no_cache)

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(args.embed_model, trust_remote_code=True)
    embed = lambda text: embedder.encode(  # noqa: E731
        text, normalize_embeddings=True
    ).tolist()

    outcomes: list[Outcome] = []
    with psycopg.connect(args.dsn) as conn, conn.cursor() as cur:
        for question in questions:
            result = run_answer(cur, question["question"], backend, embed, args.top_k)
            outcomes.append(classify(question, result))

    counts = Counter(o.outcome for o in outcomes)
    answerable = [o for o in outcomes if not o.expected.startswith("REFUSE_")]
    refusals = [o for o in outcomes if o.expected.startswith("REFUSE_")]

    print(f"config: {label}")
    print(f"{len(questions)} questions — {len(answerable)} answerable, "
          f"{len(refusals)} refusals")
    print()
    print(f"  answered correctly   {counts['answered_correctly']:>3} / {len(answerable)}")
    print(f"  refused correctly    {counts['refused_correctly']:>3} / {len(refusals)}")
    print(f"  wrongly refused      {counts['wrongly_refused']:>3}"
          "      (the answer is in the corpus)")
    print(f"  WRONGLY ANSWERED     {counts['wrongly_answered']:>3}"
          "      (a verified claim, and wrong)")
    print()

    if counts["wrongly_answered"]:
        print("A wrongly answered question is the failure this design exists to")
        print("prevent: it reaches a reader as a plausible number they cannot")
        print("distinguish from a correct one. Every one is worth reading.")
        print()

    by_category: dict[str, Counter] = defaultdict(Counter)
    for o in outcomes:
        by_category[o.category][o.outcome] += 1
    width = max(len(c) for c in by_category)
    print(f"{'category':<{width}}  correct  refused  wrong-refuse  WRONG-ANSWER")
    for category, c in sorted(by_category.items()):
        print(f"{category:<{width}}  {c['answered_correctly']:>7}  "
              f"{c['refused_correctly']:>7}  {c['wrongly_refused']:>12}  "
              f"{c['wrongly_answered']:>12}")
    print()

    if args.show_failures:
        for o in outcomes:
            if o.outcome in {"wrongly_refused", "wrongly_answered"}:
                print(f"{o.question_id}  {o.outcome}")
                print(f"  expected: {o.expected}")
                print(f"  got:      {o.detail[:200] or '(refused)'}")
                print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
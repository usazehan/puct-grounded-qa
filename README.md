# Grounded QA over Texas PUC Utility Filings

![tests](https://github.com/usazehan/puct-grounded-qa/actions/workflows/tests.yml/badge.svg)

Question answering over Texas Public Utility Commission rate-case filings.
Every claim is verified against the span it cites before the answer is
returned. When verification fails, the system refuses.

A rate case decides what millions of customers pay for electricity, and the
documents run to thousands of pages. Generic RAG fails here in a specific way:
it produces a *plausible* number, and a reader cannot tell a plausible number
from a correct one without doing the original research anyway.

This is a research aid, not a compliance tool. It reduces time-to-source; it
does not replace reading the filing.

---

## Results

23 hand-written questions in `evals/questions.jsonl`, scored end to end:

| | |
|---|---|
| answered correctly | **16–17 / 20** |
| refused correctly | **3 / 3** |
| wrongly refused | 1–3 |
| wrongly answered | 1–2 |

The ranges are not hedging. Two questions in twenty flip between runs at
temperature 0, same prompt and same evidence, so 16/20 and 17/20 are the same
result.

A wrong answer is never summed with a wrong refusal. A refusal wastes a
reader's time; a wrong answer hands them a figure they cannot check without
redoing the work.

Retrieval, over the 18 questions with a single source page:

| config | recall@3 | MRR |
|---|---|---|
| lexical | 78% | 0.602 |
| dense | 56% | 0.426 |
| **hybrid** | **83%** | **0.657** |

Per answered question: ~2.9 s, ~3,700 input and ~85 output tokens, ~$0.012.
Cost is almost entirely the evidence sent, so `top_k` is the knob that moves
the bill.

### What the number is worth

Twenty-three questions, written by me, from searches I ran while building the
thing being tested. It measures the paths I knew to look at.

Two questions were relabelled after seeing the output. I wrote "What is the
residential PBRAF?" as a refusal, assuming a system facing three values across
three schedules would pick one. It enumerated them with attribution instead,
which is a better answer. The taxonomy was wrong, not the system — but
relabelling after reading the output is grading next to the answer.

Prose retrieves worse than figures: every question naming a figure or an
acronym retrieves at 100%, while questions about what a witness argued sit at
60%. Roughly half of what these filings contain is argument.

Three failures are worth naming. One is a retrieval miss on a prose question.
Two are verified, cited, and not answers — asked what an adjustment *consisted
of*, the system explained the witness's rationale; asked why approval is in the
public interest, it gave two of three reasons and stopped. Nothing in the guard
catches either.

---

## The service

```bash
curl localhost:8000/health
# {"status": "ok", "embedded_chunks": 817, "retrievable_sets": 6, "backend": "anthropic"}

curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What return on equity was approved in the Final Order?"}'
```
```json
{
  "refused": false,
  "claims": [{
    "assertion": "The Final Order approved a return on equity of 9.4% for CenterPoint Houston.",
    "quoted_span": "It is appropriate for CenterPoint Houston to have an overall rate of return of 6.51%, based on a cost of debt of 4.38%, a return on equity of 9.4%, ...",
    "citation": {
      "document": "49421_792_1054963.pdf", "page": 11,
      "anchor_scheme": "bates", "anchor": "Bates 000000010",
      "weakly_anchored": false
    }
  }],
  "usage": {"latency_ms": 3745, "input_tokens": 3683, "output_tokens": 83, "cost_usd": 0.012294}
}
```

**A refusal returns 200**, carrying its reason. It is the system working, not a
client error, and a 4xx would tempt a client to retry it.

**`weakly_anchored`** says whether the citation names a position in the record or
merely in a file. 14% of this corpus resolves only to a PDF page.

**`/health` reports whether the corpus is queryable**, not just whether the
process is up — migrations applied with chunks never built is the failure that
would otherwise look healthy.

`POST /ask` takes an optional `review` flag, off by default; it roughly doubles
cost and adds ~18 s.

---

---

## What it does end to end

```
question
   │
   ├─ hybrid retrieval over retrieval-eligible sets ──── top 5 chunks
   │
   ├─ segmentation ─────────────────────────────────── evidence units, each
   │                                                    with document offsets
   ├─ a model selects unit IDS from a closed list ───── never quotes text
   │
   ├─ the guard verifies each claim ─────────────────── span, figures, predicate
   │
   └─ answer with citations, or refuse
```

**The model never writes a span.** It selects ids; code resolves them to text
and offsets recorded during segmentation. Asking a model to quote its source
makes verbatim copying a capability requirement — a model that paraphrases
fails span verification even when its claim is right, and the guard ends up
compensating for model behaviour rather than for OCR damage. Selecting from a
closed list makes exact quotation an invariant.

Page headers are shown to the model and are **not** citable. A table row is six
bare values — `Metal Halide (175w) $9.24 12,900 210 N/A 70` — and only the
header says which is the T&D charge. Without it the model was handed the answer
and declined, correctly. With the header citable, a claim could rest on a column
heading, which asserts nothing. Context is for reading; spans are for citing.

---

---

## The grounding guard

Three checks, separate because they fail for different reasons.

**Span verification is fuzzy.** The quoted span must appear in the cited chunk,
but not byte for byte — item 795-A contains `PERIODIC BILLING RE UIREMENT` and
`ATIACHMENT D` in a document measuring 99.8% word accuracy.

**Numeric verification is exact.** Digits survive extraction where the corpus
can be measured, so a figure that does not match is wrong rather than rendered
differently. Sign and unit are part of identity: `(1,234)` is not `1,234`.

**Predicate verification** catches what neither can. Retrieval's top chunk for
*"what return on equity did CenterPoint request?"* is the page stating the
**agreed** 9.4% rather than the requested 10.4%. A claim built on it quotes
accurately and its figure is present. Three figures in this docket are one word
apart — 10.4% requested, 9.45% recommended, 9.4% approved — so the check
compares what a claim asserts against what its source states. It is a word-set
test fitted to this docket and does not generalise.

All three are tested against hand-written claims from real corpus text, half
correct and half wrong in ways that matter. No model is involved, which is the
point: the verifier was built and tested before the thing it verifies existed.

### A fourth check, different in kind

Verification asks whether a claim is supported by its evidence. It never asks
whether the claim answers the question, or whether it says everything the
evidence says. So there is a review pass judging **responsive** and
**complete**, which sees the retrieved units the answer did *not* cite — a
partial answer's omission sits in the evidence it declined to select.

This is a model grading a model, which the rest of the design avoids. It runs
after verification, downgrades only, and records an unparseable review as
*unreviewed* rather than clean, so a results table can report the deterministic
checks apart from this one. Validated against a deliberately truncated answer:
given one of three reasons, it returned `complete: false` and named the two
omissions with their unit ids.

It costs about $0.025 against $0.012 and 21 s against 3 s, and is off by default.

### What none of them catch

**Contradiction between claims.** Nothing deterministic asks whether other
retrieved evidence gives a different answer to the same question.

**Verification is document-level where it should be occurrence-level.**
`numeric_verifiable` is a flag on a set, so item 795-A counts as verified while
50 of its 8,081 figures did not round-trip and nothing marks which.

`DESIGN.md` §17 has the full account, including the six selection rules and four
association models that failed on the way here.

---

## What the corpus can and cannot support

Measured against the Interchange's own Native Files bundles. 19 items group into
16 comparable sets, because several filings are served in parts — item 1 as 71
PDFs, item 795 as four, twice over.

| Verdict | Sets | Meaning |
|---|---|---|
| Verified | 6 | figures round-trip against a covering native |
| Below numeric floor | 6 | figures round-trip at 89.6–98.6% |
| No usable ground truth | 3 | no native in the bundle covers the filing |
| Deferred | 1 | several plausible natives; a human must name the operative one |

**Verification coverage, not accuracy, is the binding constraint.** Ten of sixteen
sets cannot currently support exact numeric verification — not because extraction
is bad, but because the corpus does not supply ground truth for them. The
best-measured set (item 795-A, a 370-page compliance tariff) round-trips 8,031 of
8,081 figures against a native covering 95.4% of its text. That is what the design
rests on.

Prose is consistently better than numbers: word accuracy sits at 99.6–100% on
almost every set while numeric accuracy ranges from 89.6% to 100%. That is the
reverse of the asymmetry the design originally assumed, and it means fuzzy span
matching is doing less work than expected while exact numeric matching is doing
more.

---

---

## Quickstart

```bash
cp .env.example .env
docker compose up          # API + Postgres/pgvector, migrations applied
make test                  # 233 tests, no network, no database
```

`data/sample/` holds **synthetic fixtures**, not real filings, so the tests and
`make probe DIR=data/sample` run for a reviewer who has downloaded nothing. The
corpus itself is not redistributable; with PDFs in `data/raw/` the pipeline is
four commands — manifest, fidelity measurement, ingest, chunk and embed.

Everything runs locally except claim generation, which takes one of three
backends: `echo` (no key, not intelligent, exists so the pipeline is testable),
`ollama` (local), or `anthropic` (a key, about a cent per question).

---

---

## Scope

**In:** docket 49421 (CenterPoint rate case), 19 curated items covering the
procedural spine: application, preliminary order, intervenor testimony, rebuttal,
briefs, proposal for decision, settlement, final order, compliance tariff.

**Out:** OCR performed by this project — documents with no text layer are detected
and excluded with coverage reported; ratepayer comment forms (excluded at ingestion
— they contain names, home addresses, and phone numbers); administrative chaff such
as mail logs and transmittal letters; cross-docket reasoning; any model training.

Selection is two-stage and deliberate. `filters.evaluate()` triages 797 filings
down to the substantive record with conservative rules, logging every exclusion
with the rule that caused it. The final 19 are then listed by hand in
`CORPUS_ITEMS`: choosing which eight of forty direct testimonies belong in the
corpus is editorial judgement, and encoding it as regex would be false precision.

---

---

## Layout

```
src/puctqa/
  sources.py     DocumentSource protocol, manifest loading, set grouping
  extract.py     PDF extraction, page-offset mapping, anchor resolution
  filters.py     Stage 1 triage with auditable exclusion reasons
  chunk.py       layout-aware chunking; prose by budget, tables by row
  evidence.py    chunks -> addressable units with document offsets
  retrieve.py    dense + lexical + trigram arms, fused by reciprocal rank
  generate.py    claim proposal; echo, ollama and anthropic backends
  guard.py       span, numeric, and predicate verification
  review.py      responsiveness and completeness; model-based, downgrades only
  api.py         FastAPI service; /ask and /health
scripts/
  build_manifest.py      export -> manifest; merges, never overwrites assertions
  probe_extraction.py    corpus report + threshold calibration
  ocr_accuracy.py        extraction fidelity against native bundles
  ingest.py              persist sets, documents, text, page spans, anchors
  chunk_corpus.py        chunk the eligible sets and persist with spans
  embed_chunks.py        embed persisted chunks; records the model per row
  eval_retrieval.py      recall and MRR per configuration and category
  answer.py              one question, end to end
  eval_answers.py        the whole pipeline against the question set
  find_answers.py        search the eligible corpus while writing questions
migrations/
  001_init.sql           filings, documents, chunks, claims, eval tables
  002_document_sets.sql  sets, verdicts, page anchors, refusal provenance
  003_chunk_layout.sql   chunk kind and header spans
  004_embeddings.sql     embedding provenance
evals/
  questions.jsonl        15 hand-written questions, 5 of them refusals
tests/                   no network or DB required
```

---

---

## Roadmap

Acquisition, extraction fidelity measurement, schema, ingestion, layout-aware
chunking, embeddings, hybrid retrieval, the grounding guard, claim generation,
the review pass, the API, cost accounting and CI are built and tested.
`DESIGN.md` has the week-by-week plan and §17 records where the build diverged
from it.

Two things are open, both deliberately:

**Contradiction detection.** Nothing deterministic asks whether other retrieved
evidence gives a different answer to the same question. Knowing that 40.4859%
and 64.9176% compete requires knowing both are residential PBRAFs from different
schedules — metadata this corpus does not carry in structured form.

**Verification is document-level where it should be occurrence-level.**
`numeric_verifiable` is a flag on a set, so item 795-A counts as verified while
50 of its 8,081 figures did not round-trip and nothing marks which.

The eval set grows by 10–15 questions a week; five of the twenty-three expect a
refusal, and they have found more real problems than the other eighteen.
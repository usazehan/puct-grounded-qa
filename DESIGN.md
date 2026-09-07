# DESIGN.md — Grounded Q&A over Texas PUC Utility Rate Filings

> Status: draft / pre-build
> Owner: Zehan
> Target: 4 weeks @ ~15 hrs/week

---

## 1. Problem

Texas utility rate cases are decided in public dockets that run to thousands of
pages per proceeding. The documents are public but effectively unsearchable: the
Public Utility Commission of Texas (PUCT) Interchange is a browser-oriented
portal with no official developer API, and answers to questions like *"what
return on equity did CenterPoint request, and where is that stated?"* require
manually paging through PDFs.

The obvious fix is retrieval-augmented generation. The obvious failure mode is
that RAG systems fabricate numbers, and in a regulatory context a fabricated
number is worse than no answer at all.

**This project builds a question-answering API over PUCT filings in which every
quantitative claim is mechanically traceable to a span in a source document, and
the system refuses to answer when it cannot establish that traceability.**

### What makes this different from a generic RAG demo

1. A **deterministic grounding guard** — citations are verified by string and
   numeric matching, not by asking a model to grade itself.
2. A **shipped evaluation harness** with a labeled question set, including
   deliberately unanswerable questions to measure refusal behavior.
3. Production concerns treated as first-class: idempotent ingestion, structured
   logging, per-request cost accounting, containerization, CI, deployment.

### Non-goals

- Not a compliance tool. This is a research and analysis aid. No claim is made
  that outputs are suitable for regulatory filing or legal reliance.
- No model training or fine-tuning. The LLM is an API call. Modeling depth lives
  in a separate project.
- No OCR *performed* by this project. PUCT already OCRs its scans; documents
  whose text layer is absent are detected and excluded (see §10).
- Not a product. No auth, no multi-tenancy, no billing.

---

## 2. Corpus

Source: PUCT Interchange — `https://interchange.puc.texas.gov/`

The Commission regulates electric, telecommunications, and water/sewer utilities.
Filings are organized by **control number** (docket), then **item number**, then
individual documents.

### Access pattern

There is no official API. The portal is scrapeable via query parameters:

```
/search/filings/?UtilityType=E&ControlNumber=58481&DocumentType=ALL
                &SortBy=FilingParty&SortOrder=Ascending
```

Documents resolve to a predictable path:

```
/Documents/{control_number}_{item_number}_{document_id}.PDF
```

`UtilityType` codes: `E` electric, `W` water, `T` telephone, `O` other, `A` all.

### Scope: two dockets

| Control No. | Case style | Why |
|---|---|---|
| 58481 | Rulemaking to implement large load interconnection standards under PURA 37.0561 | Data-center interconnection. Topical; ties the project to current infrastructure debates. |
| 49421 *or* 53719 | Application of CenterPoint Energy Houston Electric / Entergy Texas for authority to change rates | A real rate case with cost-of-service exhibits and numeric testimony. |

Within each docket, ingest only the spine: application, direct testimony, staff
recommendation, intervenor testimony, final order. Target **80–200
PDFs**.

**Lock this on Day 1.** Corpus expansion is the most common way this kind of
project fails to ship.

### Excluded by policy

Ratepayer comment forms are excluded at ingestion by document type. They are
public but contain names, street addresses, phone numbers, and email addresses,
and there is no reason for them to enter a public demo index. Filtering is
applied before any text is persisted.

---

## 3. Architecture

```
                    ┌─────────────────┐
                    │  PUCT Crawler   │  (background job, scheduled)
                    │  polite rate    │
                    │  limit + cache  │
                    └────────┬────────┘
                             │ raw PDFs + metadata
                             ▼
                    ┌─────────────────┐
                    │   Ingestion     │  extract → filter PII types
                    │   Pipeline      │  → chunk w/ offsets → embed
                    └────────┬────────┘
                             ▼
              ┌──────────────────────────────┐
              │   Postgres + pgvector        │
              │   dockets / filings /        │
              │   documents / chunks /       │
              │   queries / claims / evals   │
              └──────────┬───────────────────┘
                         │
       ┌─────────────────┴──────────────────┐
       ▼                                    ▼
┌──────────────┐                   ┌──────────────────┐
│  Query API   │                   │  Eval Harness    │
│  FastAPI     │                   │  runs labeled Q  │
│              │                   │  set, writes     │
│ retrieve →   │                   │  metrics         │
│ generate →   │                   └──────────────────┘
│ GUARD →      │
│ cite/refuse  │
└──────────────┘
```

### Query path

1. Embed question
2. Vector search top-k over `chunks`
3. If max similarity < threshold → **refuse early** (reason: `no_relevant_context`)
4. Build prompt with numbered context blocks
5. LLM returns structured JSON (claims + citations + quoted spans)
6. **Grounding guard** verifies every claim (§5)
7. Return answer with citations, or refuse (reason: `verification_failed`)
8. Persist query, claims, verification outcomes, latency, tokens, cost

---

## 4. Data model

```sql
dockets(
  control_number    text PRIMARY KEY,
  case_style        text,
  utility_type      char(1),
  first_seen_at     timestamptz
)

filings(
  id                bigserial PRIMARY KEY,
  control_number    text REFERENCES dockets,
  item_number       int,
  filing_date       date,
  filing_party      text,
  item_type         text,
  UNIQUE (control_number, item_number)
)

documents(
  id                bigserial PRIMARY KEY,
  filing_id         bigint REFERENCES filings,
  source_url        text,
  filename          text,
  sha256            text,
  page_count        int,
  has_text_layer    boolean,
  extraction_status text,          -- ok | no_text_layer | failed | excluded_pii
  UNIQUE (sha256)
)

chunks(
  id                bigserial PRIMARY KEY,
  document_id       bigint REFERENCES documents,
  page_start        int,
  page_end          int,
  char_start        int,
  char_end          int,
  text              text,
  embedding         vector(768)
)

queries(
  id                bigserial PRIMARY KEY,
  question          text,
  answer            text,
  refused           boolean,
  refusal_reason    text,
  config_id         text,          -- retrieval/prompt version tag
  latency_ms        int,
  input_tokens      int,
  output_tokens     int,
  cost_usd          numeric(10,6),
  created_at        timestamptz
)

claims(
  id                bigserial PRIMARY KEY,
  query_id          bigint REFERENCES queries,
  chunk_id          bigint REFERENCES chunks,
  claim_text        text,
  quoted_span       text,
  span_verified     boolean,
  numerics_verified boolean
)

eval_runs(id, config_id, started_at, metrics jsonb)
eval_items(id, question, expected_answer, answerable boolean,
           gold_document_id, question_type)
```

### The load-bearing detail

`chunks` stores `page_start/page_end` and `char_start/char_end`. This is what
allows a citation to resolve to *page 47, characters 1200–1310 of this specific
PDF* rather than an opaque chunk ID. Every grounding guarantee in §5 depends on
these offsets surviving the chunking step. Test them directly.

### Idempotency

Crawler upserts on `(control_number, item_number)` for filings and on `sha256`
for documents. Re-running the crawler over an already-ingested docket must
produce zero new rows. This is an explicit test, not an aspiration.

---

## 5. The grounding guard

This is the core of the project. Budget the most time here.

### Structured generation

The model is instructed to return JSON only:

```json
{
  "answer": "CenterPoint requested a return on equity of 10.4%.",
  "claims": [
    {
      "text": "CenterPoint requested a return on equity of 10.4%",
      "chunk_id": 8821,
      "quoted_span": "a return on equity of 10.4 percent"
    }
  ],
  "insufficient_context": false
}
```

### Verification — deterministic, no LLM judge

**0. Citation anchor — Bates, not PDF page.** Verified against 49421-788: PDF
page 1 is the Interchange barcode cover sheet, page 2 is the file-stamped title
page, and the document's own labels restart at attachments ("Page 1 of 9", then
"Page 1 of 3"). Only the Bates stamp is unique and monotonic across a filing —
and it is what a lawyer actually cites ("at 0000004"). Pages with no recoverable
stamp fall back to filing page and are flagged, never silently emitted.

**1. Span check.** Does `quoted_span` actually occur in chunk `8821`? Normalize
whitespace, ligatures, and hyphenation artifacts from PDF extraction, then fuzzy
match above a similarity threshold. Catches fabricated quotations outright.

Fuzzy is the primary path, not a fallback, and this is a measured decision.
PUCT serves OCR'd scans; measured against the native .docx for 49421-788,
word-type accuracy is 99.9% — roughly 1 in 700 word types is corrupted by
`rn`→`m` ligature errors. Exact matching would reject legitimate quotations and
inflate the false-refusal rate for no correctness gain.

**2. Numeric check.** Extract every number appearing in `answer`. Each must be
traceable to a cited chunk.

Exact matching here is sound, and that too is measured rather than assumed:
numeric-token accuracy on the same document was **100% (106/106)**, with zero
letters found inside numeric tokens. OCR degrades prose ligatures and leaves
digits alone. Re-run `scripts/ocr_accuracy.py` across the whole corpus before
relying on this — if numeric accuracy is ever below 100%, exact numeric
verification is unsound and numeric claims must be refused rather than checked.

Normalization must handle:

- `$1.2 billion` ↔ `1,200,000,000` ↔ `1.2B`
- `10.4%` ↔ `10.4 percent` ↔ `.104`
- Thousands separators, parenthesized negatives `(1,234)` = `-1234`
- Currency symbols, unit suffixes

Any number that cannot be traced is either stripped with the surrounding claim,
or triggers refusal — configurable, defaults to refusal.

**3. Refusal.** Triggered by:

| Reason | Condition |
|---|---|
| `no_relevant_context` | top-k max similarity below threshold |
| `model_declined` | `insufficient_context: true` in model output |
| `verification_failed` | > N% of claims fail span or numeric check |
| `untraceable_numeric` | any number in answer not found in cited context |

All refusals are logged with reason. Refusal is a first-class outcome, not an
error path.

### Prompt injection

Retrieved filings are untrusted input. Intervenor testimony in a contested rate
case is written by adversarial parties, and nothing prevents text in a PDF from
addressing the model directly.

- Delimit retrieved context in clearly marked blocks; instruct the model that
  context is data to be cited, never instructions to be followed.
- Never let retrieved text influence tool use, routing, or refusal thresholds.
- The span verifier is a second layer of defense: an injected instruction cannot
  manufacture a quotation that survives verification against the chunk it claims
  to come from.

Worth a short README subsection. Very few RAG portfolios address this at all, and
"my corpus is adversarial by construction" is an unusually good answer to the
question of why you thought about it.

### Why this framing matters

"I verify citations with string and numeric matching rather than asking a model
to grade itself" is the sentence this whole project exists to earn. Deterministic
verification has clean failure modes you can characterize and discuss; LLM-as-
judge does not.

Be ready to discuss its limits: the guard catches *fabricated* numbers, not
*misinterpreted* ones. A correctly-quoted figure applied to the wrong entity
still passes. Say so in the README before an interviewer says it for you.

---

## 6. Evaluation

### Question set: 40–60 items, three types

| Type | Count | Measures |
|---|---|---|
| Factual lookup | ~15 | retrieval quality (dates, parties, procedural facts) |
| Numeric | ~15 | numeric accuracy (ROE, revenue requirement, rate change) |
| **Unanswerable** | ~15 | **refusal behavior** — plausible questions the corpus cannot support |

The unanswerable bucket is the point. Most RAG portfolios never measure whether
a system knows when to stop.

### Metrics

- Retrieval hit rate @ k, MRR
- Numeric accuracy at 1% tolerance
- Faithfulness: % of claims passing verification
- **Refusal rate on unanswerable questions** (headline)
- **False refusal rate on answerable questions** (its necessary counterpart)
- p50 / p95 latency
- Cost per query

### Labeling discipline

Label 10–15 questions per week starting Week 2. Deferring all labeling to the end
is the single most likely cause of this project not shipping. Budget 8–12 hours
total; it is tedious and it is the work.

---

## 7. Observability

- `structlog` JSON logs, request ID propagated through retrieval → generation →
  verification
- Per-request: latency breakdown by stage, token counts, computed cost
- `/metrics` endpoint or admin route exposing aggregate cost, p95 latency,
  refusal rate over a rolling window
- Every query persisted, so the eval harness and production traffic share a
  schema

---

## 8. Testing

| Layer | What |
|---|---|
| Crawler | idempotency (re-run → zero new rows), pagination, retry/backoff |
| Extraction | offset integrity — `chunk.text` must equal source slice at recorded offsets |
| Numeric normalizer | table-driven cases incl. negatives, units, separators, percent forms |
| Span verifier | true positives, fabricated quotes, whitespace/ligature variants |
| Refusal logic | each refusal reason triggers under its condition and no other |
| API | contract tests on the response shape |

The numeric normalizer deserves the most cases. It is where the interesting bugs
live and it demos well.

---

## 9. Deployment

- `docker compose up` brings up API + Postgres/pgvector + worker. **One command,
  no manual steps.** This single detail separates serious repos from toy ones.
- A sample corpus (~20 PDFs) committed to the repo so a reviewer can run
  everything without crawling. Nobody evaluates a project that requires a
  three-hour crawl first.
- GitHub Actions: lint, tests, build image
- Deploy: Fly.io or Render

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **OCR quality.** PUCT filings are scans that PUCT has OCR'd. MEASURED on 49421-788 against its native .docx: word-type accuracy 99.9%, numeric-token accuracy 100% (106/106). Errors are `rn`->`m` ligature confusions in prose; digits are untouched. | Numeric verification stays EXACT (sound, per measurement). Span verification goes FUZZY-primary. Run `scripts/ocr_accuracy.py` over every PDF/native pair and publish the table. |
| **Missing text layer.** Some documents are pure imagery. | Detect via chars-per-page; mark `has_text_layer=false` and exclude. Report text-layer coverage as a stat. |
| **Eval labeling fatigue.** | 40 questions is sufficient. 10–15/week from Week 2. |
| **Scraper brittleness / rate limiting.** | Polite delays, aggressive local caching, committed sample corpus so the repo works offline. |
| **Scope creep toward modeling.** | The LLM is an API call. No fine-tuning. Modeling story lives elsewhere. |
| **Corpus expansion temptation.** | Two dockets. Locked Day 1. |
| **Timeline vs. competing commitments.** | Cut list in §11 exists precisely for this. |

---

## 11. Milestones

### Week 1 — Ingestion
Crawler with rate limiting and caching. Postgres schema applied. PyMuPDF
extraction with offset tracking. PII document-type filter.
**Done when:** re-running the crawler produces zero duplicate rows, and filings
are queryable by party and date.

### Week 2 — Retrieval
Chunking with offsets preserved, embeddings, pgvector index, top-k search, basic
generate-with-context. Begin eval labeling.
**Done when:** end-to-end question → answer with chunk IDs attached.

### Week 3 — The guard
Structured output, span verifier, numeric verifier, refusal logic, eval harness
producing a metrics table. Test suite, with heavy coverage on numeric
normalization.
**Done when:** a real results table exists with real numbers in it.

### Week 4 — Ship
Docker Compose for the full stack. structlog with request IDs and cost
accounting. GitHub Actions. Deploy. README with architecture diagram, metrics
table, tradeoffs section, demo GIF.

### Cut list, in order
1. Reranker
2. Hybrid search (BM25 + vector)
3. Any UI
4. Public deployment (Compose + a recorded demo is acceptable)

**Never cut the eval harness.** It is the entire reason this project beats the
other RAG repos.

---

## 12. Tradeoffs to document in the README

Interviewers read this section. Write it as though defending choices out loud.

- **Postgres + pgvector over a dedicated vector DB.** One datastore for metadata
  and embeddings; joins between chunks and filing metadata come free. Costs
  ceiling on index performance at scale — acceptable at 200 documents, would
  revisit at 10M.
- **Deterministic verification over LLM-as-judge.** Characterizable failure
  modes; no second model to evaluate. Costs recall — paraphrased-but-correct
  claims can fail the span check.
- **Refusal defaults over answer-with-caveats.** Domain-driven: a wrong number
  in a rate case is worse than no number. Measured explicitly via false-refusal
  rate.
- **Scoped to PDFs with a usable text layer.** Documents with no text layer are
  detected and excluded rather than OCR'd by this project. OCR accuracy on
  included documents is measured, not assumed -- see the results table.
- **(superseded)** OCR quality on scanned exhibits was below what
  a grounding guarantee could tolerate.
- **Fixed chunk size vs. structure-aware chunking.** Record what was chosen and
  what the alternative would buy.

---

## 13. Open questions

- [ ] Chunk size and overlap — tune against retrieval hit rate, don't guess
- [ ] Embedding model: hosted API vs. local `sentence-transformers` (cost vs.
      reproducibility for anyone cloning the repo)
- [ ] Should numeric-check failure strip the claim or refuse the whole answer?
      Make it configurable, pick a default, justify it.
- [ ] Similarity threshold for early refusal — derive from the eval set, not
      intuition

---

## 14. README outline

Write this last, but sketch it Week 1 — it clarifies what the project has to
prove.

1. **What and why** — the invariant, in two sentences, above the fold. A reader
   should understand the guarantee before they scroll.
2. Architecture diagram
3. Quickstart — `docker compose up`, one command, sample corpus included
4. **Results table** — metrics with methodology, including both refusal rates
5. How the grounding guard works — the three checks, with one worked example
   showing a caught fabrication
6. Scope and limitations — measured OCR accuracy, table extraction, PII handling,
   single-docket questions, and the misinterpretation gap from §5
7. Tradeoffs — §12, written out
8. Demo GIF

Put the results table high. It is the thing that distinguishes this repo, and
most readers will not reach section 7.

---

## 15. Interview questions to prepare for

Rehearse these out loud before the first interview where this project comes up.

**On evaluation**
- How do you evaluate a RAG system without ground truth?
- Why both refusal rate and false-refusal rate? What does one look like without
  the other?
- How did you pick the similarity threshold?

**On the guard**
- Why deterministic verification instead of an LLM judge? What does it cost you?
- Walk me through a false refusal you saw. What caused it?
- How does numeric normalization break?
- How do you handle prompt injection from adversarial testimony?

**On engineering**
- What makes your ingestion idempotent, and why does that matter?
- What's your chunking strategy, and what does it get wrong?
- How would you scale this to all fifty state commissions?
- What would you change to serve this at 100 QPS?

**The one that matters**
- *Where does this system still hallucinate?*

"Nowhere" is the wrong answer and every interviewer knows it. The real answer is
in §5: the guard catches fabricated numbers, not misattributed ones. A figure
quoted correctly but applied to the wrong party or the wrong test year passes
every check. Say it before they do.

---

## 16. Resume line (fill in real numbers on completion)

> Built a grounded question-answering service over Texas PUC utility rate
> filings (FastAPI, Postgres/pgvector). Implemented a deterministic citation
> verifier that traces every numeric claim to a source span, achieving __%
> refusal on unanswerable questions at __% false-refusal; instrumented
> per-request cost and p95 latency; containerized with CI and deployed.

---

## 17. What changed, and why

*Appended after the build. Everything above is the plan as written before any
document was downloaded. This section records where it held and where the corpus
disagreed with it. Nothing above has been edited to match — a design document
that turned out to be right about everything is a design document nobody
followed.*

### The philosophy held

Deterministic verification over LLM-as-judge, refusal as a first-class outcome
rather than an error path, chunk offsets as the load-bearing invariant, and
idempotent ingestion all survived unchanged. So did §6's insistence on
unanswerable questions: five of the twenty-three eval questions expect a refusal,
and they found more real problems than the other eighteen.

The one process instruction that mattered most was §6's labeling discipline.
Questions written while reading a document turned out to be the only ones that
found anything; questions written to fill a category did not.

### The OCR asymmetry is backwards

§5 and §10 rest on a measurement: word-type accuracy 99.9%, numeric-token
accuracy 100% (106/106), on item 49421-788 against its native `.docx`. The
conclusion — fuzzy span verification, exact numeric verification — is still what
the code does. The reasoning was wrong twice over.

**The 106/106 was computed against an empty comparison.** The measurement counted
unique numeric tokens as a set, over tokens of length ≥ 4. On a document whose
paired native has no text layer, that is 0 of 0 matching, which the arithmetic
reports as 100%. Item 773 passed this way for two weeks. There is now a
`no_ground_truth` verdict checked before anything else is believed, requiring the
native to carry real text *and* to account for at least 60% of the served
document — because the next-best native for 773 was a two-page memo covering
23.6% of a 26-page filing, which also scored 100%.

**Across the whole corpus, numbers are the fragile side.** Word accuracy sits at
99.6–100% on almost every set; numeric occurrence accuracy ranges from 89.6% to
100%. Prose renders faithfully and figures do not, which is the reverse of what
§10 predicted. Exact numeric matching remains correct — a figure that does not
match is wrong rather than rendered differently — but it is doing more work than
the design assumed, and fuzzy span matching less.

**And provenance is not recoverable.** §10 assumes documents can be sorted into
scans and born-digital text. Four detectors were tried — chars-per-page,
character-confusion patterns, split-decimal frequency, and PDF font and image
structure — and none separated a known-bad scan from the corpus, because the
Interchange re-renders everything through one pipeline. The probe now reports
what it measures, *pages with a text layer*, and trust comes from a covering
native rather than from a guess about the file.

### Misattribution starts before the model

§5 and §15 name the guard's limit precisely: it catches fabricated numbers, not
misinterpreted ones, and a correctly-quoted figure applied to the wrong entity
passes. Both sections frame this as an LLM interpretation problem.

It is not, or not only. Asked *"what return on equity did CenterPoint request?"*,
retrieval's top result is the findings-of-fact page stating the **agreed** 9.4%
rather than the background page stating the **requested** 10.4%. A claim built on
it quotes accurately and its figure is genuinely present. The failure is upstream
of anything a model does, and nothing downstream of retrieval has the information
to notice.

Three figures in this docket are one word apart — 10.4% requested, 9.45%
recommended by the ALJs, 9.4% approved — and nothing in a quoted figure
distinguishes them. So the guard gained a third check comparing what a claim
*asserts* against what its source *states*. That check is a hand-built word-set
test fitted to this docket's language, and it does not generalise: grouping
`agreed` with `approved` is correct here, because the signatories agreed and the
Commission approved the same figure in one document, and separating them produced
a false refusal on the clearest question in the eval set.

### A set layer between filings and documents

§4 models one document per filing row. Item 795 is a 371-page compliance tariff
served as four ~100-page PDFs, twice over, three months apart — two batches with
identical page-range descriptions and different content. Item 593 is the same
shape at smaller scale.

That forced three changes. A `document_sets` layer between filings and documents,
because the filing as served is not the same thing as a file. Extraction verdicts
attached to the set rather than the document, because comparing one part against
a whole native caps its accuracy at that part's share of the filing — item 795
measured 26% that way and 99.4% correctly. And `retrieval_eligible` separate from
presence, defaulting to false, because which of two refiled sets the Commission
treats as operative is not determinable from the Interchange, and grounding a
rate answer in a superseded tariff is a failure no verification catches.

§5's "Bates, not PDF page" became a hierarchy for the same reason: Bates is not
universal across filings. Anchors resolve Bates, then the document's own
"Page N of M", then PDF page, and the scheme is recorded per page. 14% of chunks
fall back to PDF page, which is a position in a file rather than in the record —
and the document with the best numeric fidelity has the weakest anchors.

**Page-range descriptions are not used to derive offsets.** The design does not
mention them; the first implementation trusted them. Two documents in item 795
are both described "Pages 101 to 200" and begin at different content, so a
description is a claim about a file rather than a fact about it, and
`verify_offset_integrity()` cannot catch the difference — it checks that page
spans tile the extracted text, which is internal consistency.

### The model never writes a span

§5's structured output has the model return `quoted_span` as text. That makes
verbatim copying a model capability requirement: a model that paraphrases fails
the span check even when its claim is right, and the guard ends up compensating
for model behaviour rather than for OCR damage.

Chunks are instead segmented into **evidence units** with document offsets, and
the model selects unit ids from a closed list. Exact quotation becomes an
invariant rather than a capability. Page headers are shown to the model and are
not citable — a table row is six bare values, `Metal Halide (175w) $9.24 12,900
210 N/A 70`, and only the header says which is the T&D charge, but a claim
resting on a column heading asserts nothing.

### Chunking is layout-dependent

§13 lists chunk size as an open question to tune against retrieval hit rate. The
answer was not a size. Pages classify as prose or table by median line length,
and table pages — which serialize one cell per line — split on row boundaries
rather than character counts, because a chunk of bare numerals is retrievable by
nothing and citable to nothing.

Three details cost real time. The median rather than the mean, because every
tariff sheet opens with a six-line header that drags a mean above the threshold.
Pleading line numbers dropped from prose pages and kept on table pages, because a
bare two-digit line is margin furniture in testimony and a cell value in a rate
schedule — the monthly kWh column reads 70, 98, 159, 367. And tables of contents
skipped entirely, since they classify as tables by line length and answer nothing.

### Hybrid retrieval was on the cut list

§11 lists "hybrid search (BM25 + vector)" as the second thing to cut. It is the
thing that works. Measured over the eval set: hybrid reaches recall@3 of 83%
against 78% lexical and 56% dense, and the arms fail on *different* questions —
lexical misses the conceptually-phrased one, dense misses those naming a figure
or an acronym. Lexical alone outranks dense on a corpus where questions name
`PBRAF`, `LGS` and `Sheet 6.7.2`.

Ranks are fused rather than scores, because cosine distance and `ts_rank` are not
on a common scale and any weighting between them would be a constant nobody could
defend.

### A fourth check the design does not have

§5 has three checks. There is now a fourth, and it is different in kind: a review
pass asking whether an answer is **responsive** to the question and **complete**
against the retrieved evidence. Three eval questions turned on that gap — one
explained a rationale when asked what an adjustment consisted of, one gave two of
three reasons and stopped, one could have named a single schedule out of three.
All produce answers that are true, cited, and not what a reader needed.

It is a model grading a model, which the rest of the design avoids, so it runs
after verification, downgrades only, and records its verdict separately. §12's
tradeoff list should say so rather than implying every check is deterministic.

### What the eval set learned about itself

§6 proposes three question types: factual, numeric, unanswerable. The
unanswerable bucket turned out to hold two different things.

*"What is the residential PBRAF?"* has three answers across three schedules. That
is not a contradiction — schedule is a stable qualifier that explains the
variation — so enumeration with attribution answers it, and refusing would be
wrong. *"What is the LOS-A distribution charge?"* is `$0.000000` on two sheets and
`$0.430730` on a third with nothing to distinguish them, and refusing is right.
Same surface shape, opposite correct behaviour.

So the taxonomy is four modes rather than answerable/unanswerable:
`ANSWER_SINGLE`, `ANSWER_QUALIFIED_SET`, `REFUSE_NO_SUPPORT`,
`REFUSE_UNRESOLVABLE`. Qualified sets are scored per claim rather than over
joined text, because an answer pairing the right figures with the wrong schedules
satisfies every required substring while being exactly the misattribution the
guard exists to prevent.

Two questions were relabelled after seeing the system's output, which is grading
adjacent to the answer and is recorded as such in the README.

### Numbers the design asked for

| | |
|---|---|
| corpus | 1 docket, 19 items, 109 PDFs, 10,000 pages |
| retrieval-eligible | 6 sets, 817 chunks |
| answered correctly | 16–17 / 20 |
| refused correctly | 3 / 3 |
| cost per answer | ~$0.012, or ~$0.025 with the review pass |
| latency | ~2.9 s, ~21 s with review |

Two questions in twenty flip between runs at temperature 0. A point estimate
would invite more confidence than the measurement earns.

### Still open

**Contradiction detection.** Nothing deterministic asks whether other retrieved
evidence gives a different answer to the same question. The review pass is the
closest thing and it is a model's judgement; it also only sees what retrieval
returned, so cross-schedule contradiction requires retrieving every schedule
first.

**Verification is document-level where it should be occurrence-level.**
`numeric_verifiable` is a flag on a set, so item 795-A is "verified" while 50 of
its 8,081 figures did not round-trip and nothing marks which. The claims table
already models claims individually; the flag is the part that does not.

**Whether the review pass earns its cost** has not been measured across the eval
set — only validated against a deliberately truncated answer, where it named the
omissions correctly.
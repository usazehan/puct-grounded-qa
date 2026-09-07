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
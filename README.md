# Grounded QA over Texas PUC Utility Filings

Question answering over Texas Public Utility Commission rate-case filings, where
**every factual claim is verified against a source span before the answer is
returned**. When verification fails, the system refuses rather than guessing.

> **Status: Week 1 complete.** Acquisition, extraction, citation anchoring,
> extraction-fidelity measurement, schema, ingestion, and layout-aware chunking
> are built and tested. Retrieval, the grounding guard, and the eval harness are
> not. See `DESIGN.md` for the full spec.

---

## Why refusal is the design center

A rate case determines what millions of customers pay for electricity. The
documents deciding it run to thousands of pages. Generic LLM question answering
fails here in a specific way: it produces a *plausible* number, and the reader
cannot tell a plausible number from a correct one without doing the original
research anyway — which defeats the purpose.

So the invariant is: no claim without a verifiable source span. Numbers
especially.

This is a research aid, not a compliance tool. It reduces time-to-source; it does
not replace reading the filing.

---

## Document acquisition

The PUCT Interchange hosts these filings as public records, but its robots.txt
appears to disallow automated access to the search endpoint. This project
therefore **does not ship a crawler pointed at the agency**.

The corpus is acquired manually — 19 curated items from one closed docket, served
as 109 PDFs across 10,000 pages — and dropped into `data/raw/`. Where the
Interchange offers a "Native Files (Zip)" bundle, it is extracted to
`data/native/<item>/` and used as ground truth. Acquisition sits behind an
interface (`DocumentSource`) so nothing downstream depends on how bytes arrived:

- `LocalFolderSource` — the default. No network access at all.
- `HttpSource` — implemented, tested, and **disabled by default**. Enable only if
  the agency confirms automated retrieval is permitted.

---

## Quickstart

```bash
cp .env.example .env
make up                        # Postgres + pgvector, migrations applied on first start
make test                      # tests, no network or DB required
make probe DIR=data/sample     # extraction report on synthetic fixtures
```

`data/sample/` holds **synthetic fixtures**, not real filings — one born-digital,
one with no text layer, one ratepayer comment scan — so `make test` and the probe
run for a reviewer who has downloaded nothing.

With real documents in `data/raw/` and native bundles in `data/native/`:

```bash
python scripts/build_manifest.py data/FilingExport_49421.xlsx \
    --out data/raw/manifest.json --scan data/raw
make probe DIR=data/raw
python scripts/ocr_accuracy.py data/raw data/native --json data/ocr_report.json
python scripts/ingest.py data/raw --docket 49421 --verdicts data/ocr_report.json
```

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

## What the measurement had to get right first

Every number above is the survivor of a rule that produced a confident, wrong
answer. These are recorded in the code so they are not retried.

**A 100% that was 0 of 0 tokens.** Item 773 reported 100% word and 100% numeric
fidelity for two weeks. Its native bundle contains a PDF of the same scan, with no
text layer; it extracted to nothing, and both accuracy formulas divide by a count
that was zero. Absence of ground truth presented as perfect agreement. There is
now a `no_ground_truth` verdict checked before anything else is believed, and it
requires the native to carry real text *and* to account for at least 60% of the
served document — because the next-best native for 773 was a two-page memo
covering 23.6% of a 26-page filing, which also scored 100%.

**Set recall over unique tokens** scored 100% while occurrences were dropped,
because a repeated subtotal survives elsewhere in the document. Counting is now
per occurrence, and the gap between the two is itself the "how table-like is this
document" statistic.

**Comparing each served PDF against the native** capped accuracy at that file's
share of the filing. Item 795 measured 26% that way — the split, not the
extraction. Parts are now concatenated into a **set** and compared as one
document: 99.4%.

**Six rules for choosing which native is the ground truth**, each wrong somewhere:
format rank picked item 773's memo-only `.docx` over the memo-and-attachments
`.pdf`; word coverage picked item 785's Settlement Agreement (94.7% of served
words, 95.9% of figures) over Exhibit C (81.9%, 99.89%); unioning the bundle
scored 72.9% because the ZIP holds every exhibit filed in the docket while the
served set is a subset; raw numeric agreement picked a 270-figure cover letter at
100% over Exhibit C's 4,332 of 4,337. The script no longer chooses. It reports
every candidate and defers the verdict to a human, who records the operative
native in the manifest — the same place the other editorial decisions live.

**Four models for row association** — whether a value is still bound to its own row
label after extraction. Character windows passed a column-serialized table
trivially; nearest-label penalized wide rows; line distance assumed a row occupies
a line, when item 773 extracts **one cell per line** with currency symbols on their
own lines. Reading order — a value belongs to the nearest label above it —
survives all three. The result is reported as a **lower bound and never gates a
verdict**, because its residual mixes real misattribution with naming variance
between the native and the served rendering (`Land & Land Fees` against `Land and
Land Rights`), and the measurement cannot separate them.

**The probe cannot tell an OCR'd scan from born-digital text.** It reported 100%
"born-digital" coverage across 10,000 pages, including item 773 — whose extracted
text contains `Depredation`, `ATIACHMENT D`, and rate values with lost decimals
(`$173.97` as `173 97`). Four detectors were tried: chars-per-page, character
confusion patterns, split-decimal frequency, and PDF font/image structure. None
separated 773 from the corpus, because the Interchange re-renders everything
through the same pipeline. Provenance is not recoverable, so the probe reports what
it measures — *pages with a text layer* — and trust comes from a covering native
rather than from a guess about the file.

---

## Citation anchors resolve by hierarchy

PDF page numbers are weak: page 1 is the Interchange barcode cover sheet, and
internal labels restart at attachments. But no single alternative is universal.

Across the corpus, 2,588 of 10,000 pages fall back to PDF page numbering. Anchors
resolve in order of authority — Bates stamp, then the document's own "Page N of M"
header, then PDF page — and **the scheme is recorded per page** in `page_anchors`.
A PDF-page anchor is a position in a file, not a position in the record; a citation
that can't say which it is has no business claiming to be verifiable.

**Page-range descriptions are not used to derive offsets.** The Interchange
describes two different documents in item 795 as "Pages 101 to 200", and they begin
at different content. A description is a claim about a file, not a fact about it,
and trusting it displaces every citation from that document silently —
`verify_offset_integrity()` checks that page spans tile the extracted text, which
is internal consistency and cannot detect that page 1 of the file is not page 101
of the record. `page_offset` is asserted by a human in the manifest or it stays 0.

---

## Sets, and why presence is not retrievability

An item can be served as several PDFs, and the same filing can be served twice.
Item 795 is a 371-page tariff served as four parts, in two batches three months
apart. So the schema has three layers:

- **`documents`** — one row per served PDF. Provenance; everything on disk.
- **`document_sets`** — the filing as served. Carries the extraction verdict,
  because that is how it was measured.
- **`retrieval_eligible`** — separate from presence, and `false` by default.

Both 795 sets are `undetermined` and therefore not retrievable. Which one the
Commission treats as the operative tariff is not determinable from the Interchange
— it does not mark supersession, and nothing in the bytes says so. Grounding a rate
answer in a superseded tariff is a failure **no amount of span or numeric
verification catches**, because the text would be quoted correctly from a document
that no longer controls. A `CHECK` constraint enforces that an undetermined set
cannot be retrievable.

The reasoning lives in `selection_note` on each set, and `build_manifest.py`
carries those notes forward rather than overwriting them on regeneration.

---

## Chunking is layout-dependent

Pages classify as prose or table by mean characters per line — item 773's schedules
run 6.8–12.7 while its memo pages run 25.8–72.5, a clean gap at ~15. Numeric
density does not separate them: one page is 30% numeric at 37.6 chars per line
(prose with figures) and another is 39% numeric at 17.3 (a real table).

Table pages serialize one cell per line, so a character-count splitter cuts rows in
half and produces chunks of bare numerals — retrievable by nothing, citable to
nothing. Tables are split on **row boundaries** instead, and every table chunk
carries its page header, including `(amounts in thousands)`. Without that, a
verified `4,231` is wrong by a factor of a thousand and the guard cannot catch it,
because the digits match exactly.

Which means chunk text is **not** a contiguous document slice: it is header plus
body. Both are real spans and both are recorded, so the guard must verify a claim
against one span or the other, never the concatenation.

---

## The load-bearing invariant

`chunks` records `page_start/page_end` and `char_start/char_end`. That turns a
citation from "chunk 8821" into *page 47, characters 1200–1310 of
`49421_795_1057873.pdf`* — a location a human can go verify.

`verify_offset_integrity()` asserts that page spans tile the document text exactly,
with no gaps or overlaps. It runs in tests, in the probe, and in ingest before
anything is persisted. `verify_chunk_spans()` plays the same role for chunks. If
either fails, every citation downstream is suspect.

A chunk belongs to exactly one document. Support that spans a part boundary within
a set is not citable — one citation is one document and one span — and must be
refused deliberately rather than half-answered from whichever part was retrieved.

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

## Layout

```
src/puctqa/
  sources.py     DocumentSource protocol, manifest loading, set grouping
  extract.py     PDF extraction, page-offset mapping, anchor resolution
  filters.py     Stage 1 triage with auditable exclusion reasons
  chunk.py       layout-aware chunking; prose by budget, tables by row
scripts/
  build_manifest.py      export -> manifest; merges, never overwrites assertions
  probe_extraction.py    corpus report + threshold calibration
  ocr_accuracy.py        extraction fidelity against native bundles
  ingest.py              persist sets, documents, text, page spans, anchors
migrations/
  001_init.sql           filings, documents, chunks, claims, eval tables
  002_document_sets.sql  sets, verdicts, page anchors, refusal provenance
tests/                   no network or DB required
```

---

## Roadmap

- [x] **W1** Acquisition, extraction, offsets, anchoring, fidelity measurement,
      schema, ingestion, chunking
- [ ] **W2** Embeddings, pgvector retrieval over retrieval-eligible sets
- [ ] **W3** Grounding guard: span + numeric verification, refusal, eval harness
- [ ] **W4** Structured logging, cost accounting, CI, deploy, results table

Eval labeling runs continuously from W2 — 10–15 questions per week. The eval set
needs negative controls from the start: a claim like *"CenterPoint reported 14
outage events"* can pass exact numeric verification against a testimony chunk
because pleading line numbers put a `14` in every chunk. Questions whose correct
answer is a refusal are worth more than ordinary ones.
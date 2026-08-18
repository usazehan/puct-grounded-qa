# Grounded QA over Texas PUC Utility Filings

Question answering over Texas Public Utility Commission rate-case filings, where
**every factual claim is verified against a source span before the answer is
returned**. When verification fails, the system refuses rather than guessing.

> **Status: Week 1 complete.** Acquisition, extraction, citation anchoring,
> extraction-fidelity measurement, schema, and ingestion are built and tested.
> Retrieval, the grounding guard, and the eval harness are not. See `DESIGN.md`
> for the full spec.

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

The corpus is acquired manually — a fixed set of 19 curated items from one closed
docket, so this is a one-time download, not a recurring chore — and dropped into
`data/raw/`. Acquisition sits behind an interface (`DocumentSource`) so nothing
downstream depends on how bytes arrived:

- `LocalFolderSource` — the default. No network access at all.
- `HttpSource` — implemented, tested, and **disabled by default**. Enable only if
  the agency confirms automated retrieval is permitted.

---

## Quickstart

```bash
cp .env.example .env
make up                        # Postgres + pgvector, migrations applied on first start
make test                      # 127 tests, no network or DB required
make probe DIR=data/sample     # extraction report on synthetic fixtures
```

`data/sample/` holds **synthetic fixtures**, not real filings — one born-digital,
one with no text layer, one ratepayer comment scan — so `make test` and the probe
run for a reviewer who has downloaded nothing.

With real documents in `data/raw/` and native bundles in `data/native/`:

```bash
make probe DIR=data/raw
python scripts/ocr_accuracy.py data/raw data/native --json data/ocr_report.json
python scripts/ingest.py data/raw --docket 49421 --verdicts data/ocr_report.json
```

---

## Measured findings

Measured against the Interchange's own Native Files bundles as ground truth.
Nine documents, 776 pages, all born-digital.

| Set | Word | Numeric (per occurrence) | Row association | Verdict |
|---|---|---|---|---|
| 773 — Commission number run | 100% | **100%** | 77.4% | `exact_structure_reported` |
| 795-A — tariff filing, 370pp | 99.8% | 99.4% | — | `exact_unverified_structure` |
| 795-B — tariff refiling, 371pp | 99.6% | 97.9% | — | `refuse_numerics` |

**Digits survive extraction on tables.** Item 773 is a Commission number run —
a staff accounting memorandum plus eight schedule attachments — at 100% word and
100% numeric fidelity across 26 pages of rate schedules. This is the measurement
exact numeric verification rests on. Everything measured before it was clean
prose, where the result would not have generalized.

**Prose diverges slightly, so span verification is fuzzy.** The 20 word misses
across 795 are lost whitespace (`centerpointenergy`, `allocclass`) and a typo in
the native itself (`exension`) — not OCR error. Every page in this corpus carries
a text layer.

**795-B is unverified, not corrupt.** The Interchange served this item twice.
Only the earlier set has a Native Files bundle, so B was measured against A's
native and its residual (`1999` ×33, `2009` ×8, `2002` ×7) is the delta between
two filings rather than extraction failure. Its `refuse_numerics` verdict should
not be read as corruption. This is recorded in `measured_against` on the set row.

```bash
python scripts/ocr_accuracy.py data/raw data/native --show-errors
```

### What the measurement had to get right first

Three earlier versions of this script produced confident numbers that meant
nothing. Each failure is recorded in the code so it is not retried:

- **Set recall over unique tokens** scored 100% while occurrences were being
  dropped, because a repeated subtotal survives elsewhere in the document.
  Counting is now per occurrence, and the gap between the two is itself the
  "how table-like is this document" statistic.
- **Comparing each served PDF against the native** capped accuracy at that
  file's share of the filing. Item 795 is served as four ~100-page parts, so it
  measured 26% — the split, not the extraction. Parts are now concatenated into
  a **set** and compared as one document: 99.4%.
- **Selecting the native by format rank** picked item 773's memo-only `.docx`
  over the memo-and-attachments `.pdf`, which scores near 100% while measuring a
  fifth of the filing. The primary is now chosen by how much of the served
  document it covers.

Row association — whether a value is still bound to its own row label after
extraction — went through four models. Character windows passed a
column-serialized table trivially; nearest-label penalized wide rows;
line-distance assumed a row occupies a line, when item 773 extracts **one cell
per line** with currency symbols on their own lines. Reading order survives all
three. The resulting 77.4% is reported as a **lower bound and never gates a
verdict**, because its residual mixes real misattribution (`Accumulated
Depreciation` against `Total Accumulated Depreciation`) with naming variance
between the native and the printed attachment (`Land & Land Fees` against `Land
and Land Rights`), and the measurement cannot separate them.

### Citation anchors resolve by hierarchy

PDF page numbers are weak: page 1 is the Interchange barcode cover sheet, and
internal labels restart at attachments. But no single alternative is universal
either.

| Set | Bates | Page label | Falls back to PDF page |
|---|---|---|---|
| 773 (number run) | 1 | 1 | **23** |
| 795-A (tariff) | 8 | 354 | 7 |
| 795-B (tariff) | 8 | 329 | 33 |

Anchors resolve in order of authority — Bates, then the document's own
"Page N of M" header, then PDF page — and **the scheme is recorded per page** in
`page_anchors`. A PDF-page anchor is a position in a file, not a position in the
record; a citation that can't say which it is has no business claiming to be
verifiable. Note that the document with the best numeric fidelity has the weakest
anchors: 23 of 773's 26 pages cite by PDF page alone.

**Page-range descriptions are not used to derive offsets.** The Interchange
describes two different documents in item 795 as "Pages 101 to 200", and they
begin at different content. A description is a claim about a file, not a fact
about it, and trusting it displaces every citation from that document silently —
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
Commission treats as the operative tariff is not determinable from the
Interchange — it does not mark supersession, and nothing in the bytes says so.
Grounding a rate answer in a superseded tariff is a failure **no amount of span
or numeric verification catches**, because the text would be quoted correctly
from a document that no longer controls. A `CHECK` constraint enforces that an
undetermined set cannot be retrievable.

The reasoning lives in `selection_note` on each set, and `build_manifest.py`
carries those notes forward rather than overwriting them on regeneration.

---

## The load-bearing invariant

`chunks` records `page_start/page_end` and `char_start/char_end`. That turns a
citation from "chunk 8821" into *page 47, characters 1200–1310 of
`49421_795_1057873.pdf`* — a location a human can go verify.

`verify_offset_integrity()` asserts that page spans tile the document text
exactly, with no gaps or overlaps. It runs in tests, in the probe, and in ingest
before anything is persisted. If it ever fails, every citation downstream is
suspect.

A chunk belongs to exactly one document. Support that spans a part boundary
within a set is not citable — one citation is one document and one span — and
must be refused deliberately rather than half-answered from whichever part was
retrieved.

---

## Scope

**In:** docket 49421 (CenterPoint rate case), 19 curated items covering the
procedural spine: application, preliminary order, intervenor testimony, rebuttal,
briefs, proposal for decision, settlement, final order, compliance tariff.

**Out:** OCR performed by this project — PUCT already OCRs its scans, and
documents with no text layer are detected and excluded with coverage reported;
ratepayer comment forms (excluded at ingestion — they contain names, home
addresses, and phone numbers); administrative chaff such as mail logs and
transmittal letters; cross-docket reasoning; any model training.

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
scripts/
  build_manifest.py      export -> manifest; merges, never overwrites assertions
  probe_extraction.py    corpus report + threshold calibration
  ocr_accuracy.py        extraction fidelity against native bundles
  ingest.py              persist sets, documents, text, page spans, anchors
migrations/
  001_init.sql           filings, documents, chunks, claims, eval tables
  002_document_sets.sql  sets, verdicts, page anchors, refusal provenance
tests/                   127 tests, no network or DB required
```

---

## Roadmap

- [x] **W1** Acquisition, extraction, offsets, anchoring, fidelity measurement,
      schema, ingestion
- [ ] **W2** Chunking, embeddings, pgvector retrieval
- [ ] **W3** Grounding guard: span + numeric verification, refusal, eval harness
- [ ] **W4** Structured logging, cost accounting, CI, deploy, results table

Eval labeling runs continuously from W2 — 10–15 questions per week. The eval set
needs negative controls from the start: a claim like *"CenterPoint reported 14
outage events"* can pass exact numeric verification against a testimony chunk
because pleading line numbers put a `14` in every chunk. Questions whose correct
answer is a refusal are worth more than ordinary ones.
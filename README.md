# Grounded QA over Texas PUC Utility Filings

Question answering over Texas Public Utility Commission rate-case and rulemaking
filings, where **every factual claim is verified against a source span before the
answer is returned**. When verification fails, the system refuses rather than
guessing.

> **Status: Week 1 scaffold.** Ingestion foundation and extraction are in place.
> Retrieval, the grounding guard, and the eval harness are not yet built.
> See `DESIGN.md` for the full spec.

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

The corpus is acquired manually — it is a fixed set of roughly 80–200 documents
from two closed dockets, so this is a one-time download, not a recurring chore —
and dropped into `data/raw/`. Acquisition sits behind an interface
(`DocumentSource`) so that nothing downstream depends on how bytes arrived:

- `LocalFolderSource` — the default. No network access at all.
- `HttpSource` — implemented, tested, and **disabled by default**. Enable only if
  the agency confirms automated retrieval is permitted.

Everything after acquisition — content-hash dedup, idempotent re-ingest,
extraction status tracking, offset-preserving chunking — is identical either way.

---

## Quickstart

```bash
cp .env.example .env
make up                        # Postgres + pgvector, schema applied on first start
make test                      # 16 tests, no network or DB required
make probe DIR=data/sample     # extraction report on the sample corpus
```

Point the probe at your own documents once you have them:

```bash
make probe DIR=data/raw
```

---

## Measured findings

Verified against two real filings in docket 49421 — item 788 (Staff testimony,
native `.docx`) and item 786 (CenterPoint testimony, born-digital native `.pdf`
from the Native Files ZIP) — using the native file as ground truth:

| Measurement | Result | Consequence |
|---|---|---|
| Text-layer coverage | 100% of pages across both filings | Usable without OCR work |
| Word-type accuracy | 99.9% (item 788), 100% (item 786) | Span matching must be **fuzzy** |
| Numeric-token accuracy | **100% on both (142/142)** | Numeric matching can be **exact** |

OCR quality varies by scan — 788 shows `rn`→`m` errors, 786 is clean — so the
measurement is per-document, not a single corpus-wide claim.

PUCT serves scanned filings that PUCT has already OCR'd. The OCR degrades prose
ligatures (`amount` → `arnount`, `O&M` → `08tM`) and leaves digits untouched.
That asymmetry is the whole reason the guard verifies prose fuzzily and numbers
exactly — a measured decision, not a guess.

```bash
python scripts/ocr_accuracy.py data/raw data/native --show-errors
```

### Citation anchors resolve by hierarchy

PDF page numbers are not citable: page 1 is the Interchange barcode cover sheet,
and internal labels restart at attachments. But no single alternative is
universal either — measured across two filings in the same docket:

| Item | Bates | Page label | Falls back to PDF page |
|---|---|---|---|
| 788 (Staff testimony) | 12 pages | — | 1 |
| 786 (Colvin testimony) | none | 11 pages | 3 |

So anchors resolve in order of authority — Bates, then the document's own
"Page N of M" header, then PDF page — and **the scheme is recorded alongside the
citation**. A PDF-page anchor is a position in a file, not a position in the
record; a citation that can't say which it is has no business claiming to be
verifiable.

---

## Run the probe first

`scripts/probe_extraction.py` is the day-one de-risking tool. Before building
anything on top of the corpus, it answers two questions:

1. What fraction of pages are actually born-digital?
2. Where does the chars-per-page threshold separate real text from scanned
   imagery **in this corpus**?

```
file                                pages  chars/pg status         parsed
----------------------------------------------------------------------------
49421_1_1274481.PDF                     3    2870.0 ok             yes
49421_88_1275000.PDF                    4       1.0 no_text_layer  yes

Born-digital pages:   5  (50.0% coverage)
Offset integrity: OK on all documents.
```

The default threshold of 200 chars/page is a starting guess, not a finding. Look
at the histogram, find the gap between the scanned cluster and the text cluster,
and put the threshold there.

---

## The load-bearing invariant

`chunks` records `page_start/page_end` and `char_start/char_end`. That is what
turns a citation from "chunk 8821" into *page 47, characters 1200–1310 of
`49421_312_1274481.PDF`* — a location a human can go verify.

`verify_offset_integrity()` asserts that page spans tile the document text
exactly, with no gaps or overlaps. It runs in tests and in the probe. If it ever
fails, every citation downstream is suspect.

---

## Scope

**In:** two dockets — 58481 (large load interconnection rulemaking) and one rate
case (49421 CenterPoint or 53719 Entergy Texas). Within each, only the procedural
spine: application, direct testimony, staff recommendation, intervenor testimony,
final order.

**Out:** OCR performed by this project — PUCT already OCRs its scans, and
documents with no text layer are detected and excluded with coverage reported;
ratepayer comment forms (excluded at ingestion — they contain names, home
addresses, and phone numbers); spreadsheet attachments; cross-docket reasoning;
any model training.

---

## Layout

```
src/puctqa/
  sources.py     DocumentSource protocol, LocalFolderSource, gated HttpSource
  extract.py     PDF extraction, page-offset mapping, scanned detection
scripts/
  probe_extraction.py    corpus report + threshold calibration
migrations/
  001_init.sql   full schema including chunks, claims, eval tables
tests/
  test_extract.py        offset integrity, scanned detection, source behavior
```

---

## Roadmap

- [x] **W1** Ingestion foundation: source abstraction, extraction, offsets,
      Bates anchoring, OCR measurement, schema
- [ ] **W2** Chunking, embeddings, pgvector retrieval
- [ ] **W3** Grounding guard: span + numeric verification, refusal, eval harness
- [ ] **W4** Structured logging, cost accounting, CI, deploy, results table

Eval labeling runs continuously from W2 — 10–15 questions per week.
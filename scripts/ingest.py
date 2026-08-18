"""Persist a curated corpus into Postgres.

Reads the manifest, extracts each served PDF, and writes filings, document sets,
documents, full text with page spans, and per-page citation anchors.

Idempotent on content hash. Re-running over the same folder updates metadata and
leaves the extracted text alone; a document whose bytes changed is re-extracted.

WHAT THIS DOES NOT DO
---------------------
It does not decide anything. Two judgements are reserved for a human and are
carried, never derived:

  - `page_offset`, which turns an in-PDF page into a page of the record. Item 795
    serves two documents both described "Pages 101 to 200" that begin at
    different content, so the Interchange description cannot be trusted to
    supply it. Absent an explicit assertion the offset stays 0 and citations
    resolve to PDF pages -- a position in a file, honestly labelled.

  - `status` and `retrieval_eligible` on a set. Which of two refiled sets
    controls is a reading of the record. The schema defaults a set to
    undetermined and not retrievable, and CHECK (undetermined_not_retrievable)
    keeps it that way until someone decides.

Usage:
    python scripts/ingest.py data/raw --docket 49421
    python scripts/ingest.py data/raw --docket 49421 --verdicts data/ocr_report.json
    python scripts/ingest.py data/raw --docket 49421 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg  # noqa: E402
from psycopg.types.json import Json  # noqa: E402

from puctqa.extract import extract_document, verify_offset_integrity  # noqa: E402
from puctqa.sources import (  # noqa: E402
    DocumentRef,
    LocalFolderSource,
    group_document_sets,
    sha256_bytes,
)

DEFAULT_DSN = "postgresql://puctqa:puctqa@localhost:5432/puctqa"


@dataclass
class Counts:
    filings: int = 0
    sets: int = 0
    documents: int = 0
    reextracted: int = 0
    unchanged: int = 0
    skipped: int = 0
    anchors: int = 0


# --- Verdicts -------------------------------------------------------------


def load_verdicts(path: Path | None) -> dict[str, dict]:
    """Extraction verdicts from ocr_accuracy.py --json, keyed by set label.

    The report labels a multi-part set "49421_795 [1057872..1057875]" and a
    single document by filename, so entries are re-keyed onto set_id by their
    constituent parts. Verdicts belong to the set because that is how they were
    measured: comparing one ~100-page part against a 371-page native caps its
    accuracy at that part's share of the filing.
    """
    if not path:
        return {}
    payload = json.loads(path.read_text())
    by_first_part: dict[str, dict] = {}
    for report in payload.get("documents", []):
        parts = report.get("parts") or [report["document"]]
        by_first_part[parts[0]] = report
    return by_first_part


def verdict_for(members: list[DocumentRef], verdicts: dict[str, dict]) -> dict | None:
    return verdicts.get(members[0].filename or "")


# --- Writes ---------------------------------------------------------------


def upsert_filing(cur, control: str, item: int, meta: dict) -> int:
    cur.execute(
        """
        INSERT INTO filings (control_number, item_number, filing_date,
                             filing_party, item_type)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (control_number, item_number) DO UPDATE
            SET filing_date  = COALESCE(EXCLUDED.filing_date, filings.filing_date),
                filing_party = COALESCE(EXCLUDED.filing_party, filings.filing_party)
        RETURNING id
        """,
        (
            control,
            item,
            meta.get("filing_date"),
            meta.get("filing_party"),
            meta.get("item_type"),
        ),
    )
    return cur.fetchone()[0]


def upsert_set(cur, filing_id: int, set_id: str, meta: dict, verdict: dict | None) -> int:
    """Write a set, carrying the human judgements rather than inventing them.

    `status` and `retrieval_eligible` come from the manifest only when present.
    COALESCE keeps whatever is already in the database when the manifest is
    silent, so re-running ingest cannot quietly revoke a decision someone made
    in SQL -- nor grant one.
    """
    cur.execute(
        """
        INSERT INTO document_sets (
            filing_id, set_id, status, retrieval_eligible, selection_note,
            verdict, numeric_verifiable, word_accuracy, numeric_accuracy,
            association_rate, measured_against, measured_at
        )
        VALUES (%s, %s, COALESCE(%s, 'undetermined'), COALESCE(%s, false), %s,
                %s, %s, %s, %s, %s, %s, CASE WHEN %s THEN now() END)
        ON CONFLICT (set_id) DO UPDATE SET
            status             = COALESCE(EXCLUDED.status, document_sets.status),
            retrieval_eligible = COALESCE(EXCLUDED.retrieval_eligible,
                                          document_sets.retrieval_eligible),
            selection_note     = COALESCE(EXCLUDED.selection_note,
                                          document_sets.selection_note),
            verdict            = COALESCE(EXCLUDED.verdict, document_sets.verdict),
            numeric_verifiable = COALESCE(EXCLUDED.numeric_verifiable,
                                          document_sets.numeric_verifiable),
            word_accuracy      = COALESCE(EXCLUDED.word_accuracy,
                                          document_sets.word_accuracy),
            numeric_accuracy   = COALESCE(EXCLUDED.numeric_accuracy,
                                          document_sets.numeric_accuracy),
            association_rate   = COALESCE(EXCLUDED.association_rate,
                                          document_sets.association_rate),
            measured_against   = COALESCE(EXCLUDED.measured_against,
                                          document_sets.measured_against),
            measured_at        = COALESCE(EXCLUDED.measured_at, document_sets.measured_at)
        RETURNING id
        """,
        (
            filing_id,
            set_id,
            meta.get("status"),
            meta.get("retrieval_eligible"),
            meta.get("set_selection_note") or meta.get("selection_note"),
            (verdict or {}).get("verdict"),
            (verdict or {}).get("numeric_verifiable"),
            (verdict or {}).get("word_type_accuracy"),
            (verdict or {}).get("numeric_occurrence_accuracy"),
            (verdict or {}).get("association_rate"),
            # Provenance of the verdict: which native it was measured against, and
            # whether that native actually belongs to this set. 795's refiled
            # batch has no bundle of its own, so its numbers are unverified
            # rather than verified-and-clean, and the row must say so.
            Json(
                {
                    "native": verdict.get("native"),
                    "native_kind": verdict.get("native_kind"),
                    "folded_workpapers": verdict.get("folded_workpapers"),
                }
            )
            if verdict
            else None,
            # Whether a verdict was supplied at all, not the verdict itself:
            # measured_at records when this set was last measured.
            verdict is not None,
        ),
    )
    return cur.fetchone()[0]


def upsert_document(
    cur, filing_id: int, set_db_id: int, ordinal: int, ref: DocumentRef, extracted, digest: str
) -> tuple[int, bool]:
    """Write the document row. Returns (id, content_changed)."""
    cur.execute(
        "SELECT id, sha256 FROM documents WHERE filing_id = %s AND document_id = %s",
        (filing_id, ref.document_id),
    )
    existing = cur.fetchone()

    cur.execute(
        """
        INSERT INTO documents (
            filing_id, set_id, part_ordinal, document_id, source_url, filename,
            description, page_offset, sha256, page_count, chars_per_page,
            has_text_layer, extraction_status, extraction_error
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (filing_id, document_id) DO UPDATE SET
            set_id            = EXCLUDED.set_id,
            part_ordinal      = EXCLUDED.part_ordinal,
            filename          = EXCLUDED.filename,
            description       = EXCLUDED.description,
            page_offset       = EXCLUDED.page_offset,
            sha256            = EXCLUDED.sha256,
            page_count        = EXCLUDED.page_count,
            chars_per_page    = EXCLUDED.chars_per_page,
            has_text_layer    = EXCLUDED.has_text_layer,
            extraction_status = EXCLUDED.extraction_status,
            extraction_error  = EXCLUDED.extraction_error
        RETURNING id
        """,
        (
            filing_id,
            set_db_id,
            ordinal,
            ref.document_id,
            ref.source_url,
            ref.filename,
            ref.description,
            ref.page_offset,
            digest,
            extracted.page_count,
            round(extracted.chars_per_page, 2),
            extracted.has_text_layer,
            extracted.status.value,
            extracted.error,
        ),
    )
    doc_id = cur.fetchone()[0]
    changed = existing is None or existing[1] != digest
    return doc_id, changed


def write_text_and_anchors(cur, doc_id: int, extracted) -> int:
    """Full text, page spans, and per-page anchors.

    Page spans are stored so a citation's character offsets can be resolved and
    audited later without re-parsing the PDF -- if the offsets are wrong, every
    citation built on them is wrong, and re-deriving them from a possibly
    different library version would hide that rather than reveal it.
    """
    cur.execute(
        """
        INSERT INTO document_text (document_id, text, page_spans)
        VALUES (%s, %s, %s)
        ON CONFLICT (document_id) DO UPDATE
            SET text = EXCLUDED.text, page_spans = EXCLUDED.page_spans
        """,
        (
            doc_id,
            extracted.text,
            Json(
                [
                    {
                        "page_number": span.page_number,
                        "char_start": span.char_start,
                        "char_end": span.char_end,
                        "bates": span.bates,
                        "page_label": span.page_label,
                        "is_cover_sheet": span.is_cover_sheet,
                    }
                    for span in extracted.pages
                ]
            ),
        ),
    )

    # Cover sheets are excluded: the Interchange barcode page is not part of the
    # record and nothing should ever cite it.
    cur.execute("DELETE FROM page_anchors WHERE document_id = %s", (doc_id,))
    rows = [
        (doc_id, span.page_number, span.anchor_scheme.value, span.citation)
        for span in extracted.citable_pages()
    ]
    if rows:
        cur.executemany(
            """
            INSERT INTO page_anchors (document_id, pdf_page, anchor_scheme, anchor_value)
            VALUES (%s, %s, %s, %s)
            """,
            rows,
        )
    return len(rows)


# --- Driver ---------------------------------------------------------------


def ingest(
    root: Path, docket: str, dsn: str, verdicts_path: Path | None, dry_run: bool
) -> Counts:
    source = LocalFolderSource(root)
    refs = source.list_documents(docket)
    if not refs:
        raise SystemExit(f"No parseable documents in {root}")

    verdicts = load_verdicts(verdicts_path)
    sets = group_document_sets(refs)
    counts = Counts()

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dockets (control_number, case_style)
            VALUES (%s, %s) ON CONFLICT (control_number) DO NOTHING
            """,
            (docket, f"Docket {docket}"),
        )

        seen_filings: dict[int, int] = {}
        for set_id, members in sorted(sets.items()):
            first_meta = source.metadata(members[0])
            item = members[0].item_number

            if item not in seen_filings:
                seen_filings[item] = upsert_filing(cur, docket, item, first_meta)
                counts.filings += 1
            filing_id = seen_filings[item]

            set_db_id = upsert_set(
                cur, filing_id, set_id, first_meta, verdict_for(members, verdicts)
            )
            counts.sets += 1

            for ordinal, ref in enumerate(members):
                path = root / (ref.filename or "")
                payload = path.read_bytes()
                digest = sha256_bytes(payload)

                extracted = extract_document(payload, page_offset=ref.page_offset)
                if extracted.status.value == "ok":
                    # If page spans do not tile the text exactly, every citation
                    # from this document is suspect. Fail loudly rather than
                    # persist offsets that cannot be trusted.
                    verify_offset_integrity(extracted)

                doc_id, changed = upsert_document(
                    cur, filing_id, set_db_id, ordinal, ref, extracted, digest
                )
                counts.documents += 1

                if changed:
                    counts.anchors += write_text_and_anchors(cur, doc_id, extracted)
                    counts.reextracted += 1
                else:
                    counts.unchanged += 1

                print(
                    f"  {ref.filename:<30} {extracted.page_count:>4}p "
                    f"{extracted.status.value:<14} "
                    f"{'re-extracted' if changed else 'unchanged'}"
                )

        if dry_run:
            conn.rollback()
            print("\nDry run: rolled back.")
        else:
            conn.commit()

    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="folder of served PDFs plus manifest.json")
    ap.add_argument("--docket", required=True)
    ap.add_argument("--dsn", default=os.environ.get("PUCTQA_DSN", DEFAULT_DSN))
    ap.add_argument("--verdicts", type=Path, help="ocr_report.json from ocr_accuracy.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    counts = ingest(args.root, args.docket, args.dsn, args.verdicts, args.dry_run)

    print()
    print(f"Filings:      {counts.filings}")
    print(f"Sets:         {counts.sets}")
    print(f"Documents:    {counts.documents}  ({counts.reextracted} extracted, "
          f"{counts.unchanged} unchanged)")
    print(f"Page anchors: {counts.anchors}")
    print()
    print("Sets are undetermined and not retrievable until someone decides which")
    print("version of the record controls. Set status and retrieval_eligible in")
    print("the manifest, then re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
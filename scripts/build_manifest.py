#!/usr/bin/env python3
"""Turn the Interchange Excel export into a corpus manifest.

Two-stage selection, deliberately:

  STAGE 1 (automatic) -- filters.evaluate() triages 797 filings down to the
  substantive record. Rules are conservative and every exclusion is logged with
  the rule that caused it, so the drop is auditable.

  STAGE 2 (human) -- you pick the final corpus from the survivors by listing
  item numbers in CORPUS_ITEMS below.

Stage 2 is not laziness. Choosing which eight of forty direct testimonies belong
in a 50-document corpus is editorial judgment about what the corpus should be
able to answer; encoding that as regex would be false precision, and the
resulting corpus would be unexplainable. An explicit list is reviewable by
someone who knows rate cases, and it is a record of a decision rather than an
accident of pattern matching.

Note the export omits the item-type column shown on the web page (TEST, BR,
CONF, RFI, TARF, ADMN), so rules run on description and party only.

Usage:
    python scripts/build_manifest.py FilingExport.xlsx --report
    python scripts/build_manifest.py FilingExport.xlsx --out data/raw/manifest.json
    python scripts/build_manifest.py FilingExport.xlsx --out data/raw/manifest.json \
        --scan data/raw

The manifest is per DOCUMENT, not per item, because an item can be served as
several PDFs -- item 795 is a 371-page tariff served as four ~100-page parts,
twice over. Those parts group into a set, and the set is what carries the
operative/superseded judgement and the extraction verdicts, since both are
properties of the whole filing rather than of a quarter of it.

--scan fills the document rows from what you have actually downloaded. Without
it the items are emitted with empty sets, awaiting download.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.filters import evaluate, summarize  # noqa: E402
from puctqa.sources import DocumentRef, group_document_sets  # noqa: E402

# ---------------------------------------------------------------------------
# STAGE 2: the curated corpus.
#
# Chosen to support contested questions -- what a party asked for versus what
# was ultimately agreed -- rather than lookup alone. Every entry survived the
# automatic filter; the note records why it is here.
# ---------------------------------------------------------------------------
CORPUS_ITEMS: dict[int, str] = {
    # --- Spine: the case as filed and as resolved ---
    1: "Application. Narrative petition (first fragments only; exhibits excluded).",
    142: "Preliminary Order. Frames the issues the Commission will decide.",
    720: "PROPOSAL FOR DECISION (SOAH). The ALJs' recommendation -- the baseline "
         "the settlement departs from. Both settlement witnesses reference it.",
    785: "Stipulation and Settlement Agreement. The controlling document.",
    792: "FINAL ORDER. Authoritative resolved numbers.",
    795: "Tariff filing in compliance with Final Order. Table-heavy; rate schedules.",
    # --- Settlement testimony (both sides describing the same agreement) ---
    786: "Colvin (CenterPoint) testimony supporting the Agreement.",
    788: "Tietjen (Staff) testimony supporting the Stipulation.",
    # --- Original positions, for contested-value questions ---
    416: "Pollock direct (TIEC). Industrial consumer position.",
    414: "Woolridge direct (TCUC). Cost of equity -- contrast with settled 9.4% ROE.",
    413: "Garrett direct (City of Houston). Revenue requirement adjustments.",
    405: "Nalepa direct (OPUC). Ratepayer advocate position.",
    410: "Chriss direct (Walmart). Rate design.",
    # --- Utility rebuttal ---
    603: "Hevert rebuttal. CenterPoint's cost-of-equity witness, named in Colvin's "
         "testimony. Direct counterpart to Woolridge -- pairs for the ROE question.",
    593: "Reed rebuttal. Broad response to intervenor positions.",
    600: "Watson rebuttal. Depreciation rates; cited in the Agreement as Exhibit F.",
    # --- The numbers behind the baselines ---
    773: "Commission number run memo. Basis for TCOS/TCRF/DCRF baselines; tables.",
    675: "Commission Staff's Initial Brief. Staff's litigated position pre-settlement.",
    669: "CenterPoint's Initial Brief. The company's litigated position.",
}


# Fields a human asserts and the generator must never overwrite. Everything
# else -- filing dates, parties, descriptions, the document list -- is derived
# from the export and the folder, and is safe to regenerate.
PRESERVED_SET_FIELDS = ("status", "retrieval_eligible", "selection_note")
PRESERVED_DOCUMENT_FIELDS = ("page_offset", "description")


def load_prior(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Existing human assertions, keyed by set_id and by filename.

    Regenerating a manifest must not silently discard the reasoning written into
    it. `status`, `retrieval_eligible`, and `page_offset` are decisions someone
    made by reading the record; a `selection_note` is the record of why. Losing
    those to a routine --scan would be worse than not regenerating at all, since
    the file would still look complete.
    """
    if not path.exists():
        return {}, {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} exists but is not valid JSON ({exc}). Refusing to "
                         "overwrite it -- move it aside if you meant to start fresh.")
    if isinstance(payload, list):  # legacy flat manifest
        return {}, {e["filename"]: e for e in payload if e.get("filename")}

    sets: dict[str, dict] = {}
    docs: dict[str, dict] = {}
    for item in payload.get("items", []):
        for doc_set in item.get("sets", []):
            if doc_set.get("set_id"):
                sets[doc_set["set_id"]] = doc_set
            for doc in doc_set.get("documents", []):
                if doc.get("filename"):
                    docs[doc["filename"]] = doc
    return sets, docs


def carry_over(fresh: dict, prior: dict | None, fields: tuple[str, ...]) -> dict:
    """Prior assertions win; nulls in the prior do not overwrite."""
    if not prior:
        return fresh
    for field in fields:
        if prior.get(field) is not None:
            fresh[field] = prior[field]
    return fresh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path, help="FilingExport.xlsx from the Interchange")
    ap.add_argument("--out", type=Path, help="Write manifest.json here")
    ap.add_argument("--candidates", type=Path, help="Write surviving filings to CSV for review")
    ap.add_argument("--report", action="store_true", help="Print the triage breakdown")
    ap.add_argument(
        "--scan",
        type=Path,
        help="directory of downloaded PDFs; fills document rows and groups them into sets",
    )
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pip install pandas openpyxl")

    df = pd.read_excel(args.export)
    required = {"Item #", "File Stamp", "Filing Party", "Filing Description"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Export is missing columns: {sorted(missing)}")

    control_number = str(df["Control #"].iloc[0]) if "Control #" in df.columns else "unknown"

    verdicts = [
        evaluate(
            filing_description=row["Filing Description"],
            filing_party=row["Filing Party"],
        )
        for _, row in df.iterrows()
    ]
    df = df.assign(
        keep=[v.include for v in verdicts],
        reason=[v.reason.value if v.reason else "" for v in verdicts],
        rule=[v.matched_rule or "" for v in verdicts],
    )

    if args.report:
        print(f"Docket {control_number} -- {len(df)} filings\n")
        print("STAGE 1 automatic triage:")
        for key, count in summarize(verdicts).items():
            share = count / len(df) * 100
            print(f"  {key:<16} {count:>4}  ({share:>4.1f}%)")

        print(f"\nSTAGE 2 curated corpus: {len(CORPUS_ITEMS)} items")
        selected = df[df["Item #"].isin(CORPUS_ITEMS)]
        for _, row in selected.sort_values("Item #").iterrows():
            flag = "" if row["keep"] else f"  <-- WARNING: filter would drop ({row['reason']})"
            print(f"  {int(row['Item #']):>4}  {str(row['Filing Description'])[:58]:<58}{flag}")

        absent = sorted(set(CORPUS_ITEMS) - set(df["Item #"]))
        if absent:
            print(f"\n  Items not present in this export: {absent}")

    if args.candidates:
        cols = ["Item #", "File Stamp", "Filing Party", "Filing Description"]
        df[df.keep][cols].to_csv(args.candidates, index=False)
        print(f"\nWrote {int(df.keep.sum())} candidates to {args.candidates}")
        print("Review these and edit CORPUS_ITEMS in this script.")

    if args.out:
        downloaded: dict[int, dict[str, list[DocumentRef]]] = {}
        if args.scan:
            refs = []
            for path in sorted(Path(args.scan).glob("*.[pP][dD][fF]")):
                try:
                    refs.append(DocumentRef.from_filename(path.name))
                except ValueError:
                    print(f"  skipping unparseable filename: {path.name}")
            for set_id, members in group_document_sets(refs).items():
                downloaded.setdefault(members[0].item_number, {})[set_id] = members
            stray = sorted(set(downloaded) - set(CORPUS_ITEMS))
            if stray:
                print(f"  WARNING: downloaded items not in CORPUS_ITEMS: {stray}")
                print("  These would become corpus documents at ingest. Add a")
                print("  selection_note for them or delete the files.")

        prior_sets, prior_docs = load_prior(args.out)
        carried = 0

        entries = []
        for _, row in df[df["Item #"].isin(CORPUS_ITEMS)].iterrows():
            item = int(row["Item #"])
            sets = []
            for set_id, members in sorted(downloaded.get(item, {}).items()):
                prior_set = prior_sets.get(set_id)
                if prior_set and any(
                    prior_set.get(f) is not None for f in PRESERVED_SET_FIELDS
                ):
                    carried += 1
                sets.append(
                    carry_over(
                        {
                            "set_id": set_id,
                            # operative | superseded | undetermined. Nothing
                            # derives this: which of two refiled sets controls is
                            # a reading of the record, not a property of the bytes.
                            "status": "undetermined",
                            # Presence and retrievability are separate. A
                            # superseded set stays in the corpus for provenance;
                            # grounding an answer in it would cite the wrong
                            # version of the record.
                            "retrieval_eligible": None,
                            "selection_note": None,
                            "documents": [
                                carry_over(
                                    {
                                        "filename": ref.filename,
                                        "document_id": ref.document_id,
                                        # Null until a human reads the document
                                        # and asserts it. Never derived from the
                                        # Interchange page-range description --
                                        # item 795 serves two documents both
                                        # described "Pages 101 to 200" that
                                        # start at different content.
                                        "page_offset": None,
                                        "description": None,
                                    },
                                    prior_docs.get(ref.filename or ""),
                                    PRESERVED_DOCUMENT_FIELDS,
                                )
                                for ref in members
                            ],
                        },
                        prior_set,
                        PRESERVED_SET_FIELDS,
                    )
                )
            entries.append(
                {
                    "item_number": item,
                    "control_number": control_number,
                    "sets": sets,
                    "filing_date": str(row["File Stamp"])[:10],
                    "filing_party": row["Filing Party"],
                    "filing_description": row["Filing Description"],
                    "selection_note": CORPUS_ITEMS[item],
                }
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"control_number": control_number, "items": entries}
        args.out.write_text(json.dumps(payload, indent=2))

        documents = sum(len(st["documents"]) for e in entries for st in e["sets"])
        set_count = sum(len(e["sets"]) for e in entries)
        awaiting = [e["item_number"] for e in entries if not e["sets"]]
        print(f"\nWrote {len(entries)} items, {set_count} set(s), {documents} document(s)")
        print(f"to {args.out}")
        if awaiting:
            print(f"  Awaiting download ({len(awaiting)}): {awaiting}")
        multi = [
            st["set_id"] for e in entries for st in e["sets"] if len(st["documents"]) > 1
        ]
        if multi:
            print(f"  Multi-part sets: {multi}")
        if carried:
            print(f"  Carried forward assertions on {carried} existing set(s).")
        if any(st["status"] == "undetermined" for e in entries for st in e["sets"]):
            print()
            print("Set 'status' and 'retrieval_eligible' on each set. A refiled set left")
            print("retrievable lets an answer cite the superseded version of the record,")
            print("which no amount of verification catches.")
        print("Leave 'page_offset' null unless you have read the document and")
        print("confirmed where it starts; citations fall back to PDF page.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puctqa.filters import evaluate, summarize  # noqa: E402

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path, help="FilingExport.xlsx from the Interchange")
    ap.add_argument("--out", type=Path, help="Write manifest.json here")
    ap.add_argument("--candidates", type=Path, help="Write surviving filings to CSV for review")
    ap.add_argument("--report", action="store_true", help="Print the triage breakdown")
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
        entries = []
        for _, row in df[df["Item #"].isin(CORPUS_ITEMS)].iterrows():
            item = int(row["Item #"])
            entries.append(
                {
                    # filename is filled in after download; document IDs are only
                    # visible on the item page, not in the export.
                    "filename": None,
                    "control_number": control_number,
                    "item_number": item,
                    "filing_date": str(row["File Stamp"])[:10],
                    "filing_party": row["Filing Party"],
                    "filing_description": row["Filing Description"],
                    "selection_note": CORPUS_ITEMS[item],
                }
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(entries, indent=2))
        print(f"\nWrote {len(entries)} manifest entries to {args.out}")
        print("Fill in 'filename' and 'description' as you download each document.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
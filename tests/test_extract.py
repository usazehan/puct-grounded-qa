"""Tests for extraction and source acquisition.

The offset integrity tests are the important ones. Every citation the system
emits depends on chunk offsets resolving to the right page of the right
document; if that breaks, the grounding guarantee is decorative.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from puctqa.extract import (
    ExtractionStatus,
    extract_document,
    normalize_text,
    verify_offset_integrity,
)
from puctqa.sources import DocumentRef, LocalFolderSource, sha256_bytes


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        y = 100
        for line in body.split("\n"):
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    payload = doc.tobytes()
    doc.close()
    return payload


def make_image_only_pdf(page_count: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(page_count):
        page = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200))
        pix.set_rect(pix.irect, (255, 255, 255))
        page.insert_image(fitz.Rect(72, 72, 272, 272), pixmap=pix)
    payload = doc.tobytes()
    doc.close()
    return payload


# --------------------------------------------------------------------------
# Offset integrity
# --------------------------------------------------------------------------


def test_page_spans_tile_document_exactly():
    extracted = extract_document(make_pdf(["Alpha one", "Beta two", "Gamma three"]))
    verify_offset_integrity(extracted)
    assert extracted.page_count == 3
    assert extracted.pages[0].char_start == 0
    assert extracted.pages[-1].char_end == len(extracted.text)


def test_offset_resolves_to_correct_page():
    extracted = extract_document(make_pdf(["Alpha", "Beta", "Gamma"]))
    for needle, expected_page in [("Alpha", 1), ("Beta", 2), ("Gamma", 3)]:
        offset = extracted.text.index(needle)
        assert extracted.page_for_offset(offset) == expected_page


def test_slice_returns_exact_source_text():
    """A citation's quoted span must equal the text at its recorded offsets."""
    extracted = extract_document(
        make_pdf(["A return on equity of 10.4 percent was requested."])
    )
    needle = "return on equity of 10.4 percent"
    start = extracted.text.index(needle)
    end = start + len(needle)
    assert extracted.slice(start, end) == needle


def test_page_boundaries_do_not_fuse_words():
    """Without a separator, last word of page 1 would glue to first of page 2."""
    extracted = extract_document(make_pdf(["ending", "beginning"]))
    assert "endingbeginning" not in extracted.text


def test_offset_beyond_document_raises():
    extracted = extract_document(make_pdf(["short"]))
    with pytest.raises(IndexError):
        extracted.page_for_offset(len(extracted.text) + 100)


# --------------------------------------------------------------------------
# Scanned detection and failure handling
# --------------------------------------------------------------------------


def test_image_only_pdf_flagged_as_no_text_layer():
    extracted = extract_document(make_image_only_pdf())
    assert extracted.has_text_layer is False
    assert extracted.status is ExtractionStatus.NO_TEXT_LAYER


def test_dense_text_passes_threshold():
    dense = "\n".join(["The quick brown fox jumps over the lazy dog."] * 40)
    extracted = extract_document(make_pdf([dense]), scan_threshold=200)
    assert extracted.has_text_layer is True
    assert extracted.status is ExtractionStatus.OK


def test_threshold_is_configurable():
    """Sparse page passes under a permissive threshold, fails under a strict one."""
    payload = make_pdf(["a few words only on this page"])
    assert extract_document(payload, scan_threshold=10).has_text_layer is True
    assert extract_document(payload, scan_threshold=5000).has_text_layer is False


def test_corrupt_pdf_fails_without_raising():
    extracted = extract_document(b"this is not a pdf at all")
    assert extracted.status is ExtractionStatus.FAILED
    assert extracted.error
    verify_offset_integrity(extracted)  # empty doc is still internally consistent


def test_normalize_folds_ligatures_before_offsets_are_taken():
    assert normalize_text("\ufb01ling") == "filing"
    assert normalize_text("soft\u00adhyphen") == "softhyphen"


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def test_filename_parses_into_provenance():
    ref = DocumentRef.from_filename("49421_312_1274481.PDF")
    assert ref.control_number == "49421"
    assert ref.item_number == 312
    assert ref.document_id == "1274481"
    assert ref.natural_key == ("49421", 312, "1274481")


def test_unexpected_filename_raises_with_guidance():
    with pytest.raises(ValueError, match="manifest"):
        DocumentRef.from_filename("testimony_final_v2.pdf")


def test_local_source_filters_by_docket(tmp_path: Path):
    (tmp_path / "49421_1_100.PDF").write_bytes(make_pdf(["in scope"]))
    (tmp_path / "49421_2_101.PDF").write_bytes(make_pdf(["also in scope"]))
    (tmp_path / "53719_1_200.PDF").write_bytes(make_pdf(["other docket"]))
    (tmp_path / "notes.txt").write_text("ignored")

    source = LocalFolderSource(tmp_path)
    refs = source.list_documents("49421")
    assert [r.item_number for r in refs] == [1, 2]


def test_local_source_reads_manifest_metadata(tmp_path: Path):
    (tmp_path / "49421_1_100.PDF").write_bytes(make_pdf(["hello"]))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "filename": "49421_1_100.PDF",
                    "filing_party": "CenterPoint Energy Houston Electric",
                    "item_type": "Application",
                    "source_url": "https://interchange.puc.texas.gov/Documents/49421_1_100.PDF",
                }
            ]
        )
    )
    source = LocalFolderSource(tmp_path)
    ref = source.list_documents("49421")[0]
    assert ref.source_url.endswith("49421_1_100.PDF")
    assert source.metadata(ref)["filing_party"].startswith("CenterPoint")


def test_identical_content_yields_identical_hash():
    """Idempotency depends on this: re-ingesting the same bytes is a no-op."""
    payload = make_pdf(["stable content"])
    assert sha256_bytes(payload) == sha256_bytes(payload)
    assert sha256_bytes(payload) != sha256_bytes(make_pdf(["different content"]))


def test_http_source_refuses_when_disabled():
    from puctqa.sources import HttpSource

    source = HttpSource(enabled=False)
    ref = DocumentRef(control_number="49421", item_number=1, document_id="100")
    with pytest.raises(RuntimeError, match="disabled"):
        source.fetch(ref)


# --------------------------------------------------------------------------
# Split-filing page offsets
#
# The Interchange serves large filings as ~100-page PDF fragments. Page 1 of the
# "Pages 101 to 200" fragment is page 101 of the filing. Citations that ignore
# this are off by a constant -- wrong in exactly the way this project exists to
# prevent.
# --------------------------------------------------------------------------


def test_parse_page_offset_from_description():
    from puctqa.sources import parse_page_offset

    assert parse_page_offset("Pages 1 to 100") == 0
    assert parse_page_offset("Pages 101 to 200") == 100
    assert parse_page_offset("Pages 901 to 1000") == 900
    assert parse_page_offset("Pages 101-200") == 100


def test_parse_page_offset_defaults_to_zero():
    from puctqa.sources import parse_page_offset

    assert parse_page_offset(None) == 0
    assert parse_page_offset("Native Files (Zip)") == 0
    assert parse_page_offset("Direct Testimony of Jane Doe") == 0


def test_filing_page_accounts_for_fragment_offset():
    extracted = extract_document(make_pdf(["alpha", "beta", "gamma"]), page_offset=100)
    offset = extracted.text.index("gamma")
    assert extracted.page_for_offset(offset) == 3        # page within the PDF
    assert extracted.filing_page_for_offset(offset) == 103  # page within the filing


def test_unsplit_document_reports_identical_pages():
    extracted = extract_document(make_pdf(["only page"]))
    offset = extracted.text.index("only")
    assert extracted.page_for_offset(offset) == extracted.filing_page_for_offset(offset)


def test_local_source_derives_offset_from_manifest_description(tmp_path: Path):
    (tmp_path / "49421_1_1013635.PDF").write_bytes(make_pdf(["fragment"]))
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"filename": "49421_1_1013635.PDF", "description": "Pages 101 to 200"}])
    )
    ref = LocalFolderSource(tmp_path).list_documents("49421")[0]
    assert ref.page_offset == 100


# --------------------------------------------------------------------------
# Bates stamps and cover sheets
#
# Bates is the citation anchor. PDF page index is wrong twice over: it counts
# the Interchange barcode cover sheet, and internal document labels restart at
# attachments ("Page 1 of 9" ... then "Page 1 of 3").
# --------------------------------------------------------------------------


def test_bates_extracted_from_page_footer():
    extracted = extract_document(
        make_pdf(["body text here\n0000004", "more body text\n0000005"])
    )
    assert [p.bates for p in extracted.pages] == ["0000004", "0000005"]


def test_last_bates_wins_over_body_numbers():
    """A docket or exhibit number in the body must not be mistaken for the stamp."""
    extracted = extract_document(make_pdf(["See Docket 0000099 for detail\n0000004"]))
    assert extracted.pages[0].bates == "0000004"


def test_bates_for_offset_resolves_citation_anchor():
    extracted = extract_document(
        make_pdf(["alpha content\n0000002", "the ROE is 9.4 percent\n0000003"])
    )
    offset = extracted.text.index("9.4 percent")
    assert extracted.bates_for_offset(offset) == "0000003"


def test_missing_bates_returns_none_rather_than_guessing():
    """Real filings have unstamped pages; a citation must flag, not fabricate."""
    extracted = extract_document(make_pdf(["no stamp on this page"]))
    assert extracted.bates_for_offset(0) is None


def test_interchange_cover_sheet_detected_and_excluded():
    cover = "Control Number: 49421\nItem Number: 788\nAddendum StartPage: 0"
    extracted = extract_document(make_pdf([cover, "real content\n0000002"]))
    assert extracted.pages[0].is_cover_sheet is True
    assert extracted.pages[1].is_cover_sheet is False
    assert len(extracted.citable_pages()) == 1


def test_ordinary_page_not_mistaken_for_cover_sheet():
    extracted = extract_document(make_pdf(["Control Number: 49421 appears in a heading"]))
    assert extracted.pages[0].is_cover_sheet is False
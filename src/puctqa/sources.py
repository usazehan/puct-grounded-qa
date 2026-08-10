"""Document acquisition.

The corpus can be acquired two ways: manually downloaded into a local folder, or
fetched over HTTP. Everything downstream (dedup, extraction, chunking) is
identical either way, so acquisition sits behind a single interface.

LocalFolderSource is the default and the only one enabled out of the box.
HttpSource exists, is tested, and is gated behind an explicit config flag --
see the note on its class docstring before turning it on.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Protocol

# Interchange document filenames encode their provenance:
#   {control_number}_{item_number}_{document_id}.PDF
FILENAME_RE = re.compile(
    r"^(?P<control_number>\d+)_(?P<item_number>\d+)_(?P<document_id>\d+)\.pdf$",
    re.IGNORECASE,
)


# Large filings are split into ~100-page PDFs whose description carries a page
# range: "Pages 101 to 200".
#
# That description is NOT used to derive a page offset, and this is deliberate.
# Item 795 serves two documents both described "Pages 101 to 200" that begin at
# different content, so the description is a claim the Interchange makes about a
# file rather than a fact about the file. Trusting it displaces every citation
# from that document by an unknown amount, and it fails silently:
# verify_offset_integrity() checks that page spans tile the extracted text, which
# is internal consistency and cannot detect that page 1 of the file is not page
# 101 of the record.
#
# A page offset must therefore be asserted explicitly in the manifest, by a human
# who checked. With no offset, page_offset stays 0 and citations resolve against
# the PDF -- a position in a file, honestly labelled as such, rather than a
# position in the record that the file cannot support.
PAGE_RANGE_RE = re.compile(r"pages?\s+(?P<start>\d+)\s*(?:to|-|–)\s*(?P<end>\d+)", re.IGNORECASE)


def parse_page_range(description: str | None) -> tuple[int, int] | None:
    """The page range a description claims, or None.

    Used for grouping and reporting only. Never for citation offsets: see the
    note above.
    """
    if not description:
        return None
    m = PAGE_RANGE_RE.search(description)
    if not m:
        return None
    return int(m["start"]), int(m["end"])


@dataclass(frozen=True)
class DocumentRef:
    """Identity of one filed document, independent of how it was obtained."""

    control_number: str
    item_number: int
    document_id: str
    source_url: str | None = None
    filename: str | None = None
    description: str | None = None
    page_offset: int = 0  # added to in-PDF page numbers to get true filing pages

    @property
    def natural_key(self) -> tuple[str, int, str]:
        return (self.control_number, self.item_number, self.document_id)

    @classmethod
    def from_filename(cls, name: str, source_url: str | None = None) -> "DocumentRef":
        m = FILENAME_RE.match(Path(name).name)
        if not m:
            raise ValueError(
                f"Filename {name!r} does not match the Interchange pattern "
                "{control}_{item}_{docid}.PDF. Rename it or add a manifest entry."
            )
        return cls(
            control_number=m["control_number"],
            item_number=int(m["item_number"]),
            document_id=m["document_id"],
            source_url=source_url,
            filename=Path(name).name,
        )


class DocumentSource(Protocol):
    """Where raw PDF bytes come from."""

    def list_documents(self, control_number: str) -> list[DocumentRef]: ...

    def fetch(self, ref: DocumentRef) -> bytes: ...


class LocalFolderSource:
    """Reads documents you downloaded yourself.

    Expects files named as the Interchange serves them
    (49421_312_1274481.PDF) in a flat directory. Optionally reads a
    manifest.json alongside them carrying filing metadata that isn't in the
    filename -- filing_date, filing_party, item_type.

    This is the default source. It requires no network access, which also means
    a reviewer can clone the repo and run the whole pipeline against the
    committed sample corpus.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, dict]:
        path = self.root / "manifest.json"
        if not path.exists():
            return {}
        with path.open() as fh:
            entries = json.load(fh)
        return {e["filename"]: e for e in entries}

    def list_documents(self, control_number: str) -> list[DocumentRef]:
        refs: list[DocumentRef] = []
        for path in sorted(self.root.glob("*.[pP][dD][fF]")):
            try:
                ref = DocumentRef.from_filename(path.name)
            except ValueError:
                continue
            if ref.control_number != control_number:
                continue
            meta = self._manifest.get(path.name, {})
            description = meta.get("description")
            # Explicit assertion only. An absent offset means the document
            # stands alone and cites by PDF page.
            offset = meta.get("page_offset") or 0
            refs.append(
                DocumentRef(
                    control_number=ref.control_number,
                    item_number=ref.item_number,
                    document_id=ref.document_id,
                    source_url=meta.get("source_url"),
                    filename=path.name,
                    description=description,
                    page_offset=int(offset),
                )
            )
        return refs

    def fetch(self, ref: DocumentRef) -> bytes:
        assert ref.filename, "LocalFolderSource requires a filename on the ref"
        return (self.root / ref.filename).read_bytes()

    def metadata(self, ref: DocumentRef) -> dict:
        return self._manifest.get(ref.filename or "", {})


class HttpSource:
    """Fetches documents over HTTP. DISABLED BY DEFAULT.

    The PUCT Interchange robots.txt should be checked before this is enabled;
    at time of writing automated access to the search endpoint appeared to be
    disallowed. Do not turn this on without confirming current policy, and
    prefer contacting PUCT Central Records for guidance on programmatic access.

    Note the asymmetry this class is built around: listing documents requires
    the search endpoint, fetching a known document does not. If policy permits
    document retrieval but not crawling, use list_documents from a manifest you
    assembled by hand and let this class handle only fetch().
    """

    DOCUMENT_URL = "https://interchange.puc.texas.gov/Documents/{control}_{item}_{doc}.PDF"

    def __init__(
        self,
        enabled: bool = False,
        delay_seconds: float = 2.0,
        user_agent: str = "puct-grounded-qa/0.1 (research; contact: you@example.com)",
        max_retries: int = 3,
    ):
        self.enabled = enabled
        self.delay_seconds = delay_seconds
        self.user_agent = user_agent
        self.max_retries = max_retries
        self._last_request_at = 0.0

    def _guard(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "HttpSource is disabled. Set SOURCE_HTTP_ENABLED=true only after "
                "confirming the agency permits automated retrieval. Use "
                "LocalFolderSource otherwise."
            )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def list_documents(self, control_number: str) -> list[DocumentRef]:
        self._guard()
        raise NotImplementedError(
            "Listing requires the search endpoint. Assemble a manifest manually "
            "and use LocalFolderSource, or implement this only if policy allows."
        )

    def fetch(self, ref: DocumentRef) -> bytes:
        self._guard()
        import urllib.request  # local import keeps the module importable offline

        url = ref.source_url or self.DOCUMENT_URL.format(
            control=ref.control_number, item=ref.item_number, doc=ref.document_id
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read()
            except Exception as exc:  # noqa: BLE001 - retried and re-raised below
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts") from last_error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def iter_documents(
    source: DocumentSource, control_number: str
) -> Iterator[tuple[DocumentRef, bytes, str]]:
    """Yield (ref, payload, sha256) for each document in a docket."""
    for ref in source.list_documents(control_number):
        payload = source.fetch(ref)
        yield ref, payload, sha256_bytes(payload)


__all__ = [
    "DocumentRef",
    "DocumentSource",
    "LocalFolderSource",
    "HttpSource",
    "iter_documents",
    "sha256_bytes",
    "asdict",
]
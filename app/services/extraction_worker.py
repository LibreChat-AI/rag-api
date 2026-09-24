"""Isolated document parsing. Run only as ``python -m app.services.extraction_worker``.

The web process does not import native parsing bindings. Child exit, crash, and
SIGKILL cannot terminate the API process or leave its event loop blocked.
"""

import json
import sys
import zipfile
from importlib.metadata import version
from pathlib import Path

MAX_ARCHIVE_ENTRIES = 4096
MAX_ENTRY_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_OUTPUT_BYTES = 15 * 1024 * 1024
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".jp2",
    ".jpx",
    ".avif",
    ".heic",
    ".heif",
    ".emf",
    ".wmf",
    ".svg",
)


class ExtractionRefusal(Exception):
    def __init__(self, code: str):
        self.code = code


def inspect_docx(path: Path) -> bool:
    """Validate *actually inflated* archive bytes before handing any to AnyDoc.

    Reading in 64-KiB chunks catches false central-directory sizes without
    keeping decompressed entries in memory. This is deliberately separate from
    the text-output limit, since even an empty parse can inflate a zip bomb.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise ExtractionRefusal("ZIP_BOMB")
            names = {entry.filename for entry in entries}
            if not {"[Content_Types].xml", "word/document.xml"} <= names:
                raise ExtractionRefusal("ARCHIVE_INVALID")
            total = 0
            may_omit_content = False
            for entry in entries:
                if entry.is_dir():
                    continue
                if (
                    entry.file_size > MAX_ENTRY_BYTES
                    or total + entry.file_size > MAX_TOTAL_BYTES
                ):
                    raise ExtractionRefusal("ZIP_BOMB")
                name = entry.filename.lower()
                if not name.startswith(("docprops/", "thumbnails/")) and name.endswith(
                    IMAGE_EXTENSIONS
                ):
                    may_omit_content = True
                entry_bytes = 0
                with archive.open(entry) as stream:
                    while chunk := stream.read(64 * 1024):
                        entry_bytes += len(chunk)
                        total += len(chunk)
                        if entry_bytes > MAX_ENTRY_BYTES or total > MAX_TOTAL_BYTES:
                            raise ExtractionRefusal("ZIP_BOMB")
            return may_omit_content
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        RuntimeError,
        EOFError,
        NotImplementedError,
        OSError,
    ) as exc:
        raise ExtractionRefusal("ARCHIVE_INVALID") from exc


def extract(path: Path) -> dict:
    may_omit_content = inspect_docx(path)
    try:
        import anydoc

        # The MIME/extension are never passed to the binding. DOCX identity was
        # checked in the archive above; explicit format avoids filename fallback.
        text = anydoc.to_markdown_bytes(path.read_bytes(), "docx")
    except ImportError as exc:
        raise ExtractionRefusal("PARSER_UNAVAILABLE") from exc
    except Exception as exc:
        # Native error messages can echo source content. Never send them to clients.
        raise ExtractionRefusal("PARSE_FAILED") from exc
    if not isinstance(text, str) or not text.strip():
        raise ExtractionRefusal("NO_DOCUMENT_TEXT")
    if len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ExtractionRefusal("PARSER_OUTPUT_LIMIT")
    return {
        "profile": "document-v1",
        "text": text,
        "format": "markdown",
        "completeness": "partial" if may_omit_content else "complete",
        "may_omit_content": may_omit_content,
        "pages_needing_ocr": [],
        "truncated": False,
        "parser": {"name": "anydoc", "version": version("firecrawl-anydoc")},
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(2)
    try:
        payload = {"ok": True, "result": extract(Path(sys.argv[1]))}
    except ExtractionRefusal as exc:
        payload = {"ok": False, "code": exc.code}
    except Exception:
        payload = {"ok": False, "code": "PARSE_FAILED"}
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized.encode("utf-8")) > MAX_OUTPUT_BYTES:
        serialized = json.dumps({"ok": False, "code": "PARSER_OUTPUT_LIMIT"})
    sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
